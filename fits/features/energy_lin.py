from ase.io import read, write
import hickle
import ase
import json
from tqdm import tqdm
class tqdm_reusable:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        return tqdm(*self._args, **self._kwargs).__iter__()
import torch
device = "cpu"
torch.set_default_dtype(torch.float64)
import ast 
import copy 
import numpy as np
import scipy as sp
from equistore.io import load,save
from equistore import Labels, TensorBlock, TensorMap
from itertools import product
from rascaline import SphericalExpansion
from rascaline import SphericalExpansionByPair as PairExpansion
from equistore import operations
import sys, os
sys.path.append(os.getcwd())
from feat_settings import *

import scipy
from sklearn.decomposition import PCA
e0 = -198.27291671238572
def StructureMap(samples_structure, device="cpu"):
    unique_structures, unique_structures_idx = np.unique(
        samples_structure, return_index=True
    )
    new_samples = samples_structure[np.sort(unique_structures_idx)]
    # we need a list keeping track of where each atomic contribution goes
    # (e.g. if structure ids are [3,3,3,1,1,1,6,6,6] that will be stored as
    # the unique structures [3, 1, 6], structure_map will be
    # [0,0,0,1,1,1,2,2,2]
    replace_rule = dict(zip(unique_structures, range(len(unique_structures))))
    structure_map = torch.tensor(
        [replace_rule[i] for i in samples_structure],
        dtype=torch.long,
        device=device,
    )
    return structure_map, new_samples, replace_rule
class BlockEncoderDecoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encode_type = kwargs['encode_type']
        self.input_size = kwargs["input_size"]
        self.intermediate_size = kwargs["intermediate_size"]
        self.hidden_size = kwargs["hidden_size"]
        if 'output_size' in kwargs:
            self.output_size = kwargs["output_size"]
        else: 
            self.output_size = self.input_size

        if self.encode_type == 'linear':
            self.encoder_output_layer = torch.nn.Linear(
            in_features=self.input_size, out_features=self.intermediate_size)
            self.encode_nn = torch.nn.Sequential(self.encoder_output_layer)
        else:
            self.encoder_hidden_layer = torch.nn.Linear(
                in_features=self.input_size, out_features=self.hidden_size)
            self.encoder_hidden_layer2= torch.nn.Linear(
                in_features=self.hidden_size, out_features=self.hidden_size)

            self.encoder_output_layer = torch.nn.Linear(
                in_features=self.hidden_size, out_features=self.intermediate_size)

            self.encode_nn = torch.nn.Sequential(self.encoder_hidden_layer,
                                                 torch.nn.LayerNorm(normalized_shape=self.hidden_size),
                                                 torch.nn.SiLU(),
                                                 self.encoder_hidden_layer2,
                                                 torch.nn.LayerNorm(normalized_shape=self.hidden_size),
                                                 torch.nn.SiLU(),
                                                 self.encoder_output_layer,
            )


        self.decoder_hidden_layer = torch.nn.Linear(
            in_features=self.intermediate_size, out_features=self.hidden_size)

        self.decoder_hidden_layer2 = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size)

        self.decoder_output_layer = torch.nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size)
        self.decode_nn = torch.nn.Sequential(self.decoder_hidden_layer,
                                             torch.nn.LayerNorm(normalized_shape=self.hidden_size),
                                             torch.nn.SiLU(),
                                             self.decoder_hidden_layer2,
                                             torch.nn.LayerNorm(normalized_shape=self.hidden_size),
                                             torch.nn.SiLU(),
                                             self.decoder_output_layer,
                                             )

        self.sum_triples=None
        self.to(device)

    def structure_wise_feats(self, x, samples):
        encoded = self.encode_nn(x)
        structure_map, new_samples, _ = StructureMap(
            samples["structure"], "cpu"
        )
        self.sum_triples =  torch.zeros((len(new_samples), encoded.shape[-1]), device=x.device)
        self.sum_triples.index_add_(0, structure_map, encoded)

        return self.sum_triples

    def forward(self, x):
        decoded = self.decode_nn(x)
        return decoded

class SingleEnergy(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_size = kwargs["input_size"]
        self.intermediate_size = kwargs["intermediate_size"]
        self.hidden_size = kwargs["hidden_size"]
        self.input_layer = torch.nn.Linear(
            in_features=self.input_size, out_features=self.intermediate_size)

        self.hidden_layer = torch.nn.Linear(
                in_features=self.intermediate_size, out_features=self.hidden_size
            )

        self.out_layer = torch.nn.Linear(
                in_features=self.hidden_size, out_features=1
            )

        self.model = torch.nn.Sequential( self.input_layer,
                                        torch.nn.GroupNorm(4,self.intermediate_size),
                                         torch.nn.SiLU(),
                                         self.hidden_layer,
                                         torch.nn.GroupNorm(4,self.hidden_size),
                                         torch.nn.SiLU(),
                                         self.out_layer
            )

        self.to(device)


    def forward(self, x):
        pred = self.model(x)
        return pred



frames = read(frames_file, ':')
triple_sample_array = np.load(feat_file + 'triple_samples.npy')
triple_sample_names = np.load(feat_file + 'triple_samples_names.npy')
triple_samples = Labels(list(triple_sample_names), triple_sample_array)
for fi, f in enumerate(frames):
    f.info["energy_rel_eV"] = (f.info["energy_ha"]-e0)#*Hartree
    for n, v in zip( ("index", "label","r", "z_1", "z_2", "psi", "phi_1", "phi_2"), ast.literal_eval(f.info["pars"]) ):
        f.info[n] = v 
    if fi%2 ==1:
        frames[fi].info["delta"] = np.abs(frames[fi].info["energy_rel_eV"]-frames[fi-1].info["energy_rel_eV"])
        frames[fi-1].info["delta"] = frames[fi].info["delta"]

energy = torch.tensor([f.info["energy_rel_eV"] for f in frames], device=device)
print(torch.min(energy), torch.max(energy), energy.shape)

feats_n2nu1 = hickle.load(feat_file+ 'feat_3cnu1_PCA.hickle' )
feats_n2nu1 = torch.tensor(feats_n2nu1, device=device)

isize = 128
hsize = 128
input_size = feats_n2nu1.shape[-1]
hidden_size = 512
intermediate_size = 128
encode_type = 'linear'
output_size =  8000#feats_nu_to7.shape[-1]

model_block = BlockEncoderDecoder(input_size=input_size, hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, encode_type=encode_type,
                                  output_size = output_size

                                                  )
feat_checkpoint = torch.load('./models/lin-best.pt')
model_block.load_state_dict(feat_checkpoint['model_state_dict'])
encoded = model_block.structure_wise_feats(feats_n2nu1, triple_samples)
encoded = encoded.detach()
print(encoded.shape, "should be ", intermediate_size) 
enmodel = SingleEnergy(input_size = intermediate_size,
                               intermediate_size = isize,
                               hidden_size = hsize
                              )



def mse_loss(pred, target):
    return torch.sum((pred.flatten() - target.flatten()) ** 2)

def iterate_minibatches(inputs, outputs, batch_size):
    for index in range(0, int(inputs.shape[0]), batch_size):
        yield inputs[index : (index + batch_size)], outputs[index : index + batch_size], index



total = int(len(frames)/2)

ntrain=7000
BATCH_SIZE = 250

optimizer = torch.optim.Adam(
        enmodel.parameters(),
        lr=1e-3
    )
best = 1e10
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience = 100)
n_epochs = 50000

#checkpoint = torch.load('./models/nl-adam-best_double_silu.pt')
#model_block.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#epoch_load = checkpoint['epoch']
#loss = checkpoint['loss']
#best = checkpoint['valerror']
#print(epoch_load, loss)
epoch_load = 0
for epoch in range(epoch_load, epoch_load + n_epochs):
    all_predictions = []
    enmodel.train(True)
    for feat, target, index in iterate_minibatches(encoded[:ntrain], energy[:ntrain] , BATCH_SIZE):
        predicted = enmodel.forward(feat) 
        all_predictions.append(predicted.data.detach().numpy())
        l = mse_loss(predicted, target)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    all_predictions = np.concatenate(all_predictions, axis = 0) 
    
    train_error = mse_loss(torch.tensor(all_predictions), energy[:ntrain])
    scheduler.step(train_error) 
    enmodel.train(False)
    test_all_predictions = []
    for feat, target,index in iterate_minibatches(encoded[ntrain:], energy[ntrain:], BATCH_SIZE):
        predicted = enmodel.forward(feat) 
        test_all_predictions.append(predicted.data.detach().numpy())
        
    test_all_predictions = np.concatenate(test_all_predictions, axis = 0)
    val_error = mse_loss(torch.tensor(test_all_predictions), energy[ntrain:])

    print("epoch: ", epoch, "RMSE train: ", np.sqrt(train_error.detach().numpy()/(ntrain)),
          "RMSE test: ", np.sqrt(val_error.detach().numpy()/((8000-ntrain))))
    if val_error< best: 
        best = val_error
        torch.save({
            'epoch':epoch,
            'model_state_dict': enmodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss':train_error,
            'valerror':val_error
        }, './energy_models/energy-lin-best.pt')

    

