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
from ase.units import Bohr, Hartree
import ast
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

class TripletEnergy(torch.nn.Module):
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
                                        self.out_layer
            )
        
        self.sum_triples=None
        self.to(device)

    def structure_wise_energy(self, x, samples):

        structure_map, new_samples, _ = StructureMap(
            samples["structure"], "cpu"
        )
#         print(encoded.shape)
        self.sum_triples =  torch.zeros((len(new_samples), x.shape[-1]), device=x.device)
        self.sum_triples.index_add_(0, structure_map, x)

        return self.sum_triples
    
    def forward(self, x, samples):
        pred = self.model(x)
        pred = self.structure_wise_energy(pred, samples)
        return pred


frames = read(frames_file, ':8000')
triple_sample_array = np.load(feat_file + 'triple_samples.npy')
triple_sample_names = np.load(feat_file + 'triple_samples_names.npy')
triple_samples = Labels(list(triple_sample_names), triple_sample_array)
ntrain=7000
triple_samples_train = Labels(triple_sample_names, triple_sample_array[:ntrain*64])
triple_samples_test = Labels(triple_sample_names, triple_sample_array[ntrain*64:])
#e0 = minstruc.info['energy_ha']
for fi, f in enumerate(frames):
    f.info["energy_rel_eV"] = (f.info["energy_ha"]-e0)#*Hartree
    for n, v in zip( ("index", "label","r", "z_1", "z_2", "psi", "phi_1", "phi_2"), ast.literal_eval(f.info["pars"]) ):
        f.info[n] = v
    if fi%2 ==1:
        frames[fi].info["delta"] = np.abs(frames[fi].info["energy_rel_eV"]-frames[fi-1].info["energy_rel_eV"])
        frames[fi-1].info["delta"] = frames[fi].info["delta"]

energy = torch.tensor([f.info["energy_rel_eV"] for f in frames], device=device)
print(torch.min(energy), torch.max(energy))

feats_n2nu1 = hickle.load(feat_file+ 'feat_3cnu1_PCA.hickle' )
feats_n2nu1 = torch.tensor(feats_n2nu1, device=device)

triple_enmodel = TripletEnergy(input_size = feats_n2nu1.shape[-1],
                               intermediate_size = 128,
                               hidden_size = 128, 
                              )
print("\n Originial 3cnu1 shape", feats_n2nu1.shape)
print(triple_enmodel)
def mse_loss(pred, target):
    return torch.sum((pred.flatten() - target.flatten()) ** 2) 

best = 1e10
energy_optimizer = torch.optim.Adam(
        triple_enmodel.parameters(),
        lr=1e-3,
        #line_search_fn="strong_wolfe",
        #history_size=128
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(energy_optimizer, factor = 0.8, patience = 200)
#checkpoint = torch.load('./models/triple_p0-lin-best.pt')
#triple_enmodel.load_state_dict(checkpoint['model_state_dict'])
#energy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch_load = checkpoint['epoch']
epoch_load = 0
print(epoch_load)

def iterate_minibatches(inputs, outputs, batch_size):
    for index in range(0, int(inputs.shape[0]/64), batch_size):
        yield inputs[index*64 : (index + batch_size)*64], outputs[index : index + batch_size], index

ntrain=7000
BATCH_SIZE = 250
n_epochs = 20000

for epoch in range(epoch_load, epoch_load + n_epochs):
    all_predictions = []
    triple_enmodel.train(True)
    for feat, target, index in iterate_minibatches(feats_n2nu1[:ntrain*64], energy[:ntrain], BATCH_SIZE):
        samples= Labels(triple_sample_names, triple_sample_array[index*64:(index+BATCH_SIZE)*64])
        predicted = triple_enmodel.forward(feat, samples)
#         print(index,samples)
        all_predictions.append(predicted.data.detach().numpy())
        l = mse_loss(predicted, target)
        l.backward()
        #print(l)
        energy_optimizer.step()
        energy_optimizer.zero_grad()

    all_predictions = np.concatenate(all_predictions, axis = 0)

    train_error = mse_loss(torch.tensor(all_predictions), energy[:ntrain])
    scheduler.step(train_error)
    triple_enmodel.train(False)
    test_all_predictions = []
    for feat, target,index in iterate_minibatches(feats_n2nu1[ntrain*64:], energy[ntrain:], BATCH_SIZE):
        tsamples = Labels(triple_sample_names, triple_sample_array[(ntrain+index)*64:(ntrain+index+BATCH_SIZE)*64])
#         print(index,tsamples)
#     print(struct_feat)
        predicted = triple_enmodel.forward(feat, tsamples)
        test_all_predictions.append(predicted.data.detach().numpy())

    test_all_predictions = np.concatenate(test_all_predictions, axis = 0)
    val_error = mse_loss(torch.tensor(test_all_predictions), energy[ntrain:])

    print("epoch: ", epoch, "RMSE train: ", np.sqrt(train_error.detach().numpy()/(ntrain)),
          "RMSE test: ", np.sqrt(val_error.detach().numpy()/((8000-ntrain))))
    if val_error< best:
        best = val_error
        torch.save({
            'epoch':epoch,
            'model_state_dict': triple_enmodel.state_dict(),
            'optimizer_state_dict': energy_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss':train_error,
            'valerror':val_error
        }, './models/batchmc2_triple_p0-lin-best.pt')
