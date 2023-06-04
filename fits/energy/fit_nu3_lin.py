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
                                         self.out_layer
            )
        
        self.to(device)

    
    def forward(self, x):
        pred = self.model(x)
        return pred


frames = read(frames_file, ':8000')
ntrain=7000
samples_train = slice(0, ntrain) 
#e0 = minstruc.info['energy_ha']
for fi, f in enumerate(frames):
    f.info["energy_rel_eV"] = (f.info["energy_ha"]-e0)#*Hartree
    for n, v in zip( ("index", "label","r", "z_1", "z_2", "psi", "phi_1", "phi_2"), ast.literal_eval(f.info["pars"]) ):
        f.info[n] = v
    if fi%2 ==1:
        frames[fi].info["delta"] = np.abs(frames[fi].info["energy_rel_eV"]-frames[fi-1].info["energy_rel_eV"])
        frames[fi-1].info["delta"] = frames[fi].info["delta"]

energy = torch.tensor([f.info["energy_rel_eV"] for f in frames], device=device)
print(torch.min(energy), torch.max(energy), energy.shape)

feats_nu_to3 = hickle.load(feat_file + 'feat_123_PCA.hickle')
feats_nu_to3 = torch.tensor(feats_nu_to3, device=device)

enmodel = SingleEnergy(input_size = feats_nu_to3.shape[-1],
                               intermediate_size = 128,
                               hidden_size = 128, 
                              )
print("\n Originial feats shape", feats_nu_to3.shape)
print(enmodel)

def mse_loss(pred, target):
    return torch.sum((pred.flatten() - target.flatten()) ** 2) 
best = 1e10
energy_optimizer = torch.optim.Adam(
        enmodel.parameters(),
        lr=1e-3,
        #line_search_fn="strong_wolfe",
        #history_size=128
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(energy_optimizer, factor = 0.8, patience = 200)
def iterate_minibatches(inputs, outputs, batch_size):
    for index in range(0, int(inputs.shape[0]), batch_size):
        yield inputs[index : (index + batch_size)], outputs[index : index + batch_size], index

ntrain=7000
BATCH_SIZE = 250
n_epochs = 100000
for epoch in range(0, n_epochs):
    all_predictions = []
    enmodel.train(True)

    for feat, target, index in iterate_minibatches(feats_nu_to3[:ntrain], energy[:ntrain], BATCH_SIZE):
        predicted = enmodel.forward(feat)
        all_predictions.append(predicted.data.detach().numpy())
        l = mse_loss(predicted, target)
        l.backward()
        energy_optimizer.step()
        energy_optimizer.zero_grad()

    all_predictions = np.concatenate(all_predictions, axis = 0)

    train_error = mse_loss(torch.tensor(all_predictions), energy[:ntrain])
    scheduler.step(train_error)
    enmodel.train(False)
    test_all_predictions = []
    for feat, target,index in iterate_minibatches(feats_nu_to3[ntrain:], energy[ntrain:], BATCH_SIZE):
        predicted = enmodel.forward(feat)
        test_all_predictions.append(predicted.data.detach().numpy())

    test_all_predictions = np.concatenate(test_all_predictions, axis = 0)
    val_error = mse_loss(torch.tensor(test_all_predictions), energy[ntrain:])
    if epoch==0:
        print("train shape", all_predictions.shape, "test.shape", test_all_predictions.shape)
    print("epoch: ", epoch, "RMSE train: ", np.sqrt(train_error.detach().numpy()/(ntrain)),
          "RMSE test: ", np.sqrt(val_error.detach().numpy()/((8000-ntrain))))
    if val_error< best:
        best = val_error
        torch.save({
            'epoch':epoch,
            'model_state_dict': enmodel.state_dict(),
            'optimizer_state_dict': energy_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss':train_error,
            'valerror':val_error
        }, './models/batchmc2_single_p0-lin-best-NU3.pt')

