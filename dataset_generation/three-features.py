from ase.io import read, write
import hickle
from sklearn.decomposition import PCA
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
import copy 
import numpy as np
import scipy as sp
from equistore.io import load,save
from equistore import Labels, TensorBlock, TensorMap
from itertools import product
from equistore_utils.clebsh_gordan import ClebschGordanReal
from rascaline import SphericalExpansion
from rascaline import SphericalExpansionByPair as PairExpansion
from equistore import operations
import sys, os
sys.path.append(os.getcwd())
from feat_settings import *
cg = ClebschGordanReal(5)
from equistore_utils.mp_utils import *

import scipy
frames = read(frames_file, frames_slice)
print(len(frames))
#
calculator = SphericalExpansion(**hypers)
rhoi = calculator.compute(frames)
rhoi = rhoi.keys_to_properties(['species_neighbor'])
rho1i = acdc_standardize_keys(rhoi)
## selects only one environment
rho1i = operations.slice(rho1i, samples=Labels(['center'],np.array([[0]], np.int32)) )
##norm_rho1 = np.sqrt(np.sum([(b.values**2).sum(axis=(1,2)) for b in rho1i.blocks()],axis=0).mean())
##for b in rho1i.blocks():
##     b.values[:]/=norm_rho1
##print(np.sqrt(np.sum([(b.values**2).sum(axis=(1,2)) for b in rho1i.blocks()],axis=0).mean()))
#
calculator = PairExpansion(**hypers)
gij = calculator.compute(frames)
gij = operations.slice(gij, samples=Labels(['first_atom'],np.array([[0]], np.int32)) )
gij =  acdc_standardize_keys(gij)
##for b in gij.blocks():
##    b.values[:]/=norm_rho1
#
test = operations.sum_over_samples(gij, samples_names=["neighbor"]) 
print("test (should be zero)", np.linalg.norm(rho1i.block(0).values/test.block(0).values -1))
#
# Compute feature 
rhoii1i2_nu0 = cg_combine(gij, gij, clebsch_gordan=cg, other_keys_match=['species_center'], lcut=3)
rhoii1i2_nu1 =  cg_combine(rho1i, rhoii1i2_nu0, clebsch_gordan=cg, other_keys_match = ['species_center'], lcut=0)
save(feat_file+ 'rhoii1i2_nu0', rhoii1i2_nu0 )
save(feat_file+ 'rhoii1i2_nu1', rhoii1i2_nu1 )
np.save(feat_file + 'triple_samples_names', rhoii1i2_nu1.block(0).samples.names)
np.save(feat_file + 'triple_samples', rhoii1i2_nu1.block(0).samples.asarray())

#Perform PCA
raw = np.hstack([rhoii1i2_nu1.block(0).values.squeeze(),rhoii1i2_nu1.block(1).values.squeeze()])
print(raw.shape)
feats_n2nu1 = PCA(n_components=min(raw.shape[0],raw.shape[-1])).fit_transform(raw)
hickle.dump (raw, feat_file+ 'feat_3cnu1.hickle')
hickle.dump(feats_n2nu1, feat_file+'feat_3cnu1_PCA.hickle')

