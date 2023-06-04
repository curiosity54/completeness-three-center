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

def compute_rhoi_pca(rhoi, npca):
    """ computes PCA contraction with combined elemental and radial channels.
    returns the contraction matrices
    """
    if isinstance(npca, list): 
        assert len(npca) == len(rhoi)
    else: 
        npca = [npca]*len(rhoi)
    pca_vh_all = []
    s_sph_all = []
    pca_blocks = []
    for idx, (key, block) in enumerate(rhoi):
        nu, sigma, l, spi = key
        xl = block.values.reshape((len(block.samples)*len(block.components[0]),-1))
#         print(xl.shape)
        u, s, vt = scipy.sparse.linalg.svds(xl, k=(min(min(xl.shape)-1,npca[idx]*2)), return_singular_vectors='vh')
#          k=(min(xl.shape)-1)
#         k=(min(min(xl.shape)-1,npca*2))
        s_sph_all.append(s[::-1])
        pca_vh_all.append(vt[-npca[idx]:][::-1].T)
#         print(s.shape, vt.shape)
        print("singular values", s[::-1][:npca[idx]]/s[::-1][0])
        pblock = TensorBlock( values = vt[-npca[idx]:][::-1].T ,
                                 components = [],
                                 samples = Labels(["pca"], np.asarray([i for i in range(len(block.properties))], dtype=np.int32).reshape(-1,1)),
                                 properties = Labels(["pca"], np.asarray([i for i in range(vt[-npca[idx]:][::-1].T.shape[-1])], dtype=np.int32).reshape(-1,1))
                                )
        pca_blocks.append(pblock)
    pca_tmap = TensorMap(rhoi.keys, pca_blocks)
    return pca_tmap, pca_vh_all, s_sph_all

def get_pca_tmap(rhoi, pca_vh_all):
    assert len(rhoi.keys) == len(pca_vh_all)
    pca_blocks = []
    for idx, (key, block) in enumerate(rhoi):
        vt = pca_vh_all[idx]
        print(vt.shape)
        pblock = TensorBlock( values = vt ,
                                 components = [], 
                                 samples = Labels(["pca"], np.asarray([i for i in range(len(block.properties))], dtype=np.int32).reshape(-1,1)),
                                 properties = Labels(["pca"], np.asarray([i for i in range(vt.shape[-1])], dtype=np.int32).reshape(-1,1))
                                )   
        pca_blocks.append(pblock)
    pca_tmap = TensorMap(rhoi.keys, pca_blocks)
    return pca_tmap

def apply_pca(rhoi, pca_tmap):
    new_blocks = []
    for idx, (key, block) in enumerate(rhoi):
        nu, sigma, l, spi = key
        xl = block.values.reshape((len(block.samples)*len(block.components[0]),-1))
        vt = pca_tmap.block(spherical_harmonics_l = l, inversion_sigma = sigma).values
        xl_pca = (xl@vt).reshape((len(block.samples),len(block.components[0]),-1))
#         print(xl_pca.shape)
        pblock = TensorBlock( values = xl_pca,
                                 components = block.components,
                                 samples = block.samples,
                                 properties = Labels(["pca"], np.asarray([i for i in range(xl_pca.shape[-1])], dtype=np.int32).reshape(-1,1))
                                )
        new_blocks.append(pblock)
    pca_tmap = TensorMap(rhoi.keys, new_blocks)
    return pca_tmap

frames = read(frames_file, frames_slice)

calculator = SphericalExpansion(**hypers)
rhoi = calculator.compute(frames)
rhoi = rhoi.keys_to_properties(['species_neighbor'])
rho1i = acdc_standardize_keys(rhoi)
## selects only one environment
rho1i = operations.slice(rho1i, samples=Labels(['center'],np.array([[0]], np.int32)) )
##norm_rho1 = np.sqrt(np.sum([(b.values**2).sum(axis=(1,2)) for b in rho1i.blocks()],axis=0).mean())
##for b in rho1i.blocks():
##     b.values[:]/=norm_rho1
##print(norm_rho1)    
save(feat_file+'rho1i', rho1i )
print("rho1i.shape" , rho1i.block(0).values.shape)
#
rho2i = cg_increment(rho1i, rho1i, clebsch_gordan=cg, lcut=lcut, other_keys_match=["species_center"])
print(rho2i.keys, len(rho2i.keys))
rho2i_forpca = operations.slice(rho2i, samples=Labels(['structure'],np.array(list(range(0,8000,5)), np.int32).reshape(-1,1)) )
rho2i_projection, rho2i_vh_blocks, rho2i_eva_blocks = compute_rhoi_pca(rho2i_forpca, npca=[25,25,40,25, 15,20,22])
print("rho2i eva", rho2i_eva_blocks)
rho2i_pca= apply_pca(rho2i, rho2i_projection)
save(feat_file+'rho2i_pca', rho2i_pca )
np.save(feat_file+ 'rho2i_vh_blocks.npy',rho2i_vh_blocks)
np.save(feat_file+ 'rho2i_eva_blocks.npy', rho2i_eva_blocks)

#
rho3i = cg_increment(rho2i_pca, rho1i, clebsch_gordan=cg, lcut=lcut, other_keys_match=["species_center"])
rho3i_forpca = operations.slice(rho3i, samples=Labels(['structure'],np.array(list(range(0,8000,5)), np.int32).reshape(-1,1)) )
rho3i_projection, rho3i_vh_blocks, rho3i_eva_blocks = compute_rhoi_pca(rho3i_forpca, npca=[80, 180, 200, 200, 120, 120, 180,30])
print("rho3i eva", rho3i_eva_blocks)
#
rho3i_pca= apply_pca(rho3i, rho3i_projection)
save(feat_file+'rho3i_pca', rho3i_pca )
np.save(feat_file+ 'rho3i_vh_blocks.npy',rho3i_vh_blocks)
np.save(feat_file+ 'rho3i_eva_blocks.npy', rho3i_eva_blocks)

#
rho4i = cg_increment(rho3i_pca, rho1i, clebsch_gordan=cg, lcut=3, other_keys_match=["species_center"])
rho4i_forpca = operations.slice(rho4i, samples=Labels(['structure'],np.array(list(range(0,8000,5)), np.int32).reshape(-1,1)) )
rho4i_projection, rho4i_vh_blocks, rho4i_eva_blocks = compute_rhoi_pca(rho4i, npca=[150, 500, 400, 500, 500, 500, 600,120])
print("rho4i eva", rho4i_eva_blocks)
rho4i_pca= apply_pca(rho4i, rho4i_projection)
save(feat_file+'rho4i_pca', rho4i_pca )
np.save(feat_file+ 'rho4i_vh_blocks.npy',rho4i_vh_blocks)
np.save(feat_file+ 'rho4i_eva_blocks.npy', rho4i_eva_blocks)

#
rho5i = cg_increment(rho4i_pca, rho1i, clebsch_gordan=cg, lcut=3, other_keys_match=["species_center"])
rho5i_forpca = operations.slice(rho5i, samples=Labels(['structure'],np.array(list(range(0,8000,8)), np.int32).reshape(-1,1)) )
rho5i_projection, rho5i_vh_blocks, rho5i_eva_blocks = compute_rhoi_pca(rho5i_forpca, npca=[200, 700, 700, 700, 600, 600, 700,150])
print("rho5i eva", rho5i_eva_blocks)
rho5i_pca= apply_pca(rho5i, rho5i_projection)
save(feat_file+'rho5i_pca', rho5i_pca )
np.save(feat_file+ 'rho5i_vh_blocks.npy',rho5i_vh_blocks)
np.save(feat_file+ 'rho5i_eva_blocks.npy', rho5i_eva_blocks)


rho6i = cg_increment(rho5i_pca, rho1i, clebsch_gordan=cg, lcut=3, other_keys_match=["species_center"])
rho6i_forpca = operations.slice(rho6i, samples=Labels(['structure'],np.array(list(range(0,8000,10)), np.int32).reshape(-1,1)) )
rho6i_projection, rho6i_vh_blocks, rho6i_eva_blocks = compute_rhoi_pca(rho6i, npca= [300, 800, 800, 800, 800, 800, 800,250])
rho6i_pca= apply_pca(rho6i, rho6i_projection)
save(feat_file+'rho6i_pca', rho6i_pca )
np.save(feat_file+ 'rho6i_vh_blocks.npy',rho6i_vh_blocks)
np.save(feat_file+ 'rho6i_eva_blocks.npy', rho6i_eva_blocks)

rho7i = cg_increment(rho6i_pca, rho1i, clebsch_gordan=cg, lcut=0, other_keys_match=["species_center"])
save(feat_file+ 'rho7i', rho7i)

# Perform final feature PCA to retain as many features as the number of structures

raw = np.hstack([
    rho1i.block(inversion_sigma=1,spherical_harmonics_l=0).values.squeeze(),
    rho2i_pca.block(inversion_sigma=1,spherical_harmonics_l=0).values.squeeze(),
    rho3i_pca.block(inversion_sigma=1,spherical_harmonics_l=0).values.squeeze(),
    rho3i_pca.block(inversion_sigma=-1,spherical_harmonics_l=0).values.squeeze(),
    rho4i_pca.block(inversion_sigma=1,spherical_harmonics_l=0).values.squeeze(),
    rho4i_pca.block(inversion_sigma=-1,spherical_harmonics_l=0).values.squeeze(),
    rho5i_pca.block(inversion_sigma=1,spherical_harmonics_l=0).values.squeeze(),
    rho5i_pca.block(inversion_sigma=-1,spherical_harmonics_l=0).values.squeeze(),
    rho6i_pca.block(inversion_sigma=1,spherical_harmonics_l=0).values.squeeze(),
    rho6i_pca.block(inversion_sigma=-1,spherical_harmonics_l=0).values.squeeze(),
    rho7i.block(inversion_sigma=1,spherical_harmonics_l=0).values.squeeze(),
    rho7i.block(inversion_sigma=-1,spherical_harmonics_l=0).values.squeeze()
])
print(raw.shape)
hickle.dump (raw, feat_file+ 'feat_1234567.hickle')
feats = PCA(n_components=min(raw.shape[0],raw.shape[-1])).fit_transform(raw) 
hickle.dump(feats, feat_file + 'feat_1234567_PCA.hickle')

