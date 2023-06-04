feat_prefix = "../../dataset_generation/n6l3" # Replace by path to features

# files
root = "../../data"
frames_file = root + "boron_8000_pbeccpvdz.xyz"

feat_file = feat_prefix+ '/' 

# feature calculation parameters
pca_stride = 2 

hypers = { 
    "cutoff": 2.0,
    "max_radial": 6,
    "max_angular": 3,
    "atomic_gaussian_width": 0.2,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
    "center_atom_weight": 0.0,    
}
lcut = 3 

