feat_prefix = "n6l3"

# files
root = "../"
frames_file = root + "data/"+"boron8_8000_pbeccpvdz.xyz"

feat_file = feat_prefix+ '/'

# feature calculation parameters

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
frames_slice = slice(0, 8000) # '4000:8000'
