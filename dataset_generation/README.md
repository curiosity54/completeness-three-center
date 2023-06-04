This subdirectory contains the scripts to 

1. generate bispectrum degenerate pairs using the methodology described in Ref. [1]
2. generate the bispectrum degenerate dataset for boron (B8), perform DFT calculations using PySCF, `dft_scripts.py`
3. generate and compress  N=2, nu=1 (3 center, 1 neighbor) features, `three-features.py`
4. generate and perform feature compression on the nu=7 single atom-centered features, `get-features.py` using the scheme of NICE [2]

`feat-settings.py` contains the hyperparameters and the paths to the data files that are used for the feature generation. 

[1] S. Pozdnyakov et al. Incompleteness of atomic structure representations, Physical Review Letters. 2020;125(16):166001.
[2] J. Nigam, S. Pozdnyakov, M. Ceriotti, N-body Iterative COntraction of Equivariants, J. Chem. Phys. 2020; 153 (12): 121101. https://doi.org/10.1063/5.0021116
