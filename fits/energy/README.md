This subdirectory contains the scripts to fit the energies directly using the single-center (ACDC) features with correlation order ($\nu$) upto 3 (bispectrum) which is incomplete, using ACDC features upto 7(which should be complete for this dataset), and using the three-center features. As per the main paper, 

1. `fit_triple_lin.py` corresponds to the B_{L} model (linear model built on N=2, nu=1 features)  
2. `fit_triple_nl.py` corresponds to the B_{NL} model (non-linear model built on N=2 nu=1 features)
3. `fit_nu7_lin.py` corresponds to the C_{L} model (linear model built on nu=7 features) 
4. `fit_nu7_nl.py` corresponds to the C_{NL} model (non-linear model built on nu=7 features)
5. `fit_nu3_lin.py` corresponds to the D_{L} model (linear model built on nu=3 features)
6. `fit_nu3_nl.py` corresponds to the D_{NL} model (non-linear model built on nu=3 features)

The linear models consist of two linear layers sequentially reducing the input features to the a latent-space size of 128 features and finally to a single scalar output per structure.
Nonlinear models on the other hand first compress the features to a hidden size of 128 following which a SiLU nonlinearity is applied and the procedure is repeated for another intermediate layer before producing a single energy output.

`feat-settings.py` paths to the data files that are used for training the models
