This subdirectory contains the scripts to fit the single-center (ACDC) features with correlation order upto 7(which should be complete for this dataset) using the three-center features. 
The encoder-decoder architectures for this reconstruction differs in the transformations employed in the encoder. (The same decoder is used for all models) 

Linearly encoded models employed just a single linear layer to contract the input, whereas a nonlinear encoder sequentially contracted the input to the final latent size over three layers. The first linear layer compressed the dimension to a hidden size of 512 neurons, following which the output was normalized (using LayerNorm) and transformed with a SiLU nonlinearity. The second layer preserved the hidden dimension but also involved normalization and nonlinearity. The final layer condensed the features to the latent space size (fixed to be 128 in all models). Both the encoders described above were subsequently used with a decoder comprising three linear layers interspersed with a normalization layer (LayerNorm) and
SiLU nonlinearities as in the case of the nonlinear encoder, except that the dimensions sequentially increased from 128 to 512 to the output size

1. `fit_lin.py` corresponds to the A_{L}^{\rho} model (linear model built on N=2, nu=1 features to reconstruct n=7 features)  
2. `fit_nl.py` corresponds to the A_{NL}^{\rho} model (non-linear model built on N=2 nu=1 features)


We further use the encoded features to build models for the energy

As per the main paper, 


1. `fit_energy_lin.py` corresponds to the A_{L} model (non-linear model built on encoded features from best A_{L}^{\rho} to target energies)  
2. `fit_energy_nl.py` corresponds to the A_{NL} model (non-linear model built on encoded features from best A_{NL}^{\rho} to target energies)

Both the energy models themselves are non-linear, as detailed in `../energy/README.md'

`feat-settings.py` paths to the data files that are used for training the models
