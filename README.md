# Open Macromolecular Genome

This repository contains python scripts to construct Open Macromolecular Genome (OMG) data from eMolecules and train generative models.

<p align="center">
<img src="https://github.com/TheJacksonLab/OpenMacromolecularGenome/blob/main/data/figure/schematic_diagram.jpg" width="300" height="300">
</p>

## Python environment
```
conda env create -f environment.yml
```
To run a script, a file path in the script should be modified to be consistent with a user environment.  

## Script components
### 1. data
This directory contains scripts to construct the OMG database from eMolecules.

### 2. molecule_chef 
This directory contains the Molecule Chef generative model. The scripts were written referring to the original work,
[Bradshaw, J.; Paige, B.; Kusner, M. J.; Segler, M.; Hernández-Lobato, J. M. A Model to Search for Synthesizable Molecules. 
In Advances in Neural Information Processing Systems; Curran Associates, Inc., 2019; Vol. 32.](https://arxiv.org/abs/1906.05221), 
and their scripts, https://github.com/john-bradshaw/molecule-chef. 

### 3. polymerization
This directory contains the OMG polymerization algorithms based.

### 4. scscore
This directory contains a script to calculate SC score obtained from https://github.com/connorcoley/scscore.

### 5. vae 
This directory contains a variational autoencoder model. The scripts were written referring to https://github.com/aspuru-guzik-group/selfies/blob/master/examples/vae_example/chemistry_vae.py

### 6. train
This directory contains scripts to train Molecule Chef and SELFIES VAE. These scripts were also written by referring to 
https://github.com/john-bradshaw/molecule-chef and https://github.com/aspuru-guzik-group/selfies/blob/master/examples/vae_example/chemistry_vae.py. 

### 7. SELFIES 
This directory contains modified SELFIES scripts to incorporate asterisk (*). The asterisk rules were added to the original work, https://github.com/aspuru-guzik-group/selfies  

## Authors
Seonghwan Kim, Charles M. Schroeder, and Nicholas E. Jackson

## Funding Acknowledgements
This work was supported by the IBM-Illinois Discovery Accelerator Institute. N.E.J. thanks the 3M Nontenured Faculty Award for support of this research. We thank Jed Pitera and Jeffrey Moore for critical readings of the manuscript and Prof. Tengfei Luo for assistance with the PI1M dataset.

<p align="right">
<img src="https://github.com/TheJacksonLab/OpenMacromolecularGenome/blob/main/data/figure/OMG.png" width="200" height="60"> 
</p>

