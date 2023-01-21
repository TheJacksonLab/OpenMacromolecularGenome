# Open Macromolecular Genome

This repository contains python scripts to construct the Open Macromolecular Genome (OMG) database from [eMolecules](https://www.emolecules.com/) and train generative models.

<p align="center">
<img src="https://github.com/TheJacksonLab/OpenMacromolecularGenome/blob/main/data/figure/schematic_diagram.jpg" width="500" height="500">
</p>

## Set up Python environment with Anaconda 
```
conda env create -f environment.yml
``` 

## Script components
To run a script, a file path in the script should be modified to be consistent with an attempted directory.

### 1. data
This directory contains scripts to construct the OMG database from [eMolecules](https://www.emolecules.com/).

### 2. scscore
This directory contains a script to calculate SC score obtained from https://github.com/connorcoley/scscore.

### 3. polymerization
This directory contains the OMG polymerization algorithms.

### 4. selfies 
This directory contains modified SELFIES scripts to incorporate asterisk (*). The asterisk rules were added to the original work, https://github.com/aspuru-guzik-group/selfies

### 5. molecule_chef 
This directory contains the Molecule Chef generative model. The scripts were written referring to the original work,
[Bradshaw, J.; Paige, B.; Kusner, M. J.; Segler, M.; Hernández-Lobato, J. M. A Model to Search for Synthesizable Molecules. 
In Advances in Neural Information Processing Systems; Curran Associates, Inc., 2019; Vol. 32.](https://arxiv.org/abs/1906.05221), 
and their scripts, https://github.com/john-bradshaw/molecule-chef.

### 6. vae 
This directory contains a variational autoencoder model. The scripts were written referring to https://github.com/aspuru-guzik-group/selfies/blob/master/examples/vae_example/chemistry_vae.py

### 7. train
This directory contains scripts to train Molecule Chef and SELFIES VAE. These scripts were written by referring to 
https://github.com/john-bradshaw/molecule-chef and https://github.com/aspuru-guzik-group/selfies/blob/master/examples/vae_example/chemistry_vae.py.

## Authors
Seonghwan Kim, Charles M. Schroeder, and Nicholas E. Jackson

## Funding Acknowledgements
This work was supported by the IBM-Illinois Discovery Accelerator Institute. N.E.J. thanks the 3M Nontenured Faculty Award for support of this research. We thank Jed Pitera and Jeffrey Moore for critical readings of the manuscript and Prof. Tengfei Luo for assistance with the PI1M dataset.

<p align="right">
<img src="https://github.com/TheJacksonLab/OpenMacromolecularGenome/blob/main/data/figure/OMG.png" width="200" height="60"> 
</p>

