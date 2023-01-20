# Open Macromolecular Genome

##### This repository contains python scripts to construct Open Macromolecular Genome (OMG) data from eMolecules and train generative models.

[comment]: <> (![alt text]&#40;https://github.com/TheJacksonLab/OpenMacromolecularGenome/blob/main/data/figure/schematic_diagram.jpg?raw=True&#41;)
<p align="center">
<img src="C:\Users\wpxl3\git\data\figure\schematic_diagram.jpg" width="500" height="500">
</p>
[comment]: <> (![alt text]&#40;C:\Users\wpxl3\git\data\figure\schematic_diagram.jpg?=300x50&#41;)

## Python environment
```
conda env create -f environment.yml
```
To run a script, a file path in the script should be modified to be consistent with a user environment.  

## Script components
### 1. data
This directory contains scripts to construct the OMG database from eMolecules.

### 2. molecule_chef -> based on the original work
This directory contains the Molecule Chef generative model. The scripts were written referring to the original work,
Bradshaw, J.; Paige, B.; Kusner, M. J.; Segler, M.; Hernández-Lobato, J. M. A Model to Search for Synthesizable Molecules. 
In Advances in Neural Information Processing Systems; Curran Associates, Inc., 2019; Vol. 32. https://github.com/john-bradshaw/molecule-chef 

### 3. polymerization
### 4. scscore
### 5. vae -> based on the SELFIES VAE
### 6. train
### SELFIES -> added asterisk

## Authors

## Funding Acknowledgements

