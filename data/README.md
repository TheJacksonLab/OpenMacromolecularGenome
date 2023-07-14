This directory contains .py scripts to generate the OMG database.
### 1. `./script/preprocess.py`
This script extracts monomer reactants having polymerization functional groups from the eMolecule database. 
Please note that the eMolecule raw data (`version.smi`) should be placed in the `data` directory. The `version.smi` can be downloaded at https://marketing.emolecules.com/incremental-file-download. 
The detailed usage can be checked by `python preprocess.py -h`

### 2. `./get_reactant_bags_reactions_script`
This directory contains .py scripts to generate the OMG polymer database using the monomer reactants extracted from the eMolecules database. 
These scripts combine monomer reactants to satisfy predefined 17 polymerization mechanisms as described in the paper. 