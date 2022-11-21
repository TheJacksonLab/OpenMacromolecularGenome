import os
import pandas as pd

from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Descriptors import NumRadicalElectrons


def atomic_num_check(mol, min_atomic_number, max_atomic_number):
    """
    This function screens if a molecule passes the OMG atom type standardization test
    Allowed atomic list = [0, 1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 35]
    :param mol: RDkit mol object
    :param min_atomic_number: minimum atomic number allowed in a keep list
    :param max_atomic_number: maximum atomic number allowed in a keep list
    :return: 0 if passes, 1 if fails
    """
    # making keep list
    keep_list = [0, 1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 35]
    for value in keep_list:
        if value < min_atomic_number or value > max_atomic_number:
            keep_list.remove(value)
    # screen
    num_atm = mol.GetNumAtoms()
    flag = 0
    for i in range(num_atm):
        atomic_num = mol.GetAtomWithIdx(i).GetAtomicNum()
        if atomic_num not in keep_list:
            flag = 1
            break
    return flag


def check_isotope(mol):
    cnt = 0
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            cnt += 1
            break
    return cnt


def classify_monomers_to_csv(df_smi: pd.DataFrame, sub_structure_dict: dict, save_directory, iteration):
    # set save directory
    save_directory = os.path.join(os.getcwd(), save_directory)
    if not os.path.exists(save_directory):
        Path(save_directory).mkdir(parents=True)

    # make df_smi
    df_smi = df_smi.rename(columns={'isosmiles': 'smiles'})
    df_smi = df_smi.dropna(axis=0)

    # filter smiles containing '.'
    df_smi['point'] = df_smi['smiles'].apply(lambda x: '.' in x)
    df_smi = df_smi[~df_smi['point']]

    # convert to mol objects
    df_smi['mol'] = df_smi['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df_smi = df_smi.dropna(axis=0)

    # canonical smiles without stereochemistry
    df_smi = df_smi.drop(labels=['smiles'], axis=1)
    df_smi['smiles'] = df_smi['mol'].apply(lambda x: Chem.MolToSmiles(x, isomericSmiles=False))
    df_smi = df_smi.reset_index(drop=True)

    # filter explicit hydrogen and atom index mappings
    df_smi['flag'] = df_smi['smiles'].apply(lambda x: ('[H]' in x) | (':' in x))
    df_smi = df_smi[df_smi['flag'] == 0]

    # filter non-zero formal charges - remove positive
    df_smi['flag'] = df_smi['smiles'].apply(lambda x: '+' in x)
    df_smi = df_smi[df_smi['flag'] == 0]

    # filter non-zero formal charges - remove negative
    df_smi['flag'] = df_smi['mol'].apply(lambda x: Chem.GetFormalCharge(x))
    df_smi = df_smi[df_smi['flag'] == 0]

    # remove radicals
    df_smi['flag'] = df_smi['mol'].apply(lambda x: NumRadicalElectrons(x))
    df_smi = df_smi[df_smi['flag'] == 0]

    # atomic number check
    df_smi['flag'] = df_smi['mol'].apply(
        lambda x: atomic_num_check(x, min_atomic_number=1, max_atomic_number=35)
    )
    df_smi = df_smi[df_smi['flag'] == 0]

    # exception - hydrogen isotopes
    df_smi['flag'] = df_smi['mol'].apply(lambda x: check_isotope(x))
    df_smi = df_smi[df_smi['flag'] == 0]

    # drop duplicates
    df_smi = df_smi.drop_duplicates(subset=['smiles'])

    # recognize substructure
    for key, value in sub_structure_dict.items():
        sub_structure_mol = Chem.MolFromSmarts(value)
        df_smi['%s' % key] = df_smi['mol'].apply(lambda x: len(x.GetSubstructMatches(sub_structure_mol)))

    # hydroxy_carboxylic_acid should have both OH and COOH
    target = ['hydroxy_carboxylic_acid_OH', 'hydroxy_carboxylic_acid_COOH']
    df_smi['hydroxy_carboxylic_acid'] = df_smi[target[0]] & df_smi[target[1]]

    # drop unnecessary columns
    df_smi = df_smi.drop(labels=[target[0], target[1]], axis=1)
    df_smi = df_smi.drop(labels=['mol'], axis=1)
    df_smi = df_smi.drop(labels=['version_id', 'parent_id', 'point'], axis=1)
    df_smi = df_smi.drop(labels=['flag'], axis=1)

    # save .csv files
    df_smi.to_csv(os.path.join(save_directory, f'batch_{iteration}.csv'))
