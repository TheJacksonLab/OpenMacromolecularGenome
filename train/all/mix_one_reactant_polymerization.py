import os
import sys
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import RWMol, BondType, Atom

from itertools import combinations


def extend_polymer(p_smile_list):
    mol_1 = Chem.MolFromSmiles(p_smile_list[0])
    mol_2 = Chem.MolFromSmiles(p_smile_list[1])
    mw_1 = RWMol(mol_1)
    mw_2 = RWMol(mol_2)

    # store
    asterisk_idx = list()
    mol_1_del_list = list()
    mol_2_del_list = list()

    # find asterisk idx
    for idx, atom in enumerate(mol_1.GetAtoms()):
        if atom.GetSymbol() == '*':
            asterisk_idx.append(idx)
    mol_1_del_list.append(asterisk_idx[1])
    mol_2_del_list.append(asterisk_idx[0])

    # modify index of monomer 2
    modified_mol_2_del_list = [idx + mol_1.GetNumAtoms() for idx in mol_2_del_list]

    # combine
    new_polymer = RWMol(Chem.CombineMols(mw_1, mw_2))
    new_polymer.AddBond(mol_1_del_list[0], modified_mol_2_del_list[0], BondType.SINGLE)

    # rearrange atom idx
    new_polymer_smi = Chem.MolToSmiles(new_polymer)
    asterisk_idx_smi = list()
    for idx, char in enumerate(new_polymer_smi):
        if char == '*':
            asterisk_idx_smi.append(idx)
    asterisk_idx_smi = asterisk_idx_smi[1:-1]
    new_polymer_smi = new_polymer_smi[:asterisk_idx_smi[0]] + new_polymer_smi[asterisk_idx_smi[1] + 1:]

    return Chem.CanonSmiles(new_polymer_smi)


def get_sub_df(reactant_bags, product_bags, mix_reaction_idx, random_seq):
    reactant_1_list = list()
    reactant_2_list = list()
    product_list = list()

    for idx in random_seq:
        reactant_bag = reactant_bags[idx]
        product_bag = product_bags[idx]
        print(idx, flush=True)
        repeat_unit = extend_polymer(product_bag)
        if repeat_unit is None:
            print("Merged repeat unit is None", flush=True)
            exit()

        # canonical form
        mol = Chem.MolFromSmiles(repeat_unit)
        if mol is None:  # *C(=O)CC(=O)CC(=O)[NH2:8][CH2:7][c:6]1[cH:1][cH:2][n:3][c:4]([NH2:9]*)[cH:5]1
            print("Merged repeat unit Mol is None", flush=True)
            exit()

        p_smi = Chem.MolToSmiles(mol)
        product_list.append(p_smi)

        # append
        reactant_1_list.append(reactant_bag[0])
        reactant_2_list.append(reactant_bag[1])

    sub_df = pd.DataFrame(
        {'reaction_idx': [mix_reaction_idx] * len(reactant_1_list), 'reactant_1': reactant_1_list,
         'reactant_2': reactant_2_list, 'product': product_list}
    )

    return sub_df


if __name__ == '__main__':
    # This code mixes one-reactant polymers (not containing the self-coupling) -> N*(N-1)/2 combinations
    # environmental variables
    load_directory = '/home/sk77/PycharmProjects/publish/OMG/Data'
    save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all'

    # load OMG polymers
    df = pd.read_csv(os.path.join(load_directory, 'OMG_polymers.csv'))
    df_reactant_bags = pd.DataFrame(columns=['reaction_idx', 'reactant_1', 'reactant_2', 'product'])
    sample_number_dict = {
        7: 20000,
        8: 30000,
        9: 20000,
        10: 3000,
        11: 3000,
        12: 6000,
        13: 3000,
        14: 500,  # cyclic carbonate -> total case is 1,000
        15: 3000,
        16: 3000,
        17: 20000,
    }

    # mixing one-reactant polymers
    for reaction_idx in range(7, 17 + 1):
        df_one_reactant = df[df['reaction_idx'] == reaction_idx]

        # mix one-reactant polymers
        print(f"Mixing {reaction_idx} starts", flush=True)
        product = df_one_reactant['product'].tolist()
        reactant = df_one_reactant['reactant_1'].tolist()  # reactant_2 is the same as the reactant_1

        product_bags = list(combinations(product, r=2))
        reactant_bags = list(combinations(reactant, r=2))
        length = len(product_bags)

        # random sequence
        sample_number = sample_number_dict[reaction_idx]
        random_arr = np.random.RandomState(seed=42).choice(a=length, size=sample_number, replace=False)

        # construct mixed reactant bags
        df_mixed = get_sub_df(
            reactant_bags=reactant_bags,
            product_bags=product_bags,
            mix_reaction_idx=reaction_idx,
            random_seq=random_arr
        )
        df_reactant_bags = pd.concat([df_reactant_bags, df_mixed], axis=0)
        print(f"Mixing {reaction_idx} is done", flush=True)

    # save
    print(df_reactant_bags.shape)
    print(df_reactant_bags.head().to_string())
    df_reactant_bags.to_csv(
        os.path.join(save_directory, 'mixed_one_reaction_30000_20000_10000_6000_3000_500.csv'), index=False
    )
