import os
import sys
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import product

sys.path.append('/home/sk77/PycharmProjects/polymer')
from polymerization import BasePolymerization, Polymerization


reaction_idx_dict = {
    '[step_growth]_[di_amine]_[di_carboxylic_acid]': 1,
    '[step_growth]_[di_acid_chloride]_[di_amine]': 2,
    '[step_growth]_[di_carboxylic_acid]_[di_ol]': 3,
    '[step_growth]_[di_acid_chloride]_[di_ol]': 4,
    '[step_growth]_[di_amine]_[di_isocyanate]': 5,
    '[step_growth]_[di_isocyanate]_[di_ol]': 6,
    '[step_growth]_[hydroxy_carboxylic_acid]': 7,
    '[chain_growth]_[vinyl]': 8,
    '[chain_growth]_[acetylene]': 9,
    '[chain_growth_ring_opening]_[lactone]': 10,
    '[chain_growth_ring_opening]_[lactam]': 11,
    '[chain_growth_ring_opening]_[cyclic_ether]': 12,
    '[chain_growth_ring_opening]_[cyclic_olefin]': 13,
    '[chain_growth_ring_opening]_[cyclic_carbonate]': 14,
    '[chain_growth_ring_opening]_[cyclic_sulfide]': 15,
    '[metathesis]_[terminal_diene]': 16,
    '[metathesis]_[conjugated_di_bromide]': 17,
}


def get_sub_df(reactant_bags):
    reaction_idx_list = list()
    reactant_1_list = list()
    reactant_2_list = list()
    product_list = list()
    for idx, reactant_bag in enumerate(reactant_bags):
        print(idx, flush=True)
        repeat_unit = reactor.polymerize(reactant_bag)
        if repeat_unit is None:
            continue

        # for save reaction name
        reaction_name_list = list(repeat_unit[1].keys())
        reaction_name_list += list(repeat_unit[1].values())[0]

        # append
        reactant_1_list.append(reactant_bag[0])
        if len(reactant_bag) == 1:
            reactant_2_list.append(reactant_bag[0])
        else:
            reactant_2_list.append(reactant_bag[1])

        # canonical form
        mol = Chem.MolFromSmiles(repeat_unit[0])
        if mol is None:
            print(reactant_bag)
            print(repeat_unit[0])
            exit()
        p_smi = Chem.MolToSmiles(mol)
        product_list.append(p_smi)

        # get reaction name
        reaction_name = ''
        for name in reaction_name_list:
            reaction_name += '_' + '[' + name + ']'
        reaction_name = reaction_name[1:]

        reaction_idx_list.append(reaction_idx_dict[reaction_name])

    sub_df = pd.DataFrame(
        {'reaction_idx': reaction_idx_list, 'reactant_1': reactant_1_list,
         'reactant_2': reactant_2_list, 'product': product_list}
    )

    return sub_df


if __name__ == '__main__':
    # load eMolecule reactants
    save_directory = sys.argv[1]
    df = pd.read_csv(os.path.join(
        save_directory, sys.argv[2]
    )).reset_index(drop=True)
    print(df.shape, flush=True)

    # add mol columns
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

    # add class columns
    base_polymerization = BasePolymerization()
    for key, value in base_polymerization._sub_structure_dict.items():
        sub_structure_mol = Chem.MolFromSmarts(value)
        df['%s' % key] = df['mol'].apply(lambda x: len(x.GetSubstructMatches(sub_structure_mol)))

    # drop mol columns
    df = df.drop('mol', axis=1)

    # call polymerization class
    reactor = Polymerization()

    # construct monomer reactant bags
    df_reactant_bags = pd.DataFrame(columns=['reaction_idx', 'reactant_1', 'reactant_2', 'product'])

    # sort monomer reactants #
    # 3) ring opening
    lactam = df[df['lactam'] == 1]
    cyclic_ether = df[df['cyclic_ether'] == 1]
    cyclic_olefin = df[df['cyclic_olefin'] == 1]
    cyclic_carbonate = df[df['cyclic_carbonate'] == 1]
    cyclic_sulfide = df[df['cyclic_sulfide'] == 1]

    # 4) metathesis
    terminal_diene = df[df['terminal_diene'] == 2]
    conjugated_di_bromide = df[df['conjugated_di_bromide'] == 2]

    # construct monomer reactant bags #
    '''
    Note: Reaction index below may not be accurate. The accurate idx is obtained from polymerization function.
    e.g) Molecule can be thought of cyclic ether, but lactone in reality
    '''

    # 11) Ring opening of lactam
    reaction_idx = 11
    print(f"Reaction {reaction_idx} starts", flush=True)
    target_list = lactam['smiles'].tolist()
    reactant_bags = [[reactant] for reactant in target_list]
    df_sub = get_sub_df(reactant_bags=reactant_bags)
    df_reactant_bags = pd.concat([df_reactant_bags, df_sub], axis=0).reset_index(drop=True)
    print(f"Reaction {reaction_idx} is done", flush=True)
    df_reactant_bags.to_csv(
        os.path.join(save_directory, f'eMolecule_reactant_bags_reaction_idx_{reaction_idx}.csv'), index=False
    )

    # 12) Ring opening of cyclic ether
    reaction_idx = 12
    target_list = cyclic_ether['smiles'].tolist()
    reactant_bags = [[reactant] for reactant in target_list]
    df_sub = get_sub_df(reactant_bags=reactant_bags)
    df_reactant_bags = pd.concat([df_reactant_bags, df_sub], axis=0).reset_index(drop=True)
    print(f"Reaction {reaction_idx} is done", flush=True)
    df_reactant_bags.to_csv(
        os.path.join(save_directory, f'eMolecule_reactant_bags_reaction_idx_{reaction_idx}.csv'), index=False
    )

    # 13) Ring opening of cyclic olefin
    reaction_idx = 13
    target_list = cyclic_olefin['smiles'].tolist()
    reactant_bags = [[reactant] for reactant in target_list]
    df_sub = get_sub_df(reactant_bags=reactant_bags)
    df_reactant_bags = pd.concat([df_reactant_bags, df_sub], axis=0).reset_index(drop=True)
    print(f"Reaction {reaction_idx} is done", flush=True)
    df_reactant_bags.to_csv(
        os.path.join(save_directory, f'eMolecule_reactant_bags_reaction_idx_{reaction_idx}.csv'), index=False
    )

    # 14) Ring opening of cyclic carbonate
    reaction_idx = 14
    target_list = cyclic_carbonate['smiles'].tolist()
    reactant_bags = [[reactant] for reactant in target_list]
    df_sub = get_sub_df(reactant_bags=reactant_bags)
    df_reactant_bags = pd.concat([df_reactant_bags, df_sub], axis=0).reset_index(drop=True)
    print(f"Reaction {reaction_idx} is done", flush=True)
    df_reactant_bags.to_csv(
        os.path.join(save_directory, f'eMolecule_reactant_bags_reaction_idx_{reaction_idx}.csv'), index=False
    )

    # 15) Ring opening of cyclic sulfide
    reaction_idx = 15
    print(f"Reaction {reaction_idx} starts", flush=True)
    target_list = cyclic_sulfide['smiles'].tolist()
    reactant_bags = [[reactant] for reactant in target_list]
    df_sub = get_sub_df(reactant_bags=reactant_bags)
    df_reactant_bags = pd.concat([df_reactant_bags, df_sub], axis=0).reset_index(drop=True)
    print(f"Reaction {reaction_idx} is done", flush=True)
    df_reactant_bags.to_csv(
        os.path.join(save_directory, f'eMolecule_reactant_bags_reaction_idx_{reaction_idx}.csv'), index=False
    )

    # 16) Acyclic diene metathesis (ADMET)
    reaction_idx = 16
    target_list = terminal_diene['smiles'].tolist()
    reactant_bags = [[reactant] for reactant in target_list]
    df_sub = get_sub_df(reactant_bags=reactant_bags)
    df_reactant_bags = pd.concat([df_reactant_bags, df_sub], axis=0).reset_index(drop=True)
    print(f"Reaction {reaction_idx} is done", flush=True)
    df_reactant_bags.to_csv(
        os.path.join(save_directory, f'eMolecule_reactant_bags_reaction_idx_{reaction_idx}.csv'), index=False
    )

    # 17) Grignard metathesis method (GRIM)
    reaction_idx = 17
    target_list = conjugated_di_bromide['smiles'].tolist()
    reactant_bags = [[reactant] for reactant in target_list]
    df_sub = get_sub_df(reactant_bags=reactant_bags)
    df_reactant_bags = pd.concat([df_reactant_bags, df_sub], axis=0).reset_index(drop=True)
    print(f"Reaction {reaction_idx} is done", flush=True)
    df_reactant_bags.to_csv(
        os.path.join(save_directory, f'eMolecule_reactant_bags_reaction_idx_{reaction_idx}.csv'), index=False
    )