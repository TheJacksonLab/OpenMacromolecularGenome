import os
import pandas as pd

from rdkit import Chem

sub_structure_dict = {
    'acetylene': '[CX2]#[CX2]',
    'di_acid_chloride': '[CX3](=O)[Cl]',
    'conjugated_di_bromide': '[c;R][Br]',
    'cyclic_carbonate': '[OX1]=[CX3;R]([OX2;R][C;R])[OX2;R][C;R]',
    'cyclic_ether': '[C;R][O;R]([C;R])',  # '[OX2;R]([CX2;R][C;R])[CX2;R][C;R]'
    'cyclic_olefin': '[CH1;R][CH1;R]=[CH1;R][CH1;R]',
    'cyclic_sulfide': '[C;R][S;R]([C;R])',
    'di_amine': '[NX3H2;!$(NC=O)]',
    'di_carboxylic_acid': '[CX3](=O)[OX2H]',
    'di_isocyanate': '[NX2]=[CX2]=[OX1]',
    'di_ol': '[C,c;!$(C=O)][OX2H1]',
    'hydroxy_carboxylic_acid_OH': '[!$(C=O)][OX2H1]',
    'hydroxy_carboxylic_acid_COOH': '[CX3](=O)[OX2H]',
    'lactam': '[NH1;R][C;R](=O)',
    'lactone': '[O;R][C;R](=O)',
    'terminal_diene': '[CX3H2]=[CX3H1]',
    'vinyl': '[CX3;!R]=[CX3]'
}


if __name__ == '__main__':
    # read .csv file
    df_OMG_monomers = pd.read_csv('/home/sk77/PycharmProjects/publish/OMG/data/OMG_monomers.csv')
    print(df_OMG_monomers.head().to_string())
    print(df_OMG_monomers.shape)

    # load monomers
    monomer_dict = dict()
    monomer_names = ['conjugated_di_bromide', 'hydroxy_carboxylic_acid',
                     'cyclic_ether', 'cyclic_olefin', 'di_isocyanate', 'di_ol',
                     'terminal_diene', 'lactone', 'cyclic_sulfide', 'cyclic_carbonate',
                     'di_acid_chloride', 'di_amine', 'vinyl', 'di_carboxylic_acid',
                     'acetylene', 'lactam']
    for monomer in monomer_names:
        monomer_dict[monomer] = 0

    # count
    cnt = 0
    for monomer in monomer_names:
        if 'di' in monomer:
            df_sub = df_OMG_monomers[df_OMG_monomers[monomer] == 2]
            monomer_dict[monomer] = df_sub.shape[0]
            cnt += df_sub.shape[0]
        else:
            df_sub = df_OMG_monomers[df_OMG_monomers[monomer] == 1]
            monomer_dict[monomer] = df_sub.shape[0]
            cnt += df_sub.shape[0]

    # sort dictionary
    sorted_dict = {k: v for k, v in sorted(monomer_dict.items(), key=lambda item: item[1])}
    print(cnt)
    print(sorted_dict)

    # sum
    total_counts = 0
    for value in sorted_dict.values():
        total_counts += value
    print(total_counts)


