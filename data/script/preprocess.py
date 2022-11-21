import sys
import pandas as pd

from screen_reactant_smiles import classify_monomers_to_csv

if __name__ == "__main__":
    # load a law data
    df_smi = pd.read_csv('/home/sk77/PycharmProjects/publish/OMG/data/version.smi', sep=' ')

    # load environmental variables
    save_directory = sys.argv[1]
    print(save_directory, flush=True)
    iteration = int(sys.argv[2])
    start_idx = int(sys.argv[3])
    print(start_idx, flush=True)
    end_idx = int(sys.argv[4])
    print(end_idx, flush=True)

    # parallel process
    df_smi = df_smi.iloc[start_idx: end_idx].reset_index(drop=True)

    # SMARTS - SMiles ARbitrary Target Specification (SMARTS) annotations
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
    classify_monomers_to_csv(
        df_smi=df_smi,
        sub_structure_dict=sub_structure_dict,
        save_directory=save_directory,
        iteration=iteration
    )
