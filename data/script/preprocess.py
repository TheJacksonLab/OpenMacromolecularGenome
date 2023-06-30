import sys
from pathlib import Path
import pandas as pd
import argparse
from screen_reactant_smiles import classify_monomers_to_csv


def get_cli_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process command-line arguments.')
    # Add arguments
    # Add positional arguments
    parser.add_argument('save_directory', type=str, nargs='?', help='Directory to save the data',
                        default='database')
    parser.add_argument('positional_iteration', type=int, nargs='?', help='Iteration number', default=1)
    parser.add_argument('positional_start_idx', type=int, nargs='?', help='Start index', default=None)
    parser.add_argument('positional_end_idx', type=int, nargs='?', help='End index', default=None)
    # Add optional arguments with flags
    parser.add_argument('-i', '--iteration', type=int, help='Iteration Number', default=None)
    parser.add_argument('-s', '--start_idx', type=int, help='Start Index', default=None)
    parser.add_argument('-e', '--end_idx', type=int, help='End Index', default=None)
    # Parse the arguments
    return parser.parse_args()


if __name__ == "__main__":
    # load a law data
    version_path = Path(__file__).resolve().parent.parent / 'version.smi'
    df_smi = pd.read_csv(version_path, sep=' ')

    args = get_cli_args()

    # Retrieve the argument values
    save_directory = Path(args.save_directory)
    iteration = args.iteration or args.positional_iteration
    start_idx = args.start_idx or args.positional_start_idx or 0
    end_idx = args.end_idx or args.positional_end_idx
    end_idx = end_idx if end_idx is not None else len(df_smi)

    # Print the loaded variables (for demonstration purposes)
    print(f"Save Directory: {save_directory}", flush=True)
    print(f"Iteration: {iteration}", flush=True)
    print(f"Start Index: {start_idx}", flush=True)
    print(f"End Index: {end_idx}", flush=True)

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
