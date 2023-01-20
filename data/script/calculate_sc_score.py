import os
import sys
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from rdkit import Chem
from rdkit.Chem import RDConfig
from scipy import stats

sys.path.append('/home/sk77/PycharmProjects/publish/OMG')
from scscore.scscore import SCScorer


if __name__ == '__main__':
    # eMolecules monomer
    load_dir = '/home/sk77/PycharmProjects/publish/OMG/data/OMG_monomers'
    df_total = pd.DataFrame(data=None, columns=['acetylene', 'di_acid_chloride', 'conjugated_di_bromide',
                                                'cyclic_carbonate', 'cyclic_ether', 'cyclic_olefin', 'cyclic_sulfide',
                                                'di_amine', 'di_carboxylic_acid', 'di_isocyanate', 'di_ol', 'lactam',
                                                'lactone', 'terminal_diene', 'vinyl', 'hydroxy_carboxylic_acid',
                                                'smiles'])
    for file in os.listdir(load_dir):
        df = pd.read_csv(os.path.join(load_dir, file))
        df_total = pd.concat([df_total, df], axis=0)

    # drop duplicates
    df_total = df_total.drop_duplicates(subset='smiles')
    print(df_total.shape, flush=True)

    # load SC score class
    model = SCScorer()
    model.restore()

    # calculate SC Score
    df_total['SC_score'] = df_total['smiles'].apply(lambda x: model.get_score_from_smi(x)[1])

    # filter
    df_total = df_total[df_total['SC_score'] <= 2.163110]
    print(df_total.shape, flush=True)

    # save - 2.163110 -> Mean SC score of PolyInfo monomers
    # df_polyinfo.to_csv('./polyinfo_total_monomers_additional_functional_groups.csv', index=False)
    df_total.to_csv(os.path.join('/home/sk77/PycharmProjects/publish/OMG/data', 'OMG_monomers.csv'))
