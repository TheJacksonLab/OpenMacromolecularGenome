import os
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

if __name__ == '__main__':
    # df = pd.read_csv(os.path.join(save_directory, 'OMG_polymers.csv')).sample(n=2000000, random_state=42)
    # df_OMG = pd.read_csv(os.path.join(save_directory, 'OMG_polymers.csv'))
    # print(df_OMG.shape)
    # exit()
    df_OMG = pd.read_csv('/home/sk77/PycharmProjects/publish/OMG/data/OMG_polymers.csv').sample(n=2000000, random_state=42)
    df_OMG = df_OMG.reset_index(drop=True)

    df_OMG['mol'] = df_OMG['product'].apply(lambda x: Chem.MolFromSmiles(x))
    fps = np.array(
        df_OMG['mol'].apply(lambda x: GetMorganFingerprintAsBitVect(x, radius=2, nBits=1024)).tolist()
    ).astype('float32')

    # save
    np.save(os.path.join('/home/sk77/PycharmProjects/publish/OMG/data', 'fps_OMG_polymers_random_2M'), fps)
