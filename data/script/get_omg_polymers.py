import os
import pandas as pd

if __name__ == '__main__':
    save_directory = '/home/sk77/PycharmProjects/publish/OMG/data/OMG_polymers_batch'
    df_total = pd.DataFrame(columns=['reaction_idx', 'reactant_1', 'reactant_2', 'product'])
    for file in os.listdir(save_directory):
        df_sub = pd.read_csv(os.path.join(save_directory, file))
        # concat
        df_total = pd.concat([df_total, df_sub], axis=0)

    # print
    print(df_total.shape)
    df_total = df_total.drop_duplicates(subset=['product'])
    print(df_total.shape)
    df_total.to_csv(os.path.join('/home/sk77/PycharmProjects/publish/OMG/data', 'OMG_polymers.csv'), index=False)
