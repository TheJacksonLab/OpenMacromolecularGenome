import os
import pandas as pd

if __name__ == '__main__':
    # get name of monomer
    load_dir = '/home/sk77/PycharmProjects/publish/OMG/data/OMG_monomer_process_batch'
    save_dir = '/home/sk77/PycharmProjects/publish/OMG/data/OMG_monomers'
    df_total = pd.DataFrame(data=None, columns=['acetylene', 'di_acid_chloride', 'conjugated_di_bromide',
                                                'cyclic_carbonate', 'cyclic_ether', 'cyclic_olefin', 'cyclic_sulfide',
                                                'di_amine', 'di_carboxylic_acid', 'di_isocyanate', 'di_ol', 'lactam',
                                                'lactone', 'terminal_diene', 'vinyl', 'hydroxy_carboxylic_acid',
                                                'smiles'])
    for file in os.listdir(load_dir):
        if file.endswith('.csv'):
            # merge
            df = pd.read_csv(os.path.join(load_dir, file)).drop(columns=['Unnamed: 0'], axis=1)
            df_total = pd.concat([df_total, df], axis=0)

    # save df
    print(df_total.shape)
    df_total = df_total.drop_duplicates(subset='smiles')
    df_total = df_total.reset_index(drop=True)
    print(df_total.shape)

    # save .csv files
    for col in df_total.columns:
        if col != 'smiles' and 'di' not in col:
            df_condition = df_total[df_total[col] == 1]
            csv_file_path = os.path.join(save_dir, '%s' % col + '.csv')
            df_condition.to_csv(path_or_buf=csv_file_path, index=False)
        elif col != 'smiles' and 'di' in col:
            df_condition = df_total[df_total[col] == 2]
            csv_file_path = os.path.join(save_dir, '%s' % col + '.csv')
            df_condition.to_csv(path_or_buf=csv_file_path, index=False)
