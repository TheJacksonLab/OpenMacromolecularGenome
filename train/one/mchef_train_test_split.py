import os
import torch
import pandas as pd

if __name__ == '__main__':
    # load and split data
    load_directory = '/home/sk77/PycharmProjects/publish/OMG/data'
    save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/one'

    df = pd.read_csv(os.path.join(load_directory, 'OMG_polymers.csv'))
    reaction_idx = 3  # the largest one - a single polymerization mechanism

    # target df
    df = df[df['reaction_idx'] == reaction_idx]
    df = df.reset_index(drop=True)

    # random sample
    df_sample = df.sample(n=150000, random_state=42)
    df_sample = df_sample.reset_index(drop=True)
    df_sample.to_csv(os.path.join(save_directory, 'reaction_3_150K.csv'), index=False)

    # get monomer_bags idx
    train_bag_idx = df_sample.index.to_list()
    test_bag_idx = list()
    len_train_idx = len(train_bag_idx)
    for idx in range(len_train_idx):
        idx_unique_monomer_sets = set()
        reactant_1 = df_sample.iloc[idx]['reactant_1']
        reactant_2 = df_sample.iloc[idx]['reactant_2']
        idx_unique_monomer_sets.add(reactant_1)
        idx_unique_monomer_sets.add(reactant_2)

        # get unique reactants
        remaining_unique_monomer_sets = set()
        train_bag_idx.remove(idx)
        for temp_reactant_1, temp_reactant_2 in zip(df_sample.iloc[train_bag_idx]['reactant_1'], df_sample.iloc[train_bag_idx]['reactant_2']):
            remaining_unique_monomer_sets.add(temp_reactant_1)
            remaining_unique_monomer_sets.add(temp_reactant_2)

        if idx_unique_monomer_sets.issubset(remaining_unique_monomer_sets):
            test_bag_idx.append(idx)
        else:
            # recover the idx
            train_bag_idx.append(idx)
        print(f'{idx} is done', flush=True)

    torch.save(train_bag_idx, os.path.join(save_directory, 'train_bag_idx_reaction_3_150K.pth'))
    torch.save(test_bag_idx, os.path.join(save_directory, 'test_bag_idx_reaction_3_150K.pth'))
