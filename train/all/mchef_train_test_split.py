import os
import torch
import pandas as pd

if __name__ == '__main__':
    # load and split data
    load_directory = '/home/sk77/PycharmProjects/publish/OMG/data'
    save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all'

    # random sample - two-reactant reactions
    df_OMG = pd.read_csv(os.path.join(load_directory, 'OMG_polymers.csv'))

    two_reactant_df = pd.DataFrame(columns=['reaction_idx', 'reactant_1', 'reactant_2', 'product'])
    for reaction_idx in [1, 3]:
        df_sub = df_OMG[df_OMG['reaction_idx'] == reaction_idx]
        df_sub = df_sub.sample(n=10000, random_state=42)
        two_reactant_df = pd.concat([two_reactant_df, df_sub], axis=0)
        print(f'{reaction_idx} is done', flush=True)

    for reaction_idx in [2, 4]:
        df_sub = df_OMG[df_OMG['reaction_idx'] == reaction_idx]
        df_sub = df_sub.sample(n=6000, random_state=42)
        two_reactant_df = pd.concat([two_reactant_df, df_sub], axis=0)
        print(f'{reaction_idx} is done', flush=True)

    for reaction_idx in [5, 6]:
        df_sub = df_OMG[df_OMG['reaction_idx'] == reaction_idx]
        df_sub = df_sub.sample(n=3000, random_state=42)
        two_reactant_df = pd.concat([two_reactant_df, df_sub], axis=0)
        print(f'{reaction_idx} is done', flush=True)

    # concatenate - one-reactant reactions
    mixed_one_reactant_df = pd.read_csv(os.path.join(save_directory, 'mixed_one_reaction_30000_20000_10000_6000_3000_500.csv'))
    two_reactant_df = pd.concat([two_reactant_df, mixed_one_reactant_df], axis=0)

    two_reactant_df = two_reactant_df.reset_index(drop=True)
    two_reactant_df.to_csv(os.path.join(save_directory, 'all_reactions_30000_20000_10000_6000_3000_500.csv'), index=False)

    # get monomer_bags id
    train_bag_idx = two_reactant_df.index.to_list()
    test_bag_idx = list()
    len_train_idx = len(train_bag_idx)
    for idx in range(len_train_idx):
        idx_unique_monomer_sets = set()
        reactant_1 = two_reactant_df.iloc[idx]['reactant_1']
        reactant_2 = two_reactant_df.iloc[idx]['reactant_2']
        idx_unique_monomer_sets.add(reactant_1)
        idx_unique_monomer_sets.add(reactant_2)

        # get unique reactants
        remaining_unique_monomer_sets = set()
        train_bag_idx.remove(idx)
        for temp_reactant_1, temp_reactant_2 in zip(two_reactant_df.iloc[train_bag_idx]['reactant_1'], two_reactant_df.iloc[train_bag_idx]['reactant_2']):
            remaining_unique_monomer_sets.add(temp_reactant_1)
            remaining_unique_monomer_sets.add(temp_reactant_2)

        if idx_unique_monomer_sets.issubset(remaining_unique_monomer_sets):
            test_bag_idx.append(idx)
        else:
            # recover the idx
            train_bag_idx.append(idx)
        print(f'{idx} is done', flush=True)

    torch.save(train_bag_idx, os.path.join(save_directory, 'all_reactions_train_bag_idx_30000_20000_10000_6000_3000_500.pth'))
    torch.save(test_bag_idx, os.path.join(save_directory, 'all_reactions_test_bag_idx_30000_20000_10000_6000_3000_500.pth'))
