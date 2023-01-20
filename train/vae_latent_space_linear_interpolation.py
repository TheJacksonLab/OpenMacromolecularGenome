import os
import sys
import time

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as f

sys.path.append('/home/sk77/PycharmProjects/publish/OMG')
import selfies as sf

from math import ceil

from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdmolops import GetShortestPath


from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_smile_to_hot, multiple_selfies_to_hot
from vae.decoder.torch import Decoder
from vae.encoder.torch import CNNEncoder
from vae.property_predictor.torch import PropertyNetworkPredictionModule


def get_number_of_backbone_atoms(target_mol):
    asterisk_idx_list = list()
    for idx in range(target_mol.GetNumAtoms()):
        atom = target_mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == '*':
            asterisk_idx_list.append(idx)
    dist = GetShortestPath(target_mol, asterisk_idx_list[0], asterisk_idx_list[1])
    backbone_length = len(dist) - 2

    return backbone_length


def reconstruct_polymers(encoding_alphabet, one_hot_encoded_vector):
    # set parameters
    reconstructed_molecules = []

    molecules_num = one_hot_encoded_vector.shape[0]
    reconstruct_one_hot = one_hot_encoded_vector.clone().detach()
    for recon_idx in range(molecules_num):
        gathered_atoms = ''
        one_hot_idx = reconstruct_one_hot[recon_idx]
        for recon_value in one_hot_idx:
            gathered_atoms += encoding_alphabet[recon_value]

        generated_molecule = gathered_atoms.replace('[nop]', '')
        generated_molecule = '[*]' + generated_molecule  # add the first asterisk
        smiles_generated_molecule = sf.decoder(generated_molecule)

        # convert to canonical smiles
        mol = Chem.MolFromSmiles(smiles_generated_molecule)
        if mol is None:
            print(smiles_generated_molecule, flush=True)
            reconstructed_molecules.append(None)
            continue

        canonical_smiles_generated_molecule = Chem.MolToSmiles(mol)

        # check valid P-smiles
        # (1) the number of '*' should be 2
        asterisk_cnt = 0
        for char in canonical_smiles_generated_molecule:
            if char == '*':
                asterisk_cnt += 1

        if asterisk_cnt == 0:
            # print("Asterisk 0", flush=True)
            reconstructed_molecules.append(None)
            continue

        if asterisk_cnt == 1:
            # print("Asterisk 1", flush=True)
            reconstructed_molecules.append(None)
            continue

        elif asterisk_cnt >= 3:
            # print("Asterisk >= 3", flush=True)
            reconstructed_molecules.append(None)
            continue

        # (2) '*' should be connected through single bond or terminal diene
        flag = 1
        atoms = mol.GetAtoms()
        bond_order = list()
        for atom in atoms:
            atom_symbol = atom.GetSymbol()
            if atom_symbol == '*':
                if atom.GetDegree() != 1:  # asterisk should be the end of molecule
                    flag = 0  # exclude
                    break
                else:
                    bond_order.append(atom.GetExplicitValence())  # 1 or 2 (terminal diene) is expected
        if flag == 0:
            print("The asterisk is not at the end of the polymer", flush=True)
            if len(canonical_smiles_generated_molecule) >= 2:
                print(canonical_smiles_generated_molecule, flush=True)
                print(generated_molecule)
            reconstructed_molecules.append(None)
            continue

        # check bond order
        if bond_order[0] == 1 and bond_order[1] == 1:
            pass
        # elif bond_order[0] == 2 and bond_order[1] == 2:  # terminal diene -> fixed in the new SMART version
        #     pass
        else:
            flag = 0

        # exclude error
        if flag == 0:
            print("The connectivity of asterisks is not valid", flush=True)
            reconstructed_molecules.append(None)
            continue
        reconstructed_molecules.append(canonical_smiles_generated_molecule)

    return reconstructed_molecules


def linear_interpolation(arr1, arr2, t):
    return (arr2 - arr1) * t + arr1


if __name__ == '__main__':
    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # set torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    # load vae parameters
    load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/six'
    save_directory = os.path.join(load_directory, 'vae_optuna_50000_15000_6000_objective_1_10_10')
    model_save_directory = os.path.join(save_directory, 'divergence_weight_2.948_latent_dim_499_learning_rate_0.000')

    vae_parameters = torch.load(os.path.join(model_save_directory, 'vae_parameters.pth'), map_location=device)
    nop_idx = vae_parameters['nop_idx']
    asterisk_idx = vae_parameters['asterisk_idx']
    dtype = vae_parameters['dtype']

    # load data
    # df_polymer = pd.read_csv(vae_parameters['data_path'])

    # load SELFIES data - due to the random order
    encoding_list = torch.load(os.path.join(save_directory, 'encoding_list.pth'))
    encoding_alphabet = torch.load(os.path.join(save_directory, 'encoding_alphabet.pth'))
    largest_molecule_len = torch.load(os.path.join(save_directory, 'largest_molecule_len.pth'))

    # set encoder
    vae_encoder = CNNEncoder(
        in_channels=vae_parameters['encoder_in_channels'],
        feature_dim=vae_parameters['encoder_feature_dim'],
        convolution_channel_dim=vae_parameters['encoder_convolution_channel_dim'],
        kernel_size=vae_parameters['encoder_kernel_size'],
        layer_1d=vae_parameters['encoder_layer_1d'],
        layer_2d=vae_parameters['encoder_layer_2d'],
        latent_dimension=vae_parameters['latent_dimension']
    ).to(device)

    # set decoder
    vae_decoder = Decoder(
        input_size=vae_parameters['decoder_input_dimension'],
        num_layers=vae_parameters['decoder_num_gru_layers'],
        hidden_size=vae_parameters['latent_dimension'],
        out_dimension=vae_parameters['decoder_output_dimension'],
        bidirectional=vae_parameters['decoder_bidirectional']
    ).to(device)

    # set property prediction network
    property_predictor = PropertyNetworkPredictionModule(
        latent_dim=vae_parameters['latent_dimension'],
        property_dim=vae_parameters['property_dim'],
        property_network_hidden_dim_list=vae_parameters['property_network_hidden_dim_list'],
        dtype=dtype,
        device=device,
        weights=vae_parameters['property_weights']
    ).to(device)

    # load state dictionary
    vae_encoder.load_state_dict(torch.load(os.path.join(model_save_directory, 'encoder.pth'), map_location=device))
    vae_decoder.load_state_dict(torch.load(os.path.join(model_save_directory, 'decoder.pth'), map_location=device))
    property_predictor.load_state_dict(torch.load(os.path.join(model_save_directory, 'property_predictor.pth'),
                                       map_location=device))

    # preparation training data
    # df_polymer = pd.read_csv(vae_parameters['data_path'])
    # df_polymer['mol'] = df_polymer['product'].apply(lambda x: Chem.MolFromSmiles(x))
    # df_polymer['LogP'] = df_polymer['mol'].apply(lambda x: MolLogP(x) / get_number_of_backbone_atoms(x))

    # property_value = df_polymer['LogP'].to_numpy().reshape(-1, 1)

    property_value = torch.load(os.path.join(save_directory, 'property_value.pth'))

    # split data
    train_idx = torch.load(os.path.join(save_directory, 'train_idx.pth'))
    property_train = property_value[train_idx]

    # property scaling
    property_scaler = StandardScaler()
    property_scaler.fit(property_train)

    # load PCA data
    eigenvector_1 = np.load(os.path.join(model_save_directory, 'PC_mean_eigenvector_1.npy'))
    eigenvector_2 = np.load(os.path.join(model_save_directory, 'PC_mean_eigenvector_2.npy'))
    eigenvector_list = [eigenvector_1, eigenvector_2]

    z_mean = np.load(os.path.join(model_save_directory, 'z_mean.npy'))
    property_z_mean = np.load(os.path.join(model_save_directory, 'property_z_mean.npy'))

    principal_df = pd.DataFrame(
        data=None,
        columns=['PC%d' % num for num in range(1, 3)] + [f'z_{num + 1}_mean' for num in range(z_mean.shape[1])] +
                [f'property_{num + 1}_z_mean' for num in range(property_z_mean.shape[1])]
    )

    # fill values
    for idx, arr in enumerate(z_mean.T):
        principal_df[f'z_{idx + 1}_mean'] = arr

    for idx, arr in enumerate(property_z_mean.T):
        principal_df[f'property_{idx + 1}_z_mean'] = arr

    # projection
    data_mean = np.mean(z_mean, axis=0)
    data = z_mean - data_mean
    for idx, eigenvector in enumerate(eigenvector_list):
        # get PC coefficient
        idx += 1
        principal_df['PC%d' % idx] = data.dot(eigenvector)

    # get decoding point list
    decoding_point = list()
    # principal_df_sorted = principal_df.sort_values(by=['PC1'], ascending=True)  # lowest
    # target_1_point = principal_df_sorted.iloc[25]

    # principal_df_sorted = principal_df.sort_values(by=['PC1'], ascending=False)  # highest
    # target_2_point = principal_df_sorted.iloc[35]

    principal_df_sorted = principal_df.sort_values(by=['property_1_z_mean'], ascending=True)  # minimum of property
    target_1_point = principal_df_sorted.iloc[300]

    principal_df_sorted = principal_df.sort_values(by=['property_1_z_mean'], ascending=False)  # maximum of property
    target_2_point = principal_df_sorted.iloc[100]

    # linear interpolation
    target_1_z = target_1_point.to_numpy()[2:-1]  # exclude PC1, PC2, and property
    target_2_z = target_2_point.to_numpy()[2:-1]  # exclude PC1, PC2, and property
    t_points = np.arange(start=0, stop=1.01, step=0.25)  # np.arange didn't include the stop

    # PCA or target point tensor
    target_point_arr = np.array([linear_interpolation(target_1_z, target_2_z, t) for t in t_points])

    # projection
    data = target_point_arr - data_mean
    interpolation_df = pd.DataFrame(data=None, columns=['PC1', 'PC2'])
    for idx, eigenvector in enumerate(eigenvector_list):
        # get PC coefficient
        idx += 1
        interpolation_df['PC%d' % idx] = data.dot(eigenvector)

    print('Projection is done', flush=True)

    # decode
    generated_polymer_list = []
    property_prediction_list = []
    with torch.no_grad():
        # eval
        vae_encoder.eval()
        vae_decoder.eval()
        property_predictor.eval()

        target_point_tensor = torch.tensor(
            np.array([linear_interpolation(target_1_z, target_2_z, t) for t in t_points]),
            dtype=dtype, device=device)

        property_prediction = property_predictor(target_point_tensor).cpu()

        # property scaler
        predicted_property = property_scaler.inverse_transform(property_prediction)
        property_prediction_list.append(predicted_property)

        for latent_tensor in target_point_tensor:
            # decode
            hidden = vae_decoder.init_hidden(latent_tensor)
            out_one_hot = torch.zeros(size=(1, largest_molecule_len, len(encoding_alphabet)),
                                      dtype=dtype, device=device)
            nop_tensor = -torch.ones(size=(1,), dtype=dtype, device=device)
            asterisk_tensor = -torch.ones(size=(1,), dtype=dtype, device=device)
            updated_nop_tensor = -torch.ones(size=(1,), dtype=dtype, device=device)
            x_input = torch.zeros(size=(1, 1, len(encoding_alphabet)), dtype=dtype, device=device)
            rnn_max_length = largest_molecule_len

            # no teacher forcing
            for seq_index in range(rnn_max_length):
                out_one_hot_line, hidden = vae_decoder(x=x_input, hidden=hidden)
                # find 'nop' idx
                boolean_tensor = out_one_hot_line[0].argmax(dim=-1) == nop_idx
                nonzero_tensor = boolean_tensor.nonzero()

                for nonzero_num in range(nonzero_tensor.shape[0]):
                    nonzero_idx = nonzero_tensor[nonzero_num][0].item()
                    # store
                    if nop_tensor[nonzero_idx] < 0:
                        nop_tensor[nonzero_idx] = seq_index

                # find 'asterisk' idx
                boolean_tensor = out_one_hot_line[0].argmax(dim=-1) == asterisk_idx
                nonzero_tensor = boolean_tensor.nonzero()
                for nonzero_num in range(nonzero_tensor.shape[0]):
                    nonzero_idx = nonzero_tensor[nonzero_num][0].item()
                    # store
                    if asterisk_tensor[nonzero_idx] < 0:
                        asterisk_tensor[nonzero_idx] = 0.0  # generated one asterisk

                    elif asterisk_tensor[nonzero_idx] == 0:
                        asterisk_tensor[
                            nonzero_idx] = seq_index  # generated two asterisks other than the initiator

                # change value of asterisk tensor: 0 -> -1
                # asterisk_tensor[asterisk_tensor == 0.0] = -1.0  # yield bug -> more than 2 asterisks

                # update nop tensor
                for idx in range(asterisk_tensor.shape[0]):
                    if nop_tensor[idx] == -1.0 and asterisk_tensor[idx] <= 0.0:
                        continue
                    elif nop_tensor[idx] < 0 < asterisk_tensor[idx]:
                        updated_nop_tensor[idx] = asterisk_tensor[idx]
                    elif nop_tensor[idx] > 0 >= asterisk_tensor[idx]:
                        updated_nop_tensor[idx] = nop_tensor[idx]
                    elif nop_tensor[idx] >= asterisk_tensor[idx]:
                        updated_nop_tensor[idx] = asterisk_tensor[idx]
                    else:
                        updated_nop_tensor[idx] = nop_tensor[idx]

                # stop generation if all tensors contain 'nop': elements of nop tensor is not negative
                flag = 1  # flag = 1 means 'stop'
                for idx, value in enumerate(updated_nop_tensor):
                    if value < 0:  # there is non 'nop'
                        flag = 0
                        break
                if flag == 1:
                    break

                out_one_hot[:, seq_index, :] = out_one_hot_line[0]

                # change input - no teacher forcing (one-hot)
                x_input = out_one_hot_line.argmax(dim=-1)
                x_input = f.one_hot(x_input, num_classes=len(encoding_alphabet)).to(torch.float)

            # x_indices
            x_hat_prob = f.softmax(out_one_hot, dim=-1)
            x_hat_indices = x_hat_prob.argmax(dim=-1)

            # modify x_hat_indices value to nop_idx after the nop appears
            for idx, value in enumerate(updated_nop_tensor):
                if value >= 0.0:
                    x_hat_indices[idx, int(value): rnn_max_length] = nop_idx

            # convert to SMILES string
            generated_molecules = reconstruct_polymers(encoding_alphabet, x_hat_indices)

            # store
            for polymer in generated_molecules:
                if polymer is None:
                    generated_polymer_list.append(None)
                    continue
                else:
                    generated_polymer_list.append(polymer)

    print(generated_polymer_list)
    print(property_prediction_list)

    # seaborn JointGrid
    g = sns.JointGrid(x='PC1', y='PC2', data=principal_df)

    # scatter plot
    cmap = LinearSegmentedColormap.from_list("", ["#2F65ED", "#F5EF7E", "#F89A3F"])
    sns.scatterplot(x='PC1', y='PC2', hue='property_1_z_mean', data=principal_df, palette=cmap, ax=g.ax_joint,
                    edgecolor=None, alpha=0.75, legend=False, size=3.0)

    sns.lineplot(x='PC1', y='PC2', data=interpolation_df, ax=g.ax_joint, linestyle='--', color='k', alpha=0.75)
    sns.scatterplot(x='PC1', y='PC2', data=interpolation_df, color='k', ax=g.ax_joint,
                    edgecolor=None, alpha=0.75, legend=False, size=10000.0, marker='*')

    # kde plots on the marginal axes
    sns.kdeplot(x='PC1', data=principal_df, ax=g.ax_marg_x, shade=True, color='c')
    sns.kdeplot(y='PC2', data=principal_df, ax=g.ax_marg_y, shade=True, color='c')

    # tick parameters
    g.ax_joint.tick_params(labelsize=14)
    g.ax_joint.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    g.ax_joint.set_yticks([-12, -8, -4, 0, 4, 8, 12, 16])
    g.set_axis_labels('Principal component 1', 'Principal component 2', fontsize=16)
    g.fig.tight_layout()

    # color bars
    norm = plt.Normalize(principal_df['property_1_z_mean'].min(), principal_df['property_1_z_mean'].max())
    scatter_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Make space for the colorbar
    g.fig.subplots_adjust(bottom=0.2)
    cax = g.fig.add_axes([0.18, 0.08, 0.6, 0.02])  # l, b, w, h
    cbar = g.fig.colorbar(scatter_map, cax=cax, orientation='horizontal', ticks=[-0.5, 0.0, 0.5, 1.0, 1.5])
    cbar.set_label(label='LogP', size=12)
    cbar.ax.tick_params(labelsize=12)

    g.fig.savefig(os.path.join(model_save_directory, "seaborn.png"), dpi=800)

    # matplotlib plot
    # plt.figure(figsize=(6, 6), dpi=300)
    # plt.scatter(principal_df['PC1'], principal_df['PC2'], c=property_z_mean, cmap='rainbow', s=2.0, alpha=1.0)
    # plt.plot(interpolation_df['PC1'], interpolation_df['PC2'], 'k--', alpha=0.5)
    # plt.plot(interpolation_df['PC1'].iloc[1:-1], interpolation_df['PC2'].iloc[1:-1], 'k*', markersize=5.0, alpha=0.5,
    #          label='Interpolation')
    #
    # plt.title('PCA of the latent space of MoleculeChef', fontsize=10)
    # plt.xlabel('PC1', fontsize=10)
    # plt.ylabel('PC2', fontsize=10)
    # plt.colorbar(orientation='horizontal').set_label(label='LogP', size=10)
    # plt.legend(fontsize=10)
    # plt.savefig(os.path.join(model_save_directory, f'latent_space_linear_interpolation_1_2.png'))
    # plt.show()
    # plt.close()
