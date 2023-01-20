import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as f

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


sys.path.append('/home/sk77/PycharmProjects/publish/OMG')
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


def reconstruct_molecules(encoding_alphabet, one_hot_encoded_vector):
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
        canonical_smiles_generated_molecule = Chem.MolToSmiles(mol)

        reconstructed_molecules.append(canonical_smiles_generated_molecule)

    return reconstructed_molecules


if __name__ == '__main__':
    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    # set torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    # load vae parameters
    load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all'
    save_directory = os.path.join(load_directory, 'vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay')
    model_save_directory = os.path.join(save_directory, 'divergence_weight_4.345_latent_dim_152_learning_rate_0.002')

    vae_parameters = torch.load(os.path.join(model_save_directory, 'vae_parameters.pth'), map_location=device)
    nop_idx = vae_parameters['nop_idx']
    asterisk_idx = vae_parameters['asterisk_idx']
    dtype = vae_parameters['dtype']

    # load SELFIES data - due to the random order
    encoding_list = torch.load(os.path.join(save_directory, 'encoding_list.pth'))
    encoding_alphabet = torch.load(os.path.join(save_directory, 'encoding_alphabet.pth'))
    largest_molecule_len = torch.load(os.path.join(save_directory, 'largest_molecule_len.pth'))

    print('[VAE] Constructing selfies one_hot_encoded vectors..', flush=True)
    data = multiple_selfies_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)
    print('[VAE] One_hot_encoded vectors are constructed successfully', flush=True)

    # exclude the first [*]
    data = data[:, 1:, :]

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

    # load data
    train_idx = torch.load(os.path.join(save_directory, 'train_idx.pth'))
    valid_idx = torch.load(os.path.join(save_directory, 'valid_idx.pth'))
    test_idx = torch.load(os.path.join(save_directory, 'test_idx.pth'))

    # preparation training data
    property_value = torch.load(os.path.join(save_directory, 'property_value.pth'))

    # split data
    data_train = data[train_idx]
    data_valid = data[valid_idx]
    data_test = data[test_idx]

    property_train = property_value[train_idx]
    property_valid = property_value[valid_idx]
    property_test = property_value[test_idx]

    # property scaling
    property_scaler = StandardScaler()
    property_train_scaled = property_scaler.fit_transform(property_train)
    property_valid_scaled = property_scaler.transform(property_valid)
    property_test_scaled = property_scaler.transform(property_test)

    # set property prediction network
    property_predictor = PropertyNetworkPredictionModule(
        latent_dim=vae_parameters['latent_dimension'],
        property_dim=vae_parameters['property_dim'],
        property_network_hidden_dim_list=vae_parameters['property_network_hidden_dim_list'],
        dtype=dtype,
        device=device,
        weights=vae_parameters['property_weights']
    ).to(device)

    data_train_tensor = torch.tensor(data_train, dtype=dtype).to(device)
    data_valid_tensor = torch.tensor(data_valid, dtype=dtype).to(device)
    data_test_tensor = torch.tensor(data_test, dtype=dtype).to(device)
    y_train_tensor = torch.tensor(property_train_scaled, dtype=dtype).to(device)
    y_valid_tensor = torch.tensor(property_valid_scaled, dtype=dtype).to(device)
    y_test_tensor = torch.tensor(property_test_scaled, dtype=dtype).to(device)

    # load state dictionary
    vae_encoder.load_state_dict(torch.load(os.path.join(model_save_directory, 'encoder.pth'), map_location=device))
    vae_decoder.load_state_dict(torch.load(os.path.join(model_save_directory, 'decoder.pth'), map_location=device))
    property_predictor.load_state_dict(torch.load(os.path.join(model_save_directory, 'property_predictor.pth'),
                                       map_location=device))

    # store latent z points & property values
    z_mean_train = list()
    z_mean_valid = list()
    z_mean_test = list()

    train_property_prediction_list = list()
    valid_property_prediction_list = list()
    test_property_prediction_list = list()

    train_reconstruction_list = list()
    valid_reconstruction_list = list()
    test_reconstruction_list = list()

    train_r2_score_list = list()
    valid_r2_score_list = list()
    test_r2_score_list = list()

    # check reconstruction rate - encode and decode several times to get uncertainty
    trial = 10
    for trial_idx in range(1, trial + 1):
        print(f'Trial {trial_idx} started', flush=True)
        trial_train_property_prediction = list()
        trial_valid_property_prediction = list()
        trial_test_property_prediction = list()

        trial_z_mean_train = list()
        trial_z_mean_valid = list()
        trial_z_mean_test = list()

        # check reconstruction (train data)
        batch_size = 100
        train_batch_iteration = ceil(data_train_tensor.shape[0] / batch_size)
        valid_batch_iteration = ceil(data_valid_tensor.shape[0] / batch_size)
        test_batch_iteration = ceil(data_test_tensor.shape[0] / batch_size)

        count_of_train_reconstruction = 0
        count_of_valid_reconstruction = 0
        count_of_test_reconstruction = 0

        train_answer_polymer = list()
        train_wrong_prediction_polymer = list()

        valid_answer_polymer = list()
        valid_wrong_prediction_polymer = list()

        test_answer_polymer = list()
        test_wrong_prediction_polymer = list()

        # eval
        vae_encoder.eval()
        vae_decoder.eval()
        property_predictor.eval()
        with torch.no_grad():
            tqdm.write("Train Reconstruction ...")
            pbar = tqdm(range(train_batch_iteration), total=train_batch_iteration, leave=True)
            time.sleep(1.0)
            with pbar as t:
                for batch_iteration in t:
                    # manual batch iterations
                    start_idx = batch_iteration * batch_size
                    stop_idx = (batch_iteration + 1) * batch_size

                    if batch_iteration == train_batch_iteration - 1:
                        stop_idx = data_train_tensor.shape[0]

                    data_train_batch = data_train_tensor[start_idx: stop_idx]

                    # find maximum length
                    length_tensor = torch.zeros_like(data_train_batch[:, 0, 0])
                    for idx in range(data_train_batch.shape[0]):
                        boolean = (data_train_batch[idx, :, nop_idx] == 1.0).nonzero()
                        if boolean.shape[0] != 0:
                            length = (data_train_batch[idx, :, nop_idx] == 1.0).nonzero()[0][0].item()
                        else:
                            length = data_train_batch.shape[1]
                        length_tensor[idx] = length
                    rnn_max_length = length_tensor.max()

                    # encode
                    inp_flat_one_hot_train = data_train_batch.transpose(dim0=1, dim1=2)
                    latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot_train)

                    # store
                    trial_z_mean_train += mus.cpu().tolist()

                    # property_prediction
                    trial_train_property_prediction += property_predictor(latent_points).cpu().tolist()

                    # use latent vectors as hidden (not zero-initialized hidden)
                    hidden = vae_decoder.init_hidden(latent_points)
                    out_one_hot = torch.zeros_like(data_train_batch, dtype=dtype, device=device)
                    x_input = torch.zeros_like(data_train_batch[:, 0, :].unsqueeze(0), dtype=dtype, device=device)
                    nop_tensor = -torch.ones_like(data_train_batch[:, 0, 0], dtype=dtype, device=device)
                    asterisk_tensor = -torch.ones_like(data_train_batch[:, 0, 0], dtype=dtype, device=device)

                    # no teacher forcing
                    for seq_index in range(int(rnn_max_length.item())):
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

                        out_one_hot[:, seq_index, :] = out_one_hot_line[0]

                        # change input - no teacher forcing (one-hot)
                        x_input = out_one_hot_line.argmax(dim=-1)
                        x_input = f.one_hot(x_input, num_classes=data_train_batch.shape[2]).to(torch.float)

                    # x_indices
                    x_indices = data_train_batch.argmax(dim=-1)
                    x_hat_prob = f.softmax(out_one_hot, dim=-1)
                    x_hat_indices = x_hat_prob.argmax(dim=-1)

                    # change value of asterisk tensor: 0 -> -1
                    asterisk_tensor[asterisk_tensor == 0.0] = -1.0

                    # update nop tensor
                    updated_nop_tensor = -torch.ones_like(data_train_batch[:, 0, 0], dtype=dtype, device=device)
                    for idx in range(asterisk_tensor.shape[0]):
                        if nop_tensor[idx] == -1.0 and asterisk_tensor[idx] == -1.0:
                            continue
                        elif nop_tensor[idx] < 0 < asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = asterisk_tensor[idx]
                        elif nop_tensor[idx] > 0 > asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = nop_tensor[idx]
                        elif nop_tensor[idx] >= asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = asterisk_tensor[idx]
                        else:
                            updated_nop_tensor[idx] = nop_tensor[idx]

                    # modify x_hat_indices value to nop_idx after the nop appears
                    max_length = x_hat_indices.shape[1]
                    for idx, value in enumerate(nop_tensor):
                        if value >= 0.0:
                            x_hat_indices[idx, int(value): max_length] = nop_idx

                    # convert to SMILES string
                    answer_molecules = reconstruct_molecules(encoding_alphabet, x_indices)
                    decoded_molecules = reconstruct_molecules(encoding_alphabet, x_hat_indices)

                    # count right & wrong molecules
                    for count_idx in range(len(answer_molecules)):
                        if answer_molecules[count_idx] == decoded_molecules[count_idx]:
                            count_of_train_reconstruction += 1
                        else:
                            train_answer_polymer.append(answer_molecules[count_idx])
                            train_wrong_prediction_polymer.append(decoded_molecules[count_idx])

            tqdm.write(f'Train reconstruction rate is {(100 * (count_of_train_reconstruction/data_train_tensor.shape[0])):.3f}%')

            # store & inverse transform
            train_reconstruction_list.append((100 * (count_of_train_reconstruction/data_train_tensor.shape[0])))

            trial_train_inverse_transformed_property_prediction = property_scaler.inverse_transform(
                trial_train_property_prediction
            )
            train_property_prediction_list.append(trial_train_inverse_transformed_property_prediction)

            # get r2 score
            property_name = ['LogP', 'SC_score']
            for property_idx in range(property_predictor.property_dim):
                train_r2_score_list.append(r2_score(y_true=property_train[:, property_idx], y_pred=trial_train_inverse_transformed_property_prediction[:, property_idx]))

            z_mean_train.append(trial_z_mean_train)

            tqdm.write('Validation Reconstruction ...')
            pbar = tqdm(range(valid_batch_iteration), total=valid_batch_iteration, leave=True)
            time.sleep(1.0)
            with pbar as t:
                for batch_iteration in t:
                    # manual batch iterations
                    start_idx = batch_iteration * batch_size
                    stop_idx = (batch_iteration + 1) * batch_size

                    if batch_iteration == valid_batch_iteration - 1:
                        stop_idx = data_valid_tensor.shape[0]

                    data_valid_batch = data_valid_tensor[start_idx: stop_idx]

                    # find maximum length
                    length_tensor = torch.zeros_like(data_valid_batch[:, 0, 0])
                    for idx in range(data_valid_batch.shape[0]):
                        boolean = (data_valid_batch[idx, :, nop_idx] == 1.0).nonzero()
                        if boolean.shape[0] != 0:
                            length = (data_valid_batch[idx, :, nop_idx] == 1.0).nonzero()[0][0].item()
                        else:
                            length = data_valid_batch.shape[1]
                        length_tensor[idx] = length
                    rnn_max_length = length_tensor.max()

                    # encode
                    inp_flat_one_hot_valid = data_valid_batch.transpose(dim0=1, dim1=2)
                    latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot_valid)

                    # store
                    trial_z_mean_valid += mus.cpu().tolist()

                    # property_prediction
                    trial_valid_property_prediction += property_predictor(latent_points).cpu().tolist()

                    # use latent vectors as hidden (not zero-initialized hidden)
                    hidden = vae_decoder.init_hidden(latent_points)
                    out_one_hot = torch.zeros_like(data_valid_batch, dtype=dtype, device=device)
                    x_input = torch.zeros_like(data_valid_batch[:, 0, :].unsqueeze(0), dtype=dtype, device=device)
                    nop_tensor = -torch.ones_like(data_valid_batch[:, 0, 0], dtype=dtype, device=device)
                    asterisk_tensor = -torch.ones_like(data_valid_batch[:, 0, 0], dtype=dtype, device=device)

                    # no teacher forcing
                    for seq_index in range(int(rnn_max_length.item())):
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

                        out_one_hot[:, seq_index, :] = out_one_hot_line[0]

                        # change input - no teacher forcing (one-hot)
                        x_input = out_one_hot_line.argmax(dim=-1)
                        x_input = f.one_hot(x_input, num_classes=data_valid_batch.shape[2]).to(dtype)

                    # x_indices
                    x_indices = data_valid_batch.argmax(dim=-1)
                    x_hat_prob = f.softmax(out_one_hot, dim=-1)
                    x_hat_indices = x_hat_prob.argmax(dim=-1)

                    # change value of asterisk tensor: 0 -> -1
                    asterisk_tensor[asterisk_tensor == 0.0] = -1.0

                    # update nop tensor
                    updated_nop_tensor = -torch.ones_like(data_valid_batch[:, 0, 0], dtype=dtype, device=device)
                    for idx in range(asterisk_tensor.shape[0]):
                        if nop_tensor[idx] == -1.0 and asterisk_tensor[idx] == -1.0:
                            continue
                        elif nop_tensor[idx] < 0 < asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = asterisk_tensor[idx]
                        elif nop_tensor[idx] > 0 > asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = nop_tensor[idx]
                        elif nop_tensor[idx] >= asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = asterisk_tensor[idx]
                        else:
                            updated_nop_tensor[idx] = nop_tensor[idx]

                    # modify x_hat_indices value to nop_idx after the nop appears
                    max_length = x_hat_indices.shape[1]
                    for idx, value in enumerate(nop_tensor):
                        if value >= 0.0:
                            x_hat_indices[idx, int(value): max_length] = nop_idx

                    # convert to SMILES string
                    answer_molecules = reconstruct_molecules(encoding_alphabet, x_indices)
                    decoded_molecules = reconstruct_molecules(encoding_alphabet, x_hat_indices)

                    # count right & wrong molecules
                    for count_idx in range(len(answer_molecules)):
                        if answer_molecules[count_idx] == decoded_molecules[count_idx]:
                            count_of_valid_reconstruction += 1
                        else:
                            valid_answer_polymer.append(answer_molecules[count_idx])
                            valid_wrong_prediction_polymer.append(decoded_molecules[count_idx])

            tqdm.write(f'Valid reconstruction rate is {(100 * (count_of_valid_reconstruction / data_valid_tensor.shape[0])):.3f}%')

            # store & inverse transform
            valid_reconstruction_list.append((100 * (count_of_valid_reconstruction/data_valid_tensor.shape[0])))
            trial_valid_inverse_transformed_property_prediction = property_scaler.inverse_transform(
                trial_valid_property_prediction
            )
            valid_property_prediction_list.append(trial_valid_inverse_transformed_property_prediction)
            z_mean_valid.append(trial_z_mean_valid)

            # get r2 score
            property_name = ['LogP', 'SC_score']
            for property_idx in range(property_predictor.property_dim):
                valid_r2_score_list.append(r2_score(y_true=property_valid[:, property_idx],
                                                     y_pred=trial_valid_inverse_transformed_property_prediction[:,
                                                            property_idx]))

            tqdm.write('Test Reconstruction ...')
            pbar = tqdm(range(test_batch_iteration), total=test_batch_iteration, leave=True)
            time.sleep(1.0)
            with pbar as t:
                for batch_iteration in t:
                    # manual batch iterations
                    start_idx = batch_iteration * batch_size
                    stop_idx = (batch_iteration + 1) * batch_size

                    if batch_iteration == test_batch_iteration - 1:
                        stop_idx = data_test_tensor.shape[0]

                    data_test_batch = data_test_tensor[start_idx: stop_idx]

                    # find maximum length
                    length_tensor = torch.zeros_like(data_test_batch[:, 0, 0])
                    for idx in range(data_test_batch.shape[0]):
                        boolean = (data_test_batch[idx, :, nop_idx] == 1.0).nonzero()
                        if boolean.shape[0] != 0:
                            length = (data_test_batch[idx, :, nop_idx] == 1.0).nonzero()[0][0].item()
                        else:
                            length = data_test_batch.shape[1]
                        length_tensor[idx] = length
                    rnn_max_length = length_tensor.max()

                    # encode
                    inp_flat_one_hot_test = data_test_batch.transpose(dim0=1, dim1=2)
                    latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot_test)

                    # store
                    trial_z_mean_test += mus.cpu().tolist()

                    # property_prediction
                    trial_test_property_prediction += property_predictor(latent_points).cpu().tolist()

                    # use latent vectors as hidden (not zero-initialized hidden)
                    hidden = vae_decoder.init_hidden(latent_points)
                    out_one_hot = torch.zeros_like(data_test_batch, dtype=dtype, device=device)
                    x_input = torch.zeros_like(data_test_batch[:, 0, :].unsqueeze(0), dtype=dtype, device=device)
                    nop_tensor = -torch.ones_like(data_test_batch[:, 0, 0], dtype=dtype, device=device)
                    asterisk_tensor = -torch.ones_like(data_test_batch[:, 0, 0], dtype=dtype, device=device)

                    # no teacher forcing
                    for seq_index in range(int(rnn_max_length.item())):
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

                        out_one_hot[:, seq_index, :] = out_one_hot_line[0]

                        # change input - no teacher forcing (one-hot)
                        x_input = out_one_hot_line.argmax(dim=-1)
                        x_input = f.one_hot(x_input, num_classes=data_test_batch.shape[2]).to(dtype)

                    # x_indices
                    x_indices = data_test_batch.argmax(dim=-1)
                    x_hat_prob = f.softmax(out_one_hot, dim=-1)
                    x_hat_indices = x_hat_prob.argmax(dim=-1)

                    # change value of asterisk tensor: 0 -> -1
                    asterisk_tensor[asterisk_tensor == 0.0] = -1.0

                    # update nop tensor
                    updated_nop_tensor = -torch.ones_like(data_test_batch[:, 0, 0], dtype=dtype, device=device)
                    for idx in range(asterisk_tensor.shape[0]):
                        if nop_tensor[idx] == -1.0 and asterisk_tensor[idx] == -1.0:
                            continue
                        elif nop_tensor[idx] < 0 < asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = asterisk_tensor[idx]
                        elif nop_tensor[idx] > 0 > asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = nop_tensor[idx]
                        elif nop_tensor[idx] >= asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = asterisk_tensor[idx]
                        else:
                            updated_nop_tensor[idx] = nop_tensor[idx]

                    # modify x_hat_indices value to nop_idx after the nop appears
                    max_length = x_hat_indices.shape[1]
                    for idx, value in enumerate(nop_tensor):
                        if value >= 0.0:
                            x_hat_indices[idx, int(value): max_length] = nop_idx

                    # convert to SMILES string
                    answer_molecules = reconstruct_molecules(encoding_alphabet, x_indices)
                    decoded_molecules = reconstruct_molecules(encoding_alphabet, x_hat_indices)

                    # count right & wrong molecules
                    for count_idx in range(len(answer_molecules)):
                        if answer_molecules[count_idx] == decoded_molecules[count_idx]:
                            count_of_test_reconstruction += 1
                        else:
                            test_answer_polymer.append(answer_molecules[count_idx])
                            test_wrong_prediction_polymer.append(decoded_molecules[count_idx])

            tqdm.write(
                f'Test reconstruction rate is {(100 * (count_of_test_reconstruction / data_test_tensor.shape[0])):.3f}%')

            # store & inverse transform
            test_reconstruction_list.append((100 * (count_of_test_reconstruction/data_test_tensor.shape[0])))
            trial_test_inverse_transformed_property_prediction = property_scaler.inverse_transform(
                trial_test_property_prediction
            )
            test_property_prediction_list.append(trial_test_inverse_transformed_property_prediction)
            z_mean_test.append(trial_z_mean_test)

            # get r2 score
            property_name = ['LogP', 'SC_score']
            for property_idx in range(property_predictor.property_dim):
                test_r2_score_list.append(r2_score(y_true=property_test[:, property_idx],
                                                     y_pred=trial_test_inverse_transformed_property_prediction[:,
                                                            property_idx]))

    # calculating reconstruction percentage / uncertainty
    # train
    train_reconstruction_arr = np.array(train_reconstruction_list)
    train_reconstruction_mean = np.mean(train_reconstruction_arr)
    train_reconstruction_std = np.std(train_reconstruction_arr)

    train_property_prediction_arr = np.array(train_property_prediction_list)
    train_property_prediction_mean = np.mean(train_property_prediction_arr, axis=0)
    train_property_prediction_std = np.std(train_property_prediction_arr, axis=0)

    z_mean_train_arr = np.array(z_mean_train)
    z_mean_train_mean = np.mean(z_mean_train_arr, axis=0)
    z_mean_train_std = np.std(z_mean_train_arr, axis=0)

    # valid
    valid_reconstruction_arr = np.array(valid_reconstruction_list)
    valid_reconstruction_mean = np.mean(valid_reconstruction_arr)
    valid_reconstruction_std = np.std(valid_reconstruction_arr)

    valid_property_prediction_arr = np.array(valid_property_prediction_list)  # shape - (trial, # of data, property_dim)
    valid_property_prediction_mean = np.mean(valid_property_prediction_arr, axis=0)  # shape - (# of data, 1)
    valid_property_prediction_std = np.std(valid_property_prediction_arr, axis=0)  # shape - (# of data, 1)

    z_mean_valid_arr = np.array(z_mean_valid)  # shape - (trial, # of data, latent dim)
    z_mean_valid_mean = np.mean(z_mean_valid_arr, axis=0)
    z_mean_valid_std = np.std(z_mean_valid_arr, axis=0)  # all zero

    # test
    test_reconstruction_arr = np.array(test_reconstruction_list)
    test_reconstruction_mean = np.mean(test_reconstruction_arr)
    test_reconstruction_std = np.std(test_reconstruction_arr)

    test_property_prediction_arr = np.array(test_property_prediction_list)
    test_property_prediction_mean = np.mean(test_property_prediction_arr, axis=0)
    test_property_prediction_std = np.std(test_property_prediction_arr, axis=0)

    z_mean_test_arr = np.array(z_mean_test)
    z_mean_test_mean = np.mean(z_mean_test_arr, axis=0)
    z_mean_test_std = np.std(z_mean_test_arr, axis=0)

    print(f'Total {trial} trials were executed', flush=True)
    print(f'Train mean reconstruction is {train_reconstruction_mean:.3f}% / uncertainty is {train_reconstruction_std:.3f}%', flush=True)
    print(train_reconstruction_arr, flush=True)

    print(
        f'Valid mean reconstruction is {valid_reconstruction_mean:.3f}% / uncertainty is {valid_reconstruction_std:.3f}%',
        flush=True)
    print(valid_reconstruction_arr, flush=True)

    print(
        f'Test mean reconstruction is {test_reconstruction_mean:.3f}% / uncertainty is {test_reconstruction_std:.3f}%',
        flush=True)
    print(test_reconstruction_arr, flush=True)

    # calculating property prediction r2 score / uncertainty
    train_r2_score_arr = np.array(train_r2_score_list)
    train_r2_score_mean = np.mean(train_r2_score_arr)
    train_r2_score_std = np.std(train_r2_score_arr)

    valid_r2_score_arr = np.array(valid_r2_score_list)
    valid_r2_score_mean = np.mean(valid_r2_score_arr)
    valid_r2_score_std = np.std(valid_r2_score_arr)

    test_r2_score_arr = np.array(test_r2_score_list)
    test_r2_score_mean = np.mean(test_r2_score_arr)
    test_r2_score_std = np.std(test_r2_score_arr)

    print(f'Train R2 mean is {train_r2_score_mean:.3f}% / uncertainty is {train_r2_score_std:.3f}%', flush=True)
    print(train_r2_score_arr, flush=True)
    print(f'Valid R2 mean is {valid_r2_score_mean:.3f}% / uncertainty is {valid_r2_score_std:.3f}%', flush=True)
    print(valid_r2_score_arr, flush=True)
    print(f'Test R2 mean is {test_r2_score_mean:.3f}% / uncertainty is {test_r2_score_std:.3f}%', flush=True)
    print(test_r2_score_arr, flush=True)
