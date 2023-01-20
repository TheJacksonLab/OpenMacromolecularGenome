import os
import sys
import time
import numpy as np
import pandas as pd

import selfies as sf

import torch
import torch.nn.functional as f

from torch.distributions.multivariate_normal import MultivariateNormal

from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdmolops import GetShortestPath

sys.path.append('/home/sk77/PycharmProjects/publish/OMG')
from vae.decoder.torch import Decoder
from vae.encoder.torch import Encoder, CNNEncoder
from vae.property_predictor.torch import PropertyNetworkPredictionModule
from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_smile_to_hot, multiple_selfies_to_hot


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


if __name__ == '__main__':
    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # set torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    # load vae parameters
    load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all/'
    save_directory = os.path.join(load_directory, 'vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay')
    model_save_directory = os.path.join(save_directory, 'divergence_weight_4.345_latent_dim_152_learning_rate_0.002')

    vae_parameters = torch.load(os.path.join(model_save_directory, 'vae_parameters.pth'), map_location=device)
    nop_idx = vae_parameters['nop_idx']
    asterisk_idx = vae_parameters['asterisk_idx']
    dtype = vae_parameters['dtype']

    # load data
    # data_path = os.path.join(load_directory, 'reaction_3_150K.csv')
    data_path = os.path.join(load_directory, '50000_15000_6000.csv')
    data_path = os.path.join(load_directory, 'all_reactions_10000_6000_3000_500.csv')
    df_polymer = pd.read_csv(data_path)

    # load SELFIES data - due to the random order
    encoding_list = torch.load(os.path.join(save_directory, 'encoding_list.pth'), map_location=device)
    encoding_alphabet = torch.load(os.path.join(save_directory, 'encoding_alphabet.pth'), map_location=device)
    largest_molecule_len = torch.load(os.path.join(save_directory, 'largest_molecule_len.pth'), map_location=device)

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
    vae_encoder.load_state_dict(torch.load(os.path.join(model_save_directory, 'encoder.pth'), map_location=device))

    # set decoder
    vae_decoder = Decoder(
        input_size=vae_parameters['decoder_input_dimension'],
        num_layers=vae_parameters['decoder_num_gru_layers'],
        hidden_size=vae_parameters['latent_dimension'],
        out_dimension=vae_parameters['decoder_output_dimension'],
        bidirectional=vae_parameters['decoder_bidirectional']
    ).to(device)
    vae_decoder.load_state_dict(torch.load(os.path.join(model_save_directory, 'decoder.pth'), map_location=device))

    # set property prediction network
    property_predictor = PropertyNetworkPredictionModule(
        latent_dim=vae_parameters['latent_dimension'],
        property_dim=vae_parameters['property_dim'],
        property_network_hidden_dim_list=vae_parameters['property_network_hidden_dim_list'],
        dtype=dtype,
        device=device,
        weights=vae_parameters['property_weights']
    ).to(device)
    property_predictor.load_state_dict(torch.load(os.path.join(model_save_directory, 'property_predictor.pth'), map_location=device))

    # eval
    vae_encoder.eval()
    vae_decoder.eval()
    property_predictor.eval()

    # gaussian prior generation
    number_of_generation = 30000
    multivariate_normal = MultivariateNormal(
        torch.zeros(vae_parameters['latent_dimension'], dtype=dtype, device=device),
        torch.eye(vae_parameters['latent_dimension'], dtype=dtype, device=device)
    )

    # generated polymer maximum length is less than original polymers
    rnn_max_length = largest_molecule_len
    generated_polymer_list = list()

    tqdm.write("Generation ...")
    pbar = tqdm(range(number_of_generation), total=number_of_generation, leave=True)
    time.sleep(1.0)
    with pbar as t:
        for iteration in t:
            with torch.no_grad():
                # sample gaussian prior
                random_prior = multivariate_normal.sample().unsqueeze(0)
                hidden = vae_decoder.init_hidden(random_prior)
                out_one_hot = torch.zeros(size=(1, largest_molecule_len, len(encoding_alphabet)),
                                          dtype=dtype, device=device)
                nop_tensor = -torch.ones(size=(1, ), dtype=dtype, device=device)
                asterisk_tensor = -torch.ones(size=(1,), dtype=dtype, device=device)
                updated_nop_tensor = -torch.ones(size=(1,), dtype=dtype, device=device)
                x_input = torch.zeros(size=(1, 1, len(encoding_alphabet)), dtype=dtype, device=device)

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
                        continue
                    else:
                        generated_polymer_list.append(polymer)

    # 1) check validity
    number_of_valid_polymers = len(generated_polymer_list)

    print(f"Gaussian prior generated valid {number_of_valid_polymers} polymers "
          f"{100 * (number_of_valid_polymers/number_of_generation):.3f}% ",
          flush=True)
    print('====================', flush=True)

    # 2) check uniqueness
    generated_unique_polymer_list = list(set(generated_polymer_list))

    print(f"Gaussian prior generated unique {len(generated_unique_polymer_list)} polymers "
          f"{100 * (len(generated_unique_polymer_list)/number_of_valid_polymers):.3f}% "
          f"among valid polymers", flush=True)

    print('====================', flush=True)

    # 3) check novelty
    # novel polymers, not overlapping with training polymers

    # load train data
    train_idx = torch.load(os.path.join(save_directory, 'train_idx.pth'))
    train_polymers = df_polymer['product'][train_idx]

    generated_novel_unique_polymer_list = list(set(generated_unique_polymer_list) - set(train_polymers))

    print(f"Gaussian prior generated novel, unique {len(generated_novel_unique_polymer_list)} polymers "
          f"{100 * (len(generated_novel_unique_polymer_list)/number_of_valid_polymers):.3f}% "
          f"among valid polymers", flush=True)

    print('====================', flush=True)

    # save
    torch.save(generated_novel_unique_polymer_list, os.path.join(
        model_save_directory, f'gaussian_prior_{number_of_generation}_generations_polymers.pth'))
