import os
import pickle
import sys
import time
import torch
import random

import numpy as np
import pandas as pd

import optuna

from optuna.trial import TrialState

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from multiprocessing import Manager
from joblib import parallel_backend
# "https://github.com/rapidsai/dask-cuda/issues/789"

from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdmolops import GetShortestPath

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append('/home/sk77/PycharmProjects/publish/OMG')
from vae.decoder.torch import Decoder
from vae.encoder.torch import CNNEncoder
from vae.preprocess import get_selfie_and_smiles_encodings_for_dataset, multiple_selfies_to_hot
from vae.property_predictor.torch import PropertyNetworkPredictionModule
from vae.training_optuna import train_model
from vae.utils.save import VAEParameters


def get_number_of_backbone_atoms(target_mol):
    asterisk_idx_list = list()
    for idx in range(target_mol.GetNumAtoms()):
        atom = target_mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == '*':
            asterisk_idx_list.append(idx)
    dist = GetShortestPath(target_mol, asterisk_idx_list[0], asterisk_idx_list[1])
    backbone_length = len(dist) - 2

    return backbone_length


class Objective:
    def __init__(self, gpu_queue):
        # Shared queue to manage GPU IDs.
        self.gpu_queue = gpu_queue

    def __call__(self, trial):
        # Fetch GPU ID for this trial.
        gpu_id = self.gpu_queue.get()

        # optimization
        with torch.cuda.device(gpu_id):
            # set torch details class
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device, flush=True)
            dtype = torch.float32

            # set hyperparameters
            divergence_weight = trial.suggest_float(name="divergence_weight", low=1e-1, high=10, log=True)
            latent_space_dim = trial.suggest_int(name="latent_space_dim", low=64, high=512, log=True)
            learning_rate = trial.suggest_float(name="learning_rate", low=1e-5, high=1e-2, log=True)

            # encoder_layer_1d = trial.suggest_int(name="encoder_layer_1d", low=512, high=1024, log=True)
            # encoder_layer_2d = trial.suggest_int(name="encoder_layer_2d", low=256, high=512, log=True)
            encoder_layer_1d = 1024
            encoder_layer_2d = 512

            # CNN parameters
            # channel_1_dim = trial.suggest_categorical(name="channel_1", choices=[3, 5, 7, 9])
            # kernel_1_dim = trial.suggest_categorical(name="kernel_1", choices=[3, 5, 7, 9])
            #
            # channel_2_dim = trial.suggest_categorical(name="channel_2", choices=[3, 5, 7, 9])
            # kernel_2_dim = trial.suggest_categorical(name="kernel_2", choices=[3, 5, 7, 9])
            #
            # channel_3_dim = trial.suggest_categorical(name="channel_3", choices=[3, 5, 7, 9])
            # kernel_3_dim = trial.suggest_categorical(name="kernel_3", choices=[3, 5, 7, 9])
            channel_1_dim = 9
            kernel_1_dim = 9

            channel_2_dim = 9
            kernel_2_dim = 9

            channel_3_dim = 9
            kernel_3_dim = 9

            # decoder parameter
            # decoder_num_gru_layers = trial.suggest_int(name="decoder_num_gru_layers", low=2, high=8)
            decoder_num_gru_layers = 4

            data_train_tensor = torch.tensor(data_train, dtype=dtype).to(device)
            data_valid_tensor = torch.tensor(data_valid, dtype=dtype).to(device)
            y_train_tensor = torch.tensor(property_train_scaled, dtype=dtype).to(device)
            y_valid_tensor = torch.tensor(property_valid_scaled, dtype=dtype).to(device)

            # set VAE parameters
            vae_parameters = VAEParameters(
                data_path=data_path,
                save_directory=save_directory,
                nop_idx=nop_idx,
                asterisk_idx=asterisk_idx,
                latent_dimension=latent_space_dim,
                encoder_in_channels=data.shape[2],
                encoder_feature_dim=data.shape[1],
                encoder_convolution_channel_dim=[channel_1_dim, channel_2_dim, channel_3_dim],
                encoder_kernel_size=[kernel_1_dim, kernel_2_dim, kernel_3_dim],
                encoder_layer_1d=encoder_layer_1d,
                encoder_layer_2d=encoder_layer_2d,
                decoder_input_dimension=len_alphabet,
                decoder_output_dimension=len_alphabet,
                decoder_num_gru_layers=decoder_num_gru_layers,
                decoder_bidirectional=True,
                property_dim=1,
                property_network_hidden_dim_list=[[64, 16]],
                property_weights=(0.5,),
                dtype=dtype,
                device=device,
                test_size=0.1,
                random_state=42
            )

            # set encoder
            encoder = CNNEncoder(
                in_channels=vae_parameters.encoder_in_channels,
                feature_dim=vae_parameters.encoder_feature_dim,
                convolution_channel_dim=vae_parameters.encoder_convolution_channel_dim,
                kernel_size=vae_parameters.encoder_kernel_size,
                layer_1d=vae_parameters.encoder_layer_1d,
                layer_2d=vae_parameters.encoder_layer_2d,
                latent_dimension=vae_parameters.latent_dimension
            ).to(device)

            # set decoder
            decoder = Decoder(
                input_size=vae_parameters.decoder_input_dimension,
                num_layers=vae_parameters.decoder_num_gru_layers,
                hidden_size=vae_parameters.latent_dimension,
                out_dimension=vae_parameters.decoder_output_dimension,
                bidirectional=vae_parameters.decoder_bidirectional
            ).to(device)

            # set property prediction network
            property_network_module = PropertyNetworkPredictionModule(
                latent_dim=vae_parameters.latent_dimension,
                property_dim=vae_parameters.property_dim,
                property_network_hidden_dim_list=vae_parameters.property_network_hidden_dim_list,
                dtype=dtype,
                device=device,
                weights=vae_parameters.property_weights
            ).to(device)

            # training
            model_save_directory = os.path.join(save_directory,
                                                f'divergence_weight_{divergence_weight:.3f}_latent_dim_{latent_space_dim}_learning_rate_{learning_rate:.3f}')
            if not os.path.exists(model_save_directory):
                Path(model_save_directory).mkdir(parents=True)

            print("start training", flush=True)
            start = time.time()
            reconstruction_loss, divergence_loss, property_loss = train_model(
                vae_encoder=encoder,
                vae_decoder=decoder,
                property_predictor=property_network_module,
                nop_idx=vae_parameters.nop_idx,
                asterisk_idx=vae_parameters.asterisk_idx,
                data_train=data_train_tensor,
                data_valid=data_valid_tensor,
                y_train=y_train_tensor,
                y_valid=y_valid_tensor,
                y_scaler=property_scaler,
                num_epochs=60,
                batch_size=32,
                lr_enc=learning_rate,
                lr_dec=learning_rate,
                lr_property=learning_rate,
                weight_decay=1e-5,
                mmd_weight=divergence_weight,
                save_directory=model_save_directory,
                dtype=dtype,
                device=device
            )

            # save parameter
            save_parameter_dict = dict()
            info = vars(vae_parameters)
            for key, value in info.items():
                save_parameter_dict[key] = value

            torch.save(save_parameter_dict, os.path.join(model_save_directory, 'vae_parameters.pth'))

            end = time.time()
            print("total %.3f minutes took for training" % ((end - start) / 60.0), flush=True)

            # return GPU ID to the queue
            time.sleep(random.randint(1, 8))
            self.gpu_queue.put(gpu_id)

            return reconstruction_loss + divergence_loss * 10 + property_loss * 10


def tune_hyperparameters():
    """
    Tune hyperparmeters using Optuna
    https://github.com/optuna/optuna/issues/1365
    """
    study = optuna.create_study(
        direction='minimize'
    )

    # parallel optimization
    n_gpus = 2
    with Manager() as manager:
        # Initialize the queue by adding available GPU IDs.
        gpu_queue = manager.Queue()
        # for i in range(n_gpus):
        #     gpu_queue.put(i)
        for i in range(2, 4):
            gpu_queue.put(i)

        with parallel_backend("dask", n_jobs=n_gpus):
            study.optimize(
                func=Objective(gpu_queue),
                n_trials=32,
                n_jobs=n_gpus
            )

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ", flush=True)
    print("  Number of finished trials: ", len(study.trials), flush=True)
    print("  Number of pruned trials: ", len(pruned_trials), flush=True)
    print("  Number of complete trials: ", len(complete_trials), flush=True)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Show results.
    print(study.trials_dataframe().to_string())


if __name__ == '__main__':
    # set clients
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all'
    data_path = os.path.join(load_directory, 'all_reactions_10000_6000_3000_500.csv')

    # load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/six'
    # data_path = os.path.join(load_directory, '50000_15000_6000.csv')

    # load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/one'
    # data_path = os.path.join(load_directory, 'reaction_3_150K.csv')

    df_polymer = pd.read_csv(data_path)
    df_polymer = df_polymer.drop_duplicates().reset_index(drop=True)
    print(df_polymer.shape, flush=True)

    save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all/vae_optuna_10000_6000_3000_500_objective_1_10_10_1e-5_weight_decay'
    # save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/six/vae_optuna_50000_15000_6000_objective_1_10_10_1e-5_weight_decay'
    # save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/one/vae_optuna_1e-5_weight_decay'

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    print('Representation: SELFIES', flush=True)
    encoding_list, encoding_alphabet, largest_molecule_len, _, _, _ = \
        get_selfie_and_smiles_encodings_for_dataset(df_polymer)
    print('--> Creating one-hot encoding...', flush=True)
    data = multiple_selfies_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)
    print('Finished creating one-hot encoding.', flush=True)

    # exclude the first character - [*]
    data = data[:, 1:, :]

    # save data - due to the random order
    torch.save(data, os.path.join(save_directory, 'data.pth'), pickle_protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(encoding_list, os.path.join(save_directory, 'encoding_list.pth'))
    torch.save(encoding_alphabet, os.path.join(save_directory, 'encoding_alphabet.pth'))
    torch.save(largest_molecule_len, os.path.join(save_directory, 'largest_molecule_len.pth'))

    # set parameters
    print(data.shape, flush=True)
    len_max_molec = data.shape[1]
    print(len_max_molec, flush=True)
    len_alphabet = data.shape[2]
    print(len_alphabet, flush=True)

    nop_idx = 0
    asterisk_idx = 0

    # find '[nop]' and '[*]'
    for idx, component in enumerate(encoding_alphabet):
        if component == '[nop]':
            nop_idx = idx
        if component == '[*]':
            asterisk_idx = idx

    print(f'nop idx is {nop_idx}', flush=True)
    print(f'asterisk idx is {asterisk_idx}', flush=True)

    # load idx
    train_monomer_bags_idx = torch.load(os.path.join(load_directory, 'all_reactions_train_bag_idx_10000_6000_3000_500.pth'))
    temp_idx = torch.load(os.path.join(load_directory, 'all_reactions_test_bag_idx_10000_6000_3000_500.pth'))

    # train_monomer_bags_idx = torch.load(os.path.join(load_directory, 'train_bag_idx_50000_15000_6000.pth'))
    # temp_idx = torch.load(os.path.join(load_directory, 'test_bag_idx_50000_15000_6000.pth'))

    # train_monomer_bags_idx = torch.load(os.path.join(load_directory, 'train_bag_idx_reaction_3_150K.pth'))
    # temp_idx = torch.load(os.path.join(load_directory, 'test_bag_idx_reaction_3_150K.pth'))

    # train : valid : test split - 8 : 1 : 1
    temp_train_idx, test_monomer_bags_idx = train_test_split(temp_idx, test_size=int(df_polymer.shape[0] * (1 / 10)),
                                                             random_state=42)
    temp_train_idx, valid_monomer_bags_idx = train_test_split(temp_train_idx, test_size=int(df_polymer.shape[0] * (1 / 10)),
                                                              random_state=42)
    train_monomer_bags_idx += temp_train_idx

    torch.save(train_monomer_bags_idx, os.path.join(save_directory, 'train_idx.pth'))
    torch.save(valid_monomer_bags_idx, os.path.join(save_directory, 'valid_idx.pth'))
    torch.save(test_monomer_bags_idx, os.path.join(save_directory, 'test_idx.pth'))

    # preparation training data
    df_polymer['mol'] = df_polymer['product'].apply(lambda x: Chem.MolFromSmiles(x))
    df_polymer['LogP'] = df_polymer['mol'].apply(lambda x: MolLogP(x) / get_number_of_backbone_atoms(x))
    df_polymer = df_polymer.drop(['mol'], axis=1)

    property_value = df_polymer['LogP'].to_numpy().reshape(-1, 1)
    torch.save(property_value, os.path.join(save_directory, 'property_value.pth'))

    data_train = data[train_monomer_bags_idx]
    data_valid = data[valid_monomer_bags_idx]
    property_train = property_value[train_monomer_bags_idx]
    property_valid = property_value[valid_monomer_bags_idx]

    # property scaling
    property_scaler = StandardScaler()
    property_train_scaled = property_scaler.fit_transform(property_train)
    property_valid_scaled = property_scaler.transform(property_valid)

    tune_hyperparameters()


