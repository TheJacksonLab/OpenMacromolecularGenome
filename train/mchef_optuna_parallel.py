import os
import torch
import torch.nn as nn
import torch.nn.functional as f

import optuna

from optuna.trial import TrialState

from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from multiprocessing import Manager
from joblib import parallel_backend
# "https://github.com/rapidsai/dask-cuda/issues/789"

import time
import random

import sys
sys.path.append('/home/sk77/PycharmProjects/publish/OMG')

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.rdmolops import GetShortestPath
from rdkit.Chem.Crippen import MolLogP

from pathlib import Path

from sklearn.model_selection import train_test_split

from molecule_chef.mchef.molecule_chef_optuna import MoleculeChef
from molecule_chef.module.ggnn_base import GGNNParams
from molecule_chef.module.gated_graph_neural_network import GraphFeaturesStackIndexAdd
from molecule_chef.module.encoder import Encoder
from molecule_chef.module.decoder import Decoder
from molecule_chef.module.gated_graph_neural_network import GGNNSparse
from molecule_chef.module.utils import TorchDetails, FullyConnectedNeuralNetwork, save_model
from molecule_chef.module.utils import MChefParameters, PropertyNetworkPredictionModule
from molecule_chef.module.preprocess import AtomFeatureParams


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
            torch_details = TorchDetails(device=device, data_type=torch.float32)

            # set hyperparameters
            # divergence_weight = trial.suggest_float(name="divergence_weight", low=1e-1, high=10, log=True)
            # ggnn_num_layers = trial.suggest_int(name="ggnn_num_layers", low=1, high=3)
            # graph_embedding_dim = trial.suggest_int(name="graph_embedding_dim", low=64, high=101)
            # encoder_layer_1d_dim = trial.suggest_int(name='encoder_layer_1d_dim', low=48, high=64)
            # latent_space_dim = trial.suggest_int(name="latent_space_dim", low=24, high=48)
            # learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=1e-2, log=True)
            # decoder_num_of_layers = trial.suggest_int(name="decoder_num_of_layers", low=2, high=5)
            # decoder_neural_net_hidden_dim = trial.suggest_int(name="decoder_neural_net_hidden_dim", low=64, high=101)

            divergence_weight = trial.suggest_float(name="divergence_weight", low=1e-1, high=10, log=True)
            latent_space_dim = trial.suggest_int(name="latent_space_dim", low=25, high=50)
            learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=1e-2, log=True)
            ggnn_num_layers = 4
            graph_embedding_dim = 50
            encoder_layer_1d_dim = 200
            decoder_num_of_layers = 2
            decoder_neural_net_hidden_dim = 128

            # atomic feature parameters for graph embeddings
            atom_feature_parameters = AtomFeatureParams()

            # set parameters
            mchef_parameters = MChefParameters(
                h_layer_size=atom_feature_parameters.atom_feature_length,  # 101
                ggnn_num_layers=ggnn_num_layers,
                graph_embedding_dim=graph_embedding_dim,
                latent_dim=latent_space_dim,
                encoder_layer_1d_dim=encoder_layer_1d_dim,
                decoder_num_of_layers=decoder_num_of_layers,
                decoder_max_steps=5,
                decoder_neural_net_hidden_dim=[decoder_neural_net_hidden_dim],
                property_network_hidden_sizes=[[32, 16]],
                property_dim=1,  # property dimension
                property_weights=(1.0,),
                dtype=torch_details.data_type,
                device=device,
            )

            # set gated graph neural network parameters
            graph_neural_network_parameters = GGNNParams(
                h_layer_size=mchef_parameters.h_layer_size,
                edge_names=atom_feature_parameters.bond_names,
                num_layers=mchef_parameters.ggnn_num_layers,
                torch_details=torch_details
            )

            # set graph featurization networks
            graph_featurization = GraphFeaturesStackIndexAdd(
                neural_net_project=FullyConnectedNeuralNetwork(
                    input_dim=mchef_parameters.h_layer_size,
                    output_dim=mchef_parameters.graph_embedding_dim,
                    hidden_sizes=[]
                ),
                neural_net_gate=FullyConnectedNeuralNetwork(
                    input_dim=mchef_parameters.h_layer_size,
                    output_dim=mchef_parameters.graph_embedding_dim,
                    hidden_sizes=[]
                ),
                torch_details=torch_details
            )

            # graph neural network
            graph_neural_network = GGNNSparse(
                params=graph_neural_network_parameters, graph_feature=graph_featurization
            ).to(device)

            # set encoder
            encoder = Encoder(
                in_dimension=mchef_parameters.graph_embedding_dim,
                layer_1d=mchef_parameters.encoder_layer_1d_dim,
                latent_dimension=mchef_parameters.latent_dim
            ).to(device)

            # set decoder
            decoder = Decoder(
                number_of_layers=mchef_parameters.decoder_num_of_layers,
                max_steps=mchef_parameters.decoder_max_steps,
                graph_embedding_dim=mchef_parameters.graph_embedding_dim,
                latent_dimension=mchef_parameters.latent_dim,
                gru_neural_net_hidden_dim=mchef_parameters.decoder_neural_net_hidden_dim,
                torch_details=torch_details
            ).to(device)

            # set property prediction network
            property_network_module = PropertyNetworkPredictionModule(
                latent_dim=mchef_parameters.latent_dim,
                property_dim=mchef_parameters.property_dim,
                property_network_hidden_dim_list=mchef_parameters.property_network_hidden_sizes,
                dtype=torch_details.data_type,
                device=torch_details.device,
                weights=mchef_parameters.property_weights
            ).to(device)

            # set stop embedding
            stop_embedding = nn.Parameter(
                torch.empty(mchef_parameters.graph_embedding_dim, dtype=torch_details.data_type,
                            device=torch_details.device)
            )
            bound = 1 / np.sqrt(mchef_parameters.graph_embedding_dim)
            nn.init.uniform_(stop_embedding, -bound, bound)

            # instantiate molecule chef class
            molecule_chef = MoleculeChef(
                graph_neural_network=graph_neural_network,
                encoder=encoder,
                decoder=decoder,
                property_network=property_network_module,
                stop_embedding=stop_embedding,
                torch_details=torch_details
            )

            # train Molecule Chef
            model_save_directory = os.path.join(save_directory,
                                                f'divergence_weight_{divergence_weight:.3f}_latent_dim_{latent_space_dim}_learning_rate_{learning_rate:.3f}')
            reconstruction_loss, divergence_loss, property_loss = molecule_chef.train(
                num_epochs=60,
                batch_size=32,
                save_directory=model_save_directory,
                lr=learning_rate,
                weight_decay=1e-4,
                mmd_weight=divergence_weight,
                train_monomer_bags=train_monomer_bags,
                test_monomer_bags=valid_monomer_bags,
                unique_monomer_sets=unique_monomer_sets,
                unique_mols=unique_mols,
                train_property_data=train_property,
                test_property_data=valid_property
            )

        save_parameter_dict = dict()
        info = vars(mchef_parameters)
        for key, value in info.items():
            save_parameter_dict[key] = value

        # save model
        info = vars(molecule_chef)
        cpu_device = torch.device('cpu')
        for key, value in info.items():
            if isinstance(value, torch.nn.Module):
                save_parameter_dict['nn_graph_neural_network'] = value.graph_neural_network.state_dict()
                save_parameter_dict['nn_encoder'] = value.encoder.state_dict()
                save_parameter_dict['nn_decoder'] = value.decoder.state_dict()
                save_parameter_dict['nn_property_network'] = value.property_network.state_dict()
                save_parameter_dict['nn_stop_embedding'] = value.stop_embedding

            elif isinstance(value, torch.Tensor):
                value.to(cpu_device)
                save_parameter_dict[key] = value

            else:
                save_parameter_dict[key] = value

        save_model(save_parameter_dict, save_directory=model_save_directory, name='mchef_parameters.pth')

        # plot result
        molecule_chef.plot_learning_curve()

        # return GPU ID to the queue
        time.sleep(random.randint(1, 8))
        self.gpu_queue.put(gpu_id)

        return reconstruction_loss + divergence_loss * 100 + property_loss * 10


def tune_hyperparameters():
    """
    Tune hyperparmeters using Optuna
    https://github.com/optuna/optuna/issues/1365
    """
    study = optuna.create_study(
        direction='minimize'
    )

    # parallel optimization
    n_gpus = 1
    with Manager() as manager:
        # Initialize the queue by adding available GPU IDs.
        gpu_queue = manager.Queue()
        for i in range(2, 3):
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

    # set directory
    # load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/six'
    load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all'
    # load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/one'

    # save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/one/mchef_optuna_3_150K_message_4_objective_1_100_10'
    # save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/six/mchef_50000_15000_6000_message_4_objective_1_100_10'
    save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all/mchef_optuna_10000_6000_3000_500_message_4_objective_1_100_10'

    if not os.path.exists(save_directory):
        Path(save_directory).mkdir(parents=True)

    # load and split data
    # df = pd.read_csv(os.path.join(load_directory, '50000_15000_6000.csv'))
    df = pd.read_csv(os.path.join(load_directory, 'all_reactions_10000_6000_3000_500.csv'))
    # df = pd.read_csv(os.path.join(load_directory, 'reaction_3_150K.csv'))

    print(df.shape, flush=True)

    # load idx
    # train_monomer_bags_idx = torch.load(os.path.join(load_directory, 'train_bag_idx_reaction_3_150K.pth'))
    # temp_idx = torch.load(os.path.join(load_directory, 'test_bag_idx_reaction_3_150K.pth'))

    # train_monomer_bags_idx = torch.load(os.path.join(load_directory, 'train_bag_idx_50000_15000_6000.pth'))
    # temp_idx = torch.load(os.path.join(load_directory, 'test_bag_idx_50000_15000_6000.pth'))

    train_monomer_bags_idx = torch.load(os.path.join(load_directory, 'all_reactions_train_bag_idx_10000_6000_3000_500.pth'))
    temp_idx = torch.load(os.path.join(load_directory, 'all_reactions_test_bag_idx_10000_6000_3000_500.pth'))

    # train : valid : test split - 8 : 1 : 1
    temp_train_idx, test_monomer_bags_idx = train_test_split(temp_idx, test_size=int(df.shape[0] * (1 / 10)),
                                                             random_state=42)
    temp_train_idx, valid_monomer_bags_idx = train_test_split(temp_train_idx, test_size=int(df.shape[0] * (1 / 10)),
                                                              random_state=42)
    train_monomer_bags_idx += temp_train_idx

    # get monomer bags
    train_monomer_bags = list()
    valid_monomer_bags = list()
    test_monomer_bags = list()

    train_property = list()
    valid_property = list()
    test_property = list()

    # get LogP
    df['mol'] = df['product'].apply(lambda x: Chem.MolFromSmiles(x))
    df['LogP'] = df['mol'].apply(lambda x: MolLogP(x) / get_number_of_backbone_atoms(x))
    df = df.drop(['mol'], axis=1)

    # get two reactants train bags
    for row_idx in train_monomer_bags_idx:
        row = df.loc[row_idx, :]
        train_monomer_bags.append([row['reactant_1'], row['reactant_2']])
        property_list = list()
        property_list.append(row['LogP'])
        train_property.append(property_list)

    # get two reactants valid bags
    for row_idx in valid_monomer_bags_idx:
        row = df.loc[row_idx, :]
        valid_monomer_bags.append([row['reactant_1'], row['reactant_2']])
        property_list = list()
        property_list.append(row['LogP'])
        valid_property.append(property_list)

    # get two reactants test bags
    for row_idx in test_monomer_bags_idx:
        row = df.loc[row_idx, :]
        test_monomer_bags.append([row['reactant_1'], row['reactant_2']])
        property_list = list()
        property_list.append(row['LogP'])
        test_property.append(property_list)

    train_property = np.array(train_property)
    valid_property = np.array(valid_property)
    test_property = np.array(test_property)

    print(train_property.shape, flush=True)
    print(valid_property.shape, flush=True)
    print(test_property.shape, flush=True)

    # get unique monomer bags
    unique_monomer_sets = np.array(
        list(set([monomer for monomer_bag in train_monomer_bags for monomer in monomer_bag])), dtype=object
    )
    print(f'The number of unique monomers is {len(unique_monomer_sets)}', flush=True)

    unique_mols = [Chem.MolFromSmiles(monomer) for monomer in unique_monomer_sets]
    for mol in unique_mols:  # kekulize
        Chem.Kekulize(mol)

    # save train / valid / test data
    save_model(train_monomer_bags, save_directory=save_directory, name='train_bags.pth')
    save_model(valid_monomer_bags, save_directory=save_directory, name='valid_bags.pth')
    save_model(test_monomer_bags, save_directory=save_directory, name='test_bags.pth')

    save_model(train_property, save_directory=save_directory, name='train_property.pth')
    save_model(valid_property, save_directory=save_directory, name='valid_property.pth')
    save_model(test_property, save_directory=save_directory, name='test_property.pth')
    save_model(unique_monomer_sets, save_directory=save_directory, name='unique_monomer_sets.pth')

    tune_hyperparameters()
