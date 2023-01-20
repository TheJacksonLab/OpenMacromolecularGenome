import os
import sys
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.path.append('/home/sk77/PycharmProjects/publish/OMG')
from molecule_chef.mchef.molecule_chef import MoleculeChef
from molecule_chef.module.ggnn_base import GGNNParams
from molecule_chef.module.gated_graph_neural_network import GraphFeaturesStackIndexAdd
from molecule_chef.module.encoder import Encoder
from molecule_chef.module.decoder import Decoder
from molecule_chef.module.gated_graph_neural_network import GGNNSparse
from molecule_chef.module.utils import TorchDetails, FullyConnectedNeuralNetwork, MChefParameters
from molecule_chef.module.utils import get_correct_reactant_bags_batch, PropertyNetworkPredictionModule
from molecule_chef.module.preprocess import AtomFeatureParams

if __name__ == '__main__':
    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    # load parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all/mchef_optuna_10000_6000_3000_500_message_4_objective_1_100_10'
    model_save_directory = os.path.join(save_directory, 'divergence_weight_1.677_latent_dim_38_learning_rate_0.001')

    state_dict = torch.load(os.path.join(model_save_directory, 'mchef_parameters.pth'), map_location=device)
    mchef_parameters = MChefParameters(
        h_layer_size=state_dict['h_layer_size'],
        ggnn_num_layers=state_dict['ggnn_num_layers'],
        graph_embedding_dim=state_dict['graph_embedding_dim'],
        latent_dim=state_dict['latent_dim'],
        encoder_layer_1d_dim=state_dict['encoder_layer_1d_dim'],
        decoder_num_of_layers=state_dict['decoder_num_of_layers'],
        decoder_max_steps=state_dict['decoder_max_steps'],
        property_dim=state_dict['property_dim'],
        decoder_neural_net_hidden_dim=state_dict['decoder_neural_net_hidden_dim'],
        property_network_hidden_sizes=state_dict['property_network_hidden_sizes'],
        property_weights=state_dict['property_weights'],
        dtype=state_dict['dtype'],
        device=state_dict['device']
    )

    # set torch details class
    torch_details = TorchDetails(device=device, data_type=mchef_parameters.dtype)

    # atomic feature parameters for graph embeddings
    atom_feature_parameters = AtomFeatureParams()

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
    ).to(device)

    # graph neural network
    graph_neural_network = GGNNSparse(
        params=graph_neural_network_parameters, graph_feature=graph_featurization
    ).to(device)
    graph_neural_network.load_state_dict(state_dict['nn_graph_neural_network'])

    # set encoder
    encoder = Encoder(
        in_dimension=mchef_parameters.graph_embedding_dim,
        layer_1d=mchef_parameters.encoder_layer_1d_dim,
        latent_dimension=mchef_parameters.latent_dim
    ).to(device)
    encoder.load_state_dict(state_dict['nn_encoder'])

    # set decoder
    decoder = Decoder(
        number_of_layers=mchef_parameters.decoder_num_of_layers,
        max_steps=mchef_parameters.decoder_max_steps,
        graph_embedding_dim=mchef_parameters.graph_embedding_dim,
        latent_dimension=mchef_parameters.latent_dim,
        gru_neural_net_hidden_dim=mchef_parameters.decoder_neural_net_hidden_dim,
        torch_details=torch_details
    ).to(device)
    decoder.load_state_dict(state_dict['nn_decoder'])

    # set property network
    property_network = PropertyNetworkPredictionModule(
        latent_dim=mchef_parameters.latent_dim,
        property_dim=mchef_parameters.property_dim,
        property_network_hidden_dim_list=mchef_parameters.property_network_hidden_sizes,
        dtype=mchef_parameters.dtype,
        device=device,
        weights=mchef_parameters.property_weights
    ).to(device)
    property_network.load_state_dict(state_dict['nn_property_network'])

    stop_embedding = state_dict['nn_stop_embedding']

    # property_scaler
    train_property_values = torch.load(os.path.join(save_directory, 'train_property.pth'))
    property_scaler = StandardScaler()
    property_scaler.fit(train_property_values)

    y_train_true = train_property_values
    y_valid_true = torch.load(os.path.join(save_directory, 'valid_property.pth'))
    y_test_true = torch.load(os.path.join(save_directory, 'test_property.pth'))

    # instantiate molecule chef class
    molecule_chef = MoleculeChef(
        graph_neural_network=graph_neural_network,
        encoder=encoder,
        decoder=decoder,
        property_network=property_network,
        stop_embedding=stop_embedding,
        torch_details=torch_details
    )

    # check accuracy
    train_monomer_bags = torch.load(os.path.join(save_directory, 'train_bags.pth'), map_location=device)
    valid_monomer_bags = torch.load(os.path.join(save_directory, 'valid_bags.pth'), map_location=device)
    test_monomer_bags = torch.load(os.path.join(save_directory, 'test_bags.pth'), map_location=device)
    unique_monomer_sets = torch.load(os.path.join(save_directory, 'unique_monomer_sets.pth'), map_location=device)

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
        trial_train_property_prediction_list = list()
        trial_valid_property_prediction_list = list()
        trial_test_property_prediction_list = list()

        trial_z_mean_train = list()
        trial_z_mean_valid = list()
        trial_z_mean_test = list()

        # reconstruct
        batch_size = 100
        num_batches_train = int(math.ceil(len(train_monomer_bags) / batch_size))
        with torch.no_grad():
            molecule_chef.mchef_module.eval()
            pbar = tqdm(range(num_batches_train), total=num_batches_train, leave=True)
            train_correct_bags = 0
            time.sleep(1.0)
            with pbar as t:
                for batch_idx in t:
                    # divide batch data
                    start_idx = batch_idx * batch_size
                    stop_idx = (batch_idx + 1) * batch_size
                    if batch_idx == num_batches_train - 1:
                        stop_idx = len(train_monomer_bags)
                    batch_train_monomer_bags = train_monomer_bags[start_idx: stop_idx]

                    # get train answer dict
                    train_answer_dict = molecule_chef.get_answer_dict(
                        monomer_bags=batch_train_monomer_bags, unique_monomer_sets=unique_monomer_sets
                    )

                    # get graph embeddings of whole monomer bags
                    all_monomers_graph_embedding_tensor = state_dict['all_monomers_graph_embedding_tensor'].to(device)
                    monomer_bags_graph_embedding_tensor = torch.zeros_like(all_monomers_graph_embedding_tensor[0]).view(1, -1)

                    # get monomer bags embedding tensors
                    for idx_of_bag in range(len(batch_train_monomer_bags)):
                        monomer_bags_graph_embedding = torch.zeros_like(all_monomers_graph_embedding_tensor[0])
                        for monomer_idx in train_answer_dict[idx_of_bag][:-1]:
                            monomer_bags_graph_embedding += all_monomers_graph_embedding_tensor[monomer_idx]
                        monomer_bags_graph_embedding_tensor = torch.cat(
                            [monomer_bags_graph_embedding_tensor, monomer_bags_graph_embedding.view(1, -1)], dim=0
                        )

                    # encode
                    z_samples, mus, log_vars = molecule_chef.mchef_module.encoder(monomer_bags_graph_embedding_tensor[1:])

                    # property prediction
                    property_prediction = molecule_chef.mchef_module.property_network(z_samples)

                    # decode
                    reconstruction_loss, decoded_dict = molecule_chef.mchef_module.decoder(
                        z_samples=z_samples, all_monomer_tensors=all_monomers_graph_embedding_tensor,
                        answer_dict=train_answer_dict, monomer_bag_idx=range(len(train_answer_dict)), teacher_forcing=False
                    )

                    # get the number of correctly predicted bags
                    correct_bags = get_correct_reactant_bags_batch(
                        decoded_dict=decoded_dict, answer_dict=train_answer_dict, device=device,
                        dtype=mchef_parameters.dtype
                    )

                    train_correct_bags += correct_bags

                    # store
                    property_prediction = property_prediction.cpu().tolist()
                    trial_train_property_prediction_list += property_prediction
                    trial_z_mean_train += mus.cpu().tolist()

            # store & inverse transform
            trial_train_inverse_transformed_property_prediction = property_scaler.inverse_transform(
                trial_train_property_prediction_list)
            train_property_prediction_list.append(trial_train_inverse_transformed_property_prediction)
            z_mean_train.append(trial_z_mean_train)
            train_reconstruction_list.append(100 * (train_correct_bags / len(train_monomer_bags)))
            print(f'The train bag accuracy is {100 * (train_correct_bags/len(train_monomer_bags)):.3f}%', flush=True)

            # get r2 score
            property_name = ['LogP', 'SC_score']
            for property_idx in range(mchef_parameters.property_dim):
                train_r2_score_list.append(r2_score(y_true=y_train_true[:, property_idx],
                                                    y_pred=trial_train_inverse_transformed_property_prediction[:, property_idx]))

            # valid
            num_batches_valid = int(math.ceil(len(valid_monomer_bags) / batch_size))
            pbar = tqdm(range(num_batches_valid), total=num_batches_valid, leave=True)
            valid_correct_bags = 0
            time.sleep(1.0)
            with pbar as t:
                for batch_idx in t:
                    # divide batch data
                    start_idx = batch_idx * batch_size
                    stop_idx = (batch_idx + 1) * batch_size
                    if batch_idx == num_batches_valid - 1:
                        stop_idx = len(valid_monomer_bags)
                    batch_valid_monomer_bags = valid_monomer_bags[start_idx: stop_idx]

                    # get valid answer dict
                    valid_answer_dict = molecule_chef.get_answer_dict(
                        monomer_bags=batch_valid_monomer_bags, unique_monomer_sets=unique_monomer_sets
                    )

                    # get graph embeddings of whole monomer bags
                    all_monomers_graph_embedding_tensor = state_dict['all_monomers_graph_embedding_tensor'].to(device)
                    monomer_bags_graph_embedding_tensor = torch.zeros_like(all_monomers_graph_embedding_tensor[0]).view(1, -1)

                    # get monomer bags embedding tensors
                    for idx_of_bag in range(len(batch_valid_monomer_bags)):
                        monomer_bags_graph_embedding = torch.zeros_like(all_monomers_graph_embedding_tensor[0])
                        for monomer_idx in valid_answer_dict[idx_of_bag][:-1]:
                            monomer_bags_graph_embedding += all_monomers_graph_embedding_tensor[monomer_idx]
                        monomer_bags_graph_embedding_tensor = torch.cat(
                            [monomer_bags_graph_embedding_tensor, monomer_bags_graph_embedding.view(1, -1)], dim=0
                        )

                    # encode
                    z_samples, mus, log_vars = molecule_chef.mchef_module.encoder(monomer_bags_graph_embedding_tensor[1:])

                    # property prediction
                    property_prediction = molecule_chef.mchef_module.property_network(z_samples)

                    # decode
                    reconstruction_loss, decoded_dict = molecule_chef.mchef_module.decoder(
                        z_samples=z_samples, all_monomer_tensors=all_monomers_graph_embedding_tensor,
                        answer_dict=valid_answer_dict, monomer_bag_idx=range(len(valid_answer_dict)), teacher_forcing=False,
                        generate=True
                    )

                    # get the number of correctly predicted bags
                    correct_bags = get_correct_reactant_bags_batch(
                        decoded_dict=decoded_dict, answer_dict=valid_answer_dict, device=device,
                        dtype=mchef_parameters.dtype
                    )

                    valid_correct_bags += correct_bags

                    # store
                    property_prediction = property_prediction.cpu().tolist()
                    trial_valid_property_prediction_list += property_prediction
                    trial_z_mean_valid += mus.cpu().tolist()

            # store & inverse transform
            trial_valid_inverse_transformed_property_prediction = property_scaler.inverse_transform(
                trial_valid_property_prediction_list)
            valid_property_prediction_list.append(trial_valid_inverse_transformed_property_prediction)
            z_mean_valid.append(trial_z_mean_valid)
            valid_reconstruction_list.append(100 * (valid_correct_bags / len(valid_monomer_bags)))
            print(f'The valid bag accuracy is {100 * (valid_correct_bags / len(valid_monomer_bags)):.3f}%', flush=True)

            # get r2 score
            property_name = ['LogP', 'SC_score']
            for property_idx in range(mchef_parameters.property_dim):
                valid_r2_score_list.append(r2_score(y_true=y_valid_true[:, property_idx],
                                                    y_pred=trial_valid_inverse_transformed_property_prediction[:,
                                                           property_idx]))

            # test
            num_batches_test = int(math.ceil(len(test_monomer_bags) / batch_size))
            pbar = tqdm(range(num_batches_test), total=num_batches_test, leave=True)
            test_correct_bags = 0
            time.sleep(1.0)
            with pbar as t:
                for batch_idx in t:
                    # divide batch data
                    start_idx = batch_idx * batch_size
                    stop_idx = (batch_idx + 1) * batch_size
                    if batch_idx == num_batches_test - 1:
                        stop_idx = len(test_monomer_bags)
                    batch_test_monomer_bags = test_monomer_bags[start_idx: stop_idx]

                    # get test answer dict
                    test_answer_dict = molecule_chef.get_answer_dict(
                        monomer_bags=batch_test_monomer_bags, unique_monomer_sets=unique_monomer_sets
                    )

                    # get graph embeddings of whole monomer bags
                    all_monomers_graph_embedding_tensor = state_dict['all_monomers_graph_embedding_tensor'].to(device)
                    monomer_bags_graph_embedding_tensor = torch.zeros_like(all_monomers_graph_embedding_tensor[0]).view(1, -1)

                    # get monomer bags embedding tensors
                    for idx_of_bag in range(len(batch_test_monomer_bags)):
                        monomer_bags_graph_embedding = torch.zeros_like(all_monomers_graph_embedding_tensor[0])
                        for monomer_idx in test_answer_dict[idx_of_bag][:-1]:
                            monomer_bags_graph_embedding += all_monomers_graph_embedding_tensor[monomer_idx]
                        monomer_bags_graph_embedding_tensor = torch.cat(
                            [monomer_bags_graph_embedding_tensor, monomer_bags_graph_embedding.view(1, -1)], dim=0
                        )

                    # encode
                    z_samples, mus, log_vars = molecule_chef.mchef_module.encoder(monomer_bags_graph_embedding_tensor[1:])

                    # property prediction
                    property_prediction = molecule_chef.mchef_module.property_network(z_samples)

                    # decode
                    reconstruction_loss, decoded_dict = molecule_chef.mchef_module.decoder(
                        z_samples=z_samples, all_monomer_tensors=all_monomers_graph_embedding_tensor,
                        answer_dict=test_answer_dict, monomer_bag_idx=range(len(test_answer_dict)), teacher_forcing=False,
                        generate=True
                    )

                    # get the number of correctly predicted bags
                    correct_bags = get_correct_reactant_bags_batch(
                        decoded_dict=decoded_dict, answer_dict=test_answer_dict, device=device,
                        dtype=mchef_parameters.dtype
                    )

                    test_correct_bags += correct_bags

                    # store
                    property_prediction = property_prediction.cpu().tolist()
                    trial_test_property_prediction_list += property_prediction
                    trial_z_mean_test += mus.cpu().tolist()

            # store & inverse transform
            trial_test_inverse_transformed_property_prediction = property_scaler.inverse_transform(trial_test_property_prediction_list)
            test_property_prediction_list.append(trial_test_inverse_transformed_property_prediction)
            z_mean_test.append(trial_z_mean_test)
            test_reconstruction_list.append(100 * (test_correct_bags / len(test_monomer_bags)))
            print(f'The test bag accuracy is {100 * (test_correct_bags / len(test_monomer_bags)):.3f}%', flush=True)

            # get r2 score
            property_name = ['LogP', 'SC_score']
            for property_idx in range(mchef_parameters.property_dim):
                test_r2_score_list.append(r2_score(y_true=y_test_true[:, property_idx],
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

    # plot property prediction
    y_train_true = train_property_values
    y_valid_true = torch.load(os.path.join(save_directory, 'valid_property.pth'))
    y_test_true = torch.load(os.path.join(save_directory, 'test_property.pth'))

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
