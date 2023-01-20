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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # load parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all/mchef_optuna_10000_6000_3000_500_message_4_objective_1_100_10'
    model_save_directory = os.path.join(save_directory, 'divergence_weight_1.940_latent_dim_25_learning_rate_0.000')

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

    # check reconstruction rate
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
                property_prediction = molecule_chef.mchef_module.property_network(mus)

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
                z_mean_train += mus.cpu().tolist()
                property_prediction = property_prediction.cpu().tolist()
                train_property_prediction_list += property_prediction

        print(f'The train bag accuracy is {100 * (train_correct_bags/len(train_monomer_bags)):.3f}%', flush=True)

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
                property_prediction = molecule_chef.mchef_module.property_network(mus)

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
                z_mean_valid += mus.cpu().tolist()
                property_prediction = property_prediction.cpu().tolist()
                valid_property_prediction_list += property_prediction

        print(f'The valid bag accuracy is {100 * (valid_correct_bags / len(valid_monomer_bags)):.3f}%', flush=True)

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
                property_prediction = molecule_chef.mchef_module.property_network(mus)

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
                z_mean_test += mus.cpu().tolist()
                property_prediction = property_prediction.cpu().tolist()
                test_property_prediction_list += property_prediction

        print(f'The test bag accuracy is {100 * (test_correct_bags / len(test_monomer_bags)):.3f}%', flush=True)

    # plot property prediction
    y_train_true = train_property_values
    y_valid_true = torch.load(os.path.join(save_directory, 'valid_property.pth'))
    y_test_true = torch.load(os.path.join(save_directory, 'test_property.pth'))

    y_train_prediction = property_scaler.inverse_transform(train_property_prediction_list)
    y_valid_prediction = property_scaler.inverse_transform(valid_property_prediction_list)
    y_test_prediction = property_scaler.inverse_transform(test_property_prediction_list)

    # get r2 score & plot
    property_name = ['LogP', 'SC_score']
    for idx in range(mchef_parameters.property_dim):
        train_r2_score = r2_score(y_true=y_train_true[:, idx], y_pred=y_train_prediction[:, idx])
        valid_r2_score = r2_score(y_true=y_valid_true[:, idx], y_pred=y_valid_prediction[:, idx])
        test_r2_score = r2_score(y_true=y_test_true[:, idx], y_pred=y_test_prediction[:, idx])

        plt.figure(figsize=(6, 6), dpi=300)
        plt.plot(y_train_true[:, idx], y_train_prediction[:, idx], 'bo',
                 label='Train R2 score: %.3f' % train_r2_score)
        plt.plot(y_valid_true[:, idx], y_valid_prediction[:, idx], 'go',
                 label='Valid R2 score: %.3f' % valid_r2_score)
        plt.plot(y_test_true[:, idx], y_test_prediction[:, idx], 'ro',
                 label='Test R2 score: %.3f' % test_r2_score)

        plt.legend()
        plt.xlabel('True')
        plt.ylabel('Prediction')

        plt.savefig(os.path.join(model_save_directory, "recovered_property_prediction_%s.png" % property_name[idx]))
        plt.show()
        plt.close()

    # convert to numpy
    z_mean = np.array(z_mean_train)
    property_value = np.array(y_train_prediction)

    # PCA
    n_components = 24
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(z_mean)

    principal_df = pd.DataFrame(
        data=principal_components,
        columns=['PC%d' % (num + 1) for num in range(n_components)]
    )
    print('PCA is done', flush=True)

    # plot explained variance by principal component analysis
    exp_var_pca = pca.explained_variance_ratio_
    plt.bar(range(1, len(exp_var_pca) + 1), exp_var_pca, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_directory, 'Explained_ratio.png'), dpi=300)
    plt.show()
    plt.close()

    # save eigenvector
    for idx, eigenvector in enumerate(pca.components_):
        np.save(os.path.join(model_save_directory, f'PC_mean_eigenvector_{idx + 1}.npy'), eigenvector)
        if idx == 2:
            break

    # save mean
    np.save(os.path.join(model_save_directory, 'z_mean.npy'), z_mean)
    np.save(os.path.join(model_save_directory, 'property_z_mean.npy'), property_value)

    # plot
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(principal_df['PC1'], principal_df['PC2'], c=property_value, cmap='rainbow', s=2.0, alpha=1.0, label='OMG Polymers')
    plt.colorbar(orientation='horizontal').set_label(label='LogP', size=10)
    plt.title('PCA of the latent space of MoleculeChef', fontsize=10)
    plt.xlabel('PC1', fontsize=10)
    plt.ylabel('PC2', fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(model_save_directory, f'latent_space_1_2.png'))
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(principal_df['PC1'], principal_df['PC3'], c=property_value, cmap='rainbow', s=2.0, alpha=1.0, label='OMG Polymers')
    plt.colorbar(orientation='horizontal').set_label(label='LogP', size=10)
    plt.title('PCA of the latent space of MoleculeChef', fontsize=10)
    plt.xlabel('PC1', fontsize=10)
    plt.ylabel('PC3', fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(model_save_directory, f'latent_space_1_3.png'))
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(principal_df['PC2'], principal_df['PC3'], c=property_value, cmap='rainbow', s=2.0, alpha=1.0, label='OMG Polymers')
    plt.colorbar(orientation='horizontal').set_label(label='LogP', size=10)
    plt.title('PCA of the latent space of MoleculeChef', fontsize=10)
    plt.xlabel('PC2', fontsize=10)
    plt.ylabel('PC3', fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(model_save_directory, f'latent_space_2_3.png'))
    plt.show()
    plt.close()
