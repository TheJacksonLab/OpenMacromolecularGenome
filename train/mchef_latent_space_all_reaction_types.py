import os
import sys
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Draw

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


sys.path.append('/home/sk77/PycharmProjects/publish/OMG')

from polymerization import Polymerization

from molecule_chef.mchef.molecule_chef import MoleculeChef
from molecule_chef.module.ggnn_base import GGNNParams
from molecule_chef.module.gated_graph_neural_network import GraphFeaturesStackIndexAdd
from molecule_chef.module.encoder import Encoder
from molecule_chef.module.decoder import Decoder
from molecule_chef.module.gated_graph_neural_network import GGNNSparse
from molecule_chef.module.utils import TorchDetails, PropertyNetworkPredictionModule, FullyConnectedNeuralNetwork
from molecule_chef.module.utils import MChefParameters, get_correct_reactant_bags_batch
from molecule_chef.module.preprocess import AtomFeatureParams


def linear_interpolation(arr1, arr2, t):
    return (arr2 - arr1) * t + arr1


if __name__ == '__main__':
    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all'
    save_directory = os.path.join(load_directory, 'mchef_optuna_10000_6000_3000_500_message_4_objective_1_100_10')
    model_save_directory = os.path.join(save_directory, 'divergence_weight_1.677_latent_dim_38_learning_rate_0.001')

    # load data
    df = pd.read_csv(os.path.join(load_directory, 'all_reactions_10000_6000_3000_500.csv'))

    # load idx
    train_monomer_bags_idx = torch.load(os.path.join(load_directory, 'all_reactions_train_bag_idx_10000_6000_3000_500.pth'))
    temp_idx = torch.load(os.path.join(load_directory, 'all_reactions_test_bag_idx_10000_6000_3000_500.pth'))

    # train : valid : test split - 8 : 1 : 1
    temp_train_idx, test_monomer_bags_idx = train_test_split(temp_idx, test_size=int(df.shape[0] * (1 / 10)),
                                                             random_state=42)
    temp_train_idx, valid_monomer_bags_idx = train_test_split(temp_train_idx, test_size=int(df.shape[0] * (1 / 10)),
                                                              random_state=42)
    train_monomer_bags_idx += temp_train_idx

    # get train df
    df_train = df.iloc[train_monomer_bags_idx, :]
    df_train = df_train.reset_index(drop=True)

    # load MChef
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
    dtype = mchef_parameters.dtype
    torch_details = TorchDetails(device=device, data_type=dtype)

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

    # instantiate molecule chef class
    molecule_chef = MoleculeChef(
        graph_neural_network=graph_neural_network,
        encoder=encoder,
        decoder=decoder,
        property_network=property_network,
        stop_embedding=stop_embedding,
        torch_details=torch_details
    )

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

    print('Projection is done', flush=True)

    # get decoding point list
    decoding_point = list()
    principal_df_sorted = principal_df.sort_values(by=['PC1'], ascending=True)
    target_1_point = principal_df_sorted.iloc[0]
    #
    principal_df_sorted = principal_df.sort_values(by=['PC1'], ascending=False)
    target_2_point = principal_df_sorted.iloc[5]

    # principal_df_sorted = principal_df.sort_values(by=['property_1_z_mean'], ascending=True)  # minimum of property
    # target_1_point = principal_df_sorted.iloc[0]

    # principal_df_sorted = principal_df.sort_values(by=['property_1_z_mean'], ascending=False) # maximum of property
    # target_2_point = principal_df_sorted.iloc[0]

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

    with torch.no_grad():
        target_point_tensor = torch.tensor(
            np.array([linear_interpolation(target_1_z, target_2_z, t) for t in t_points]),
            dtype=dtype, device=device)

        # property prediction
        property_prediction = molecule_chef.mchef_module.property_network(target_point_tensor)

        # property scaler
        train_property_values = torch.load(os.path.join(save_directory, 'train_property.pth'))
        property_scaler = StandardScaler()
        property_scaler.fit(train_property_values)

        # inverse transform
        property_prediction_arr = property_scaler.inverse_transform(property_prediction.cpu().numpy())

        # decode
        all_monomers_graph_embedding_tensor = state_dict['all_monomers_graph_embedding_tensor'].to(device)
        _, decoded_dict = molecule_chef.mchef_module.decoder(
            z_samples=target_point_tensor, all_monomer_tensors=all_monomers_graph_embedding_tensor,
            answer_dict=None, monomer_bag_idx=range(len(target_point_tensor)), teacher_forcing=False, generate=True
        )
        # get decoded results
        decoder_max_steps = mchef_parameters.decoder_max_steps
        decoded_result = torch.empty(size=(decoder_max_steps,
                                           decoded_dict[0].shape[0]), dtype=dtype, device=device)

        for step_num in range(decoder_max_steps):
            decoded_result[step_num, :] = decoded_dict[step_num].clone()
        decoded_result = torch.transpose(decoded_result, dim0=0, dim1=1)

        # get reactants
        unique_monomer_sets = torch.load(os.path.join(save_directory, 'unique_monomer_sets.pth'), map_location=device)
        stop_embedding_idx = len(unique_monomer_sets)
        monomer_bag_list = list()
        for num, decoded_monomer_indices in enumerate(decoded_result):
            monomer_idx_list = list()
            for decoded_monomer_idx in decoded_monomer_indices:
                if not decoded_monomer_idx >= stop_embedding_idx:
                    monomer_idx_list.append(decoded_monomer_idx.to(torch.long).cpu().numpy().item())
                else:
                    break

            # get reactants
            monomer_bag = set()
            for idx in monomer_idx_list:
                decoded_monomer = unique_monomer_sets[idx]
                monomer_bag.add(decoded_monomer)
            monomer_bag_list.append(list(monomer_bag))

    # print
    print(property_prediction_arr)
    print(monomer_bag_list)

    # polymerization
    print('===== Polymerization =====', flush=True)
    reactor = Polymerization()
    for monomer_bag in monomer_bag_list:
        result = reactor.polymerize(monomer_bag)
        if result is None:
            print('None', flush=True)
        else:
            print(result, flush=True)
            print(monomer_bag, flush=True)

    # plot according to the reaction type
    plt.figure(figsize=(6, 6), dpi=300)
    df_train = df_train.reset_index(drop=True)

    # 1) step-growth polymerization
    df_step_growth = df_train[df_train['reaction_idx'] <= 7]
    principal_df_step_growth = principal_df.iloc[df_step_growth.index.to_list(), :]

    # 2) chain-growth addition
    df_chain_growth_addition = df_train[df_train['reaction_idx'].between(left=8, right=9, inclusive='both')]
    principal_df_chain_growth_addition = principal_df.iloc[df_chain_growth_addition.index.to_list(), :]

    # 3) chain-growth ring-opening
    df_chain_growth_ring_opening = df_train[df_train['reaction_idx'].between(left=10, right=15, inclusive='both')]
    principal_df_chain_growth_ring_opening = principal_df.iloc[df_chain_growth_ring_opening.index.to_list(), :]

    # 4) metathesis
    df_metathesis = df_train[df_train['reaction_idx'].between(left=16, right=17, inclusive='both')]
    principal_df_metathesis = principal_df.iloc[df_metathesis.index.to_list(), :]

    # construct dataframe for plot
    principal_df_step_growth['mechanism'] = ['Step Growth'] * principal_df_step_growth.shape[0]
    principal_df_chain_growth_addition['mechanism'] = ['Chain Growth'] * principal_df_chain_growth_addition.shape[0]
    principal_df_chain_growth_ring_opening['mechanism'] = ['Ring Opening'] * principal_df_chain_growth_ring_opening.shape[0]
    principal_df_metathesis['mechanism'] = ['Metathesis'] * principal_df_metathesis.shape[0]

    plot_df = pd.concat(
        [principal_df_step_growth, principal_df_chain_growth_addition, principal_df_chain_growth_ring_opening, principal_df_metathesis],
        ignore_index=True, axis=0
    )

    # plot_df = pd.concat(
    #     [principal_df_metathesis],
    #     ignore_index=True, axis=0
    # )

    # plot seaborn
    g = sns.jointplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="mechanism",
        kind='scatter',
        joint_kws=dict(alpha=0.5),
        hue_order=['Chain Growth', 'Metathesis', 'Ring Opening', 'Step Growth'],
        # hue_order=['Metathesis'],
        edgecolor=None,
        palette=['#32B565', '#7953A2', '#3C76BC', '#ED1C24'],
        # palette=['m'],
        s=20.0,
        marginal_kws={'common_norm': False}
    )
    g.ax_joint.tick_params(labelsize=16)
    g.ax_joint.legend(title=None, fontsize=10, loc='upper right')
    g.ax_joint.set_xticks([-6, -4, -2, 0, 2, 4, 6, 8])
    g.ax_joint.set_yticks([-6, -4, -2, 0, 2, 4, 6, 8])
    g.ax_joint.legend_.remove()
    # for lh in g.ax_joint.legend_.legendHandles:
    #     lh.set_alpha(0.5)
    #     lh.set_sizes([10])
    g.set_axis_labels('Principal component 1', 'Principal component 2', fontsize=16)
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(model_save_directory, "reaction_type.png"), dpi=800)
    exit()

    plt.plot(principal_df_step_growth['PC1'], principal_df_step_growth['PC2'], 'ro', label='Step Growth', alpha=0.5,
             markersize=2.0)

    plt.plot(principal_df_chain_growth_addition['PC1'], principal_df_chain_growth_addition['PC2'], 'go',
             label='Chain Growth Addition', alpha=0.5, markersize=2.0)
    plt.plot(principal_df_chain_growth_ring_opening['PC1'], principal_df_chain_growth_ring_opening['PC2'], 'bo',
             label='Ring Opening', alpha=0.5, markersize=2.0)
    plt.plot(principal_df_metathesis['PC1'], principal_df_metathesis['PC2'], 'mo', label='Metathesis',
             alpha=0.5, markersize=2.0)

    # plt.scatter(principal_df['PC1'], principal_df['PC2'], c=property_z_mean, cmap='rainbow', s=2.0, alpha=1.0)
    # plt.plot(interpolation_df['PC1'], interpolation_df['PC2'], 'k--', alpha=0.5)
    # plt.plot(interpolation_df['PC1'].iloc[1:-1], interpolation_df['PC2'].iloc[1:-1], 'k*', markersize=5.0, alpha=0.5,
    #          label='Interpolation')

    plt.title('PCA of the latent space of MoleculeChef', fontsize=10)
    # plt.xlim(-6.0, 13.0)
    # plt.ylim(-6.0, 6.0)
    plt.xlabel('PC1', fontsize=10)
    plt.ylabel('PC2', fontsize=10)
    # plt.colorbar(orientation='horizontal').set_label(label='LogP', size=10)
    plt.legend(fontsize=8, loc='upper left')
    plt.savefig(os.path.join(model_save_directory, f'latent_space_linear_reaction_type.png'))
    # plt.savefig(os.path.join(save_directory, f'latent_space_linear_reaction_type_step_growth.png'))
    # plt.savefig(os.path.join(save_directory, f'latent_space_linear_reaction_type_chain_growth_addition.png'))
    # plt.savefig(os.path.join(save_directory, f'latent_space_linear_reaction_type_ring_opening.png'))
    # plt.savefig(os.path.join(save_directory, f'latent_space_linear_reaction_type_metathesis.png'))
    plt.show()
    plt.close()
