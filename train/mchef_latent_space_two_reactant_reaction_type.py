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

    load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/six/'
    save_directory = os.path.join(load_directory, 'mchef_50000_15000_6000_message_4_objective_1_100_10')
    model_save_directory = os.path.join(save_directory, 'divergence_weight_0.750_latent_dim_36_learning_rate_0.000')

    # load data
    df = pd.read_csv(os.path.join(load_directory, '50000_15000_6000.csv'))

    # load idx
    train_monomer_bags_idx = torch.load(os.path.join(load_directory, 'train_bag_idx_50000_15000_6000.pth'))
    temp_idx = torch.load(os.path.join(load_directory, 'test_bag_idx_50000_15000_6000.pth'))

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

    # step-growth polymerization
    df_step_growth_1 = df_train[df_train['reaction_idx'] == 1]
    principal_df_step_growth_1 = principal_df.iloc[df_step_growth_1.index.to_list(), :]

    df_step_growth_2 = df_train[df_train['reaction_idx'] == 2]
    principal_df_step_growth_2 = principal_df.iloc[df_step_growth_2.index.to_list(), :]

    df_step_growth_3 = df_train[df_train['reaction_idx'] == 3]
    principal_df_step_growth_3 = principal_df.iloc[df_step_growth_3.index.to_list(), :]

    df_step_growth_4 = df_train[df_train['reaction_idx'] == 4]
    principal_df_step_growth_4 = principal_df.iloc[df_step_growth_4.index.to_list(), :]

    df_step_growth_5 = df_train[df_train['reaction_idx'] == 5]
    principal_df_step_growth_5 = principal_df.iloc[df_step_growth_5.index.to_list(), :]

    df_step_growth_6 = df_train[df_train['reaction_idx'] == 6]
    principal_df_step_growth_6 = principal_df.iloc[df_step_growth_6.index.to_list(), :]

    # construct dataframe for plot
    principal_df_step_growth_1['mechanism'] = ['Amide (carboxylic acid + amine)'] * principal_df_step_growth_1.shape[0]
    principal_df_step_growth_2['mechanism'] = ['Amide (acid chloride + amine)'] * principal_df_step_growth_2.shape[0]
    principal_df_step_growth_3['mechanism'] = ['Ester (carboxylic acid + alcohol)'] * principal_df_step_growth_3.shape[0]
    principal_df_step_growth_4['mechanism'] = ['Ester (acid chloride + alcohol)'] * principal_df_step_growth_4.shape[0]
    principal_df_step_growth_5['mechanism'] = ['Urea (amine + isocyanate)'] * principal_df_step_growth_5.shape[0]
    principal_df_step_growth_6['mechanism'] = ['Urethane (alcohol + isocyanate)'] * principal_df_step_growth_6.shape[0]

    plot_df = pd.concat(
        [principal_df_step_growth_1, principal_df_step_growth_2, principal_df_step_growth_3, principal_df_step_growth_4,
         principal_df_step_growth_5, principal_df_step_growth_6],
        ignore_index=True, axis=0
    )

    # plot_df = pd.concat(
    #     [principal_df_step_growth_3, principal_df_step_growth_6, principal_df_step_growth_4, principal_df_step_growth_1,
    #      principal_df_step_growth_5, principal_df_step_growth_2],
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
        # hue_order=['Reaction 1', 'Reaction 2', 'Reaction 3', 'Reaction 4', 'Reaction 5', 'Reaction 6'],
        hue_order=['Ester (carboxylic acid + alcohol)', 'Ester (acid chloride + alcohol)', 'Urethane (alcohol + isocyanate)',
                   'Urea (amine + isocyanate)', 'Amide (acid chloride + amine)', 'Amide (carboxylic acid + amine)'],
        edgecolor=None,
        # palette=['b', 'm', 'y', 'c', 'g', 'r'],
        palette=['#F89A41', '#F2EE81', '#BCDA7F', '#5EBC59', '#4CC5D7', '#4766B0'],
        s=15.0,  # size
        marginal_kws={'common_norm': False}
    )
    g.ax_joint.tick_params(labelsize=16)
    # legend = g.ax_joint.legend(title=None, fontsize=8, loc='lower right')
    # legend.get_frame().set_edgecolor('k')
    # legend.get_frame().set_linewidth(1.0)
    # for lh in g.ax_joint.legend_.legendHandles:
    #     lh.set_alpha(1.0)
    #     lh.set_sizes([10])
    #     lh.set_edgecolor(None)
    g.ax_joint.legend_.remove()
    g.set_axis_labels('Principal component 1', 'Principal component 2', fontsize=16)

    #####
    # draw examples
    # principal_df_step_growth_4['mechanism'] = ['Ester (acid chloride + alcohol)'] * principal_df_step_growth_4.shape[0]
    principal_df_step_growth_4_examples_condition_1 = (principal_df_step_growth_4['PC1'] <= 3) & (principal_df_step_growth_4['PC2'] <= 3)
    principal_df_step_growth_4_examples_condition_2 = (principal_df_step_growth_4['PC1'] >= 2) & (principal_df_step_growth_4['PC2'] >= 2)
    principal_df_step_growth_4_examples = principal_df_step_growth_4[principal_df_step_growth_4_examples_condition_1 & principal_df_step_growth_4_examples_condition_2]
    print(principal_df_step_growth_4_examples.head(20).to_string())
    # sns.scatterplot(data=principal_df_step_growth_4_examples, x='PC1', y='PC2', palette='k', ax=g.ax_joint)
    #####

    g.fig.tight_layout()
    g.fig.savefig(os.path.join(model_save_directory, "figure_7.png"), dpi=800)
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
    plt.savefig(os.path.join(save_directory, f'latent_space_linear_reaction_type.png'))
    # plt.savefig(os.path.join(save_directory, f'latent_space_linear_reaction_type_step_growth.png'))
    # plt.savefig(os.path.join(save_directory, f'latent_space_linear_reaction_type_chain_growth_addition.png'))
    # plt.savefig(os.path.join(save_directory, f'latent_space_linear_reaction_type_ring_opening.png'))
    # plt.savefig(os.path.join(save_directory, f'latent_space_linear_reaction_type_metathesis.png'))
    plt.show()
    plt.close()
