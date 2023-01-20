import os
import sys
import torch

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import numpy as np
import pandas as pd
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import RWMol, BondType, Atom
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

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


def get_morgan_fingerprints(smi_list):
    fps_list = list()
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        fps = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fps_list.append(fps)

    return np.array(fps_list)


def bulk_tanimoto_similarity(fps_1, fps_list):
    tanimoto_similarity_list = list()
    for fps_2 in fps_list:
        and_count = np.bitwise_and(fps_1, fps_2).sum()
        or_count = np.bitwise_or(fps_1, fps_2).sum()
        tanimoto_similarity_list.append(and_count/or_count)
    return np.array(tanimoto_similarity_list)


def calculate_similarity(reaxys_fps, reactant_1, reactant_2):
    # calculate Tanimoto similarity with reactant_1
    reactant_1_fps = get_morgan_finger_prints(reactant_1)
    reactant_1_tanimoto_similarity = bulk_tanimoto_similarity(reactant_1_fps, reaxys_fps)
    largest_similarity_reactant_1 = max(reactant_1_tanimoto_similarity)

    # calculate Tanimoto similarity with reactant_2
    reactant_2_fps = get_morgan_finger_prints(reactant_2)
    reactant_2_tanimoto_similarity = bulk_tanimoto_similarity(reactant_2_fps, reaxys_fps)
    largest_similarity_reactant_2 = max(reactant_2_tanimoto_similarity)
    # most_similar_reactant_2 = reactant_2_tanimoto_similarity.argmax()

    return (largest_similarity_reactant_1 + largest_similarity_reactant_2) / 2


def linear_interpolation(arr1, arr2, t):
    return (arr2 - arr1) * t + arr1


def combine_polymer(p_smile_1, p_smile_2):
    mol_1 = Chem.MolFromSmiles(p_smile_1)
    mol_2 = Chem.MolFromSmiles(p_smile_2)
    mw_1 = RWMol(mol_1)
    mw_2 = RWMol(mol_2)

    # store
    asterisk_idx = list()
    mol_1_del_list = list()
    mol_2_del_list = list()

    # find asterisk idx
    for idx, atom in enumerate(mol_1.GetAtoms()):
        if atom.GetSymbol() == '*':
            asterisk_idx.append(idx)
    mol_1_del_list.append(asterisk_idx[1])
    mol_2_del_list.append(asterisk_idx[0])

    # modify index of monomer 2
    modified_mol_2_del_list = [idx + mol_1.GetNumAtoms() for idx in mol_2_del_list]

    # combine
    new_polymer = RWMol(Chem.CombineMols(mw_1, mw_2))
    new_polymer.AddBond(mol_1_del_list[0], modified_mol_2_del_list[0], BondType.SINGLE)

    # rearrange atom idx
    new_polymer_smi = Chem.MolToSmiles(new_polymer)
    asterisk_idx_smi = list()
    for idx, char in enumerate(new_polymer_smi):
        if char == '*':
            asterisk_idx_smi.append(idx)
    asterisk_idx_smi = asterisk_idx_smi[1:-1]
    new_polymer_smi = new_polymer_smi[:asterisk_idx_smi[0]] + new_polymer_smi[asterisk_idx_smi[1] + 1:]

    return Chem.CanonSmiles(new_polymer_smi)


def search_reaxys_condition(df_omg):
    # construct OMG reactant library
    omg_reactant_set = set()
    for reactant_1 in df_omg['reactant_1']:
        omg_reactant_set.add(reactant_1)
    for reactant_2 in df_omg['reactant_2']:
        omg_reactant_set.add(reactant_2)

    omg_reactant_dict_idx2react = dict()
    for idx, reactant in enumerate(omg_reactant_set):
        omg_reactant_dict_idx2react[idx] = reactant

    omg_reactant_dict_react2idx = dict()
    for key, value in omg_reactant_dict_idx2react.items():
        omg_reactant_dict_react2idx[value] = key

    # calculate all fingerprints of OMG reactants
    omg_reactant_fps = get_morgan_fingerprints(list(omg_reactant_set))
    print(omg_reactant_fps.shape)

    # construct reaxys reactant library
    df_reaxys = pd.read_csv(
        '/home/sk77/PycharmProjects/polymer/reaxys_parallel_version_3/reaxys_polymerization_filtered.csv'
    )

    # add Reaxys vocabulary
    reaxys_reactant_set = set()
    for reaction in df_reaxys['Reaction']:
        reactant_list = reaction.split('.')
        for reactant in reactant_list:
            reaxys_reactant_set.add(reactant)

    reaxys_reactant_dict_idx2react = dict()
    for idx, reactant in enumerate(reaxys_reactant_set):
        reaxys_reactant_dict_idx2react[idx] = reactant

    reaxys_reactant_dict_react2idx = dict()
    for key, value in reaxys_reactant_dict_idx2react.items():
        reaxys_reactant_dict_react2idx[value] = key

    # calculate all fingerprints of Reaxys reactants
    reaxys_reactant_fps = get_morgan_fingerprints(list(reaxys_reactant_set))

    # construct tanimoto similarity array - [omg, reaxys]
    tanimoto_array = np.zeros(shape=[omg_reactant_fps.shape[0], reaxys_reactant_fps.shape[0]])
    for row_idx, omg_fps in enumerate(omg_reactant_fps):
        for col_idx, reaxys_fps in enumerate(reaxys_reactant_fps):
            and_count = np.bitwise_and(omg_fps, reaxys_fps).sum()
            or_count = np.bitwise_or(omg_fps, reaxys_fps).sum()
            tanimoto_array[row_idx, col_idx] = and_count / or_count

    # construct similarity array - between OMG reaction i and Reaxys reaction j
    similarity_array = np.zeros(shape=(df_omg.shape[0], df_reaxys.shape[0], 2))  # 2 refers to the two reactants
    for omg_idx in range(df_omg.shape[0]):
        print(omg_idx, flush=True)
        omg_reaction = df_omg.iloc[omg_idx]
        # omg reactant idx
        omg_reactant_1 = omg_reaction['reactant_1']
        omg_reactant_1_idx = omg_reactant_dict_react2idx[omg_reactant_1]
        omg_reactant_2 = omg_reaction['reactant_2']
        omg_reactant_2_idx = omg_reactant_dict_react2idx[omg_reactant_2]

        # type of polymerization
        omg_polymerization_type = omg_reaction['pair']

        for reaxys_idx in range(df_reaxys.shape[0]):
            reaxys_reaction = df_reaxys.iloc[reaxys_idx]['Reaction']
            number_of_reactants = df_reaxys.iloc[reaxys_idx]['reactant_count']

            # initialize similarity
            omg_reactant_1_similarity = -1.0
            omg_reactant_2_similarity = -1.0

            # check the number of reactants
            if omg_polymerization_type == 1:  # two reactant reactions
                if number_of_reactants != 2:
                    omg_reactant_1_similarity = 0.0
                    omg_reactant_2_similarity = 0.0

                    # append to the similarity array
                    similarity_array[omg_idx, reaxys_idx, 0] = omg_reactant_1_similarity
                    similarity_array[omg_idx, reaxys_idx, 1] = omg_reactant_2_similarity
                    continue

            elif omg_polymerization_type == 0:  # one reactant reaction
                if number_of_reactants != 1:
                    omg_reactant_1_similarity = 0.0
                    omg_reactant_2_similarity = 0.0

                    # append to the similarity array
                    similarity_array[omg_idx, reaxys_idx, 0] = omg_reactant_1_similarity
                    similarity_array[omg_idx, reaxys_idx, 1] = omg_reactant_2_similarity
                    continue

            for reaxys_reactant in reaxys_reaction.split('.'):
                reaxys_reactant_idx = reaxys_reactant_dict_react2idx[reaxys_reactant]

                # update omg reactant 1 similarity
                similarity = tanimoto_array[omg_reactant_1_idx, reaxys_reactant_idx]
                if similarity > omg_reactant_1_similarity:
                    omg_reactant_1_similarity = similarity
                # update omg reactant 2 similarity
                similarity = tanimoto_array[omg_reactant_2_idx, reaxys_reactant_idx]
                if similarity > omg_reactant_2_similarity:
                    omg_reactant_2_similarity = similarity

            # append to the similarity array
            similarity_array[omg_idx, reaxys_idx, 0] = omg_reactant_1_similarity
            similarity_array[omg_idx, reaxys_idx, 1] = omg_reactant_2_similarity

    # add columns to df_omg
    df_omg_reaxys = df_omg.copy()
    df_omg_reaxys['arithmetic_score_1'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['arithmetic_score_2'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['arithmetic_score_3'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['arithmetic_score_4'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['arithmetic_score_5'] = np.zeros(df_omg_reaxys.shape[0])

    df_omg_reaxys['arithmetic_score_1_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['arithmetic_score_2_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['arithmetic_score_3_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['arithmetic_score_4_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['arithmetic_score_5_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])

    df_omg_reaxys['geometric_score_1'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['geometric_score_2'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['geometric_score_3'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['geometric_score_4'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['geometric_score_5'] = np.zeros(df_omg_reaxys.shape[0])

    df_omg_reaxys['geometric_score_1_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['geometric_score_2_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['geometric_score_3_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['geometric_score_4_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['geometric_score_5_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])

    df_omg_reaxys['harmonic_score_1'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['harmonic_score_2'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['harmonic_score_3'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['harmonic_score_4'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['harmonic_score_5'] = np.zeros(df_omg_reaxys.shape[0])

    df_omg_reaxys['harmonic_score_1_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['harmonic_score_2_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['harmonic_score_3_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['harmonic_score_4_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])
    df_omg_reaxys['harmonic_score_5_reaxys_idx'] = np.zeros(df_omg_reaxys.shape[0])

    # recommend reaxys reactions
    cnt = 0
    record_top = 5
    epsilon = 1e-10
    for omg_idx in range(df_omg_reaxys.shape[0]):
        df_recommend = pd.DataFrame(data=None, columns=['arithmetic_score', 'geometric_score', 'harmonic_score'])

        # list to store scores
        reaxys_arithmetic_score_list = list()
        reaxys_geometric_score_list = list()
        reaxys_harmonic_score_list = list()

        for reaxys_idx in range(df_reaxys.shape[0]):
            similarity_1 = similarity_array[omg_idx, reaxys_idx, 0]
            similarity_2 = similarity_array[omg_idx, reaxys_idx, 1]
            arithmetic_score = (similarity_1 + similarity_2) / 2
            geometric_score = np.sqrt(similarity_1 * similarity_2)
            harmonic_score = 2 / ((1 / (similarity_1 + epsilon)) + (1 / (similarity_2 + epsilon)))

            # append
            reaxys_arithmetic_score_list.append(arithmetic_score)
            reaxys_geometric_score_list.append(geometric_score)
            reaxys_harmonic_score_list.append(harmonic_score)

        # append
        df_recommend['arithmetic_score'] = reaxys_arithmetic_score_list
        df_recommend['geometric_score'] = reaxys_geometric_score_list
        df_recommend['harmonic_score'] = reaxys_harmonic_score_list

        # sort
        df_recommend_arithmetic = df_recommend.sort_values(by='arithmetic_score', ascending=False)
        df_recommend_geometric = df_recommend.sort_values(by='geometric_score', ascending=False)
        df_recommend_harmonic = df_recommend.sort_values(by='harmonic_score', ascending=False)

        # store to OMG dataframe
        for num in range(record_top):
            df_omg_reaxys.loc[omg_idx, f'arithmetic_score_{num + 1}'] = df_recommend_arithmetic['arithmetic_score'].iloc[num]
            df_omg_reaxys.loc[omg_idx, f'geometric_score_{num + 1}'] = df_recommend_arithmetic['geometric_score'].iloc[num]
            df_omg_reaxys.loc[omg_idx, f'harmonic_score_{num + 1}'] = df_recommend_arithmetic['harmonic_score'].iloc[num]

            df_omg_reaxys.loc[omg_idx, f'arithmetic_score_{num + 1}_reaxys_idx'] = int(df_recommend_arithmetic.index[num])
            df_omg_reaxys.loc[omg_idx, f'geometric_score_{num + 1}_reaxys_idx'] = int(df_recommend_geometric.index[num])
            df_omg_reaxys.loc[omg_idx, f'harmonic_score_{num + 1}_reaxys_idx'] = int(df_recommend_harmonic.index[num])

        cnt += 1
        print(f'{cnt} is done', flush=True)

    return df_omg_reaxys


if __name__ == '__main__':
    # set environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/six/mchef_50000_15000_6000_message_4_objective_1_100_10'
    model_save_directory = '/home/sk77/PycharmProjects/publish/OMG/train/six/mchef_50000_15000_6000_message_4_objective_1_100_10/divergence_weight_0.750_latent_dim_36_learning_rate_0.000'

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
    target_1_point = principal_df_sorted.iloc[60]

    principal_df_sorted = principal_df.sort_values(by=['PC1'], ascending=False)
    target_2_point = principal_df_sorted.iloc[40]

    # principal_df_sorted = principal_df.sort_values(by=['property_1_z_mean'], ascending=True)  # minimum of property
    # target_1_point = principal_df_sorted.iloc[15]
    # #
    # principal_df_sorted = principal_df.sort_values(by=['property_1_z_mean'], ascending=False)  # maximum of property
    # target_2_point = principal_df_sorted.iloc[5]

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
    df_omg = pd.DataFrame(data=None, columns=['result_number', 'reactant_1', 'reactant_2', 'product', 'pair'])

    # polymerization
    print('\n', flush=True)
    print('===== Polymerization =====', flush=True)
    reactor = Polymerization()
    for idx, monomer_bag in enumerate(monomer_bag_list):
        result = reactor.polymerize(monomer_bag)
        if result is None:  # no reaction
            if len(monomer_bag) == 2:  # check novel one-reactant polymerization
                result_1 = reactor.polymerize([monomer_bag[0]])
                result_2 = reactor.polymerize([monomer_bag[1]])
                # compare polymerization mechanism
                if result_1 is not None and result_2 is not None and result_1[1] == result_2[1]:
                    # combine two polymers
                    combined_polymer = combine_polymer(result_1[0], result_2[0])
                    print(f"===== Result {idx + 1} ======", flush=True)
                    print(monomer_bag[0], monomer_bag[1], result_1[1], flush=True)
                    print(combined_polymer, flush=True)
                    df_sub = pd.DataFrame({'result_number': [idx + 1], 'reactant_1': [monomer_bag[0]],
                                           'reactant_2': [monomer_bag[1]], 'product': [combined_polymer],
                                           'pair': [1]})
                    df_omg = pd.concat([df_omg, df_sub], axis=0)

                else:
                    print(f"===== Result {idx + 1} ======", flush=True)
                    print('None', flush=True)

            else:  # no novel one-reactant polymerization
                print(f"===== Result {idx + 1} ======", flush=True)
                print('None', flush=True)

        else:  # reacted
            print(f"===== Result {idx + 1} ======", flush=True)
            print(monomer_bag, result[1], flush=True)
            print(result[0], flush=True)
            if len(monomer_bag) == 2:
                df_sub = pd.DataFrame({'result_number': [idx + 1], 'reactant_1': [monomer_bag[0]],
                                       'reactant_2': [monomer_bag[1]], 'product': [result[0]],
                                       'pair': [1]})
            else:
                df_sub = pd.DataFrame({'result_number': [idx + 1], 'reactant_1': [monomer_bag[0]],
                                       'reactant_2': [monomer_bag[0]], 'product': [result[0]],
                                       'pair': [0]})
            df_omg = pd.concat([df_omg, df_sub], axis=0)

    # search!
    df_omg = df_omg.reset_index(drop=True)
    df_omg_reaxys = search_reaxys_condition(df_omg)
    df_omg_reaxys = df_omg_reaxys[df_omg_reaxys['harmonic_score_1'] >= 0.5]
    print(df_omg_reaxys.head().to_string())

    # seaborn
    # seaborn JointGrid
    g = sns.JointGrid(x='PC1', y='PC2', data=principal_df)

    cmap = LinearSegmentedColormap.from_list("", ["#2F65ED", "#F5EF7E", "#F89A3F"])

    # scatter plot
    sns.scatterplot(x='PC1', y='PC2', hue='property_1_z_mean', data=principal_df, palette=cmap, ax=g.ax_joint,
                    edgecolor=None, alpha=0.75, legend=False, size=3.0)
    sns.scatterplot(x='PC1', y='PC2', data=interpolation_df, color='k', ax=g.ax_joint,
                    edgecolor=None, alpha=0.75, legend=False, size=1000.0, marker='*')
    sns.lineplot(x='PC1', y='PC2', data=interpolation_df, ax=g.ax_joint, linestyle='--', color='k', alpha=0.75)

    # kde plots on the marginal axes
    sns.kdeplot(x='PC1', data=principal_df, ax=g.ax_marg_x, shade=True, color='c')
    sns.kdeplot(y='PC2', data=principal_df, ax=g.ax_marg_y, shade=True, color='c')

    # tick parameters
    g.ax_joint.tick_params(labelsize=16)
    g.ax_joint.set_xticks([-6, -3, 0, 3, 6, 9])
    g.ax_joint.set_yticks([-6, -4, -2, 0, 2, 4, 6, 8])
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

    g.fig.savefig(os.path.join(model_save_directory, "seaborn_linear_interpolation.png"), dpi=800)

    # # plot
    # plt.figure(figsize=(6, 6), dpi=300)
    # plt.scatter(principal_df['PC1'], principal_df['PC2'], c=property_z_mean, cmap=cmap, s=2.0, alpha=1.0)
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
