import os
import sys
import math
import numpy as np
import pandas as pd
import time

import selfies as sf

import torch
import torch.nn as nn
import torch.nn.functional as f

from torch.distributions.multivariate_normal import MultivariateNormal

from rdkit import Chem
from rdkit.Chem.rdchem import RWMol, BondType, Atom
from rdkit.Chem.rdmolops import GetShortestPath
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append('/home/sk77/PycharmProjects/publish/OMG')

from polymerization import Polymerization

from molecule_chef.mchef.molecule_chef import MoleculeChef
from molecule_chef.module.ggnn_base import GGNNParams
from molecule_chef.module.gated_graph_neural_network import GraphFeaturesStackIndexAdd
from molecule_chef.module.encoder import Encoder
from molecule_chef.module.decoder import Decoder
from molecule_chef.module.gated_graph_neural_network import GGNNSparse
from molecule_chef.module.utils import TorchDetails, save_model, MChefParameters
from molecule_chef.module.utils import FullyConnectedNeuralNetwork, PropertyNetworkPredictionModule
from molecule_chef.module.preprocess import AtomFeatureParams


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

    # terminal diene
    # if '==' in new_polymer_smi:
    #     target_idx = new_polymer_smi.find('==')
    #     new_polymer_smi = new_polymer_smi[:target_idx] + new_polymer_smi[target_idx + 1:]

    return Chem.CanonSmiles(new_polymer_smi)


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

    # mchef load parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_directory = '/home/sk77/PycharmProjects/publish/OMG/train/all/mchef_optuna_10000_6000_3000_500_message_4_objective_1_100_10'
    model_save_directory = os.path.join(load_directory, 'divergence_weight_1.677_latent_dim_38_learning_rate_0.001')

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
    torch_details = TorchDetails(device=device, data_type=mchef_parameters.dtype)
    dtype = mchef_parameters.dtype

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

    # load stop embedding
    stop_embedding = state_dict['nn_stop_embedding']

    # load property prediction network
    property_network = PropertyNetworkPredictionModule(
        latent_dim=mchef_parameters.latent_dim,
        property_dim=mchef_parameters.property_dim,
        property_network_hidden_dim_list=mchef_parameters.property_network_hidden_sizes,
        dtype=mchef_parameters.dtype,
        device=device,
        weights=mchef_parameters.property_weights
    ).to(device)
    property_network.load_state_dict(state_dict['nn_property_network'])

    # instantiate molecule chef class
    molecule_chef = MoleculeChef(
        graph_neural_network=graph_neural_network,
        encoder=encoder,
        decoder=decoder,
        property_network=property_network,
        stop_embedding=stop_embedding,
        torch_details=torch_details
    )

    # prior generation
    number_of_generation = 30000
    molecule_chef.mchef_module.eval()
    pbar = tqdm(range(number_of_generation), total=number_of_generation, leave=True)
    time.sleep(1.0)

    # store bag
    generated_monomer_bag_list = list()
    generated_monomer_bag_list_before_polymerization = list()

    multivariate_normal = MultivariateNormal(
        torch.zeros(mchef_parameters.latent_dim, dtype=dtype, device=device),
        torch.eye(mchef_parameters.latent_dim, dtype=dtype, device=device)
    )

    # get graph embeddings of whole monomer bags
    unique_monomer_sets = torch.load(os.path.join(load_directory, 'unique_monomer_sets.pth'), map_location=device)
    train_monomer_bags = torch.load(os.path.join(load_directory, 'train_bags.pth'), map_location=device)
    all_monomers_graph_embedding_tensor = state_dict['all_monomers_graph_embedding_tensor'].to(device)

    with pbar as t:
        for iteration in t:
            with torch.no_grad():
                # gaussian prior generation
                random_prior = multivariate_normal.sample().unsqueeze(0)

                # decode
                _, decoded_dict = molecule_chef.mchef_module.decoder(
                    z_samples=random_prior, all_monomer_tensors=all_monomers_graph_embedding_tensor,
                    answer_dict=None, monomer_bag_idx=range(random_prior.shape[0]), teacher_forcing=False,
                    generate=True
                )

                # get decoded results
                decoder_max_steps = mchef_parameters.decoder_max_steps
                decoded_result = torch.empty(size=(decoder_max_steps,
                                                   decoded_dict[0].shape[0]), dtype=dtype, device=device)

                for step_num in range(decoder_max_steps):
                    decoded_result[step_num, :] = decoded_dict[step_num].clone()
                decoded_result = torch.transpose(decoded_result, dim0=0, dim1=1)

                # get reactants
                stop_embedding_idx = len(unique_monomer_sets)

                for decoded_monomer_indices in decoded_result:
                    monomer_idx_list = list()
                    for decoded_monomer_idx in decoded_monomer_indices:
                        if not decoded_monomer_idx >= stop_embedding_idx:
                            monomer_idx_list.append(decoded_monomer_idx.to(torch.long).cpu().numpy().item())
                        else:
                            break

                    # get reactants
                    # if len(monomer_idx_list) >= 3:  # exclude three reactants
                    #     continue

                    monomer_bag = set()
                    for idx in monomer_idx_list:
                        decoded_monomer = unique_monomer_sets[idx]
                        monomer_bag.add(decoded_monomer)
                    monomer_bag = frozenset(monomer_bag)
                    generated_monomer_bag_list.append(monomer_bag)
                    generated_monomer_bag_list_before_polymerization.append(monomer_bag)

    # 1) check validity - valid after polymerization
    # polymerization
    reactor = Polymerization()
    generated_synthesizable_monomer_bag_list = list()
    wrong_list = list()

    for bag_set in generated_monomer_bag_list:
        if len(bag_set) >= 3:
            wrong_list.append(bag_set)
            continue
        result = reactor.polymerize(list(bag_set))
        if result is None:
            if len(bag_set) == 2:  # check novel one-reactant polymerization
                result_1 = reactor.polymerize([list(bag_set)[0]])
                result_2 = reactor.polymerize([list(bag_set)[1]])
                # compare polymerization mechanism
                if result_1 is not None and result_2 is not None and result_1[1] == result_2[1]:
                    generated_synthesizable_monomer_bag_list.append(bag_set)
            wrong_list.append(bag_set)
        else:
            generated_synthesizable_monomer_bag_list.append(bag_set)

    # save wrong list
    torch.save(wrong_list,
               os.path.join(model_save_directory, f'wrong_gaussian_prior_generated_monomer_bags_{number_of_generation}_generations.pth'))

    number_of_valid_generated_polymers = len(generated_synthesizable_monomer_bag_list)

    print(f"Gaussian prior generated valid {len(generated_monomer_bag_list_before_polymerization)} polymers "
          f"{100 * (len(generated_monomer_bag_list_before_polymerization)/number_of_generation):.3f}% "
          f"before polymerization", flush=True)

    print(f"Gaussian prior generated valid {number_of_valid_generated_polymers} polymers "
          f"{100 * (number_of_valid_generated_polymers/number_of_generation):.3f}% "
          f"after polymerization", flush=True)

    print('====================', flush=True)

    # 2) check uniqueness - uniqueness after polymerization
    # random search - drop duplicates
    generated_synthesizable_unique_monomer_bag_set = set(generated_synthesizable_monomer_bag_list)
    generated_synthesizable_unique_monomer_bag_list = list(generated_synthesizable_unique_monomer_bag_set)

    generated_unique_monomber_bag_list_before_polymerization = list(set(generated_monomer_bag_list_before_polymerization))

    print(f"Gaussian prior generated unique {len(generated_unique_monomber_bag_list_before_polymerization)} polymers "
          f"{100 * (len(generated_unique_monomber_bag_list_before_polymerization)/len(generated_monomer_bag_list_before_polymerization)):.3f}% "
          f"among valid polymers before polymerization", flush=True)

    print(f"Gaussian prior generated unique {len(generated_synthesizable_unique_monomer_bag_list)} polymers "
          f"{100 * (len(generated_synthesizable_unique_monomer_bag_list)/number_of_valid_generated_polymers):.3f}% "
          f"among valid polymers after polymerization", flush=True)

    print('====================', flush=True)

    # 3) check novelty - novelty among valid polymers
    # compare with train_monomer_bags
    train_monomer_bags_set = set(frozenset(monomer_bag) for monomer_bag in train_monomer_bags)

    novel_unique_generated_monomer_bag_list = list(set(generated_synthesizable_unique_monomer_bag_list) - train_monomer_bags_set)
    novel_unique_generated_monomer_bag_list_before_polymerization = list(set(generated_unique_monomber_bag_list_before_polymerization) - train_monomer_bags_set)

    print(f"Gaussian prior generated novel, unique {len(novel_unique_generated_monomer_bag_list_before_polymerization)} polymers "
          f"{100 * (len(novel_unique_generated_monomer_bag_list_before_polymerization)/len(generated_monomer_bag_list_before_polymerization)):.3f}% "
          f"among valid polymers before polymerization", flush=True)

    print(f"Gaussian prior generated novel, unique {len(novel_unique_generated_monomer_bag_list)} polymers "
          f"{100 * (len(novel_unique_generated_monomer_bag_list)/number_of_valid_generated_polymers):.3f}% "
          f"among valid polymers after polymerization", flush=True)

    print('====================', flush=True)

    # # save polymerization mechanism
    # generated_synthesizable_polymer_list = list()
    # generated_synthesizable_polymer_mechanism_list = list()
    #
    # for bag_set in novel_unique_generated_monomer_bag_list:
    #     result = reactor.polymerize(list(bag_set))
    #     if result is None:
    #         if len(bag_set) == 2:  # check novel one-reactant polymerization
    #             result_1 = reactor.polymerize([list(bag_set)[0]])
    #             result_2 = reactor.polymerize([list(bag_set)[1]])
    #             # compare polymerization mechanism
    #             if result_1 is not None and result_2 is not None and result_1[1] == result_2[1]:
    #                 # combine two polymers
    #                 combined_polymer = combine_polymer(result_1[0], result_2[0])
    #                 generated_synthesizable_polymer_list.append(combined_polymer)
    #                 generated_synthesizable_polymer_mechanism_list.append(result_1[1])
    #         else:
    #             ValueError("Error!")
    #
    #     else:
    #         # canonical form
    #         p_mol = Chem.MolFromSmiles(result[0])
    #         p_smi = Chem.MolToSmiles(p_mol)
    #         generated_synthesizable_polymer_list.append(p_smi)
    #         generated_synthesizable_polymer_mechanism_list.append(result[1])
    #
    # # reaxys condition search
    # df_omg = pd.DataFrame(data=None, columns=['reactant_1', 'reactant_2', 'product', 'mechanism', 'pair'])
    #
    # # append
    # for bag_set, polymer, mechanism in zip(novel_unique_generated_monomer_bag_list, generated_synthesizable_polymer_list, generated_synthesizable_polymer_mechanism_list):
    #     bag = list(bag_set)
    #     if len(bag) == 2:
    #         df_sub = pd.DataFrame({'reactant_1': [bag[0]], 'reactant_2': [bag[1]], 'product': [polymer],
    #                                'mechanism': [mechanism], 'pair': [1]})
    #     else:
    #         df_sub = pd.DataFrame({'reactant_1': [bag[0]], 'reactant_2': [bag[0]], 'product': [polymer],
    #                                'mechanism': [mechanism], 'pair': [0]})
    #     # concat
    #     df_omg = pd.concat([df_omg, df_sub], axis=0)
    #
    # # search!
    # df_omg = df_omg.reset_index(drop=True)
    # df_omg_reaxys = search_reaxys_condition(df_omg)
    # df_omg_reaxys = df_omg_reaxys[df_omg_reaxys['harmonic_score_1'] >= 0.5]
    # print(df_omg_reaxys.shape)
    # df_omg_reaxys.to_csv(os.path.join(model_save_directory, f'./mchef_{number_of_generation}_prior_generation_reaxys.csv'), index=False)

    # save
    # torch.save(novel_unique_generated_monomer_bag_list,
    #            os.path.join(model_save_directory, f'gaussian_prior_generated_monomer_bags_{number_of_generation}_generations.pth'))
    # torch.save(generated_synthesizable_polymer_list,
    #            os.path.join(model_save_directory, f'gaussian_prior_generated_polymers_{number_of_generation}_generations.pth'))
    # torch.save(generated_synthesizable_polymer_mechanism_list,
    #            os.path.join(model_save_directory, f'gaussian_prior_generated_polymers_mechanism_{number_of_generation}_generations.pth'))
