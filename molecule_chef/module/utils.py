import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as f

from abc import ABC
from pathlib import Path
from rdkit import Chem
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TORCH_FLT = torch.float32
NP_LONG = np.int64


class TorchDetails(object):
    def __init__(self, device, data_type):
        self.device = device
        self.data_type = data_type

    @ property
    def device_str(self):
        return self.device


class FullyConnectedNeuralNetwork(nn.Module, ABC):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int]):
        super(FullyConnectedNeuralNetwork, self).__init__()
        self.layer_sizes = [input_dim] + hidden_sizes + [output_dim]
        self.layer_dims = zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        self.linear_layers = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for input_dim, output_dim in self.layer_dims
        ])
        # self.batch_norm_layers = nn.ModuleList([
        #     nn.BatchNorm1d(dim) for dim in hidden_sizes
        # ])

    def forward(self, input_tensor: torch.Tensor):
        hidden_tensor = input_tensor
        for layer_num, layer in enumerate(self.linear_layers):
            hidden_tensor = layer(hidden_tensor)
            if layer_num < len(self.linear_layers) - 1:
                # hidden_tensor = self.batch_norm_layers[layer_num](hidden_tensor)
                hidden_tensor = f.relu(hidden_tensor)

        return hidden_tensor


class GraphAsAdjList(object):
    # from Molecule Chef's code
    def __init__(self, atomic_feature_vectors, edge_type_to_adjacency_list_map, node_to_graph_id):
        self.atomic_feature_vectors = atomic_feature_vectors
        self.edge_type_to_adjacency_list_map = edge_type_to_adjacency_list_map
        self.node_to_graph_id = node_to_graph_id
        self.max_num_graphs = self.node_to_graph_id.max() + 1  # plus one to deal with fact that index from zero.
        self.edge_type_to_adjacency_list_bi_directed_map = dict()
        # get bi_directed_map
        for key, value in self.edge_type_to_adjacency_list_map.items():
            if value.shape[0] == 0:
                self.edge_type_to_adjacency_list_bi_directed_map[key] = None
            else:
                self.edge_type_to_adjacency_list_bi_directed_map[key] = torch.cat((value, torch.flip(value, [0, 1])), dim=1)


class MChefParameters(object):
    def __init__(self, h_layer_size, ggnn_num_layers, graph_embedding_dim, latent_dim, encoder_layer_1d_dim,
                 decoder_num_of_layers, decoder_max_steps, property_dim, property_weights, dtype, device,
                 decoder_neural_net_hidden_dim: list, property_network_hidden_sizes: list):

        self.h_layer_size = h_layer_size
        self.ggnn_num_layers = ggnn_num_layers
        self.graph_embedding_dim = graph_embedding_dim
        self.latent_dim = latent_dim
        self.encoder_layer_1d_dim = encoder_layer_1d_dim
        self.decoder_num_of_layers = decoder_num_of_layers
        self.decoder_max_steps = decoder_max_steps
        self.property_dim = property_dim
        self.property_weights = property_weights
        self.decoder_neural_net_hidden_dim = decoder_neural_net_hidden_dim
        self.property_network_hidden_sizes = property_network_hidden_sizes
        self.dtype = dtype
        self.device = device


class PropertyNetworkPredictionModule(nn.Module):
    def __init__(self, latent_dim, property_dim, property_network_hidden_dim_list, dtype, device, weights=(0.5, 0.5)):
        super(PropertyNetworkPredictionModule, self).__init__()
        if len(property_network_hidden_dim_list) != property_dim:
            raise ValueError('Insert the property network structure corresponding to the property dimension')
        if len(weights) != property_dim:
            raise ValueError('Insert the weights corresponding to the property dimension')

        self.latent_dim = latent_dim
        self.property_dim = property_dim
        self.dtype = dtype
        self.device = device

        weights = torch.tensor(weights, dtype=dtype, device=device)
        self.normalized_weights_tensor = f.normalize(weights, p=1.0, dim=0)

        property_network_list = list()
        if property_dim >= 1:
            for dim_list in property_network_hidden_dim_list:
                property_network = FullyConnectedNeuralNetwork(
                    input_dim=latent_dim,
                    hidden_sizes=dim_list,
                    output_dim=1
                )
                property_network_list.append(property_network)

        else:
            raise ValueError("There is no property target")

        self.property_network_module = nn.ModuleList(property_network_list)

    def forward(self, latent_points):
        prediction = torch.zeros(size=(latent_points.shape[0], 1),
                                 device=latent_points.device, dtype=latent_points.dtype)

        for property_idx in range(self.property_dim):
            prediction_tensor = self.property_network_module[property_idx](latent_points)
            prediction = torch.cat([prediction, prediction_tensor], dim=-1)

        return prediction[:, 1:]


def construct_graph_dictionary(mol_list, ggnn, embedding_dim, torch_details):
    # flatten
    mol_list_np = np.array(mol_list).flatten()

    # construct graph dict
    graph_embeddings_dict = dict()
    for idx, mol in enumerate(mol_list_np):
        graph_embeddings_dict[idx] = ggnn(mol)

    # set HALT embeddings
    stop_embedding = nn.Parameter(torch.Tensor(embedding_dim).to(torch_details.data_type))
    bound = 1 / np.sqrt(embedding_dim)
    nn.init.uniform_(stop_embedding, -bound, bound)

    # add HALT embeddings to the graph_dict
    graph_embeddings_dict[mol_list_np.shape[0]] = stop_embedding

    return graph_embeddings_dict


def save_model(model, save_directory: str, name: str):
    save_directory = os.path.join(os.getcwd(), save_directory)
    if not os.path.exists(save_directory):
        Path(save_directory).mkdir(parents=True)
    torch.save(model, os.path.join(save_directory, name))


def get_accuracy(decoded_dict: dict, answer_dict: dict, dtype, device):
    max_step = len(decoded_dict)
    decoded_result = torch.empty(size=(max_step, decoded_dict[0].shape[0]), dtype=dtype, device=device).cpu().numpy()
    stop_embedding_idx = answer_dict[0][-1]
    # get decoded results
    for step in range(max_step):
        decoded_result[step, :] = decoded_dict[step].clone().cpu().numpy()
    decoded_result = np.transpose(decoded_result)

    # count correctly encoded monomer bags
    bag_count = 0
    element_count = 0
    total_element = 0
    for key in range(len(answer_dict)):
        length = len(answer_dict[key])
        answer = set(answer_dict[key])
        stop_bool = decoded_result[key] == stop_embedding_idx

        # find stop index
        stop_idx = max_step - 1
        for idx, value in enumerate(stop_bool):
            if value:
                stop_idx = idx
                break
        prediction = set(decoded_result[key][:stop_idx + 1])

        # count
        bag_count += int(answer == prediction)
        element_count += len(answer.intersection(prediction))
        total_element += length

    # get accuracy
    element_accuracy = element_count / total_element * 100
    bag_accuracy = bag_count / decoded_result.shape[0] * 100

    return element_accuracy, bag_accuracy


def get_correct_reactant_bags_batch(decoded_dict: dict, answer_dict: dict, dtype, device):
    max_step = len(decoded_dict)
    decoded_result = torch.empty(size=(max_step, decoded_dict[0].shape[0]), dtype=dtype, device=device).cpu().numpy()
    stop_embedding_idx = answer_dict[0][-1]
    # get decoded results
    for step in range(max_step):
        decoded_result[step, :] = decoded_dict[step].clone().cpu().numpy()
    decoded_result = np.transpose(decoded_result)

    # count correctly encoded monomer bags
    bag_count = 0
    for key in range(len(answer_dict)):
        answer = set(answer_dict[key])
        stop_bool = decoded_result[key] == stop_embedding_idx

        # find stop index
        stop_idx = max_step - 1
        for idx, value in enumerate(stop_bool):
            if value:
                stop_idx = idx
                break
        prediction = set(decoded_result[key][:stop_idx + 1])

        # count
        bag_count += int(answer == prediction)

    return bag_count


def get_correct_reactant_bags_batch_error_analysis(decoded_dict: dict, answer_dict: dict, dtype, device):
    max_step = len(decoded_dict)
    decoded_result = torch.empty(size=(max_step, decoded_dict[0].shape[0]), dtype=dtype, device=device).cpu().numpy()
    stop_embedding_idx = answer_dict[0][-1]
    # get decoded results
    for step in range(max_step):
        decoded_result[step, :] = decoded_dict[step].clone().cpu().numpy()
    decoded_result = np.transpose(decoded_result)

    # right bags
    right_idx = []

    # count correctly encoded monomer bags
    bag_count = 0
    for key in range(len(answer_dict)):
        answer = set(answer_dict[key])
        stop_bool = decoded_result[key] == stop_embedding_idx

        # find stop index
        stop_idx = max_step - 1
        for idx, value in enumerate(stop_bool):
            if value:
                stop_idx = idx
                break
        prediction = set(decoded_result[key][:stop_idx + 1])

        # count
        bag_count += int(answer == prediction)

        if answer == prediction:
            # append
            right_idx.append(key)

    return bag_count, right_idx


def get_accuracy_check(decoded_dict: dict, answer_dict: dict, dtype, device):
    max_step = len(decoded_dict)
    decoded_result = torch.empty(size=(max_step, decoded_dict[0].shape[0]), dtype=dtype, device=device).cpu().numpy()
    stop_embedding_idx = answer_dict[0][-1]
    # get decoded results
    for step in range(max_step):
        decoded_result[step, :] = decoded_dict[step].clone().cpu().numpy()
    decoded_result = np.transpose(decoded_result)

    # count correctly encoded monomer bags
    bag_count = 0
    element_count = 0
    total_element = 0
    for key in range(len(answer_dict)):
        length = len(answer_dict[key])
        answer = set(answer_dict[key])
        stop_bool = decoded_result[key] == stop_embedding_idx

        # find stop index
        stop_idx = max_step - 1
        for idx, value in enumerate(stop_bool):
            if value:
                stop_idx = idx
                break
        prediction = set(decoded_result[key][:stop_idx + 1])

        # count
        bag_count += int(answer == prediction)
        element_count += len(answer.intersection(prediction))
        total_element += length

    # get accuracy
    element_accuracy = element_count / total_element * 100
    bag_accuracy = bag_count / decoded_result.shape[0] * 100

    return element_accuracy, bag_accuracy


def get_maximum_mean_discrepancy(posterior_z_samples: torch.Tensor):
    """
    Detailed explanation is in: https://www.kaggle.com/onurtunali/maximum-mean-discrepancy
    Prior distribution is assumed to be a multi-variate Gaussian
    MMD is a symmetric distance between samples from two different probability distributions,
    i.e. MMD(P, Q) = MMD(Q, P)
    Also refer to Wasserstein Auto-Encoders by Tolstikhin et al (2017).
    """
    # set dtype and device
    dtype = posterior_z_samples.dtype
    device = posterior_z_samples.device

    # set dimension and number of samples
    latent_dimension = posterior_z_samples.shape[1]
    number_of_samples = posterior_z_samples.shape[0]

    # get prior samples
    gaussian_sampler = MultivariateNormal(
        loc=torch.zeros(latent_dimension, dtype=dtype, device=device),
        covariance_matrix=torch.eye(n=latent_dimension, dtype=dtype, device=device)
    )
    prior_z_samples = gaussian_sampler.sample(sample_shape=(number_of_samples,))

    # calculate Maximum Mean Discrepancy with inverse multi-quadratics kernel
    # set value of c - refer to Sec.4 of Wasserstein paper
    c = 2 * latent_dimension * (1.0**2)

    # calculate pp term (p means prior)
    pp = torch.mm(prior_z_samples, prior_z_samples.t())
    pp_diag = pp.diag().unsqueeze(0).expand_as(pp)
    kernel_pp = c / (c + pp_diag + pp_diag.t() - 2 * pp)
    kernel_pp = (torch.sum(kernel_pp) - number_of_samples) / (number_of_samples * (number_of_samples - 1))

    # calculate qq term (q means posterior)
    qq = torch.mm(posterior_z_samples, posterior_z_samples.t())
    qq_diag = qq.diag().unsqueeze(0).expand_as(qq)
    kernel_qq = c / (c + qq_diag + qq_diag.t() - 2 * qq)
    kernel_qq = (torch.sum(kernel_qq) - number_of_samples) / (number_of_samples * (number_of_samples - 1))

    # calculate pq term
    pq = pp_diag.t() - torch.mm(prior_z_samples, posterior_z_samples.t()) \
        - torch.mm(posterior_z_samples, prior_z_samples.t()) + qq_diag
    kernel_pq = -2 * torch.sum(c / (c + pq)) / number_of_samples**2

    mmd = kernel_pp + kernel_qq + kernel_pq

    return mmd


def estimate_maximum_mean_discrepancy(posterior_z_samples: torch.Tensor):
    # https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution
    # set dtype and device
    dtype = posterior_z_samples.dtype
    device = posterior_z_samples.device

    # set dimension and number of samples
    latent_dimension = posterior_z_samples.shape[1]
    number_of_samples = posterior_z_samples.shape[0]

    # get prior samples
    prior_z_samples = torch.randn(size=(number_of_samples, latent_dimension), device=device, dtype=dtype)
    # gaussian_sampler = MultivariateNormal(
    #     loc=torch.zeros(latent_dimension, dtype=dtype, device=device),
    #     covariance_matrix=torch.eye(n=latent_dimension, dtype=dtype, device=device)
    # )
    # prior_z_samples = gaussian_sampler.sample(sample_shape=(number_of_samples,))

    # calculate Maximum Mean Discrepancy with inverse multi-quadratics kernel
    # set value of c - refer to Sec.4 of Wasserstein paper
    c = 2 * latent_dimension * (1.0**2)

    # calculate pp term (p means prior)
    pp = torch.mm(prior_z_samples, prior_z_samples.t())
    pp_diag = pp.diag().unsqueeze(0).expand_as(pp)

    # calculate qq term (q means posterior)
    qq = torch.mm(posterior_z_samples, posterior_z_samples.t())
    qq_diag = qq.diag().unsqueeze(0).expand_as(qq)

    # calculate pq term (q means posterior)
    pq = torch.mm(prior_z_samples, posterior_z_samples.t())

    # calculate kernel
    kernel_pp = torch.mean(c / (c + pp_diag + pp_diag.t() - 2 * pp))
    kernel_qq = torch.mean(c / (c + qq_diag + qq_diag.t() - 2 * qq))
    kernel_pq = torch.mean(c / (c + qq_diag + pp_diag.t() - 2 * pq))

    # estimate mmd
    mmd = kernel_pp + kernel_qq - 2*kernel_pq

    return mmd


def preprocess_df(file):
    df = pd.read_csv(file)
    df['end_point'] = df['smiles'].apply(lambda x: x.find('.'))
    df = df[df['end_point'] == -1]
    df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df['num_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
    df = df[df['num_atoms'] <= 15]
    df = df[['smiles', 'num_atoms']].reset_index(drop=True)
    df = df.iloc[:500]
    df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
    df = df.drop(['num_atoms'], axis=1)

    return df


def plot_grad_flow(named_parameters, file_name):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('%s_gradient_flow.png' % file_name)


