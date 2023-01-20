import torch
import torch.nn as nn

from abc import ABC
from molecule_chef.module.ggnn_base import GGNNBase, APPENDER_TO_HIDDEN_NAMES, GraphFeaturesFromStackedNodeFeaturesBase
from molecule_chef.module.utils import GraphAsAdjList
from .utils import TORCH_FLT


class GGNNPad(GGNNBase, ABC):
    """
    Gated Graph Neural Network: takes in adjacency matrix and node features
    """
    def __init__(self, params, graph_feature):
        super(GGNNPad, self).__init__(params=params)
        # graph feature to calculate graph representation vector
        self.graph_feature = graph_feature

    def forward(
            self,
            atom_features: torch.FloatTensor,
            adjacency_matrix: torch.FloatTensor,
    ):
        """
        Computes node representation tensors
        b: batch size
        e: the number of edge types
        v: the number of nodes of adjacency matrix in one graph
        :param atom_features: initial node representation tensors. shape: [b, v, h]
        :param adjacency_matrix: adjacency matrix. shape: [b, v, v, e]
        :return: the computed features (node representation vectors) for each node. shape: [b, v, h]
        """

        batch_size, num_nodes, num_hidden = atom_features.shape
        hidden = atom_features.view(-1, num_hidden)  # hidden shape: [b*v, h]

        for time_step in range(self.params.num_layers):
            message = torch.zeros(
                size=(batch_size, num_nodes, self.params.h_layer_size),
                device=self.params.torch_details.device,
                dtype=TORCH_FLT
            )  # size [b, v, h]

            for e, edge_type in enumerate(self.params.edge_names):
                # A_hidden shape: [h, h]
                message_t = self.A_hidden[edge_type + APPENDER_TO_HIDDEN_NAMES](hidden)  # message_t shape: [b*v, h]
                # TODO shape check
                message_t = message_t.view(batch_size, num_nodes, num_hidden)  # message_t shape: [b, v, h]
                # perform batch matrix-matrix product : [b, v, v] @ [b, v, h] = [b, v, h]
                message = message + torch.bmm(adjacency_matrix[:, :, :, e], message_t)

            message_unrolled = message.view(-1, num_hidden)  # shape: [b*v, h]
            hidden = self.GRU_hidden(
                input=message_unrolled,
                hx=hidden
            )  # shape: [b*v, h]

        hidden = hidden.view(batch_size, num_nodes, num_hidden)  # [b, v, h] - computed node representation vectors

        graph_representation_vector = self.graph_feature(
            propagated_node_features=hidden,
            initial_node_features=atom_features
        )

        return graph_representation_vector


class GraphFeaturization(nn.Module, ABC):
    """
    This class runs on node representation vectors and computes graph level representation vectors
    """
    def __init__(self, neural_net_project, neural_net_gate):
        super(GraphFeaturization, self).__init__()
        self.neural_net_project = neural_net_project  # neural network which goes from [*, 2h] to [*, h]
        self.neural_net_gate = neural_net_gate  # neural network which goes from [*, h] to [*, 1]

    def forward(self, propagated_node_features, initial_node_features):
        """
        :param propagated_node_features: shape is [b, v, h]
        :param initial_node_features: shape is [b, v, h]
        :return: shape [b, q]
        """
        # set dimensions
        b, v, h = propagated_node_features.shape  # the same shape as initial features

        # concatenate two tensors. shape: [b, v, 2h]
        concatenated_feature_vectors = torch.cat((propagated_node_features, initial_node_features), dim=-1)

        proj = self.neural_net_project(concatenated_feature_vectors.view(-1, 2*h)).view(b, v, -1)  # [b, v, h']

        # calculate gate value
        gate_tensor = self.neural_net_gate(concatenated_feature_vectors.view(-1, 2*h)).view(b, v, -1)  # [b, v, h']
        gate = torch.sigmoid(gate_tensor)  # [b, v, h']

        gated_sum = torch.sum(proj * gate, dim=1)  # [b, h']
        result = torch.tanh(gated_sum)

        return result


class GGNNSparse(GGNNBase):
    # from molecule chef code
    def __init__(self, params, graph_feature):
        super(GGNNSparse, self).__init__(params=params)
        # graph feature to calculate graph representation vector
        self.graph_feature = graph_feature

    def forward(self, graphs: GraphAsAdjList):
        hidden = graphs.atomic_feature_vectors.to(device=self.params.torch_details.device)
        num_nodes = hidden.shape[0]

        for t in range(self.params.num_layers):
            message = torch.zeros(
                num_nodes, self.params.h_layer_size, device=self.params.torch_details.device,
                dtype=self.params.torch_details.data_type
            )  # [total number of atoms, h_layer_size]
            for edge_name, projection in self.get_edge_names_and_projections():
                adj_list = graphs.edge_type_to_adjacency_list_bi_directed_map[edge_name]  # get bond projection
                if adj_list is None:
                    continue  # no edges of this type

                # memory
                # adj_list = adj_list.to(self.params.torch_details.device)
                # print(adj_list.device, flush=True)

                # projection
                projected_feats = projection(hidden)  # [total number of atoms, h_layer_size]
                #Todo: potentially wasteful doing this projection on all nodes (ie many may not
                # be connected by all kinds of edge)
                message.index_add_(0, adj_list[0], projected_feats.index_select(0, adj_list[1]))

            hidden = self.GRU_hidden(message, hidden)

        # memory
        # graphs.node_to_graph_id = graphs.node_to_graph_id.to(self.params.torch_details.device)

        graph_embedding_tensor = self.graph_feature(hidden, graphs.node_to_graph_id)

        # memory
        # graphs.node_to_graph_id = graphs.node_to_graph_id.to(torch.device('cuda:1'))

        return graph_embedding_tensor


class GraphFeaturesStackIndexAdd(nn.Module):
    def __init__(self, neural_net_project, neural_net_gate, torch_details):
        super(GraphFeaturesStackIndexAdd, self).__init__()
        self.neural_net_project = neural_net_project
        self.neural_net_gate = neural_net_gate
        self.torch_details = torch_details
    """
    Do the sum by Pytorch's index_add method.
    """
    def forward(self, node_features, node_to_graph_id):
        """
        :param node_features: [v*, h]
        :param node_to_graph_id:  for each node index the graph it belongs to [v*]
        """
        proj_up = self.neural_net_project(node_features)  # [v*, j]
        gate_logit = self.neural_net_gate(node_features)  # [v*, _]
        gate = torch.sigmoid(gate_logit)  # [v*, _]
        gated_vals = gate * proj_up

        num_graphs = node_to_graph_id.max() + 1
        graph_sums = torch.zeros(
            num_graphs, gated_vals.shape[1], device=self.torch_details.device, dtype=self.torch_details.data_type
        )  # [g, j]
        graph_sums.index_add_(0, node_to_graph_id, gated_vals)

        return graph_sums
