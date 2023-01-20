from abc import abstractmethod
from typing import NamedTuple, List
from .utils import TorchDetails

import torch.nn as nn

APPENDER_TO_HIDDEN_NAMES = '_bond_proj_'  # to avoid name clashes with other attributes


class GGNNParams(NamedTuple):
    h_layer_size: int
    edge_names: List[str]  # string of the nodes which are associated with different relationships
    num_layers: int  # number of the time steps to do message passing. often denoted as T in papers
    torch_details: TorchDetails


class GGNNBase(nn.Module):
    """
    Gated Graph Neural Network (node features)
    Li, Y., Tarlow, D., Brockschmidt, M. and Zemel, R., 2015. Gated graph sequence neural networks.
    arXiv preprint arXiv:1511.05493.
    see also
    Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O. and Dahl, G.E., 2017.
    Neural message passing for quantum chemistry. arXiv preprint arXiv:1704.01212.
    """

    def __init__(self, params: GGNNParams):
        super(GGNNBase, self).__init__()
        self.params = params
        self.GRU_hidden = nn.GRUCell(
            input_size=self.params.h_layer_size,
            hidden_size=self.params.h_layer_size,
            bias=True
        )
        self.A_hidden = nn.ModuleDict(
            {name + APPENDER_TO_HIDDEN_NAMES: nn.Linear(
                self.params.h_layer_size, self.params.h_layer_size) for name in self.params.edge_names}
        )

    def get_edge_names_and_projections(self):
        return ((key[:-len(APPENDER_TO_HIDDEN_NAMES)], value) for key, value in self.A_hidden.items())

    @ abstractmethod
    def forward(self, *args):
        raise NotImplementedError


class GraphFeaturesFromStackedNodeFeaturesBase(nn.Module):
    """
    Attention weighted sum using the computed features.
    The trickiness in performing these operations is that we need to do a sum over nodes. For different graphs we
    have different numbers of nodes and so batching is difficult. The children of this class try doing this in different
    ways.
    Base class for modules that take in the stacked node feature matrix [v*, h] and produce embeddings of graphs
     [g, h']. These are called aggregation functions by Johnson (2017).
    Johnson DD (2017) Learning Graphical State Transitions. In: ICLR, 2017.
    Li Y, Vinyals O, Dyer C, et al. (2018) Learning Deep Generative Models of Graphs.
    arXiv [cs.LG]. Available at: http://arxiv.org/abs/1803.03324.
    """
    def __init__(self, mlp_project_up, mlp_gate, mlp_func, torch_details):
        super().__init__()
        self.mlp_project_up = mlp_project_up  # net that goes from [None, h'] to [None, j] with j>h usually
        self.mlp_gate = mlp_gate  # net that goes from [None, h'] to [None, 1 or h']
        self.mlp_func = mlp_func  # net that goes from [None, j] to [None, q]
        self.cuda_details = torch_details
