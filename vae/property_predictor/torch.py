import abc

import torch
import torch.nn as nn
import torch.nn.functional as f

from sklearn.preprocessing import StandardScaler

from typing import List


class PropertyPredictor(nn.Module, abc.ABC):
    def __init__(self, latent_dimension, layer_1d, layer_2d, property_dimension):
        super(PropertyPredictor, self).__init__()
        self.latent_dimension = latent_dimension
        self.property_dimension = property_dimension

        self.predict_nn = nn.Sequential(
            nn.Linear(in_features=latent_dimension, out_features=layer_1d),
            nn.BatchNorm1d(layer_1d),
            nn.LeakyReLU(),
            nn.Linear(in_features=layer_1d, out_features=layer_2d),
            nn.BatchNorm1d(layer_2d),
            nn.LeakyReLU(),
            nn.Linear(in_features=layer_2d, out_features=property_dimension)
        )

    def forward(self, x):
        targeted_property = self.predict_nn(x)
        return targeted_property


class PropertyPredictorOptuna(nn.Module, abc.ABC):
    def __init__(self, latent_dimension, sequential, property_dimension, y_scaler=StandardScaler()):
        super(PropertyPredictorOptuna, self).__init__()
        self.latent_dimension = latent_dimension
        self.property_dimension = property_dimension
        self.y_scaler = y_scaler
        self.predict_nn = sequential

    def forward(self, x):
        targeted_property = self.predict_nn(x)
        return targeted_property


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


class FullyConnectedNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int]):
        super(FullyConnectedNeuralNetwork, self).__init__()
        self.layer_sizes = [input_dim] + hidden_sizes + [output_dim]
        self.layer_dims = zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        self.linear_layers = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for input_dim, output_dim in self.layer_dims
        ])
        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in hidden_sizes
        ])

    def forward(self, input_tensor: torch.Tensor):
        hidden_tensor = input_tensor
        for layer_num, layer in enumerate(self.linear_layers):
            hidden_tensor = layer(hidden_tensor)
            if layer_num < len(self.linear_layers) - 1:
                hidden_tensor = self.batch_norm_layers[layer_num](hidden_tensor)
                hidden_tensor = f.leaky_relu(hidden_tensor)

        return hidden_tensor




