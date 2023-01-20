from abc import ABC

import torch
import torch.nn as nn


class Encoder(nn.Module, ABC):
    def __init__(self, in_dimension, layer_1d, latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(Encoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.output_dimension = latent_dimension * 2

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, self.output_dimension),
        )

    @staticmethod
    def reparameterize(mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # Get results of encoder network
        h1 = self.encode_nn(x)

        mu = h1[..., :self.latent_dimension]
        log_var = h1[..., self.latent_dimension:]

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
