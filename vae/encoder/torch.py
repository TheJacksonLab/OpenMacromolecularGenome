import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d,
                 latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(Encoder, self).__init__()
        self.latent_dimension = latent_dimension

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.BatchNorm1d(layer_1d),
            nn.LeakyReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.BatchNorm1d(layer_2d),
            nn.LeakyReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.BatchNorm1d(layer_3d),
            nn.LeakyReLU()
        )

        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)

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

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class CNNEncoder(nn.Module):

    def __init__(self, in_channels, feature_dim, convolution_channel_dim: list or tuple, kernel_size: list or tuple,
                 layer_1d, layer_2d, latent_dimension):
        """
        CNN encoder to encode molecule to latent space
        """
        super(CNNEncoder, self).__init__()
        if len(convolution_channel_dim) != len(kernel_size):
            ValueError("Convolution channel dim should have the same number as kernel size")

        self.latent_dimension = latent_dimension
        self.convolution_channel_dim = convolution_channel_dim
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        # construct Conv1D encoder
        conv_layers = list()
        for idx in range(len(convolution_channel_dim)):
            if idx == 0:
                conv_layers.append(
                    nn.Conv1d(in_channels=self.in_channels, out_channels=self.convolution_channel_dim[idx],
                              kernel_size=self.kernel_size[idx])
                )
            else:
                conv_layers.append(
                    nn.Conv1d(in_channels=self.convolution_channel_dim[idx - 1], out_channels=self.convolution_channel_dim[idx],
                              kernel_size=self.kernel_size[idx])
                )
            conv_layers.append(nn.BatchNorm1d(self.convolution_channel_dim[idx]))
            conv_layers.append(nn.LeakyReLU())
        self.conv_encode_nn = nn.Sequential(*conv_layers)

        # get FCNN in dimension
        self.fcnn_in_dim = feature_dim
        for dim in kernel_size:
            self.fcnn_in_dim -= (dim - 1)
        self.fcnn_in_dim *= self.convolution_channel_dim[-1]

        print(f'[VAE] CNN Encoder dim {self.fcnn_in_dim} to FCNN', flush=True)
        # construct FCNN encoder
        self.layer_1d = layer_1d
        self.layer_2d = layer_2d
        self.encode_linear = nn.Sequential(
            nn.Linear(self.fcnn_in_dim, self.layer_1d),
            nn.BatchNorm1d(self.layer_1d),
            nn.LeakyReLU(),
            nn.Linear(self.layer_1d, self.layer_2d),
            nn.BatchNorm1d(self.layer_2d),
            nn.LeakyReLU()
        )
        self.encode_mu = nn.Linear(self.layer_2d, self.latent_dimension)
        self.encode_log_var = nn.Linear(self.layer_2d, self.latent_dimension)

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
        h1 = self.conv_encode_nn(x)
        h1 = h1.view(h1.size(0), -1)

        # latent space
        h1 = self.encode_linear(h1)

        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class EncoderOptuna(nn.Module):
    def __init__(self, sequential, last_layer_dimension, latent_dimension):
        super(EncoderOptuna, self).__init__()
        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = sequential

        # Latent space mean
        self.encode_mu = nn.Linear(last_layer_dimension, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(last_layer_dimension, latent_dimension)

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # Get results of encoder network
        h1 = self.encode_nn(x)

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var