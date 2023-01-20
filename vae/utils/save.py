import os
import pickle


def save_model(model, model_name, save_directory):
    file_path = os.path.join(save_directory, model_name) + '.pickle'
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model


class VAEParameters(object):
    def __init__(self, data_path: str, save_directory, nop_idx, asterisk_idx, latent_dimension, encoder_in_channels,
                 encoder_feature_dim, encoder_convolution_channel_dim: list, encoder_kernel_size: list,
                 encoder_layer_1d, encoder_layer_2d, decoder_input_dimension, decoder_output_dimension,
                 decoder_num_gru_layers, decoder_bidirectional: bool, property_dim,
                 property_network_hidden_dim_list: list, property_weights: tuple, dtype, device, test_size=0.1,
                 random_state=42):

        self.data_path = data_path
        self.save_directory = save_directory
        self.nop_idx = nop_idx
        self.asterisk_idx = asterisk_idx
        self.latent_dimension = latent_dimension
        self.encoder_in_channels = encoder_in_channels
        self.encoder_feature_dim = encoder_feature_dim
        self.encoder_convolution_channel_dim = encoder_convolution_channel_dim
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_layer_1d = encoder_layer_1d
        self.encoder_layer_2d = encoder_layer_2d
        self.decoder_input_dimension = decoder_input_dimension
        self.decoder_output_dimension = decoder_output_dimension
        self.decoder_num_gru_layers = decoder_num_gru_layers
        self.decoder_bidirectional = decoder_bidirectional
        self.property_dim = property_dim
        self.property_network_hidden_dim_list = property_network_hidden_dim_list
        self.property_weights = property_weights
        self.dtype = dtype
        self.device = device
        self.test_size = test_size
        self.random_state = random_state
