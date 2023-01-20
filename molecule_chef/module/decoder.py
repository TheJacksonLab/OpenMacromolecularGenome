from abc import ABC
from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as f

from torch.distributions.gumbel import Gumbel

from molecule_chef.module.utils import FullyConnectedNeuralNetwork


class Decoder(nn.Module, ABC):
    def __init__(
            self,
            number_of_layers,
            max_steps,
            graph_embedding_dim,
            latent_dimension,
            gru_neural_net_hidden_dim: List[int],
            torch_details
    ):
        super(Decoder, self).__init__()
        self.number_of_layers = number_of_layers
        self.max_steps = max_steps
        self.latent_dimension = latent_dimension
        self.graph_embedding_dim = graph_embedding_dim
        self.gru_neural_net = FullyConnectedNeuralNetwork(
            input_dim=self.graph_embedding_dim,
            hidden_sizes=gru_neural_net_hidden_dim,
            output_dim=self.graph_embedding_dim
        )
        self.torch_details = torch_details  # contain GPU information and data type

        # convert the dimension: latent dimension -> graph embedding dimension
        self.linear_projection_z_to_hidden = nn.Linear(
            in_features=self.latent_dimension, out_features=self.graph_embedding_dim, bias=True
        )
        self.gru = nn.GRU(
            input_size=self.graph_embedding_dim,
            hidden_size=self.graph_embedding_dim,
            num_layers=number_of_layers,
            bias=True
        )

    def forward(self, z_samples, all_monomer_tensors, answer_dict, monomer_bag_idx, teacher_forcing=True,
                generate=False):
        # initial hidden shape: [b, graph_embedding_dim]
        hidden = self.linear_projection_z_to_hidden(z_samples)

        # initial message shape: [b, graph_embedding_dim]
        message = torch.zeros(
            size=hidden.shape, dtype=self.torch_details.data_type, device=self.torch_details.device
        )
        # updated massage shape: [b, graph_embedding_dim]
        updated_message = torch.zeros(
            size=hidden.shape, dtype=self.torch_details.data_type, device=self.torch_details.device
        )
        # change hidden shape
        hidden_in = hidden.unsqueeze(0).repeat(self.number_of_layers, 1, 1)  # [2, b, graph_embedding (out_size)]

        # transpose
        all_monomer_tensors_transposed = all_monomer_tensors.transpose(dim0=0, dim1=1)

        # set reconstruction loss
        recon_loss = torch.zeros(size=(1,), dtype=self.torch_details.data_type, device=self.torch_details.device)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # decoded molecules
        decoded_idx = dict()

        # set stop_flag. If stop embedding appears, stop contributing to the reconstruction error from the next decoding
        stop_flag = torch.zeros(size=(z_samples.shape[0],), dtype=torch.bool, device=self.torch_details.device)
        # stop_embedding_idx = all_monomer_tensors.shape[0] - 1  # idx = length - 1

        # get length tensor
        length = torch.zeros(size=(z_samples.shape[0],), dtype=torch.long, device=self.torch_details.device)
        if not generate:
            for idx, bag_idx in enumerate(monomer_bag_idx):
                length[idx] = len(answer_dict[bag_idx])
        else:
            for idx, bag_idx in enumerate(monomer_bag_idx):
                length[idx] = self.max_steps

        # run RNN - calculate hidden tensors
        for step in range(self.max_steps):
            message_in = message.unsqueeze(0)  # [1, b, graph_embedding (input_size)]
            output, hidden_in = self.gru(input=message_in, hx=hidden_in)  # [1, b, graph_embedding (out_size)]
            output_mapped = self.gru_neural_net(output.squeeze(0))  # [b, graph_embedding]

            # memory
            # output_mapped = output_mapped.to(torch.device('cuda:1'))

            # print(output_mapped.device, flush=True)
            # print(all_monomer_tensors_transposed.device, flush=True)

            # calculate dot product with all monomers embedding
            # output_mapped: [b, graph_embedding_dim]. total_embeddings_transpose: [graph_embedding_dim, total_num]
            dot_product = torch.matmul(output_mapped, all_monomer_tensors_transposed)  # shape: [b, total_num]

            # memory
            # dot_product = dot_product.to(self.torch_details.device)

            # set Gumbel noise: refer to https://arxiv.org/pdf/1611.01144.pdf
            # gumbel_dist = Gumbel(0, 1)
            # gumbel_noise = gumbel_dist.sample(sample_shape=dot_product.shape).to(self.torch_details.device)
            # decoded_idx[step] = torch.argmax(f.softmax(dot_product + gumbel_noise, dim=-1), dim=-1).detach()
            decoded_idx[step] = torch.argmax(dot_product, dim=-1).detach()

            # construct target and update message
            target = torch.zeros(
                size=(dot_product.shape[0],), dtype=torch.long, device=self.torch_details.device
            )
            if not generate:
                for idx, bag_idx in enumerate(monomer_bag_idx):
                    idx_step = step
                    # if step is larger than the length of the bag, take it to HALT
                    if idx_step >= len(answer_dict[bag_idx]):
                        idx_step = len(answer_dict[bag_idx]) - 1
                    target[idx] = deepcopy(answer_dict[bag_idx][idx_step])

                    if teacher_forcing:  # train (teacher forcing)
                        # update message vector (teacher forcing)
                        updated_message[idx] = all_monomer_tensors[answer_dict[bag_idx][idx_step]].clone()
                    else:  # test (not teacher forcing)
                        updated_message[idx] = all_monomer_tensors[decoded_idx[step][idx]].clone()
            else:
                for idx, bag_idx in enumerate(monomer_bag_idx):
                    updated_message[idx] = all_monomer_tensors[decoded_idx[step][idx]].clone()

            # change message
            message = updated_message.clone()

            # calculate cross entropy loss
            recon_loss_step = criterion(dot_product, target)[~stop_flag]
            if len(recon_loss_step) == 0:  # if there is no components in recon_loss_step
                recon_loss_step = 0
            else:
                recon_loss_step = recon_loss_step.sum()
            recon_loss += recon_loss_step

            # update stop_flag
            stop_flag[length == step + 1] = True
            # stop_flag[decoded_idx[step] == stop_embedding_idx] = True

        return recon_loss / length.shape[0], decoded_idx
