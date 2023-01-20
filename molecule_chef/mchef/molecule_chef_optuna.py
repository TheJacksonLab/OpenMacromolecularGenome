import os
import time
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

from torch.optim.lr_scheduler import ExponentialLR

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from molecule_chef.mchef.base import BaseMoleculeChef
from molecule_chef.module.utils import TorchDetails, get_accuracy, estimate_maximum_mean_discrepancy
from molecule_chef.module.encoder import Encoder
from molecule_chef.module.decoder import Decoder


class EarlyStopping(object):
    def __init__(self, tolerance=5, min_delta=1e-5):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.previous_validation_loss = 1e31
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss > (self.previous_validation_loss + self.min_delta):
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.tolerance:
            self.early_stop = True

        # change
        self.previous_validation_loss = validation_loss


class MoleculeChefLearningModule(nn.Module, ABC):
    def __init__(self, graph_neural_network, encoder, decoder, property_network, stop_embedding):
        super(MoleculeChefLearningModule, self).__init__()
        self.graph_neural_network = graph_neural_network
        self.encoder = encoder
        self.decoder = decoder
        self.property_network = property_network
        self.stop_embedding = stop_embedding

    def forward(self):
        pass


class MoleculeChef(BaseMoleculeChef):
    def __init__(
            self,
            graph_neural_network,
            encoder: Encoder,
            decoder: Decoder,
            property_network,
            stop_embedding,
            torch_details=TorchDetails(device='cpu', data_type=torch.float32)
    ):
        super(MoleculeChef, self).__init__(torch_details=torch_details)

        # combine module
        self.mchef_module = MoleculeChefLearningModule(
            graph_neural_network=graph_neural_network,
            encoder=encoder,
            decoder=decoder,
            property_network=property_network,
            stop_embedding=stop_embedding
        )
        # set property scaler
        self.scaler = StandardScaler()

        # save info
        self.optimizer = None
        self.lr_scheduler = None
        self.save_directory = None

        # collect loss
        self.divergence_loss = list()
        self.reconstruction_loss = list()
        self.property_loss = list()
        self.train_reconstruction_loss = list()
        self.train_divergence_loss = list()
        self.teacher_forcing_reconstruction_loss = list()

        # batch dependent attributes
        self.monomer_bags_graph_embedding_dict = OrderedDict()
        self.all_monomers_graph_embedding_tensor = None
        self.monomer_bags_graph_embedding_tensor = None

    @staticmethod
    def get_answer_dict(monomer_bags, unique_monomer_sets):
        # get answer dictionary
        answer_dict = dict()
        for idx_of_bag in range(len(monomer_bags)):
            idx_list = list()
            for monomer in monomer_bags[idx_of_bag]:
                idx = np.where(unique_monomer_sets == monomer)[0].item()  # np.where returns nd.arr tuple
                idx_list.append(idx)
            idx_list.append(len(unique_monomer_sets))
            answer_dict[idx_of_bag] = deepcopy(idx_list)
        return answer_dict

    def get_graph_embeddings_all_monomers(self, graph_adj_list):
        # get graph embeddings
        graph_embedding_tensor = self.mchef_module.graph_neural_network(graph_adj_list)
        # add stop embedding
        self.all_monomers_graph_embedding_tensor = torch.cat(
            (graph_embedding_tensor, self.mchef_module.stop_embedding.unsqueeze(0)), dim=0
        )

    def get_graph_embeddings_monomer_bags_dict(self, monomer_bags_idx, answer_dict):
        # get monomer bags embeddings
        for idx_of_bag in monomer_bags_idx:
            self.monomer_bags_graph_embedding_dict[idx_of_bag] = 0  # initialize
            for monomer_idx in answer_dict[idx_of_bag][:-1]:  # exclude stop embeddings
                self.monomer_bags_graph_embedding_dict[idx_of_bag] += self.all_monomers_graph_embedding_tensor[
                    monomer_idx]

    def get_graph_embeddings_monomer_bags_tensor(self):
        temp = list(self.monomer_bags_graph_embedding_dict.values())[0].view(1, -1)
        for key in list(self.monomer_bags_graph_embedding_dict.keys())[1:]:
            result = torch.cat(
                [temp, self.monomer_bags_graph_embedding_dict[key].view(1, -1)], dim=0
            )
            temp = result
        self.monomer_bags_graph_embedding_tensor = temp

    def train(self,
              train_monomer_bags: list or tuple or np.ndarray, test_monomer_bags: list or tuple or np.ndarray,
              train_property_data: list or tuple or np.ndarray, test_property_data: list or tuple or np.ndarray,
              unique_mols, unique_monomer_sets, num_epochs, batch_size: int, save_directory: str, lr=0.001,
              weight_decay=1e-4, mmd_weight=10.0):
        # get graph data
        graph_adj_list = self.get_atomic_feature_vectors_and_adjacency_list(unique_mols)
        # self.get_graph_embeddings_all_monomers(graph_adj_list=graph_adj_list)  # memory issue 

        # get answer dict
        train_answer_dict = self.get_answer_dict(train_monomer_bags, unique_monomer_sets)
        test_answer_dict = self.get_answer_dict(test_monomer_bags, unique_monomer_sets)

        # scale
        self.scaler.fit(train_property_data)
        scaled_train_property = self.scaler.transform(train_property_data)
        scaled_test_property = self.scaler.transform(test_property_data)

        # set save directory
        self.save_directory = save_directory
        if not os.path.exists(os.path.join(os.getcwd(), self.save_directory)):
            Path(os.path.join(os.getcwd(), self.save_directory)).mkdir(parents=True)

        # set optimizer
        self.optimizer = torch.optim.Adam(
            params=self.mchef_module.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # set learning rate scheduler
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.1)

        # early stopping
        early_stopping = EarlyStopping(tolerance=5, min_delta=1e-10)

        # training
        num_batches_train = int(math.ceil(len(train_monomer_bags) / batch_size))
        for epoch in range(num_epochs):
            train_reconstruction_loss = torch.zeros(size=(1,), dtype=self.dtype, device=self.device)
            train_divergence_loss = torch.zeros(size=(1,), dtype=self.dtype, device=self.device)
            tqdm.write("===== EPOCH %d =====" % (epoch + 1))
            train_monomer_bags_idx = np.random.permutation(range(len(train_answer_dict)))  # shuffle monomer bags
            start = time.time()  # start time

            # train mode
            self.mchef_module.train()

            # batch training
            tqdm.write("Training ...")
            pbar = tqdm(range(num_batches_train), total=num_batches_train, leave=True)
            time.sleep(1.0)
            with pbar as t:
                for batch_idx in t:
                    # divide batch data
                    start_idx = batch_idx * batch_size
                    stop_idx = (batch_idx + 1) * batch_size
                    if batch_idx == num_batches_train - 1:
                        stop_idx = len(train_monomer_bags)
                    batch_monomer_bags_idx = train_monomer_bags_idx[start_idx: stop_idx]

                    # clear monomer bags graph embeddings dictionary
                    self.monomer_bags_graph_embedding_dict.clear()

                    # get answer dict and graph embeddings
                    self.get_graph_embeddings_all_monomers(graph_adj_list=graph_adj_list)
                    self.get_graph_embeddings_monomer_bags_dict(
                        monomer_bags_idx=batch_monomer_bags_idx, answer_dict=train_answer_dict)
                    self.get_graph_embeddings_monomer_bags_tensor()

                    # encode
                    z_samples, mus, log_vars = self.mchef_module.encoder(self.monomer_bags_graph_embedding_tensor)

                    # get maximum mean discrepancy loss
                    divergence_loss = estimate_maximum_mean_discrepancy(
                        posterior_z_samples=z_samples
                    )
                    weighted_divergence_loss = mmd_weight * divergence_loss

                    # get property loss
                    batch_train_property_tensor = torch.tensor(
                        scaled_train_property[batch_monomer_bags_idx], dtype=self.dtype, device=self.device
                    )
                    prediction = self.mchef_module.property_network(z_samples)
                    criteria = nn.MSELoss(reduction='none')

                    # property_loss = 50.0 * criteria(input=prediction, target=batch_train_property_tensor)
                    property_loss = criteria(input=prediction, target=batch_train_property_tensor)

                    # weight property loss (elementwise multiplication)
                    property_loss = torch.sum(
                        torch.mean(property_loss, dim=0) * self.mchef_module.property_network.normalized_weights_tensor,
                        dim=0
                    )

                    # decode (training - teacher forcing)
                    reconstruction_loss, decoded_idx = self.mchef_module.decoder(
                        z_samples=z_samples, all_monomer_tensors=self.all_monomers_graph_embedding_tensor,
                        answer_dict=train_answer_dict, monomer_bag_idx=batch_monomer_bags_idx, teacher_forcing=True
                    )
                    # back propagate
                    self.optimizer.zero_grad()

                    # get gradients
                    loss = weighted_divergence_loss + reconstruction_loss + property_loss
                    # loss = divergence_loss + reconstruction_loss
                    loss.backward()

                    # combine reconstruction loss
                    train_reconstruction_loss += float(reconstruction_loss)
                    train_divergence_loss += float(divergence_loss)

                    # updates
                    self.optimizer.step()

                    # update progress bar
                    t.set_postfix(reconstruction_loss='%.2f' % float(reconstruction_loss))
                    t.update()
                t.close()

            # append train_reconstruction_loss
            self.train_reconstruction_loss.append((train_reconstruction_loss / num_batches_train).cpu().numpy())
            self.train_divergence_loss.append((train_divergence_loss / num_batches_train).cpu().numpy())

            tqdm.write("Test ...")
            # test
            with torch.no_grad():
                # change mode
                self.mchef_module.eval()

                # clear answer dict and graph embeddings
                self.monomer_bags_graph_embedding_dict.clear()

                # get answer dict and graph embeddings
                self.get_graph_embeddings_all_monomers(graph_adj_list=graph_adj_list)
                self.get_graph_embeddings_monomer_bags_dict(
                    monomer_bags_idx=range(len(test_monomer_bags)), answer_dict=test_answer_dict
                )
                self.get_graph_embeddings_monomer_bags_tensor()

                # encode
                z_samples, mus, log_vars = self.mchef_module.encoder(self.monomer_bags_graph_embedding_tensor)

                # get maximum mean discrepancy loss
                divergence_loss = estimate_maximum_mean_discrepancy(
                    posterior_z_samples=z_samples
                )

                # get property loss
                criteria = nn.MSELoss(reduction='none')
                test_property_tensor = torch.tensor(scaled_test_property, dtype=self.dtype, device=self.device)
                prediction = self.mchef_module.property_network(z_samples)
                # property_loss = 50. * criteria(input=prediction, target=test_property_tensor)
                property_loss = criteria(input=prediction, target=test_property_tensor)

                # weight property loss (elementwise multiplication)
                property_loss = torch.sum(
                    torch.mean(property_loss, dim=0) * self.mchef_module.property_network.normalized_weights_tensor,
                    dim=0
                )

                # decode (no teacher forcing)
                reconstruction_loss, decoded_dict = self.mchef_module.decoder(
                    z_samples=z_samples, all_monomer_tensors=self.all_monomers_graph_embedding_tensor,
                    answer_dict=test_answer_dict, monomer_bag_idx=range(len(test_monomer_bags)),
                    teacher_forcing=False
                )
                # decode (teacher forcing)
                teacher_forcing_reconstruction_loss, _ = self.mchef_module.decoder(
                    z_samples=z_samples, all_monomer_tensors=self.all_monomers_graph_embedding_tensor,
                    answer_dict=test_answer_dict, monomer_bag_idx=range(len(test_monomer_bags)),
                    teacher_forcing=True
                )

                # loss append
                self.divergence_loss.append(float(divergence_loss))
                self.reconstruction_loss.append(float(reconstruction_loss))
                self.teacher_forcing_reconstruction_loss.append(float(teacher_forcing_reconstruction_loss))
                self.property_loss.append(float(property_loss))

                # get loss and accuracy
                element_accuracy, bag_accuracy = get_accuracy(
                    decoded_dict=decoded_dict, answer_dict=test_answer_dict, device=self.device,
                    dtype=self.dtype
                )
                tqdm.write("Divergence loss is %.3f" % divergence_loss)
                tqdm.write("Train Divergence loss is %.3f" % self.train_divergence_loss[-1])
                tqdm.write("Reconstruction loss (no teacher forcing) is %.3f" % reconstruction_loss)
                tqdm.write("Reconstruction loss (teacher forcing) is %.3f" % teacher_forcing_reconstruction_loss)
                tqdm.write("Train reconstruction loss (teacher forcing) is %.3f" % self.train_reconstruction_loss[-1])
                tqdm.write("Property loss is %.3f" % property_loss)
                tqdm.write("Decoded molecule element score is %.3f" % element_accuracy)
                tqdm.write("Decoded molecule bag score is %.3f" % bag_accuracy)

                # check NaN loss
                return_loss = [self.reconstruction_loss[-1], self.divergence_loss[-1], self.property_loss[-1]]
                flag = 0
                for loss in return_loss:
                    if math.isnan(loss):
                        flag = 1

                if flag:
                    return np.Inf, np.Inf, np.Inf

                # early stopping
                early_stopping(validation_loss=self.reconstruction_loss[-1])

                if early_stopping.early_stop:
                    tqdm.write(f'===== Early Stopping Is Executed At Epoch {epoch + 1} =====')
                    return self.reconstruction_loss[-1], self.divergence_loss[-1], self.property_loss[-1]

                # step learning rate scheduler
                # if epoch % 40 == 0 and epoch / 40 > 0.9:
                #     self.lr_scheduler.step()
                #     print(self.optimizer, flush=True)

            # end time
            end = time.time()
            tqdm.write("EPOCH %d takes %.3f minutes" % ((epoch + 1), (end - start) / 60))

        # return loss
        with torch.no_grad():
            print("Plot property prediction ...", flush=True)
            # eval mode
            self.mchef_module.eval()

            # clear answer dict and graph embeddings
            self.monomer_bags_graph_embedding_dict.clear()

            # get answer dict and graph embeddings - train tensor values
            self.get_graph_embeddings_all_monomers(graph_adj_list=graph_adj_list)
            self.get_graph_embeddings_monomer_bags_dict(
                monomer_bags_idx=range(len(train_monomer_bags)), answer_dict=train_answer_dict
            )
            self.get_graph_embeddings_monomer_bags_tensor()

            # encode
            z_samples, mus, log_vars = self.mchef_module.encoder(self.monomer_bags_graph_embedding_tensor)

            # get property loss (inverse transform)
            train_prediction = self.scaler.inverse_transform(
                self.mchef_module.property_network(z_samples).cpu().numpy())
            # train_prediction = self.mchef_module.property_network(z_samples).cpu().numpy()

            # get answer dict and graph embeddings - test tensor values
            self.monomer_bags_graph_embedding_dict.clear()
            self.get_graph_embeddings_monomer_bags_dict(
                monomer_bags_idx=range(len(test_monomer_bags)), answer_dict=test_answer_dict
            )
            self.get_graph_embeddings_monomer_bags_tensor()

            # encode
            z_samples, mus, log_vars = self.mchef_module.encoder(self.monomer_bags_graph_embedding_tensor)

            # get property loss (inverse transform)
            test_prediction = self.scaler.inverse_transform(
                self.mchef_module.property_network(z_samples).cpu().numpy())
            # test_prediction = self.mchef_module.property_network(z_samples).cpu().numpy()

            # check NaN loss
            return_loss = [self.reconstruction_loss[-1], self.divergence_loss[-1], self.property_loss[-1]]
            flag = 0
            for loss in return_loss:
                if math.isnan(loss):
                    flag = 1

            if flag:
                return np.Inf, np.Inf, np.Inf

            # get r2 score & plot
            property_name = ['LogP', 'SC_score']
            for idx in range(self.mchef_module.property_network.property_dim):
                train_r2_score = r2_score(y_true=train_property_data[:, idx], y_pred=train_prediction[:, idx])
                test_r2_score = r2_score(y_true=test_property_data[:, idx], y_pred=test_prediction[:, idx])

                plt.figure(figsize=(6, 6), dpi=300)
                plt.plot(train_property_data[:, idx], train_prediction[:, idx], 'bo', label='Train R2 score: %.3f' % train_r2_score)
                plt.plot(test_property_data[:, idx], test_prediction[:, idx], 'ro', label='Validation R2 score: %.3f' % test_r2_score)

                plt.legend()
                plt.xlabel('True')
                plt.ylabel('Prediction')

                plt.savefig(os.path.join(self.save_directory, "property_prediction_%s.png" % property_name[idx]))
                plt.show()
                plt.close()

        return self.reconstruction_loss[-1], self.divergence_loss[-1], self.property_loss[-1]

    def plot_learning_curve(self):
        # set parameters
        title = 'molecule_chef'
        epoch = len(self.divergence_loss)

        # plot
        plt.figure(figsize=(8, 8), dpi=300)
        plt.plot(range(1, epoch + 1), self.divergence_loss, 'r-', label='Test Divergence Loss')  # actually valid
        plt.plot(range(1, epoch + 1), self.train_divergence_loss, 'b-', label='Train Divergence Loss')
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("%s Learning Curve" % title, fontsize=12)
        plt.legend()
        plt.savefig(os.path.join(self.save_directory, "divergence_loss.png"))
        plt.show()
        plt.close()

        plt.figure(figsize=(8, 8), dpi=300)
        plt.plot(range(1, epoch + 1), self.reconstruction_loss, 'r-',
                 label='Test Reconstruction Loss (No teacher forcing)')
        plt.plot(range(1, epoch + 1), self.teacher_forcing_reconstruction_loss, 'g-',
                 label='Test Reconstruction Loss (Teacher forcing)')
        plt.plot(range(1, epoch + 1), self.train_reconstruction_loss, 'b-', label='Train Reconstruction Loss')
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("%s Learning Curve" % title, fontsize=12)
        plt.legend()
        plt.savefig(os.path.join(self.save_directory, "reconstruction_loss.png"))
        plt.show()
        plt.close()
