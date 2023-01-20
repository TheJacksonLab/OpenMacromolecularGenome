import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import selfies as sf

from math import ceil
from tqdm import tqdm

from torch.nn import functional as f
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from matplotlib.lines import Line2D
from rdkit.Chem import MolFromSmiles
from sklearn.metrics import r2_score
from vae.utils.save import save_model


def plot_grad_flow(named_parameters, file_path, file_name):
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
    # plt.ylim(bottom=-0.001)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(os.path.join(file_path, '%s_gradient_flow.png' % file_name))


def train_model(vae_encoder, vae_decoder, property_predictor, nop_idx, asterisk_idx, data_train, data_valid, y_train,
                y_valid, y_scaler, num_epochs, batch_size, lr_property, lr_enc, lr_dec, kld_alpha, save_directory,
                dtype, device, mmd_weight=10.0):
    """
    Train the Variational Auto-Encoder
    """
    print('num_epochs: ', num_epochs, flush=True)

    # set optimizer
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc, weight_decay=1e-5)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec, weight_decay=1e-5)
    optimizer_property_predictor = torch.optim.Adam(property_predictor.parameters(), lr=lr_property, weight_decay=1e-5)

    # learning rate scheduler
    step_size = 40
    gamma = 0.1
    encoder_lr_scheduler = ExponentialLR(optimizer=optimizer_encoder, gamma=gamma)
    decoder_lr_scheduler = ExponentialLR(optimizer=optimizer_decoder, gamma=gamma)
    property_predictor_lr_scheduler = ExponentialLR(optimizer=optimizer_property_predictor, gamma=gamma)

    # encoder_lr_scheduler = StepLR(optimizer=optimizer_encoder, step_size=step_size, gamma=gamma)
    # decoder_lr_scheduler = StepLR(optimizer=optimizer_decoder, step_size=step_size, gamma=gamma)

    # train and valid data
    num_batches_train = ceil(len(data_train) / batch_size)
    num_batches_valid = ceil(len(data_valid) / batch_size)

    data_train_reconstruction_loss_list_teacher_forcing, data_train_divergence_loss_list_teacher_forcing, data_train_property_loss_list_teacher_forcing = [], [], []
    # data_train_reconstruction_loss_list_no_teacher_forcing, data_train_divergence_loss_list_no_teacher_forcing = [], []
    data_valid_reconstruction_loss_list_no_teacher_forcing, data_valid_divergence_loss_list_no_teacher_forcing, data_valid_property_loss_list_no_teacher_forcing= [], [], []

    # training
    for epoch in range(num_epochs):
        start = time.time()
        train_reconstruction_loss = torch.zeros(size=(1,), dtype=dtype).cpu()
        train_divergence_loss = torch.zeros(size=(1,), dtype=dtype).cpu()
        train_property_loss = torch.zeros(size=(1,), dtype=dtype).cpu()
        tqdm.write("==== Epoch %d Training" % (epoch + 1))

        # change mode
        vae_encoder.train()
        vae_decoder.train()
        property_predictor.train()

        # random permutation
        rand_idx = torch.randperm(data_train.size()[0])
        y_train_epoch = y_train.clone()[rand_idx]

        # mini-batch training
        tqdm.write("Training ...")
        pbar = tqdm(range(num_batches_train), total=num_batches_train, leave=True)
        with pbar as t:
            for batch_iteration in t:
                # manual batch iterations
                start_idx = batch_iteration * batch_size
                stop_idx = (batch_iteration + 1) * batch_size

                if batch_iteration == num_batches_train - 1:
                    stop_idx = data_train.shape[0]

                data_train_batch = data_train[rand_idx[start_idx: stop_idx]]
                y_train_batch = y_train_epoch[start_idx: stop_idx]

                # find max length
                length_tensor = torch.zeros_like(data_train_batch[:, 0, 0])
                for idx in range(data_train_batch.shape[0]):
                    boolean = (data_train_batch[idx, :, nop_idx] == 1.0).nonzero()
                    if boolean.shape[0] != 0:
                        length = (data_train_batch[idx, :, nop_idx] == 1.0).nonzero()[0][0].item()
                    else:
                        length = data_train_batch.shape[1]
                    length_tensor[idx] = length
                rnn_max_length = length_tensor.max()

                # reshaping for efficient parallelization
                # inp_flat_one_hot = data_train_batch.flatten(start_dim=1)
                inp_flat_one_hot = data_train_batch.transpose(dim0=1, dim1=2)  # convolution
                latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

                # TODO property predictor - input: latent points [batch, latent_dimensions], output: corresponding melting points [batch, 1]
                criterion = nn.MSELoss(reduction='none').to(device)
                predicted_train_property = property_predictor(latent_points)

                # compute train loss
                property_loss = criterion(input=predicted_train_property, target=y_train_batch)
                property_loss = torch.sum(
                    torch.mean(property_loss, dim=0) * property_predictor.normalized_weights_tensor,
                    dim=0
                )

                # use latent vectors as hidden (not zero-initialized hidden)
                hidden = vae_decoder.init_hidden(latent_points)
                out_one_hot = torch.zeros_like(data_train_batch, dtype=dtype, device=device)
                x_input = torch.zeros_like(data_train_batch[:, 0, :].unsqueeze(0), dtype=dtype, device=device)
                nop_tensor = -torch.ones_like(data_train_batch[:, 0, 0], dtype=dtype, device=device)
                asterisk_tensor = -torch.ones_like(data_train_batch[:, 0, 0], dtype=dtype, device=device)

                # teacher forcing
                for seq_index in range(int(rnn_max_length.item())):
                    out_one_hot_line, hidden = vae_decoder(x=x_input, hidden=hidden)
                    # find 'nop' idx
                    boolean_tensor = out_one_hot_line[0].argmax(dim=-1) == nop_idx
                    nonzero_tensor = boolean_tensor.nonzero()
                    for nonzero_num in range(nonzero_tensor.shape[0]):
                        nonzero_idx = nonzero_tensor[nonzero_num][0].item()
                        # store
                        if nop_tensor[nonzero_idx] < 0:
                            nop_tensor[nonzero_idx] = seq_index

                    # find 'asterisk' idx
                    boolean_tensor = out_one_hot_line[0].argmax(dim=-1) == asterisk_idx
                    nonzero_tensor = boolean_tensor.nonzero()
                    for nonzero_num in range(nonzero_tensor.shape[0]):
                        nonzero_idx = nonzero_tensor[nonzero_num][0].item()
                        # store
                        if asterisk_tensor[nonzero_idx] < 0:
                            asterisk_tensor[nonzero_idx] = 0.0  # generated one asterisk

                        elif asterisk_tensor[nonzero_idx] == 0:
                            asterisk_tensor[nonzero_idx] = seq_index  # generated two asterisks other than the initiator

                    out_one_hot[:, seq_index, :] = out_one_hot_line[0]
                    x_input = data_train_batch[:, seq_index, :].unsqueeze(0)  # answer

                # compute train loss
                # reconstruction_loss, divergence_loss = compute_elbo(
                #     data_train_batch, out_one_hot, nop_tensor, mus, log_vars, kld_alpha
                # )

                # change value of asterisk tensor: 0 -> -1
                asterisk_tensor[asterisk_tensor == 0.0] = -1.0

                # update nop tensor
                updated_nop_tensor = -torch.ones_like(data_train_batch[:, 0, 0], dtype=dtype, device=device)
                for idx in range(asterisk_tensor.shape[0]):
                    if nop_tensor[idx] == -1.0 and asterisk_tensor[idx] == -1.0:
                        continue
                    elif nop_tensor[idx] < 0 < asterisk_tensor[idx]:
                        updated_nop_tensor[idx] = asterisk_tensor[idx]
                    elif nop_tensor[idx] > 0 > asterisk_tensor[idx]:
                        updated_nop_tensor[idx] = nop_tensor[idx]
                    elif nop_tensor[idx] >= asterisk_tensor[idx]:
                        updated_nop_tensor[idx] = asterisk_tensor[idx]
                    else:
                        updated_nop_tensor[idx] = nop_tensor[idx]

                reconstruction_loss, _ = compute_elbo(
                    data_train_batch, out_one_hot, updated_nop_tensor, mus, log_vars, kld_alpha
                )
                divergence_loss = estimate_maximum_mean_discrepancy(posterior_z_samples=latent_points)
                weighted_divergence_loss = mmd_weight * divergence_loss

                # total_train_loss = property_train_loss + variational_train_loss
                # total_train_loss = reconstruction_loss + divergence_loss
                total_train_loss = reconstruction_loss + weighted_divergence_loss + property_loss

                # perform back propagation
                optimizer_encoder.zero_grad()
                optimizer_decoder.zero_grad()
                optimizer_property_predictor.zero_grad()

                # property_train_loss.backward(retain_graph=True)
                total_train_loss.backward()

                # nn.utils.clip_grad_norm_(vae_encoder.parameters(), max_norm=0.05, norm_type='inf')
                # nn.utils.clip_grad_norm_(vae_decoder.parameters(), max_norm=0.05, norm_type='inf')

                # plot_grad_flow(vae_encoder.named_parameters(), os.path.join(save_directory),
                #                f'encoder_epoch_{epoch}_batch_iter_{batch_iteration}')
                # plot_grad_flow(vae_decoder.named_parameters(), os.path.join(save_directory),
                #                f'decoder_epoch_{epoch}_batch_iter_{batch_iteration}')

                optimizer_encoder.step()
                optimizer_decoder.step()
                optimizer_property_predictor.step()

                # combine loss for average
                train_reconstruction_loss += float(reconstruction_loss.detach().cpu())
                train_divergence_loss += float(divergence_loss.detach().cpu())
                train_property_loss += float(property_loss.detach().cpu())

                # update progress bar
                t.set_postfix(
                    {'reconstruction_loss': '%.2f' % float(reconstruction_loss),
                     'divergence_loss': '%.2f' % float(divergence_loss),
                     'property_loss': '%.2f' % float(property_loss)}
                )
                # t.update()
            # t.close()

        # append train loss
        data_train_reconstruction_loss_list_teacher_forcing.append((train_reconstruction_loss / num_batches_train))
        data_train_divergence_loss_list_teacher_forcing.append((train_divergence_loss / num_batches_train))
        data_train_property_loss_list_teacher_forcing.append((train_property_loss / num_batches_train))

        # validation
        tqdm.write("Validation ...")
        with torch.no_grad():
            # predict property of valid data
            vae_encoder.eval()
            vae_decoder.eval()
            property_predictor.eval()

            # calculate validation reconstruction percentage
            valid_count_of_reconstruction_no_teacher_forcing = 0
            valid_reconstruction_loss = torch.zeros(size=(1,), dtype=dtype).cpu()
            valid_divergence_loss = torch.zeros(size=(1,), dtype=dtype).cpu()
            valid_property_loss = torch.zeros(size=(1,), dtype=dtype).cpu()

            y_valid_epoch = y_valid.clone()

            # mini-batch training
            pbar = tqdm(range(num_batches_valid), total=num_batches_valid, leave=True)
            with pbar as t:
                for batch_iteration in t:
                    # manual batch iterations
                    start_idx = batch_iteration * batch_size
                    stop_idx = (batch_iteration + 1) * batch_size

                    if batch_iteration == num_batches_train - 1:
                        stop_idx = data_valid.shape[0]

                    data_valid_batch = data_valid[start_idx: stop_idx]
                    y_valid_batch = y_valid_epoch[start_idx: stop_idx]

                    # find max length
                    length_tensor = torch.zeros_like(data_valid_batch[:, 0, 0])
                    for idx in range(data_valid_batch.shape[0]):
                        boolean = (data_valid_batch[idx, :, nop_idx] == 1.0).nonzero()
                        if boolean.shape[0] != 0:
                            length = (data_valid_batch[idx, :, nop_idx] == 1.0).nonzero()[0][0].item()
                        else:
                            length = data_valid_batch.shape[1]
                        length_tensor[idx] = length
                    rnn_max_length = length_tensor.max()

                    # encode
                    inp_flat_one_hot_valid = data_valid_batch.transpose(dim0=1, dim1=2)  # convolution
                    latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot_valid)

                    # property prediction
                    criterion = nn.MSELoss(reduction='none').to(device)
                    predicted_valid_property = property_predictor(latent_points)
                    property_loss = criterion(input=predicted_valid_property, target=y_valid_batch)
                    property_loss = torch.sum(
                        torch.mean(property_loss, dim=0) * property_predictor.normalized_weights_tensor,
                        dim=0
                    )

                    # use latent vectors as hidden (not zero-initialized hidden)
                    hidden = vae_decoder.init_hidden(latent_points)
                    out_one_hot = torch.zeros_like(data_valid_batch, dtype=dtype, device=device)
                    x_input = torch.zeros_like(data_valid_batch[:, 0, :].unsqueeze(0), dtype=dtype, device=device)
                    nop_tensor = -torch.ones_like(data_valid_batch[:, 0, 0], dtype=dtype, device=device)
                    asterisk_tensor = -torch.ones_like(data_valid_batch[:, 0, 0], dtype=dtype, device=device)

                    # no teacher forcing
                    for seq_index in range(int(rnn_max_length.item())):
                        out_one_hot_line, hidden = vae_decoder(x=x_input, hidden=hidden)
                        # find 'nop' idx
                        boolean_tensor = out_one_hot_line[0].argmax(dim=-1) == nop_idx
                        nonzero_tensor = boolean_tensor.nonzero()
                        for nonzero_num in range(nonzero_tensor.shape[0]):
                            nonzero_idx = nonzero_tensor[nonzero_num][0].item()
                            # store
                            if nop_tensor[nonzero_idx] < 0:
                                nop_tensor[nonzero_idx] = seq_index

                        # find 'asterisk' idx
                        boolean_tensor = out_one_hot_line[0].argmax(dim=-1) == asterisk_idx
                        nonzero_tensor = boolean_tensor.nonzero()
                        for nonzero_num in range(nonzero_tensor.shape[0]):
                            nonzero_idx = nonzero_tensor[nonzero_num][0].item()
                            # store
                            if asterisk_tensor[nonzero_idx] < 0:
                                asterisk_tensor[nonzero_idx] = 0.0  # generated one asterisk

                            elif asterisk_tensor[nonzero_idx] == 0:
                                asterisk_tensor[nonzero_idx] = seq_index  # generated two asterisks other than the initiator

                        out_one_hot[:, seq_index, :] = out_one_hot_line[0]

                        # change input - no teacher forcing (one-hot)
                        x_input = out_one_hot_line.argmax(dim=-1)
                        x_input = f.one_hot(x_input, num_classes=data_valid_batch.shape[2]).to(torch.float)

                    # x_indices
                    x_indices = data_valid_batch.argmax(dim=-1)
                    x_hat_prob = f.softmax(out_one_hot, dim=-1)
                    x_hat_indices = x_hat_prob.argmax(dim=-1)

                    # change value of asterisk tensor: 0 -> -1
                    asterisk_tensor[asterisk_tensor == 0.0] = -1.0

                    # update nop tensor
                    updated_nop_tensor = -torch.ones_like(data_valid_batch[:, 0, 0], dtype=dtype, device=device)
                    for idx in range(asterisk_tensor.shape[0]):
                        if nop_tensor[idx] == -1.0 and asterisk_tensor[idx] == -1.0:
                            continue
                        elif nop_tensor[idx] < 0 < asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = asterisk_tensor[idx]
                        elif nop_tensor[idx] > 0 > asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = nop_tensor[idx]
                        elif nop_tensor[idx] >= asterisk_tensor[idx]:
                            updated_nop_tensor[idx] = asterisk_tensor[idx]
                        else:
                            updated_nop_tensor[idx] = nop_tensor[idx]

                    # modify x_hat_indices value to nop_idx after the nop appears
                    max_length = x_hat_indices.shape[1]
                    for idx, value in enumerate(updated_nop_tensor):
                        if value >= 0.0:
                            x_hat_indices[idx, int(value): max_length] = nop_idx

                    # count right molecules
                    difference_tensor = torch.abs(x_indices - x_hat_indices)
                    difference_tensor = difference_tensor.sum(dim=-1)

                    for idx in range(len(difference_tensor)):
                        if difference_tensor[idx] == 0:
                            valid_count_of_reconstruction_no_teacher_forcing += 1

                    # compute no teacher forcing loss
                    # reconstruction_loss, divergence_loss = compute_elbo(
                    #     data_valid_batch, out_one_hot, nop_tensor, mus, log_vars, kld_alpha
                    # )
                    reconstruction_loss, _ = compute_elbo(
                        data_valid_batch, out_one_hot, updated_nop_tensor, mus, log_vars, kld_alpha
                    )
                    divergence_loss = estimate_maximum_mean_discrepancy(posterior_z_samples=latent_points)

                    # combine loss for average
                    valid_reconstruction_loss += float(reconstruction_loss.detach().cpu())
                    valid_divergence_loss += float(divergence_loss.detach().cpu())
                    valid_property_loss += float(property_loss.detach().cpu())

                    # update progress bar
                    t.set_postfix(
                        {'reconstruction_loss': '%.2f' % float(reconstruction_loss),
                         'divergence_loss': '%.2f' % float(divergence_loss),
                         'property_loss': '%.2f' % float(property_loss)}
                    )
                    # t.update()
                # t.close()

            # append validation loss
            data_valid_reconstruction_loss_list_no_teacher_forcing.append((valid_reconstruction_loss / num_batches_valid))
            data_valid_divergence_loss_list_no_teacher_forcing.append((valid_divergence_loss / num_batches_valid))
            data_valid_property_loss_list_no_teacher_forcing.append((valid_property_loss / num_batches_valid))
            print('[VAE] Valid Reconstruction (no teacher forcing) Percent Rate is %.3f' %
                  (100 * valid_count_of_reconstruction_no_teacher_forcing / data_valid.shape[0]), flush=True)

            # train
            vae_encoder.train()
            vae_decoder.train()
            property_predictor.train()

        end = time.time()
        print(f"Epoch {epoch + 1} took {(end - start) / 60.0:.3f} minutes took for training", flush=True)

        # step learning rate scheduler  # trial -> delete learning scheduler
        if (epoch + 1) % step_size == 0:
            encoder_lr_scheduler.step()
            decoder_lr_scheduler.step()
            property_predictor_lr_scheduler.step()

            print(optimizer_encoder, flush=True)
            print(optimizer_decoder, flush=True)
            print(optimizer_property_predictor, flush=True)

        # learning rate decay
        # if (epoch + 1) % step_size == 0 and (epoch + 1) / step_size > 0.9:
        #     print("Learning rate is decayed", flush=True)
        #     encoder_lr_scheduler.step()
        #     decoder_lr_scheduler.step()

    # property prediction
    with torch.no_grad():
        print("Plot property prediction ...", flush=True)
        # eval mode
        vae_encoder.eval()
        vae_decoder.eval()
        property_predictor.eval()

        # train encode
        inp_flat_one_hot = data_train.transpose(dim0=1, dim1=2)  # convolution
        latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)
        train_prediction = y_scaler.inverse_transform(property_predictor(latent_points).cpu().numpy())

        # valid encode
        inp_flat_one_hot = data_valid.transpose(dim0=1, dim1=2)  # convolution
        latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)
        valid_prediction = y_scaler.inverse_transform(property_predictor(latent_points).cpu().numpy())

        # get r2 score & plot
        property_name = ['LogP', 'SC_score']
        y_train_true_original_scale = y_scaler.inverse_transform(y_train.cpu().numpy())
        y_valid_true_original_scale = y_scaler.inverse_transform(y_valid.cpu().numpy())
        for idx in range(property_predictor.property_dim):
            y_train_true = y_train_true_original_scale[:, idx]
            y_valid_true = y_valid_true_original_scale[:, idx]
            y_train_prediction = train_prediction[:, idx]
            y_valid_prediction = valid_prediction[:, idx]

            train_r2_score = r2_score(y_true=y_train_true, y_pred=y_train_prediction)
            test_r2_score = r2_score(y_true=y_valid_true, y_pred=y_valid_prediction)

            plt.figure(figsize=(6, 6), dpi=300)
            plt.plot(y_train_true, y_train_prediction, 'bo',
                     label='Train R2 score: %.3f' % train_r2_score)
            plt.plot(y_valid_true, y_valid_prediction, 'ro',
                     label='Validation R2 score: %.3f' % test_r2_score)

            plt.legend()
            plt.xlabel('True')
            plt.ylabel('Prediction')

            plt.savefig(os.path.join(save_directory, "property_prediction_%s.png" % property_name[idx]))
            plt.show()
            plt.close()

    # save encoder, decoder, and property network
    torch.save(vae_encoder.state_dict(), os.path.join(save_directory, 'encoder.pth'))
    torch.save(vae_decoder.state_dict(), os.path.join(save_directory, 'decoder.pth'))
    torch.save(property_predictor.state_dict(), os.path.join(save_directory, 'property_predictor.pth'))

    # plot results
    iteration_list = list(range(1, num_epochs + 1))

    plot_learning_curve(
        iteration_list=iteration_list,
        train_loss_teacher_forcing=torch.tensor(data_train_reconstruction_loss_list_teacher_forcing).numpy(),
        valid_loss_no_teacher_forcing=torch.tensor(data_valid_reconstruction_loss_list_no_teacher_forcing).numpy(),
        save_directory=save_directory,
        title='Reconstruction Loss'
    )
    plot_learning_curve(
        iteration_list=iteration_list,
        train_loss_teacher_forcing=torch.tensor(data_train_divergence_loss_list_teacher_forcing).numpy(),
        valid_loss_no_teacher_forcing=torch.tensor(data_valid_divergence_loss_list_no_teacher_forcing).numpy(),
        save_directory=save_directory,
        title='Divergence Loss'
    )
    plot_learning_curve(
        iteration_list=iteration_list,
        train_loss_teacher_forcing=torch.tensor(data_train_property_loss_list_teacher_forcing).numpy(),
        valid_loss_no_teacher_forcing=torch.tensor(data_valid_property_loss_list_no_teacher_forcing).numpy(),
        save_directory=save_directory,
        title='Property Loss'
    )


def compute_elbo(x, x_hat, nop_tensor, mus, log_vars, kld_alpha):
    # get values
    batch_size = x.shape[0]
    max_length = x.shape[1]

    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    recon_loss = criterion(inp, target)
    recon_loss = recon_loss.reshape(batch_size, max_length)  # reshape

    # exclude error contribution from nop
    err_boolean_tensor = torch.ones_like(recon_loss)

    for idx, value in enumerate(nop_tensor):
        if value >= 0.0:
            err_boolean_tensor[idx, int(value + 1): max_length] = 0.0

    # multiply
    recon_loss = recon_loss * err_boolean_tensor
    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())

    return recon_loss.sum() / err_boolean_tensor.count_nonzero(), kld_alpha * kld


def plot_learning_curve(iteration_list,
                        train_loss_teacher_forcing,
                        # train_loss_no_teacher_forcing,
                        valid_loss_no_teacher_forcing,
                        save_directory,
                        title: str):

    plt.figure(figsize=(8, 8), dpi=300)
    plt.plot(iteration_list, train_loss_teacher_forcing, 'b-', label='Train Loss (teacher forcing)')
    # plt.plot(iteration_list, train_loss_no_teacher_forcing, 'g-', label='Train Loss (no teacher forcing)')
    plt.plot(iteration_list, valid_loss_no_teacher_forcing, 'r-', label='Validation Loss (no teacher forcing)')

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("%s" % title, fontsize=12)

    plt.legend()
    plt.savefig(os.path.join(save_directory, "%s_learning_curve.png" % title))
    plt.show()
    plt.close()


def estimate_maximum_mean_discrepancy(posterior_z_samples: torch.Tensor):
    # set dtype and device
    dtype = posterior_z_samples.dtype
    device = posterior_z_samples.device

    # set dimension and number of samples
    latent_dimension = posterior_z_samples.shape[1]
    number_of_samples = posterior_z_samples.shape[0]

    # get prior samples
    prior_z_samples = torch.randn(size=(number_of_samples, latent_dimension), device=device, dtype=dtype)

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
