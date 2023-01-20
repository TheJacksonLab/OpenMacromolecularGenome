import os

import selfies as sf

import torch
import torch.nn.functional as f

from rdkit import Chem
from rdkit.Chem import Draw

from vae.preprocess import multiple_selfies_to_hot

from rdkit.Chem import MolFromSmiles


def is_correct_smiles(smiles):
    """
    Using RDKit to calculate whether molecule is syntactically and
    semantically valid.
    """
    if smiles == "":
        return False

    try:
        return MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False


def reconstruct_molecules(type_of_encoding, encoding_alphabet, one_hot_encoded_vector):
    # set parameters
    total_valid = 0
    valid_molecules_idx = []
    valid_molecules = []

    molecules_num = one_hot_encoded_vector.shape[0]
    reconstruct_one_hot = one_hot_encoded_vector.clone().detach()

    for idx in range(molecules_num):
        gathered_atoms = ''
        one_hot_idx = reconstruct_one_hot[idx]

        for value in one_hot_idx:
            gathered_atoms += encoding_alphabet[value]
        # selfies
        if type_of_encoding == 1:
            generated_molecule = gathered_atoms.replace('[nop]', '')
        # smiles
        else:
            generated_molecule = None
        # selfies to smiles
        smiles_generated_molecule = sf.decoder(generated_molecule)
        if is_correct_smiles(smiles_generated_molecule):
            total_valid += 1
            valid_molecules_idx.append(idx)
            valid_molecules.append(smiles_generated_molecule)

    return total_valid, valid_molecules_idx, valid_molecules


def one_point_latent_sampling(encoder, decoder, encoding_list, encoding_alphabet,
                              largest_molecule_len, molecule_idx, repeat_num, save_directory):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoding_list_func = encoding_list.copy()
    trial_molecule_set = [encoding_list_func[molecule_idx] for _ in range(repeat_num)]

    # data & torch tensor
    data = multiple_selfies_to_hot(trial_molecule_set, largest_molecule_len, encoding_alphabet)
    data_tensor = torch.tensor(data, dtype=torch.float).to(device)

    # eval
    encoder.eval()
    decoder.eval()

    # encoding
    inp_flat_one_hot = data_tensor.flatten(start_dim=1)
    latent_points, mus, log_vars = encoder(inp_flat_one_hot)
    latent_points = latent_points.unsqueeze(0)

    # decoding
    hidden = decoder.init_hidden(batch_size=repeat_num)
    out_one_hot = torch.zeros_like(data_tensor, device=device)
    for seq_index in range(data_tensor.shape[1]):
        out_one_hot_line, hidden = decoder(latent_points, hidden)
        out_one_hot[:, seq_index, :] = out_one_hot_line[0]

    x_hat_prob = f.softmax(out_one_hot, dim=-1)
    x_hat_indices = x_hat_prob.argmax(dim=-1)

    # generating molecules
    _, _, molecules = reconstruct_molecules(
        type_of_encoding=1,
        encoding_alphabet=encoding_alphabet,
        one_hot_encoded_vector=x_hat_indices
    )
    # # generating images
    # image_save_directory = os.path.join(save_directory, 'sample_repeat_images')
    # if not os.path.exists(image_save_directory):
    #     os.mkdir(image_save_directory)

    # for mol_idx in range(len(molecules)):
    #     mol = Chem.MolFromSmiles(molecules[mol_idx])
    #     Draw.MolToFile(mol, os.path.join(image_save_directory, "molecule_%d.png" % mol_idx))
    return molecules


def two_points_latent_interpolate(encoder, decoder, encoding_list, encoding_alphabet,
                                  largest_molecule_len, molecule_idx_1, molecule_idx_2, interpolation_number,
                                  save_directory):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoding_list_func = encoding_list.copy()

    # eval
    encoder.eval()
    decoder.eval()

    two_molecules = [encoding_list_func[molecule_idx_1], encoding_list_func[molecule_idx_2]]

    # data & torch tensor
    data = multiple_selfies_to_hot(two_molecules, largest_molecule_len, encoding_alphabet)
    data_tensor = torch.tensor(data, dtype=torch.float).to(device)

    # encoding
    inp_flat_one_hot = data_tensor.flatten(start_dim=1)
    latent_points, mus, log_vars = encoder(inp_flat_one_hot)

    # TODO make an interpolation function between two torch tensors of size [2, 25], and unsqueeze(0)
    weights = torch.linspace(start=0, end=1, steps=interpolation_number).to(device)
    linear_interpolate = torch.empty_like(latent_points).to(device)
    input_tensor = latent_points[0].view(-1, latent_points.shape[1])
    output_tensor = latent_points[1].view(-1, latent_points.shape[1])

    for weight in weights:
        interpolation = torch.lerp(input=input_tensor, end=output_tensor, weight=weight)
        linear_interpolate = torch.cat([linear_interpolate, interpolation])
    linear_interpolate = linear_interpolate[2:]
    latent_points = linear_interpolate.unsqueeze(0)

    # decoding
    hidden = decoder.init_hidden(batch_size=interpolation_number)
    out_one_hot = torch.zeros(size=[interpolation_number, data_tensor.shape[1], data_tensor.shape[2]])
    # out_one_hot = torch.zeros_like(data_tensor, device=device)
    for seq_index in range(out_one_hot.shape[1]):
        out_one_hot_line, hidden = decoder(latent_points, hidden)
        out_one_hot[:, seq_index, :] = out_one_hot_line[0]

    x_hat_prob = f.softmax(out_one_hot, dim=-1)
    x_hat_indices = x_hat_prob.argmax(dim=-1)

    # generating molecules
    _, _, molecules = reconstruct_molecules(
        type_of_encoding=1,
        encoding_alphabet=encoding_alphabet,
        one_hot_encoded_vector=x_hat_indices
    )
    print('[VAE] Linear interpolated molecules are ...', flush=True)
    print(molecules)

    # generating images
    image_save_directory = os.path.join(save_directory, 'interpolation_images')
    if not os.path.exists(image_save_directory):
        os.mkdir(image_save_directory)

    for mol_idx in range(len(molecules)):
        mol = Chem.MolFromSmiles(molecules[mol_idx])
        Draw.MolToFile(mol, os.path.join(image_save_directory, "molecule_%d.png" % mol_idx))






