import sys

import torch

from collections import OrderedDict
from copy import deepcopy
from rdkit.Chem.rdchem import Mol
from typing import List

from molecule_chef.module.preprocess import AtomFeatureParams
from molecule_chef.module.utils import TorchDetails, GraphAsAdjList


class BaseMoleculeChef(object):
    def __init__(
            self,
            atomic_feature_parameters=AtomFeatureParams(),
            torch_details=TorchDetails(device='cpu', data_type=torch.float32)
    ):
        super(BaseMoleculeChef, self).__init__()
        self.atomic_feature_parameters = atomic_feature_parameters
        self.electron_negativity = {
            'Ag': 1.93, 'Al': 1.61, 'Ar': 3.98, 'As': 2.18, 'Au': 2.54, 'B': 2.04, 'Ba': 0.89, 'Be': 1.57,
            'Bi': 2.02, 'Br': 2.96, 'C': 2.55, 'Ca': 1.0, 'Cd': 1.69, 'Ce': 1.12, 'Cl': 3.16, 'Co': 1.88,
            'Cr': 1.66, 'Cs': 0.79, 'Cu': 1.90, 'Dy': 1.22, 'Eu': 3.98, 'F': 3.98, 'Fe': 1.83, 'Ga': 1.81,
            'Ge': 2.01, 'H': 2.20, 'He': 3.98, 'Hf': 1.3, 'Hg': 2.0, 'I': 2.66, 'In': 1.78, 'Ir': 2.20,
            'K': 0.82, 'La': 1.10, 'Li': 0.98, 'Mg': 1.31, 'Mn': 1.55, 'Mo': 2.16, 'N': 3.04, 'Na': 0.93,
            'Nd': 1.14, 'Ni': 1.91, 'O': 3.44, 'Os': 2.20, 'P': 2.19, 'Pb': 2.33, 'Pd': 2.20, 'Pr': 1.13,
            'Pt': 2.28, 'Pu': 1.28, 'Ra': 0.9, 'Rb': 0.82, 'Re': 1.9, 'Rh': 2.28, 'Rn': 3.98, 'Ru': 2.2,
            'S': 2.58, 'Sb': 2.05, 'Sc': 1.36, 'Se': 2.55, 'Si': 1.90, 'Sm': 1.17, 'Sn': 1.96, 'Sr': 0.95,
            'Ta': 1.5, 'Tb': 3.98, 'Tc': 1.9, 'Te': 2.1, 'Th': 1.3, 'Ti': 1.54, 'Tl': 1.62, 'Tm': 1.25,
            'U': 1.38, 'V': 1.63, 'W': 2.36, 'Xe': 2.6, 'Y': 1.22, 'Yb': 3.98, 'Zn': 1.65, 'Zr': 1.33
        }
        self.dtype = torch_details.data_type
        self.device = torch_details.device

    def _get_atom_features(self, atom: Mol):
        # construct atom feature dictionary
        feature_dict = OrderedDict()
        feature_dict['atomic_symbol'] = self._encode_boolean_information(
            atom.GetSymbol(), self.atomic_feature_parameters.atom_types
        )
        # how many atoms
        feature_dict['atomic_degree'] = self._encode_boolean_information(
            atom.GetDegree(), self.atomic_feature_parameters.degrees
        )
        # how many bonds
        feature_dict['atomic_explicit_valences'] = self._encode_boolean_information(
            atom.GetExplicitValence(), self.atomic_feature_parameters.explicit_valences
        )
        feature_dict['atomic_hybridization'] = self._encode_boolean_information(
            atom.GetHybridization(), self.atomic_feature_parameters.hybridization, if_missing_set_as_last_flag=True
        )
        feature_dict['atomic_electron_negativity'] = [self.electron_negativity[atom.GetSymbol()]]
        feature_dict['atomic_number'] = [atom.GetAtomicNum()]
        feature_dict['atomic_hydrogen_number'] = [atom.GetTotalNumHs()]
        feature_dict['atomic_aromaticity'] = [atom.GetIsAromatic()]
        # feature_dict['formal_charge'] = [atom.GetFormalCharge()]

        # convert dictionary to torch tensor
        feature_list = list()
        for value in feature_dict.values():
            feature_list = feature_list + deepcopy(value)

        return torch.tensor(feature_list, dtype=self.dtype, device=self.device)

        # memory
        # return torch.tensor(feature_list, dtype=self.dtype, device=torch.device('cuda:1'))
        # return torch.tensor(feature_list, dtype=self.dtype, device=torch.device('cuda:1'))

    @ staticmethod
    def _encode_boolean_information(value, allowable_set, if_missing_set_as_last_flag=False):
        if value not in set(allowable_set):
            if if_missing_set_as_last_flag:
                value = allowable_set[-1]
            else:
                raise RuntimeError
        return list(map(lambda s: value == s, allowable_set))

    def _get_atomic_feature_vectors_and_adjacency_matrix(self, mol_array: List[Mol]):
        # set maximum num atoms
        maximum_num_atoms = 0
        for mol in mol_array:
            num_atoms = mol.GetNumAtoms()
            if maximum_num_atoms < num_atoms:
                maximum_num_atoms = num_atoms

        # initialize atomic feature vectors and adjacency matrix
        num_of_molecules = len(mol_array)
        atomic_feature_vectors = torch.zeros(
            size=(num_of_molecules, maximum_num_atoms, self.atomic_feature_parameters.atom_feature_length),
            dtype=self.dtype,
            device=self.device
        )
        adjacency_matrix = torch.zeros(
            (num_of_molecules, maximum_num_atoms, maximum_num_atoms, self.atomic_feature_parameters.num_bond_types),
            dtype=self.dtype,
            device=self.device
        )

        # get atomic feature vectors and adjacency matrix
        for idx, mol in enumerate(mol_array):
            # set atomic parameter
            atoms = mol.GetAtoms()

            # get atomic feature vectors - index and feature
            for atom in atoms:
                atomic_feature_vectors[idx, atom.GetIdx(), :] = self._get_atom_features(atom)
            
            # get adjacency dictionary
            adjacency_dict = OrderedDict()
            for bond_type in self.atomic_feature_parameters.bond_names:
                adjacency_dict[bond_type] = list()

            for bond in mol.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                bond_type = self.atomic_feature_parameters.get_bond_name(bond)
                adjacency_dict[bond_type].append(
                    (begin_atom.GetIdx(), end_atom.GetIdx())
                )

            # get adjacency matrix
            # bond type idx -> aromatic: 0, single: 1, double: 2, triple: 3
            for bond_type, [_, values] in enumerate(adjacency_dict.items()):
                for idx_set in values:
                    begin_atom_idx = idx_set[0]
                    end_atom_idx = idx_set[1]
                    adjacency_matrix[idx, begin_atom_idx, end_atom_idx, bond_type] = 1.0
                    adjacency_matrix[idx, end_atom_idx, begin_atom_idx, bond_type] = 1.0

        return atomic_feature_vectors, adjacency_matrix

    def get_atomic_feature_vectors_and_adjacency_list(self, mol_array: List[Mol]):
        # assign later
        total_number_of_atoms = 0
        node_to_graph_id = list()
        edge_type_to_adjacency_list_map = OrderedDict()

        atomic_feature_vectors = list()
        for bond_type in self.atomic_feature_parameters.bond_names:  # set edge type to adjacency list dict
            edge_type_to_adjacency_list_map[bond_type] = list()

        # assign
        for mol_idx, mol in enumerate(mol_array):
            # update node to graph id
            num_atoms = mol.GetNumAtoms()
            node_to_graph_id[total_number_of_atoms: num_atoms] = num_atoms * [mol_idx]

            # update atomic feature vectors
            for atom in mol.GetAtoms():
                atomic_feature_vectors.append(self._get_atom_features(atom))

            # update bond_idx
            for bond in mol.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                bond_type = self.atomic_feature_parameters.get_bond_name(bond)
                bond_idx = [total_number_of_atoms + begin_atom.GetIdx(), total_number_of_atoms + end_atom.GetIdx()]
                edge_type_to_adjacency_list_map[bond_type].append(bond_idx)

            # update total number of atoms
            total_number_of_atoms += num_atoms

        # convert to tensor
        node_to_graph_id = torch.tensor(node_to_graph_id, dtype=torch.long, device=self.device)

        # memory
        # node_to_graph_id = torch.tensor(node_to_graph_id, dtype=torch.long, device=torch.device('cuda:1'))

        atomic_feature_vectors = torch.stack(atomic_feature_vectors)
        for bond_type in self.atomic_feature_parameters.bond_names:
            edge_type_to_adjacency_list_map[bond_type] = torch.tensor(
                edge_type_to_adjacency_list_map[bond_type], dtype=torch.long, device=self.device
            ).t()

            # memory
            # edge_type_to_adjacency_list_map[bond_type] = torch.tensor(
            #     edge_type_to_adjacency_list_map[bond_type], dtype=torch.long, device=torch.device('cuda:1')
            # ).t()

        graph_adjacency_list = GraphAsAdjList(
            atomic_feature_vectors=atomic_feature_vectors,
            edge_type_to_adjacency_list_map=edge_type_to_adjacency_list_map,
            node_to_graph_id=node_to_graph_id
        )

        return graph_adjacency_list
