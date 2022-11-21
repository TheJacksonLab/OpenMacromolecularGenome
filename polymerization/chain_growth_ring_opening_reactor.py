import copy

from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.rdchem import BondType, Atom

from ._base import BasePolymerization


class ChainGrowthRingOpeningReactor(BasePolymerization):
    def __init__(self, reaction_monomers, reaction_groups, reaction_sites, mechanism):
        super(ChainGrowthRingOpeningReactor, self).__init__()
        self.reaction_monomers = reaction_monomers
        self.reaction_groups = reaction_groups
        self.reaction_sites = reaction_sites
        self.mechanism = mechanism

        # get monomer_1_key -> only 'monomer_1' (there is only one monomer in chain growth)
        self.monomer_1_key = 'monomer_1'

        # find smiles and reactions sites for monomer_1
        self.monomer_1_smiles = self.reaction_monomers[self.monomer_1_key]
        self.monomer_1_mol = Chem.MolFromSmiles(self.monomer_1_smiles)
        self.monomer_1_reaction_sites = self.reaction_sites[self.monomer_1_key]

        # set new monomer and repeating unit
        self.new_monomer = copy.deepcopy(self.monomer_1_mol)
        self.repeating_unit_smiles = None

        # get break atomic and bond idx
        self.break_bond_atomic_site_1 = None
        self.break_bond_atomic_site_2 = None
        self.break_bond_site = None
        self.get_break_bond_site()

        # set rewritable mol object
        self.mw_1 = RWMol(self.new_monomer)

    def get_break_bond_site(self):
        reaction_cluster = self.reaction_sites['monomer_1'][0]
        # (1) lactone
        if self.mechanism['chain_growth_ring_opening'] == ['lactone']:
            for i in range(len(reaction_cluster)):
                for j in range(i + 1, len(reaction_cluster)):  # find a specific element
                    # get site and atom
                    site_i = reaction_cluster[i]
                    site_i_atom = self.monomer_1_mol.GetAtomWithIdx(site_i).GetSymbol()
                    site_j = reaction_cluster[j]
                    site_j_atom = self.monomer_1_mol.GetAtomWithIdx(site_j).GetSymbol()
                    atom_list = [site_i_atom, site_j_atom]
                    # check two atoms are connected
                    if self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j) is not None:
                        bond_site = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetIdx()
                        bond_type = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetBondType()
                        if 'O' in atom_list and 'C' in atom_list and bond_type == BondType.SINGLE:
                            self.break_bond_atomic_site_1 = site_i
                            self.break_bond_atomic_site_2 = site_j
                            self.break_bond_site = bond_site

        # (2) lactam
        elif self.mechanism['chain_growth_ring_opening'] == ['lactam']:
            for i in range(len(reaction_cluster)):
                for j in range(i + 1, len(reaction_cluster)):  # find a specific element
                    # get site and atom
                    site_i = reaction_cluster[i]
                    site_i_atom = self.monomer_1_mol.GetAtomWithIdx(site_i).GetSymbol()
                    site_j = reaction_cluster[j]
                    site_j_atom = self.monomer_1_mol.GetAtomWithIdx(site_j).GetSymbol()
                    atom_list = [site_i_atom, site_j_atom]
                    # check two atoms are connected
                    if self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j) is not None:
                        bond_site = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetIdx()
                        bond_type = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetBondType()
                        if 'N' in atom_list and 'C' in atom_list and bond_type == BondType.SINGLE:
                            self.break_bond_atomic_site_1 = site_i
                            self.break_bond_atomic_site_2 = site_j
                            self.break_bond_site = bond_site

        # (3) cyclic ether - break bond with less bulky carbon. If degrees are equal, randomly break a bond
        elif self.mechanism['chain_growth_ring_opening'] == ['cyclic_ether']:
            hydrogen_number = 0
            for i in range(len(reaction_cluster)):
                for j in range(i + 1, len(reaction_cluster)):  # find a specific element
                    # get site and atom
                    site_i = reaction_cluster[i]
                    site_i_atom = self.monomer_1_mol.GetAtomWithIdx(site_i).GetSymbol()
                    site_j = reaction_cluster[j]
                    site_j_atom = self.monomer_1_mol.GetAtomWithIdx(site_j).GetSymbol()
                    atom_list = [site_i_atom, site_j_atom]
                    # check two atoms are connected
                    if self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j) is not None:
                        bond_site = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetIdx()
                        bond_type = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetBondType()
                        if 'O' in atom_list and 'C' in atom_list and bond_type == BondType.SINGLE:
                            number_of_h_at_site_i = self.monomer_1_mol.GetAtomWithIdx(site_i).GetTotalNumHs()
                            number_of_h_at_site_j = self.monomer_1_mol.GetAtomWithIdx(site_j).GetTotalNumHs()
                            new_hydrogen_number = number_of_h_at_site_i + number_of_h_at_site_j
                            if new_hydrogen_number >= hydrogen_number:
                                hydrogen_number = new_hydrogen_number
                                self.break_bond_atomic_site_1 = site_i
                                self.break_bond_atomic_site_2 = site_j
                                self.break_bond_site = bond_site

        # (4) cyclic olefin
        elif self.mechanism['chain_growth_ring_opening'] == ['cyclic_olefin']:
            for i in range(len(reaction_cluster)):
                for j in range(i + 1, len(reaction_cluster)):  # find a specific element
                    # get site and atom
                    site_i = reaction_cluster[i]
                    site_i_atom = self.monomer_1_mol.GetAtomWithIdx(site_i).GetSymbol()
                    site_j = reaction_cluster[j]
                    site_j_atom = self.monomer_1_mol.GetAtomWithIdx(site_j).GetSymbol()
                    atom_list = [site_i_atom, site_j_atom]
                    # check two atoms are connected
                    if self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j) is not None:
                        bond_site = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetIdx()
                        bond_type = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetBondType()
                        if 'C' in atom_list and 'C' in atom_list and bond_type == BondType.DOUBLE:
                            self.break_bond_atomic_site_1 = site_i
                            self.break_bond_atomic_site_2 = site_j
                            self.break_bond_site = bond_site

        # (5) cyclic carbonate
        # TODO decide break which bond (if asymmetric) -> current decide a position randomly
        elif self.mechanism['chain_growth_ring_opening'] == ['cyclic_carbonate']:
            carbon_idx = None
            oxygen_idx_list = []
            # find carbon with double bond with oxygen
            for i in range(len(reaction_cluster)):
                for j in range(i + 1, len(reaction_cluster)):  # find a specific element
                    # get site and atom
                    site_i = reaction_cluster[i]
                    site_i_atom = self.monomer_1_mol.GetAtomWithIdx(site_i).GetSymbol()
                    site_j = reaction_cluster[j]
                    site_j_atom = self.monomer_1_mol.GetAtomWithIdx(site_j).GetSymbol()
                    atom_idx_list = [site_j, site_j]
                    atom_list = [site_i_atom, site_j_atom]
                    # check two atoms are connected with double bond
                    if self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j) is not None:
                        bond_type = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetBondType()
                        if bond_type == BondType.DOUBLE:
                            carbon_idx = atom_idx_list[atom_list.index('C')]
                        elif bond_type == BondType.SINGLE and 'O' in atom_list:
                            oxygen_idx_list.append(atom_idx_list[atom_list.index('O')])

            self.break_bond_atomic_site_1 = carbon_idx
            self.break_bond_atomic_site_2 = oxygen_idx_list[0]
            self.break_bond_site = self.monomer_1_mol.GetBondBetweenAtoms(carbon_idx, oxygen_idx_list[0]).GetIdx()

        # (6) cyclic sulfide
        elif self.mechanism['chain_growth_ring_opening'] == ['cyclic_sulfide']:
            for i in range(len(reaction_cluster)):
                for j in range(i + 1, len(reaction_cluster)):  # find a specific element
                    # get site and atom
                    site_i = reaction_cluster[i]
                    site_i_atom = self.monomer_1_mol.GetAtomWithIdx(site_i).GetSymbol()
                    site_j = reaction_cluster[j]
                    site_j_atom = self.monomer_1_mol.GetAtomWithIdx(site_j).GetSymbol()
                    atom_list = [site_i_atom, site_j_atom]
                    # check two atoms are connected
                    if self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j) is not None:
                        bond_site = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetIdx()
                        bond_type = self.monomer_1_mol.GetBondBetweenAtoms(site_i, site_j).GetBondType()
                        if 'S' in atom_list and 'C' in atom_list and bond_type == BondType.SINGLE:
                            self.break_bond_atomic_site_1 = site_i
                            self.break_bond_atomic_site_2 = site_j
                            self.break_bond_site = bond_site

    def react(self):
        # break bond
        self.mw_1.RemoveBond(self.break_bond_atomic_site_1, self.break_bond_atomic_site_2)
        # set wildcard (*) idx
        wildcard_id_1 = self.mw_1.AddAtom(Atom('*'))
        wildcard_id_2 = self.mw_1.AddAtom(Atom('*'))

        # add bond
        if self.mechanism['chain_growth_ring_opening'] != ['cyclic_olefin']:
            self.mw_1.AddBond(self.break_bond_atomic_site_1, wildcard_id_1, BondType.SINGLE)
            self.mw_1.AddBond(self.break_bond_atomic_site_2, wildcard_id_2, BondType.SINGLE)
        elif self.mechanism['chain_growth_ring_opening'] == ['cyclic_olefin']:
            # add one more carbon
            carbon_id = self.mw_1.AddAtom(Atom('C'))
            # choose smaller idx
            if self.break_bond_atomic_site_1 < self.break_bond_atomic_site_2:
                self.mw_1.AddBond(self.break_bond_atomic_site_1, carbon_id, BondType.DOUBLE)
                self.mw_1.AddBond(carbon_id, wildcard_id_1, BondType.SINGLE)
                self.mw_1, modified_monomer_1_bnd_list = self.remove_atoms_and_relabel(
                    self.mw_1, [self.break_bond_atomic_site_2], [self.break_bond_atomic_site_2 + 1, wildcard_id_2]
                )
                self.mw_1.AddBond(modified_monomer_1_bnd_list[0], modified_monomer_1_bnd_list[1], BondType.SINGLE)

            elif self.break_bond_atomic_site_1 > self.break_bond_atomic_site_2:
                self.mw_1.AddBond(self.break_bond_atomic_site_2, carbon_id, BondType.DOUBLE)
                self.mw_1.AddBond(carbon_id, wildcard_id_1, BondType.SINGLE)
                self.mw_1, modified_monomer_1_bnd_list = self.remove_atoms_and_relabel(
                    self.mw_1, [self.break_bond_atomic_site_1], [self.break_bond_atomic_site_1 + 1, wildcard_id_2]
                )
                self.mw_1.AddBond(modified_monomer_1_bnd_list[0], modified_monomer_1_bnd_list[1], BondType.SINGLE)

        # convert to smiles
        self.repeating_unit_smiles = Chem.MolToSmiles(self.mw_1)

        return self.repeating_unit_smiles, self.mechanism
