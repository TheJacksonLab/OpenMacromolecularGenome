import copy

from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.rdchem import BondType, Atom

from ._base import BasePolymerization


class MetathesisReactor(BasePolymerization):
    def __init__(self, reaction_monomers, reaction_groups, reaction_sites, mechanism):
        super(MetathesisReactor, self).__init__()
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

        # set rewritable mol object
        self.mw_1 = RWMol(self.new_monomer)

        # get monomer del and bond list
        self.monomer_1_del_list = []
        self.monomer_1_bond_list = []
        self.get_del_bond_list()

    def get_del_bond_list(self):
        # (1) terminal diene - ADMET
        if self.mechanism['metathesis'] == ['terminal_diene']:
            for idx, (site_1, site_2) in enumerate(self.reaction_sites['monomer_1']):
                # get number of hydrogen connected to terminal diene
                number_of_hydrogen_at_site_1 = self.monomer_1_mol.GetAtomWithIdx(site_1).GetTotalNumHs()
                number_of_hydrogen_at_site_2 = self.monomer_1_mol.GetAtomWithIdx(site_2).GetTotalNumHs()
                if idx == 0:  # terminal diene -> different from the other reactions (asymmetric) -> delete one side
                    self.monomer_1_del_list.append(site_1)
                    self.monomer_1_del_list.append(site_2)
                    if number_of_hydrogen_at_site_1 == 1:
                        self.monomer_1_bond_list.append(site_1 + 1)
                    elif number_of_hydrogen_at_site_2 == 1:
                        self.monomer_1_bond_list.append(site_2 + 1)
                else:  # terminal diene -> preserve the other side
                    if number_of_hydrogen_at_site_1 == 2:
                        self.monomer_1_bond_list.append(site_1)
                    elif number_of_hydrogen_at_site_2 == 2:
                        self.monomer_1_bond_list.append(site_2)

        # (2) conjugated dibromide - GRIM
        elif self.mechanism['metathesis'] == ['conjugated_di_bromide']:
            for site_1, site_2 in self.reaction_sites['monomer_1']:
                site_1_atom = self.monomer_1_mol.GetAtomWithIdx(site_1).GetSymbol()
                site_2_atom = self.monomer_1_mol.GetAtomWithIdx(site_2).GetSymbol()
                if site_1_atom == 'C':
                    self.monomer_1_bond_list.append(site_1)
                    self.monomer_1_del_list.append(site_2)
                elif site_2_atom == 'C':
                    self.monomer_1_bond_list.append(site_2)
                    self.monomer_1_del_list.append(site_1)

    def react(self):
        # remove atom - sort monomer_1_del_list in order not to affect other elements
        self.monomer_1_del_list.sort(reverse=True)
        for del_idx in self.monomer_1_del_list:
            self.mw_1.RemoveAtom(del_idx)
            for j in range(len(self.monomer_1_bond_list)):
                if self.monomer_1_bond_list[j] > del_idx:
                    self.monomer_1_bond_list[j] -= 1

        # get wildcard id
        wildcard_id_1 = self.mw_1.AddAtom(Atom('*'))
        wildcard_id_2 = self.mw_1.AddAtom(Atom('*'))

        # add wildcard atoms
        add_idx_1 = self.monomer_1_bond_list[0]
        add_idx_2 = self.monomer_1_bond_list[1]

        # (1) ADMET (terminal diene)
        if self.mechanism['metathesis'] == ['terminal_diene']:
            self.mw_1.AddBond(add_idx_1, wildcard_id_1, BondType.SINGLE)
            self.mw_1.AddBond(add_idx_2, wildcard_id_2, BondType.SINGLE)

        # (2) GRIM (conjugated dibromide)
        elif self.mechanism['metathesis'] == ['conjugated_di_bromide']:
            self.mw_1.AddBond(add_idx_1, wildcard_id_1, BondType.SINGLE)
            self.mw_1.AddBond(add_idx_2, wildcard_id_2, BondType.SINGLE)

        # convert to smiles
        self.repeating_unit_smiles = Chem.MolToSmiles(self.mw_1)

        return self.repeating_unit_smiles, self.mechanism
