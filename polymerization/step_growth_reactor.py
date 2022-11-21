from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.rdchem import BondType, Atom

from ._base import BasePolymerization


class StepGrowthReactor(BasePolymerization):
    def __init__(self, reaction_monomers, reaction_groups, reaction_sites, mechanism):
        super(StepGrowthReactor, self).__init__()
        self.reaction_monomers = reaction_monomers
        self.reaction_groups = reaction_groups
        self.reaction_sites = reaction_sites
        self.mechanism = mechanism

        # for self-condensation of 'hydroxy_carboxylic_acid'
        if self.reaction_groups['monomer_1'] != 'hydroxy_carboxylic_acid':
            # get monomer_1_key and monomer_2_key ('monomer_1' or 'monomer_2')
            # match with alphabetically ordered predefined_mechanism
            self.monomer_1_key = [key for key, value in self.reaction_groups.items() if
                                  value == self.mechanism['step_growth'][0]][0]
            self.monomer_2_key = [key for key, value in self.reaction_groups.items() if
                                  value == self.mechanism['step_growth'][1]][0]

        elif self.reaction_groups['monomer_1'] == 'hydroxy_carboxylic_acid':
            [self.monomer_1_key, self.monomer_2_key] = ['monomer_1', 'monomer_1']

        # find smiles and reactions sites for monomer_1 and monomer_2
        self.monomer_1_smiles = self.reaction_monomers[self.monomer_1_key]
        self.monomer_1_mol = Chem.MolFromSmiles(self.monomer_1_smiles)
        self.monomer_1_reaction_sites = self.reaction_sites[self.monomer_1_key]

        self.monomer_2_smiles = self.reaction_monomers[self.monomer_2_key]
        self.monomer_2_mol = Chem.MolFromSmiles(self.monomer_2_smiles)
        self.monomer_2_reaction_sites = self.reaction_sites[self.monomer_2_key]

        # set rewritable mol object
        self.mw_1 = RWMol(self.monomer_1_mol)
        self.mw_2 = RWMol(self.monomer_2_mol)

        # find monomer_1_del_list, monomer_1_bond_list, monomer_2_del_list, and monomer_2_bond_list
        self.monomer_1_del_list = []
        self.monomer_1_bond_list = []
        self.monomer_2_del_list = []
        self.monomer_2_bond_list = []
        self.get_del_bond_list()

        # set new monomer and repeating unit
        self.new_monomer = None
        self.repeating_unit_smiles = None

    def get_del_bond_list(self):
        # (1) di_amine and di_carboxylic_acid
        if self.mechanism['step_growth'] == ['di_amine', 'di_carboxylic_acid']:
            for reaction_positions in self.monomer_1_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_1_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == '':
                        self.monomer_1_del_list.append(idx)
                    elif symbol == 'N':
                        self.monomer_1_bond_list.append(idx)

            for reaction_positions in self.monomer_2_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_2_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == 'O' and self.monomer_2_mol.GetAtomWithIdx(idx).GetTotalNumHs() == 1:
                        self.monomer_2_del_list.append(idx)
                    elif symbol == 'C':
                        self.monomer_2_bond_list.append(idx)

        # (2) di_acid_chloride and di_amine
        elif self.mechanism['step_growth'] == ['di_acid_chloride', 'di_amine']:
            for reaction_positions in self.monomer_1_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_1_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == 'Cl':
                        self.monomer_1_del_list.append(idx)
                    elif symbol == 'C':
                        self.monomer_1_bond_list.append(idx)

            for reaction_positions in self.monomer_2_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_2_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == '':
                        self.monomer_2_del_list.append(idx)
                    elif symbol == 'N':
                        self.monomer_2_bond_list.append(idx)

        # (3) di_carboxylic_acid and di_ol
        elif self.mechanism['step_growth'] == ['di_carboxylic_acid', 'di_ol']:
            for reaction_positions in self.monomer_1_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_1_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == 'O' and self.monomer_1_mol.GetAtomWithIdx(idx).GetTotalNumHs() == 1:
                        self.monomer_1_del_list.append(idx)
                    elif symbol == 'C':
                        self.monomer_1_bond_list.append(idx)

            for reaction_positions in self.monomer_2_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_2_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == '':
                        self.monomer_2_del_list.append(idx)
                    elif symbol == 'O' and self.monomer_2_mol.GetAtomWithIdx(idx).GetTotalNumHs() == 1:
                        self.monomer_2_bond_list.append(idx)

        # (4) di_acid_chloride and di_ol
        elif self.mechanism['step_growth'] == ['di_acid_chloride', 'di_ol']:
            for reaction_positions in self.monomer_1_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_1_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == 'Cl':
                        self.monomer_1_del_list.append(idx)
                    elif symbol == 'C':
                        self.monomer_1_bond_list.append(idx)

            for reaction_positions in self.monomer_2_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_2_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == '':
                        self.monomer_2_del_list.append(idx)
                    elif symbol == 'O' and self.monomer_2_mol.GetAtomWithIdx(idx).GetTotalNumHs() == 1:
                        self.monomer_2_bond_list.append(idx)

        # (5) di_amine and di_isocyanate
        elif self.mechanism['step_growth'] == ['di_amine', 'di_isocyanate']:
            # down-convert nitrogen double bond to single bond
            nitrogen_site = []
            carbon_site = []
            for reaction_positions in self.monomer_1_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_1_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == '':
                        self.monomer_1_del_list.append(idx)
                    elif symbol == 'N':
                        self.monomer_1_bond_list.append(idx)

            for reaction_positions in self.monomer_2_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_2_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == '':
                        self.monomer_2_del_list.append(idx)
                    elif symbol == 'C':
                        self.monomer_2_bond_list.append(idx)
                        carbon_site.append(idx)
                    elif symbol == 'N':
                        nitrogen_site.append(idx)
            # down-convert
            for nitrogen_idx, carbon_idx in zip(nitrogen_site, carbon_site):
                self.mw_2.GetBondBetweenAtoms(
                    nitrogen_idx,
                    carbon_idx
                ).SetBondType(BondType.SINGLE)

        # (6) di_isocyanate and di_ol
        elif self.mechanism['step_growth'] == ['di_isocyanate', 'di_ol']:
            # down-convert nitrogen double bond to single bond
            nitrogen_site = []
            carbon_site = []
            for reaction_positions in self.monomer_1_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_1_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == '':
                        self.monomer_1_del_list.append(idx)
                    elif symbol == 'C':
                        self.monomer_1_bond_list.append(idx)
                        carbon_site.append(idx)
                    elif symbol == 'N':
                        nitrogen_site.append(idx)
            # down-convert
            for nitrogen_idx, carbon_idx in zip(nitrogen_site, carbon_site):
                self.mw_1.GetBondBetweenAtoms(
                    nitrogen_idx,
                    carbon_idx
                ).SetBondType(BondType.SINGLE)

            for reaction_positions in self.monomer_2_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_2_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == '':
                        self.monomer_2_del_list.append(idx)
                    elif symbol == 'O' and self.monomer_2_mol.GetAtomWithIdx(idx).GetTotalNumHs() == 1:
                        self.monomer_2_bond_list.append(idx)

        # (7) self_condensation of hydroxy_carboxylic_acid
        elif self.mechanism['step_growth'] == ['hydroxy_carboxylic_acid']:
            for reaction_positions in self.monomer_1_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_1_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == 'C' and len(reaction_positions) == 3:  # COOH
                        self.monomer_1_bond_list.append(idx)
                    elif symbol == 'O' and self.monomer_1_mol.GetAtomWithIdx(idx).GetTotalNumHs() == 1 and \
                            len(reaction_positions) == 3:  # COOH
                        self.monomer_1_del_list.append(idx)
                    elif symbol == 'O' and self.monomer_1_mol.GetAtomWithIdx(idx).GetTotalNumHs() == 1:
                        self.monomer_1_bond_list.append(idx)

            for reaction_positions in self.monomer_2_reaction_sites:
                for idx in reaction_positions:
                    symbol = self.monomer_2_mol.GetAtomWithIdx(idx).GetSymbol()
                    if symbol == 'C' and len(reaction_positions) == 3:  # COOH
                        self.monomer_2_bond_list.append(idx)
                    elif symbol == 'O' and self.monomer_2_mol.GetAtomWithIdx(idx).GetTotalNumHs() == 1 and \
                            len(reaction_positions) == 3:  # COOH
                        self.monomer_2_del_list.append(idx)
                    elif symbol == 'O' and self.monomer_2_mol.GetAtomWithIdx(idx).GetTotalNumHs() == 1:
                        self.monomer_2_bond_list.append(idx)

    def react(self):
        # Use modified_monomer_1, modified_monomer_2
        modified_monomer_1, modified_monomer_1_bnd_list = self.remove_atoms_and_relabel(
            self.mw_1, self.monomer_1_del_list, self.monomer_1_bond_list
        )
        modified_monomer_2, modified_monomer_2_bnd_list = self.remove_atoms_and_relabel(
            self.mw_2, self.monomer_2_del_list, self.monomer_2_bond_list
        )
        # modify index of monomers - after monomer 1
        modified_monomer_2_bnd_list = [idx + modified_monomer_1.GetNumAtoms()
                                       for idx in modified_monomer_2_bnd_list]
        # make new monomer and add bond
        self.new_monomer = RWMol(Chem.CombineMols(modified_monomer_1, modified_monomer_2))
        self.new_monomer.AddBond(modified_monomer_1_bnd_list[1], modified_monomer_2_bnd_list[0], BondType.SINGLE)

        # add wildcard atoms
        wildcard_1_id = self.new_monomer.AddAtom(Atom('*'))
        self.new_monomer.AddBond(modified_monomer_1_bnd_list[0], wildcard_1_id, BondType.SINGLE)
        wildcard_2_id = self.new_monomer.AddAtom(Atom('*'))
        self.new_monomer.AddBond(modified_monomer_2_bnd_list[1], wildcard_2_id, BondType.SINGLE)

        # remove hydrogen
        self.repeating_unit_smiles = Chem.MolToSmiles(self.new_monomer)

        return self.repeating_unit_smiles, self.mechanism

