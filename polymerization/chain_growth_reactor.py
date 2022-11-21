import copy

from rdkit import Chem
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem.rdchem import BondType, Atom

from ._base import BasePolymerization


class ChainGrowthReactor(BasePolymerization):
    def __init__(self, reaction_monomers, reaction_groups, reaction_sites, mechanism):
        super(ChainGrowthReactor, self).__init__()
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

        # down-convert bond for chain growth
        if self.mechanism['chain_growth'] == ['vinyl']:
            self.new_monomer.GetBondBetweenAtoms(
                self.monomer_1_reaction_sites[0][0],
                self.monomer_1_reaction_sites[0][1]
            ).SetBondType(BondType.SINGLE)
        elif self.mechanism['chain_growth'] == ['acetylene']:
            self.new_monomer.GetBondBetweenAtoms(
                self.monomer_1_reaction_sites[0][0],
                self.monomer_1_reaction_sites[0][1]
            ).SetBondType(BondType.DOUBLE)

        # set rewritable mol object
        self.mw_1 = RWMol(self.new_monomer)

    def react(self):
        # set wildcard_id
        wildcard_id_1 = self.mw_1.AddAtom(Atom('*'))
        wildcard_id_2 = self.mw_1.AddAtom(Atom('*'))

        # draw single bond
        self.mw_1.AddBond(self.monomer_1_reaction_sites[0][0], wildcard_id_1, BondType.SINGLE)
        self.mw_1.AddBond(self.monomer_1_reaction_sites[0][1], wildcard_id_2, BondType.SINGLE)

        # convert to smiles
        self.repeating_unit_smiles = Chem.MolToSmiles(self.mw_1)

        return self.repeating_unit_smiles, self.mechanism
