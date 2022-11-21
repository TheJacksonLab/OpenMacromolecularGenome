import pandas as pd

from rdkit import Chem

from ._base import BasePolymerization


class Polymerization(BasePolymerization):
    def __init__(self):
        super(Polymerization, self).__init__()
        self.find_mechanism = False
        self._mechanism = None
        self._reaction_sites = None
        self._reaction_groups = None

    # find a proper polymerization mechanism
    def _search_mechanism(self, monomers_bag: list or tuple):
        # set default value
        self.find_mechanism = False

        # change monomers to mol objects
        mol = [Chem.MolFromSmiles(smiles) for smiles in monomers_bag]
        if None in mol:
            # print("[POLYMER] There is a chemically invalid SMILES in a monomers bag", flush=True)
            return

        # find functional group components
        df_sub_structure = pd.DataFrame({'smiles': monomers_bag, 'mol': mol})
        for key, value in self._sub_structure_dict.items():
            sub_structure_mol = Chem.MolFromSmarts(value)
            df_sub_structure['%s' % key] = df_sub_structure['mol'].apply(
                lambda x: len(x.GetSubstructMatches(sub_structure_mol))
            )

        # hydroxy_carboxylic_acid should have both OH and COOH
        target = ['hydroxy_carboxylic_acid_OH', 'hydroxy_carboxylic_acid_COOH']
        df_sub_structure['hydroxy_carboxylic_acid'] = df_sub_structure[target[0]] * df_sub_structure[target[1]]
        df_sub_structure = df_sub_structure.drop(labels=[target[0], target[1]], axis=1)
        # deal with exception
        for col in df_sub_structure.columns:
            for row in df_sub_structure.index.values:
                # reduce di_* value to 0 if the value is 1. di_* value should be 2
                if 'di' in col and df_sub_structure.loc[row, col] == 1:
                    df_sub_structure.loc[row, col] = 0
                # reduce di_* value to 0 if the value is larger than 2.
                if 'di' in col and df_sub_structure.loc[row, col] >= 3:
                    df_sub_structure.loc[row, col] = 0
                # hydroxy_carboxylic_acid should have only one pair of OH and COOH
                if col == 'hydroxy_carboxylic_acid' and df_sub_structure.loc[row, col] >= 2:
                    # print("[POLYMER] There is more than one hydroxy_carboxylic group in a monomer", flush=True)
                    return
                # if there are both vinyl group and cyclic olefin -> cyclic olefin has a priority (arbitrary)
                if col == 'vinyl' and df_sub_structure.loc[row, col] == 1 and \
                        df_sub_structure.loc[row, 'cyclic_olefin'] == 1:
                    df_sub_structure.loc[row, col] = 0
                # if there are both vinyl group and terminal diene -> terminal diene has a priority (arbitrary)
                if col == 'vinyl' and df_sub_structure.loc[row, col] == 2 and \
                        df_sub_structure.loc[row, 'terminal_diene'] == 2:
                    df_sub_structure.loc[row, col] = 0
                # number of vinyl functional group should be 1
                if col == 'vinyl' and df_sub_structure.loc[row, col] >= 2:
                    # print("[POLYMER] There is more than one vinyl group in a monomer", flush=True)
                    return
                # number of acetylene functional group should be 1
                if col == 'acetylene' and df_sub_structure.loc[row, col] >= 2:
                    # print("[POLYMER] There is more than one acetylene group in a monomer", flush=True)
                    return
                # if there are both cyclic ether and lactone -> lactone has a priority (arbitrary)
                if col == 'cyclic_ether' and df_sub_structure.loc[row, col] == 1 and \
                    df_sub_structure.loc[row, 'lactone'] == 1:
                    df_sub_structure.loc[row, col] = 0
                # if there are both cyclic ether and cyclic_carbonate -> cyclic carbonate has a priority (arbitrary)
                if col == 'cyclic_ether' and df_sub_structure.loc[row, col] == 2 and \
                    df_sub_structure.loc[row, 'cyclic_carbonate'] == 1:
                    df_sub_structure.loc[row, col] = 0
                # if there are both lactone and cyclic_carbonate -> cyclic carbonate has a priority (arbitrary)
                if col == 'lactone' and df_sub_structure.loc[row, col] == 2 and \
                    df_sub_structure.loc[row, 'cyclic_carbonate'] == 1:
                    df_sub_structure.loc[row, col] = 0
                # count the number of atoms in a ring of cyclic ether and cyclic sulfide
                # -> only 3, 4, and larger than 6 are allowed
                if (col == 'cyclic_ether' or col == 'cyclic_sulfide') and df_sub_structure.loc[row, col] >= 1:
                    cyclic_smiles = df_sub_structure.loc[row, 'mol']
                    ring_info = cyclic_smiles.GetRingInfo()
                    for ring_cluster in ring_info.AtomRings():
                        ring_idx = ring_cluster[0]
                        min_num_atoms_in_ring = ring_info.MinAtomRingSize(ring_idx)
                        if min_num_atoms_in_ring in (5, 6):
                            df_sub_structure.loc[row, col] = 0
                # number of functional group should be 1
                if 'di' not in col and col != 'smiles' and col != 'mol' and df_sub_structure.loc[row, col] >= 2:
                    # print("[POLYMER] There is more than one cyclic functional group in a monomer", flush=True)
                    return
  
        # decide if there are more than one functional groups in a molecule
        reaction_sites = dict()
        reaction_groups = dict()
        reaction_monomers = dict()
        for row_idx in range(df_sub_structure.shape[0]):
            count = 0
            for col in df_sub_structure.columns[2:]:  # exclude 'smiles' and 'mol' columns
                # not self-condensation of hydrocarboxylic acid
                if col != 'hydroxy_carboxylic_acid' and df_sub_structure.iloc[row_idx][col] != 0:
                    count += 1
                    # find react sites and reaction groups in SMILES molecules
                    reaction_groups['monomer_%d' % (row_idx + 1)] = col
                    reaction_monomers['monomer_%d' % (row_idx + 1)] = df_sub_structure.iloc[row_idx]['smiles']
                    reaction_sites['monomer_%d' % (row_idx + 1)] = df_sub_structure.iloc[row_idx]['mol']. \
                        GetSubstructMatches(Chem.MolFromSmarts(self._sub_structure_dict[col]))

                # self-condensation of hydroxycarboxylic acid
                elif col == 'hydroxy_carboxylic_acid' and df_sub_structure.iloc[row_idx][col] != 0:
                    count += 1
                    reaction_groups['monomer_%d' % (row_idx + 1)] = col
                    reaction_monomers['monomer_%d' % (row_idx + 1)] = df_sub_structure.iloc[row_idx]['smiles']
                    # add reaction sites of self-condensation of hyroxycarboxylic acid
                    sites = ()
                    for key, value in self._sub_structure_dict.items():
                        if key.startswith(col):
                            sites += df_sub_structure.iloc[row_idx]['mol'].\
                                GetSubstructMatches(Chem.MolFromSmarts(value))
                            reaction_sites['monomer_%d' % (row_idx + 1)] = sites

            # check if there are more than two functional groups
            if count >= 2:
                # print("[POLYMER] There is more than one functional group in a monomer", flush=True)
                return
            # check if there is no functional groups
            elif count == 0:
                # print("[POLYMER] There is a monomer that doesn't have a functional group", flush=True)
                return

        # find a possible polymerization mechanism
        reaction_list = list(reaction_groups.values())
        reaction_list.sort()
        flag = 0
        for mechanism, reactions in self._predefined_mechanism.items():
            for reaction in reactions:
                sorted_reaction = reaction.copy()
                sorted_reaction.sort()
                if sorted_reaction == reaction_list:
                    flag = 1
                    self._mechanism = {'%s' % mechanism: sorted_reaction}
        # store values
        self.find_mechanism = True if flag == 1 else False
        self._reaction_sites = reaction_sites
        self._reaction_groups = reaction_groups
        self._reaction_monomers = reaction_monomers

    def polymerize(self, monomers_bag):
        # search polymerization mechanism
        self._search_mechanism(monomers_bag)
        # check if a polymerization mechanism was found
        if not self.find_mechanism:
            # print('Failed to find a polymerization mechanism', flush=True)
            return None
        # classify mechanism - step_growth, chain_growth, chain_growth_ring_opening, or metathesis
        mechanism = list(self._mechanism.keys())[0]
        reactor = self.call_polymerization_reactor(mechanism)(
            reaction_monomers=self._reaction_monomers,
            reaction_groups=self._reaction_groups,
            reaction_sites=self._reaction_sites,
            mechanism=self._mechanism
        )
        product, mechanism = reactor.react()

        return product, mechanism
