from rdkit.Chem.rdchem import BondType, HybridizationType


class AtomFeatureParams(object):
    def __init__(self):
        self.atom_types = ['Ag', 'Al', 'Ar', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Br', 'C',
                    'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Eu', 'F',
                    'Fe', 'Ga', 'Ge', 'H', 'He', 'Hf', 'Hg', 'I', 'In', 'Ir', 'K', 'La',
                    'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nd', 'Ni', 'O', 'Os', 'P', 'Pb',
                    'Pd', 'Pr', 'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se',
                    'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Te', 'Ti', 'Tl', 'V', 'W', 'Xe', 'Y',
                    'Yb', 'Zn', 'Zr']
        #     [
        #     'Br', 'B', 'C', 'O', 'S', 'N', 'I', 'P', 'Mg', 'Cl', 'F', 'Si', 'Sn', 'Zn', 'Al', 'K', 'Na', 'Cu', 'Cr',
        #     'Mn', 'Se', 'Li'
        # ]
        # self.bond_dict = {
        #     BondType.AROMATIC: 'aromatic',
        #     BondType.SINGLE: 'single',
        #     BondType.DOUBLE: 'double',
        #     BondType.TRIPLE: 'triple'
        # }
        self.bond_dict = {
            BondType.SINGLE: 'single',
            BondType.DOUBLE: 'double',
            BondType.TRIPLE: 'triple'
        }
        self.degrees = [0., 1., 2., 3., 4., 5., 6., 7., 10.]
        self.explicit_valences = [0., 1., 2., 3., 4., 5., 6., 7., 8., 10., 12., 14.]
        # self.degrees = [0., 1., 2., 3., 4., 5.]
        # self.explicit_valences = [0., 1., 2., 3., 4., 5., 6., 7.]
        # self.hybridization = [
        #     HybridizationType.S,
        #     HybridizationType.SP,
        #     HybridizationType.SP2,
        #     HybridizationType.SP3,
        #     HybridizationType.SP3D,
        #     HybridizationType.SP3D2
        # ]
        self.hybridization = [
            HybridizationType.SP,
            HybridizationType.SP2,
            HybridizationType.SP3,
            0
        ]
        # self.bond_names = ['aromatic', 'single', 'double', 'triple']
        self.bond_names = ['single', 'double', 'triple']
        self.num_bond_types = len(self.bond_names)
        # self.atom_feature_length = len(self.atom_types) + len(self.degrees) + len(self.explicit_valences) + \
        #     len(self.hybridization) + len(
        #     ['electron_negativity', 'atomic_number', 'hydrogen_number', 'aromaticity', 'formal_charge']
        # )
        self.atom_feature_length = len(self.atom_types) + len(self.degrees) + len(self.explicit_valences) + \
            len(self.hybridization) + len(
            ['electron_negativity', 'atomic_number', 'hydrogen_number', 'aromaticity']
        )

    def get_bond_name(self, bond):
        bond_type = bond.GetBondType()
        return self.bond_dict[bond_type]
