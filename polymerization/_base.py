import importlib


class BasePolymerization(object):
    def __init__(self):
        # SMART annotation
        self._sub_structure_dict = {
            'acetylene': '[CX2]#[CX2]',
            'di_acid_chloride': '[CX3](=O)[Cl]',
            'conjugated_di_bromide': '[c;R][Br]',
            'cyclic_carbonate': '[OX1]=[CX3;R]([OX2;R][C;R])[OX2;R][C;R]',
            'cyclic_ether': '[C;R][O;R]([C;R])',  # '[OX2;R]([CX2;R][C;R])[CX2;R][C;R]'
            'cyclic_olefin': '[CH1;R][CH1;R]=[CH1;R][CH1;R]',
            'cyclic_sulfide': '[C;R][S;R]([C;R])',
            'di_amine': '[NX3H2;!$(NC=O)]',
            'di_carboxylic_acid': '[CX3](=O)[OX2H]',
            'di_isocyanate': '[NX2]=[CX2]=[OX1]',
            'di_ol': '[C,c;!$(C=O)][OX2H1]',
            'hydroxy_carboxylic_acid_OH': '[!$(C=O)][OX2H1]',
            'hydroxy_carboxylic_acid_COOH': '[CX3](=O)[OX2H]',
            'lactam': '[NH1;R][C;R](=O)',
            'lactone': '[O;R][C;R](=O)',
            'terminal_diene': '[CX3H2]=[CX3H1]',
            'vinyl': '[CX3;!R]=[CX3]'
        }
        # chain_growth, step_growth, ring opening (chain_growth), and metathesis -> alphabetically ordered
        # ADMET and GRIM in methathesis refer to Acyclic Diene METathesis and GRIgnard Metathesis, respectively
        self._predefined_mechanism = {
            'step_growth': [['di_amine', 'di_carboxylic_acid'], ['di_acid_chloride', 'di_amine'],
                            ['di_carboxylic_acid', 'di_ol'], ['di_acid_chloride', 'di_ol'],
                            ['di_amine', 'di_isocyanate'], ['di_isocyanate', 'di_ol'], ['hydroxy_carboxylic_acid']],
            'chain_growth': [['vinyl'], ['acetylene']],
            'chain_growth_ring_opening': [['lactone'], ['lactam'], ['cyclic_ether'], ['cyclic_olefin'],
                                          ['cyclic_carbonate'], ['cyclic_sulfide']],
            'metathesis': [['terminal_diene'], ['conjugated_di_bromide']]
        }

    @ staticmethod
    def call_polymerization_reactor(reaction_mechanism: str):
        # match reaction_mechanism and reaction_reactor class name
        reactor_dict = {
            'step_growth': 'StepGrowthReactor',
            'chain_growth': 'ChainGrowthReactor',
            'chain_growth_ring_opening': 'ChainGrowthRingOpeningReactor',
            'metathesis': 'MetathesisReactor'
        }
        # check if an inserted mechanism is supported or not
        if reaction_mechanism not in reactor_dict.keys():
            print('Inserted %s is not currently supported' % reaction_mechanism, flush=True)
            exit()
        # call reactor class
        reactor_class_name = reactor_dict[reaction_mechanism]
        reactor_class = getattr(importlib.import_module('polymerization'), reactor_class_name)

        return reactor_class

    @ staticmethod
    def remove_atoms_and_relabel(monomer, del_list, bnd_list):
        # reverse sort del_list, so we can delete without affecting other elements of del_list
        arr = del_list.copy()
        arr.sort(reverse=True)
        for del_id in arr:
            monomer.RemoveAtom(del_id)
            # modify bnd_list idx
            for j in range(len(bnd_list)):
                if bnd_list[j] > del_id:
                    bnd_list[j] -= 1
        return monomer, bnd_list
