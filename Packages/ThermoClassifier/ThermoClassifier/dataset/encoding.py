import pkg_resources

import pandas as pd
import numpy as np


class Encoder(object):
    def __init__(self):
        super(Encoder, self).__init__()

        # Load the element-phase excel sheet to get all phase and to create the labels
        element_phase_filename = r'../data/Phases.xlsx'
        stream = pkg_resources.resource_stream(__name__, element_phase_filename)
        element_phase_data = pd.read_excel(stream, sheet_name='Phases')
        self.phases = element_phase_data['Index']
        self.elements = element_phase_data.columns.values

    def __call__(self, inp):
        # If input is a Series of strings, encode the Series
        if type(inp) == pd.Series and type(inp[0]):
            return self.encode_phase(inp)
        # If input is a Series of integers, decode the Series
        elif type(inp) == pd.Series and type(inp[0]) == np.int32:
            return self.decode_phase(inp)
        elif type(inp) == str:
            return self.encode_element(inp)
        elif type(inp) == int:
            return self.decode_element(inp)

    def encode_phase(self, inp):
        """
        Encodes element phases from name to integer label

        Parameters
        ----------
        inp : pd.Series
            Series containing strings (decoded phases)

        Returns
        -------
        pd.Series
            Encoded phases as pd.Series

        """
        # Remove the first part of the string as this indicates the measurement and is not needed
        to_remove = inp[0].split('_')[0] + '_'

        # Remove the measurement indicator from the Series
        inp = inp.str.replace(to_remove, '', regex=False).astype('category')

        # Rename categories based on index of category (i.e., a phase) in self.phases and return
        new_cats = {}
        for cat in inp.cat.categories:
            new_cats[cat] = self.phases[self.phases == cat].index[0]

        return inp.cat.rename_categories(new_cats).astype('int')

    def decode_phase(self, inp):
        """
        Decodes element phase from integer label to name

        Parameters
        ----------
        inp : pd.Series
            Series containing integers (encoded phases)

        Returns
        -------
        pd.Series
            Decoded phases as pd.Series

        """
        inp = inp.astype('category')

        # Rename categories based on name of category (i.e., a phase) in self.phases and return
        new_cats = {}
        for cat in inp.cat.categories:
            new_cats[cat] = self.phases.iloc[cat]

        return inp.cat.rename_categories(new_cats).astype('str')

    def encode_element(self, inp):
        """
        Encodes element from name to integer label

        Parameters
        ----------
        inp : str
            Element name

        Returns
        -------
        int
            Element label
        """

        return np.where(self.elements == inp)[0].item() - 1

    def decode_element(self, inp):
        """
        Decodes element from integer label to name

        Parameters
        ----------
        inp : int
            Element label

        Returns
        -------
        str
            Element name
        """

        return self.elements[inp + 1]

