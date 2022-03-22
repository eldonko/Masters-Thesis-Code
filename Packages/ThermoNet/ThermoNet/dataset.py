import pkg_resources

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sgte.handler import SGTEHandler
from thermoclassifier.dataset.encoding import Encoder


class ThermoDataset(Dataset):
    """
    The dataset for training and testing the thermonet

    Parameters
    ----------
    element : str
        element for which the dataset is created
    phase : str
        phase of the element for which the dataset is created
    start_temp : int
        lower value of temperature range (Default value = 200)
    end_temp : int
        upper value of temperature range (Default value = 2000)
    step : float
        step size of the SGTEHandler to create the data. The smaller step, the larger the dataset (Default value = 1.)
    """
    def __init__(self, element, phase, start_temp=200, end_temp=2000, step=1.):
        super(ThermoDataset, self).__init__()

        sgte_handler = SGTEHandler(element)
        sgte_handler.evaluate_equations(start_temp, end_temp, 1e5, plot=False, phases=phase, entropy=True,
                                        enthalpy=True,
                                        heat_capacity=True, step=step)
        data = sgte_handler.equation_result_data

        # Get values
        self.temp = torch.tensor(data['Temperature'], dtype=torch.float64)
        data.iloc[:, 1] /= 1000
        data.iloc[:, 3] /= 1000

        self.targets = torch.Tensor(data.iloc[:, 1:].values)

        self._samples = list(zip(self.temp, self.targets))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i: int):
        return self._samples[i]

    def get_data(self):
        """
        Returns
        -------
        tuple of torch.tensors
            (temperature, target values) where target values are the Gibbs energy [kJ], the entropy [J/K], the enthalpy
            [kJ] and the heat capacity [J/K].
        """
        return self.temp.unsqueeze(-1), self.targets


class ThermoDatasetNew(Dataset):
    """
    The dataset for training and testing the thermonet

    Parameters
    ----------
    elements : list or None
        element for which the dataset is created
    start_temp : int
        lower value of temperature range (Default value = 200)
    end_temp : int
        upper value of temperature range (Default value = 2000)
    step : float
        step size of the SGTEHandler to create the data. The smaller step, the larger the dataset (Default value = 1.)
    inp_phases : list or None
        if None, all phases of an element are considered. Can only be list if elements contains only one element
    """

    def __init__(self, elements, start_temp=200, end_temp=2000, step=1., inp_phases=None):
        super(ThermoDatasetNew, self).__init__()

        # Input checking
        assert inp_phases is None or type(inp_phases) == list
        if type(inp_phases) == list and (elements is None or (type(elements) == list and len(elements) != 1)):
            raise ValueError('phases can only be list if elements is a list of exactly one element')

        # Initialize dataset samples
        self._samples = np.empty(shape=(0, 7))

        # Load the element phase data
        element_phase_filename = r'./data/Phases.xlsx'
        stream = pkg_resources.resource_stream(__name__, element_phase_filename)
        self.element_phase_data = pd.read_excel(stream, sheet_name='Phases').set_index('Index')

        # If only certain phases shall be selected, then just retrieve those phases from element_phase_data
        if elements is None:
            elements = self.element_phase_data.columns.values
        else:
            self.element_phase_data = self.element_phase_data[elements]

        # Loop through the elements and load the data
        for element in elements:
            if inp_phases is None:  # Use all phases in this case
                phases = self.get_element_phases(element)
            else:
                phases = inp_phases

            sgte_handler = SGTEHandler(element)
            sgte_handler.evaluate_equations(start_temp, end_temp, 1e5, plot=False, phases=phases, entropy=True,
                                            enthalpy=True,
                                            heat_capacity=True, step=step)
            data = sgte_handler.equation_result_data

            # Get the data by phase
            data = data.set_index('Temperature')
            data_by_phase = self.split_data_by_phase(data, element)

            # Add to samples
            self._samples = np.concatenate((self._samples, data_by_phase), 0)

    def __getitem__(self, i):
        return self._samples[i]

    def __len__(self):
        return len(self._samples)

    def get_data(self):
        return self._samples

    def get_element_phases(self, element):
        """
        Retrieves the phases that exist (in the SGTE data) for a given element from the provided excel sheet

        Parameters
        ----------
        element : str
            element to load the phases for

        Returns
        -------
        list :
            list of strings for the phases that exist for a given element
        """

        return self.element_phase_data[element][self.element_phase_data[element] == 1].index.tolist()

    @staticmethod
    def split_data_by_phase(data, element):
        """
        Splits data so that for every phase in data, an array of temperature, Gibbs energy, entropy, enthalpy,
        heat capacity and the element label is returned

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the data from the solved SGTE equations for all phases and functions
        element : str
            element name

        Returns
        -------
        list :
            list of array of temperature, Gibbs energy, entropy, enthalpy, heat capacity, phase label and
            element label
        """
        encoder = Encoder()

        # Encode the element
        element_enc = encoder(element)

        phase_data_np_total = np.empty(shape=(0, 7))

        # Slice data so that the 4 functions for each phase can be sliced out
        for i in range(0, len(data.columns.values), 4):
            # Get the data for only one phase
            phase_data = data[data.columns.values[i:i + 4]]

            # Encode the phase name
            phase_enc = encoder(pd.Series(phase_data.columns.values[0]))[0]

            # Convert the phase data the a numpy array
            phase_data = phase_data.reset_index()
            phase_data_np = np.array(phase_data)

            # Convert Gibbs energy and enthalpy to kJ for numerical reasons
            phase_data_np[:, 1] /= 1000
            phase_data_np[:, 3] /= 1000

            # Add element and phase labels
            labels = np.ones(shape=(phase_data_np.shape[0], 2))
            labels[:, 0] *= element_enc
            labels[:, 1] *= phase_enc
            phase_data_np = np.concatenate((phase_data_np, labels), -1)

            # Stack the data to the total data
            phase_data_np_total = np.concatenate((phase_data_np_total, phase_data_np), 0)

        return phase_data_np_total
