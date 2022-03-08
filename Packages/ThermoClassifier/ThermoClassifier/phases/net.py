import os
import pkg_resources

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Softmax, LSTM, Sequential, BatchNorm1d, Tanh
from torch.utils.data import DataLoader
import pandas as pd


class PhaseClassifier(nn.Module):
    """
	phases classifies thermodynamic measurement data in the sense that given measurement data for the
	Gibbs energy, entropy, enthalpy or the heat capacity at certain temperatures, it is able to tell which phase of
	a given element this measurement data belongs to.
	"""

    def __init__(self, element, train=False, measurement='G', hidden_layers=1, hidden_size=32):
        """

        Parameters
        ----------

		element: str :
		    Which element the measurements are from (and the phase should be predicted for)
		train: bool :
		    If train is True, a network with random weights is generated and trained, if train is False, an
		    already trained network for this element and measurement type will be loaded. (Default value = False)
		measurement: str :
		    Defines for which properties the classification should be made. Must be one of 'G', 'S', 'H'
		    or 'C'. (Default value = 'G')
		hidden_layers: int :
		    number of hidden layers
		hidden_size: int :
		    number of nodes in the hidden layers
		"""
        super(PhaseClassifier, self).__init__()

        # Input checking
        assert measurement in ['G', 'S', 'H', 'C']
        self.measurement = measurement

        self.train = train

        # File paths
        self.element_data_path = r'../data/Phases.xlsx'
        self.models_path = r'./models/new/'

        # Load element data
        self.element_data = None
        self.load_element_data()
        assert element in self.element_data.columns.values
        self.element = element

        # Fully connected net for classification
        self.num_classes = self.element_data[element].sum()
        self.in_features = 2
        self.hidden_size_linear = hidden_size

        # For training, create a new network, else load a pre-trained network
        def create_net():
            """
            fc = Sequential(
                Linear(self.in_features, self.hidden_size_linear),
                ReLU(),
                Linear(self.hidden_size_linear, self.hidden_size_linear),
                ReLU(),
                # Linear(self.hidden_size_linear, self.hidden_size_linear),
                # Tanh(),
                Linear(self.hidden_size_linear, self.num_classes)
            )"""

            fc = Sequential()
            fc.add_module('in', Linear(self.in_features, self.hidden_size_linear))
            fc.add_module('a_in', ReLU())
            for i in range(hidden_layers):
                fc.add_module('h_' + str(i + 1), Linear(self.hidden_size_linear, self.hidden_size_linear))
                fc.add_module('a_' + str(i + 1), ReLU())
            fc.add_module('out', Linear(self.hidden_size_linear, self.num_classes))

            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
                    m.bias.data.fill_(0.01)

            fc.apply(init_weights)

            return fc

        if train:
            self.fc = create_net()
        else:
            # Try to load existing network
            try:
                self.load_network()
            except:
                self.fc = create_net()
                print('New network had to be initialized as no network exists for this element and measurement. '
                      'Training necessary!')

    def forward(self, x):
        """
        Forward pass of the network

        Parameters
        ----------
        x : torch.tensor
            network input

        Returns
        -------
        torch.tensor
            network output

        """
        return self.fc(x)

    def load_element_data(self):
        """
        Loads the element data from the excel sheet provided at self.element_data_path
        """

        stream = pkg_resources.resource_stream(__name__, self.element_data_path)
        self.element_data = pd.read_excel(stream, sheet_name='Phases', index_col='Index')

    def load_network(self):
        """
        Loads a network from the directory provided at self.models_path
        """
        model_name = self.models_path + self.element + '_' + self.measurement + '.pth'  # e.g., Fe_G
        stream = pkg_resources.resource_stream(__name__, model_name)
        self.fc = torch.load(stream)
