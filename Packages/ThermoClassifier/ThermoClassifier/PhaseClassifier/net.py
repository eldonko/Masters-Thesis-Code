import os
import pkg_resources

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Softmax, LSTM, Sequential, BatchNorm1d, Tanh
import pandas as pd


class PhaseClassifier(nn.Module):
    """
	PhaseClassifier classifies thermodynamic measurement data in the sense that given measurement data for the
	Gibbs energy, entropy, enthalpy or the heat capacity at certain temperatures, it is able to tell which phase of
	a given element this measurement data belongs to.
	"""

    def __init__(self, element, train=False, measurement='G'):
        """

        Parameters
        ----------

		element : str
		    Which element the measurements are from (and the phase should be predicted for)
		train : bool
		    If train is True, a network with random weights is generated and trained, if train is False, an
		    already trained network for this element and measurement type will be loaded. (Default value = False)
		measurement : str
		    Defines for which properties the classification should be made. Must be one of 'G', 'S', 'H'
		    or 'C'. (Default value = 'G')
		"""
        super(PhaseClassifier, self).__init__()

        # Input checking
        assert measurement in ['G', 'S', 'H', 'C']
        self.measurement = measurement

        # File paths
        self.element_data_path = r'../data/Phases.xlsx'
        self.models_path = r'./models'

        # Load element data
        self.element_data = None
        self.load_element_data()
        assert element in self.element_data.columns.values
        self.element = element

        # Fully connected net for classification
        self.num_classes = self.element_data[element].sum()
        self.in_features = 2
        self.hidden_size_linear = 32

        # For training, create a new network, else load a pre-trained network
        if train or (not train and not os.path.exists(os.path.join(self.models_path, element + '_' + measurement))):
            # Error handling
            if not train and not os.path.exists(os.path.join(self.models_path, element + '_' + measurement)):
                print('New network had to be initialized as no network exists for this element and measurement. '
                      'Training necessary!')

            self.fc = Sequential(
                Linear(self.in_features, self.hidden_size_linear),
                Tanh(),
                Linear(self.hidden_size_linear, self.hidden_size_linear),
                Tanh(),
                # Linear(self.hidden_size_linear, self.hidden_size_linear),
                # Tanh(),
                Linear(self.hidden_size_linear, self.num_classes)
            )

            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
                    m.bias.data.fill_(0.01)

            self.fc.apply(init_weights)
        else:
            self.load_network()

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
        model_name = self.element + '_' + self.measurement  # e.g., Fe_G
        self.fc = torch.load(os.path.join(self.models_path, model_name))