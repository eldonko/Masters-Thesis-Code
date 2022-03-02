import os
import warnings
import pkg_resources

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Softmax, LSTM, Sequential, BatchNorm1d, Tanh
from torch.nn.functional import pad
import pandas as pd


class ElementClassifier(nn.Module):
    """
    phases classifies thermodynamic measurement data in the sense that given measurement data for the
    Gibbs energy, entropy, enthalpy or the heat capacity at certain temperatures, it is able to tell which  element
    this measurement data belongs to.

    Version: Input padding to maximum 10 measurements

    """

    def __init__(self, train=False, in_features=20, hidden_size_linear=64):
        """

        Parameters
        ----------
		train : bool
		    If train is True, a network with random weights is generated and trained, if train is False, an
		    already trained network for this element and measurement type will be loaded.
		measurement : str
		    Defines for which properties the classification should be made. Must be one of 'G', 'S', 'H'
		    or 'C'.
		"""
        super(ElementClassifier, self).__init__()

        # File paths
        self.element_data_path = r'../data/Phases.xlsx'
        self.models_path = r'./models/'

        # Load element data
        self.element_data = None
        self.load_element_data()  # Load element data

        # Fully connected net for classification
        self.num_classes = len(self.element_data.columns)
        self.in_features = in_features
        self.hidden_size_linear = hidden_size_linear

        # For training, create a new network, else load a pre-trained network
        if train or (not train and not os.path.exists(os.path.join(self.models_path, 'elements'))):
            # Error handling
            if not train and not os.path.exists(os.path.join(self.models_path, 'elements')):
                print('New network had to be initialized as no network exists yet. Training necessary!')

            self.fc = Sequential(
                Linear(self.in_features, self.hidden_size_linear),
                ReLU(),
                Linear(self.hidden_size_linear, self.hidden_size_linear),
                ReLU(),
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
        """Forward pass of the network

        Parameters
        ----------
        x : torch.tensor [batch_size, n, 2]
            network input. Should be a torch.tensor of shape [batch_size, n, 2] where n is the number of measurements.
            If n is smaller than max_num_measurements, the tensor will be padded with 0 so that the tensor has a shape
            of [batch_size, max_num_measurements, 2]. In case n is greater than max_num_measurements, all the inputs
            with an index above max_num_measurements will not be considered for the input.

        Returns
        -------
        torch.tensor
            network output

        """

        # Maximum number of measurements
        max_num_measurements = self.in_features / 2

        # Input checking
        if not len(x.shape) == 3:
            raise ValueError('The network input must be a torch.tensor of shape (batch_size, n, 2), where n is an '
                             'arbitrary number smaller or equal 10. The tensor you provided has shape ', x.shape)

        if x.shape[2] != 2:
            raise ValueError('The network input must be a torch.tensor of shape (batch_size, n, 2), where n is an '
                             'arbitrary number smaller or equal 10. The tensor you provided has shape ', x.shape)

        if x.shape[1] > max_num_measurements:
            warnings.warn('The second dimension of the input tensor should not be greater than 10. The entries with '
                          'index equal or greater than 10 will be left out!')
            x = x[:, :max_num_measurements]

        # If the second dimension of the input (=number of measurements) is smaller than 10, pad the tensor with 0 after
        # the measurements
        if x.shape[1] < max_num_measurements:
            num_pad = max_num_measurements - x.shape[1]
            p2d = (0, 0, 0, num_pad)  # pad last dim by (0, 0) and 2nd to last by (0, num_pad)
            x = pad(x, p2d, 'constant', 0)

        x = torch.flatten(x, start_dim=1)
        return self.fc(x)

    def load_element_data(self):
        """Loads the element data from the excel sheet provided at self.element_data_path"""

        stream = pkg_resources.resource_stream(__name__, self.element_data_path)
        self.element_data = pd.read_excel(stream, sheet_name='Phases', index_col='Index')

    def load_network(self):
        """Change the path where already trained models are stored. Needs to be a directory"""
        model_name = self.models_path + 'elements'
        stream = pkg_resources.resource_stream(__name__, model_name)
        self.fc = torch.load(stream)
