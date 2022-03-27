import os
import pkg_resources

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Softmax, LSTM, Sequential, BatchNorm1d, Tanh, Sigmoid


class BinaryPredictor(nn.Module):
    """
    """

    def __init__(self, train=False, in_features=200, out_features=20, hidden_size_linear=256, hidden_layers=1):
        super(BinaryPredictor, self).__init__()

        # File paths
        self.models_path = r'./models/'

        # Fully connected net for classification
        self.out_features = out_features
        self.in_features = in_features
        self.hidden_size_linear = hidden_size_linear

        # For training, create a new network, else load a pre-trained network
        if train or (not train and not os.path.exists(os.path.join(self.models_path, 'binary.pth'))):
            # Error handling
            if not train and not os.path.exists(os.path.join(self.models_path, 'binary.pth')):
                print('New network had to be initialized as no network exists yet. Training necessary!')

            # Create the network
            self.fc = Sequential()
            self.fc.add_module('in', Linear(self.in_features, self.hidden_size_linear))
            self.fc.add_module('a_in', ReLU())
            for i in range(hidden_layers):
                self.fc.add_module('h_' + str(i + 1), Linear(self.hidden_size_linear, self.hidden_size_linear))
                self.fc.add_module('a_' + str(i + 1), ReLU())
            self.fc.add_module('h_last', Linear(self.hidden_size_linear, self.hidden_size_linear))
            self.fc.add_module('a_last', Tanh())
            self.fc.add_module('out', Linear(self.hidden_size_linear, self.out_features))
            self.fc.add_module('a_out', Sigmoid())

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
        x : torch.tensor [batch_size, n, 2]

        Returns
        -------
        torch.tensor
            network output

        """

        return self.fc(x)

    def load_network(self):
        """Change the path where already trained models are stored. Needs to be a directory"""
        model_name = self.models_path + 'elements'
        stream = pkg_resources.resource_stream(__name__, model_name)
        self.fc = torch.load(stream)


class DerivativeNet(nn.Module):
    """
    Given the values of two functions as input, DerivativeNet predicts an approximated polynomial by its coefficients.
    The approximated polynomial defines the connection between the x-values of both input functions, so that the first
    derivative is the same for both.
    """

    def __init__(self, train=False, in_features=200, out_features=5, hidden_size_linear=128, hidden_layers=1):
        super(DerivativeNet, self).__init__()

        # File paths
        self.models_path = r'./models/'

        # Fully connected net for classification
        self.out_features = out_features
        self.in_features = in_features
        self.hidden_size_linear = hidden_size_linear

        # For training, create a new network, else load a pre-trained network
        if train or (not train and not os.path.exists(os.path.join(self.models_path, 'DerivativeNet.pth'))):
            # Error handling
            if not train and not os.path.exists(os.path.join(self.models_path, 'binary.pth')):
                print('New network had to be initialized as no network exists yet. Training necessary!')

            # Create the network
            self.fc = Sequential()
            self.fc.add_module('in', Linear(self.in_features, self.hidden_size_linear))
            self.fc.add_module('a_in', ReLU())
            for i in range(hidden_layers):
                self.fc.add_module('h_' + str(i + 1), Linear(self.hidden_size_linear, self.hidden_size_linear))
                self.fc.add_module('a_' + str(i + 1), ReLU())
            self.fc.add_module('out', Linear(self.hidden_size_linear, self.out_features))

            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
                    m.bias.data.fill_(0.001)

            self.fc.apply(init_weights)
        else:
            self.load_network()

    def forward(self, x):
        """
        Forward pass of the network

        Parameters
        ----------
        x : torch.tensor [batch_size, n, 2]

        Returns
        -------
        torch.tensor
            network output

        """

        return self.fc(x)

    def load_network(self):
        """Change the path where already trained models are stored. Needs to be a directory"""
        model_name = self.models_path + 'elements'
        stream = pkg_resources.resource_stream(__name__, model_name)
        self.fc = torch.load(stream)