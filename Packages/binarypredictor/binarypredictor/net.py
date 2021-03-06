import os
import pkg_resources

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Softmax, LSTM, Sequential, BatchNorm1d, Tanh, Sigmoid


class TangentNet(nn.Module):
    """
    Given the values of two functions as input, DerivativeNet predicts an approximated polynomial by its coefficients.
    The approximated polynomial defines the connection between the x-values of both input functions, so that the first
    derivative is the same for both.
    """

    def __init__(self, train=False, in_features=200, out_features=10000, hidden_size_linear=128, hidden_layers=1):
        super(TangentNet, self).__init__()

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

            layers_size = [in_features] + hidden_layers * [hidden_size_linear] + [out_features]

            # Create the network
            self.fc = Sequential()
            self.fc.add_module('in', Linear(int(layers_size[0]), (int(layers_size[1]))))
            self.fc.add_module('a_in', ReLU())
            for i in range(1, len(layers_size) - 2, 1):
                self.fc.add_module('h_' + str(i + 1), Linear(int(layers_size[i]), int(layers_size[i + 1])))
                self.fc.add_module('a_' + str(i + 1), ReLU())
            self.fc.add_module('out', Linear(int(layers_size[-2]), int(layers_size[-1])))
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
    Given the values of one function as input, DerivativeNet predicts the first derivative of this function.
    """

    def __init__(self, train=False, in_features=100, out_features=100, hidden_size_linear=128, hidden_layers=1
                 , net='DerivativeNet.pth'):
        super(DerivativeNet, self).__init__()

        # File paths
        self.models_path = r'./models/'
        self.net = net

        # Fully connected net for classification
        self.out_features = out_features
        self.in_features = in_features
        self.hidden_size_linear = hidden_size_linear

        # For training, create a new network, else load a pre-trained network
        if train:
            layers_size = [in_features] + hidden_layers * [hidden_size_linear] + [out_features]

            # Create the network
            self.fc = Sequential()
            self.fc.add_module('in', Linear(int(layers_size[0]), (int(layers_size[1]))))
            self.fc.add_module('a_in', ReLU())
            for i in range(1, len(layers_size) - 2, 1):
                self.fc.add_module('h_' + str(i + 1), Linear(int(layers_size[i]), int(layers_size[i + 1])))
                self.fc.add_module('a_' + str(i + 1), ReLU())
            self.fc.add_module('out', Linear(int(layers_size[-2]), int(layers_size[-1])))

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
        model_name = self.models_path + self.net
        stream = pkg_resources.resource_stream(__name__, model_name)
        self.fc = torch.load(stream)


class FixedSystemNet(nn.Module):
    """
    Given the values of one function as input, DerivativeNet predicts the first derivative of this function.
    """

    def __init__(self, train=False, in_features=1, out_features=2, hidden_size_linear=32, hidden_layers=1
                 , net='FixedSystemNet.pth'):
        super(FixedSystemNet, self).__init__()

        # File paths
        self.models_path = r'./models/'
        self.net = net

        # Fully connected net for classification
        self.out_features = out_features
        self.in_features = in_features
        self.hidden_size_linear = hidden_size_linear

        # For training, create a new network, else load a pre-trained network
        if train:
            layers_size = [in_features] + hidden_layers * [hidden_size_linear] + [out_features]

            # Create the network
            self.fc = Sequential()
            self.fc.add_module('in', Linear(int(layers_size[0]), (int(layers_size[1]))))
            self.fc.add_module('a_in', ReLU())
            for i in range(1, len(layers_size) - 2, 1):
                self.fc.add_module('h_' + str(i + 1), Linear(int(layers_size[i]), int(layers_size[i + 1])))
                self.fc.add_module('a_' + str(i + 1), ReLU())
            self.fc.add_module('out', Linear(int(layers_size[-2]), int(layers_size[-1])))
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
        model_name = self.models_path + self.net
        stream = pkg_resources.resource_stream(__name__, model_name)
        self.fc = torch.load(stream)