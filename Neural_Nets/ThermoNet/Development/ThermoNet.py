import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
import numpy as np
from Data_Handling.SGTEHandler.Development.SGTEHandler import SGTEHandler
from Neural_Nets.ThermoNetActFuncs.Development.ThermoNetActFuncs import ChenSundman, Softplus, Sigmoid, ELUFlipped


class ThermoNet(nn.Module):
    """
    Implementation of neural network that aims to predict Gibbs energy, entropy, enthalpy and heat capacity for any
    element from the periodic table given temperature and data for the element
    """

    def __init__(self):
        super(ThermoNet, self).__init__()


class ThermoRegressionNet(nn.Module):
    """
    This network aims to learn the Gibbs energy, entropy, enthalpy and heat capacity for any element given temperature
    value. It will be used in ThermoNet after an initial network which outputs modified temperature values and active
    phases based on the element data which it receives as input.
    """

    def __init__(self, hidden_layers=1, hidden_dim=16, act_func=Sigmoid()):
        super(ThermoRegressionNet, self).__init__()

        self.layers = nn.ModuleList()

        # Input layer
        il = Linear(1, hidden_dim)
        nn.init.xavier_uniform_(il.weight)
        self.layers.append(il)
        #self.layers.append(BatchNorm1d(hidden_dim))

        # Input activation
        self.layers.append(act_func)
        # self.layers.append(Softplus())

        # Hidden layers
        for i in range(hidden_layers):
            in_dim = int(hidden_dim / (2 ** i))
            out_dim = int(hidden_dim / (2 ** (i + 1)))
            hl = Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(hl.weight)
            # self.layers.append(BatchNorm1d(hidden_dim))
            self.layers.append(hl)

            self.layers.append(act_func)

        if hidden_layers < 1:
            out_dim = hidden_dim

        # Output layer
        ol = Linear(out_dim, 1)
        nn.init.xavier_normal_(ol.weight)
        self.layers.append(ol)

    def forward(self, x):
        """

        :param x: Temperature
        :return:
        """

        for layer in self.layers:
            x = layer(x.float())

        return x

    def output_all(self, temp):
        """

        :param temp: Temperature
        :return: entropy, enthalpy and heat capacity for given temperature
        """

        # Entropy
        s = self.layers[0](temp.float())
        entropy = -self.layers[-1].weight * self.input_act.first_derivative(s) @ self.input_layer.weight

        # Enthalpy
        enthalpy = self.forward(temp) + temp * entropy

        # Heat capacity
        heat_cap = -temp * self.output_layer.weight * self.input_act.second_derivative(
            s) @ self.input_layer.weight.double() ** 2

        return entropy, enthalpy, heat_cap
