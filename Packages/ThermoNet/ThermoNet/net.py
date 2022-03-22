import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Softplus


class ThermoNet(nn.Module):
    """
    Network for regression analysis of thermodynamic properties Gibbs energy, entropy, enthalpy and heat-capacity. Makes
    predictions for all four properties at once.
    """
    def __init__(self, hidden_dim=16):
        super(ThermoNet, self).__init__()

        self.layers = Sequential(
            Linear(1, hidden_dim),
            Softplus(),
            Linear(hidden_dim, hidden_dim),
            Softplus(),
            Linear(hidden_dim, hidden_dim),
            Softplus(),
            Linear(hidden_dim, 4)
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.layers.apply(init_weights)

    def forward(self, temp):
        """

        Parameters
        ----------
        temp : torch.tensor [batch_size, 1]
            temperature

        Returns
        -------
        torch.tensor [batch_size, 4]
            network output

        """
        return self.layers(temp)


class ThermoNetNew(nn.Module):
    """
    Neural network that using a regression analysis gives the Gibbs energy, entropy, enthalpy and heat capacity
    for a given element and phase at a temperature and pressure
    """
    def __init__(self, hidden_size_linear=128, hidden_layers=1, out_features=4):
        super(ThermoNetNew, self).__init__()

        self.hidden_size_linear = hidden_size_linear
        self.hidden_layers = hidden_layers

        self.in_features = 3  # Temperature, element label, phase label
        self.out_features = out_features  # Gibbs energy, entropy, enthalpy, heat capacity

        # Create the network
        self.fc = Sequential()
        self.fc.add_module('in', Linear(self.in_features, self.hidden_size_linear))
        self.fc.add_module('a_in', Softplus())
        for i in range(hidden_layers):
            self.fc.add_module('h_' + str(i + 1), Linear(self.hidden_size_linear, self.hidden_size_linear))
            self.fc.add_module('a_' + str(i + 1), Softplus())
        self.fc.add_module('out', Linear(self.hidden_size_linear, self.out_features))

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.fc.apply(init_weights)

    def forward(self, x):
        return self.fc(x)
