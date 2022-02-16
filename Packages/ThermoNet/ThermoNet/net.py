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
