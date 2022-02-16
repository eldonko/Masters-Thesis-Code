import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Softplus
from Neural_Nets.ThermoNetActFuncs.Development.ThermoNetActFuncs import ChenSundman, Softplus


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class LaengeNetModified(nn.Module):
    def __init__(self, hidden_dim=16):
        super(LaengeNetModified, self).__init__()

        self.layers = Sequential(
            Linear(1, hidden_dim),
            Softplus(),
            Linear(hidden_dim, hidden_dim),
            Softplus(),
            Linear(hidden_dim, hidden_dim),
            Softplus(),
            Linear(hidden_dim, 4)
        )

        self.layers.apply(init_weights)

    def forward(self, temp):
        return self.layers(temp)


class LaengeNetModifiedLossFunc(nn.Module):
    """
    Loss function for the LaengeNet as defined in Länge
    """

    def __init__(self):
        """
        """
        super(LaengeNetModifiedLossFunc, self).__init__()

    def __call__(self, prediction, target):
        """

        :param prediction: Predicted Gibbs energy, entropy, enthalpy and heat capacity
        :param target: True Gibbs energy, entropy, enthalpy and heat capacity
        :return: loss
        """

        return self.loss(prediction, target)

    @staticmethod
    def loss(prediction, target):
        """

        :param prediction: Predicted Gibbs energy, entropy, enthalpy and heat capacity
        :param target: True Gibbs energy, entropy, enthalpy and heat capacity
        :return: loss
        """

        return nn.MSELoss()(prediction, target)
