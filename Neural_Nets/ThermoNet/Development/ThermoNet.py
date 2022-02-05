import torch
import torch.nn as nn
from torch.nn import Linear
import torch.autograd as autograd


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

    def __init__(self, hidden_layers=1, hidden_dim=16, act_funcs=None):
        super(ThermoRegressionNet, self).__init__()

        # Needed for entropy (derivatives of forward)
        self._cache = []

        if act_funcs is None:
            act_funcs = [nn.Softplus()] * (hidden_layers + 1)

        assert len(act_funcs) == hidden_layers + 1

        self.layers = nn.ModuleList()

        # Input layer
        il = Linear(1, hidden_dim)
        nn.init.xavier_uniform_(il.weight)
        self.layers.append(il)

        # Input activation
        self.layers.append(act_funcs[0])

        # Hidden layers
        for i in range(hidden_layers):
            in_dim = int(hidden_dim / (2 ** i))
            out_dim = int(hidden_dim / (2 ** (i + 1)))
            hl = Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(hl.weight)
            self.layers.append(hl)

            self.layers.append(act_funcs[i + 1])

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
        temp.requires_grad = True
        entropy = -1 * autograd.grad(self(temp), temp, grad_outputs=torch.ones_like(self(temp)))[0]

        # Enthalpy
        enthalpy = self(temp) + temp * entropy

        # Heat capacity
        entropy.requires_grad = True
        heat_cap = autograd.grad(-1 * entropy, temp, grad_outputs=torch.ones_like(entropy), allow_unused=True)[0]

        return entropy, enthalpy, heat_cap
