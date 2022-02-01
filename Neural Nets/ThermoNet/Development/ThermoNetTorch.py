import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
import numpy as np
from SGTEHandler.Development.SGTEHandler import SGTEHandler


# Neural network containing both subnetworks
class ThermoNet(nn.Module):
    """
    ThermoNet is aiming to rebuild the network for approximating thermodynamic properties proposed in "An artificial
    neural network model for the unary description of pure iron" [Länge, M.] https://doi.org/10.1007/s00500-019-04663-3
    """
    def __init__(self, hidden_dim_sub_net_2=16):
        super(ThermoNet, self).__init__()

        self.sub_net_1 = SubNet(ChenSundman(), 1)
        self.sub_net_2 = SubNet(Softplus(), hidden_dim_sub_net_2)

    def forward(self, *args):
        if len(args) == 1:
            gibbs_1 = self.sub_net_1(*args)
            gibbs_2 = self.sub_net_2(*args)

            return gibbs_1 + gibbs_2
        elif len(args) == 4:
            gibbs_1, entropy_1, enthalpy_1, heat_cap_1 = self.sub_net_1(*args)
            gibbs_2, entropy_2, enthalpy_2, heat_cap_2 = self.sub_net_2(*args)

            return gibbs_1 + gibbs_2, entropy_1 + entropy_2, enthalpy_1 + enthalpy_2, heat_cap_1 + heat_cap_2


class SubNet(nn.Module):
    def __init__(self, activation, hidden_dim):
        super(SubNet, self).__init__()

        # NN layers
        self.layer_1 = Linear(1, hidden_dim)
        self.act_1 = activation
        self.layer_2 = Linear(hidden_dim, 1)

        self._initialize_parameters()

    def __call__(self, *temp):
        if len(temp) == 1:
            return self.gibbs(temp[0])
        elif len(temp) == 4:
            return self.gibbs(temp[0]), self.entropy(temp[1]), self.enthalpy(temp[2]), self.heat_capacity(temp[3])

    def _initialize_parameters(self):
        # Initialize parameters
        nn.init.uniform_(self.layer_1.weight, a=-0.2, b=-0.1)
        nn.init.uniform_(self.layer_1.bias, a=-0.2, b=-0.1)
        nn.init.uniform_(self.layer_2.weight, a=-0.2, b=-0.1)
        nn.init.uniform_(self.layer_2.bias, a=-0.2, b=-0.1)

    def gibbs(self, xg):
        """
        Forward pass of the network to approximate the Gibbs energy

        :param xg: Temperature value (torch.Tensor)
        :return: Gibbs energy (torch.Tensor)
        """

        s = self.layer_1(xg.float())
        a = self.act_1(s)

        # Gibbs energy
        gibbs = self.layer_2(a)

        #print(gibbs)

        return gibbs

    def entropy(self, xs):
        """
        Forward pass of the network to approximate the entropy

        :param xs: Temperature value (torch.Tensor)
        :return: entropy (torch.Tensor)
        """

        s = self.layer_1(xs.float())

        # Entropy
        entropy = -self.layer_2.weight * self.act_1.first_derivative(s) @ self.layer_1.weight

        return entropy

    def enthalpy(self, xh):
        """
        Forward pass of the network to approximate the enthalpy

        :param xh: Temperature value (torch.Tensor)
        :return: enthalpy (torch.Tensor)
        """

        return self.gibbs(xh) + xh * self.entropy(xh)

    def heat_capacity(self, xc):
        """
        Forward pass of the network to approximate the heat capacity

        :param xc: Temperature value (torch.Tensor)
        :return: heat capacity (torch.Tensor)
        """

        s = self.layer_1(xc.float())

        # Heat capacity
        heat_cap = -xc * self.layer_2.weight * self.act_1.second_derivative(s) @ self.layer_1.weight.double() ** 2

        return heat_cap.float()


# Activation function with thermodynamic parameters
class ChenSundman(nn.Module):
    """
    Implementation of activation function with learnable parameters based on Chen & Sundman model:

    f(s) = E0 + 3/2 * R * theta_E + 3 * R * s * log(1 - exp(-theta_E/s)) - 1/2 * a * s^2 - 1/6 * b * s^3

    where:
        - R: universal gas constant
        - E0, theta_E, a, b: optimized network parameters
        - s: input
    """

    def __init__(self):
        """
        Initialization of activation function and the trainable parameters
        """

        super(ChenSundman, self).__init__()

        # Initialize parameters
        self.R = Parameter(torch.tensor(8.3145))
        self.E0 = Parameter(torch.tensor(1.0))
        self.theta_E = Parameter(torch.tensor(-1.0))
        self.a = Parameter(torch.tensor(1.0))
        self.b = Parameter(torch.tensor(1.0))

        # Define require_grad
        self.R.requires_grad = False
        self.E0.requires_grad = True
        self.theta_E.requires_grad = False
        self.a.requires_grad = True
        self.b.requires_grad = True

    def forward(self, s):
        """
        This function returns the numerically stable ChenSundman activation which is needed for the entropy

        :param s: pre-activation
        :return: activation
        """

        # Restrict self.theta_E to negative values as positive values can lead to numerical instability inside the log
        if self.theta_E > 0:
            self.theta_E = Parameter(torch.tensor(0.0 - 1e-6))

        return self.E0 + 3/2 * self.R * self.theta_E + 3 * self.R * s * (torch.log(1 - torch.exp(-self.theta_E/s))) - \
                1/2 * self.a * s ** 2 - 1/6 * self.b * s ** 3

    def first_derivative(self, s):
        """
        This function returns the numerically stable first derivative of the ChenSundman activation which is needed for
        the entropy

        :param s: pre-activation
        :return: activation
        """

        # Restrict self.theta_E to positive values as negative values can lead to numerical instability inside the log
        if self.theta_E > 0:
            self.theta_E = Parameter(torch.tensor(0.0 - 1e-6))

        return 3 * self.R * (self.theta_E/(s - s * torch.exp(self.theta_E/s)) + (torch.log(1 - torch.exp(-self.theta_E/s)))) - \
                self.a * s - 1/2 * self.b * s ** 2

    def second_derivative(self, s):
        """
        This function returns the numerically stable second derivative of the ChenSundman activation which is needed for
        the heat-capacity

        :param s: pre-activation
        :return: activation
        """

        # Restrict self.theta_E to positive values as negative values can lead to numerical instability inside the log
        if self.theta_E > 0:
            self.theta_E = Parameter(torch.tensor(0.0 - 1e-6))

        return -(3 * self.theta_E ** 2 * self.R * torch.exp(self.theta_E/s))/(s ** 3 * (torch.exp(self.theta_E/s) - 1) ** 2) - \
               self.a - self.b * s


# Own implementation of Softplus activation so that derivatives can be used
class Softplus(nn.Module):
    def __init__(self):
        super(Softplus, self).__init__()

    def forward(self, s):
        """
        Forward of Softplus activation

        :param s: pre-activation
        :return: activation
        """

        return torch.log(torch.exp(s) + 1)

    @staticmethod
    def first_derivative(s):
        """
        Returns the first derivative of the softplus activation for an input s

        :param s: pre-activation
        :return: activation
        """

        return torch.exp(s) / (torch.exp(s) + 1)

    @staticmethod
    def second_derivative(s):
        """
        Returns the second derivative of the softplus activation for an input s

        :param s: pre-activation
        :return: activation
        """

        return torch.exp(s) / (torch.exp(s) + 1) ** 2


class ThermoLossFunc(nn.Module):
    """
    Loss function for the ThermoNet as defined in Länge
    """
    def __init__(self):
        super(ThermoLossFunc, self).__init__()

    def __call__(self, *args):
        """

        :param g_p: Predicted Gibbs energy
        :param g_t: True Gibbs energy
        :param s_p: Predicted entropy
        :param s_t: True entropy
        :param h_p: Predicted enthalpy
        :param h_t: True enthalpy
        :param c_p: Predicted heat capacity
        :param c_t: True heat capacity
        :return: loss
        """

        if len(args) == 2:
            return self.loss_gibbs(args[0], args[1])
        elif len(args) == 8:
            g_p, g_t, s_p, s_t, h_p, h_t, c_p, c_t = args
            return self.loss_all(g_p, g_t, s_p, s_t, h_p, h_t, c_p, c_t)

    @staticmethod
    def loss_gibbs(g_p, g_t):
        """

        :param g_p: Predicted Gibbs energy
        :param g_t: True Gibbs energy
        :return: loss
        """

        gibbs_loss = nn.MSELoss()(g_p, g_t)

        return gibbs_loss

    @staticmethod
    def loss_all(g_p, g_t, s_p, s_t, h_p, h_t, c_p, c_t):
        """

        :param g_p: Predicted Gibbs energy
        :param g_t: True Gibbs energy
        :param s_p: Predicted entropy
        :param s_t: True entropy
        :param h_p: Predicted enthalpy
        :param h_t: True enthalpy
        :param c_p: Predicted heat capacity
        :param c_t: True heat capacity
        :return: loss
        """

        gibbs_loss = nn.MSELoss()(g_p, g_t)
        entropy_loss = nn.MSELoss()(s_p, s_t)
        enthalpy_loss = nn.MSELoss()(h_p, h_t)
        heat_cap_loss = nn.MSELoss()(c_p, c_t)

        return gibbs_loss / 10 + 100000 * entropy_loss + enthalpy_loss / 10 + 10000 * heat_cap_loss


class ThermoDataset(Dataset):
    def __init__(self, element, phase):
        super(ThermoDataset, self).__init__()

        sgte_handler = SGTEHandler(element)
        sgte_handler.evaluate_equations(200, 2000, 1e5, plot=False, phases=phase, entropy=True, enthalpy=True,
                                        heat_capacity=True)
        data = sgte_handler.equation_result_data

        # Get values
        temp = torch.tensor(data['Temperature'], dtype=torch.float64)
        gibbs = torch.tensor(data.iloc[:, 1])
        entropy = torch.tensor(data.iloc[:, 2])
        enthalpy = torch.tensor(data.iloc[:, 3])
        heat_cap = torch.tensor(data.iloc[:, 4])

        self._samples = [(t, g, s, h, c) for t, g, s, h, c in zip(temp, gibbs, entropy, enthalpy, heat_cap)]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i: int):
        return self._samples[i]
