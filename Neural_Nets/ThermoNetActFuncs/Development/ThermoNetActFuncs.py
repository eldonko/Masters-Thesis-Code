import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


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
        self.R = Parameter(torch.tensor(8.3145), requires_grad=False)
        self.E0 = Parameter(torch.tensor(-10000.0), requires_grad=True)
        self.theta_E = Parameter(torch.tensor(-1000.0), requires_grad=True)
        self.a = Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = Parameter(torch.tensor(1.0), requires_grad=True)

    def _initialize_parameters(self, theta_E=-1000., E0=-10000., a=1.0, b=1.0):
        self.E0 = Parameter(torch.tensor(E0), requires_grad=True)
        self.theta_E = Parameter(torch.tensor(theta_E), requires_grad=False)
        self.a = Parameter(torch.tensor(a), requires_grad=True)
        self.b = Parameter(torch.tensor(b), requires_grad=True)

    def forward(self, s):
        """
        This function returns the numerically stable ChenSundman activation which is needed for the entropy

        :param s: pre-activation
        :return: activation
        """

        # Restrict self.theta_E to negative values as positive values can lead to numerical instability inside the log
        #self.theta_E.data = torch.tensor(min(-1, int(self.theta_E.data)), dtype=torch.float32)

        return self.E0 + 3/2 * self.R * self.theta_E + 3 * self.R * s * (torch.log(1 - torch.exp(-self.theta_E/s))) - \
                1/2 * self.a * s ** 2 - 1/6 * self.b * s ** 3

    def first_derivative(self, s):
        """
        This function returns the numerically stable first derivative of the ChenSundman activation which is needed for
        the entropy

        :param s: pre-activation
        :return: activation
        """

        # Restrict self.theta_E to negative values as positive values can lead to numerical instability inside the log
        #self.theta_E.data = torch.tensor(min(-1, int(self.theta_E.data)), dtype=torch.float32)

        return 3 * self.R * (self.theta_E/(s - s * torch.exp(self.theta_E/s)) + (torch.log(1 - torch.exp(-self.theta_E/s)))) - \
                self.a * s - 1/2 * self.b * s ** 2

    def second_derivative(self, s):
        """
        This function returns the numerically stable second derivative of the ChenSundman activation which is needed for
        the heat-capacity

        :param s: pre-activation
        :return: activation
        """

        # Restrict self.theta_E to negative values as positive values can lead to numerical instability inside the log
        #self.theta_E.data = torch.tensor(min(-1, int(self.theta_E.data)), dtype=torch.float32)

        return -(3 * self.theta_E ** 2 * self.R * torch.exp(self.theta_E/s))/(s ** 3 * (torch.exp(self.theta_E/s) - 1) ** 2) - \
               self.a - self.b * s


# Own implementation of numerically stable Softplus activation so that derivatives can be used
class Softplus(nn.Module):
    def __init__(self):
        super(Softplus, self).__init__()

    def forward(self, s):
        """
        Forward of numerically stable Softplus activation

        :param s: pre-activation
        :return: activation
        """

        return nn.ReLU()(s) + torch.log(torch.exp(-torch.abs(s)) + 1)  # Use of ReLU for max(s, 0)

    @staticmethod
    def first_derivative(s):
        """
        Returns the first derivative of the softplus activation for an input s

        :param s: pre-activation
        :return: activation
        """

        relu_prime = torch.zeros_like(s)
        relu_prime[s > 0] = 1

        return relu_prime - s/(torch.exp(abs(s)) * abs(s) + abs(s))

    @staticmethod
    def second_derivative(s):
        """
        Returns the second derivative of the softplus activation for an input s

        :param s: pre-activation
        :return: activation
        """

        second_derivative = torch.exp(s) / (torch.exp(s) + 1) ** 2
        second_derivative = torch.nan_to_num(second_derivative)

        return second_derivative
