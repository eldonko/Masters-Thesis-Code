import torch
import torch.nn as nn
from torch.nn import Linear
from .act import ChenSundman, Softplus


class LaengeNet(nn.Module):
    """
    laenge is aiming to rebuild the network for approximating thermodynamic properties proposed in "An artificial
    neural network model for the unary description of pure iron" [Länge, M.] https://doi.org/10.1007/s00500-019-04663-3

    Parameters
    ----------
    hidden_dim_sub_net_2 : int
        hidden dimension of sub network 2 (Default value = 16)
    initialize : bool
        if the network parameters should be initialized using an initialization function provided by an input (Default
        value = True)
    init_func: nn.init function
        initialization function to initialize the network parameters with (Default value = nn.init.uniform_)
    init_args : tuple or None
        arguments passed to the initialization function. Depends on the initialization function. (Default value = None)
    act_1 : activation function
        activation function for hidden layer of sub network 1
    act_2 : activation function
        activation function for hidden layer of sub network 2

    Returns
    -------

    """
    def __init__(self, hidden_dim_sub_net_2=16, initialize=True, init_func=nn.init.uniform_, init_args=None,
                 act_1=ChenSundman(), act_2=Softplus()):
        super(LaengeNet, self).__init__()

        self.sub_net_1 = SubNet(act_1, 1, initialize, init_func, init_args)
        self.sub_net_2 = SubNet(act_2, hidden_dim_sub_net_2, initialize, init_func, init_args)

    def forward(self, *args, debug=False):
        """

        Parameters
        ----------
        *args :
            temperature input for the network
            
        debug : bool
            prints debug values if True (Default value = False)

        Returns
        -------

        """
        if len(args) == 1:
            gibbs_1 = self.sub_net_1(*args, debug)
            gibbs_2 = self.sub_net_2(*args, debug)

            return gibbs_1 + gibbs_2
        elif len(args) == 4:
            gibbs_1, entropy_1, enthalpy_1, heat_cap_1 = self.sub_net_1(*args, debug=debug)
            gibbs_2, entropy_2, enthalpy_2, heat_cap_2 = self.sub_net_2(*args, debug=debug)

            return gibbs_1 + gibbs_2, entropy_1 + entropy_2, enthalpy_1 + enthalpy_2, heat_cap_1 + heat_cap_2


class SubNet(nn.Module):
    """
    Sub network as defined in Länge

    Parameters
    ----------
    activation : activation function
        activation function for the hidden layer
    hidden_dim : int
        number of nodes in the hidden layer
    initialize : bool
        if the network parameters should be initialized using an initialization function provided by an input (Default
        value = True)
    init_func: nn.init function
        initialization function to initialize the network parameters with (Default value = nn.init.uniform_)
    init_args : tuple or None
        arguments passed to the initialization function. Depends on the initialization function. (Default value = None)
    """
    def __init__(self, activation, hidden_dim, initialize=True, init_func=nn.init.uniform_, init_args=None):
        super(SubNet, self).__init__()

        # NN layers
        self.layer_1 = Linear(1, hidden_dim)
        self.act_1 = activation
        self.layer_2 = Linear(hidden_dim, 1)

        # Initialize parameters
        if initialize:
            if init_args is None:
                init_args = (-0.2, -0.1)
            self._initialize_parameters(init_func, init_args)

        # Debug settings
        self.debug = False

    def __call__(self, *temp, debug=False):
        self.debug = debug

        if len(temp) == 1:
            return self.gibbs(temp[0])
        elif len(temp) == 4:
            return self.gibbs(temp[0]), self.entropy(temp[1]), self.enthalpy(temp[2]), self.heat_capacity(temp[3])

    def _initialize_parameters(self, init_func, *args):
        """

        Parameters
        ----------
        init_func : torch.nn.init function
            
        *args :
            arguments to initialize the network parameters with. Depends on the initialization function
            

        Returns
        -------

        """
        # Initialize parameters
        if init_func == nn.init.uniform_:
            low = min(args[0])
            high = max(args[0])

            init_func(self.layer_1.weight, a=low, b=high)
            init_func(self.layer_1.bias, a=low, b=high)
            init_func(self.layer_2.weight, a=low, b=high)
            init_func(self.layer_2.bias, a=low, b=high)
        elif init_func == nn.init.xavier_normal_ or init_func == nn.init.xavier_uniform_:
            init_func(self.layer_1.weight)
            init_func(self.layer_2.weight)

    def gibbs(self, xg):
        """Forward pass of the network to approximate the Gibbs energy

        Parameters
        ----------
        xg : torch.tensor [batch_size, 1]
            Temperature values

        Returns
        -------
        torch.tensor [batch_size, 1]
            Gibbs energy

        """

        s = self.layer_1(xg.float())
        a = self.act_1(s)

        # Gibbs energy
        gibbs = self.layer_2(a)

        if self.debug:
            if not torch.isfinite(gibbs.min()):
                idx_min = gibbs.argmin().item()
                print('Negative infinity detected in Gibbs energy @ ', idx_min)
                print(xg[idx_min])
                print(s)
                print(a)

        return gibbs

    def entropy(self, xs):
        """Forward pass of the network to approximate the entropy

        Parameters
        ----------
        xs : torch.tensor [batch_size, 1]
            Temperature values

        Returns
        -------
        torch.tensor [batch_size, 1]
            entropy

        """

        s = self.layer_1(xs.float())

        # Entropy
        entropy = -self.layer_2.weight * self.act_1.first_derivative(s) @ self.layer_1.weight

        return entropy

    def enthalpy(self, xh):
        """Forward pass of the network to approximate the enthalpy

        Parameters
        ----------
        xh : torch.tensor [batch_size, 1]
            Temperature values

        Returns
        -------
        torch.tensor [batch_size, 1]
            enthalpy

        """

        return self.gibbs(xh) + xh * self.entropy(xh)

    def heat_capacity(self, xc):
        """Forward pass of the network to approximate the heat capacity

        Parameters
        ----------
        xc : torch.tensor [batch_size, 1]
            Temperature values

        Returns
        -------
        torch.tensor [batch_size, 1]
            heat capacity

        """

        s = self.layer_1(xc.float())

        # Heat capacity
        heat_cap = -xc * self.layer_2.weight * self.act_1.second_derivative(s) @ self.layer_1.weight.double() ** 2

        if self.debug:
            if not torch.isfinite(heat_cap.min()):
                idx_min = heat_cap.argmin().item()
                print('Negative infinity detected in heat cap @ ', idx_min)
                print(xc[idx_min])

        return heat_cap.float()


class LaengeNetLossFunc(nn.Module):
    """Loss function for the laenge as defined in Länge"""
    def __init__(self, weights=None):
        """
        :param weights: weights of the loss in case the loss is calculated on all 4 properties
        """
        super(LaengeNetLossFunc, self).__init__()

        if weights is not None:
            self.weights = weights
        else:
            self.weights = [0.01, 100000, 0.01, 10000]

    def __call__(self, *args, debug=False):
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

        if len(args) == 8:
            g_p, g_t, s_p, s_t, h_p, h_t, c_p, c_t = args
            return self.loss(g_p, g_t, s_p, s_t, h_p, h_t, c_p, c_t, debug)

    def loss(self, g_p, g_t, s_p, s_t, h_p, h_t, c_p, c_t, debug=False):
        """

        Parameters
        ----------
        g_p : torch.tensor
            Predicted Gibbs energy
        g_t : torch.tensor
            True Gibbs energy
        s_p : torch.tensor
            Predicted entropy
        s_t : torch.tensor
            True entropy
        h_p : torch.tensor
            Predicted enthalpy
        h_t : torch.tensor
            True enthalpy
        c_p : torch.tensor
            Predicted heat capacity
        c_t : torch.tensor
            True heat capacity
        debug : bool
            (Default value = False)

        Returns
        -------
        torch.tensor
            loss

        """

        # Get losses for each property
        gibbs_loss = nn.MSELoss()(g_p, g_t)
        entropy_loss = nn.MSELoss()(s_p, s_t)
        enthalpy_loss = nn.MSELoss()(h_p, h_t)
        heat_cap_loss = nn.MSELoss()(c_p, c_t)

        # Get the weights for the total loss
        wg, ws, wh, wc = self.weights

        # Get the total loss
        loss = wg * gibbs_loss + ws * entropy_loss + wh * enthalpy_loss + wc * heat_cap_loss

        if debug:
            print('Gibbs loss: ', gibbs_loss)
            print('g_p: ', (g_p.min().item(), g_p.max().item()), ', g_t: ', (g_t.min().item(), g_t.max().item()))
            print('Entropy loss: ', entropy_loss)
            print('s_p: ', (s_p.min().item(), s_p.max().item()), ', s_t: ', (s_t.min().item(), s_t.max().item()))
            print('Enthalpy loss:, ', enthalpy_loss)
            print('h_p: ', (h_p.min().item(), h_p.max().item()), ', h_t: ', (h_t.min().item(), h_t.max().item()))
            print('Heat cap loss: ', heat_cap_loss)
            print('c_p: ', (c_p.min().item(), c_p.max().item()), ', c_t: ', (c_t.min().item(), c_t.max().item()))
            print('Weights: ', self.weights)
            print('Total loss: ', loss, '\n')

        return loss, gibbs_loss, entropy_loss, enthalpy_loss, heat_cap_loss
