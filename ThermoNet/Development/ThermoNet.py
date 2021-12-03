import numpy as np


class ThermoNet(object):
    """
    ThermoNet is a rebuild of the network shown in https://doi.org/10.1007/s00500-019-04663-3

    This network consists of 3 layers including input and output layer
    """

    def __init__(self, hidden_dim, input_dim=1, output_dim=1, loss_func=None):
        """
        Initializes the ThermoNet

        The matrices/vectors have the following shapes:
        input_vector:                     input_dim x 1
        w_1:                     hidden_dim x input_dim
        b_1:                             hidden_dim x 1
        w_2:                    output_dim x hidden_dim
        b_2:                             output_dim x 1
        output_vector:                   output_dim x 1

        :param input_dim: Dimensionality of the input vector (Generally 1)
        :param hidden_dim: Dimensionality of the hidden layer
        :param output_dim: Dimensionality of the output layer (Generally 1)
        :param loss_func: Loss function
        """
        super(ThermoNet, self).__init__()

        # Save the input
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize the weights of and the biases of subnetwork 1
        self.w_1_1 = np.random.uniform(low=0.01, high=0.01, size=(1, 1))
        self.b_1_1 = np.random.uniform(low=0.01, high=0.01, size=(1, 1))
        self.w_1_2 = np.random.uniform(low=0.01, high=0.01, size=(1, 1))
        self.b_1_2 = np.random.uniform(low=0.01, high=0.01, size=(1, 1))

        # Initialize the gradients of subnetwork 1
        self.db_1_1 = 0
        self.dw_1_1 = 0
        self.db_1_2 = 0
        self.dw_1_2 = 0

        # Initialize the weight matrices and the bias of subnetwork 2
        self.w_2_1 = np.random.uniform(low=0.01, high=0.01, size=(self.hidden_dim, self.input_dim))
        self.b_2_1 = np.random.uniform(low=0.01, high=0.01, size=(self.hidden_dim, 1))
        self.w_2_2 = np.random.uniform(low=0.01, high=0.01, size=(self.output_dim, self.hidden_dim))
        self.b_2_2 = np.random.uniform(low=0.01, high=0.01, size=(self.output_dim, 1))

        # Initialize the gradients of subnetwork 2
        self.db_2_1 = 0
        self.dw_2_1 = 0
        self.db_2_2 = 0
        self.dw_2_2 = 0

        # Initialize the constants of the activation function
        self.r = 8.134  # Universal gas constant
        # Learnable parameters of the activation function TODO: Make the parameters actually learnable
        self.e0 = 1
        self.theta_e = 1
        self.a = 1
        self.b = 1

        # If a loss function is specified, then use this, else use the mean squared error
        if loss_func is None:
            self.loss_func = lambda out, real: 1/2 * np.mean((out - real) ** 2)
        else:
            self.loss_func = loss_func

    def __call__(self, x):
        return self.forward(x)

    def act_fun(self, s):
        """
        Activation for subnetwork 1
        :param s: preactivations
        :return:
        """
        act = self.e0 + 3 / 2 * self.r * self.theta_e + 3 * self.r * s * np.log(1 - np.exp(-self.theta_e / s)) \
              - 1 / 2 * self.a * s ** 2 - 1 / 6 * self.b * s ** 3
        return act

    def d_act_fun(self, s):
        """
        Derivative of thermodynamic activation function
        :param s: preactivations
        :return:
        """

        return 3 * self.r * (self.theta_e/(s - s * np.exp(self.theta_e/s)) + np.log(1 - np.exp(-self.theta_e/s)))

    def softplus(self, s):
        """
        Activation for subnetwork 2
        :param s: preactivations
        :return:
        """
        return np.log(np.exp(s) + 1)

    def d_softplus(self, s):
        """
        Derivative of softplus activation
        :param s: preactivations
        :return: derivative
        """

        return np.exp(s)/(np.exp(s) + 1)

    def forward(self, x):
        """
        Forward pass of the network

        :param x: temperature values
        :return: g_hat (prediction for the Gibbs energy), cache (input x)
        """
        # Store the input for the backward pass
        cache = x

        # Subnetwork 1
        output_1 = self.w_1_2 @ self.act_fun(self.w_1_1 @ x + self.b_1_1) + self.b_1_2

        # Subnetwork 2
        # Preactivations of first layer
        s = self.w_2_1 @ x + self.b_2_1

        # Activations of first layer
        s = self.softplus(s)

        # Output of the network
        output_2 = self.w_2_2 @ s + self.b_2_2

        # Get the prediction for the Gibbs energy
        g_hat = output_1 + output_2

        return g_hat, cache

    def zero_grad(self):
        """
        Resets/initializes all gradients to 0

        :return:
        """

        # Subnetwork 1
        self.dw_1_1 = 0
        self.db_1_1 = 0
        self.dw_1_2 = 0
        self.db_1_2 = 0

        # Subnetwork 2
        self.dw_2_1 = 0
        self.db_2_1 = 0
        self.dw_2_2 = 0
        self.db_2_2 = 0

    def update(self, lr):
        """
        Updates the parameters of the network (SGD optimizer)
        :param lr: learning rate
        :return:
        """

        # Subnetwork 1
        self.w_1_1 -= lr * self.dw_1_1
        self.b_1_1 -= lr * self.db_1_1
        self.w_1_2 -= lr * self.dw_1_2
        self.b_1_2 -= lr * self.db_1_2

        # Subnetwork 2
        self.w_2_1 -= lr * self.dw_2_1
        self.b_2_1 -= lr * self.db_2_1
        self.w_2_2 -= lr * self.dw_2_2
        self.b_2_2 -= lr * self.db_2_2

    def backward(self, output, g, cache):
        """
        Backward pass of the network
        :param output: network output
        :param g: actual values
        :param cache: stored inputs of the forward pass
        :return:
        """

        x = cache

        # Gradient of loss w.r.t. output
        d_out = 1/len(output) * (output - g)

        # Subnetwork 1
        # Gradient of loss w.r.t w_1_2
        self.dw_1_2 = d_out @ self.act_fun(self.w_1_1 @ x + self.b_1_1).T

        # Gradient of loss w.r.t. b_1_2
        self.db_1_2 = d_out

        # Gradient of loss w.r.t. w_1_1
        diag_w_1_1 = np.zeros([self.w_1_1.shape[0], self.w_1_1.shape[0]])
        np.fill_diagonal(diag_w_1_1, self.d_act_fun(self.w_1_1 @ x + self.b_1_1))
        self.dw_1_1 = (d_out @ x @ self.dw_1_2 @ diag_w_1_1).T

        # Gradient of loss w.r.t. b_1_1
        self.db_1_1 = (d_out @ self.dw_1_2 @ diag_w_1_1).T

        # Subnetwork 2
        # Gradient of loss w.r.t w_2_2
        self.dw_2_2 = d_out @ self.softplus(self.w_2_1 @ x + self.b_2_1).T

        # Gradient of loss w.r.t. b_2_2
        self.db_2_2 = d_out

        # Gradient of loss w.r.t. w_2_1
        diag_w_2_1 = np.zeros([self.w_2_1.shape[0], self.w_2_1.shape[0]])
        np.fill_diagonal(diag_w_2_1, self.d_softplus(self.w_2_1 @ x + self.b_2_1))
        self.dw_2_1 = (d_out @ x @ self.dw_2_2 @ diag_w_2_1).T

        # Gradient of loss w.r.t. b_2_1
        self.db_2_1 = (d_out @ self.dw_2_2 @ diag_w_2_1).T
