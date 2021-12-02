import numpy as np


class ThermoNet(object):
    """
    ThermoNet is a rebuild of the network shown in https://doi.org/10.1007/s00500-019-04663-3

    This network consists of 3 layers including input and output layer
    """

    def __init__(self, input_dim, hidden_dim, output_dim, loss_func=None):
        """
        Initializes the ThermoNet

        The matrices/vectors have the following shapes:
        input_vector:                     input_dim x 1
        w_1:                     hidden_dim x input_dim
        b_1:                             hidden_dim x 1
        w_2:                    output_dim x hidden_dim
        b_2:                             output_dim x 1
        output_vector:                   output_dim x 1

        :param input_dim: Dimensionality of the input vector
        :param hidden_dim: Dimensionality of the hidden layer
        :param output_dim: Dimensionality of the output layer
        :param loss_func: Loss function
        """
        super(ThermoNet, self).__init__()

        # Save the input
        self.db_2 = 0
        self.dw_2 = 0
        self.db_1 = 0
        self.dw_1 = 0
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize the weight matrices and the bias
        self.w_1 = np.random.uniform(low=0.01, high=0.01, size=(self.hidden_dim, self.input_dim))
        self.b_1 = np.random.uniform(low=0.01, high=0.01, size=(self.hidden_dim, 1))
        self.w_2 = np.random.uniform(low=0.01, high=0.01, size=(self.output_dim, self.hidden_dim))
        self.b_2 = np.random.uniform(low=0.01, high=0.01, size=(self.output_dim, 1))

        # If a loss function is specified, then use this, else use the mean squared error
        if loss_func is None:
            self.loss_func = lambda out, real: np.mean((out - real) ** 2)
        else:
            self.loss_func = loss_func

    def __call__(self, x, g):
        return self.forward(x, g)

    def forward(self, x, g):
        """
        Forward pass of the network

        :param x: temperature values
        :param g: actual values for the Gibbs energy
        :return: mean squared error, cache (input x)
        """

        r = 8.134  # Universal gas constant
        # Learnable parameters of the activation function TODO: Make the parameters actually learnable
        e0 = 1
        theta_e = 1
        a = 1
        b = 1

        # Define the activation function
        def act_fun(s):
            act = e0 + 3 / 2 * r * theta_e + 3 * r * s * np.log(1 - np.exp(-theta_e / s)) - 1 / 2 * a * s ** 2 \
            - 1 / 6 * b * s ** 3

            return act

        # Preactivations of first layer
        s_beta = self.w_1 @ x + self.b_1

        # Activations of first layer
        s_beta_act = act_fun(s_beta)

        # Output of the network
        output = self.w_2 @ s_beta_act + self.b_2

        # Store the input for the backward pass
        cache = x

        # Compute the loss
        loss = self.loss_func(x, g)
        print(loss)

        return loss, cache

    def zero_grad(self):
        """
        Resets/initializes all gradients to 0

        :return:
        """

        self.dw_1 = 0
        self.db_1 = 0
        self.dw_2 = 0
        self.db_2 = 0

    def update(self, lr):
        """
        Updates the parameters of the network (SGD optimizer)
        :param lr: learning rate
        :return:
        """

        self.w_1 -= lr * self.dw_1
        self.b_1 -= lr * self.db_1
        self.w_2 -= lr * self.dw_2
        self.b_2 -= lr * self.db_2

    def backward(self, loss, cache):
        """
        Backward pass of the network
        :param loss: loss
        :param cache: stored inputs of the forward pass
        :return:
        """

        x = cache
