import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import *
import math
from DataHandler.Development import DataHandler


def main():
    limit = -10
    x_data, y_data = create_data(-limit, limit, 200)

    # Split the data into train set and test set
    test_inds, train_inds = split_data(x_data, 0.9, 0.1)

    batch_size = 256
    n_epochs = 50000
    trained_net = train(train_inds, (x_data, y_data), n_epochs, batch_size)

    test(test_inds, (x_data, y_data), trained_net)


def create_data(min_inp, max_inp, size_inp):
    #x = torch.zeros(size_inp, 1)
    #for i in range(size_inp):
    #    x[i] = torch.tensor(np.random.uniform(low=min_inp, high=max_inp))
    #y = underlying_function(x)

    # Load actual Cp data
    inp = r"C:\Users\danie\Documents\Montanuni\Masterarbeit\4 Daten\Cu_100_2000_1.TXT"
    source = 'FactSage'
    handler = DataHandler(source, inp)
    handler.show_table()
    x, y = handler.get_x_y_data('T(K)_S1', 'Cp(J/K)_S1')

    x = torch.Tensor(x)
    y = torch.Tensor(y)

    return x, y


def underlying_function(x_data):
    easy_function = True
    if easy_function:
        return x_data ** 2
    else:
        y_data = torch.zeros_like(x_data)
        for i, x in enumerate(x_data):
            y_data[i] = math.sin(x)
        return y_data


def split_data(data_inp, train_size, test_size):
    # torch.utils.data.random_split can only split a dataset into two subsets. Therefore, the
    # data is first split into train_val_set and test_set. After that, train_val_set is split
    # yet again to finally obtain three subsets.

    # Compute the number of files each set must have to obey the percentages specified.
    train_size = int(train_size * len(data_inp))
    test_size = int(test_size * len(data_inp))

    # Randomize the selection process
    shuffled_indices = np.random.permutation(len(data_inp))

    test_inds = shuffled_indices[:test_size]
    train_inds = shuffled_indices[test_size:]

    return test_inds, train_inds


def train(train_inds, train_data, n_epochs, batch_size):
    def stacking_function(stacking_list: list):
        x_data = torch.zeros((batch_size, 1))
        y_data = torch.zeros((batch_size, 1))

        for i, item in enumerate(stacking_list):
            x_data[i] = train_data[0][item]
            y_data[i] = train_data[1][item]

        return x_data, y_data

    ann = ANN()
    train_loader = DataLoader(train_inds, batch_size=batch_size, collate_fn=stacking_function, num_workers=0)
    optimizer = torch.optim.Adam(ann.parameters(), lr=0.001)
    losses = []

    for i in range(n_epochs):
        for inp, output in train_loader:
            predictions, alpha, beta = ann(inp)

            optimizer.zero_grad()
            loss = nn.MSELoss()(predictions, output)
            #print('loss: ', loss)
            loss.backward()
            optimizer.step()
            losses.append(loss)

    print('last loss: ', loss, 'alpha: ', alpha, 'beta: ', beta)
    print('number of epochs: ', n_epochs)

    return ann


def test(test_inds, test_data, trained_net):
    inputs = []
    targets = []
    predictions = []
    for index in test_inds:
        x, y = test_data[0][index], test_data[1][index]
        prediction, _, _ = trained_net(x)
        inputs.append(x)
        targets.append(y)
        predictions.append(prediction)

    predictions = np.array(predictions)
    inputs = np.array(inputs)
    plt.scatter(inputs, predictions)
    plt.scatter(inputs, targets)
    plt.show()

    limit = 20
    x_new, y_new = create_data(-limit, limit, 200)

    prediction, _, _ = trained_net(x_new)

    fig, axes = plt.subplots(2)
    axes[0].scatter(np.array(x_new), prediction.detach().numpy())
    #axes[0].scatter(np.array(x_new), np.array(y_new))

    loss_per_prediction = (prediction - y_new)
    axes[1].scatter(np.array(x_new), loss_per_prediction.detach().numpy())

    plt.show()


class CustomActivationFunctions(nn.Module):
    def __init__(self, alpha=None, beta=None):
        super(CustomActivationFunctions, self).__init__()

        if alpha is None:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha))

        if beta is None:
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.beta = nn.Parameter(torch.tensor(beta))

        self.alpha.requires_grad = True
        self.beta.requires_grad = True

    def forward(self, x):
        return self.alpha * x ** 2


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.Sigmoid(),
            nn.Linear(16, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )

        self.layer_1 = nn.Linear(1, 16)
        self.layer_2 = nn.Linear(16, 16)
        self.layer_3 = nn.Linear(16, 1)

        self.custom_activation_function = CustomActivationFunctions()

    def forward(self, x):
        output = self.custom_activation_function(self.layer_1(x))
        output = self.custom_activation_function(self.layer_2(output))
        output = self.layer_3(output)

        return output, self.custom_activation_function.alpha, self.custom_activation_function.beta
        #return self.layers(x)


if __name__ == '__main__':
    main()
