import numpy as np
import random
import torch
from torch.autograd import grad


class DataGenerator(object):
    """
    Creates random function value and finds common tangents. The results serve as training and testing data for a neural
    network which learns to find common tangents between to functions.
    """
    def __init__(self):
        super(DataGenerator, self).__init__()

        # Set of basic functions that can be used to create functions
        self.funcs = {0: lambda x, a, c: c * (x - a) ** 2, 1: lambda x, a, c: c * (x - a) ** 3,
                      2: lambda x, a, c: c * (x - a) ** 4}

        # Linear offset
        self.l_off = lambda x, a, c: c * (x - a)

        # Constant offset
        self.c_off = lambda c: c

        # Get the range from 0 to 1
        self.range = torch.arange(1e-6, 100, step=1, requires_grad=True)/100

    def create_function(self):

        # Get number of basic functions from a poisson distribution
        rates = torch.rand(1) * 2  # rate parameter between 0 and 5
        poisson = int(torch.poisson(rates)) + 1  # add 1 to exclude 0 from the possible outcomes

        # Restrict the number of basic functions to the length of self.funcs
        if poisson > len(self.funcs):
            poisson = len(self.funcs)

        # Get random indices of functions in self.funcs with length poisson
        indices = torch.randperm(len(self.funcs))[:poisson].tolist()
        print(indices)

        # Get the coefficients
        cfs = torch.randint(1000, size=(poisson, ))

        # Get the function by adding the basic functions multiplied with the coefficients
        f = 0
        for i, c in zip(indices, cfs):
            a = self.range[torch.randint(high=len(self.range), size=(1,))]
            o = self.funcs[i](self.range, a, 1)
            f += o

        # Add random constant offset
        a = self.range[torch.randint(high=len(self.range), size=(1,))]
        f += self.l_off(self.range, a, 1) + self.c_off(0)

        df = grad(f, self.range, grad_outputs=torch.ones_like(f))[0]

        return f, df
