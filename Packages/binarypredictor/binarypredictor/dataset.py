import importlib_resources
import json
import os
import pkg_resources

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FunctionDataset(Dataset):
    def __init__(self, n_functions=0, filename=None, o=None):
        """

        Parameters
        ----------
        n_functions : int
            number of functions in the dataset
        filename : str
            filename to store the function parameters at
        o : int
            if None, derivative order of base function is chosen randomly, if integer (0 or 1) is provided, only
            this order will be considered
        """
        super(FunctionDataset, self).__init__()

        self.o = o

        self._samples = []  # Samples
        self.parameters = []  # Function parameters

        # Generate data
        if not os.path.exists(filename):
            for i in range(n_functions):
                self.generate_data()

            df = pd.DataFrame(self.parameters)
            df.to_csv(filename)
        else:
            self.parameters = pd.read_csv(filename)
            self.parameters = self.parameters.to_dict('records')

            for k in self.parameters:
                # Get function + derivative values
                func, der = self.get_values(**k)

                # Scale the function by the local absolute maximum
                loc_max = max(abs(torch.max(func)), abs(torch.min(func)))
                func /= loc_max
                der /= loc_max

                # Append the new values to the existing samples
                values = torch.hstack((func.unsqueeze(-1), der.unsqueeze(-1)))
                self._samples.append(values)

    def __getitem__(self, i: int):
        return self._samples[i]

    def __len__(self):
        return len(self._samples)

    def generate_data(self):
        """
        Generates Gibbs energy functions based on "Statistische Konstitutionsanalyse - Thermodynamische
        Konstitutionsmorphologie [Mager, T.; Lukas, H. L.; Petzow, G.]

        The data can be either for the base function or for its first derivative. Which to generate is chosen randomly
        and is because this data serves as input for a neural network which shall be able to handle both.

        """
        k = dict()

        # Decide if base function or first derivative
        if self.o is None:
            o = np.random.randint(0, 2)
        else:
            o = self.o
        k['o'] = o

        # Generate function parameters randomly
        k['T'] = np.random.rand() * 3000
        k['tm1'] = np.random.rand() * 3000
        k['tm2'] = np.random.rand() * 3000
        k['tm'] = (k['tm1'] + k['tm2']) / 2
        k['s1'] = 0
        k['s2'] = 0

        a_sum, a_diff = self.generate_a()
        a_1, a_2 = self.a(a_sum, a_diff, k['tm'])
        k['a'] = a_1

        # Get function + derivative values
        func, der = self.get_values(**k)

        # Scale the function by the local absolute maximum
        loc_max = max(abs(torch.max(func)), abs(torch.min(func)))
        func /= loc_max
        der /= loc_max

        # Append the new values to the existing samples
        values = torch.hstack((func.unsqueeze(-1), der.unsqueeze(-1)))
        self._samples.append(values)

        self.parameters.append(k)

    @staticmethod
    def generate_a():
        """
        Generate random A values. See above quoted paper for more information
        """
        # a_diff
        a_diff = np.random.randint(-2, 5)

        # A_sum
        a_sum_base = list(range(-2, 13, 2))
        if a_diff % 2 == 0:
            a_sum_base += list(range(1, 7, 2))
        a_sum = a_sum_base[np.random.randint(len(a_sum_base))]

        return a_sum, a_diff

    @staticmethod
    def a(a_sum, a_diff, TM):
        r = 8.314

        a_sum *= TM * r
        a_diff *= TM * r

        a_1 = (a_sum - a_diff) / 2
        a_2 = (a_sum + a_diff) / 2

        return a_1, a_2

    @staticmethod
    def base_function(**kwargs):
        r = 8.314

        x = torch.arange(1e-10, 1., step=0.01)
        t = kwargs['T']
        a = kwargs['a']
        s_1 = kwargs['s1']
        s_2 = kwargs['s2']
        tm_1 = kwargs['tm1']
        tm_2 = kwargs['tm2']

        return r * t * (x * torch.log(x) + (1 - x) * torch.log(1 - x)) + a * x * (1 - x) + s_1 * (tm_1 - t) * \
               (1 - x) + s_2 * (tm_2 - t) * x

    @staticmethod
    def first_derivative(**kwargs):
        r = 8.314

        x = torch.arange(1e-10, 1., step=0.01)
        t = kwargs['T']
        a = kwargs['a']
        s_1 = kwargs['s1']
        s_2 = kwargs['s2']
        tm_1 = kwargs['tm1']
        tm_2 = kwargs['tm2']

        return r * t * (torch.log(x) - torch.log(1 - x)) + a * (1 - 2 * x) - s_1 * tm_1 + s_2 * tm_2

    @staticmethod
    def second_derivative(**kwargs):
        r = 8.314

        x = torch.arange(1e-10, 1., step=0.01)
        t = kwargs['T']
        a = kwargs['a']

        return r * t * (1 / x + 1 / (1 - x)) - 2 * a

    def get_values(self, **kwargs):
        """
        Based on the randomly defined derivative order, return the function value and the values of its first
        derivative

        Returns
        -------
        torch.tensor, torch.tensor :
            function values, values of first derivative

        """

        o = kwargs['o']

        if o == 0:
            return self.base_function(**kwargs), self.first_derivative(**kwargs)
        elif o == 1:
            return self.first_derivative(**kwargs), self.second_derivative(**kwargs)
