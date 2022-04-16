import importlib_resources
import json
import os
import pkg_resources

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FunctionDataset(Dataset):
    def __init__(self, n_functions=0, filename=None, o=None, step=0.01, overwrite=False):
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
        step : float
            step size of the x-range
        overwrite : bool
            whether to overwrite the existing data in filename
        """

        super(FunctionDataset, self).__init__()

        self.n_functions = n_functions
        self.filename = filename
        self.o = o
        self.step = step
        self.overwrite = overwrite

        self._samples = []  # Samples
        self._parameters = []  # Function parameters

    def __getitem__(self, i: int):
        return self._samples[i]

    def __len__(self):
        return len(self._samples)

    def create_functions(self):
        # Generate data
        if not os.path.exists(self.filename) or self.overwrite:
            for i in range(self.n_functions):
                self.generate_data()

            df = pd.DataFrame(self._parameters)
            df.to_csv(self.filename)
        else:
            self._parameters = pd.read_csv(self.filename)
            self._parameters = self._parameters.to_dict('records')

            for k in self._parameters:
                # Get function + derivative values
                func, der = self.get_values(**k)

                # Scale the function by the local absolute maximum
                loc_max = max(abs(torch.max(func)), abs(torch.min(func)))
                func /= loc_max
                der /= loc_max

                # Append the new values to the existing samples
                values = torch.hstack((func.unsqueeze(-1), der.unsqueeze(-1)))
                self._samples.append(values)

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
        t_max = 3000
        s_max = 10
        k['T'] = np.random.rand() * t_max
        k['tm1'] = np.random.rand() * t_max
        k['tm2'] = np.random.rand() * t_max
        k['tm'] = (k['tm1'] + k['tm2']) / 2
        k['s1'] = np.random.rand() * s_max
        k['s2'] = np.random.rand() * s_max

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

        self._parameters.append(k)

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

    def base_function(self, x=None, **kwargs):
        r = 8.314

        if x is None:
            x = torch.arange(1e-10, 1., step=self.step)
        t = kwargs['T'] if not isinstance(kwargs['T'], torch.Tensor) else kwargs['T'].unsqueeze(-1)
        a = kwargs['a'] if not isinstance(kwargs['a'], torch.Tensor) else kwargs['a'].unsqueeze(-1)
        s_1 = kwargs['s1'] if not isinstance(kwargs['s1'], torch.Tensor) else kwargs['s1'].unsqueeze(-1)
        s_2 = kwargs['s2'] if not isinstance(kwargs['s2'], torch.Tensor) else kwargs['s2'].unsqueeze(-1)
        tm_1 = kwargs['tm1'] if not isinstance(kwargs['tm1'], torch.Tensor) else kwargs['tm1'].unsqueeze(-1)
        tm_2 = kwargs['tm2'] if not isinstance(kwargs['tm2'], torch.Tensor) else kwargs['tm2'].unsqueeze(-1)

        return r * t * (x * torch.log(x) + (1 - x) * torch.log(1 - x)) + a * x * (1 - x) + s_1 * (tm_1 - t) * \
               (1 - x) + s_2 * (tm_2 - t) * x

    def first_derivative(self, x=None, **kwargs):
        r = 8.314

        if x is None:
            x = torch.arange(1e-10, 1., step=self.step)
        t = kwargs['T'] if not isinstance(kwargs['T'], torch.Tensor) else kwargs['T'].unsqueeze(-1)
        a = kwargs['a'] if not isinstance(kwargs['a'], torch.Tensor) else kwargs['a'].unsqueeze(-1)
        s_1 = kwargs['s1'] if not isinstance(kwargs['s1'], torch.Tensor) else kwargs['s1'].unsqueeze(-1)
        s_2 = kwargs['s2'] if not isinstance(kwargs['s2'], torch.Tensor) else kwargs['s2'].unsqueeze(-1)
        tm_1 = kwargs['tm1'] if not isinstance(kwargs['tm1'], torch.Tensor) else kwargs['tm1'].unsqueeze(-1)
        tm_2 = kwargs['tm2'] if not isinstance(kwargs['tm2'], torch.Tensor) else kwargs['tm2'].unsqueeze(-1)

        return r * t * (torch.log(x) - torch.log(1 - x)) + a * (1 - 2 * x) - s_1 * tm_1 + s_2 * tm_2

    def second_derivative(self, x=None, **kwargs):
        r = 8.314

        if x is None:
            x = torch.arange(1e-10, 1., step=self.step)
        t = kwargs['T'] if not isinstance(kwargs['T'], torch.Tensor) else kwargs['T'].unsqueeze(-1)
        a = kwargs['a'] if not isinstance(kwargs['a'], torch.Tensor) else kwargs['T'].unsqueeze(-1)

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


class FunctionPairDataset(FunctionDataset):
    def __init__(self, n_functions=0, filename=None, step=0.01, overwrite=False):
        super(FunctionPairDataset, self).__init__(n_functions, filename, o=0, step=step, overwrite=overwrite)

        self.n_functions = n_functions
        self.filename = filename
        self.step = step
        self.overwrite = overwrite

        self._samples = []  # Samples
        self._parameters = []  # Function parameters
        self._scale = []  # Scaling factors

    def __getitem__(self, i):
        return self._samples[i], self._parameters[i], self._scale[i]

    def create_functions(self):
        # Generate data
        if not os.path.exists(self.filename) or self.overwrite:
            for i in range(self.n_functions):
                self.generate_data()

            df = pd.DataFrame(self._parameters)
            df.to_csv(self.filename)

    def generate_data(self):
        """
        Generates pairs of Gibbs energy functions based on "Statistische Konstitutionsanalyse - Thermodynamische
        Konstitutionsmorphologie[Mager, T.; Lukas, H.L.; Petzow, G.]

        """

        ks = []
        funcs = []
        ders = []
        global_max = None

        # Get two functions
        k = dict()
        k['o'] = 0

        # Generate function parameters randomly
        t_max = 3000
        s_max = 10
        k['T'] = np.random.rand() * t_max
        k['tm1'] = np.random.rand() * t_max
        k['tm2'] = np.random.rand() * t_max
        k['tm'] = (k['tm1'] + k['tm2']) / 2

        a_sum, a_diff = self.generate_a()
        a_1, a_2 = self.a(a_sum, a_diff, k['tm'])

        def get_function(a, s1_, s2_):
            k_ = k.copy()
            k_['a'] = a
            k_['s1'] = s1_
            k_['s2'] = s2_
            ks.append(k_)

            # Get function + derivative values
            f, d = self.get_values(**k_)
            funcs.append(f), ders.append(d)

            # Get functions maximum
            return max(abs(torch.max(f)), abs(torch.min(f)))

        # Get the two functions and in the same go the global maximum
        s1 = np.random.rand() * s_max
        s2 = np.random.rand() * s_max
        global_max = max(get_function(a_1, s1, s2), get_function(a_2, 0, 0))

        # Scale the functions by the absolute maximum
        for func, der in zip(funcs, ders):
            func /= global_max
            der /= global_max

        # Append the new values to the existing samples
        values = torch.hstack((funcs[0].unsqueeze(-1), funcs[1].unsqueeze(-1)))
        self._samples.append(values)

        self._parameters.append(ks)

        self._scale.append(global_max)
