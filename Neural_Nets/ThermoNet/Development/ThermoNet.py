import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
import numpy as np
from Data_Handling.SGTEHandler.Development.SGTEHandler import SGTEHandler
from Neural_Nets.ThermoNetActFuncs.Development.ThermoNetActFuncs import ChenSundman, Softplus, Sigmoid


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

	def __init__(self, hidden_dim=16, act_func=Sigmoid()):
		super(ThermoRegressionNet, self).__init__()

		self.layer_1 = Linear(1, hidden_dim)
		self.act_1 = act_func
		self.layer_2 = Linear(hidden_dim, 1)

		# Initialize network
		w_low = -1000
		w_high = 1000
		b_low = -10
		b_high = 10
		nn.init.uniform_(self.layer_1.weight, a=w_low, b=w_high)
		nn.init.uniform_(self.layer_1.bias, a=b_low, b=b_high)
		nn.init.uniform_(self.layer_2.weight, a=w_low, b=w_high)
		nn.init.uniform_(self.layer_2.bias, a=b_low, b=b_high)

		print(self.layer_1.weight.std())
		print(self.layer_2.weight.std())

		inp = torch.tensor([[100.]])
		print(self.layer_2.weight @ Sigmoid()(inp @ self.layer_1.weight.T + self.layer_1.bias).T + self.layer_2.bias)

		inp = torch.tensor([[2345.]])
		print(self.layer_2.weight @ Sigmoid()(inp @ self.layer_1.weight.T + self.layer_1.bias).T + self.layer_2.bias)

	def forward(self, temp):
		"""

		:param temp: Temperature
		:return:
		"""
		s = self.layer_1(temp.float())
		a = self.act_1(s)

		# Gibbs energy
		gibbs = self.layer_2(a)

		print(gibbs.std())
		print(gibbs.mean())

		return gibbs

	def output_all(self, temp):
		"""

		:param temp: Temperature
		:return: entropy, enthalpy and heat capacity for given temperature
		"""

		# Entropy
		s = self.layer_1(temp.float())
		entropy = -self.layer_2.weight * self.act_1.first_derivative(s) @ self.layer_1.weight

		# Enthalpy
		enthalpy = self.forward(temp) + temp * entropy

		# Heat capacity
		heat_cap = -temp * self.layer_2.weight * self.act_1.second_derivative(s) @ self.layer_1.weight.double() ** 2

		return entropy, enthalpy, heat_cap


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

        self._samples = [(t, g) for t, g in zip(temp, gibbs)]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i: int):
        return self._samples[i]
