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

	def __init__(self, hidden_layers=1, hidden_dim=16, act_func=Sigmoid()):
		super(ThermoRegressionNet, self).__init__()

		self.layers = nn.ModuleList()

		# Input layer
		il = Linear(1, hidden_dim)
		nn.init.xavier_uniform_(il.weight)
		self.layers.append(il)

		# Input activation
		self.layers.append(act_func)

		# Hidden layers
		for i in range(hidden_layers):
			hl = Linear(hidden_dim, hidden_dim)
			nn.init.xavier_uniform_(hl.weight)
			self.layers.append(hl)
			self.layers.append(act_func)

		# Output layer
		ol = Linear(hidden_dim, 1)
		nn.init.xavier_normal_(ol.weight)
		self.layers.append(ol)

	def forward(self, x):
		"""

		:param x: Temperature
		:return:
		"""

		for layer in self.layers:
			x = layer(x.float())

		return x

	def output_all(self, temp):
		"""

		:param temp: Temperature
		:return: entropy, enthalpy and heat capacity for given temperature
		"""

		# Entropy
		s = self.layers[0](temp.float())
		entropy = -self.layers[-1].weight * self.input_act.first_derivative(s) @ self.input_layer.weight

		# Enthalpy
		enthalpy = self.forward(temp) + temp * entropy

		# Heat capacity
		heat_cap = -temp * self.output_layer.weight * self.input_act.second_derivative(s) @ self.input_layer.weight.double() ** 2

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
