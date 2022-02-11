import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Softmax, LSTM, Sequential


class ThermoClassifier(nn.Module):
	"""
	ThermoClassifier classifies thermodynamic measurement data in the sense that given measurement data for the
	Gibbs energy, entropy, enthalpy or the heat capacity at certain temperatures, it is able to tell which element
	and which phase of the element this measurement data belongs to.
	"""

	def __init__(self):
		"""
		input_size: 2, (temperature, measurement)
		"""
		super(ThermoClassifier, self).__init__()

		# LSTM for many-to-one input
		self.input_size = 2
		self.hidden_size_lstm = 16
		self.lstm = LSTM(input_size=self.input_size, hidden_size=self.hidden_size_lstm, batch_first=False)

		# Fully connected net for classification
		self.num_classes = 5
		self.hidden_size_linear = 4 * self.hidden_size_lstm

		self.fc = Sequential(
			Linear(self.hidden_size_lstm, self.hidden_size_linear),
			ReLU(),
			Linear(self.hidden_size_linear, self.hidden_size_linear),
			ReLU(),
			Linear(self.hidden_size_linear, self.num_classes),
			Softmax()
		)

	def forward(self, x):
		lstm_out, _ = self.lstm(x)

		linear_in = lstm_out[-1]

		return self.fc(linear_in)