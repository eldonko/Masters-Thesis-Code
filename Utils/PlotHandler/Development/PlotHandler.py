import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Neural_Nets.ThermoDataset.Development.ThermoDataset import ThermoDataset


class PlotHandler:
	"""
	PlotHandler handles the plotting of different data
	"""
	def __init__(self, net='Thermo'):
		super(PlotHandler, self).__init__()

		self.net = net

	def properties_temp(self, net, element, phase, start_temp=200, end_temp=2000, scaling=True, unscale_output=True):
		"""
		Plots Gibbs energy, entropy, enthalpy and heat capacity over the temperature
		:param net: (trained) neural network which outputs shall be plotted
		:param element: element to load the dataset for
		:param phase: phase to load the dataset for
		:param start_temp: low value of the temperature interval
		:param end_temp: high value of the temperature interval
		:param scaling: determines if inputs should be scaled or not
		:param unscale_output: determines of outputs should be rescaled to original interval
		:return:
		"""

		dataset = ThermoDataset(element, phase, start_temp, end_temp, scaling=scaling)
		temp, gibbs, entropy, enthalpy, heat_cap = dataset.get_data()

		if self.net == 'Laenge':
			gibbs_p, entropy_p, enthalpy_p, heat_cap_p = net(temp, temp, temp, temp)
		elif self.net == 'Thermo':
			gibbs_p = net(temp)
			#entropy_p, enthalpy_p, heat_cap_p = net.output_all(temp_range)

		gibbs_p = gibbs_p.detach()
		#entropy_p = entropy_p.detach()
		#enthalpy_p = enthalpy_p.detach()
		#heat_cap_p = heat_cap_p.detach()

		def plot_property(prop_t, prop_p):
			plt.figure()
			plt.scatter(temp, prop_t, s=0.3, c='blue', label='True')
			plt.scatter(temp, prop_p, s=0.3, c='red', label='Prediction')
			plt.grid()
			plt.legend()
			plt.show()

		# Unscale output
		if scaling and unscale_output:
			temp_max, gibbs_max = dataset.get_maximum()
			gibbs_p *= gibbs_max
			temp *= temp_max

		plot_property(gibbs, gibbs_p)

		if self.net == 'Laenge':
			entropy_p = entropy_p.detach()
			enthalpy_p = enthalpy_p.detach()
			heat_cap_p = heat_cap_p.detach()
			plot_property(entropy, entropy_p)
			plot_property(enthalpy, enthalpy_p)
			plot_property(heat_cap, heat_cap_p)