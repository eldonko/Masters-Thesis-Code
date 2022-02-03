import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class PlotHandler:
	"""
	PlotHandler handles the plotting of different data
	"""
	def __init__(self, net='Thermo'):
		super(PlotHandler, self).__init__()

		self.net = net

	def properties_temp(self, net, dataset, start_temp=200, end_temp=2000, input_scaling=False):
		"""
		Plots Gibbs energy, entropy, enthalpy and heat capacity over the temperature
		:param net: (trained) neural network which outputs shall be plotted
		:param dataset: dataset containing the actual data
		:param start_temp: low value of the temperature interval
		:param end_temp: high value of the temperature interval
		:param input_scaling: determines if inputs should be scaled or not
		:return:
		"""

		temp_range = torch.tensor(list(range(start_temp, end_temp)), dtype=torch.float64).unsqueeze(-1)

		temp, gibbs, entropy, enthalpy, heat_cap = None, None, None, None, None

		dataloader = DataLoader(dataset, batch_size=len(dataset))
		if self.net == 'Laenge':
			for t, g, s, h, c in dataloader:
				temp, gibbs, entropy, enthalpy, heat_cap = t, g, s, h, c
		elif self.net == 'Thermo':
			for t, g in dataloader:
				temp, gibbs = t, g

		if start_temp < temp.min():
			temp_range = torch.tensor(list(range(int(temp.min()), end_temp)), dtype=torch.float64).unsqueeze(-1)

		# Input scaling
		if input_scaling:
			temp_range /= temp_range.max()

		if self.net == 'Laenge':
			gibbs_p, entropy_p, enthalpy_p, heat_cap_p = net(temp_range, temp_range, temp_range, temp_range)
		elif self.net == 'Thermo':
			gibbs_p = net(temp_range)
			entropy_p, enthalpy_p, heat_cap_p = net.output_all(temp_range)

		gibbs_p = gibbs_p.detach()
		entropy_p = entropy_p.detach()
		enthalpy_p = enthalpy_p.detach()
		heat_cap_p = heat_cap_p.detach()

		def plot_property(temp_p, prop_p, temp_t, prop_t):
			plt.figure()
			plt.scatter(temp_t, prop_t, s=0.3, c='blue')
			plt.grid()
			plt.scatter(temp_p, prop_p, s=0.3, c='red')
			plt.show()

		plot_property(temp, gibbs, temp_range, gibbs_p)

		if self.net == 'Laenge':
			plot_property(temp, entropy, temp_range, entropy_p)
			plot_property(temp, enthalpy, temp_range, enthalpy_p)
			plot_property(temp, heat_cap, temp_range, heat_cap_p)