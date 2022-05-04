import matplotlib.pyplot as plt
import torch

from thermonet.dataset import ThermoDatasetNew, ThermoDataset
from laenge.dataset import LaengeDataset


class PlotHandler(object):
    """
    PlotHandler handles the plotting of different data

    Parameters
    ----------
    net : str
    	Which network was used. Can be either 'Thermo' or 'Laenge'. (Default value = 'Thermo')
	"""

    def __init__(self, net='Thermo'):
        super(PlotHandler, self).__init__()

        self.net = net

    def properties_temp(self, net, element, phase, start_temp=200, end_temp=2000, scaling=True, unscale_output=True):
        """Plots Gibbs energy, entropy, enthalpy and heat capacity over the temperature

        Parameters
        ----------
        net : torch.nn.Module
            trained neural network which outputs shall be plotted
        element : str
            element to load the dataset for
        phase : str
            phase to load the dataset for
        start_temp : int
            low value of the temperature interval (Default value = 200)
        end_temp : int
            high value of the temperature interval (Default value = 2000)
        scaling : bool
            determines if inputs should be scaled or not (Default value = True)
        unscale_output : bool
            determines of outputs should be rescaled to original interval (Default value = True)

        Returns
        -------

        """
        if self.net == 'Laenge':
            dataset = LaengeDataset(element, phase, start_temp, end_temp)
            temp, gibbs, entropy, enthalpy, heat_cap = dataset.get_data()
            gibbs, enthalpy = gibbs/1000, enthalpy/1000
            gibbs_p, entropy_p, enthalpy_p, heat_cap_p = net(temp, temp, temp, temp)
            entropy_p *= 1000
            heat_cap_p *= 1000
        elif self.net == 'Thermo':
            dataset = ThermoDataset(element, phase, start_temp, end_temp)
            temp, gibbs, entropy, enthalpy, heat_cap = dataset.get_data()
            gibbs_p = net(temp)
            entropy_p, enthalpy_p, heat_cap_p = net.output_all(temp)

        gibbs_p = gibbs_p.detach()
        entropy_p = entropy_p.detach()
        enthalpy_p = enthalpy_p.detach()
        heat_cap_p = heat_cap_p.detach()
        temp = temp.detach()

        # Unscale output
        if scaling and unscale_output:
            temp_max, gibbs_max = dataset.get_maximum()
            gibbs_p *= gibbs_max
            gibbs *= gibbs_max
            temp *= temp_max
        elif scaling:
            entropy /= abs(entropy).max()
            enthalpy /= abs(enthalpy).max()

        fig, axes = plt.subplots(4, sharex=False, figsize=(7, 28))
        self.plot_property(temp, gibbs, gibbs_p, axes[0], 'Gibbs energy [kJ/mol] over temperature in K')
        self.plot_property(temp, entropy, entropy_p, axes[1], 'Entropy [J/(mol K)] over temperature in K')
        self.plot_property(temp, enthalpy, enthalpy_p, axes[2], 'Enthalpy [kJ/mol] over temperature in K')
        self.plot_property(temp, heat_cap, heat_cap_p, axes[3], 'Heat capacity [J/(mol K)] over temperature in K')
        plt.show()

    @staticmethod
    def plot_property(temp, prop_t, prop_p, ax, title):
        """

        Parameters
        ----------
        temp : np.ndarray
            temperature
        prop_t : np.ndarray
            true values of property
        prop_p : np.ndarray
            predicted values of property
        ax : matplotlib.pyplot axes
            ax to plot on
        title : str
            figure title

        Returns
        -------

        """
        ax.scatter(temp, prop_t, s=0.3, c='blue', label='True')
        ax.scatter(temp, prop_p, s=0.3, c='red', label='Prediction')
        ax.grid()
        ax.legend()
        ax.set_title(title)

    def properties_temp_modified(self, net, element, phase, start_temp=200, end_temp=2000):
        """

        Parameters
        ----------
       net : torch.nn.Module
            trained neural network which outputs shall be plotted
        element : str
            element to load the dataset for
        phase : str
            phase to load the dataset for
        start_temp : int
            low value of the temperature interval (Default value = 200)
        end_temp : int
            high value of the temperature interval (Default value = 2000)

        Returns
        -------

        """
        dataset = ThermoDataset(element, start_temp, end_temp, phases=[phase])
        temp, targets = dataset.get_data()

        predictions = net(temp.float())

        predictions = predictions.detach()
        temp = temp.detach()

        fig, axes = plt.subplots(4, sharex=True, figsize=(7, 28))
        self.plot_property(temp, targets[:, 0], predictions[:, 0], axes[0], 'Gibbs energy over temperature')
        self.plot_property(temp, targets[:, 1], predictions[:, 1], axes[1], 'Entropy over temperature')
        self.plot_property(temp, targets[:, 2], predictions[:, 2], axes[2], 'Enthalpy over temperature')
        self.plot_property(temp, targets[:, 3], predictions[:, 3], axes[3], 'Heat capacity over temperature')
        plt.show()

    def properties_temp_modified_new(self, net, element, phase, start_temp=200, end_temp=2000):
        """

        Parameters
        ----------
       net : torch.nn.Module
            trained neural network which outputs shall be plotted
        element : str
            element to load the dataset for
        phase : str
            phase to load the dataset for
        start_temp : int
            low value of the temperature interval (Default value = 200)
        end_temp : int
            high value of the temperature interval (Default value = 2000)

        Returns
        -------

        """
        dataset = ThermoDatasetNew(element, start_temp, end_temp, inp_phases=[phase]).get_data()

        predictions = net(torch.tensor(dataset[:, [0, -2, -1]]).float())

        predictions = predictions.detach()
        temp = dataset[:, [0]]

        fig, axes = plt.subplots(4, sharex=False, figsize=(7, 28))
        self.plot_property(temp, dataset[:, [1]], predictions[:, 0], axes[0], 'Gibbs energy [kJ/mol] over temperature in K')
        self.plot_property(temp, dataset[:, [2]], predictions[:, 1], axes[1], 'Entropy [J/(mol K)] over temperature in K')
        self.plot_property(temp, dataset[:, [3]], predictions[:, 2], axes[2], 'Enthalpy [kJ/mol] over temperature in K')
        self.plot_property(temp, dataset[:, [4]], predictions[:, 3], axes[3], 'Heat capacity [J/(mol K)] over temperature in K')
        plt.show()
