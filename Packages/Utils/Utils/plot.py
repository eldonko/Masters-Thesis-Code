import matplotlib.pyplot as plt

from thermonet.dataset import ThermoDataset


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

        dataset = ThermoDataset(element, phase, start_temp, end_temp, scaling=scaling)
        temp, gibbs, entropy, enthalpy, heat_cap = dataset.get_data()

        if self.net == 'Laenge':
            gibbs_p, entropy_p, enthalpy_p, heat_cap_p = net(temp, temp, temp, temp)
        elif self.net == 'Thermo':
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

        fig, axes = plt.subplots(4, sharex=True, figsize=(7, 28))
        self.plot_property(temp, gibbs, gibbs_p, axes[0], 'Gibbs energy over temperature')
        self.plot_property(temp, entropy, entropy_p, axes[1], 'Entropy over temperature')
        self.plot_property(temp, enthalpy, enthalpy_p, axes[2], 'Enthalpy over temperature')
        self.plot_property(temp, heat_cap, heat_cap_p, axes[3], 'Heat capacity over temperature')
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
        dataset = ThermoDataset(element, phase, start_temp, end_temp)
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
