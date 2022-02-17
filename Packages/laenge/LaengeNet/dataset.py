import torch
from torch.utils.data import Dataset

from sgte.handler import SGTEHandler


class LaengeDataset(Dataset):
    """
    The dataset for training and testing the ThermoNet

    Parameters
    ----------
    element : str
        element for which the dataset is created
    phase : str
        phase of the element for which the dataset is created
    start_temp : int
        lower value of temperature range (Default value = 200)
    end_temp : int
        upper value of temperature range (Default value = 2000)
    step : float
        step size of the SGTEHandler to create the data. The smaller step, the larger the dataset (Default value = 1.)
    scaling : bool
        whether the temperature and Gibbs energy should be scaled or not (Default value = False)
    """
    def __init__(self, element, phase, start_temp=200, end_temp=2000, step=1., scaling=False):
        super(LaengeDataset, self).__init__()

        sgte_handler = SGTEHandler(element)
        sgte_handler.evaluate_equations(start_temp, end_temp, 1e5, plot=False, phases=phase, entropy=True,
                                        enthalpy=True,
                                        heat_capacity=True, step=step)
        data = sgte_handler.equation_result_data

        # Get values
        temp = torch.tensor(data['Temperature'], dtype=torch.float64)
        gibbs = torch.tensor(data.iloc[:, 1])
        entropy = torch.tensor(data.iloc[:, 2])
        enthalpy = torch.tensor(data.iloc[:, 3])
        heat_cap = torch.tensor(data.iloc[:, 4])

        # Scale values to interval [0, 1]
        self.temp_max = temp.max()
        self.gibbs_max = abs(gibbs).max()
        if scaling:
            temp /= self.temp_max
            gibbs /= self.gibbs_max

        self._samples = [(t, g, s, h, c) for t, g, s, h, c in zip(temp, gibbs, entropy, enthalpy, heat_cap)]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i: int):
        return self._samples[i]

    def get_maximum(self):
        """

        Returns
        -------
        tuple of torch.tensors
            Maximum of actual (unscaled) temperature and maximum of actual (unscaled) Gibbs energy

        """
        return self.temp_max, self.gibbs_max

    def get_data(self):
        """
        Returns
        -------
        tuple of torch.tensors
            (temperature, gibbs, entropy, enthalpy, heat capacity)
        """
        temp, gibbs, entropy, enthalpy, heat_cap = zip(*self._samples)
        temp = torch.tensor(temp, requires_grad=False).unsqueeze(-1)
        gibbs = torch.tensor(gibbs, requires_grad=False)
        entropy = torch.tensor(entropy, requires_grad=False)
        enthalpy = torch.tensor(enthalpy, requires_grad=False)
        heat_cap = torch.tensor(heat_cap, requires_grad=False)

        return temp, gibbs, entropy, enthalpy, heat_cap
