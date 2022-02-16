import torch
from torch.utils.data import Dataset

from sgte.SGTEHandler import SGTEHandler


class ThermoDataset(Dataset):
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
    """
    def __init__(self, element, phase, start_temp=200, end_temp=2000, step=1.):
        super(ThermoDataset, self).__init__()

        sgte_handler = SGTEHandler(element)
        sgte_handler.evaluate_equations(start_temp, end_temp, 1e5, plot=False, phases=phase, entropy=True,
                                        enthalpy=True,
                                        heat_capacity=True, step=step)
        data = sgte_handler.equation_result_data

        # Get values
        self.temp = torch.tensor(data['Temperature'], dtype=torch.float64)
        data.iloc[:, 1] /= 1000
        data.iloc[:, 3] /= 1000

        self.targets = torch.Tensor(data.iloc[:, 1:].values)

        self._samples = list(zip(self.temp, self.targets))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i: int):
        return self._samples[i]

    def get_data(self):
        """
        Returns
        -------
        tuple of torch.tensors
            (temperature, target values) where target values are the Gibbs energy [kJ], the entropy [J/K], the enthalpy
            [kJ] and the heat capacity [J/K].
        """
        return self.temp.unsqueeze(-1), self.targets
