import torch
from Data_Handling.SGTEHandler.Development.SGTEHandler import SGTEHandler
from torch.utils.data import Dataset
import numpy as np


class ThermoDatasetModified(Dataset):
    def __init__(self, element, phase, start_temp=200, end_temp=2000, step=1):
        super(ThermoDatasetModified, self).__init__()

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
        return self.temp.unsqueeze(-1), self.targets
