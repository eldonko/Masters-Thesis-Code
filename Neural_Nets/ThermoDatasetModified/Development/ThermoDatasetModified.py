import torch
from Data_Handling.SGTEHandler.Development.SGTEHandler import SGTEHandler
from torch.utils.data import Dataset


class ThermoDatasetModified(Dataset):
    def __init__(self, element, phase, start_temp=200, end_temp=2000, step=1):
        super(ThermoDatasetModified, self).__init__()

        sgte_handler = SGTEHandler(element)
        sgte_handler.evaluate_equations(start_temp, end_temp, 1e5, plot=False, phases=phase, entropy=True,
                                        enthalpy=True,
                                        heat_capacity=True, step=step)
        data = sgte_handler.equation_result_data

        # Get values
        temp = torch.tensor(data['Temperature'], dtype=torch.float64)
        data.iloc[:, 1] /= 1000
        data.iloc[:, 3] /= 1000
        print(data.shape)
        prediction = torch.Tensor(data.iloc[:, 1:].values)

        self._samples = [(t, p) for t, p in zip(temp, prediction)]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i: int):
        return self._samples[i]

    def get_data(self):
        temp, gibbs, entropy, enthalpy, heat_cap = zip(*self._samples)
        temp = torch.tensor(temp, requires_grad=False).unsqueeze(-1)
        gibbs = torch.tensor(gibbs, requires_grad=False)
        entropy = torch.tensor(entropy, requires_grad=False)
        enthalpy = torch.tensor(enthalpy, requires_grad=False)
        heat_cap = torch.tensor(heat_cap, requires_grad=False)

        return temp, gibbs, entropy, enthalpy, heat_cap
