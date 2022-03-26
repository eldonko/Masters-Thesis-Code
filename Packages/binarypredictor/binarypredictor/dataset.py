import importlib_resources
import json
import os
import pkg_resources

import pandas as pd
import torch
from torch.utils.data import Dataset


class FunctionDataset(Dataset):
    def __init__(self):
        super(FunctionDataset, self).__init__()

        # Load the generated polynomial data
        with importlib_resources.open_text('binarypredictor.data', 'polyData2.json') as file:
            data = json.load(file)

        # Load samples into DataFrame
        self._samples = pd.DataFrame(data).T

        # Maximum polynomial degree
        self.max_degree = 4

        # Maximum number of common tangent points
        self.max_pts = 10

    def __getitem__(self, i: int):
        item = self._samples.iloc[i]

        # Zero-pad the lists in item so that all of them have the same shape
        p = [0] * (self.max_degree + 1 - len(item['p'])) + item['p']
        q = [0] * (self.max_degree + 1 - len(item['q'])) + item['q']
        pts = [i for a in item['pts'] for i in a] + [-1] * (self.max_pts * 2 - len(item['pts']) * 2)

        return torch.tensor(p), torch.tensor(q), torch.tensor(pts)

    def __len__(self):
        return len(self._samples)