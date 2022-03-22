import pkg_resources

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from ..dataset.encoding import Encoder


class ThermoClassifier(nn.Module):
    """
    Combines the element and phase classifiers to make predictions on both at the same time
    """

    def __init__(self):
        super(ThermoClassifier, self).__init__()

        # Load the element classifier
        stream = pkg_resources.resource_stream(__name__, '../elements/models/ElementClassifier_9782_2.pth')
        self.ec = torch.load(stream)

        # Load the phase classifier
        stream = pkg_resources.resource_stream(__name__, '../phases/models/PhaseClassifier_9563.pth')
        self.pc = torch.load(stream)

    def forward(self, x):
        """
        Forward pass of the network. First the measurements are passed through the element classifier in the package,
        then each measurement pair on its own through the phase classifier

        Parameters
        ----------
        x : torch.Tensor
            network input

        Returns
        -------
        torch.Tensor
            network output

        """
        # Predict the element
        element = self.ec(x).argmax(dim=-1)
        element = element.reshape(x.shape[0], 1, 1)

        # Add the element to x as the phase classifier needs it as input
        element_t = element * torch.ones((x.shape[0], x.shape[1], 1))
        x = torch.cat((x, element_t), -1)

        # Predict the phases
        phases = torch.zeros((x.shape[0], x.shape[1], 1))
        for i in range(x.shape[1]):
            # Get the predictions
            phase = self.pc(x[:, i]).argmax(dim=-1).unsqueeze(-1)
            # Store the predictions separately and concat later
            phases[:, i, :] = phase

        out = torch.cat((x, phases), -1)

        return out

    @staticmethod
    def decode(output):
        """
        Decodes the output

        Parameters
        ----------
        output : torch.Tensor
            network output

        Returns
        -------
        list of str
            decoded network output

        """

        elements = output[:, :, 2][:, 0]
        phases = output[:, :, 3]

        decoded = []

        for el, ph in zip(elements, phases):
            # Decode element
            element_dec = Encoder()(int(el.item()))

            # Decode phases
            ph_series = pd.Series(ph, dtype=np.int32)
            phases_dec = Encoder()(ph_series)
            decoded.append((element_dec, phases_dec.tolist()))

        return decoded