import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Softmax, LSTM, Sequential, BatchNorm1d, Tanh
import pandas as pd
import os


class PhaseClassifier(nn.Module):
    """
	PhaseClassifier classifies thermodynamic measurement data in the sense that given measurement data for the
	Gibbs energy, entropy, enthalpy or the heat capacity at certain temperatures, it is able to tell which phase of
	a GIVEN element this measurement data belongs to.
	"""

    def __init__(self, element, train=False, measurement='G', element_data_path=None, models_path=None):
        """
		input_size: 2, (temperature, measurement)

		:param element: Which element the measurements are from (and the phase should be predicted for)
		:param train: If train is True, a network with random weights is generated and trained, if train is False, an
		already trained network for this element and measurement type will be loaded.
		:param measurement: Defines for which properties the classification should be made. Must be one of 'G', 'S', 'H'
		or 'C'.
		"""
        super(PhaseClassifier, self).__init__()

        # Input checking
        if element_data_path is None:
            self.element_data_path = r"C:\Users\danie\Documents\Montanuni\Masterarbeit\4_Daten\Elements.xlsx"
        else:
            self.element_data_path = element_data_path

        if models_path is None:
            self.models_path = r"C:\Users\danie\Documents\Montanuni\Masterarbeit\5_Programmcodes\Neural_Nets" \
                          r"\Classification\PhaseClassifier\Models"
        else:
            self.models_path = models_path

        self.element_data = None
        self.load_element_data()  # Load element data
        assert element in self.element_data.columns.values
        assert measurement in ['G', 'S', 'H', 'C']

        self.element = element
        self.measurement = measurement

        # Fully connected net for classification
        self.num_classes = self.element_data[element].sum()
        self.in_features = 2
        self.hidden_size_linear = 32

        # For training, create a new network, else load a pre-trained network
        if train or (not train and not os.path.exists(os.path.join(self.models_path, element + '_' + measurement))):
            # Error handling
            if not train and not os.path.exists(os.path.join(self.models_path, element + '_' + measurement)):
                print('New network had to be initialized as no network exists for this element and measurement. '
                      'Training necessary!')

            self.fc = Sequential(
                Linear(self.in_features, self.hidden_size_linear),
                Tanh(),
                Linear(self.hidden_size_linear, self.hidden_size_linear),
                Tanh(),
                # Linear(self.hidden_size_linear, self.hidden_size_linear),
                # Tanh(),
                Linear(self.hidden_size_linear, self.num_classes)
            )

            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
                    m.bias.data.fill_(0.01)

            self.fc.apply(init_weights)
        else:
            self.load_network()

    def forward(self, x):
        return self.fc(x)

    def load_element_data(self):
        """
        Loads the element data from the excel sheet provided at self.element_data_path
        """

        self.element_data = pd.read_excel(self.element_data_path, sheet_name='Phases', index_col='Index')

    def load_network(self):
        """
        Change the path where already trained models are stored. Needs to be a directory
        """
        model_name = self.element + '_' + self.measurement  # e.g., Fe_G
        self.fc = torch.load(os.path.join(self.models_path, model_name))