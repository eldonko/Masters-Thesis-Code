import numpy as np
import pandas as pd

from sgte.handler import SGTEHandler


class ElementDatasetCreator(object):
    """
    ElementDatasetCreator loads the data for all phases of an element from the sgte data. The dataset will be used
    by dataset to create the training, testing and optionally the validation dataset.

    """

    def __init__(self, label_range, element, temp_range=(200, 2000), measurement='G', seq_len=5, splits=(0.8, 0.2),
                 validation=False, stable_only=False):
        """
		Initializes the dataset

		label_range : tuple of ints (start label, end label)
		    the label range is passed by the DatasetCreator so that for every element and phase a unique label can be
		    created. Each phase of an element receives a label in the label range. If stable_only is True, than every
		    phase of an element gets the same label
		element : str
		    the element for which the data should be loaded.
		temp_range: tuple of ints (low_temp, high_temp)
		    the temperature range for which the data should be loaded.
		measurement: str
		    Defines for which properties the classification should be made. Must be one of 'G', 'S', 'H' or 'C'.
		    (Default value = 'G')
		seq_len: int
		    Sequence length (number of measurements made for one single phase in one measurement batch) (Default value
		    = 5)
		splits : tuple of floats
		    percentage of the splits. (train set, test set, validation set). Validation set must only be
		    included if validation is True (Default value = (0.8, 0.2))
		validation : bool
		    Whether or not a validation set should be created. In this case, splits must be of length 3 (Default value =
		    False)
		stable_only : bool
		    Defines whether only measurement values from stable phases should be loaded or not (Default value = False)
		"""
        super(ElementDatasetCreator, self).__init__()

        # Input checking
        assert label_range[0] <= label_range[1]

        # Make inputs class attributes
        self.seq_len = seq_len
        self.splits = splits
        self.validation = validation
        self.element = element

        # Set the properties to load
        gibbs = True if measurement == 'G' else False
        entropy = True if measurement == 'S' else False
        enthalpy = True if measurement == 'H' else False
        heat_cap = True if measurement == 'C' else False

        # Load the data based on whether data from any phase can be included or only from stable phases
        sgte_handler = SGTEHandler(element)
        if not stable_only:
            sgte_handler.evaluate_equations(temp_range[0], temp_range[1], 1e5, plot=False, phases=['all'],
                                            gibbs=gibbs,
                                            entropy=entropy,
                                            enthalpy=enthalpy,
                                            heat_capacity=heat_cap)
            self.data = sgte_handler.equation_result_data
        else:
            sgte_handler.get_stable_properties(temp_range[0], temp_range[1], measurement=measurement)
            self.data = sgte_handler.measurements

        self.data_remainder = len(self.data) % self.seq_len

        # Check if label length and number of phases match
        assert (label_range[1] - label_range[0] + 1) == len(self.data.columns) - 1

        # Create the data sets for all phases of the element. self.val_data is only not None, if validation is True
        self.train_data, self.test_data, self.val_data = self.create_batches(label_range[0], 1)
        for (l, c) in zip(range(label_range[0] + 1, label_range[1] + 1), range(2, len(self.data.columns))):
            tr, te, val = self.create_batches(l, c)

            self.train_data = np.vstack((self.train_data, tr))
            self.test_data = np.vstack((self.test_data, te))

            if self.validation:
                self.val_data = np.vstack((self.val_data, val))

    def get_data(self):
        """ """
        return self.train_data, self.test_data, self.val_data

    def create_batches(self, label, col_index):
        """
        For each phase of an element, batches of length self.seq_len containing (temperature, measurement, label) tuples
        are created. Only full size batches are accepted. This means, that if the length of the data loaded from the
        sgte dataset can not be divided by self.seq_len without a remainder, remainder number of elements shall are
        dropped from the dataset.

        Parameters
        ----------
        label :
            label for the data
        col_index :
            index of the column the data for the phases is in self.data

        Returns
        -------
        tuple of np.ndarrays

        """

        # Create the indices and shuffle them
        indices = np.array(list(range(len(self.data))))
        np.random.shuffle(indices)

        # Drop the remainders if there are some
        if self.data_remainder > 0:
            indices = indices[:-self.data_remainder]

        # Extract the temperature and measurement
        phase_df = pd.DataFrame(self.data.iloc[:, [0, col_index]])

        # Add the label
        phase_df['label'] = label

        # Reshape the data
        phase_data = np.array(phase_df.loc[indices]).reshape((-1, self.seq_len, 3))

        # Split the data
        train_indices = int(len(phase_data) * self.splits[0])
        test_indices = int(len(phase_data) * self.splits[1])

        train_data = phase_data[:train_indices][:-1]
        if not self.validation:
            test_data = phase_data[train_indices:]
            val_data = None
        else:
            test_data = phase_data[train_indices:train_indices + test_indices]
            val_data = phase_data[train_indices + test_indices:]

        # Sort the datasets by temperature
        train_data = train_data[:, train_data[:, :, 0].argsort()][np.diag_indices(train_data.shape[0])]
        test_data = test_data[:, test_data[:, :, 0].argsort()][np.diag_indices(test_data.shape[0])]
        if val_data is not None:
            val_data = val_data[:, val_data[:, :, 0].argsort()][np.diag_indices(val_data.shape[0])]

        return train_data, test_data, val_data
