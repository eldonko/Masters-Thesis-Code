import numpy as np
import pandas as pd

from sgte.handler import SGTEHandler
from .encoding import Encoder


class ElementDatasetCreator(object):
    """
    ElementDatasetCreator loads the data for all phases of an element from the sgte data. The dataset will be used
    by dataset to create the training, testing and optionally the validation dataset.

    """

    def __init__(self, element_label, element, temp_range=(200, 2000), measurement='G', seq_len=5, splits=(0.8, 0.2),
                 validation=False, step=1., p=1e5, user='phase'):
        """
		Initializes the dataset

		element_label : int
		    the element range is passed by the DatasetCreator so that for every element a unique label can be
		    created.
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
		p : float
		    pressure at which data should be generated
		user : str
		    Defines whether the data is generated for phase or element classification. Can be either 'phase' or 'element'
		    (Default value = 'phase')
		"""
        super(ElementDatasetCreator, self).__init__()

        # Make inputs class attributes
        self.seq_len = seq_len
        self.splits = splits
        self.validation = validation
        self.element = element
        self.user = user
        self.measurement = measurement

        # Depending on the user decide whether to add phase labels
        add_phase_labels = False
        if self.user == 'phase':
            add_phase_labels = True

        # Load the data from stable phases (add the phase labels in case of phase classification)
        sgte_handler = SGTEHandler(element)
        sgte_handler.get_stable_properties(temp_range[0], temp_range[1], p=p, measurement=measurement, step=step,
                                           add_phase_labels=add_phase_labels)
        self.data = sgte_handler.measurements

        self.data_remainder = len(self.data) % self.seq_len

        # Create the data sets for all phases of the element. self.val_data is only not None, if validation is True
        self.train_data, self.test_data, self.val_data = self.create_batches(element_label)

    def get_data(self):
        """ """
        return self.train_data, self.test_data, self.val_data

    def create_batches(self, label):
        """
        For each phase of an element, batches of length self.seq_len containing (temperature, measurement, label) tuples
        are created. Only full size batches are accepted. This means, that if the length of the data loaded from the
        sgte dataset can not be divided by self.seq_len without a remainder, remainder number of elements shall are
        dropped from the dataset.

        Parameters
        ----------
        label :
            label for the data

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
        phase_df = self.data

        # Add the label
        phase_df['Label'] = label

        # Reshape the data
        shape = (-1, self.seq_len, 3)
        if self.user == 'phase':
            # Update the shape
            shape = (-1, self.seq_len, 4)

            # Encode the phases
            phase_df['Phase label'] = Encoder()(phase_df['Phase label'])
        phase_data = np.array(phase_df.loc[indices]).reshape(shape)

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
