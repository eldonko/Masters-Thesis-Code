import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from Data_Handling.SGTEHandler.Development.SGTEHandler import SGTEHandler


class DatasetCreator(object):
    """
    DataCreator loads the data for all element and phases from the SGTE data and creates the train, test and optionally
    the validation dataset. The datasets will be of type torch.utils.data.Dataset.
    """

    def __init__(self, element_phase_filename, temp_range=(200, 2000), measurement='G', seq_len=5, splits=(0.8, 0.2),
                 validation=False, elements=None, stable_only=False):
        """
        Creates the DatasetCreator

        :param element_phase_filename: specifies where the excel worksheet including the element and phase data is
        located. The notebook should contain a matrix
        :param temp_range: the temperature range for which the data should be loaded. Passed as tuple
		(low_temp, high_temp)
		:param measurement: Defines for which properties the classification should be made. Must be one of 'G', 'S', 'H'
		or 'C'.
		:param seq_len: Sequence length (number of measurements made for one single phase in one measurement batch)
		:param splits: percentage of the splits. (train set, test set, validation set). Validation set must only be
		included if validation is True
		:param validation: Whether or not a validation set should be created. In this case, splits must be of length 3
		:param elements: Which elements to load, if None, load all elements, else pass list of strings with the elements
		abbreviations (e.g. ['O', 'Fe'] if Oxygen and Iron should be loaded but nothing else)
		:param stable_only: Defines whether only measurement values from stable phases should be loaded or not
        """
        super(DatasetCreator, self).__init__()

        # Input checking
        assert temp_range[0] < temp_range[1]
        assert measurement in ['G', 'S', 'H', 'C']
        if not validation:
            assert len(splits) == 2
        else:
            assert len(splits) == 3
        assert np.array(splits).sum() == 1
        assert elements is None or type(elements) == list

        # Load the element-phase excel sheet to get all elements and to create the labels
        element_phase_data = pd.read_excel(element_phase_filename, sheet_name='Phases').set_index('Index')

        # If only certain phases shall be selected, then just retrieve those phases from element_phase_data
        if elements is not None:
            element_phase_data = element_phase_data[elements]

        # Prepare creating the labels for each element.
        phases_per_element = element_phase_data.sum()
        last_label = 0

        train_data, test_data, val_data = None, None, None

        for i in range(len(phases_per_element)):
            # Create the labels
            if not stable_only:
                label_range = (last_label, last_label + phases_per_element[i] - 1)
                last_label += phases_per_element[i]
            else:
                label_range = (last_label, last_label)
                last_label += 1

            # Create the data
            edc = ElementDatasetCreator(label_range, element=phases_per_element.index[i], temp_range=temp_range,
                                        measurement=measurement, seq_len=seq_len, splits=splits, validation=validation,
                                        stable_only=stable_only)
            train, test, val = edc.get_data()

            if train_data is None:
                train_data = train
            else:
                train_data = np.vstack((train_data, train))
            if test_data is None:
                test_data = test
            else:
                test_data = np.vstack((test_data, test))
            if validation:
                if val_data is None:
                    val_data = val
                else:
                    val_data = np.vstack((val_data, val))

        # Create the datasets
        self.train_set = ClassificationDataset()
        self.test_set = ClassificationDataset()
        self.val_set = ClassificationDataset() if validation else None

        self.train_set.set_samples(train_data)
        self.test_set.set_samples(test_data)
        if self.val_set is not None:
            self.val_set.set_samples(val_data)

    def get_datasets(self):
        return self.train_set, self.test_set, self.val_set


class ClassificationDataset(Dataset):
    """
	ClassificationDataset contains the data for training and testing the PhaseClassifier. It loads the SGTE data for
	all elements and phases. It needs to be specified whether this dataset will be used
	"""

    def __init__(self):
        """
		Initializes the dataset

		"""
        super(ClassificationDataset, self).__init__()

        self._samples = None

    def __getitem__(self, i: int):
        assert self._samples is not None
        return self._samples[i]

    def __len__(self):
        if self._samples is None:
            return None
        return len(self._samples)

    def set_samples(self, samples):
        """
        Sets the data of the Dataset self._samples to samples

        :param samples: data which the dataset should contain
        :return:
        """
        self._samples = samples


class ElementDatasetCreator(object):
    """
	ElementDataset contains the data for all phases of an element loaded from the SGTE data. The dataset will be used
	by ClassificationDataset to create the training, testing and optionally the validation data.
	"""

    def __init__(self, label_range, element, temp_range=(200, 2000), measurement='G', seq_len=5, splits=(0.8, 0.2),
                 validation=False, stable_only=False):
        """
		Initializes the dataset

		:param label_range: integer values passed as tuple (start label, end label). The label range is passed by the
		ClassificationDataset so that for every Element and phase a unique label can be created. Each phase of an
		element receives a label in the label range
		:param element: The element for which the data should be loaded.
		:param temp_range: the temperature range for which the data should be loaded. Passed as tuple
		(low_temp, high_temp)
		:param measurement: Defines for which properties the classification should be made. Must be one of 'G', 'S', 'H'
		or 'C'.
		:param seq_len: Sequence length (number of measurements made for one single phase in one measurement batch)
		:param splits: percentage of the splits. (t set, test set, validation set). Validation set must only be
		included if validation is True
		:param validation: Whether or not a validation set should be created. In this case, splits must be of length 3
		:param stable_only: Defines whether only measurement values from stable phases should be loaded or not
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
        return self.train_data, self.test_data, self.val_data

    def create_batches(self, label, col_index):
        """
        For each phase of an element, batches of length self.seq_len containing (temperature, measurement, label) tuples
        are created. Only full size batches are accepted. This means, that if the length of the data loaded from the
        SGTE dataset can not be divided by self.seq_len without a remainder, remainder number of elements shall are
        dropped from the dataset.

        :param label: label for the data
        :param col_index: index of the column the data for the phases is in self.data
        :return:
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

        return train_data, test_data, val_data