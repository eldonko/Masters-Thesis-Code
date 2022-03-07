import pkg_resources

import numpy as np
import pandas as pd

from .dataset import ClassificationDataset
from .element_dataset_creator import ElementDatasetCreator


class DatasetCreator(object):
    """
    DatasetCreator loads the data for all element and phases from the sgte data and creates the train, test and
    optionally the validation dataset. The datasets will be of type torch.utils.data.Dataset.
    """

    def __init__(self, temp_range=(200, 2000), measurement='G', seq_len=5, splits=(0.8, 0.2),
                 validation=False, elements=None, stable_only=False, step=1.):
        """
        Creates the DatasetCreator

        Parameters
        ----------
        temp_range : tuple of ints
            the temperature range for which the data should be loaded. (low_temp, high_temp)
		measurement : str
		    Defines for which properties the classification should be made. Must be one of 'G', 'S', 'H' or 'C'.
		    (Default value = 'G')
		seq_len : int
		    Sequence length (number of measurements made for one single phase in one measurement batch) (Default
		    value = 5)
		splits : tuple of floats
		    percentage of the splits. (train set, test set, validation set). Validation set must only be
		    included if validation is True (Default value = (0.8, 0.2))
		validation : bool
		    Whether or not a validation set should be created. In this case, splits must be of length 3 (Default
		    value = False)
		elements : list of str or None
		    Which elements to load, if None, load all elements, else pass list of strings with the elements
		    abbreviations (e.g. ['O', 'Fe'] if Oxygen and Iron should be loaded but nothing else) (Default value = None)
		stable_only : bool
		    Defines whether only measurement values from stable phases should be loaded or not (Default value = False)
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
        element_phase_filename = r'../data/Phases.xlsx'
        stream = pkg_resources.resource_stream(__name__, element_phase_filename)
        element_phase_data = pd.read_excel(stream, sheet_name='Phases').set_index('Index')

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
                                        stable_only=stable_only, step=step)
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

        print(train_data.shape)
        # Create the datasets
        self.train_set = ClassificationDataset()
        self.test_set = ClassificationDataset()
        self.val_set = ClassificationDataset() if validation else None

        self.train_set.set_samples(train_data)
        self.test_set.set_samples(test_data)
        if self.val_set is not None:
            self.val_set.set_samples(val_data)

    def get_datasets(self):
        """

        Returns
        -------
        tuple of ClassificationDatasets
            Training set, test set and validation set
        """
        return self.train_set, self.test_set, self.val_set