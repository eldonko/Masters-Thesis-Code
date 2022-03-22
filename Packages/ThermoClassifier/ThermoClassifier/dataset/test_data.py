import os

import pandas as pd
import numpy as np

from .dataset import ClassificationDataset
from .encoding import Encoder


class TestData(object):
    """
    This class loads data from sources other than the SGTE data to check how good the thermoclassifier can make
    predictions on data that is not from the SGTE database.
    """
    def __init__(self):
        super(TestData, self).__init__()

    def __call__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args
            args[0] : str
                file path to folder where data is stored in excel files
            args[1] : str
                name of database where the data has been taken from
                must be any of 'barin',
            args[2] : int
                sequence length (number of value pairs per package)
            args[3] : tuple
                temperature range
        kwargs

        Returns
        -------
        tuple of npndarrays
        """

        # Input checking
        assert args[1] in ['barin']
        assert args[2] > 0
        assert type(args[2]) == int
        assert args[3][0] < args[3][1]

        if args[1] == 'barin':
            return self.load_barin(args[0], args[2], args[3])

    @staticmethod
    def load_barin(file_path, seq_len, temp_range):
        """
        Loads the data from Barin database stored in 1 Excel sheet per element

        Parameters
        ----------
        file_path : str
            file path of the folder where the data Excel sheets are located
        seq_len : int
            sequence length (number of value pairs per package)
        temp_range : tuple
            temperature range

        Returns
        -------
        ClassificationDataset

        """
        test_data = None

        for i, element in enumerate(os.listdir(file_path)):
            # Load the excel sheet
            data = pd.read_excel(os.path.join(file_path, element), header=None)
            data = data.rename(columns={0: 'Temperature', 1: 'Measurement'})
            data = data[(data['Temperature'] >= temp_range[0]) & (data['Temperature'] <= temp_range[1])]
            el = os.path.splitext(element)[0]
            data['Label'] = Encoder()(el)

            # Create the indices for random selection and shuffle them
            indices = np.array(list(range(len(data))))
            np.random.shuffle(indices)

            # Get the remainder of the data points (last package that's smaller than seq_len) Drop the remainders
            # if there are some and convert to np.array
            data_remainder = len(data) % seq_len
            if data_remainder > 0:
                indices = indices[:-data_remainder]
            data = np.array(data.loc[indices], dtype=np.float).reshape((-1, seq_len, 3))

            # Sort the datasets by temperature
            data = data[:, data[:, :, 0].argsort()][np.diag_indices(data.shape[0])]

            if test_data is None:
                test_data = data
            else:
                test_data = np.vstack((test_data, data))

        # Create a dataset from the np.ndarray
        dataset = ClassificationDataset()
        dataset.set_samples(test_data)
        return dataset