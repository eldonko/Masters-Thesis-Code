from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """
    Contains the data for classification training, testing and validation

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
        """Sets the data of the Dataset self._samples to samples

        Parameters
        ----------
        samples : np.ndarray
            data which the dataset should contain

        Returns
        -------

        """
        self._samples = samples


