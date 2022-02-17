import warnings


def except_test_mode():
    """
    Raises a warning if network save function of PhaseClassifier or ElementClassifier is called in evaluation mode.

    Returns
    -------

    """

    warnings.warn('Network can not be saved in evaluation mode. Run in train mode to save the network.')