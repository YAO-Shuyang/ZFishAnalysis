"""
This code is designed for analyzing electrophysiological data.
"""

import numpy as np

def get_grating_labels(pos: np.ndarray) -> np.ndarray:
    """
    This function assigns labels to grating positions.
    It divides the position range into four segments and assigns a label (1-4)
    based on which segment the position falls into.

    Parameters
    ----------
    pos: numpy.ndarray
        An array of positions.

    Returns
    -------
    numpy.ndarray
        An array of labels corresponding to the input positions.
    """
    pos_new = np.clip(pos, 0, 99)
    # Use numpy.digitize
    bins = [0, 32, 64, 96, 100]
    labels = np.digitize(pos_new, bins)  # This will give labels from 1 to 4
    return labels

OPTO_POS = (83.5, 87.2)