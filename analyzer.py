import numpy as np

import numpy as np
from numba import njit, prange


def fast_gaussian_params(stimulus, responses, eps=1e-12):
    """
    Fast Gaussian parameter estimation for many cells using weighted moments.

    Parameters
    ----------
    stimulus : array-like, shape (n_stim,)
        Stimulus values.
    responses : array-like, shape (n_cells, n_stim)
        Response of each cell across stimuli.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    params : dict
        Dictionary with arrays of shape (n_cells,):
        - baseline
        - amplitude
        - center
        - sigma
    """
    x = np.asarray(stimulus, dtype=np.float64)
    Y = np.asarray(responses, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("stimulus must be 1D.")
    if Y.ndim != 2:
        raise ValueError("responses must be 2D: (n_cells, n_stim).")
    if Y.shape[1] != x.shape[0]:
        raise ValueError("responses.shape[1] must equal len(stimulus).")

    # baseline and amplitude
    baseline = np.min(Y, axis=1)
    ymax = np.max(Y, axis=1)
    amplitude = ymax - baseline

    # subtract baseline
    W = Y - baseline[:, None]

    # avoid negative weights from noise
    W[W < 0] = 0.0

    # sum of weights
    wsum = np.sum(W, axis=1) + eps

    # center = peak location
    center = np.argmax(W, axis=1)
    center = x[center]

    # weighted variance = sigma^2
    var = np.sum(W * (x[np.newaxis, :] - center[:, np.newaxis]) ** 2, axis=1) / wsum
    sigma = np.sqrt(np.maximum(var, eps))

    return {
        "baseline": baseline,
        "amplitude": amplitude,
        "center": center,
        "sigma": sigma,
    }