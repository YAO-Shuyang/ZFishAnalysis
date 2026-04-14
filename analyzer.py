import numpy as np

import numpy as np
from numba import njit, prange
from scipy.stats import f_oneway
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

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

def shuffle_test(
    dFF: np.ndarray,
    stim: np.ndarray,
    n_bins: int,
    trial_bound: np.ndarray,
    n_shuffles: int = 1000,
    included_cells: np.ndarray = None,
    seed: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a shuffle test (PTP) to identify stimulus-responsive cells.

    Parameters
    ----------
    dFF : array-like, shape (n_cells, n_time)
        dF/F traces for each cell.
    stim : array-like, shape (n_time,)
        Stimulus labels for each time point.
    trial_bound : array-like, shape (n_trials, 2)
        Start and end indices of each trial.
    n_shuffles : int
        Number of shuffles to perform.
    seed : int
        Random seed for reproducibility.
    included_cells : array-like, shape (n_cells,), optional
        Boolean mask of cells to include in the analysis. If None, all cells are included.

    Returns
    -------
    p_values : array-like, shape (n_cells, )
        p-values for each cell indicating stimulus responsiveness.
    ptp : array-like, shape (n_cells, )
        Observed range (max-to-min) values for each cell.
    shuf_ptp : array-like, shape (n_cells, n_shuffles)
        Range (max-to-min) values for each cell across shuffles.
    """
    
    stim = np.asarray(stim, dtype=np.int32).copy()
    trial_bound = np.asarray(trial_bound, dtype=np.int32)
    assert np.min(stim) >= 0, "stimulus labels must be non-negative integers"
    assert np.max(stim) < n_bins, "stimulus labels must be less than n_bins"
    
    if seed is None:
        np.random.seed(0)
    else:
        np.random.seed(seed)
    
    if included_cells is None:
        included_cells = np.arange(dFF.shape[0])
    
    mean_responses = np.zeros((dFF.shape[0], n_bins)) * np.nan
    bin_idx = [np.where(stim == i)[0] for i in range(n_bins)] 
    dFF_c = dFF[included_cells, :]
    
    for i in range(n_bins):
        mean_responses[included_cells, i] = np.mean(dFF_c[:, bin_idx[i]], axis=1)

    # Compute the observed PTP for each cell
    ptp = np.ptp(mean_responses, axis=1)
    
    # Shuffle test
    shuf_ptp = np.zeros((included_cells.shape[0], n_shuffles)) * np.nan
    shuf_idx= np.arange(dFF.shape[1])
    
    for i in tqdm(range(n_shuffles), desc="Shuffling"):
        # Cross trial permutation:
        for t in range(trial_bound.shape[0]):
            shuf_dist = np.random.randint(0, trial_bound[t, 1] - trial_bound[t, 0])
            shuf_idx[trial_bound[t, 0]:trial_bound[t, 1]] = np.roll(shuf_idx[trial_bound[t, 0]:trial_bound[t, 1]], shuf_dist)
        
        shuf_responses = np.zeros((included_cells.shape[0], n_bins))
        for j in range(n_bins):
            shuf_responses[:, j] = np.mean(dFF_c[:, shuf_idx[bin_idx[j]]], axis=1)
        shuf_ptp[:, i] = np.ptp(shuf_responses, axis=1)
        
    
    p_values = np.mean(shuf_ptp - ptp[included_cells, np.newaxis], axis=1)
    p_values_return = np.ones(dFF.shape[0]) * np.nan
    p_values_return[included_cells] = p_values
    
    shuf_ptp_return = np.ones((dFF.shape[0], n_shuffles)) * np.nan
    shuf_ptp_return[included_cells, :] = shuf_ptp
    return p_values_return, ptp, shuf_ptp_return

if __name__ == "__main__":
    import pickle
    import os
    import numpy as np
    
    path = r"D:\EnData\Light-sheet\10156"

    with open(os.path.join(path, "trace.pkl"), 'rb') as f:
        trace = pickle.load(f)
        
    from zfish.analyzer import shuffle_test

    idx = np.where(np.diff(trace['ms_trial']) != 0)[0] + 1
    trial_bound = np.zeros((idx.shape[0]+1, 2), dtype=np.int32)
    trial_bound[0, 0] = 0
    trial_bound[-1, 1] = trace['ms_trial'].shape[0]
    trial_bound[1:, 0] = idx
    trial_bound[:-1, 1] = idx
    
    p_values, ptp, shuf_ptp = shuffle_test(
        dFF=trace['RawTraces'][:, 200:],
        stim=trace['spike_nodes'][200:],
        n_bins=trace['rate_map_all'].shape[1],
        trial_bound=trial_bound,
        n_shuffles=1000,
        seed=42,
        included_cells=np.where(trace['snr'] >= 0.5)[0]
    )
    
    trace['p_values'] = p_values
    trace['ptp'] = ptp
    trace['shuf_ptp'] = shuf_ptp
    with open(os.path.join(path, "trace.pkl"), 'wb') as f:
        pickle.dump(trace, f)