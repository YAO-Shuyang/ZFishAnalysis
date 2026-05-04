import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time
import tifffile as tiff
from sklearn.feature_selection import mutual_info_regression

from zfish._io import import_suite2p, import16chFlt
from zfish.analyzer import fast_gaussian_params, shuffle_test

from mazepy.basic.conversion import coordinate_recording_time
from mazepy.datastruc.neuact import SpikeTrain, TuningCurve
from mazepy.datastruc.variables import VariableBin

def calc_SI(
    spikes: np.ndarray, 
    rate_map: np.ndarray, 
    t_total: float, 
    t_nodes_frac: np.ndarray
) -> np.ndarray:
    mean_rate = np.nansum(spikes, axis = 1) / t_total # mean firing rate
    logArg = (rate_map.T / mean_rate).T;
    logArg[np.where(logArg == 0)] = 1; # keep argument in log non-zero

    IC = np.nansum(t_nodes_frac * rate_map * np.log2(logArg), axis = 1) # information content
    SI = IC / mean_rate; # spatial information (bits/spike)
    return(SI)

import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def _shuffle_ptp_worker(args):
    """
    Worker function for parallel shuffle test.

    Each worker computes a subset of shuffle iterations.
    """
    (
        dFF_c,
        stim,
        bin_idx,
        trial_bound,
        n_bins,
        n_local_shuffles,
        seed,
    ) = args

    rng = np.random.default_rng(seed)

    n_cells = dFF_c.shape[0]
    n_time = dFF_c.shape[1]

    shuf_ptp = np.zeros((n_cells, n_local_shuffles), dtype=np.float64)

    for i in range(n_local_shuffles):
        # Important: reset the index for each shuffle.
        shuf_idx = np.arange(n_time)

        # Circularly shift activity within each trial.
        for t in range(trial_bound.shape[0]):
            start, end = trial_bound[t]

            if end <= start + 1:
                continue

            shuf_dist = rng.integers(0, end - start)
            shuf_idx[start:end] = np.roll(shuf_idx[start:end], shuf_dist)

        shuf_responses = np.zeros((n_cells, n_bins), dtype=np.float64)

        for j in range(n_bins):
            if bin_idx[j].size == 0:
                shuf_responses[:, j] = np.nan
            else:
                shuf_responses[:, j] = np.nanmean(
                    dFF_c[:, shuf_idx[bin_idx[j]]],
                    axis=1,
                )

        shuf_ptp[:, i] = np.nanmax(shuf_responses, axis=1) - np.nanmin(
            shuf_responses,
            axis=1,
        )

    return shuf_ptp


def shuffle_test_parallel(
    dFF: np.ndarray,
    stim: np.ndarray,
    n_bins: int,
    trial_bound: np.ndarray,
    n_shuffles: int = 1000,
    included_cells: np.ndarray | None = None,
    seed: int | None = 42,
    n_workers: int | None = None,
    shuffles_per_task: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel shuffle test for stimulus-responsive cells.

    Parameters
    ----------
    dFF : np.ndarray, shape (n_cells, n_time)
        dF/F traces.
    stim : np.ndarray, shape (n_time,)
        Stimulus/bin labels.
    n_bins : int
        Number of stimulus/spatial bins.
    trial_bound : np.ndarray, shape (n_trials, 2)
        Trial boundaries in the local time coordinates of dFF and stim.
    n_shuffles : int
        Number of shuffles.
    included_cells : np.ndarray or None
        Global cell indices to include.
    seed : int or None
        Random seed.
    n_workers : int or None
        Number of processes.

    Returns
    -------
    p_values_return : np.ndarray, shape (n_cells,)
        Shuffle-test p-values.
    ptp : np.ndarray, shape (n_cells,)
        Observed PTP values.
    shuf_ptp_return : np.ndarray, shape (n_cells, n_shuffles)
        Shuffled PTP values.
    """
    stim = np.asarray(stim, dtype=np.int32).copy()
    trial_bound = np.asarray(trial_bound, dtype=np.int32)

    assert np.min(stim) >= 0, "stimulus labels must be non-negative integers"
    assert np.max(stim) < n_bins, "stimulus labels must be less than n_bins"
    assert dFF.shape[1] == stim.shape[0], (
        f"dFF and stim length mismatch: dFF has {dFF.shape[1]} time points, "
        f"but stim has {stim.shape[0]}."
    )
    assert np.max(trial_bound) <= dFF.shape[1], (
        "trial_bound contains indices larger than the local dFF/stim length. "
        "You probably need to recompute local trial boundaries after subsetting idx."
    )

    if included_cells is None:
        included_cells = np.arange(dFF.shape[0], dtype=np.int64)
    else:
        included_cells = np.asarray(included_cells, dtype=np.int64)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    if seed is None:
        seed = 0

    # Use only included cells for expensive computation.
    dFF_c = np.ascontiguousarray(dFF[included_cells, :], dtype=np.float64)

    # Precompute bin indices.
    bin_idx = [np.where(stim == i)[0] for i in range(n_bins)]

    # Observed response map.
    mean_responses = np.zeros((included_cells.shape[0], n_bins), dtype=np.float64)

    for i in range(n_bins):
        if bin_idx[i].size == 0:
            mean_responses[:, i] = np.nan
        else:
            mean_responses[:, i] = np.nanmean(dFF_c[:, bin_idx[i]], axis=1)

    # Observed PTP for included cells.
    ptp_included = np.nanmax(mean_responses, axis=1) - np.nanmin(
        mean_responses,
        axis=1,
    )

    # Split shuffle iterations across workers.
    n_workers = min(n_workers, n_shuffles)
    base = n_shuffles // n_workers
    remainder = n_shuffles % n_workers

    local_shuffle_counts = [
        base + (1 if i < remainder else 0)
        for i in range(n_workers)
    ]

    tasks = []
    for worker_id, n_local in enumerate(local_shuffle_counts):
        if n_local == 0:
            continue

        tasks.append(
            (
                dFF_c,
                stim,
                bin_idx,
                trial_bound,
                n_bins,
                n_local,
                seed + worker_id,
            )
        )

    print(
        f"Running parallel shuffle test: "
        f"{n_shuffles} shuffles, {len(tasks)} workers, "
        f"{included_cells.shape[0]} included cells."
    )

    shuf_chunks = []

    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(_shuffle_ptp_worker, task) for task in tasks]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Shuffle workers",
        ):
            shuf_chunks.append(future.result())

    shuf_ptp_included = np.concatenate(shuf_chunks, axis=1)

    # Because workers may finish in arbitrary order, the shuffle columns are unordered.
    # That is fine for p-values, because only the distribution matters.

    # Correct one-sided p-value:
    p_values_included = (
        np.sum(
            shuf_ptp_included >= ptp_included[:, np.newaxis],
            axis=1,
        )
        + 1
    ) / (n_shuffles + 1)

    # Return full-size arrays.
    p_values_return = np.ones(dFF.shape[0], dtype=np.float64) * np.nan
    ptp_return = np.ones(dFF.shape[0], dtype=np.float64) * np.nan
    shuf_ptp_return = np.ones((dFF.shape[0], n_shuffles), dtype=np.float64) * np.nan

    p_values_return[included_cells] = p_values_included
    ptp_return[included_cells] = ptp_included
    shuf_ptp_return[included_cells, :] = shuf_ptp_included

    return p_values_return, ptp_return, shuf_ptp_return

def make_local_trial_bound_from_trial_ids(trial_ids: np.ndarray) -> np.ndarray:
    """
    Construct local trial boundaries after temporal subsetting.

    Parameters
    ----------
    trial_ids : np.ndarray, shape (n_time,)
        Trial identity for each selected time point.

    Returns
    -------
    trial_bound : np.ndarray, shape (n_trials, 2)
        Local start/end indices.
    """
    trial_ids = np.asarray(trial_ids)

    change_idx = np.where(np.diff(trial_ids) != 0)[0] + 1

    starts = np.concatenate([[0], change_idx])
    ends = np.concatenate([change_idx, [trial_ids.shape[0]]])

    return np.vstack([starts, ends]).T.astype(np.int32)

def run_LinearTrack1D(
    i: int,
    sheet_file: pd.DataFrame,
    ds_behav_to: int = 50, # Hz
    is_remove_iti: bool = True,
    n_shuffle: int = 1000
) -> None:
    """process data collected on 1D linear track.

    Parameters
    ----------
    i : int
        The index of the session in the sheet file.
    sheet_file : pd.DataFrame
        The sheet file containing the session information.
    ds_behav_to : int, optional (Hz)
        The sampling rate to downsample the behavioral data to, by default 60.
    is_remove_iti : bool, optional
        Whether to remove inter-trial intervals (ITIs) from the analysis, by default True.
    n_shuffle : int, optional
        The number of shuffles to perform in the shuffle test, by default 1000.
    """
    n_bin = 45
    n_speed_bin = 6
    speed_range = (2, 8)
    speed_smooth_win = 5
    n_map = 1#2
    map_ids = [4]#[2, 4]
    exclude_prefix = 100
    
    assert n_map == len(map_ids), "n_map should be the same as the length of map_ids."
    
    suite2p_dir = sheet_file.loc[i, 'suite2p_dir']
    behav_dir = sheet_file.loc[i, 'behav_dir']

    print(
        f"{i},  Fish ID: {sheet_file.loc[i, 'FishID']}, session: "
        f"{sheet_file.loc[i, 'session']} --------"
    )
    
    print("  1. Import Neural and behavioral data:")
    trace = import_suite2p(suite2p_dir)
    print("      a. Suite2p data imported.")
    res = import16chFlt(behav_dir, 21)
    print("      b. 16chFlt behavioral data imported.")
    save_dir = os.path.dirname(behav_dir)
    processed_file = os.path.join(save_dir, "process")
    os.makedirs(processed_file, exist_ok=True)
    tiff.imwrite(os.path.join(save_dir, "mean_image.tif"), trace['meanImg'])
    
    trace['FishID'] = sheet_file.loc[i, 'FishID']
    trace['session'] = sheet_file.loc[i, 'session']
    trace['save_dir'] = save_dir
    
    # Get time for neural activity
    print("  2. Pre-processing.")
    ms_time = np.cumsum(
        np.ones(trace['RawTraces'].shape[1])*(1/trace['fs'])*1000
    ).astype(np.int64) 
    ms_time -= ms_time[0]
    
    print("      a. Process neural data.")
    # Subtract the mean of each neuron (centering across time points)
    for k in tqdm(range(trace['RawTraces'].shape[0])):
        meanrate = np.convolve(trace['RawTraces'][k], np.ones(exclude_prefix)/exclude_prefix, mode='same')
        meanrate[:exclude_prefix] = meanrate[exclude_prefix]
        meanrate[-exclude_prefix:] = meanrate[-exclude_prefix-1]
        trace['RawTraces'][k] = (trace['RawTraces'][k]-meanrate) / meanrate
        
        mean_deconv = np.convolve(trace['DeconvSignal'][k], np.ones(exclude_prefix)/exclude_prefix, mode='same')
        mean_deconv[:exclude_prefix] = mean_deconv[exclude_prefix]
        mean_deconv[-exclude_prefix:] = mean_deconv[-exclude_prefix-1]
        trace['DeconvSignal'][k] = (trace['DeconvSignal'][k]-mean_deconv) / mean_deconv

    # Downsample behavioral data from 6000 Hz to the specified rate
    print("      b. Downsample behavioral data.")
    downsample_factor = int(6000 / ds_behav_to)
    
    # Smooth swim power before downsampling.
    res['fltCh0'] = np.convolve(res['fltCh0'], np.ones(200)/200, mode='same')
    res['fltCh1'] = np.convolve(res['fltCh1'], np.ones(200)/200, mode='same')
    
    for k in res.keys():
        res[k] = res[k][::downsample_factor]

    if res['Paradigm'][0] in [20260302, 20260404, 20260410]:
        res['behav_pos_y'] *= 2
        
    # Convert behavioral time to ms and make it int
    behav_time = (res['behav_time']*1000).astype(np.int64)
    # Position is 0-100. Outliers indicate inter-trial intervals.
    behav_pos = res['behav_pos_y'].copy().astype(np.float64)
    
    behav_pos[(behav_pos >= 100)| (behav_pos < 0)] = np.nan
    trial_based_time = np.zeros_like(behav_time, dtype=np.int64)

    # Calculate trial start and end time.
    within_trial_idx = np.where((np.isnan(behav_pos) == False)&(behav_pos >= 0)&(behav_pos < 100))[0]
    trial_change_idx = np.where(np.diff(behav_pos[within_trial_idx]) < 0)[0] + 1
    lap_beg_idx = np.concatenate(([0], trial_change_idx))
    lap_end_idx = np.concatenate((trial_change_idx, [len(within_trial_idx)]))
    trace['lap beg time'] = behav_time[within_trial_idx][lap_beg_idx]
    trace['lap end time'] = behav_time[within_trial_idx][lap_end_idx-1]
    trace['lap_beg_idx'] = within_trial_idx[lap_beg_idx]
    trace['lap_end_idx'] = within_trial_idx[lap_end_idx-1]
    
    for j in range(len(lap_beg_idx)):
        trial_based_time[within_trial_idx[lap_beg_idx[j]: lap_end_idx[j]]] = (
            behav_time[within_trial_idx][lap_beg_idx[j]: lap_end_idx[j]] - behav_time[within_trial_idx][lap_beg_idx[j]]
        )
    for j in range(len(lap_beg_idx)-1):
        trial_based_time[within_trial_idx[lap_end_idx[j]-1]: within_trial_idx[lap_beg_idx[j+1]]] = (
            behav_time[within_trial_idx[lap_end_idx[j]-1]: within_trial_idx[lap_beg_idx[j+1]]] - behav_time[within_trial_idx[lap_beg_idx[j+1]]]
        )
        
    trace['n_trials'] = len(lap_beg_idx)
    trace['map'] = res['map'][within_trial_idx][lap_beg_idx].astype(np.int64)
    
    # Calculate speed
    print("      c. Process real-time speed.")
    behav_speed = np.zeros_like(behav_pos, dtype=np.float64) * np.nan 
    behav_speed_raw = np.zeros_like(behav_pos, dtype=np.float64) * np.nan
    for j in tqdm(range(len(lap_beg_idx))):
        idx = within_trial_idx[lap_beg_idx[j]: lap_end_idx[j]]
        dt = np.append(np.diff(behav_time[idx]) / 1000, int(1/ds_behav_to))
        dx = np.append(np.diff(behav_pos[idx]), 0)
        dx[dx<0] += 100
        behav_speed_raw[idx] = dx/dt
        # Smooth speed.
        behav_speed[idx] = np.convolve(
            dx, np.ones(speed_smooth_win), mode='same'
        ) / np.convolve(dt, np.ones(speed_smooth_win), mode='same') 
    

    n_len_per_bin = 100/n_bin
    if is_remove_iti:
        trace['behav_time'] = behav_time[within_trial_idx]
        trace['behav_pos'] = behav_pos[within_trial_idx]
        trace['fltCh0'] = res['fltCh0'][within_trial_idx]
        trace['fltCh1'] = res['fltCh1'][within_trial_idx]
        trace['behav_nodes'] = (behav_pos[within_trial_idx]//n_len_per_bin).astype(np.int64)
        trace['behav_speed'] = np.clip(behav_speed[within_trial_idx], speed_range[0], speed_range[1]) 
        trace['behav_speed_raw'] = behav_speed_raw[within_trial_idx]
        trace['behav_time_aligned'] = trial_based_time[within_trial_idx]
        trace['behav_trial'] = res['n_trials'][within_trial_idx]
    else:
        trace['behav_speed_raw'] = behav_speed_raw
        trace['behav_speed'] = np.clip(behav_speed, speed_range[0], speed_range[1]) 
        trace['behav_pos'] = behav_pos        
        trace['behav_nodes'] = (behav_pos//n_len_per_bin).astype(np.int64)
        trace['behav_time'] = behav_time
        trace['behav_time_aligned'] = trial_based_time
        trace['fltCh0'] = res['fltCh0']
        trace['fltCh1'] = res['fltCh1']        
        trace['behav_trial'] = res['n_trials']
        
    
    
    if is_remove_iti:
        assert np.min(trace['behav_nodes']) >= 0 and np.max(trace['behav_nodes']) <= n_bin-1, \
            f"behav_nodes should be in the range of [0, n_bin-1], but got " \
            f"min={np.min(trace['behav_nodes'])} and max={np.max(trace['behav_nodes'])}."
    else:
        trace['behav_nodes'][np.isnan(trace['behav_nodes'])] = -1
    trace['behav_nodes'] = trace['behav_nodes'].astype(np.int64)
    
    # Coordinate neural activity and behavioral data
    print("      d. Coordinate neural activity and behavioral data.")
    coord_idx = coordinate_recording_time(ms_time, trace['behav_time'])
    ms_speed = trace['behav_speed'][coord_idx]
    ms_pos = trace['behav_pos'][coord_idx]
    ms_nodes = trace['behav_nodes'][coord_idx].astype(np.int64)
    ms_map = res['map'][coord_idx].astype(np.int64)
    ms_time_aligned = trace['behav_time_aligned'][coord_idx]
    ms_trial = trace['behav_trial'][coord_idx]
    trace['ms_time'] = ms_time
    trace['ms_time_aligned'] = ms_time_aligned
    trace['ms_speed'] = ms_speed
    trace['ms_pos'] = ms_pos
    trace['spike_nodes'] = ms_nodes
    trace['ms_map'] = ms_map
    trace['ms_trial'] = ms_trial
        
    print("  3. Calculate Mean dF/F Map")
    print("      a. Linear Map 1D")
    trace['n_neuron'] = trace['RawTraces'].shape[0]
    rate_map_all = np.zeros(
        (trace['n_neuron'], n_bin, n_map), dtype=np.float64
    )
    rate_map_fir = np.zeros_like(rate_map_all)
    rate_map_sec = np.zeros_like(rate_map_all)
    t_total = np.zeros(n_map, dtype=np.float64)
    t_nodes_frac = np.zeros((n_map, n_bin), dtype=np.float64)
    
    map_indices = [
        np.where(ms_map[exclude_prefix:] == map_id)[0]+exclude_prefix for map_id in map_ids
    ]
    map_checked_points = np.vstack([
        [iit[0], int((iit[0]+iit[-1])/2), iit[-1]] for iit in map_indices
    ])
    for n in range(n_map):
        for i in tqdm(range(n_bin)):
            idx = np.where((ms_nodes[exclude_prefix:] == i) & (ms_map[exclude_prefix:] == map_ids[n]))[0] + exclude_prefix
            rate_map_all[:, i, n] = np.nanmean(trace['RawTraces'][:, idx], axis=1)
            t_nodes_frac[n, i] = idx.shape[0] / trace['fs']
            
            onset, mid, offset = map_checked_points[n, :]
            idx_fir = np.where((ms_nodes[onset:mid] == i))[0] + onset
            
            if idx_fir.shape[0] > 0:
                rate_map_fir[:, i, n] = np.nanmean(trace['RawTraces'][:, idx_fir], axis=1)
                
            idx_sec = np.where((ms_nodes[mid:offset] == i))[0] + mid
            if idx_sec.shape[0] > 0:
                rate_map_sec[:, i, n] = np.nanmean(trace['RawTraces'][:, idx_sec], axis=1)
                
        t_total[n] = np.sum(t_nodes_frac[n, :])
        t_nodes_frac[n, :] /= (t_total[n]+1e-8)
    
    rate_map_all[np.isnan(rate_map_all)] = 0.0
    rate_map_fir[np.isnan(rate_map_fir)] = 0.0
    rate_map_sec[np.isnan(rate_map_sec)] = 0.0
    trace['rate_map_all'] = rate_map_all
    trace['rate_map_fir'] = rate_map_fir
    trace['rate_map_sec'] = rate_map_sec
    
    # Smooth the rate map along the spatial dimension.
    sigma = 1
    gkernel = np.exp(-0.5 * (np.linspace(-3, 3, 7) / sigma)**2)
    gkernel /= gkernel.sum()
    print("          Smooth the rate map.")
    smooth_map_all = np.zeros_like(rate_map_all)
    smooth_map_fir = np.zeros_like(rate_map_fir)
    smooth_map_sec = np.zeros_like(rate_map_sec)
    for n in range(n_map):
        for i in tqdm(range(trace['n_neuron'])):
            smooth_map_all[i, :, n] = np.convolve(
                rate_map_all[i, :, n], gkernel, mode='same'
            )
            smooth_map_fir[i, :, n] = np.convolve(
                rate_map_fir[i, :, n], gkernel, mode='same'
            )
            smooth_map_sec[i, :, n] = np.convolve(
                rate_map_sec[i, :, n], gkernel, mode='same'
            )
    trace['smooth_map_all'] = smooth_map_all
    trace['smooth_map_fir'] = smooth_map_fir
    trace['smooth_map_sec'] = smooth_map_sec
    
    half_half_corr = np.zeros((trace['n_neuron'], n_map), dtype=np.float64)
    print("          Calculate the half-half correlation of the rate map.")
    for n in range(n_map):
        for i in tqdm(range(trace['n_neuron']), desc=f"Map {map_ids[n]}"):
            half_half_corr[i, n] = np.corrcoef(
                rate_map_fir[i, :, n], rate_map_sec[i, :, n]
            )[0, 1]
            
    trace['fir_sec_corr'] = half_half_corr

    # Generate tuning curve estimated by fast Gaussian fitting.
    print("          Fast Gaussian fitting.")
    tuning_params = fast_gaussian_params(
        np.linspace(100/n_bin/2, 100-100/n_bin/2, n_bin),
        smooth_map_all[:, :, 0]
    )
    trace['tuning_params'] = tuning_params
    
    """
    print("      b. Speed Coupled Map 1D")
    speed_bin_size = (speed_range[1]-speed_range[0]+1e-8)/n_speed_bin
    trace['ms_speed_bin'] = (
        (ms_speed-speed_range[0]) // speed_bin_size
    )
    trace['ms_speed_bin'][
        (trace['ms_speed_bin'] < 0) | 
        (trace['ms_speed_bin'] >= n_speed_bin) |
        (np.isnan(trace['ms_speed_bin']))
    ] = -1
    trace['ms_speed_bin'] = trace['ms_speed_bin'].astype(np.int64)
    ms_speed_bin = trace['ms_speed_bin']
    
    speed_map_all = np.zeros(
        (trace['n_neuron'], n_bin, n_speed_bin, n_map), dtype=np.float64
    )
    ms_nodes_speed = ms_nodes + n_bin * ms_speed_bin
    for n in range(n_map):
        for i in range(n_bin*n_speed_bin):
            idx = np.where((ms_nodes_speed == i) & (ms_map == map_ids[n]))[0]
            speed_map_all[:, :, :, n] = np.reshape(
                np.mean(trace['RawTraces'][:, idx], axis=1), (trace['n_neuron'], n_bin, n_speed_bin)
            )
    trace['speed_map_all'] = speed_map_all
    
    # Smooth the speed map along the spatial dimension.
    print("          Smooth the speed map.")
    speed_map_smooth = np.zeros_like(speed_map_all)
    for n in range(n_map):
        for k in range(n_speed_bin):
            for i in range(trace['n_neuron']):
                speed_map_smooth[i, :, k, n] = np.convolve(
                    speed_map_all[i, :, k, n], gkernel, mode='same'
                )
    trace['speed_map_smooth'] = speed_map_smooth
    """
    # Save the processed data
    save_path = os.path.join(save_dir, f"trace.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(trace, f)
        
    print("      c. Save the processed data.")
    print("  4. Shuffle test for spatial tuning.")
    p_values_all = np.zeros((trace['n_neuron'], n_map), dtype=np.float64)
    ptp_all = np.zeros((trace['n_neuron'], n_map), dtype=np.float64)
    shuf_ptp_all = np.zeros(
        (trace['n_neuron'], n_map, n_shuffle),
        dtype=np.float64,
    )

    for n in range(n_map):
        print(f"      Map {map_ids[n]}:")

        idx = np.where(trace['ms_map'][exclude_prefix:] == map_ids[n])[0] + exclude_prefix

        included_cells = np.where(trace['snr'] >= 0.5)[0]

        # Local trial boundaries after subsetting by idx.
        local_trial_bound = make_local_trial_bound_from_trial_ids(
            trace['ms_trial'][idx]
        )

        p_values, ptp, shuf_ptp = shuffle_test_parallel(
            dFF=trace['RawTraces'][:, idx],
            stim=trace['spike_nodes'][idx],
            n_bins=trace['rate_map_all'].shape[1],
            trial_bound=local_trial_bound,
            n_shuffles=n_shuffle,
            seed=42,
            included_cells=included_cells,
            n_workers=5
        )

        p_values_all[:, n] = p_values
        ptp_all[:, n] = ptp
        shuf_ptp_all[:, n, :] = shuf_ptp
    
    trace['p_values'] = p_values_all
    trace['ptp'] = ptp_all
    trace['shuf_ptp'] = shuf_ptp_all
    with open(os.path.join(save_dir, f"trace.pkl"), 'wb') as f:
        pickle.dump(trace, f)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}      Shuffle test done and results saved.")

    print("  5. Calculate mutual information.")
    MI = np.zeros((trace['RawTraces'].shape[0], n_map), np.float32)
    map_idx = [np.where(trace['ms_map'][exclude_prefix:] == map_ids[n])[0]+exclude_prefix for n in range(n_map)]
    for n in range(n_map):
        print(f"      Map {map_ids[n]}:")

        idx = map_idx[n]

        # X: rows are time points, columns are neurons.
        # Each column/neuron gets one MI value against y.
        X = np.ascontiguousarray(trace['RawTraces'][:, idx].T, dtype=np.float64)
        y = np.asarray(trace['spike_nodes'][idx], dtype=np.float64)

        mi_this_map = mutual_info_regression(
            X,
            y,
            discrete_features=False,
            n_neighbors=3,
            random_state=42,
            n_jobs=10,
        )

        MI[:, n] = mi_this_map.astype(np.float32)
    trace['mutual_info'] = MI
    with open(os.path.join(save_dir, f"trace.pkl"), 'wb') as f:
        pickle.dump(trace, f)
    # Print the finished time of processing
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}      Mutual information calculation done and results saved.")
    print(f"Done.", end="\n\n\n")

if __name__ == "__main__":
    info = {
        "FishID": ["10138"],
        "session": [2],
        "suite2p_dir": [r"D:\EnData\Light-sheet\10138\snr filtered"],
        "behav_dir": [r"D:\EnData\Light-sheet\10138\S3\res.16chFlt"]
    }
    sheet_file = pd.DataFrame(info)
    for i in range(len(sheet_file)):
        run_LinearTrack1D(i, sheet_file)