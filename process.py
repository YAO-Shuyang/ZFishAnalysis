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

def run_LinearTrack1D(
    i: int,
    sheet_file: pd.DataFrame,
    ds_behav_to: int = 50, # Hz
    is_remove_iti: bool = True
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
    """
    n_bin = 45
    n_speed_bin = 6
    speed_range = (2, 8)
    speed_smooth_win = 5
    n_map = 2#2
    map_ids = [4,5]#[2, 4]
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
    idx = np.where(np.diff(trace['ms_trial']) != 0)[0] + 1
    trial_bound = np.zeros((idx.shape[0]+1, 2), dtype=np.int32)
    trial_bound[0, 0] = 0
    trial_bound[-1, 1] = trace['ms_trial'].shape[0]
    trial_bound[1:, 0] = idx
    trial_bound[:-1, 1] = idx
    
    p_values_all = np.zeros((trace['n_neuron'], n_map), dtype=np.float64)
    ptp_all = np.zeros((trace['n_neuron'], n_map), dtype=np.float64)
    shuf_ptp_all = np.zeros((trace['n_neuron'], n_map, 1000), dtype=np.float64)
    for n in range(n_map):
        print(f"      Map {map_ids[n]}:")
        idx = np.where(trace['ms_map'][exclude_prefix:] == map_ids[n])[0]+exclude_prefix
        p_values, ptp, shuf_ptp = shuffle_test(
            dFF=trace['RawTraces'][:, idx],
            stim=trace['spike_nodes'][idx],
            n_bins=trace['rate_map_all'].shape[1],
            trial_bound=trial_bound,
            n_shuffles=1000,
            seed=42,
            included_cells=np.where(trace['snr'] >= 0.5)[0]
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
    for i in tqdm(range(trace['RawTraces'].shape[0])):
        for n in range(n_map):
            MI[i, n] = mutual_info_regression(
                trace['RawTraces'][i, map_idx[n]].reshape(-1, 1), 
                trace['spike_nodes'][map_idx[n]]
            )[0]
    trace['mutual_info'] = MI
    with open(os.path.join(save_dir, f"trace.pkl"), 'wb') as f:
        pickle.dump(trace, f)
    # Print the finished time of processing
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}      Mutual information calculation done and results saved.")
    print(f"Done.", end="\n\n\n")

if __name__ == "__main__":
    info = {
        "FishID": ["10162"],
        "session": [2],
        "suite2p_dir": [r"D:\EnData\Light-sheet\10162\snr filtered"],
        "behav_dir": [r"D:\EnData\Light-sheet\spatial preference vr1\10162\S2\res.16chFlt"]
    }
    sheet_file = pd.DataFrame(info)
    for i in range(len(sheet_file)):
        run_LinearTrack1D(i, sheet_file)