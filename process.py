import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time

from zfish._io import import_suite2p, import16chFlt

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
    n_bin = 50
    n_speed_bin = 6
    speed_range = (2, 8)
    speed_smooth_win = 5
    n_map = 2
    map_ids = [2, 4]
    assert n_map == len(map_ids), "n_map should be the same as the length of map_ids."
    
    suite2p_dir = sheet_file.loc[i, 'suite2p_dir']
    behav_dir = sheet_file.loc[i, 'behav_dir']
    trace = import_suite2p(suite2p_dir)
    res = import16chFlt(behav_dir, 21)
    print(
        f"{i},  Fish ID: {sheet_file.loc[i, 'FishID']}, session: "
        f"{sheet_file.loc[i, 'session']} --------"
    )
    print("  1. Neural and behavioral data imported.")
    save_dir = os.path.dirname(behav_dir)
    
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
    for k in range(trace['RawTraces'].shape[0]):
        meanrate = np.convolve(trace['RawTraces'][k], np.ones(200)/200, mode='same')
        meanrate[:200] = meanrate[200]
        meanrate[-200:] = meanrate[-201]
        trace['RawTraces'][k] = trace['RawTraces'][k] - meanrate
        
        mean_deconv = np.convolve(trace['DeconvSignal'][k], np.ones(200)/200, mode='same')
        mean_deconv[:200] = mean_deconv[200]
        mean_deconv[-200:] = mean_deconv[-201]
        trace['DeconvSignal'][k] = trace['DeconvSignal'][k] - mean_deconv

    # Downsample behavioral data from 6000 Hz to the specified rate
    print("      b. Downsample behavioral data.")
    downsample_factor = int(6000 / ds_behav_to)
    for k in res.keys():
        res[k] = res[k][::downsample_factor]
    # Convert behavioral time to ms and make it int
    behav_time = (res['behav_time']*1000).astype(np.int64)
    # Position is 0-100. Outliers indicate inter-trial intervals.
    behav_pos = res['behav_pos_y'].copy().astype(np.float64)
    behav_pos[(behav_pos >= 100)| (behav_pos < 0)] = np.nan
    trace['behav_time'] = behav_time
    trial_based_time = np.zeros_like(behav_time, dtype=np.int64)

    # Calculate trial start and end time.
    within_trial_idx = np.where(np.isnan(behav_pos) == False)[0]
    trial_change_idx = np.where(np.diff(behav_pos[within_trial_idx]) < 0)[0] + 1
    lap_beg_idx = np.concatenate(([0], trial_change_idx))
    lap_end_idx = np.concatenate((trial_change_idx, [len(within_trial_idx)]))
    trace['lap beg time'] = behav_time[within_trial_idx][lap_beg_idx]
    trace['lap end time'] = behav_time[within_trial_idx][lap_end_idx-1]
    trace['lap_beg_idx'] = within_trial_idx[lap_beg_idx]
    trace['lap_end_idx'] = within_trial_idx[lap_end_idx-1]
    print()
    
    for j in range(len(lap_beg_idx)):
        trial_based_time[within_trial_idx[lap_beg_idx[j]: lap_end_idx[j]]] = (
            behav_time[within_trial_idx][lap_beg_idx[j]: lap_end_idx[j]] - behav_time[within_trial_idx][lap_beg_idx[j]]
        )
    for j in range(len(lap_beg_idx)-1):
        trial_based_time[within_trial_idx[lap_end_idx[j]-1]: within_trial_idx[lap_beg_idx[j+1]]] = (
            behav_time[within_trial_idx[lap_end_idx[j]-1]: within_trial_idx[lap_beg_idx[j+1]]] - behav_time[within_trial_idx[lap_beg_idx[j+1]]]
        )
        print(trial_based_time[within_trial_idx[lap_end_idx[j]-1]: within_trial_idx[lap_beg_idx[j+1]]])
    
    trace['n_trials'] = len(lap_beg_idx)
    trace['map'] = res['map'][within_trial_idx][lap_beg_idx].astype(np.int64)
    trace['behav_time_aligned'] = trial_based_time
    
    # Calculate speed
    print("      c. Process real-time speed.")
    behav_speed = np.zeros_like(behav_pos, dtype=np.float64) * np.nan 
    behav_speed_raw = np.zeros_like(behav_pos, dtype=np.float64) * np.nan
    for j in tqdm(range(len(lap_beg_idx))):
        idx = within_trial_idx[lap_beg_idx[j]: lap_end_idx[j]]
        dt = np.append(np.diff(behav_time[idx]) / 1000, int(1/ds_behav_to))
        dx = np.append(np.diff(behav_pos[idx]), 0)
        behav_speed_raw[idx] = dx/dt
        # Smooth speed.
        behav_speed[idx] = np.convolve(
            dx, np.ones(speed_smooth_win), mode='same'
        ) / np.convolve(dt, np.ones(speed_smooth_win), mode='same') 
    trace['behav_speed_raw'] = behav_speed_raw
    trace['behav_speed'] = np.clip(behav_speed, speed_range[0], speed_range[1]) 
    trace['behav_pos'] = behav_pos
    trace['behav_nodes'] = (behav_pos//2)
    trace['behav_nodes'][np.isnan(trace['behav_nodes'])] = -1
    trace['behav_nodes'] = trace['behav_nodes'].astype(np.int64)
    
    # Coordinate neural activity and behavioral data
    print("      d. Coordinate neural activity and behavioral data.")
    coord_idx = coordinate_recording_time(ms_time, behav_time)
    ms_speed = behav_speed[coord_idx]
    ms_pos = behav_pos[coord_idx]
    ms_nodes = trace['behav_nodes'][coord_idx].astype(np.int64)
    ms_map = res['map'][coord_idx].astype(np.int64)
    ms_time_aligned = trace['behav_time_aligned'][coord_idx]
    trace['ms_time'] = ms_time
    trace['ms_time_aligned'] = ms_time_aligned
    trace['ms_speed'] = ms_speed
    trace['ms_pos'] = ms_pos
    trace['spike_nodes'] = ms_nodes
    trace['ms_map'] = ms_map
    spikes = np.where(
        trace['DeconvSignal'] - 3*np.std(trace['DeconvSignal'], axis=1, keepdims=True) >= 0, 
        1, 0
    )
    trace['Spikes'] = spikes
        
    print("  3. Calculate Mean dF/F Map")
    print("      a. Linear Map 1D")
    trace['n_neuron'] = trace['RawTraces'].shape[0]
    rate_map_all = np.zeros(
        (trace['n_neuron'], n_bin, n_map), dtype=np.float64
    )
    t_total = np.zeros(n_map, dtype=np.float64)
    t_nodes_frac = np.zeros((n_map, n_bin), dtype=np.float64)
    for n in range(n_map):
        idx = np.where((ms_nodes >= 0) & (ms_map == map_ids[n]))[0]
        spike_train = SpikeTrain(
            spikes[:, idx],
            time=trace['ms_time'][idx],
            variable=VariableBin(ms_nodes[idx])
        ) 
        rate_map_all[:, :, n] = spike_train.calc_tuning_curve(nbins=n_bin, t_interv_limits=2000).to_array()
        t_total[n] = spike_train.calc_total_time(t_interv_limits=2000) / 1000
        t_nodes_frac[n, :] = spike_train.calc_occu_time(t_interv_limits=2000, nbins=n_bin) / 1000 / (t_total[n] + 1e-8)
    trace['rate_map_all'] = rate_map_all
    
    # Smooth the rate map along the spatial dimension.
    sigma = 1
    gkernel = np.exp(-0.5 * (np.linspace(-3, 3, 7) / sigma)**2)
    gkernel /= gkernel.sum()
    print("          Smooth the rate map.")
    smooth_map_all = np.zeros_like(rate_map_all)
    for n in range(n_map):
        for i in range(trace['n_neuron']):
            smooth_map_all[i, :, n] = np.convolve(
                rate_map_all[i, :, n], gkernel, mode='same'
            )
    trace['smooth_map_all'] = smooth_map_all
    
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
    for n in range(n_map):
        idx = np.where((ms_nodes >= 0) & (ms_speed_bin >= 0) & (ms_map == map_ids[n]))[0]
        spike_train = SpikeTrain(
            spikes[:, idx],
            time=trace['ms_time'][idx],
            variable=VariableBin(ms_nodes[idx] + n_bin * ms_speed_bin[idx])
        )
        tuning_curve_speed = spike_train.calc_tuning_curve(nbins=n_bin*n_speed_bin).to_array()
        speed_map_all[:, :, :, n] = np.reshape(
            tuning_curve_speed, (trace['n_neuron'], n_bin, n_speed_bin)
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
    
    # Calculate information score
    SI = np.zeros((trace['n_neuron'], n_map), dtype=np.float64)
    for map in range(n_map):
        idx = np.where((ms_nodes >= 0) & (ms_map == map_ids[map]))[0]
        SI[:, map] = calc_SI(
            spikes[:, idx], 
            rate_map_all[:, :, map], 
            t_total[map], 
            t_nodes_frac[map, :]
        )
    trace['SI'] = SI
    
    # Save the processed data
    save_path = os.path.join(save_dir, f"trace.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(trace, f)
    # Print the finished time of processing
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}      Done.", end="\n\n\n")
    

if __name__ == "__main__":
    info = {
        "FishID": ["10136"],
        "session": [1],
        "suite2p_dir": [r"E:\10136\S1_20260223_222901\combined"],
        "behav_dir": [r"E:\10136\S1_20260223_222901\res.16chFlt"]
    }
    sheet_file = pd.DataFrame(info)
    for i in range(len(sheet_file)):
        run_LinearTrack1D(i, sheet_file)