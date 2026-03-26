# -*- coding: utf-8 -*-
"""
Created on Tue Feb 04 16:40:04 2014
script for importing an .10ch files
@author: nvladimus

Modifications made on Mon Aug 25 15:51:06 2025
to accommodate latest versions of Python, idiomatic use of NumPy, 
and add some annotations for future learners.
@Shuyang Yao

Modifications made on Mon Feb 22 18:01:06 2026
for
1. Define import 16 channel
"""
import numpy as np
from ._classes import SwimDataDict
import os
import pickle

class FileExtensionError(Exception):
    pass

def import10ch(filename: str) -> SwimDataDict:
    """ 
    Imports *.10ch or *.10chFlt file and parses it into Excel-like
    data formats (Dictionary). Dictionary can be easily converted to
    pandas DataFrame for further analysis.

    Parameters
    ----------
    filename : str
        The name of the file to import.

    Returns
    -------
    data : dict
        The imported data.\\
        For *.10ch file, the data dictionary contains the following keys:
        - 't': time vector
        - 'ch0': channel 0 data
        - 'ch1': channel 1 data
        - 'fltCh0': filtered channel 0 data
        - 'fltCh1': filtered channel 1 data
        - 'camTrigger': camera trigger signal
        - '2pTrigger': two-photon trigger signal
        - 'drift': drift signal
        - 'speed': speed signal
        - 'gain': gain signal
        - 'temp': ?
        \\
        For *.10chFlt files, the data dictionary contains the following keys:
        - 't': time vector
        - 'ch0': channel 0 data
        - 'ch1': channel 1 data
        - 'fltCh0': filtered channel 0 data
        - 'fltCh1': filtered channel 1 data
        - 'camTrigger': camera trigger signal
        - 'drift': drift signal
        - 'gain': gain signal

    Note
    ----
    10ch files contain 10 channels of data, with each channel represented
    some specific parameters of the experiment (e.g., time, channel 0,
    channel 1, gain, drift, speed etc.)
    """ 
    if (
        not filename.endswith('.10ch') and 
        not filename.endswith('.10chFlt')
    ):
        raise FileExtensionError(
            f"File must be of type .10ch, or .10chFlt, but got "
            f"{filename.split('.')[-1]}."
        )
    """
    f = open(filename, 'rb')
    A =  np.fromfile(f, np.float32).reshape((-1,10)).T
    f.close()
    """
    with open(filename, 'rb') as f:
        A = np.fromfile(f, np.float32).reshape((-1, 10)).T
    
    if filename.endswith('.10ch'):
        data = {}
        # Create a Gaussian kernel for smoothing with sigma = 20
        ker = np.exp(-np.arange(-60, 61)**2 / (2 * 20**2.))
        ker /= np.sum(ker)
        ch1 = A[0,:]
        smch1 = np.convolve(ch1, ker, mode='same')
        pow1 = (ch1 - smch1)**2
        ch2 = A[1, :]
        smch2 = np.convolve(ch2, ker, mode='same')
        pow2 = (ch2 - smch2)**2    
        data['t'] = np.arange(1, A.shape[1] + 1) / 6000
        data['ch0'] = ch1
        data['ch1'] = ch2
        data['fltCh0'] = np.convolve(pow1, ker, mode='same')
        data['fltCh1'] = np.convolve(pow2, ker, mode='same')
        data['gain'] = A[4, :]
        data['drift'] = A[5, :]
        data['speed'] = A[6, :]
        data['camTrigger'] = A[7, :]
        data['2pTrigger'] = A[8, :]
        data['temp'] = A[9, :]
        data = SwimDataDict(data, extension='.10ch')
        
    elif filename.endswith('.10chFlt'):
        data = {}
        # Create a Gaussian kernel for smoothing with sigma = 20
        ker = np.exp(-np.arange(-60, 61)**2 / (2 * 20**2.))
        ker /= np.sum(ker)
        ch1 = A[0, :]
        smch1 = np.convolve(ch1, ker, mode='same')
        pow1 = (ch1 - smch1)**2
        ch2 = A[1, :]
        smch2 = np.convolve(ch2, ker, mode='same')
        pow2 = (ch2 - smch2)**2    
        data['t'] = np.arange(1, A.shape[1] + 1) / 6000
        data['ch0'] = ch1
        data['ch1'] = ch2
        data['fltCh0'] = np.convolve(pow1, ker, mode='same')
        data['fltCh1'] = np.convolve(pow2, ker, mode='same')
        data['camTrigger'] = A[2, :]
        data['drift'] = A[6, :]
        data['gain'] = A[9, :]
        data = SwimDataDict(data, extension='.10chFlt')
    
    return data

def import12chFlt(filename: str) -> SwimDataDict:
    """
    Imports *.12chFlt file and parses it into Excel-like
    data formats (Dictionary). Dictionary can be easily converted to
    pandas DataFrame for further analysis.
    
    Parameters
    ----------
    filename : str
        The name of the file to import.
        
    Returns
    -------
    data : dict
        The imported data.\\
        For *.12chFlt files, the data dictionary contains the following keys:
        - 'behav_time': time vector
        - 'ch0': channel 0 data
        - 'ch1': channel 1 data
        - 'fltCh0': filtered channel 0 data
        - 'fltCh1': filtered channel 1 data
        - 'n_trials': number of trials
        - 'behav_pos': behavioral position
        - 'in_trial_time': in-trial time
        - 'behav_speed': behavioral speed
        - 'opto_states': optogenetic states
        - 'Paradigm': paradigm type
        - 'Stim Type': stimulus type
        - 'pass_speed': passive speed
        - 'active_gain': active gain
        - 'swim_speed': swim speed
    """
    if not filename.endswith('.12chFlt'):
        raise FileExtensionError(
            f"File must be of type .12chFlt, but got "
            f"{filename.split('.')[-1]}."
        )
    with open(filename, 'rb') as f:
        A = np.fromfile(f, np.float32).reshape((-1, 12)).T

    data = {}
    # Create a Gaussian kernel for smoothing with sigma = 20
    ker = np.exp(-np.arange(-60, 61)**2 / (2 * 20**2.))
    ker /= np.sum(ker)
    ch1 = A[0, :]
    smch1 = np.convolve(ch1, ker, mode='same')
    pow1 = (ch1 - smch1)**2
    ch2 = A[1, :]
    smch2 = np.convolve(ch2, ker, mode='same')
    pow2 = (ch2 - smch2)**2
    data['behav_time'] = (np.arange(1, A.shape[1] + 1) / 6000).astype(np.float64)
    data['ch0'] = ch1.astype(np.float64)
    data['ch1'] = ch2.astype(np.float64)
    data['fltCh0'] = np.convolve(pow1, ker, mode='same').astype(np.float64)
    data['fltCh1'] = np.convolve(pow2, ker, mode='same').astype(np.float64)
    data['n_trials'] = A[2, :].astype(np.int64)
    data['behav_pos'] = A[3, :].astype(np.float64)
    data['in_trial_time'] = A[4, :].astype(np.float64)
    data['behav_speed'] = A[5, :].astype(np.float64)
    data['opto_states'] = A[6, :].astype(np.int64)
    data['Paradigm'] = A[7, :].astype(np.int64)
    data['Stim Type'] = A[8, :].astype(np.int64)
    data['pass_speed'] = A[9, :].astype(np.float64)
    data['active_gain'] = A[10, :].astype(np.float64)
    data['swim_speed'] = A[11, :].astype(np.float64)
    data = SwimDataDict(data, extension='.12chFlt')
        
    return data

def import16chFlt(filename: str, nchannel: int=21) -> SwimDataDict:
    """
    Imports *.16chFlt file and parses it into Excel-like
    data formats (Dictionary). Dictionary can be easily converted to
    pandas DataFrame for further analysis.
    
    Parameters
    ----------
    filename : str
        The name of the file to import.
        
    Returns
    -------
    data : dict
        The imported data.\\
        For *.16chFlt files, the data dictionary contains the following keys:
        - 'behav_time': time vector
        - 'ch0': channel 0 data
        - 'ch1': channel 1 data
        - 'fltCh0': filtered channel 0 data
        - 'fltCh1': filtered channel 1 data
        - 'n_trials': number of trials
        - 'behav_pos_x': behavioral position x
        - 'behav_pos_y': behavioral position y
        - 'in_trial_time': in-trial time
        - 'behav_speed_x': behavioral speed x
        - 'behav_speed_y': behavioral speed y
        - 'behav_orient': behavioral orientation
        - 'opto_states': optogenetic states
        - 'Paradigm': paradigm type
        - 'Stim Type': stimulus type
        - 'pass_speed': passive speed
        - 'swim_gain': active gain
        - 'swim_speed': swim speed
        - 'turn_gain': turn gain
        - 'led_power': LED power (if available)
        - 'AI2': Green Camera signal (if available)
        - 'AI3': Red Camera signal (if available)
        - 'AI4': IR camera (if available)
        - 'map': Map ID (if available)
    """
    if not filename.endswith('.16chFlt'):
        raise FileExtensionError(
            f"File must be of type .16chFlt, but got "
            f"{filename.split('.')[-1]}."
        )
    try:
        with open(filename, 'rb') as f:
            A = np.fromfile(f, np.float32).reshape((-1, nchannel)).T
    except Exception as exc:
        raise IOError(
            f"Error reading file {filename}: {exc}\n"
            f"The number of channels (nchannel) may need to be adjusted based "
            f"on the actual file format."
        )

    data = {}
    # Create a Gaussian kernel for smoothing with sigma = 20
    ker = np.exp(-np.arange(-60, 61)**2 / (2 * 20**2.))
    ker /= np.sum(ker)
    ch1 = A[0, :]
    smch1 = np.convolve(ch1, ker, mode='same')
    
    pow1 = (ch1 - smch1)**2
    ch2 = A[1, :]
    smch2 = np.convolve(ch2, ker, mode='same')
    pow2 = (ch2 - smch2)**2
    data['behav_time'] = (np.arange(1, A.shape[1] + 1) / 6000).astype(np.float64)
    data['ch0'] = ch1.astype(np.float64)
    data['ch1'] = ch2.astype(np.float64)
    data['fltCh0'] = np.convolve(pow1, ker, mode='same').astype(np.float64)
    data['fltCh1'] = np.convolve(pow2, ker, mode='same').astype(np.float64)
    data['n_trials'] = A[2, :].astype(np.int64)
    data['behav_pos_x'] = A[3, :].astype(np.float64)
    data['behav_pos_y'] = A[4, :].astype(np.float64)
    data['in_trial_time'] = A[5, :].astype(np.float64)
    data['behav_speed_x'] = A[6, :].astype(np.float64)
    data['behav_speed_y'] = A[7, :].astype(np.float64)
    data['behav_orient'] = A[8, :].astype(np.float64)
    
    data['opto_states'] = A[9, :].astype(np.int64)
    data['Paradigm'] = A[10, :].astype(np.int64)
    data['Stim Type'] = A[11, :].astype(np.int64)
    data['pass_speed'] = A[12, :].astype(np.float64)
    data['swim_gain'] = A[13, :].astype(np.float64)
    data['swim_speed'] = A[14, :].astype(np.float64)
    data['turn_gain'] = A[15, :].astype(np.float64)
    try:
        data['led_power'] = A[16, :].astype(np.float64)
        data['AI2'] = A[17, :].astype(np.float64)
        data['AI3'] = A[18, :].astype(np.float64)
        data['AI4'] = A[19, :].astype(np.float64)
        data['map'] = A[20, :].astype(np.int64)
    except:
        pass
        
    data = SwimDataDict(data, extension='.16chFlt')
    
    return data

def import_suite2p(file_dir: str)-> dict:
    with open(os.path.join(file_dir, 'F.npy'), 'rb') as f:
        RawTraces: np.ndarray = np.asarray(np.load(f, allow_pickle=True), dtype=np.float32)
        
    with open(os.path.join(file_dir, 'Fneu.npy'), 'rb') as f:
        NeuropilTraces: np.ndarray = np.asarray(np.load(f, allow_pickle=True), dtype=np.float32)

    dF = RawTraces.copy() - 0.7 * NeuropilTraces
    snr = 1 - 0.5 * np.diff(dF, axis=1).var(axis=1) / dF.var(axis=1)  
        
    with open(os.path.join(file_dir, 'stat.npy'), 'rb') as f:
        stats: list = np.load(f, allow_pickle=True)
    
    try:
        snr = np.array([stat['snr'] for stat in stats], dtype=np.float32)
    except:
        dF = RawTraces.copy() - 0.7 * NeuropilTraces
        snr = 1 - 0.5 * np.diff(dF, axis=1).var(axis=1) / dF.var(axis=1)  
        
    rois = []
    for stat in stats:
        xpix = np.median(stat['xpix'])
        ypix = np.median(stat['ypix'])
        iplane = stat['iplane']
        rois.append((xpix, ypix, iplane))
    rois = np.asarray(rois, dtype=np.int32)
    
    idx = snr > 0.15
        
    with open(os.path.join(file_dir, 'spks.npy'), 'rb') as f:
        DeconvSignal: np.ndarray = np.asarray(np.load(f, allow_pickle=True), dtype=np.float32)

    RawTraces = RawTraces[idx, :]
    DeconvSignal = DeconvSignal[idx, :]
    Fneu = NeuropilTraces[idx, :]
    snr = snr[idx]
    rois = rois[idx, :]
    
    with open(os.path.join(file_dir, 'ops.npy'), 'rb') as f:
        obj: dict = np.load(f, allow_pickle=True).item()
    
    trace={
        'RawTraces': RawTraces,
        'DeconvSignal': DeconvSignal,
        'Fneu': Fneu,
        'iscell': idx,
        'roi_coord': rois,
        'snr': snr,
        'suite2p_version': obj['suite2p_version'],
        'nplanes': obj['nplanes'],
        'nchannels': obj['nchannels'],
        'tau': obj['tau'],
        'fs': obj['fs'],
        'pretrained_model': obj['pretrained_model'],
        'lenX': obj['lenX'],
        'lenY': obj['lenY'],
        'lenZ': obj['lenZ'],
        'meanImg': obj['meanImg'],
        'nframes': obj['nframes']
    }
    return trace