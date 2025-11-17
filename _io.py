# -*- coding: utf-8 -*-
"""
Created on Tue Feb 04 16:40:04 2014
script for importing an .10ch files
@author: nvladimus

Modifications made on Mon Aug 25 15:51:06 2025
to accommodate latest versions of Python, idiomatic use of NumPy, 
and add some annotations for future learners.
@Shuyang Yao
"""
import numpy as np
from ._classes import SwimDataDict

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

def import16chFlt(filename: str) -> SwimDataDict:
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
    """
    if not filename.endswith('.16chFlt'):
        raise FileExtensionError(
            f"File must be of type .16chFlt, but got "
            f"{filename.split('.')[-1]}."
        )
    with open(filename, 'rb') as f:
        A = np.fromfile(f, np.float32).reshape((-1, 16)).T

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
    data = SwimDataDict(data, extension='.16chFlt')
    
    return data

def importSuite2p(dir_path: str) -> dict:
    """
    Imports Suite2p output files from the specified directory.

    Parameters
    ----------
    dir_path : str
        The directory path containing Suite2p output files.

    Returns
    -------
    data : dict
        A dictionary containing Suite2p data.
    """
    # open numpy files
    file_names = [
        'ops.npy', 'stat.npy', 'iscell.npy', 'F.npy', 'Fneu.npy', 'spks.npy'
    ]
    data = {}
    for file_name in file_names:
        file_path = f"{dir_path}/{file_name}"
        data[file_name.split('.')[0]] = np.load(file_path, allow_pickle=True)
    return data

def stackInits(data: dict, thrMag: float = 3.8) -> np.ndarray:
    """
    finds stack onset indices in ephys data.
    
    Parameter
    ---------
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
    thrMag: float, by default 3.8
        The threshold magnitude for detecting stack events.
    """
    stackInits = np.where(data['camTrigger'][:] > thrMag)[0]
    initDiffs = np.where(np.diff(stackInits) > 1)[0]
    initDiffs = np.concatenate(([0], initDiffs + 1))    
    stackInits = stackInits[initDiffs]
    return stackInits

def getSwims(fltch, th = 2.5):

    peaksT,peaksIndT = getPeaks(fltch)
    thr = getThreshold(fltch,peaksT,90000, th)
    burstIndT = peaksIndT[np.where(fltch[peaksIndT] > thr[peaksIndT])]
    if len(burstIndT):
        burstT = np.zeros(fltch.shape)
        burstT[burstIndT] = 1
        
        interSwims = np.diff(burstIndT)
        swimEndIndB = np.where(interSwims > 800)[0]
        swimEndIndB = np.concatenate((swimEndIndB,[burstIndT.size-1]))

        swimStartIndB = swimEndIndB[0:-1] + 1
        swimStartIndB = np.concatenate(([0], swimStartIndB))
        nonShort = np.where(swimEndIndB != swimStartIndB)[0]
        swimStartIndB = swimStartIndB[nonShort]
        swimEndIndB = swimEndIndB[nonShort]
      
        bursts = np.zeros(fltch.size)
        starts = np.zeros(fltch.size)
        stops = np.zeros(fltch.size)
        bursts[burstIndT] = 1
        starts[burstIndT[swimStartIndB]] = 1
        stops[burstIndT[swimEndIndB]] = 1
    else:
        starts = []
        stops = []
    return starts, stops, thr, bursts


# filter signal, extract power
def smoothPower(ch, kern):
    smch = np.convolve(ch, kern, 'same')
    power = (ch - smch)**2
    fltch = np.convolve(power, kern, 'same')
    return fltch

# get peaks
def getPeaks(fltch,deadTime=40):
    
    aa = np.diff(fltch)
    peaks = (aa[0:-1] > 0) * (aa[1:] < 0)
    inds = np.where(peaks)[0]    

    # take the difference between consecutive indices
    dInds = np.diff(inds)
                    
    # find differences greater than deadtime
    toKeep = (dInds > deadTime)    
    
    # only keep the indices corresponding to differences greater than deadT 
    inds[1::] = inds[1::] * toKeep
    inds = inds[inds.nonzero()]
    
    peaks = np.zeros(fltch.size)
    peaks[inds] = 1
    
    return peaks,inds

# find threshold
def getThreshold(fltch,peaks,wind=90000,shiftScale=2.5):
    ''' edited by nvladimus '''
    ''' edited by eyang '''
    th = np.zeros(fltch.shape)
    
    for t in np.arange(0,fltch.size-wind, wind):

        interval = np.arange(0, t+wind)
        peaksInd = np.where(peaks[interval])        
        xH = np.arange(0,np.min([fltch[peaksInd].max(),0.005]),1e-5) # changed upper value to max of signal
        # make histogram of peak values
        peakHist = np.histogram(fltch[peaksInd], xH)[0]
        # find modal value
        mx = np.min(np.where(peakHist == np.max(peakHist)))        
        # find distance between mode and foot of left side of histogram
        if xH[mx] <  np.percentile(fltch[peaksInd],40):
            
            bound=np.median(fltch[peaksInd])
            
        else:
            bound=xH[mx]
            
        # in case the signal is skewered, use median instead
        if peakHist[0] < peakHist[mx]/100.0:
            mn = np.max(np.where(peakHist[0:mx] < peakHist[mx]/100.0))  
        else:
            mn = 0
            
        th[t:t+wind] =  bound + shiftScale * ( bound- xH[mn])
        
        
    th[t+wind:] = th[t+wind-1]
    return th
