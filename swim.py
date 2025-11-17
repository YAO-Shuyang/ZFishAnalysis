"""
This code is designed for analyzing electrophysiological data.
"""

import numpy as np
import numba
from tqdm import tqdm
from ._classes import SwimDataDict, SwimRealignedDict
from typing import Optional

def _check_binary(array: np.ndarray) -> bool:
    if not isinstance(array, np.ndarray):
        return False
    
    unique_values = np.unique(array)
    if len(unique_values) != 2:
        return False
    
    if not np.array_equal(unique_values, [0, 1]):
        return False
    
    return True

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

# ===================== For GoDarkGo paradigm ===============================
@numba.jit(nopython=True)
def get_mean_top_values(data, v_percent: float | int = 99) -> float:
    threshold = np.percentile(data, v_percent)
    top_data = data[data >= threshold]
    mean_top = np.mean(top_data)
    return mean_top

def calc_engagement_score(
    avg_signals: np.ndarray,
    state: np.ndarray,
    n_shuffle: int = 1000,
    v_percent: float | int = 99
) -> tuple[float, float, np.ndarray]:
    """Calculate engagement score for paradigms like 'GoDarkGo' task.

    Parameters
    ----------
    avg_signals : np.ndarray
        The average swim power across two
    state : np.ndarray
        The binary behavioral states. By default, it assumes that state 0 is
        the engaged state.
    n_shuffle : int, optional
        The number of times for randomly rolling shuffles, by default 1000
    v_percent : float | int, optional
        The percentile of top swim signals used for the computation, by 
        default 99

    Returns
    -------
    engagement_score : float
        The engagement score.
    p_value : float
        The p-value determined by the shuffle test.
    engagement_shuf : np.ndarray, shape (n_shuffle,)
        The shuffled scores.
    """
    
    if avg_signals.shape[0] != state.shape[0]:
        raise ValueError(
            "The length of avg_signals and state must be the same."
            f" Got {avg_signals.shape[0]} and {state.shape[0]}."
        )
    
    # Validate if state is binary and if the two states are represented by 0 and 1
    if not _check_binary(state):
        raise ValueError(
            "The state array must be binary and contain only 0 and 1 values."
            f" Got unique values: {np.unique(state)}."
        )
    
    state0_idx = np.where(state == 0)[0]
    state1_idx = np.where(state == 1)[0]
    
    mean_top_state0 = get_mean_top_values(avg_signals[state0_idx], v_percent)
    mean_top_state1 = get_mean_top_values(avg_signals[state1_idx], v_percent)
    engagement_score = mean_top_state0 / (mean_top_state0 + mean_top_state1)
    
    engagement_shuf = np.zeros(n_shuffle)
    roll_shift = np.random.randint(avg_signals.shape[0], size=n_shuffle)

    for i in tqdm(range(n_shuffle)):
        state0_idx_shuf = (state0_idx + roll_shift[i]) % avg_signals.shape[0]
        state1_idx_shuf = (state1_idx + roll_shift[i]) % avg_signals.shape[0]
        mean_top_shuf_state0 = get_mean_top_values(avg_signals[state0_idx_shuf], v_percent)
        mean_top_shuf_state1 = get_mean_top_values(avg_signals[state1_idx_shuf], v_percent)
        engagement_shuf[i] = mean_top_shuf_state0 / (mean_top_shuf_state0 + mean_top_shuf_state1)

    p_value = np.mean(engagement_shuf >= engagement_score)

    return engagement_score, p_value, engagement_shuf


def event_based_realignment(
    data: SwimDataDict,
    state: np.ndarray | str = 'opto_states',
    segmented_keys: list | str = 'all',
    event_onset_state: int = 1,
    time_range: tuple[float, float] = (-2, 2),
    bin_size: float = 0.05,
    excluded_idx: Optional[np.ndarray] = None,
    is_shuffle: bool = False,
    **shuf_kw
) -> SwimRealignedDict:
    """
    Realign the timestamps of swim data based on the defined events.
    \\
    For example, if the state 1 represents the `onset of optogenetic stimulation`,
    then the time will be aligned to the onset of optogenetic stimulation.
    Only data points within the defined time range around the event onset will 
    be kept.

    Parameters
    ----------
    data: dict
        The dictionary that you want to extract subset from.
    state: np.ndarray
        The binary state array used for segmentation.
    segmented_keys: list or str, default ('all')
        Contains keys that you want to keep in subset. If 'all', all keys
        will be kept.
    event_onset_state : int, optional
        The state value that indicates the onset of the event, by default 1.
    time_range : tuple[float, float], optional, unit: seconds
        The time range around the event onset to include in the segmentation,
        by default (-2, 2). Others will be excluded.
    bin_size : float, optional, unit: seconds
        The bin size for realigned data, by default 0.05s.
    excluded_idx : np.ndarray, optional
        The excluded indexes are used to exclude specific events that fall 
        within this range. It should be a 1D numpy array with all the indices 
        that you want to exclude. \\
        e.g., exclude events that occur too close to the onset or offset of each
        trial. 
    is_shuffle : bool, optional
        Whether to perform shuffling on the state array before realignment, by 
        default False.
    **shuf_kw : keyword arguments
        Additional keyword arguments for shuffling function if is_shuffle is 
        True.
        - n_shuffle : int, optional
            The number of times for randomly rolling shuffles, by default 100.

    Raises
    ------
    ValueError
        If segmented_keys is empty.
        If any key in segmented_keys is not in data.keys().
        If state is str but not in data.keys().
        If manually input state array length does not match data arrays length.
        If state array is not binary.
    TypeError
        If state is neither str nor np.ndarray.

    Returns
    -------
    segm_data: SwimRealignedDict
        The segmented dictionary containing only the elements where state == 1.
    """
    if isinstance(segmented_keys, str):
        if segmented_keys == 'all':
            segmented_keys = list(data.keys())
        else:
            raise ValueError(
                f"Only 'all' is valid when segmented_keys is str, but got {segmented_keys}"
            )
    
    if len(segmented_keys) == 0:
        raise ValueError("segmented_keys should not be an empty list!")
    
    # all keys in segmented_keys should be in data.keys()
    for k in segmented_keys:
        if k not in data.keys():
            raise ValueError(
                f"All keys in segmented_keys should be in data.keys(). "
                f"Got {k} which is not in data.keys(): \n {data.keys()}."
            )
    
    if isinstance(state, str):
        if state not in data.keys():
            raise ValueError(
                f"When state is str, it should be one of the keys in data. "
                f"Got {state}, but data keys are {data.keys()}."
            )
        state_array = data[state]
    elif isinstance(state, np.ndarray):
        state_array = state
        if state_array.shape[0] != len(data[segmented_keys[0]]):
            raise ValueError(
                "Any manually input state array must have a length that "
                f"match the length of the data arrays."
            )
    else:
        raise TypeError(
            f"The type of state should be str or np.ndarray, but got {type(state)}."
        )
    
    if _check_binary(state_array) == False:
        raise ValueError(
            "The state array must be binary and contain only 0 and 1 values."
            f" Got unique values: {np.unique(state_array)}."
        )

    state_transitions = np.diff(state_array)
    event_onset_indices = np.where(
        (state_array[1:] == event_onset_state) &
        (state_transitions != 0)
    )[0] + 1
    
    behav_time = data['behav_time']
    
    realigned_data = {
        "realigned_time": [],
        'n_events': []
    }
    for key_names in segmented_keys:
        if key_names in ['behav_time', 'n_trials']:
            continue
        realigned_data[f"realigned_{key_names}"] = []
        
    n_bins = int((time_range[1] - time_range[0]) / bin_size)
    realigned_time = np.linspace(
        time_range[0], time_range[1], n_bins
    ) + bin_size / 2

    if is_shuffle:
        try:
            n_shuffle = shuf_kw.get('n_shuffle', 10)
        except:
            n_shuffle = 10
    else:
        n_shuffle = 1

    for n in tqdm(range(n_shuffle)):
        if is_shuffle:
            event_onset_indices = np.random.choice(
                behav_time.shape[0], size=len(event_onset_indices), replace=False
            )
        
        # Iterate through each event onset index
        for i, onset_idx in enumerate(event_onset_indices):
            # Exclude events that fall within the exclude range
            if excluded_idx is not None:
                if onset_idx in excluded_idx:
                    continue
                
            event_time = behav_time[onset_idx]
            start_time = event_time + time_range[0]
            end_time = event_time + time_range[1]

            time_mask = np.where(
                (behav_time >= start_time) & (behav_time <= end_time)
            )[0]

            for j, dt in enumerate(np.arange(time_range[0], time_range[1], bin_size)):
                if j >= n_bins:
                    break
                
                realigned_data['realigned_time'].append(realigned_time[j])
                realigned_data['n_events'].append(i)
                
                within_bin_mask = np.where(
                    (behav_time[time_mask] >= event_time + dt) &
                    (behav_time[time_mask] < event_time + (dt + bin_size)) &
                    (data['behav_speed_y'][time_mask] >= 0)
                )[0]
                
                for key_names in segmented_keys:
                    if key_names in ['behav_time', 'n_trials']:
                        continue
                    realigned_data[f"realigned_{key_names}"].append(
                        np.nanmean(data[key_names][time_mask][within_bin_mask])
                    )
    
    for k in realigned_data.keys():
        realigned_data[k] = np.asarray(realigned_data[k])
        
    return SwimRealignedDict(realigned_data)