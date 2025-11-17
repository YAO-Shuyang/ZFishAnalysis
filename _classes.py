import numpy as np
from typing import KeysView, Union, Optional

class LengthAlignedDict(dict):
    """ A dictionary subclass for storing length-aligned data. """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'LengthAlignedDict' object has no attribute '{key}'")
        
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        
        self._check_key_lengths()
        
    def _check_key_lengths(self):
        try:
            self._length = self[list(self.keys())[0]].shape[0]
        except:
            self._length = 0
            return
        
        for k in self.keys():
            if not isinstance(self[k], np.ndarray):
                raise TypeError(
                    "All values in LengthAlignedDict must be numpy arrays!"
                    f" Got {k} with type {type(self[k])}."
                )

            if self[k].shape[0] != self._length:
                raise ValueError(
                    "All arrays in LengthAlignedDict must have the same length!"
                    f" Got {k} with length {self[k].shape[0]}, "
                    f"while other arrays have length {self._length}."
                )
                
    def subdict(
        self, 
        idx: np.ndarray, 
        dict_keys: Optional[str | list | KeysView] = 'all'
    ) -> 'LengthAlignedDict':
        """
        Return a subset of the LengthAlignedDict based on the provided indices.
        """
        subdic = {}
        if isinstance(dict_keys, str):
            if dict_keys == 'all':
                dict_keys = self.keys()
            else:
                raise ValueError(
                    f"Only 'all' is valid when keys is str, but got {dict_keys}"
                )
        else:
            assert len(dict_keys) != 0, "keys should not be an empty list!"
            
        for k in dict_keys:
            subdic[k] = self[k][idx]
        return LengthAlignedDict(subdic)
    
    def __add__(self, other: 'LengthAlignedDict') -> 'LengthAlignedDict':
        """
        Merge two LengthAlignedDicts by concatenating their arrays along the 
        first axis.
        """
        if not isinstance(other, LengthAlignedDict):
            raise TypeError("Can only merge with another LengthAlignedDict!")
        merged_dict = {}
        for k in self.keys():
            if k in other:
                merged_dict[k] = np.concatenate((self[k], other[k]), axis=0)
            else:
                raise KeyError(f"Key {k} not found in both LengthAlignedDicts.")
        return LengthAlignedDict(merged_dict)
    
    @property
    def shape(self):
        return (self._length,)
    
    def __len__(self):
        return self._length
    
    def copy(self) -> 'LengthAlignedDict':
        """
        Create a deep copy of the LengthAlignedDict.
        """
        copied_dict = {k: np.copy(v) for k, v in self.items()}
        return LengthAlignedDict(copied_dict)

class SwimDataDict(LengthAlignedDict):
    """ A dictionary subclass for storing data with attribute-style access. """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DataDict' object has no attribute '{key}'")

    def __init__(self, *args, extension, **kwargs):
        super().__init__(*args, **kwargs)
        self._extension = extension

        self._check_required_keys()

    @property
    def extension(self):
        return self._extension
    
    def _check_required_keys(self):
        dict_keys = list(self.keys())
        if self.extension == '.10ch':
            for required_keys in [
                't', 'ch0', 'ch1', 'fltCh0', 'fltCh1', 'gain', 'drift', 'speed',
                'camTrigger', '2pTrigger', 'temp'
            ]:
                if required_keys not in dict_keys:
                    raise KeyError(f"Missing required key: {required_keys}")
        
        elif self.extension == '.10chFlt':
            for required_keys in [
                't', 'ch0', 'ch1', 'fltCh0', 'fltCh1', 'gain', 'drift', 
                'camTrigger'
            ]:
                if required_keys not in dict_keys:
                    raise KeyError(f"Missing required key: {required_keys}")
                
        elif self.extension == '.12chFlt':
            for required_keys in [
                'behav_time', 'ch0', 'ch1', 'fltCh0', 'fltCh1', 'n_trials',
                'behav_pos', 'in_trial_time', 'behav_speed', 'opto_states', 
                'Paradigm', 'Stim Type', 'pass_speed', 'active_gain',
                'swim_speed'
            ]:
                if required_keys not in dict_keys:
                    raise KeyError(f"Missing required key: {required_keys}")
                
        elif self.extension == '.16chFlt':
            for required_keys in [
                'behav_time', 'ch0', 'ch1', 'fltCh0', 'fltCh1', 'n_trials',
                'behav_pos_x', 'behav_pos_y', 'in_trial_time', 
                'behav_speed_x', 'behav_speed_y', 'behav_orient',
                'opto_states', 'Paradigm', 'Stim Type', 'pass_speed',
                'swim_gain', 'swim_speed', 'turn_gain'
            ]:
                if required_keys not in dict_keys:
                    raise KeyError(f"Missing required key: {required_keys}")
                
        else:
            raise NotImplementedError(
                f"Key checking not implemented for extension {self.extension}." 
            )

class SwimRealignedDict(LengthAlignedDict):
    """ A dictionary subclass for storing realigned data based on events. """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'SwimRealignedDict' object has no attribute '{key}'"
            )
        
    def __init__(self, data, *args, **kwargs):
        if 'n_events' not in data.keys():
            raise ValueError("SwimRealignedDict must contain 'n_events' key!")
        super().__init__(data, *args, **kwargs)