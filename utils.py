import pickle
import cv2
import h5py
import copy as cp
import time
import shutil
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib.axes import Axes

from typing import KeysView, Union, Optional

from tqdm import tqdm
import scipy.stats
import sklearn.preprocessing
from scipy.stats import pearsonr, linregress
from scipy.stats import ttest_1samp, ttest_ind, levene, f_oneway, ttest_rel
from matplotlib.gridspec import GridSpec
import gc

import numpy as np
import os
import matplotlib.pyplot as plt
from os.path import join, exists
from .swim import OPTO_POS
from ._classes import SwimDataDict, LengthAlignedDict

ChannelColors = ['#d56e9e', '#3c619a']

def mkdir(path: str) -> bool:
    """
    Create a directory if it does not exist.
    
    Parameter
    ---------
    path: str
        The path of the directory to be created.
        
    Returns
    -------
    bool
        True if the directory was created, False if it already existed.
    """
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(f"        {path} is made up successfully!")
        return True
    else:
        print(f"        {path} is already existed!")
        return False
    
def DateTime(is_print: bool = False) -> str:
    """
    Return current time with certain format: e.g. 2022-09-16 13:50:42

    Parameter
    ---------
    is_print: bool, default (False)
        whether print the time.

    Return
    ------
    str
        the current time with certain form.
    """
    if is_print:
        t1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print(t1)
        return t1
    else:
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    
def Clear_Axes(
    ax: Axes,
    close_spines: list[str] = ['top','bottom','left','right'],
    xticks: list[float] | np.ndarray = [],
    yticks: list[float] | np.ndarray = [],
    ifxticks: bool = False,
    ifyticks: bool = False
) -> Axes:
    """
    Notes
    -----
    This function is used to clear the edges and axes of the input axes object.
    It operates over the original axes object and return the modified axes 
    object, rather than copy the input axes object and return a new one.

    Parameters
    ----------
    axes: <class 'matplotlib.axes._subplots.AxesSubplot'>
        Input the canvas axes object whose edges and axes should be cleared.
    close_spines: list[str]
        Contains str objects. Only 'top', 'bottom','left' and 'right' are valid
        values. The input edge(s)/spine(s) will be cleared, for example, if 
        input ['top','right'], Only the top and right edges will be removed and 
        the bottom and left edges will be maintained.
    xticks: list[float] or numpy.ndarray object. 
        The default value is an empty list and it means no tick of x axis will 
        be shown on the figure. If you input a list, the number in this list 
        will be shown as x ticks on the figure.
    yticks: list[float] or numpy.ndarray object. 
        Similar as xticks. The default value is an empty list and it means no 
        tick of y axis will be shown on the figure. If you input a list, the 
        number in this list will be shown as y ticks on the figure.
    ifxticks: bool and The default value is False.
        If it is False, the xticks will not shown unless the parameter xticks 
        is not an empty list. If it is true, the xticks will shown automatically.
        So if you want to manually set the value of x ticks, you remain this 
        parameter as False and input the xticks list or np.ndarray object that 
        you want to set as xticks through parameter 'xticks'.
    ifyticks: bool and The default value is False.
        If it is False, the yticks will not shown unless the parameter yticks 
        is not an empty list. If it is true, the yticks will shown automatically.
        So if you want to manually set the value of y ticks, you remain this 
        parameter as False and input the yticks list or np.ndarray object that
        you want to set as yticks through parameter 'yticks'.

    Returns
    -------
    - axes: <class 'matplotlib.axes._subplots.AxesSubplot'>. Return the axes 
    object after clearing.
    """
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    
    ax.tick_params(width=0.5)
    
    for s in close_spines:
        ax.spines[s].set_visible(False)
    
    if ifxticks == False:
        ax.set_xticks(xticks)
    if ifyticks == False:
        ax.set_yticks(yticks)
    
    return ax

def SubDict(
    data: LengthAlignedDict,
    idx: np.ndarray, 
    keys: Optional[str | list | KeysView] = 'all'
) -> LengthAlignedDict:
    """
    Return a subset of data. 
    
    Notes
    -----
    Each key should have values with the same length, otherwise an error will
    be raised. The idx should be a list or np.ndarray object, which contains

    Parameters
    ----------
    data: dict, 
        The dictionary that you want to extract subset from.
    idx: list or np.ndarray object, default (None)
        The index of elements that you want to keep in subset.
    keys: list, 
        Contains keys that you want to keep in subset.

    Returns
    -------
    subdic: dict
        The subset of input dictionary.
    """
    if isinstance(data, dict) and not isinstance(data, LengthAlignedDict):
        data = LengthAlignedDict(data)
    
    assert isinstance(data, LengthAlignedDict), "data should be a LengthAlignedDict!"
    assert isinstance(idx, np.ndarray), "idx should be a np.ndarray object!"
    
    return data.subdict(idx, dict_keys=keys)

def MergeDicts(
    dict1: LengthAlignedDict,
    dict2: LengthAlignedDict,
    keys: Optional[str | list | KeysView] = 'all'
) -> LengthAlignedDict:
    """
    Merge two LengthAlignedDict objects by concatenating their values along 
    the first axis.

    Parameters
    ----------
    dict1: LengthAlignedDict
        The first LengthAlignedDict to merge.
    dict2: LengthAlignedDict
        The second LengthAlignedDict to merge.

    Returns
    -------
    LengthAlignedDict
        A new LengthAlignedDict containing the merged data.
    """
    if isinstance(dict1, dict) and not isinstance(dict1, LengthAlignedDict):
        dict1 = LengthAlignedDict(dict1)
    if isinstance(dict2, dict) and not isinstance(dict2, LengthAlignedDict):
        dict2 = LengthAlignedDict(dict2)
    
    return (
        dict1.subdict(np.arange(dict1.shape[0]), dict_keys=keys) + 
        dict2.subdict(np.arange(dict2.shape[0]), dict_keys=keys)
    )


def readout_paradigm(paradm_code: int) -> str:
    """
    Read out the paradigm name according to the input paradigm code.

    Parameters
    ----------
    paradm_code: int
        The code of the paradigm.

    Returns
    -------
    str
        The name of the paradigm.
    """
    if isinstance(paradm_code, SwimDataDict):
        # the user incorrectly input the results of import16chFlt function
        paradm_code = paradm_code['Paradigm'][0]
    elif isinstance(paradm_code, np.ndarray):
        # the user input the 'Paradigm' field of import16chFlt function result
        # but forget to get the first element (because it is a 1-element array)
        paradm_code = paradm_code[0]
    elif isinstance(paradm_code, int) or isinstance(paradm_code, np.integer):
        pass
    else:
        raise TypeError(
            f'paradm_code should be int, dict or np.ndarray, but got '
            f'{type(paradm_code)}'
        )

    if paradm_code == 59595:
        return 'GoDarkGo'
    elif paradm_code == 20250916:
        return '1D Open-loop Navigation'
    elif paradm_code == 20250921 or paradm_code == 20250920: 
        # I don't know why some old results have paradm_code 20250920
        return '1D Close-loop Navigation'
    elif paradm_code == 20251102:
        return '2D Close-loop Open Field'
    elif paradm_code == 20251109:
        return '2D Set Offset'
    elif paradm_code == 20251113 or paradm_code == 20251112: 
        # In some old results, the paradigm code is 202511112'
        return '1D Close-loop Speed-coupled Optogenetics'
    elif paradm_code == 20251114:
        return 'Manual Optogenetics Pretest'
    else:
        raise ValueError(f'Unknown paradigm code: {paradm_code}')