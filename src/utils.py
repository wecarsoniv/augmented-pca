# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  misc_utils.py
# Author:  Billy Carson
# Date written:  01-27-2020
# Last modified:  01-27-2020

"""
Description:  Miscellaneous utilities.
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT MODULES
# ----------------------------------------------------------------------------------------------------------------------

# Import modules
import os
import warnings
import numpy as np
from numpy import around, sqrt, ravel
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# Returns time in days, hours, minutes, seconds
def get_converted_time(time_sec):
    """
    Description
    -----------
    Displays time in days, hours, minutes, seconds when given time in seconds.

    Parameters
    ----------
    time_sec : float; number of seconds

    Returns
    -------
    time_str : string; string of elapsed time in days, hours, minutes, seconds
    """

    # Calculate number of days, hours, minutes, and seconds
    n_days = int(time_sec / (24 * 60 * 60))
    n_hours = int((time_sec - (n_days * 24 * 60 * 60)) / (60 * 60))
    n_mins = int((time_sec - (n_days * 24 * 60 * 60) - (n_hours * 60 * 60)) / 60)
    n_sec = int(time_sec - (n_days * 24 * 60 * 60) - (n_hours * 60 * 60) - (n_mins * 60))

    # Create time string
    if (n_days == 0) & (n_hours == 0) & (n_mins == 0):
        time_str = str(n_sec) + ' sec'
    elif (n_days == 0) & (n_hours == 0):
        time_str = str(n_mins) + ' min, ' + str(n_sec) + ' sec'
    elif (n_days == 0):
        if n_hours == 1:
            time_str = str(n_hours) + ' hour, ' + str(n_mins) + ' min, ' + str(n_sec) + ' sec'
        else:
            time_str = str(n_hours) + ' hours, ' + str(n_mins) + ' min, ' + str(n_sec) + ' sec'
    else:
        if n_days == 1:
            if n_hours == 1:
                time_str = str(n_days) + ' day, ' + str(n_hours) + ' hour, ' + str(n_mins) + ' min, ' + str(n_sec) + ' sec'
            else:
                time_str = str(n_days) + ' day, ' + str(n_hours) + ' hours, ' + str(n_mins) + ' min, ' + str(n_sec) + ' sec'
        else:
            if n_hours == 1:
                time_str = str(n_days) + ' days, ' + str(n_hours) + ' hour, ' + str(n_mins) + ' min, ' + str(n_sec) + ' sec'
            else:
                time_str = str(n_days) + ' days, ' + str(n_hours) + ' hours, ' + str(n_mins) + ' min, ' + str(n_sec) + ' sec'

    # Return time string
    return time_str


# Converts numpy array of class labels to one-hot array
def convert_to_one_hot(labels):
    """
    Description
    -----------
    Converts numpy array of class labels to one-hot array.

    Parameters
    ----------
    labels : list or ndarray; class labels

    Returns
    -------
    one_hot_arr : ndarray; one-hot encoding of class labels
    """

    # Convert labels to one-hot encoding
    if type(labels) is list:
        labels_arr = np.array(labels)
    else:
        labels_arr = labels
    one_hot_arr = np.zeros((labels_arr.size, labels_arr.max() + 1))
    one_hot_arr[np.arange(labels_arr.size), labels_arr] = 1

    # Return one-hot encoding array
    return one_hot_arr

