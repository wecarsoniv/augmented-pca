# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  metrics.py
# Author:  Billy Carson
# Date written:  04-17-2021
# Last modified:  04-20-2021

"""
Description:  Augmented Principal Component Analysis (APCA) evaluation metrics definitions file.
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT MODULES
# ----------------------------------------------------------------------------------------------------------------------

# Import modules
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# Reconstruction error function
def reconstruct_error(a, a_recon, reduction='mean'):
    """
    Description
    -----------
    
    Parameters
    ----------
    
    Returns
    -------
    """
    
    # Check array types
    if (type(a) is not np.ndarray) | (type(a_recon) is not np.ndarray):
        raise TypeError('Arrays must be of type numpy.ndarray.')
    
    # Check array sizes
    if a.shape != a_recon.shape:
        raise ValueError('Arrays must be the same shape.')
    
    # Check reduction type and value
    if (reduction != 'mean') & (reduction != 'sum'):
        raise ValueError('Only \"mean\" and \"sum\" supported for reconstruction error array reduction.')
    
    # Calculate array of element reconstruction errors
    err_arr = (a - a_recon) ** 2
    
    # Reduce reconstruction error array
    if reduction == 'mean':
        err = np.mean(err_arr)
    else:
        err = np.sum(err_arr)
    
    # Return reconstruction error
    return err

