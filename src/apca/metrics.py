# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  metrics.py
# Author:  Billy Carson
# Date written:  04-17-2021
# Last modified:  11-04-2021

r"""
AugmentedPCA evaluation metric functions.
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT STATEMENTS
# ----------------------------------------------------------------------------------------------------------------------

# Import statements
import numpy
from numpy import mean, sum


# ----------------------------------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# Reconstruction error function
def reconstruct_error(a: numpy.ndarray, a_recon: numpy.ndarray, reduction: str='mean') -> numpy.ndarray:
    r"""
    Computes the reconstruction error between two matrices. This function can be used to determine the reconstruction 
    error between original data matrices (primary or augmenting) and data matrices reconstructed via APCA.
    
    Parameters
    ----------
    a : numpy.ndarray
        Original data matrix.
    a_recon : numpy.ndarray
        Reconstructed data matrix.
    reduction : str; optional, default is 'mean'
        Specifies the reduction to apply to the output: 'mean' or 'sum'. 'mean':  mean of the squared error is taken, 
        'sum': sum of the squared error is taken.
    
    Returns
    -------
    err : float
        Reconstruction error between `a` and `a_recon`. 
    """
    
    # Check array types
    if (type(a) is not numpy.ndarray) | (type(a_recon) is not numpy.ndarray):
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
        err = mean(err_arr)
    else:
        err = sum(err_arr)
    
    # Return reconstruction error
    return err

