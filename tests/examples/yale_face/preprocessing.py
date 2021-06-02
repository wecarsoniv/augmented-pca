# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  preprocessing.py
# Author:  Billy Carson
# Date written:  05-12-2021
# Last modified:  05-12-2021

r"""
Image data preprocessing functions.
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT MODULES
# ----------------------------------------------------------------------------------------------------------------------

# Import modules
import os
import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# Remove outliers function
def remove_img_outliers(X, *arrs, outlier_thresh, copy=True):
    r"""
    """
    
    non_outlier_idx = np.where(np.mean(X, axis=1) > outlier_thresh)[0]
    
    if copy:
        non_outlier_arr_list = [X.copy()[non_outlier_idx]]
    else:
        non_outlier_arr_list = [X[non_outlier_idx, :]]
    
    for arr in arrs:
        if copy:
            non_outlier_arr_list.append(arr.copy()[non_outlier_idx])
        else:
            non_outlier_arr_list.append(arr[[non_outlier_idx]])
    
    return non_outlier_arr_list

    
def ref_img_adjust(img_ref_arr, img_recon_arr, ccmt, ccmt_thresh, copy=True):
    r"""
    """
    
    if copy:
        img_recon_arr = img_recon_arr.copy()
    
    col_half_idx = int(0.5 * img_ref_arr.shape[1])
    
    if ccmt > ccmt_thresh:
        ref_non_ccmt_mean = np.mean(img_ref_arr[:, :col_half_idx])
        recon_non_ccmt_mean = np.mean(img_recon_arr[:, :col_half_idx])
    elif ccmt < ccmt_thresh:
        ref_non_ccmt_mean = np.mean(img_ref_arr[:, col_half_idx:])
        recon_non_ccmt_mean = np.mean(img_recon_arr[:, col_half_idx:])
    else:
        ref_non_ccmt_mean = np.mean(img_ref_arr[:, col_half_idx:col_half_idx + 2])
        recon_non_ccmt_mean = np.mean(img_recon_arr[:, col_half_idx:col_half_idx + 2])
    
    intensity_dif = ref_non_ccmt_mean - recon_non_ccmt_mean
    
    return img_recon_arr + intensity_dif

    
# ----------------------------------------------------------------------------------------------------------------------
# CLASS DEFINITIONS
# ----------------------------------------------------------------------------------------------------------------------

# Range scaler object
class RangeScaler():
    r"""
    """

    # Instantiation method
    def __init__(self, feature_range=(0, 1), copy=True):
        r"""
        """
        self.feature_range = feature_range
        self.copy = copy
        
        self.data_min_ = None
        self.data_max_ = None
        self.data_mean_ = None
        self.data_scaled_mean_ = None
    
    # Fit method
    def fit(self, X, y=None):
        r"""
        """
        
        self.data_min_ = np.min(X)
        self.data_max_ = np.max(X)
        self.data_mean_ = np.mean(X)
    
    # Transform method
    def transform(self, X):
        r"""
        """
        
        if self.copy:
            X = X.copy()
        
        X_scaled = (X - self.data_min_) / (self.data_max_ - self.data_min_)
        X_scaled = (X_scaled * (self.feature_range[1] - self.feature_range[0])) + self.feature_range[0]
        
        self.data_scaled_mean_ = np.mean(X_scaled)

        return X_scaled
    
    # Fit-transform method
    def fit_transform(self, X, y=None):
        r"""
        """
        
        self.fit(X)
        X_scaled = self.transform(X)
        
        return X_scaled
    
    # Inverse transform method
    def inverse_transform(self, X, y=None):
        r"""
        """
        
        if self.copy:
            X = X.copy()
        
        X = (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        X = (X * (self.data_max_ - self.data_min_)) + self.data_min_
        
        return X
        
        
        
        