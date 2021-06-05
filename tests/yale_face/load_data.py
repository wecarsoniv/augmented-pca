# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  load_data.py
# Author:  Billy Carson
# Date written:  05-12-2021
# Last modified:  05-12-2021

r"""
Load images from the extended Yale face dataset.
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

# Load cropped Yale face dataset B into matrix
def load_cropped_yale_face(dir_path, resize_factor=0.25):
    # Initialize feature and label lists
    X = []
    Y = []
    labels_id = []
    labels_shadow = []
    
    # Get list of sub-directories
    subdir_list = [x for x in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, x))]
    
    # Iterate over sub-directories
    for subdir in subdir_list:
        # Get list of files in sub-directory
        subdir_path = dir_path + subdir + '/PNG/'
        file_list = [x for x in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, x))]
        
        # Load images in subdirectory
        for file in file_list:
            # Only load images
            if ('.png' in file) & ('bad' not in file) & ('Ambient' not in file) & \
            (('A-0' in file) | ('A+0' in file)) & ('E+00' in file):
                # Load image into array
                img = Image.open(subdir_path + file)
                img_width, img_height = img.size
                img_resize = img.resize((int(resize_factor * img_width), int(resize_factor * img_height)), Image.ANTIALIAS)
                img_arr = np.array(img_resize)
                
                # Extract azimuth as concomitant variable
                azimuth = int(file[12:16])
                
                # Extract elevation as concomitant variable
                elevation = int(file[17:20])
                
                # Extract identity label
                label_id = int(file[5:7]) - 1
                
                # Extract shadow location label (left = 0, right = 1)
                if azimuth < 0.0:
                    label_shadow = 0
                elif azimuth > 0.0:
                    label_shadow = 1
                else:
                    label_shadow = 0.5
                
                # Append to lists
                X.append(img_arr.ravel())
                Y.append(np.array([azimuth]).ravel())
                labels_id.append(label_id)
                labels_shadow.append(label_shadow)
    
    # Convert to arrays
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    labels_id = np.array(labels_id)
    labels_shadow = np.array(labels_shadow)
    
    # Return arrays
    return X, Y, labels_id, labels_shadow

    
def save_arr_as_img(img_orig_arr, img_recon_arr, Y_test_scaled, scaler):
    img_orig_arr = np.reshape(img_orig_arr, newshape=(img_height, img_width))
    img_orig_arr = scaler.inverse_transform(img_orig_arr)
    img_orig_arr = np.clip(img_orig_arr, a_min=0.0, a_max=255.0)
    
    img_recon_arr = np.reshape(img_recon_arr, newshape=(img_height, img_width))
    img_recon_arr = scaler.inverse_transform(img_recon_arr)
    img_recon_arr = ref_img_adjust(img_orig_arr, img_recon_arr, ccmt=Y_test_scaled[[i], :],
                                   ccmt_thresh=0.5, copy=True)
    img_recon_arr = np.clip(img_recon_arr, a_min=0.0, a_max=255.0)
    
    img = Image.fromarray(img_recon_arr)
    img = img.resize((299, 299), Image.ANTIALIAS).convert('RGB')
    
    file = 'apca_recon_yale_face_test/' + 'train_recon_' + str(i) + '.png'
    img = img.save(file)

