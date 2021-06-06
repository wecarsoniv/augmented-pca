:github_url: https://github.com/wecarsoniv/augmented-pca

.. role:: python(code)
   :language: python


Examples
========================================================================================================================

Here, usage and efficacy of AugmentedPCA models is demonstrated on real world, open-source datasets.


sAPCA Example - 2-Dimensional Cancer Gene Expression Clustering
---------------------------------------------------------------

The ability of sAPCA to create representations with greater class fidelity is demonstrated using a 
`gene expression dataset from the UCI machine learning repository <https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq>`_. 
This dataset contains RNA-Seq gene expression samples from patients with five different typesof tumors. Dimensionality 
reduction techniques, such as PCA, are often used in gene expression analysis to visualize clustering of samples in 
2-dimensional(2D) space or as a preprocessing step for downstream classification. However, sometimes principal axes of 
variance may represent patient-specific gene expression variance rather than variance specific to condition or disease. 
Here, sAPCA is used to create representations that, in addition to representing the variance in the gene expression 
data, are aligned with the data labels.

First, Python functions, modules, and libraries used in this example are imported.

.. code-block:: python

    # Import functions, modules, and libraries
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, scale, StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt

Next, AugmentedPCA factor models are imported from the :python:`apca.models` module.

.. code-block:: python

    # Import AugmentedPCA models
    from apca.models import *



aAPCA Example - Removal of Nuisance Variables in Image Data
------------------------------------------------------------------------------------------------------------------------
    
The ability of aAPCA to create representations invariant to concomitant data or nuisance variables is demonstrated 
using images from the `Extended Yale Face Database B <http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html>`_. 
This dataset contains facial images of 38 human subjects taken with the light source at varying angles of azimuth and 
elevation, resulting in shadows cast across subject faces. Here, the nuisance variable is the variable lighting angles 
resulting in shadows that obscure parts of the image, and by extension features of subject identity. Here, aAPCA is 
used to create representations that, in addition to representing the variance in the image data, are invariant to this 
shadow nuisance variable.

First, Python functions, modules, and libraries used in this example are imported.

.. code-block:: python

    # Import functions, modules, and libraries
    import os
    import numpy as np
    import time
    import PIL
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, mean_squared_error, roc_auc_score
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

Next, AugmentedPCA factor models are imported from the :python:`apca.models` module.

.. code-block:: python

    # Import all APCA models
    from apca.models import *
