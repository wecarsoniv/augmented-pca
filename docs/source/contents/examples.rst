:github_url: https://github.com/wecarsoniv/augmented-pca

.. role:: python(code)
   :language: python


Examples
========================================================================================================================

Here, usage and efficacy of AugmentedPCA models is demonstrated on real world, open-source datasets.


sAPCA Example - Gene Expression Clustering
------------------------------------------------------------------------------------------------------------------------

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
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt

Next, AugmentedPCA factor models are imported from the :python:`apca.models` module.

.. code-block:: python

    # Import AugmentedPCA models
    from apca.models import *
    

Gene expression data is loaded and formatted into a matrix :python:`X`, where each row represents a different tumor 
gene expression sample and each column represents a different gene. 

.. code-block:: python

    # Display data dimensionality
    print('Cancer gene expression dataset dimensions:\n')
    print('  Gene expression data:  (%d, %d)' % (X.shape))
    print('  Supervision data:  (%d, %d)' % (Y.shape))
    print('  Labels:  (%d,)' % (y.shape))
    
    >>>   Gene expression data:  (801, 20531)
    >>>   Supervision data:  (801, 5)
    >>>   Labels:  (801,)

.. code-block:: python

    # Subset of original data
    X_subset = X[:, :2000]

    # Split data
    X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(X_subset, Y, y, test_size=0.5,
                                                                         shuffle=True, random_state=random_state)
    

.. code-block:: python

    # Instantiate standard scaler
    scaler = StandardScaler()

    # Scale gene expression data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

.. code-block:: python

    # Instantiate logistic regression model
    model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=10000, multi_class='auto', random_state=0)
    

.. code-block:: python

    # PCA decomposition
    n_components = 2
    pca = PCA(n_components=n_components)
    S_train = pca.fit_transform(X_train)
    S_test = pca.transform(X_test)

    # Fit model to training data
    model.fit(S_train, y_train)

    # Get model predictions
    y_pred_train = model.predict(S_train)
    y_pred_test = model.predict(S_test)
    train_acc = accuracy_score(y_pred_train, y_train)
    test_acc = accuracy_score(y_pred_test, y_test)

    # Model prediction accuracy
    print('Model performance using PCA components (# components = %d):' % (n_components))
    print('  Train set:  %.4f' % (train_acc))
    print('  Test set:  %.4f' % (test_acc))
    
    >>> Model performance using PCA components (# components = 2):
    >>>   Train set:  0.7300
    >>>   Test set:  0.7132

    # Plot PCA components of samples in 2D space
    color_list = ['deeppink', 'dodgerblue', 'lightseagreen', 'darkorange', 'mediumorchid']
    marker_list = ['*', 'o', 's', '^', 'D']
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6.0, 4.5))
    for i, label in enumerate(list(np.unique(y_test))):
        ax1.scatter(S_test[np.where(y_test==label), 0], S_test[np.where(y_test==label), 1],
                    c=color_list[i], marker=marker_list[i], alpha=0.5, label=class_dict[i])
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.grid(alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(loc='lower right')
    plt.show()
    

.. code-block:: python

    # Number of sAPCA components
    n_components = 2

    # List of supervision strength values
    mu_lo = 0.0
    mu_hi = 5000
    mu_step = 100.0
    mu_list = list(np.arange(mu_lo, mu_hi + mu_step, mu_step))

    # Initialize test accuracy list
    train_acc_list = []
    test_acc_list = []

    # Iterate over supervision strengths
    for mu in mu_list:
        # PCA decomposition
        apca = sAPCA(n_components=2, mu=mu, inference='encoded')
        S_train = apca.fit_transform(X=X_train, Y=Y_train)
        S_test = apca.transform(X=X_test, Y=None)

        # Fit model to training data
        model.fit(S_train, y_train)

        # Predict on training data
        y_pred_train = model.predict(S_train)
        train_acc = accuracy_score(y_pred_train, y_train)
        train_acc_list.append(train_acc)

        # Predict on test data
        y_pred_test = model.predict(S_test)
        test_acc = accuracy_score(y_pred_test, y_test)
        test_acc_list.append(test_acc)

    # Model prediction accuracy
    print('Max model performance using sAPCA components (# components = %d):' % (n_components))
    print('  Train set:  %.4f' % (np.max(train_acc_list)))
    print('  Test set:  %.4f' % (np.max(test_acc_list)))
    
    >>> Max model performance using sAPCA components (# components = 2):
    >>>   Train set:  1.0000
    >>>   Test set:  0.9027
    

.. code-block:: python

    # sAPCA decomposition
    apca = sAPCA(n_components=2, mu=2500, inference='encoded')
    S_train = apca.fit_transform(X=X_train, Y=Y_train)
    S_test = apca.transform(X=X_test, Y=None)

    # Plot PCA components of samples in 2D space
    color_list = ['deeppink', 'dodgerblue', 'lightseagreen', 'darkorange', 'mediumorchid']
    marker_list = ['*', 'o', 's', '^', 'D']
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(6.0, 4.5))
    for i, label in enumerate(list(np.unique(y_test))):
        ax3.scatter(S_test[np.where(y_test==label), 0], S_test[np.where(y_test==label), 1],
                    c=color_list[i], marker=marker_list[i], alpha=0.5, label=class_dict[i])
    ax3.set_xlabel('sAPCA Component 1')
    ax3.set_ylabel('sAPCA Component 2')
    ax3.grid(alpha=0.3)
    ax3.set_axisbelow(True)
    ax3.legend(loc='lower left')
    plt.show()



aAPCA Example - Removal of Image Nuisance
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
