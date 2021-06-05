:github_url: https://github.com/wecarsoniv/augmented-pca 

.. role:: python(code)
   :language: python


Introduction
============

This library provides Python implementation of Augmented Principal Component Analysis (Augmented PCA or APCA) - a family of linear factor models that find a set of factors aligned with an *augmenting objective* in addition to the canonical PCA objective of finding factors that represent the data variance. APCA can be split into two general families of models: adversarial APCA and supervised APCA.


Models Overview
---------------

APCA has two main variants: adversarial APCA (aAPCA) and supervised APCA (sAPCA). A brief introduction to the two variants is given in the following sections.


Supervised APCA
~~~~~~~~~~~~~~~

In supervised APCA (sAPCA), the augmenting objective is to make the factors *aligned* with the data labels, or some outcome, in addition to having the factors explain the variance of the original observed or primary data. Below is a diagram depicting the relationship between primary data, supervision data, and the resulting sAPCA factors.

.. image:: ../_static/img/sapca_diagram.png
    :alt: sAPCA diagram


Adversarial APCA
~~~~~~~~~~~~~~~~

In adversarial APCA (aAPCA), the augmenting objective is to make the factors *orthogonal* to a set of concomitant data, in addition to having the factors explain the variance of the original observed or primary data. Below is a diagram depicting the relationship between primary data, concomitant data, and the resulting aAPCA factors.

.. image:: ../_static/img/aapca_diagram.png
    :alt: aAPCA diagram


Quick Introduction
------------------

Below is quick guide to using APCA. For a more in-depth examples, see the Examples section of the documentation.


Importing APCA Models
~~~~~~~~~~~~~~~~~~~~~

APCA models can be imported from the :python:`models.py` module. Below we show an example of importing the aAPCA model.

.. code-block:: python

    # Import all APCA models
    from apca.models import aAPCA
    

Alternatively, all offered APCA models can be imported at once.

.. code-block:: python

    # Import all APCA models
    from apca.models import *
    


Instantiating APCA
~~~~~~~~~~~~~~~~~~

APCA models are instantiated by assigning either an aAPCA or sAPCA object to a variable. During instantiation, one has the option to define parameters :python:`n_components`, :python:`mu`, which represent the number of components and the augmenting objective strength, respectively. Additionally, approximate inference strategy can be defined through the :python:`inference` parameter.

.. code-block:: python

    # Define model parameters
    n_components = 2        # factors will have dimensionality of 2
    mu = 1.0                # augmenting objective strength equal to 1 
    inference = 'encoded'   # encoded approximate inference strategy
    
    # Instantiate adversarial APCA model
    aapca = aAPCA(n_components=n_components, mu=mu, inference=inference)
    


Fitting APCA
~~~~~~~~~~~~

APCA models closely follow the style and implemention of `scikit-learn's PCA implementation <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_, with many of the same methods and functionality. Similar to scikit-learn models, APCA models are fit using the :python:`fit()` method. :python:`fit()` takes two parameters: :python:`X` which represents the matrix of primary data and :python:`Y` which represents the matrix of augmenting data.

.. note::
    Before fitting APCA models, it may be helpful to scale both the primary and augmenting data. Having the primary and augmenting data on the same scale will result in more consistent range of effective augmenting objective strengths (controlled by the :python:`mu` paramter) across different datasets.

.. code-block:: python

    # Import numpy
    import numpy as np
    
    # Generate synthetic data
    # Note: primary and augmenting data must have same number of samples/same first dimension size
    n_samp = 100
    X = np.random.randn(n_samp, 20)   # primary data, 100 samples with dimensionality of 20
    Y = np.random.randn(n_samp, 3)    # concomitant data, 100 samples with dimensionality of 3
    
    # Fit adversarial APCA instance
    aapca.fit(X=X, Y=Y)
    

Alternatively, APCA models can be fit using the :python:`fit_transform()` method, which takes the same parameters as the :python:`fit()` method but also returns a matrix of components or factors.

.. code-block:: python

    # Fit adversarial APCA instance and generate components
    S = aapca.fit_transform(X=X, Y=Y)
    

