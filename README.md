# Augmented Principal Component Analysis


## Overview

This library provides Python implementation of Augmented Principal Component Analysis (Augmented PCA or APCA) - a family of factor models that fit data according to an augmenting objective in addition to the canonical PCA objective of finding factors that maximize the explained data variance. APCA can be split into two general families of models: adversarial APCA and supervised APCA.


### Adversarial APCA

In adversarial APCA (aAPCA), the augmenting objective is to make the factors *orthogonal* to a set of concomitant data.


### Supervised APCA

In supervised APCA (sAPCA), the augmenting objective is to make the factors *predictive* of a label, condition, or outcome.


## Documentation

Documentation for APCA is available on this [documentation site]().

Provided documentation includes:

* Motivation - Motivation behind the APCA model and the different approximate inference strategies.

* Model formulation - Overview of different models and approximate inference strategies as well as more in-depth mathematical descriptions.

* Tutorials - Step-by-step guide on how to use the different offered APCA model.

* Examples - Use case examples for the different models.


## Dependencies

The APCA library is written in [Python](https://www.python.org/), and requires Python >= 3.6 to run. APCA relies on the following libraries and version numbers:

* Python >= 3.6
* NumPy >= 1.19.2
* SciPy >= 1.5.2


## Installation

To install the latest stable release, use [pip](https://pip.pypa.io/en/stable/reference/pip_install/). Use the following command to install:

    $ pip install augmented-pca


## Issue Tracking and Reports

Please use the [Github issue tracker](https://github.com/wecarsoniv/augmented-pca/issues) associated with the APCA repository for issue tracking, filing bug reports, and asking general questions about the library or project.


## Quick Introduction

### Importing APCA Models

APCA models can be imported by importing the `models.py` module or by importing the models themselves from the `models.py` module.

    < screenshot here >


### Instantiating an APCA Model

APCA models are instantiated by defining the model parameters `n_components`, `mu`, which represent the number of components and the augmenting objective strength, respectively. Additionally, the approximate inference strategy can be defined through the `inference` parameter.

    < screenshot here >


### Approximate Inference Strategies

In this section, we give a brief overview of the different approximate inference strategies offered for APCA. Inference strategy should be chosen based on the data on which the APCA model will be used as well as the specific use case. Both aAPCA and sAPCA models use the jointly-encoded approximate inference strategy by default.

#### Local

#### Encoded

#### Jointly-Encoded




## Citation

Plase cite our paper if you find this library helpful in your research:

    @article{carson_augmentedpca,
    title={Augmented Principal Component Analysis},
    author={{Carson IV}, William E. and Talbot, Austin and Carlson, David},
    journal={},
    year={2021}}


## Funding

This project was supported by the NIH BRAIN Initiative, award number R01 EB026937.

