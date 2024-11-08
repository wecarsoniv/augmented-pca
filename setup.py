# ----------------------------------------------------------------------------------------------------------------------
# FILE DESCRIPTION
# ----------------------------------------------------------------------------------------------------------------------

# File:  setup.py
# Author:  Billy Carson
# Date written:  04-14-2021
# Last modified:  11-08-2024

"""
Description:  Setup Python file.
"""


# ----------------------------------------------------------------------------------------------------------------------
# IMPORT STATEMENTS
# ----------------------------------------------------------------------------------------------------------------------

# Import statements
import setuptools


# ----------------------------------------------------------------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------------------------------------------------------------

with open('README.md', 'r', encoding='utf-8') as fh:
    readme_description = fh.read()

setuptools.setup(
    name='augmented-pca',
    version='0.3.0',
    author='Billy Carson',
    author_email='williamcarsoniv@gmail.com',
    description='Python implementations of supervised and adversarial linear factor models.',
    long_description=readme_description,
    long_description_content_type='text/markdown',
    keywords=[
        'augmentedpca',
        'augmented principal component analysis',
        'augmented pca',
        'principal component analysis',
        'pca',
        'factor model',
        'factor models',
        'linear models',
        'autoencoder',
        'autoencoders',
        'supervised autoencoder',
        'supervised autoencoders',
        'SAE',
        'adversarial autoencoder',
        'adversarial autoencoders',
        'fair machine learning',
        'machine learning',
        'representation learning',
        'dimensionality reduction',
    ],
    url='https://github.com/wecarsoniv/augmented-pca',
    project_urls={
        'Issue Tracker': 'https://github.com/wecarsoniv/augmented-pca/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.11',
    install_requires=[
        'numpy',
        'scipy',
    ],
)

