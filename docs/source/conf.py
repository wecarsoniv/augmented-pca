# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Import libraries
import os
import sys

# Import AugmentedPCA
sys.path.insert(0, os.path.abspath('../../src'))
import apca


# -- Import theme ------------------------------------------------------------

# Import modules and theme
import datetime
import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

# Project information
project = 'AugmentedPCA'
author = 'Billy Carson'
year = str(datetime.datetime.now().year)
copyright = year + ', ' + author

# The full version, including alpha/beta/rc tags
release = '0.1.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = [
#     'sphinx_rtd_theme',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.napoleon',
# ]
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
]

# Turn on auto-summary
autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_include_init_with_doc = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version': True,
    'logo_only': True,
}

# Side bar icon
# html_favicon = '_static/img/apca_logo.ico'
html_favicon = '_static/img/apca_logo.svg'
html_logo = '_static/img/apca_logo_full.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# HTML context
# html_context = {
#   'display_github': True,
#   'github_user': 'wecarsoniv',
#   'github_repo': 'augmented-pca',
#   'github_version': ''
# }

# Custom CSS file location
html_css_files = [
    'css/custom.css',
]

