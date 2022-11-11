# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from __future__ import unicode_literals

import os
import sys
sys.path.insert(0, os.path.abspath('...'))

# -- Project information -----------------------------------------------------

project = 'Metrics Reloaded'
year = '2022'
author = 'Carole Sudre'
copyright = '{0}, {1}'.format(year, author)

# The full version, including alpha/beta/rc tags
version = release = '0.1.0'

# -- General configuration ---------------------------------------------------
autodoc_mock_imports = [
    'numpy',
    'scipy',
    'pandas',
    'sklearn',
    'skimage',
    'nibabel',
]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx'
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_typehints = 'signature'
autodoc_docstring_signature = True
autoclass_content = 'both'
autodoc_member_order = 'bysource'

html_static_path = ["_static"]
templates_path = ['_templates']
extlinks = {
    'issue': ('https://github.com/csudre/MetricsReloaded/issues/%s', '#'),
    'pr': ('https://github.com/csudre/MetricsReloaded/pull/%s', 'PR #'),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'sphinx_rtd_theme'

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
   '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
