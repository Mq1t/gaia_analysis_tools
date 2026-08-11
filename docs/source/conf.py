import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Gaia Analysis Tools'
copyright = '2026, Matthew McGuire, Jennifer Farrell, Nakia McKay'
author = 'Matthew McGuire, Jennifer Farrell, Nakia McKay'
release = '2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # Google/NumPy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",              # Markdown support
]

templates_path = ['_templates']
exclude_patterns = []
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
