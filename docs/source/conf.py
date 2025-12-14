# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from unittest.mock import MagicMock


sys.path.insert(0, os.path.abspath('../../src'))


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return Mock()


MOCK_MODULES = ['torch', 'numpy', 'cachetools', 'pyreflex']
for mod_name in MOCK_MODULES:
    mock_module = Mock()
    mock_module.__name__ = mod_name
    mock_module.__all__ = []
    sys.modules[mod_name] = mock_module


sys.modules['torch'].__version__ = '2.1.0'


project = 'TorchFlint'
copyright = '2025, Caewinix'
author = 'Caewinix'
release = '0.0.1b16'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ["torch", "numpy", "cachetools", "pyreflex"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
