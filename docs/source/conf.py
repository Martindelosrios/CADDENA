# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'BATMAN'
copyright = '2024'
author = 'Martín de los Rios'

# The short X.Y version
version = '1.0'
# The full version, including alpha/beta/rc tags
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    # Add more extensions as needed
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further. For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # Set any theme options here
}

# -- Options for other output formats -----------------------------------------

# Add options for other output formats here, like PDF output.

# -- Extension configuration -------------------------------------------------

# Add any configuration options for extensions here.

# -- Options for LaTeX output ------------------------------------------------

# Add options for LaTeX output here.

# -- Options for manual page output -------------------------------------------

# Add options for manual page output here.

# -- Options for Texinfo output ----------------------------------------------

# Add options for Texinfo output here.

# -- Options for Epub output -------------------------------------------------

# Add options for Epub output here.

