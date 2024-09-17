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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Doxygen Generate --------------------------------------------------------

import os
import sys
import subprocess
import shutil
import textwrap

from recommonmark.parser import CommonMarkParser

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'exhale',
    'recommonmark'
]

# -- Project information -----------------------------------------------------

project = 'Leap Accelerate'
copyright = ' ICRAR/UWA - SKA Organization'
author = 'Callan Gray'

# The full version, including alpha/beta/rc tags
with open('../../version.txt') as f:
    version = f.read().strip()
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# path relative to conf.py
breathe_projects = {}

source_dir = "../../src"
#doxygen_xml = ""

# -- ReadTheDocs ------------------------------

def configureDoxyfile(input_dir: str, output_dir: str):
    with open('../Doxyfile.in', 'r') as file:
        file_data = file.read()

    file_data = file_data.replace('@DOXYGEN_INPUT_DIR@', input_dir)
    file_data = file_data.replace('@DOXYGEN_OUTPUT_DIR@', output_dir)

    with open('../Doxyfile', 'w') as file:
        file.write(file_data)

# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'
if read_the_docs_build:
    # build doxygen in docs folder
    input_dir = '../src'
    output_dir = 'build/doxygen'
    configureDoxyfile(input_dir, output_dir)
    subprocess.call('mkdir -p ' + output_dir, cwd="..", shell=True)
    subprocess.call('doxygen', cwd="..", shell=True)
    breathe_projects['LeapAccelerate'] = '../' + output_dir + '/xml'
    # doxygen_xml = '../' + output_dir + '/xml'

source_suffix = [".rst", '.md']

source_parsers = {".md": CommonMarkParser }

# Automatically generate autodoc_doxygen targets
autodoc_default_flags = ['members']

# Automatically generate stub pages
# autosummary_generate = True

cpp_id_attributes = ["__host__", "__device__", "EIGEN_DEVICE_FUNC"]

# Breathe Config

breathe_default_project = "LeapAccelerate"
breathe_default_members = ("members", "undoc-members")
breathe_separate_member_pages = True


breathe_projects_source = {
    "LeapAccelerate": (source_dir, [])
}

# breathe_doxygen_config_options = { }

breathe_domain_by_extension = {
    "h": "cpp",
    "cc": "cpp",
    "cu": "cpp"
}

# Exhale Config

exhale_args = {
    "containmentFolder":     "./api",
    "rootFileName":          "library_root.rst",
    "rootFileTitle":         "Leap Accelerate API Reference",
    "afterTitleDescription": (
        ".. note::"
        ""
        "The following documentation presents the C++ API."
    ),
    "doxygenStripFromPath": ".",

    # Suggested optional arguments
    "createTreeView":        True,
    "minifyTreeView":        False,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    #"unabridgedOrphanKinds": { "file", "namespace" },
    "exhaleExecutesDoxygen": False,
    #"exhaleDoxygenStdin":    "INPUT = ../../src",
    "lexerMapping": {
        r".*\.h": "cpp",
        r".*\.cc": "cpp",
        r".*\.cuh": "cuda",
        r".*\.cu": "cuda",
        r".*\.txt": "cmake"
    },
    "verboseBuild": True,
    "generateBreatheFileDirectives": False
}

# primary_domain = 'cpp'

# higligh_language = 'cpp'

# Extra Config

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
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']