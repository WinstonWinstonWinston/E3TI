# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
from e3ti import __version__
sys.path.insert(0, os.path.abspath("../"))

# docs/conf.py (top)
import sys, pathlib, importlib.util
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root
assert importlib.util.find_spec("e3ti"), "Sphinx can't import 'e3ti' from repo root"
autodoc_mock_imports = ["torch", "torch_geometric", "e3nn", "wandb"]


# -- Project information -----------------------------------------------------

project = "e3ti"
author = "Winston"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autodocsumm",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "jupyter_sphinx",
]

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": True, "inherited-members": True}
autodoc_typehints = "description"


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

autodoc_default_options = {
    "inherited-members": False,
    "show-inheritance": True,
    "autosummary": False,
}

# The reST default role to use for all documents.
default_role = "any"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

myst_update_mathjax = False


# Resolve function for the linkcode extension.
# Thanks to https://github.com/Lasagne/Lasagne/blob/master/docs/conf.py
def linkcode_resolve(domain, info):
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(__file__)) # type: ignore
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None

    try:
        rel_path, line_start, line_end = find_source()
        # __file__ is imported from e3nn
        filename = f"e3nn/{rel_path}#L{line_start}-L{line_end}"
    except Exception:
        # no need to be relative to core here as module includes full path.
        filename = info["module"].replace(".", "/") + ".py"

    tag = __version__
    return f"https://github.com/e3nn/e3nn/blob/{tag}/{filename}"