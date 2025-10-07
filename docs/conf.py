# -- Path setup --------------------------------------------------------------
import sys, pathlib, importlib.util
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root

# Optional: avoid heavy deps blocking import
autodoc_mock_imports = ["torch", "torch_geometric", "e3nn", "wandb"]

# Safe sanity check (optional)
assert importlib.util.find_spec("e3ti"), "Sphinx can't import 'e3ti' from repo root"

# -- Project information -----------------------------------------------------
project = "e3ti"
author = "Winston"
from e3ti import __version__ as release  # noqa: E402

# -- Extensions --------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "autoapi.extension",           # << AutoAPI
]

# AutoAPI config (scans code every build; no stub tiles)
import pathlib as _p
autoapi_type = "python"
autoapi_dirs = [str(_p.Path(__file__).resolve().parents[1] / "e3ti")]
autoapi_add_toctree_entry = False        # we place it in the sidebar ourselves
autoapi_python_class_content = "both"    # class doc + __init__ doc
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "inherited-members",
    "special-members",
    "__init__",
]
autoapi_keep_files = False
autoapi_ignore = ["*/tests/*", "*/_*/**", "*/__main__.py"]

# Typing / docstring rendering
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_class_signature = "separated"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}


# -- HTML --------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
myst_update_mathjax = False
html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
default_role = "any"

# -- linkcode (points to YOUR repo) -----------------------------------------
GITHUB_USER = "winstonwinstonwinston"
GITHUB_REPO = "E3TI"
GITHUB_BRANCH = release if isinstance(release, str) else "main"

def linkcode_resolve(domain, info):
    if domain != "py" or not info.get("module"):
        return None
    try:
        import inspect, os, importlib
        mod = importlib.import_module(info["module"])
        obj = mod
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        if not fn:
            return None
        source, lineno = inspect.getsourcelines(obj)
        rel = os.path.relpath(fn, start=str(_p.Path(__file__).resolve().parents[1]))
        return f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{rel}#L{lineno}-L{lineno+len(source)-1}"
    except Exception:
        return None
