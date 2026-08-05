"""Sphinx configuration for pypricing docs."""

from __future__ import annotations

import sys
from pathlib import Path

# -- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pypricing  # noqa: E402

# -- Project ----------------------------------------------------------------
project = "pypricing"
author = "pypricing contributors"
copyright = f"%Y, {author}"
release = pypricing.__version__
version = release
master_doc = "index"

# -- Extensions -------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = [
    "build",
    "jupyter_execute",
    "jupyter_cache",
    "**.ipynb_checkpoints",
]

# -- Autodoc / autosummary --------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- MyST-NB (don't re-run notebooks during the build) ----------------------
nb_execution_mode = "off"
nb_kernel_rgx_aliases = {".*": "python3"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]
myst_heading_anchors = 2
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- Intersphinx ------------------------------------------------------------
intersphinx_mapping = {
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# -- HTML -------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} {release}"
