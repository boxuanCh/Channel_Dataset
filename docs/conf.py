"""Sphinx configuration for the Channel Dataset documentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the source tree importable for local builds that do not install the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# NumPy is a required dependency. Importing it before an optional PyTorch install
# also avoids native-runtime initialization conflicts in some Conda environments.
import numpy  # noqa: F401

from channel_dataset import __version__

project = "Channel Dataset"
author = "Channel Dataset contributors"
copyright = "2026, Channel Dataset contributors"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autoclass_content = "class"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_preprocess_types = True

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]

# A NumPy-only environment can still build the complete API reference. Treat a
# binary-loader failure like a missing optional dependency; ``find_spec`` alone
# is insufficient when system policy or a missing runtime blocks Torch DLLs.
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    autodoc_mock_imports = ["torch"]
    suppress_warnings = ["autodoc.mocked_object"]

html_theme = "pydata_sphinx_theme"
html_title = "Channel Dataset"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True
html_theme_options = {
    "collapse_navigation": False,
    "header_links_before_dropdown": 6,
    "navigation_depth": 4,
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "show_toc_level": 2,
}

pygments_style = "sphinx"
pygments_dark_style = "monokai"
