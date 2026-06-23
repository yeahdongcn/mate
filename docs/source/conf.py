# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Allow API docs to build on hosts without visible MUSA devices.
os.environ.setdefault("MATE_MUSA_ARCH_LIST", "3.1")

import mate  # noqa: E402,F401
from packaging.version import Version  # noqa: E402

project = "MATE"
copyright = "2020-2026, MooreThreads GPU Computing Team"
author = "MooreThreads GPU Computing Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]
autodoc_mock_imports = ["tvm", "tvm_ffi", "tilelang"]

templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

version = Version(mate.__version__).base_version
release = Version(mate.__version__).base_version
html_title = f"{project} {release}"
html_short_title = html_title


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"] if (Path(__file__).parent / "_static").exists() else []
html_css_files = ["custom.css"] if html_static_path else []
html_js_files = (
    [
        "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js",
        "mermaid-init.js",
    ]
    if html_static_path
    else []
)
