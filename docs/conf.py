"""Sphinx configuration for hofmann documentation."""

project = "hofmann"
copyright = "2025, hofmann contributors"
author = "hofmann contributors"
release = "0.13.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_autodoc_typehints",
]

# -- General configuration ---------------------------------------------------

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Use Google-style docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# sphinx-autodoc-typehints settings
always_document_param_types = True
typehints_defaults = "braces"

# -- Plot directive settings --------------------------------------------------

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("svg", 150)]

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
}

# -- Auto-generate documentation figures -------------------------------------

import os
import sys


def _generate_figures(app):
    """Generate documentation figures before Sphinx reads source files."""
    if os.environ.get("SKIP_IMAGE_GEN"):
        return
    static_dir = os.path.join(os.path.dirname(__file__), "_static")
    sys.path.insert(0, static_dir)
    try:
        from generate_images import generate_docs_images

        generate_docs_images()
    finally:
        sys.path.pop(0)


def setup(app):
    app.connect("builder-inited", _generate_figures)
