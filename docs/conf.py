# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import datetime
import sys
from pathlib import Path

import nectarchain

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

# sys.path.insert(0, os.path.abspath("../src"))

pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
pyproject = tomllib.loads(pyproject_path.read_text())

project = pyproject["project"]["name"]
author = pyproject["project"]["authors"][0]["name"]
copyright = "{}.  Last updated {}".format(
    author, datetime.datetime.now().strftime("%d %b %Y %H:%M")
)
python_requires = pyproject["project"]["requires-python"]

# make some variables available to each page
rst_epilog = f"""
.. |python_requires| replace:: {python_requires}
"""

version = nectarchain.__version__
# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "numpydoc",
]

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
autosummary_generate = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

templates_path = []  # ["_templates"]
exclude_patterns = ["_build"]

# intersphinx allows referencing other packages sphinx docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "astropy": ("https://docs.astropy.org/en/latest/", None),
    "ctapipe": ("https://ctapipe.readthedocs.io/en/v0.19.2/", None),
}

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# html_theme = "alabaster"

html_logo = "_static/nectarcam.png"
html_theme_options = {
    "navigation_with_keys": False,
    "github_url": f"https://github.com/cta-observatory/{project}",
    "navbar_start": ["navbar-logo", "version-switcher"],
    "announcement": """
        <p>nectarchain is not stable yet, so expect large and rapid
        changes to structure and functionality as we explore various
        design choices before the 1.0 release.</p>
    """,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_context = {
    "default_mode": "light",
    "github_user": "cta-observatory",
    "github_repo": f"{project}",
    "github_version": "main",
    "doc_path": "docs",
}
html_file_suffix = ".html"

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = f"{project} v{release}"

# Output file base name for HTML help builder.
htmlhelp_basename = f"{project}doc"
