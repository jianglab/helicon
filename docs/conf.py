"""Sphinx configuration of the Helicon user documentation (built by Read the Docs)."""

from importlib.metadata import PackageNotFoundError, version as _version

project = "Helicon"
author = "Wen Jiang, Daoyi Li, Xiaoqi Zhang, and Helicon contributors"
copyright = "Wen Jiang, Daoyi Li, Xiaoqi Zhang, and Helicon contributors"
try:
    release = _version("helicon")
except PackageNotFoundError:  # built without the package installed
    release = ""
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "tasklist"]
myst_heading_anchors = 3

# DEVELOPMENT.md is for contributors and links to files of the repository; the folder of the
# manuscript is never published (and is not in the repository)
exclude_patterns = [
    "_build",
    "DEVELOPMENT.md",
    "manuscript",
    "tools",
    "requirements.txt",
]

html_theme = "furo"
html_title = "Helicon"
html_static_path = []
html_theme_options = {
    "source_repository": "https://github.com/jianglab/helicon/",
    "source_branch": "main",
    "source_directory": "docs/",
}
