import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _pyproject():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


def _project():
    return _pyproject()["project"]


def _names(requirements):
    """The lower-cased project names of requirement strings (no extras or versions)."""
    return {re.split(r"[\[<>=!~; ]", r, maxsplit=1)[0].lower() for r in requirements}


class TestPyprojectMetadata:
    def test_every_classifier_is_one_pypi_accepts(self):
        # PyPI rejects the whole upload (400) for a classifier it does not know
        valid = pytest.importorskip("trove_classifiers").classifiers
        bad = [c for c in _project()["classifiers"] if c not in valid]
        assert bad == []

    def test_the_license_is_an_spdx_expression(self):
        assert _project()["license"] == "MIT"

    def test_the_version_comes_from_the_git_tag(self):
        project = _project()
        assert "version" in project["dynamic"]
        assert "version" not in project

    def test_requests_is_a_runtime_dependency(self):
        # lib/path_utils.py imports requests in get_file_size() and
        # download_file_from_url(), which core code reaches (the EMDB fetch of
        # lib/dataset.py), so a plain `pip install helicon` must bring it in.
        assert "requests" in _names(_project()["dependencies"])

    def test_the_test_extra_has_what_the_suite_asks_for(self):
        # trove-classifiers so the classifier test above really runs instead of
        # importorskip'ing itself everywhere; pytest-xdist for the documented
        # everyday run `pytest tests -n auto`.
        test_extra = _pyproject()["project"]["optional-dependencies"]["test"]
        assert {"trove-classifiers", "pytest-xdist"} <= _names(test_extra)
