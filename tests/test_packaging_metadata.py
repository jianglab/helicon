import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _project():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["project"]


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
