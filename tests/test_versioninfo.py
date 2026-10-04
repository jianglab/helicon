import subprocess
import sys
import types

import pytest

from helicon import _versioninfo as vi


def _git_says(monkeypatch, stdout, returncode=0):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode, stdout, "")

    monkeypatch.setattr(vi.subprocess, "run", run)


class TestFromGit:
    def test_exactly_on_a_tag_is_the_plain_version(self, monkeypatch, tmp_path):
        _git_says(monkeypatch, "v2026.10-0-g43d8e7d\n")
        assert vi._from_git(tmp_path) == dict(
            version="2026.10", commit="43d8e7d", modified=False
        )

    def test_commits_after_a_tag_make_a_dev_version(self, monkeypatch, tmp_path):
        _git_says(monkeypatch, "v2026.10-3-g43d8e7d\n")
        info = vi._from_git(tmp_path)
        assert info["version"] == "2026.10.dev3"
        assert info["commit"] == "43d8e7d"

    def test_uncommitted_changes_are_flagged(self, monkeypatch, tmp_path):
        _git_says(monkeypatch, "v2026.10-0-g43d8e7d-dirty\n")
        assert vi._from_git(tmp_path)["modified"] is True

    def test_a_failing_git_gives_nothing(self, monkeypatch, tmp_path):
        _git_says(monkeypatch, "", returncode=128)
        assert vi._from_git(tmp_path) is None

    def test_a_missing_git_gives_nothing(self, monkeypatch, tmp_path):
        def run(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(vi.subprocess, "run", run)
        assert vi._from_git(tmp_path) is None


class TestFromBuild:
    def _built(self, monkeypatch, **values):
        fake = types.ModuleType("helicon._version")
        fake.__dict__.update(values)
        monkeypatch.setitem(sys.modules, "helicon._version", fake)
        monkeypatch.setattr(vi, "_version", fake, raising=False)

    def test_the_hash_is_shortened_without_its_g(self, monkeypatch):
        self._built(monkeypatch, version="2026.10", commit_id="gdf9d7ee2e")
        assert vi._from_build() == dict(
            version="2026.10", commit="df9d7ee", modified=False
        )

    def test_a_missing_hash_is_none(self, monkeypatch):
        self._built(monkeypatch, version="2026.10", commit_id=None)
        assert vi._from_build()["commit"] is None

    def test_no_build_file_gives_nothing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "helicon._version", None)
        assert vi._from_build() is None


class TestDescribe:
    def test_version_and_hash(self):
        info = dict(version="2026.10", commit="43d8e7d", modified=False)
        assert vi.describe(info) == "2026.10 (43d8e7d)"

    def test_modified_checkout(self):
        info = dict(version="2026.10.dev3", commit="43d8e7d", modified=True)
        assert vi.describe(info) == "2026.10.dev3 (43d8e7d, modified)"

    def test_version_only(self):
        info = dict(version="2026.10", commit=None, modified=False)
        assert vi.describe(info) == "2026.10"


class TestVersionInfo:
    def test_a_checkout_asks_git(self, monkeypatch):
        monkeypatch.setattr(
            vi, "_from_git", lambda root: dict(version="9", commit="a", modified=False)
        )
        monkeypatch.setattr(vi.Path, "exists", lambda self: True)
        assert vi.version_info()["version"] == "9"

    def test_with_nothing_known_the_version_is_a_placeholder(self, monkeypatch):
        monkeypatch.setattr(vi.Path, "exists", lambda self: False)
        monkeypatch.setattr(vi, "_from_build", lambda: None)
        assert vi.version_info()["version"] == vi.UNKNOWN

    def test_helicon_dunder_version_is_the_same(self):
        import helicon

        assert helicon.__version__ == vi.version_info()["version"]

    def test_the_command_line_prints_the_description(self, capsys, monkeypatch):
        from helicon import helicon as cli

        monkeypatch.setattr(sys, "argv", ["helicon", "--version"])
        with pytest.raises(SystemExit) as e:
            cli.main()
        assert e.value.code == 0
        assert capsys.readouterr().out.strip() == "helicon " + vi.describe()
