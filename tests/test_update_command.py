import argparse
import json
import subprocess

import pytest

from helicon.commands import update
from helicon.lib.exceptions import HeliconError


def _args(**kw):
    ns = dict(check=False, to="", yes=True)
    ns.update(kw)
    return argparse.Namespace(**ns)


def _installed(monkeypatch, version, kind="pip", commit="abc1234"):
    import helicon._versioninfo as vi

    monkeypatch.setattr(
        vi, "version_info", lambda: dict(version=version, commit=commit, modified=False)
    )
    monkeypatch.setattr(update, "install_kind", lambda: kind)


class TestArguments:
    def test_the_options_are_wired(self):
        parser = argparse.ArgumentParser()
        update.add_args(parser)
        args = parser.parse_args(["--check", "--to", "v2026.10", "--yes"])
        assert (args.check, args.to, args.yes) == (True, "v2026.10", True)

    def test_the_defaults_change_nothing(self):
        parser = argparse.ArgumentParser()
        update.add_args(parser)
        args = parser.parse_args([])
        assert (args.check, args.to, args.yes) == (False, "", False)


class TestVersionOrder:
    def test_numbers_not_text_decide(self):
        assert update._version("v2026.10") > update._version("v2026.9")

    def test_a_leading_zero_is_the_same_release(self):
        assert update._version("2026.09") == update._version("v2026.9")

    def test_text_that_is_no_version_is_none(self):
        assert update._version("vnext") is None


class TestFindingTheNewestRelease:
    def _tags(self, monkeypatch, names):
        out = "".join(f"deadbeef\trefs/tags/{n}\n" for n in names)
        monkeypatch.setattr(update, "_run", lambda cmd, cwd=None: out)

    def test_the_numerically_newest_tag_wins(self, monkeypatch):
        self._tags(monkeypatch, ["v2026.9", "v2026.10", "v2025.04"])
        assert update.tags_on_github()[-1] == "v2026.10"

    def test_release_candidates_and_odd_tags_are_not_releases(self, monkeypatch):
        self._tags(monkeypatch, ["v2026.9", "v2026.10rc1", "v2026.11.dev1", "v-x"])
        assert update.tags_on_github() == ["v2026.9"]

    def test_pypi_is_asked_first(self, monkeypatch):
        monkeypatch.setattr(update, "latest_on_pypi", lambda: "2026.10")
        monkeypatch.setattr(
            update, "tags_on_github", lambda: pytest.fail("github not needed")
        )
        assert update.latest_release() == ("2026.10", "pypi")

    def test_github_is_the_fallback(self, monkeypatch):
        monkeypatch.setattr(update, "latest_on_pypi", lambda: None)
        monkeypatch.setattr(update, "tags_on_github", lambda: ["v2026.9", "v2026.10"])
        assert update.latest_release() == ("2026.10", "github")

    def test_no_tags_is_an_error(self, monkeypatch):
        monkeypatch.setattr(update, "latest_on_pypi", lambda: None)
        monkeypatch.setattr(update, "tags_on_github", lambda: [])
        with pytest.raises(HeliconError):
            update.latest_release()

    def _pypi(self, monkeypatch, info):
        class Reply:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def read(self, *a):
                return json.dumps({"info": info}).encode()

        monkeypatch.setattr(update.urllib.request, "urlopen", lambda *a, **k: Reply())

    def test_pypi_version_of_our_project(self, monkeypatch):
        self._pypi(
            monkeypatch,
            dict(
                version="2026.10",
                project_urls={"Source": "https://github.com/jianglab/helicon"},
            ),
        )
        assert update.latest_on_pypi() == "2026.10"

    def test_a_project_of_the_same_name_by_others_is_ignored(self, monkeypatch):
        self._pypi(
            monkeypatch,
            dict(version="9.9", project_urls={"Home": "https://example.org"}),
        )
        assert update.latest_on_pypi() is None

    def test_no_network_means_not_on_pypi(self, monkeypatch):
        def fail(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr(update.urllib.request, "urlopen", fail)
        assert update.latest_on_pypi() is None


class TestMain:
    def test_up_to_date(self, monkeypatch, capsys):
        _installed(monkeypatch, "2026.10")
        monkeypatch.setattr(update, "latest_release", lambda: ("2026.10", "pypi"))
        monkeypatch.setattr(
            update, "update_installed", lambda *a: pytest.fail("must not install")
        )
        update.main(_args())
        assert "up to date" in capsys.readouterr().out

    def test_commits_after_the_newest_release_are_not_behind_it(
        self, monkeypatch, capsys
    ):
        _installed(monkeypatch, "2026.10.dev3", kind="checkout")
        monkeypatch.setattr(update, "latest_release", lambda: ("2026.10", "github"))
        monkeypatch.setattr(
            update, "update_checkout", lambda *a: pytest.fail("must not touch")
        )
        update.main(_args())
        assert "past the newest release" in capsys.readouterr().out

    def test_check_only_reports(self, monkeypatch, capsys):
        _installed(monkeypatch, "2026.9")
        monkeypatch.setattr(update, "latest_release", lambda: ("2026.10", "pypi"))
        monkeypatch.setattr(
            update, "update_installed", lambda *a: pytest.fail("must not install")
        )
        update.main(_args(check=True))
        assert "2026.10" in capsys.readouterr().out

    def test_a_pypi_install_is_upgraded_with_pip(self, monkeypatch):
        _installed(monkeypatch, "2026.9")
        monkeypatch.setattr(update, "latest_release", lambda: ("2026.10", "pypi"))
        ran = []
        monkeypatch.setattr(update, "_run", lambda cmd, cwd=None: ran.append(cmd) or "")
        update.main(_args())
        assert ran[0][-1] == "helicon==2026.10"
        assert ran[0][1:5] == ["-m", "pip", "install", "--upgrade"]

    def test_a_release_only_on_github_is_installed_from_its_tag(self, monkeypatch):
        _installed(monkeypatch, "2026.9")
        monkeypatch.setattr(update, "latest_release", lambda: ("2026.10", "github"))
        ran = []
        monkeypatch.setattr(update, "_run", lambda cmd, cwd=None: ran.append(cmd) or "")
        update.main(_args())
        assert ran[0][-1] == f"git+{update.GITHUB_URL}@v2026.10"

    def test_a_checkout_goes_to_the_tag(self, monkeypatch):
        _installed(monkeypatch, "2026.9", kind="checkout")
        monkeypatch.setattr(update, "latest_release", lambda: ("2026.10", "github"))
        got = []
        monkeypatch.setattr(
            update, "update_checkout", lambda tag: got.append(tag) or "ok"
        )
        update.main(_args())
        assert got == ["v2026.10"]

    def test_a_named_release_is_used_even_if_older(self, monkeypatch):
        _installed(monkeypatch, "2026.10", kind="checkout")
        monkeypatch.setattr(update, "latest_on_pypi", lambda: None)
        got = []
        monkeypatch.setattr(
            update, "update_checkout", lambda tag: got.append(tag) or "ok"
        )
        update.main(_args(to="v2026.9"))
        assert got == ["v2026.9"]

    def test_a_bad_release_name_is_an_error(self, monkeypatch):
        _installed(monkeypatch, "2026.9")
        with pytest.raises(HeliconError):
            update.main(_args(to="latest"))

    def test_without_a_terminal_it_will_not_guess(self, monkeypatch):
        _installed(monkeypatch, "2026.9")
        monkeypatch.setattr(update, "latest_release", lambda: ("2026.10", "pypi"))
        monkeypatch.setattr(update.sys.stdin, "isatty", lambda: False, raising=False)
        with pytest.raises(HeliconError, match="--yes"):
            update.main(_args(yes=False))


def _git(cwd, *cmd):
    subprocess.run(["git", "-C", str(cwd), *cmd], check=True, capture_output=True)


class TestCheckoutUpdate:
    @pytest.fixture
    def repos(self, tmp_path, monkeypatch):
        """A 'public' repository with two tags, and a checkout at the first."""
        public = tmp_path / "public"
        public.mkdir()
        env = ["-c", "user.name=t", "-c", "user.email=t@t"]
        _git(public, "init", "-q", "-b", "main")
        (public / "f.txt").write_text("1")
        _git(public, "add", ".")
        _git(public, *env, "commit", "-qm", "one")
        _git(public, "tag", "v1.0")
        checkout = tmp_path / "checkout"
        subprocess.run(["git", "clone", "-q", str(public), str(checkout)], check=True)
        (public / "f.txt").write_text("2")
        _git(public, *env, "commit", "-qam", "two")
        _git(public, "tag", "v2.0")
        monkeypatch.setattr(update, "GITHUB_URL", str(public))
        monkeypatch.setattr(update, "Path", lambda f: _FakeFile(checkout))
        return public, checkout

    def test_a_branch_moves_forward_to_the_tag(self, repos):
        public, checkout = repos
        msg = update.update_checkout("v2.0")
        assert "v2.0" in msg
        assert (checkout / "f.txt").read_text() == "2"
        branch = subprocess.run(
            ["git", "-C", str(checkout), "branch", "--show-current"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert branch == "main"  # still on its branch

    def test_uncommitted_changes_stop_it(self, repos):
        public, checkout = repos
        (checkout / "f.txt").write_text("mine")
        with pytest.raises(HeliconError, match="uncommitted"):
            update.update_checkout("v2.0")
        assert (checkout / "f.txt").read_text() == "mine"

    def test_a_checkout_that_has_the_tag_is_left_alone(self, repos):
        public, checkout = repos
        update.update_checkout("v2.0")
        assert "already contains" in update.update_checkout("v2.0")

    def test_a_detached_checkout_moves_to_the_tag(self, repos):
        public, checkout = repos
        _git(checkout, "checkout", "-q", "v1.0")
        update.update_checkout("v2.0")
        assert (checkout / "f.txt").read_text() == "2"

    def test_an_unknown_tag_is_an_error(self, repos):
        with pytest.raises(HeliconError):
            update.update_checkout("v9.9")


class _FakeFile:
    """Stands in for Path(__file__) so that the checkout is a temporary one."""

    def __init__(self, root):
        self._root = root

    def resolve(self):
        return self

    @property
    def parents(self):
        return [None, None, None, self._root]
