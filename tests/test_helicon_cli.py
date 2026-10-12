"""Tests for the ``helicon`` CLI entrypoint (helicon.py).

Covers the bare-``helicon`` default: with a graphical display it launches
the web apps (Home tab), or ``display`` when shiny is missing but napari is
installed; headless, or with neither installed, it falls through to the
standard subcommand help.
"""

import argparse
import os
import sys
import types

import pytest

import helicon
from helicon import helicon as helicon_mod


@pytest.fixture
def restore_argv():
    saved = sys.argv[:]
    yield
    sys.argv = saved


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)


class TestHasDisplay:
    def test_macos_always_true(self, monkeypatch, clean_env):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert helicon_mod._has_display() is True

    def test_windows_always_true(self, monkeypatch, clean_env):
        monkeypatch.setattr(sys, "platform", "win32")
        assert helicon_mod._has_display() is True

    def test_linux_no_display_false(self, monkeypatch, clean_env):
        monkeypatch.setattr(sys, "platform", "linux")
        assert helicon_mod._has_display() is False

    def test_linux_with_x11_true(self, monkeypatch, clean_env):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("DISPLAY", ":0")
        assert helicon_mod._has_display() is True

    def test_linux_with_wayland_true(self, monkeypatch, clean_env):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert helicon_mod._has_display() is True

    def test_offscreen_overrides_display(self, monkeypatch, clean_env):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        assert helicon_mod._has_display() is False


class TestDefaultCommand:
    @pytest.fixture
    def gui(self, monkeypatch, restore_argv, clean_env):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(helicon, "has_shiny", lambda: True)
        monkeypatch.setattr(helicon, "has_napari", lambda: True)
        sys.argv = ["helicon"]

    def test_no_args_launches_webapps(self, gui):
        assert helicon_mod._default_command() == "webApps"

    def test_no_shiny_launches_display(self, gui, monkeypatch):
        monkeypatch.setattr(helicon, "has_shiny", lambda: False)
        assert helicon_mod._default_command() == "display"

    def test_neither_installed(self, gui, monkeypatch):
        monkeypatch.setattr(helicon, "has_shiny", lambda: False)
        monkeypatch.setattr(helicon, "has_napari", lambda: False)
        assert helicon_mod._default_command() is None

    def test_headless(self, gui, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert helicon_mod._default_command() is None

    def test_with_subcommand(self, gui):
        sys.argv = ["helicon", "cryosparc"]
        assert helicon_mod._default_command() is None


class TestMainDispatch:
    @pytest.fixture
    def gui(self, monkeypatch, restore_argv, clean_env):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(helicon, "has_shiny", lambda: True)
        monkeypatch.setattr(helicon, "has_napari", lambda: True)
        monkeypatch.setattr(helicon_mod, "_maybe_reexec_macos_display", lambda: None)
        sys.argv = ["helicon"]

    @pytest.fixture
    def called(self, monkeypatch):
        called = {}

        def fake_import_module(name):
            # stands in for the command modules, so the tests need neither
            # shiny nor napari installed
            command = name.rsplit(".", 1)[-1]
            return types.SimpleNamespace(
                main=lambda args: called.__setitem__(command, vars(args))
            )

        def fake_get_commands(**kwargs):
            called["help"] = True

        monkeypatch.setattr(helicon_mod, "import_module", fake_import_module)
        monkeypatch.setattr(helicon_mod, "_get_commands", fake_get_commands)
        return called

    def test_bare_helicon_launches_webapps(self, gui, called):
        helicon_mod.main()
        assert called == {"webApps": {}}

    def test_bare_helicon_launches_display_without_shiny(
        self, gui, called, monkeypatch
    ):
        monkeypatch.setattr(helicon, "has_shiny", lambda: False)
        helicon_mod.main()
        assert called == {"display": {"folder": None}}
        assert sys.argv == ["helicon", "display"]

    def test_bare_helicon_falls_through_when_headless(self, gui, called, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        helicon_mod.main()
        assert called == {"help": True}
        assert sys.argv == ["helicon"]


class TestOnlyTheCommandThatRunsIsImported:
    """Importing every command (the file browser's Qt stack, the images2star
    plugins, ...) to print a list of one-line descriptions took 15 s."""

    NAMES = ["cryosparc", "display", "images2star", "proc3d", "webApps"]

    def test_the_command_is_the_first_argument_that_is_not_an_option(self):
        pick = helicon_mod._selected_command
        assert pick(["helicon", "proc3d", "--help"], self.NAMES) == "proc3d"
        assert pick(["helicon", "--version", "display"], self.NAMES) == "display"
        assert pick(["helicon", "--help"], self.NAMES) is None
        assert pick(["helicon"], self.NAMES) is None
        assert pick(["helicon", "nosuchcommand"], self.NAMES) is None

    def test_the_docstring_is_read_from_the_source_without_importing(self):
        has_add_args, doc = helicon_mod._source_doc("webApps")
        assert has_add_args is True
        assert doc.startswith("A Helicon Web app")
        assert helicon_mod._source_doc("no_such_command") == (False, None)

    def test_the_docstring_read_is_the_modules_own(self):
        import importlib

        for name in ("update", "trueFSC", "webCalEM"):
            _, doc = helicon_mod._source_doc(name)
            assert doc == importlib.import_module(f"helicon.commands.{name}").__doc__

    def _run(self, monkeypatch, argv):
        loaded = []
        real = helicon_mod.import_module
        monkeypatch.setattr(
            helicon_mod,
            "import_module",
            lambda name, *a: loaded.append(name.rsplit(".", 1)[-1]) or real(name, *a),
        )
        monkeypatch.setattr(sys, "argv", argv)
        # argparse (3.14) names the program "python -m pytest" when the tests
        # are run with python -m, instead of taking it from sys.argv
        monkeypatch.setattr(sys.modules["__main__"], "__spec__", None, raising=False)
        monkeypatch.setattr(helicon, "has_napari", lambda: True)
        monkeypatch.setattr(helicon, "has_shiny", lambda: True)
        monkeypatch.setattr(helicon, "has_streamlit", lambda: False)
        with pytest.raises(SystemExit):
            helicon_mod._get_commands(
                cli_commands=["update", "trueFSC", "webCalEM"],
                napari_commands=[],
                shiny_commands=[],
                streamlit_commands=[],
                doc_str="test",
            )
        return loaded

    def test_help_imports_no_command(self, monkeypatch, capsys):
        loaded = self._run(monkeypatch, ["helicon", "--help"])
        assert loaded == []
        out = capsys.readouterr().out
        # every command is still listed, with its description
        for name in ("update", "trueFSC", "webCalEM"):
            assert name in out

    def test_a_command_imports_only_itself(self, monkeypatch, capsys):
        loaded = self._run(monkeypatch, ["helicon", "update", "--help"])
        assert loaded == ["update"]
        assert "usage: helicon update" in capsys.readouterr().out
