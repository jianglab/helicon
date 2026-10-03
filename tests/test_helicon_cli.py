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
