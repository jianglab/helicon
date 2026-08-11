"""Tests for the ``helicon`` CLI entrypoint (helicon.py).

Covers the bare-``helicon`` → ``helicon display`` default (Option A):
a graphical display + napari launches display; headless or missing
napari falls through to the standard subcommand help.
"""
import argparse
import os
import sys

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


class TestMaybeLaunchDisplayDefault:
    def test_no_args_launches_when_gui_available(
        self, monkeypatch, restore_argv, clean_env
    ):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(helicon, "has_napari", lambda: True)
        sys.argv = ["helicon"]
        assert helicon_mod._maybe_launch_display_default() is True
        assert "display" in sys.argv

    def test_headless_does_not_launch(
        self, monkeypatch, restore_argv, clean_env
    ):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(helicon, "has_napari", lambda: True)
        sys.argv = ["helicon"]
        assert helicon_mod._maybe_launch_display_default() is False
        assert sys.argv == ["helicon"]

    def test_no_napari_does_not_launch(
        self, monkeypatch, restore_argv, clean_env
    ):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(helicon, "has_napari", lambda: False)
        sys.argv = ["helicon"]
        assert helicon_mod._maybe_launch_display_default() is False
        assert sys.argv == ["helicon"]

    def test_with_subcommand_does_not_launch(
        self, monkeypatch, restore_argv, clean_env
    ):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(helicon, "has_napari", lambda: True)
        sys.argv = ["helicon", "cryosparc"]
        assert helicon_mod._maybe_launch_display_default() is False
        assert sys.argv == ["helicon", "cryosparc"]


class TestMainDispatch:
    def test_bare_helicon_launches_display_when_gui(
        self, monkeypatch, restore_argv, clean_env
    ):
        called = {}

        def fake_display_main(args):
            called["main"] = True
            called["folder"] = args.folder

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(helicon, "has_napari", lambda: True)
        monkeypatch.setattr(
            "helicon.commands.display.main", fake_display_main
        )
        monkeypatch.setattr(
            helicon_mod, "_maybe_reexec_macos_display", lambda: None
        )
        sys.argv = ["helicon"]
        helicon_mod.main()
        assert called == {"main": True, "folder": None}

    def test_bare_helicon_falls_through_when_headless(
        self, monkeypatch, restore_argv, clean_env
    ):
        get_commands_called = {}

        def fake_get_commands(**kwargs):
            get_commands_called["called"] = True

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(helicon, "has_napari", lambda: True)
        monkeypatch.setattr(helicon_mod, "_get_commands", fake_get_commands)
        monkeypatch.setattr(
            helicon_mod, "_maybe_reexec_macos_display", lambda: None
        )
        sys.argv = ["helicon"]
        helicon_mod.main()
        assert get_commands_called == {"called": True}

    def test_bare_helicon_falls_through_when_no_napari(
        self, monkeypatch, restore_argv, clean_env
    ):
        get_commands_called = {}

        def fake_get_commands(**kwargs):
            get_commands_called["called"] = True

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(helicon, "has_napari", lambda: False)
        monkeypatch.setattr(helicon_mod, "_get_commands", fake_get_commands)
        monkeypatch.setattr(
            helicon_mod, "_maybe_reexec_macos_display", lambda: None
        )
        sys.argv = ["helicon"]
        helicon_mod.main()
        assert get_commands_called == {"called": True}
