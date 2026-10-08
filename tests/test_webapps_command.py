"""The web app launcher: the reloader (which imports the app twice) is opt-in."""

import argparse

import pytest

from helicon.commands import webApps


def _parse(*argv):
    return webApps.add_args(argparse.ArgumentParser()).parse_args(list(argv))


class TestReloadOption:
    def test_reload_is_off_by_default(self):
        assert _parse().reload is False

    def test_reload_can_be_asked_for(self):
        assert _parse("--reload").reload is True

    @pytest.mark.parametrize("flag,expected", [([], False), (["--reload"], True)])
    def test_main_passes_it_to_the_launcher(self, monkeypatch, flag, expected):
        import helicon.lib.shiny as shiny

        seen = {}
        monkeypatch.setattr(
            shiny, "launch_shiny_app", lambda app, **kw: seen.update(app=app, **kw)
        )
        webApps.main(_parse(*flag))
        assert seen["app"] == "helicon.webApps.app:app"
        assert seen["reload"] is expected and seen["block"] is True
