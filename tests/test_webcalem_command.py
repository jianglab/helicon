import argparse

import pytest

from helicon import helicon as cli
from helicon.commands import webCalEM


def _args(argv):
    parser = argparse.ArgumentParser()
    webCalEM.add_args(parser)
    return parser.parse_args(argv)


class TestWebCalEMCommand:
    def test_the_option_is_wired(self):
        assert _args([]).printUrl is False
        assert _args(["--printUrl"]).printUrl is True

    def test_it_opens_the_hosted_page(self, monkeypatch, capsys):
        opened = []
        monkeypatch.setattr(
            webCalEM.webbrowser, "open", lambda url: opened.append(url) or True
        )
        webCalEM.main(_args([]))
        assert opened == [webCalEM.URL]
        assert capsys.readouterr().out.strip() == webCalEM.URL

    def test_print_url_opens_nothing(self, monkeypatch, capsys):
        monkeypatch.setattr(
            webCalEM.webbrowser, "open", lambda url: raise_("must not open")
        )
        webCalEM.main(_args(["--printUrl"]))
        assert capsys.readouterr().out.strip() == webCalEM.URL

    def test_no_browser_only_warns(self, monkeypatch, caplog):
        monkeypatch.setattr(webCalEM.webbrowser, "open", lambda url: False)
        webCalEM.main(_args([]))
        assert webCalEM.URL in caplog.text

    def test_it_is_listed_among_the_commands(self):
        assert "webCalEM" in cli.cli_commands


class TestInTheFileBrowserAppsMenu:
    def test_the_apps_menu_runs_the_command(self):
        from helicon.lib.gui.file_browser import _APP_LAUNCH_TABLE

        rows = {row[0]: row for row in _APP_LAUNCH_TABLE}
        assert rows["WebCalEM"] == ("WebCalEM", None, "helicon.commands.webCalEM", None)

    def test_the_menu_entry_names_an_importable_command(self):
        import importlib

        from helicon.lib.gui.file_browser import _APP_LAUNCH_TABLE

        module = importlib.import_module(
            next(r[2] for r in _APP_LAUNCH_TABLE if r[0] == "WebCalEM")
        )
        assert callable(module.main) and callable(module.add_args)


def raise_(message):
    raise AssertionError(message)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class TestAppsMenuStaysComplete:
    def _widget(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        return FolderBrowserWidget(start_dir=str(tmp_path))

    def test_configure_relion_is_never_moved_to_the_macos_application_menu(
        self, tmp_path, qapp
    ):
        from PySide6.QtGui import QAction

        widget = self._widget(tmp_path, qapp)
        # with a heuristic role, macOS takes "Configure ..." out of this menu
        assert widget._configure_relion_action.menuRole() == QAction.MenuRole.NoRole

    def test_the_apps_menu_lists_the_tools_in_order(self, tmp_path, qapp):
        widget = self._widget(tmp_path, qapp)
        labels = [a.text() for a in widget._apps_menu.actions()]
        assert labels[:3] == ["Terminal", "Configure RELION…", "WebCalEM"]
