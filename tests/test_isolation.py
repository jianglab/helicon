"""A test run leaves the settings of the user alone."""

import os
from pathlib import Path


def test_the_settings_of_the_run_are_not_the_users():
    from PySide6.QtCore import QSettings

    path = Path(QSettings("helicon", "display").fileName()).resolve()
    assert Path(os.environ["XDG_CONFIG_HOME"]).resolve() in path.parents
    assert Path.home().resolve() / ".config" not in path.parents


def test_what_a_test_saves_stays_in_the_run():
    from PySide6.QtCore import QSettings

    settings = QSettings("helicon", "display")
    settings.setValue("isolation_probe", "x")
    settings.sync()
    assert Path(settings.fileName()).is_file()
    assert os.environ["XDG_CONFIG_HOME"] in settings.fileName()
    settings.remove("isolation_probe")


def test_windows_are_not_opened_on_the_screen_the_tests_run_from():
    # offscreen unless the person running the tests chose a platform
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    chosen = os.environ.get("QT_QPA_PLATFORM")
    assert app.platformName() == (chosen or "offscreen")
