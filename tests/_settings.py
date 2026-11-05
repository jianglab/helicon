"""Settings of their own for a test of the file browser (see also conftest.py, which keeps a whole
run away from the user's)."""

from PySide6.QtCore import QSettings

# the modules of the file browser that make a QSettings
_MODULES = (
    "helicon.lib.gui.file_browser",
    "helicon.lib.gui.browser_settings",
    "helicon.lib.gui.browser_style",
    "helicon.lib.gui.browser_model",
)
# the modules that make their QSettings in a function ``_settings`` (imported when used)
_ACCESSORS = ("helicon.lib.gui.job_outcome",)


def own_settings(monkeypatch, ini_path) -> None:
    """Make every module of the file browser read and write one ini file for this test."""
    import importlib

    def factory(*args):
        return QSettings(str(ini_path), QSettings.Format.IniFormat)

    for name in _MODULES:
        module = importlib.import_module(name)
        if hasattr(module, "QSettings"):
            monkeypatch.setattr(module, "QSettings", factory)
    for name in _ACCESSORS:
        monkeypatch.setattr(importlib.import_module(name), "_settings", factory)


def use_theme(monkeypatch, theme: str) -> None:
    """The saved theme of the file browser is ``theme`` (in every module that reads it)."""
    import importlib

    for name in _MODULES:
        module = importlib.import_module(name)
        if hasattr(module, "_saved_theme"):
            monkeypatch.setattr(module, "_saved_theme", lambda: theme)
