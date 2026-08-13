"""Display-theme helpers shared by every helicon window type.

The file browser is the canonical theme owner
(``helicon.lib.gui.file_browser`` persists ``_saved_theme`` and
``_resolved_theme``); these helpers translate that state into Qt
stylesheets/palettes for auxiliary display windows and into napari
theme/background colors.
"""

from __future__ import annotations

try:
    from PySide6.QtWidgets import QLayout, QWidget
except ImportError:  # pragma: no cover - only without the Qt stack
    QLayout = None
    QWidget = None

from .trackers import _napari


def _display_theme_stylesheet() -> str:
    """Return the shared Qt stylesheet for button-launched display windows."""
    from helicon.lib.gui.file_browser import (
        _THEME_COLORS,
        _resolved_theme,
        _saved_theme,
    )

    colors = _THEME_COLORS[_resolved_theme(_saved_theme())]
    return f"""
        QWidget {{
            background-color: {colors["window"]};
            color: {colors["text"]};
        }}
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 3px;
            padding: 3px;
        }}
        QToolButton, QPushButton {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 3px;
            padding: 3px 8px;
        }}
        QToolButton:hover, QPushButton:hover {{
            background-color: {colors["accent"]};
            color: #ffffff;
        }}
        QToolButton:pressed, QPushButton:pressed {{
            background-color: {colors["pressed"]};
            border: 1px solid {colors["accent_border"]};
        }}
        QComboBox QAbstractItemView {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            selection-background-color: {colors["accent"]};
        }}
    """


def _display_theme_palette():
    """Build a Qt palette matching the persisted display theme."""
    from PySide6.QtGui import QPalette, QColor
    from helicon.lib.gui.file_browser import (
        _THEME_COLORS,
        _resolved_theme,
        _saved_theme,
    )

    colors = _THEME_COLORS[_resolved_theme(_saved_theme())]
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(colors["input"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["window"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(colors["input"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    return palette


def _display_plot_theme_colors() -> dict[str, str]:
    """Return colors for plot widgets in the resolved display theme."""
    from helicon.lib.gui.file_browser import _resolved_theme, _saved_theme

    if _resolved_theme(_saved_theme()) == "Light":
        return {
            "background": "#ffffff",
            "foreground": "#202020",
            "grid": "#909090",
            "crosshair": "#707070",
            "tooltip_background": "rgba(255,255,255,220)",
            "tooltip_foreground": "#202020",
        }
    return {
        "background": "#202020",
        "foreground": "#dcdcdc",
        "grid": "#a0a0a0",
        "crosshair": "#969696",
        "tooltip_background": "rgba(0,0,0,180)",
        "tooltip_foreground": "#dcdcdc",
    }


def _refresh_display_theme_windows() -> None:
    """Reapply the saved theme to already-open auxiliary display windows."""
    if QLayout is None:
        return
    from PySide6.QtWidgets import QApplication

    stylesheet = _display_theme_stylesheet()
    palette = _display_theme_palette()
    for window in QApplication.topLevelWidgets():
        if window.property("helicon_theme_window"):
            window.setStyleSheet(stylesheet)
            window.setPalette(palette)
            for child in window.findChildren(QWidget):
                child.setStyleSheet(stylesheet)
                child.setPalette(palette)
            apply_theme = getattr(window, "_apply_display_theme", None)
            if apply_theme is not None:
                apply_theme()
            try:
                from helicon.lib.gui.gallery_widget import _apply_gallery_theme

                _apply_gallery_theme(window)
            except Exception:
                pass
    _refresh_napari_theme()


def _get_display_theme() -> str:
    """Return the persisted file-browser/web-app theme."""
    from helicon.lib.gui.file_browser import _saved_theme

    return _saved_theme()


def _napari_display_theme() -> str:
    """Map the saved Helicon theme to napari's theme names."""
    theme = _get_display_theme()
    return {"Dark": "dark", "Light": "light", "System": "system"}[theme]


def _napari_canvas_background() -> str:
    """Return the napari canvas background for the saved display theme."""
    from helicon.lib.gui.file_browser import _resolved_theme

    if _resolved_theme(_get_display_theme()) == "Light":
        return "#ffffff"
    return "#202020"


def _refresh_napari_theme() -> None:
    """Apply the saved theme to all currently open napari viewers."""
    try:
        theme = _napari_display_theme()
        background = _napari_canvas_background()
        for viewer in _napari.alive():
            viewer.theme = theme
            viewer.background_color = background
    except Exception:
        # napari is optional and may not be initialized yet.
        pass
