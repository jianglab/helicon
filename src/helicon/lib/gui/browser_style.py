"""The colours and style sheets of the file browser and of the windows it opens, and the saved theme."""

from __future__ import annotations


from PySide6.QtCore import QSettings
from PySide6.QtGui import (
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
)


_THEMES = ("Dark", "Light", "System")

_DEFAULT_THEME = "System"

_THEME_COLORS = {
    "Dark": {
        "window": "#2d2d2d",
        "input": "#3c3c3c",
        "text": "#cccccc",
        "strong_text": "#e8e8e8",
        "border": "#555555",
        "header_border": "#2d2d2d",
        "hover": "#3a3a3a",
        "accent": "#007bc2",
        "pressed": "#0069a5",
        "accent_border": "#268fcb",
        "disabled": "#7a7a7a",
        "disabled_bg": "#2b2b2b",
        "error": "#ff6b6b",
        "muted": "#8f8f8f",
        "app_text": "#4db3ec",
        "alt": "#303030",
        "folder": "#e0a93b",
        "file_icon": "#8fb0dc",
    },
    "Light": {
        "window": "#f4f4f4",
        "input": "#ffffff",
        "text": "#202020",
        "strong_text": "#202020",
        "border": "#b8b8b8",
        "header_border": "#d2d2d2",
        "hover": "#e8eef7",
        "accent": "#007bc2",
        "pressed": "#0069a5",
        "accent_border": "#00629b",
        "disabled": "#888888",
        "disabled_bg": "#e5e5e5",
        "error": "#b3261e",
        "muted": "#6e6e6e",
        "app_text": "#007bc2",
        "alt": "#eeeeee",
        "folder": "#c48a12",
        "file_icon": "#007bc2",
    },
}


def _saved_theme() -> str:
    """Return the persisted browser theme, falling back to the default."""
    value = str(QSettings("helicon", "display").value("theme", _DEFAULT_THEME))
    return value if value in _THEMES else _DEFAULT_THEME


def _save_theme(theme: str) -> None:
    """Persist a valid browser theme for the next display launch."""
    if theme in _THEMES:
        QSettings("helicon", "display").setValue("theme", theme)


def _resolved_theme(theme: str) -> str:
    """Resolve ``System`` to the current Qt light/dark palette."""
    if theme != "System":
        return theme if theme in _THEME_COLORS else _DEFAULT_THEME
    window = QApplication.palette().color(QPalette.ColorRole.Window)
    return "Dark" if window.lightness() < 128 else "Light"


def _widgets_qss(colors: dict[str, str]) -> str:
    """The look of the common widgets (sliders, scroll bars, boxes, buttons, tips) in a theme.

    Shared by the file browser and by the windows it opens, so a slider or a button
    looks the same everywhere. It sets no background for the window itself.
    """
    return f"""
        QLabel {{ background: transparent; }}
        QSlider::groove:horizontal {{
            height: 4px;
            background: {colors["border"]};
            border-radius: 2px;
        }}
        QSlider::sub-page:horizontal {{
            background: {colors["accent"]};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            width: 12px;
            height: 12px;
            margin: -5px 0;
            border-radius: 7px;
            background: {colors["strong_text"]};
            border: 2px solid {colors["accent"]};
        }}
        QSlider::handle:horizontal:hover {{ background: {colors["accent"]}; }}
        QSlider::groove:vertical {{
            width: 4px;
            background: {colors["border"]};
            border-radius: 2px;
        }}
        QSlider::add-page:vertical {{
            background: {colors["accent"]};
            border-radius: 2px;
        }}
        QSlider::handle:vertical {{
            width: 12px;
            height: 12px;
            margin: 0 -5px;
            border-radius: 7px;
            background: {colors["strong_text"]};
            border: 2px solid {colors["accent"]};
        }}
        QScrollBar:vertical {{ width: 10px; background: transparent; margin: 0; }}
        QScrollBar:horizontal {{ height: 10px; background: transparent; margin: 0; }}
        QScrollBar::handle:vertical {{
            background: {colors["border"]};
            border-radius: 5px;
            min-height: 28px;
        }}
        QScrollBar::handle:horizontal {{
            background: {colors["border"]};
            border-radius: 5px;
            min-width: 28px;
        }}
        QScrollBar::handle:hover {{ background: {colors["muted"]}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
        QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 4px;
            padding: 3px 6px;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
            border: 1px solid {colors["accent"]};
        }}
        QComboBox::drop-down {{ border: none; width: 18px; }}
        QPushButton {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 5px;
            padding: 4px 12px;
        }}
        QPushButton:hover {{ background-color: {colors["accent"]}; color: #ffffff; }}
        QPushButton:pressed {{ background-color: {colors["pressed"]}; }}
        QPushButton:checked {{
            background-color: {colors["accent"]};
            color: #ffffff;
            border: 1px solid {colors["accent_border"]};
        }}
        QPushButton:disabled {{
            color: {colors["disabled"]};
            background-color: {colors["disabled_bg"]};
        }}
        QRadioButton {{ spacing: 6px; }}
        QRadioButton::indicator {{
            width: 14px;
            height: 14px;
            border: 1px solid {colors["border"]};
            border-radius: 8px;
            background-color: {colors["input"]};
        }}
        QRadioButton::indicator:checked {{
            border: 1px solid {colors["accent"]};
            background-color: qradialgradient(
                cx: 0.5, cy: 0.5, radius: 0.5, fx: 0.5, fy: 0.5,
                stop: 0 {colors["accent"]}, stop: 0.45 {colors["accent"]},
                stop: 0.55 {colors["input"]}, stop: 1 {colors["input"]});
        }}
        QCheckBox {{ spacing: 6px; }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {colors["border"]};
            border-radius: 3px;
            background-color: {colors["input"]};
        }}
        QCheckBox::indicator:checked {{
            background-color: {colors["accent"]};
            border: 1px solid {colors["accent_border"]};
        }}
        QGroupBox {{
            border: 1px solid {colors["header_border"]};
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 6px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: {colors["muted"]};
        }}
        QProgressBar {{
            border: none;
            border-radius: 3px;
            background-color: {colors["border"]};
            text-align: center;
            color: {colors["text"]};
        }}
        QProgressBar::chunk {{ background-color: {colors["accent"]}; border-radius: 3px; }}
        QToolTip {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            padding: 4px 6px;
        }}
    """


def _browser_stylesheet(colors: dict[str, str]) -> str:
    """Build the shared stylesheet for the browser and launch buttons."""
    base = f"""
        QWidget {{
            background-color: {colors["window"]};
            color: {colors["text"]};
        }}
        QTreeView {{
            font-size: 13px;
            background-color: {colors["window"]};
            alternate-background-color: {colors["alt"]};
            color: {colors["text"]};
            border: none;
            outline: 0;
        }}
        QTreeView::item {{ padding: 4px 2px; border: none; }}
        QTreeView::item:hover {{ background-color: {colors["hover"]}; }}
        QTreeView::item:selected {{
            background-color: {colors["accent"]};
            color: #ffffff;
        }}
        QHeaderView::section {{
            padding: 6px 8px;
            font-weight: normal;
            background-color: {colors["window"]};
            color: {colors["muted"]};
            border: none;
            border-bottom: 1px solid {colors["header_border"]};
        }}
        QHeaderView::section:hover {{ color: {colors["text"]}; }}
        QSplitter::handle {{ background-color: {colors["header_border"]}; }}
        QSplitter::handle:horizontal {{ width: 1px; }}
        QTreeWidget {{
            font-size: 13px;
            background-color: {colors["window"]};
            color: {colors["text"]};
            border: none;
            outline: 0;
        }}
        QTreeWidget::item {{ padding: 3px 2px; }}
        QTreeWidget::item:hover {{ background-color: {colors["hover"]}; }}
        QTreeWidget::item:selected {{
            background-color: {colors["accent"]};
            color: #ffffff;
        }}
        #pathCrumbs {{
            background-color: {colors["input"]};
            border: 1px solid {colors["border"]};
            border-radius: 3px;
        }}
        #pathSeparator {{ color: {colors["muted"]}; background: transparent; }}
        #pathCrumbs QToolButton {{ padding: 2px 5px; }}
        #previewFacts, #previewTitle, #previewNote {{ background: transparent; }}
        QLineEdit, QComboBox {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 3px;
            padding: 3px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            selection-background-color: {colors["accent"]};
        }}
        QPushButton {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 5px;
            padding: 4px 12px;
        }}
        QPushButton[category="app"] {{
            color: {colors["app_text"]};
            border: 1px solid {colors["accent"]};
        }}
        QPushButton[category="app"]:hover {{
            background-color: {colors["accent"]};
            color: #ffffff;
        }}
        #actionBar {{
            background-color: {colors["window"]};
            border-top: 1px solid {colors["header_border"]};
        }}
        #actionBar[place="preview"] {{
            background-color: transparent;
            border: none;
        }}
        QToolButton {{
            background-color: transparent;
            color: {colors["text"]};
            border: 1px solid transparent;
            border-radius: 4px;
            padding: 4px 6px;
        }}
        QToolButton:hover {{ background-color: {colors["hover"]}; }}
        QMenuBar {{
            background-color: {colors["window"]};
            color: {colors["text"]};
            border-bottom: 1px solid {colors["border"]};
        }}
        QMenuBar::item {{
            background-color: transparent;
            padding: 4px 8px;
        }}
        QMenuBar::item:selected, QMenu::item:selected {{
            background-color: {colors["accent"]};
            color: #ffffff;
        }}
        QMenu {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
        }}
        QMenu::item {{
            padding: 4px 24px 4px 8px;
        }}
        QPushButton:hover {{
            background-color: {colors["accent"]};
            color: #ffffff;
        }}
        QToolButton:pressed, QPushButton:pressed {{
            background-color: {colors["pressed"]};
            border: 1px solid {colors["accent_border"]};
        }}
        QCheckBox {{ color: {colors["strong_text"]}; spacing: 6px; }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {colors["border"]};
            border-radius: 3px;
            background-color: {colors["input"]};
        }}
        QCheckBox::indicator:checked {{
            background-color: {colors["accent"]};
            border: 1px solid {colors["accent_border"]};
            image: none;
        }}
    """
    return base + _widgets_qss(colors)


def _disabled_button_stylesheet(colors: dict[str, str]) -> str:
    """Return the explicit disabled style used for unavailable launchers."""
    return (
        "QPushButton:disabled { "
        f"color: {colors['disabled']}; "
        f"background-color: {colors['disabled_bg']}; "
        f"border: 1px solid {colors['border']}; }}"
    )
