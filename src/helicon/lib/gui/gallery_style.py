"""The colours and style sheet of the gallery and of the orthogonal viewer, from the theme of the file browser."""

from __future__ import annotations

from PySide6.QtGui import (
    QColor,
)
from PySide6.QtWidgets import (
    QWidget,
)


def _gallery_theme_colors() -> dict[str, str]:
    """Return colors for custom-painted gallery widgets, from the shared theme."""
    from helicon.lib.gui.browser_style import (
        _THEME_COLORS,
        _resolved_theme,
        _saved_theme,
    )

    resolved = _resolved_theme(_saved_theme())
    colors = _THEME_COLORS[resolved]
    base = {
        "background": "#ffffff" if resolved == "Light" else colors["window"],
        "panel": colors["window"],
        "text": colors["text"],
        "input": colors["input"],
        "border": colors["border"],
        "accent": colors["accent"],
        "muted": colors["muted"],
    }
    if resolved == "Light":
        return {
            **base,
            # QColor does not parse CSS rgba() strings reliably in Qt6;
            # use 8-digit ARGB hex so the painter preserves the alpha.
            "label_background": "#d2ffffff",
            "scrollbar": "#d0d0d0",
            "scrollbar_thumb": "#a0a0a0",
        }
    return {
        **base,
        "label_background": "#8c000000",
        "scrollbar": "#1f1f1f",
        "scrollbar_thumb": "#5a5a5a",
    }


def _gallery_qss(colors: dict[str, str]) -> str:
    """Return a theme-aware stylesheet for gallery controls (the shared widget look)."""
    from helicon.lib.gui.browser_style import (
        _THEME_COLORS,
        _resolved_theme,
        _saved_theme,
        _widgets_qss,
    )

    theme = _THEME_COLORS[_resolved_theme(_saved_theme())]
    return f"""
        QWidget {{
            background-color: {colors["panel"]};
            color: {colors["text"]};
        }}
        QLabel, QCheckBox, QRadioButton {{ font-size: 11px; }}
        QPushButton, QComboBox, QSpinBox, QDoubleSpinBox {{ font-size: 11px; }}
        """ + _widgets_qss(
        theme
    )


def _button_icon(name: str, colors: dict[str, str] | None = None):
    """A qtawesome icon in the text colour of the theme (white when hovered), or None."""
    try:
        import qtawesome

        colors = colors or _gallery_theme_colors()
        return qtawesome.icon(
            name,
            color=colors["text"],
            color_active="#ffffff",
            color_on="#ffffff",
            color_selected="#ffffff",
        )
    except Exception:
        return None


def _apply_gallery_theme(widget: QWidget) -> None:
    """Refresh gallery child widgets after the shared theme changes."""
    for child in [widget, *widget.findChildren(QWidget)]:
        apply_theme = getattr(child, "_apply_display_theme", None)
        if apply_theme is not None:
            apply_theme()
        else:
            colors = _gallery_theme_colors()
            palette = child.palette()
            palette.setColor(child.backgroundRole(), QColor(colors["panel"]))
            child.setPalette(palette)
            child.update()
