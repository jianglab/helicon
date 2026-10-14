"""Qt widgets for napari integration."""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

from helicon.lib.cache import cache, setup_cache_dir
from helicon.lib.terminal import _open_terminal, _spawn_detached
from helicon.lib.gui.browser_parts import (
    FlowLayout,
    cryosparc_job_info,
    PathBar,
    PlacesSidebar,
    PreviewPane,
)

from PySide6.QtWidgets import (
    QWidget,
    QMainWindow,
    QTreeView,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QHeaderView,
    QMenu,
    QApplication,
    QPushButton,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QMenuBar,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QSplitter,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QPalette,
    QStandardItemModel,
    QStandardItem,
    QKeySequence,
    QShortcut,
    QAction,
    QActionGroup,
)
from PySide6.QtCore import (
    QModelIndex,
    Signal,
    Qt,
    QDir,
    QFileInfo,
    QThread,
    QObject,
    QSettings,
    QSize,
    QTimer,
)

__all__ = ["FolderBrowserWidget"]

_THEMES = ("Dark", "Light", "System")
_DEFAULT_THEME = "System"

# Standalone tools exposed through the Apps menu, in display order.
# Tuple: (menu label, shiny tab name or None, streamlit subcommand module or
# None, picker kind or None). A shiny tab is opened in the consolidated
# Helicon Lab web app without an input file; a streamlit module is launched via
# its subcommand ``main`` (WebCalEM is no streamlit app: its command only opens
# the hosted page in the browser). The picker kind selects a file picker that
# lets the user choose an input before launching the tool: "map" requires one .mrc,
# "two_maps" requires two half-map .mrc files, and "star" opens the tool's
# own in-panel file selector with no pre-set input.
_APP_LAUNCH_TABLE = [
    ("WebCalEM", None, "helicon.commands.webCalEM", None),
    ("Images2Star", None, None, "star"),
    ("Proc3D", None, None, "map"),
    ("WhereIsMyClass", "WhereIsMyClass", None, None),
    ("HelicalProjection", "HelicalProjection", None, None),
    ("HILL", "HILL", None, None),
    ("HelicalPitch", "HelicalPitch", None, None),
    ("Denovo3D", "Denovo3D", None, None),
    ("AbInitio3D", "AbInitio3D", None, None),
    ("HelicalLattice", "HelicalLattice", None, None),
    ("HI3D", "HI3D", None, None),
    ("TrueFSC", None, None, "two_maps"),
    ("Map2Seq", None, "helicon.commands.map2seq", None),
    ("ProCart", None, "helicon.commands.procart", None),
    ("CTF Simulation", None, "helicon.commands.ctfSimulation", None),
]

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


_IMAGE_EXTENSIONS = {
    ".mrc",
    ".mrcs",
    ".star",
    ".cs",
    ".lst",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".sqlite",
    ".html",
    ".htm",
}


def _get_recent_folders() -> list[str]:
    """Return up to the 5 most-recent starting folders, newest first.

    Persisted via QSettings under the ``helicon`` / ``display`` organisation
    so the list survives program restarts.
    """
    settings = QSettings("helicon", "display")
    raw = settings.value("recent_folders", [], type=list)
    seen = set()
    out = []
    for p in raw:
        p = str(p)
        if p and p not in seen and Path(p).is_dir():
            seen.add(p)
            out.append(p)
    return out[:5]


def _add_recent_folder(path: str) -> None:
    """Record ``path`` as the most-recent starting folder (max 5 kept)."""
    path = str(Path(path).resolve())
    if not Path(path).is_dir():
        return
    cur = _get_recent_folders()
    cur = [p for p in cur if p != path]
    cur.insert(0, path)
    cur = cur[:5]
    settings = QSettings("helicon", "display")
    settings.setValue("recent_folders", cur)


def _bookmark_key(path: str) -> str:
    """The form in which a folder is stored as a bookmark."""
    return str(Path(path).expanduser().resolve())


def _get_bookmarks() -> list[str]:
    """Return the bookmarked folders, in the order they were added.

    Persisted via QSettings like the recent folders. Unlike those, a folder
    that cannot be found is kept: it may be on a disk or network share that is
    only mounted now and then.
    """
    settings = QSettings("helicon", "display")
    raw = settings.value("bookmarks", [], type=list)
    out = []
    for p in raw:
        p = str(p)
        if p and p not in out:
            out.append(p)
    return out


def _save_bookmarks(paths: list[str]) -> None:
    QSettings("helicon", "display").setValue("bookmarks", list(paths))


def _is_bookmarked(path: str) -> bool:
    return _bookmark_key(path) in _get_bookmarks()


def _add_bookmark(path: str) -> None:
    """Bookmark ``path`` (at the end of the list, once)."""
    key = _bookmark_key(path)
    marks = _get_bookmarks()
    if key not in marks:
        _save_bookmarks(marks + [key])


def _remove_bookmark(path: str) -> None:
    """Remove the bookmark of ``path``, if there is one."""
    key = _bookmark_key(path)
    marks = _get_bookmarks()
    if key in marks:
        _save_bookmarks([p for p in marks if p != key])


def _move_bookmark(path: str, offset: int) -> None:
    """Move the bookmark of ``path`` ``offset`` places (negative is earlier)."""
    key = _bookmark_key(path)
    marks = _get_bookmarks()
    if key not in marks:
        return
    i = marks.index(key)
    j = min(max(i + offset, 0), len(marks) - 1)
    marks.insert(j, marks.pop(i))
    _save_bookmarks(marks)


def _clear_bookmarks() -> None:
    """Remove every bookmark."""
    _save_bookmarks([])


def _confirm_clear_bookmarks(parent) -> bool:
    """Ask before all the bookmarks are removed."""
    answer = QMessageBox.question(
        parent,
        "Clear All Bookmarks",
        "Remove all bookmarks? This cannot be undone.",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes


class _BookmarksDialog(QDialog):
    """Put the bookmarks in order, remove some, or clear them all.

    The order can be changed by dragging an entry or with the Move buttons;
    every change is saved at once.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Bookmarks")
        self.resize(560, 360)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Drag the bookmarks to put them in order."))
        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._list.model().rowsMoved.connect(self._save_order)
        self._list.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self._list)

        buttons = QHBoxLayout()
        self._up = QPushButton("Move Up")
        self._up.clicked.connect(partial(self.move_selected, -1))
        self._down = QPushButton("Move Down")
        self._down.clicked.connect(partial(self.move_selected, 1))
        self._remove = QPushButton("Remove")
        self._remove.clicked.connect(self.remove_selected)
        self._clear = QPushButton("Clear All")
        self._clear.clicked.connect(self.clear_all)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        for b in (self._up, self._down, self._remove, self._clear):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self._reload()

    def paths(self) -> list[str]:
        """The bookmarks as listed, in the order shown."""
        return [
            self._list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._list.count())
        ]

    def _reload(self, select: str | None = None) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for path in _get_bookmarks():
            label = path if Path(path).is_dir() else f"{path}  (not available)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._list.addItem(item)
            if path == select:
                self._list.setCurrentItem(item)
        self._list.blockSignals(False)
        self._update_buttons()

    def _selected(self) -> str | None:
        item = self._list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _update_buttons(self) -> None:
        row = self._list.currentRow()
        n = self._list.count()
        self._up.setEnabled(row > 0)
        self._down.setEnabled(0 <= row < n - 1)
        self._remove.setEnabled(row >= 0)
        self._clear.setEnabled(n > 0)

    def _save_order(self, *args) -> None:
        """Keep what a drag and drop did."""
        _save_bookmarks(self.paths())

    def move_selected(self, offset: int) -> None:
        path = self._selected()
        if path is not None:
            _move_bookmark(path, offset)
            self._reload(select=path)

    def remove_selected(self) -> None:
        path = self._selected()
        if path is not None:
            row = self._list.currentRow()
            _remove_bookmark(path)
            marks = _get_bookmarks()
            self._reload(select=marks[min(row, len(marks) - 1)] if marks else None)

    def clear_all(self) -> None:
        if _confirm_clear_bookmarks(self):
            _clear_bookmarks()
            self._reload()


_DIGIT_RUN = re.compile(r"(\d+)")


def _natural_sort_key(text: str) -> list:
    """Sort key that compares runs of digits as numbers.

    Plain text order puts ``J10`` before ``J2``, which scatters a CryoSPARC
    project's jobs: they are named ``J1`` to ``J121`` with no zero padding, so
    the numbering only reads continuously if the digits are compared as
    numbers. RELION's zero-padded ``job001`` is unaffected either way, and
    micrograph names with frame numbers benefit the same way jobs do.

    Digit runs sort before text at the same position, so ``J2`` precedes
    ``J2a``, and the comparison is case-insensitive.

    Parameters
    ----------
    text : str

    Returns
    -------
    list
        A list of ``(kind, number, text)`` triples, comparable between names.
    """
    key = []
    for index, part in enumerate(_DIGIT_RUN.split(text or "")):
        if index % 2:
            key.append((0, int(part), ""))
        elif part:
            key.append((1, 0, part.lower()))
    return key


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _format_item_count(n: int) -> str:
    """Format a directory's entry count with proper singular/plural phrasing.

    Parameters
    ----------
    n : int
        Number of visible entries.

    Returns
    -------
    str
        ``"1 item"`` for a single entry, ``"N items"`` otherwise.
    """
    return "1 item" if n == 1 else f"{n} items"


def _name_matches_filter(name: str, pattern: str, use_regex: bool) -> bool:
    """Return whether ``name`` matches the browser's active filter.

    An empty pattern matches everything. Regex mode uses ``re.search`` (any
    match), glob mode uses ``fnmatch.fnmatch``. Invalid regexes match
    everything so a bad pattern never hides the whole listing.

    Parameters
    ----------
    name : str
        Entry name to test.
    pattern : str
        Active filename filter (``"*"`` when no filter is set).
    use_regex : bool
        Whether ``pattern`` is a regular expression.

    Returns
    -------
    bool
        Whether the name should be shown in the listing.
    """
    if not pattern:
        return True
    if use_regex:
        import re

        try:
            return bool(re.search(pattern, name))
        except re.error:
            return True
    else:
        import fnmatch

        return fnmatch.fnmatch(name, pattern)


def _count_folder_entries(path: str | Path) -> int:
    """Count the visible (non-hidden) entries of a directory.

    Mirrors the browser's visibility rule of skipping entries whose name
    starts with a dot. Missing or unreadable directories count as zero so
    callers never have to special-case them.

    Parameters
    ----------
    path : str or Path
        Directory to count.

    Returns
    -------
    int
        Number of visible entries (``0`` when the directory cannot be read).
    """
    try:
        return sum(1 for e in Path(path).iterdir() if not e.name.startswith("."))
    except OSError:
        return 0


def _count_dir_items(path: Path, filter_pattern: str, use_regex: bool) -> int | None:
    """Count the entries of a directory the way the browser lists them.

    Mirrors ``FileBrowserModel._load_directory`` visibility rules: hidden
    entries (leading dot) are skipped and non-directory entries must match
    the active filter. Directories are always visible.

    Parameters
    ----------
    path : Path
        Directory to count.
    filter_pattern : str
        Active filename filter (``"*"`` when no filter is set).
    use_regex : bool
        Whether ``filter_pattern`` is a regular expression.

    Returns
    -------
    int or None
        Number of visible entries, or ``None`` when the directory cannot be
        read (e.g. ``PermissionError``).
    """
    try:
        count = 0
        for entry in path.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                count += 1
            elif _name_matches_filter(entry.name, filter_pattern, use_regex):
                count += 1
        return count
    except (OSError, PermissionError):
        return None


def _is_text_file(path: str) -> bool:
    """Check if a file is a text file by attempting UTF-8 decode."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


# MRC-format image extensions. RELION/CTFFIND write CTF power spectra as MRC
# maps with a ``.ctf`` suffix (e.g. ``*_PS.ctf``), so those are treated as
# MRC images too.
_MRC_EXTENSIONS = {".mrc", ".mrcs", ".map", ".ctf"}

# MRC extensions that may hold a 3D volume; ``.mrcs`` is always a 2D stack.
_MRC_VOLUME_EXTENSIONS = {".mrc", ".map", ".ctf"}

# Common raster image formats opened as 2D images (napari's built-in
# readers handle them, e.g. PNG, JPEG, TIFF).
_RASTER_IMAGE_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".webp",
    }
)

# Extensions with dedicated handlers/labels.  Everything else falls back
# to known text extension checks or uppercased extensions.
_KNOWN_EXTENSIONS = frozenset(
    {
        ".mrc",
        ".mrcs",
        ".map",
        ".ctf",
        ".star",
        ".bild",
        ".pdf",
        ".eps",
        ".cs",
        ".png",
        ".tif",
        ".tiff",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".webp",
        ".html",
        ".htm",
    }
)

_KNOWN_TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".py",
        ".pyi",
        ".md",
        ".rst",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".log",
        ".out",
        ".sh",
        ".zsh",
        ".bash",
        ".csv",
        ".tsv",
        ".ini",
        ".cfg",
        ".conf",
        ".env",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
        ".js",
        ".ts",
        ".css",
        ".mod",
        ".inp",
        ".dat",
    }
)


def _get_file_type_label(filepath: str) -> str:
    """Return a human-readable type label for the file browser.

    Detection order:
    1. Known file types by suffix (.mrc, .mrcs, .star, .bild, etc.)
    2. Known text extensions -> "Text"
    3. Anything else -> uppercased extension (or "File" if no extension)
    """
    name = os.path.basename(filepath).lower()
    ext = os.path.splitext(name)[1]

    if ext == ".star":
        if any(name.endswith(s) for s in _METADATA_STAR_SUFFIXES):
            return "Metadata"
        return "STAR"

    if ext == ".bild":
        return "3D Plot"

    if ext == ".cs":
        return "CryoSPARC"

    from helicon.lib import cryosparc_project

    if cryosparc_project.is_metadata_file(filepath):
        return "Metadata"

    if ext in (".html", ".htm"):
        return "Browser"

    if ext == ".pdf":
        return "PDF"

    if ext == ".eps":
        return "EPS"

    if ext == ".ctf":
        # RELION CTF power spectra are MRC maps with a .ctf suffix.
        return "MRC"

    if ext in _KNOWN_TEXT_EXTENSIONS:
        return "Text"

    if ext in _KNOWN_EXTENSIONS:
        return ext.lstrip(".").upper()

    return ext.lstrip(".").upper() if ext else "File"


def _get_file_info(filepath: str) -> tuple[str, str, str]:
    """Returns (info, n_images, pixel_size) tuple."""
    name = Path(filepath).name.lower()
    if name.endswith(".star") and any(
        name.endswith(s) for s in _METADATA_STAR_SUFFIXES
    ):
        # Pipeline/optimiser/model/sampling/job star files describe metadata,
        # not image data; show them as plain entries (opened as text).
        return "", "", ""
    ext = Path(filepath).suffix.lower()
    info = ""
    n_images = ""
    pixel_size = ""

    if ext in _MRC_EXTENSIONS:
        try:
            import mrcfile

            with mrcfile.open(filepath, header_only=True) as mrc:
                nx = int(mrc.header.nx)
                ny = int(mrc.header.ny)
                nz = int(mrc.header.nz)
                apix = float(mrc.voxel_size.x)
                if apix > 0:
                    pixel_size = f"{apix:.3f} Å"
                stack = ext == ".mrcs"
                if not stack and nz > 1:
                    # everything is .mrc in a CryoSPARC job: its type says whether
                    # this is a stack of class averages or particles
                    from helicon.lib import cryosparc_project

                    stack = cryosparc_project.mrc_content(filepath) == "stack"
                if stack:
                    n_images = str(nz) if nz > 1 else "1"
                    info = f"{nx}×{ny}"
                else:
                    n_images = "1"
                    info = f"{nx}×{ny}" if nz == 1 else f"{nx}×{ny}×{nz}"
        except Exception:
            pass

    elif ext in (".tif", ".tiff"):
        try:
            from PIL import Image

            with Image.open(filepath) as img:
                if hasattr(img, "n_frames") and img.n_frames > 1:
                    n_images = str(img.n_frames)
                    info = f"{img.n_frames}×{img.width}×{img.height}"
                else:
                    n_images = "1"
                    info = f"{img.width}×{img.height}"
        except Exception:
            pass

    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
        try:
            from PIL import Image

            with Image.open(filepath) as img:
                n_images = "1"
                info = f"{img.width}×{img.height}"
        except Exception:
            pass

    elif ext == ".star":
        try:
            star_dir = Path(filepath).parent

            # State for line-by-line star parser (avoids loading the entire
            # file into a DataFrame — critical for large *data.star files).
            col_names: list[str] = []
            image_col_idx = -1
            size_col_idx = -1
            apix_col_idx = -1
            in_loop = False
            in_data = False
            current_data_block = ""
            n_particles = 0
            first_image_ref: str | None = None
            optics_size: int | None = None
            optics_pixel_size: float | None = None

            with open(filepath) as f:
                for line in f:
                    raw = line.rstrip("\n\r")
                    s = raw.strip()
                    if not s or s.startswith("#"):
                        continue

                    if s.startswith("data_"):
                        current_data_block = s[5:].strip().lower()
                        in_loop = False
                        in_data = False
                        col_names = []
                        image_col_idx = -1
                        size_col_idx = -1
                        apix_col_idx = -1
                        continue

                    if s == "loop_":
                        in_loop = True
                        col_names = []
                        image_col_idx = -1
                        size_col_idx = -1
                        apix_col_idx = -1
                        continue

                    if in_loop and s.startswith("_"):
                        col_names.append(s.split()[0])
                        idx = len(col_names) - 1
                        cl = s.lower()
                        if "imagename" in cl or "micrographname" in cl:
                            if image_col_idx < 0:
                                image_col_idx = idx
                        if current_data_block == "optics":
                            if "imagesize" in cl and size_col_idx < 0:
                                size_col_idx = idx
                            if "imagepixelsize" in cl and apix_col_idx < 0:
                                apix_col_idx = idx
                        continue

                    if not in_data:
                        in_data = True

                    parts = raw.split()

                    if current_data_block == "optics":
                        if (
                            optics_size is None
                            and size_col_idx >= 0
                            and size_col_idx < len(parts)
                        ):
                            try:
                                optics_size = int(parts[size_col_idx])
                            except ValueError:
                                pass
                        if (
                            optics_pixel_size is None
                            and apix_col_idx >= 0
                            and apix_col_idx < len(parts)
                        ):
                            try:
                                optics_pixel_size = float(parts[apix_col_idx])
                            except ValueError:
                                pass
                    else:
                        n_particles += 1
                        if (
                            first_image_ref is None
                            and image_col_idx >= 0
                            and image_col_idx < len(parts)
                        ):
                            first_image_ref = parts[image_col_idx]

            if n_particles > 0:
                n_images = str(n_particles)

                if first_image_ref:
                    img_rel = (
                        first_image_ref.split("@")[-1]
                        if "@" in first_image_ref
                        else first_image_ref
                    )

                    img_path_resolved = None
                    for ancestor in [star_dir] + list(star_dir.parents):
                        candidate = ancestor / img_rel
                        if candidate.is_file():
                            img_path_resolved = candidate
                            break

                    if img_path_resolved is not None:
                        img_ext = img_path_resolved.suffix.lower()
                        if img_ext in (".mrc", ".mrcs", ".map"):
                            import mrcfile

                            with mrcfile.open(
                                str(img_path_resolved), header_only=True
                            ) as mrc:
                                nx = int(mrc.header.nx)
                                ny = int(mrc.header.ny)
                                nz = int(mrc.header.nz)
                                apix = float(mrc.voxel_size.x)
                                if apix > 0:
                                    pixel_size = f"{apix:.3f} Å"
                                if img_ext == ".mrcs" or nz == 1:
                                    info = f"{nx}×{ny}"
                                else:
                                    info = f"{nx}×{ny}×{nz}"
                        else:
                            from PIL import Image

                            with Image.open(str(img_path_resolved)) as img:
                                info = f"{img.width}×{img.height}"

                # Fallback to data_optics values when image file not found.
                if not info and optics_size is not None:
                    info = f"{optics_size}×{optics_size}"
                if (
                    not pixel_size
                    and optics_pixel_size is not None
                    and optics_pixel_size > 0
                ):
                    pixel_size = f"{optics_pixel_size:.3f} Å"
        except Exception:
            pass

    elif ext == ".pdf":
        try:
            from PySide6.QtPdf import QPdfDocument

            doc = QPdfDocument()
            doc.load(filepath)
            n = doc.pageCount()
            if n > 0:
                n_images = str(n)
        except Exception:
            pass

    elif ext in (".html", ".htm"):
        pass

    elif ext == ".cs":
        try:
            import numpy as np

            with np.load(filepath, allow_pickle=True) as data:
                if "particles" in data:
                    n = len(data["particles"])
                    n_images = str(n)
                    info = f"{n} particles"
        except Exception:
            pass

    return info, n_images, pixel_size


ROLE_KIND = Qt.ItemDataRole.UserRole + 2

# the kind of a row, for its icon: folder, image, stack, table, text, pdf, web, plot, file
_IMAGE_SUFFIXES = {
    ".mrc", ".map", ".ccp4", ".rec", ".ctf", ".tif", ".tiff", ".png", ".jpg", ".jpeg",
}  # fmt: skip
_STACK_SUFFIXES = {".mrcs", ".st", ".ali", ".hdf", ".img"}
_KIND_FOR_LABEL = {
    "Metadata": "table",
    "STAR": "table",
    "CryoSPARC": "table",
    "Text": "text",
    "PDF": "pdf",
    "EPS": "pdf",
    "Browser": "web",
    "3D Plot": "plot",
    "MRC": "image",
}
# the Font Awesome icon of each kind (qtawesome); the colour comes from the theme
_KIND_ICONS = {
    "folder": "fa5s.folder",
    "image": "fa5s.image",
    "stack": "fa5s.images",
    "table": "fa5s.table",
    "text": "fa5s.file-alt",
    "pdf": "fa5s.file-pdf",
    "web": "fa5s.globe",
    "plot": "fa5s.chart-area",
    "file": "fa5s.file",
}
_icon_cache: dict[tuple[str, str], QIcon] = {}


def _icon_kind(type_label: str, suffix: str, is_dir: bool) -> str:
    """The icon kind of a file or folder from its type label and suffix."""
    if is_dir:
        return "folder"
    suffix = suffix.lower()
    if suffix in _STACK_SUFFIXES:
        return "stack"
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    return _KIND_FOR_LABEL.get(type_label, "file")


def _themed_icon(kind: str, colors: dict[str, str]) -> QIcon:
    """The icon of a kind in the colour of the theme, cached.

    Drawn by qtawesome when it is installed (loaded here, not at import, to keep
    start-up short); otherwise the style's own folder and file icons.
    """
    color = colors["folder"] if kind == "folder" else colors["file_icon"]
    key = (kind, color)
    icon = _icon_cache.get(key)
    if icon is None:
        try:
            import qtawesome

            icon = qtawesome.icon(
                _KIND_ICONS.get(kind, "fa5s.file"),
                color=color,
                color_selected="#ffffff",  # on the accent of a selected row
            )
        except Exception:
            from PySide6.QtWidgets import QStyle

            style = QApplication.style()
            icon = style.standardIcon(
                QStyle.StandardPixmap.SP_DirIcon
                if kind == "folder"
                else QStyle.StandardPixmap.SP_FileIcon
            )
        _icon_cache[key] = icon
    return icon


def _format_modified(mtime: datetime, now: datetime | None = None) -> str:
    """The modification time for the list: "Today 21:07", "Yesterday 09:30", or the date."""
    now = now or datetime.now()
    days = (now.date() - mtime.date()).days
    if days == 0:
        return "Today " + mtime.strftime("%H:%M")
    if days == 1:
        return "Yesterday " + mtime.strftime("%H:%M")
    return mtime.strftime("%Y-%m-%d %H:%M")


COL_NAME = 0
COL_SIZE = 1
COL_TYPE = 2
COL_IMAGES = 3
COL_INFO = 4
COL_PIXELSIZE = 5
COL_MODIFIED = 6
NUM_COLUMNS = 7
ROLE_SORT = Qt.ItemDataRole.UserRole + 1
# A folder of up to this many entries is listed at once, with their sizes and dates. A
# longer one is listed in slices of the main thread (INSERT_SLICE_S seconds each, so the
# window keeps answering), and the sizes and dates are read in threads: a stat on a network
# disk takes about a millisecond, so 20,000 files froze the browser for 40 s.
SYNC_ROWS = 200
INSERT_SLICE_S = 0.03
# the columns of numbers are right-aligned
_RIGHT_COLUMNS = (COL_SIZE, COL_IMAGES, COL_PIXELSIZE)

# Star files that describe pipelines/optimisation rather than image data.
# Must stay in sync with _METADATA_STAR_SUFFIXES in commands/display.py.
_METADATA_STAR_SUFFIXES = (
    "pipeline.star",
    "optimiser.star",
    "model.star",
    "sampling.star",
    "job.star",
    "extractpick.star",
    "frameimage.star",
    "autopick.star",
)


class _StatWorker(QThread):
    """Reads the size and modification time of files off the main thread, in batches.

    ``stats_ready(epoch, batch)`` with ``batch`` a list of ``(path, size, mtime)``.
    """

    stats_ready = Signal(int, list)
    BATCH = 250

    def __init__(self, paths: list[str], epoch: int, parent=None) -> None:
        super().__init__(parent)
        self._paths = paths
        self._epoch = epoch
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        batch = []
        for path in self._paths:
            if self._stop:
                return
            try:
                st = os.stat(path)
                batch.append((path, st.st_size, st.st_mtime))
            except OSError:
                batch.append((path, 0, 0.0))
            if len(batch) >= self.BATCH:
                self.stats_ready.emit(self._epoch, batch)
                batch = []
        if batch and not self._stop:
            self.stats_ready.emit(self._epoch, batch)


class FileBrowserModel(QStandardItemModel):
    # emitted when every row of a folder is there (at once for a small folder)
    load_finished = Signal()

    def __init__(self, root_path: str, parent=None):
        super().__init__(0, NUM_COLUMNS, parent)
        self.setHorizontalHeaderLabels(
            ["Name", "Size", "Type", "Images", "Dimension", "Pixel Size", "Modified"]
        )
        self._root_path = root_path
        self._colors = _THEME_COLORS[_resolved_theme(_saved_theme())]
        self._file_infos: dict[str, tuple[str, str, str]] = {}
        self._folder_counts: dict[str, int] = {}
        self._sorting = False
        self._filter_pattern = "*"
        self._use_regex = False
        # Bumped on every directory (re)load; in-flight async info workers from
        # a previous load are discarded via this epoch so stale results never
        # land in the new model.
        self._epoch = 0
        # filepath -> the Name item of its row, for _row_for_filepath: an item knows its
        # own row, so nothing has to be rebuilt when rows are added or sorted
        self._name_items: dict[str, QStandardItem] = {}
        # a long folder is listed in slices; its sizes and dates come from threads
        self._loading = False
        self._pending: list[tuple[str, bool]] = []
        self._pending_base = Path(root_path)
        self._pending_pos = 0
        self._stat_items: dict[str, tuple] = {}
        self._stat_workers: list[_StatWorker] = []
        self._stats_running = 0
        self._cs_parents: dict[str, bool] = {}
        # the files and the folders of the folder shown, as the rows are made
        self._listed_files: list[str] = []
        self._listed_dirs: list[str] = []
        self._sort_state = (COL_NAME, Qt.SortOrder.AscendingOrder)
        self._listing_sort = self._sort_state
        self._load_directory(root_path)

    def set_theme_colors(self, colors: dict[str, str]) -> None:
        """Use the colours of a theme for the icons and the muted columns."""
        self._colors = colors
        if self.rowCount():
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, NUM_COLUMNS - 1),
                [
                    Qt.ItemDataRole.DecorationRole,
                    Qt.ItemDataRole.ForegroundRole,
                ],
            )

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        column = index.column()
        if role == Qt.ItemDataRole.DecorationRole and column == COL_NAME:
            return _themed_icon(super().data(index, ROLE_KIND) or "file", self._colors)
        if role == Qt.ItemDataRole.ForegroundRole and column != COL_NAME:
            return QBrush(QColor(self._colors["muted"]))
        if role == Qt.ItemDataRole.TextAlignmentRole and column in _RIGHT_COLUMNS:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return super().data(index, role)

    def _matches_filter(self, name: str) -> bool:
        if self._filter_pattern == "*" and not self._use_regex:
            return True  # (by far the most common; and 20,000 fnmatch calls take 0.2 s)
        return _name_matches_filter(name, self._filter_pattern, self._use_regex)

    def _load_directory(self, path: str) -> None:
        self._epoch += 1
        epoch = self._epoch
        self._cancel_loading()
        self._listing_sort = self._sort_state
        self._cs_parents = {}
        self._listed_files = []
        self._listed_dirs = []
        self._name_items = {}
        self.removeRows(0, self.rowCount())
        try:
            # a listing needs no stat of the entries (a DirEntry knows if it is a folder)
            with os.scandir(path) as listing:
                found = [(e.name, e.is_dir()) for e in listing]
        except OSError:
            self.load_finished.emit()
            return
        entries = [
            (name, is_dir)
            for name, is_dir in found
            if not name.startswith(".") and (is_dir or self._matches_filter(name))
        ]
        # in the order the list is sorted in, so a long folder needs no sort afterwards
        column, order = self._sort_state
        descending = order == Qt.SortOrder.DescendingOrder
        if column == COL_NAME:
            folders = sorted(
                (e for e in entries if e[1]),
                key=lambda e: _natural_sort_key(e[0] + "/"),
                reverse=descending,
            )
            files = sorted(
                (e for e in entries if not e[1]),
                key=lambda e: _natural_sort_key(e[0]),
                reverse=descending,
            )
            entries = folders + files
        else:
            entries.sort(key=lambda e: (not e[1], e[0].lower()))
        base = Path(path)
        if len(entries) <= SYNC_ROWS:
            for name, is_dir in entries:
                self.appendRow(self._make_row(base / name, is_dir=is_dir))
            self._finish_loading(epoch)
            return
        self._loading = True
        self._pending_base = base
        self._pending = entries  # (name, is_dir): a Path is made when the row is
        self._pending_pos = 0
        self._insert_more(epoch)

    def _insert_more(self, epoch: int) -> None:
        """Add the next rows of a long folder, for a slice of the main thread."""
        if epoch != self._epoch or not self._loading:
            return
        start = time.perf_counter()
        pending, pos, base = self._pending, self._pending_pos, self._pending_base
        while pos < len(pending):
            name, is_dir = pending[pos]
            pos += 1
            self.appendRow(self._make_row(base / name, is_dir=is_dir, light=True))
            if pos % 20 == 0 and time.perf_counter() - start > INSERT_SLICE_S:
                break
        self._pending_pos = pos
        if pos < len(pending):
            QTimer.singleShot(0, lambda e=epoch: self._insert_more(e))
        else:
            self._pending = []
            self._finish_loading(epoch)

    def _finish_loading(self, epoch: int) -> None:
        """All the rows are there: read the sizes and dates, and sort if the order asked for
        is not the one they were listed in."""
        self._loading = False
        column, order = self._sort_state
        if self._stat_items:
            self._start_stats(epoch)
        # by name the listing is already in order, unless another order was asked for while
        # it was being made; the order by size or date waits for the stats (_stats_finished)
        if column == COL_TYPE or (
            column == COL_NAME and self._sort_state != self._listing_sort
        ):
            self._do_sort(column, order)
        elif column in (COL_SIZE, COL_MODIFIED) and not self._stat_items:
            self._do_sort(column, order)
        self.load_finished.emit()

    def _cancel_loading(self) -> None:
        self._loading = False
        self._pending = []
        self._pending_pos = 0
        self._stat_items = {}
        for worker in self._stat_workers:
            worker.stop()
        self._stat_workers = [w for w in self._stat_workers if w.isRunning()]
        self._stats_running = 0

    def _start_stats(self, epoch: int) -> None:
        paths = list(self._stat_items)
        n = min(4, len(paths))
        self._stats_running = n
        for k in range(n):
            worker = _StatWorker(paths[k::n], epoch, self)
            worker.stats_ready.connect(self._apply_stats)
            worker.finished.connect(lambda e=epoch: self._stats_finished(e))
            self._stat_workers.append(worker)
            worker.start()

    def _apply_stats(self, epoch: int, batch: list) -> None:
        if epoch != self._epoch:
            return  # another folder was loaded since
        with self._batched_changes():
            for path, size, mtime in batch:
                items = self._stat_items.get(path)
                if items is None:
                    continue
                size_item, mod_item, is_dir = items
                if not is_dir:
                    size_item.setText(_format_size(size))
                    size_item.setData(size, ROLE_SORT)
                stamp = datetime.fromtimestamp(mtime) if mtime else None
                mod_item.setText(_format_modified(stamp) if stamp else "")
                mod_item.setData(stamp.isoformat() if stamp else "", ROLE_SORT)

    def _stats_finished(self, epoch: int) -> None:
        if epoch != self._epoch:
            return
        self._stats_running -= 1
        if self._stats_running <= 0:
            column, order = self._sort_state
            # the sizes and dates are there now: sort by them if that is the order
            if column in (COL_SIZE, COL_MODIFIED, COL_TYPE):
                self._do_sort(column, order)

    def set_filter(self, pattern: str, use_regex: bool = False) -> None:
        self._filter_pattern = pattern
        self._use_regex = use_regex
        self._load_directory(self._root_path)

    def _in_cryosparc_project(self, path: Path) -> bool:
        """Whether the folder holding ``path`` is a CryoSPARC project (read once per folder)."""
        parent = str(path.parent)
        known = self._cs_parents.get(parent)
        if known is None:
            try:
                known = (path.parent / "project.json").is_file()
            except OSError:
                known = False
            self._cs_parents[parent] = known
        return known

    def _make_row(
        self, path: Path, is_dir: bool | None = None, light: bool = False
    ) -> list[QStandardItem]:
        """The items of a row. ``light``: no stat of the file here (its size and date are
        filled in from a thread later, see :meth:`_start_stats`)."""
        if is_dir is None:
            is_dir = path.is_dir()
        name_item = QStandardItem(path.name + "/" if is_dir else path.name)
        name_item.setData(str(path), Qt.ItemDataRole.UserRole)
        (self._listed_dirs if is_dir else self._listed_files).append(str(path))
        self._name_items[str(path)] = name_item
        name_item.setEditable(False)

        if is_dir:
            cached_count = self._folder_counts.get(str(path))
            if cached_count is None:
                size_item = QStandardItem("")
                size_item.setData(0, ROLE_SORT)
            else:
                size_item = QStandardItem(_format_item_count(cached_count))
                size_item.setData(cached_count, ROLE_SORT)
            type_item = QStandardItem("Folder")
            type_item.setData("folder", ROLE_SORT)
            # a job folder of a CryoSPARC project says what job it is
            job = cryosparc_job_info(path) if self._in_cryosparc_project(path) else None
            if job is not None:
                type_item.setText(job["type_name"])
                tip = f"{job['uid']}  {job['title']}".rstrip()
                name_item.setToolTip(f"{tip}\n{job['raw_status'] or 'unknown'}")
            info_item = QStandardItem("")
            info_item.setData("", ROLE_SORT)
            images_item = QStandardItem("")
            images_item.setData(0, ROLE_SORT)
            apix_item = QStandardItem("")
            apix_item.setData("", ROLE_SORT)
            if light:
                mod_item = QStandardItem("")
                mod_item.setData("", ROLE_SORT)
                self._stat_items[str(path)] = (size_item, mod_item, True)
            else:
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime)
                except (OSError, ValueError):
                    mtime = datetime.now()
                mod_item = QStandardItem(_format_modified(mtime))
                mod_item.setData(mtime.isoformat(), ROLE_SORT)
        else:
            if light:
                size_item = QStandardItem("")
                size_item.setData(0, ROLE_SORT)
            else:
                try:
                    stat = path.stat()
                    size_bytes = stat.st_size
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                except (OSError, ValueError):
                    size_bytes = 0
                    mtime = datetime.now()
                size_item = QStandardItem(_format_size(size_bytes))
                size_item.setData(size_bytes, ROLE_SORT)

            type_label = _get_file_type_label(str(path))
            type_item = QStandardItem(type_label)
            type_item.setData(type_label.lower(), ROLE_SORT)

            # Dimension / Images / Pixel Size are filled asynchronously after
            # the directory is shown (see FolderBrowserWidget._populate_file_info_async)
            # so that opening a large folder stays responsive. They start empty.
            info_item = QStandardItem("")
            info_item.setData("", ROLE_SORT)

            images_item = QStandardItem("")
            images_item.setData(0, ROLE_SORT)

            apix_item = QStandardItem("")
            apix_item.setData(0, ROLE_SORT)
            if light:
                mod_item = QStandardItem("")
                mod_item.setData("", ROLE_SORT)
                self._stat_items[str(path)] = (size_item, mod_item, False)
            else:
                mod_item = QStandardItem(_format_modified(mtime))
                mod_item.setData(mtime.isoformat(), ROLE_SORT)

        name_item.setData(_icon_kind(type_item.text(), path.suffix, is_dir), ROLE_KIND)
        name_item.setDragEnabled(False)
        name_item.setDropEnabled(False)

        return [
            name_item,
            size_item,
            type_item,
            images_item,
            info_item,
            apix_item,
            mod_item,
        ]

    def _get_or_cache_info(self, filepath: str) -> tuple[str, str, str]:
        if filepath not in self._file_infos:
            self._file_infos[filepath] = _get_file_info(filepath)
        return self._file_infos[filepath]

    def current_epoch(self) -> int:
        """Return the epoch of the current directory load.

        Async info workers capture this value and only apply results whose
        epoch matches, so results from a superseded directory are ignored.
        """
        return self._epoch

    def listed_paths(self) -> tuple[list[str], list[str]]:
        """``(files, folders)`` of the folder shown, so far. No row is looked at: with
        20,000 rows, asking each for its path took 0.2 s of the main thread."""
        return list(self._listed_files), list(self._listed_dirs)

    def file_rows(self) -> list[tuple[int, str]]:
        """Return ``(row, filepath)`` pairs for every non-directory row.

        Used by the asynchronous info populator to know which rows still need
        their Dimension / Images / Pixel Size columns filled.
        """
        rows: list[tuple[int, str]] = []
        for row in range(self.rowCount()):
            name_index = self.index(row, COL_NAME)
            filepath = self.data(name_index, Qt.ItemDataRole.UserRole)
            if filepath and not self.is_dir(self.index(row, COL_NAME)):
                rows.append((row, filepath))
        return rows

    def dir_rows(self) -> list[tuple[int, str]]:
        """Return ``(row, filepath)`` pairs for every directory row.

        Used by the asynchronous count populator to know which rows still need
        their Size column filled with the number of visible entries.
        """
        rows: list[tuple[int, str]] = []
        for row in range(self.rowCount()):
            name_index = self.index(row, COL_NAME)
            filepath = self.data(name_index, Qt.ItemDataRole.UserRole)
            if filepath and self.is_dir(self.index(row, COL_NAME)):
                rows.append((row, filepath))
        return rows

    @contextmanager
    def _batched_changes(self):
        """Change many items without a ``dataChanged`` for each: one for all at the end.

        Every ``setText`` of an item of a model tells the views; for thousands of items
        that is what froze the window. Anything connected to the signals of the model hears
        only the one at the end."""
        was_blocked = self.blockSignals(True)
        try:
            yield
        finally:
            self.blockSignals(was_blocked)
            if self.rowCount():
                self.dataChanged.emit(
                    self.index(0, 0), self.index(self.rowCount() - 1, NUM_COLUMNS - 1)
                )

    def apply_file_infos(self, batch: list) -> None:
        """:meth:`apply_file_info` for a list of ``(filepath, info, n_images, apix)``."""
        with self._batched_changes():
            for filepath, info, n_images, pixel_size in batch:
                self.apply_file_info(filepath, info, n_images, pixel_size)

    def apply_dir_counts(self, batch: list) -> None:
        """:meth:`apply_dir_count` for a list of ``(filepath, n_items)``."""
        with self._batched_changes():
            for filepath, n_items in batch:
                self.apply_dir_count(filepath, n_items)

    def apply_dir_count(self, filepath: str, n_items: int) -> None:
        """Fill the Size column for ``filepath`` with its entry count.

        Resolves the row by searching for *filepath* in the Name column, so
        results remain correct even when the model has been re-sorted after
        the ``DirCountWorker`` was dispatched. Updates the sort role so sorting
        by the Size column orders folders by their entry count.

        Parameters
        ----------
        filepath : str
            Directory whose count is being applied.
        n_items : int
            Number of visible entries, or a negative value to leave the cell
            blank (used when the directory could not be read).
        """
        if not filepath or n_items < 0:
            return
        row = self._row_for_filepath(filepath)
        if row < 0 or row >= self.rowCount():
            return
        size_item = self.item(row, COL_SIZE)
        if size_item is None:
            return
        self._folder_counts[filepath] = n_items
        size_item.setText(_format_item_count(n_items))
        size_item.setData(n_items, ROLE_SORT)

    def apply_file_info(
        self, filepath: str, info: str, n_images: str, pixel_size: str
    ) -> None:
        """Fill the Dimension / Images / Pixel Size columns for ``filepath``.

        Resolves the row by searching for *filepath* in the Name column, so
        results remain correct even when the model has been re-sorted after
        the ``InfoWorker`` was dispatched. Updates the cached info and the
        per-column sort roles so subsequent sorting by those columns reflects
        the now-known values.
        """
        if not filepath:
            return
        self._file_infos[filepath] = (info, n_images, pixel_size)
        row = self._row_for_filepath(filepath)
        if row < 0 or row >= self.rowCount():
            return

        info_item = self.item(row, COL_INFO)
        if info_item is not None:
            info_item.setText(info)
            info_item.setData(info, ROLE_SORT)

        images_item = self.item(row, COL_IMAGES)
        if images_item is not None:
            images_item.setText(n_images)
            try:
                images_item.setData(int(n_images) if n_images else 0, ROLE_SORT)
            except ValueError:
                images_item.setData(0, ROLE_SORT)

        apix_item = self.item(row, COL_PIXELSIZE)
        if apix_item is not None:
            apix_item.setText(pixel_size)
            try:
                apix_item.setData(
                    float(pixel_size.split()[0]) if pixel_size else 0, ROLE_SORT
                )
            except (ValueError, IndexError):
                apix_item.setData(0, ROLE_SORT)

    def _row_for_filepath(self, filepath: str) -> int:
        """Return the model row containing *filepath*, or -1 if not found."""
        item = self._name_items.get(filepath)
        if item is None:
            return -1
        try:
            return item.row()
        except RuntimeError:  # the item is gone with its row
            return -1

    def refresh(self) -> None:
        self._file_infos.clear()
        self._load_directory(self._root_path)

    def set_root_path(self, path: str) -> None:
        self._root_path = path
        self.refresh()

    def file_path(self, index: QModelIndex) -> str | None:
        name_index = self.index(index.row(), COL_NAME)
        return self.data(name_index, Qt.ItemDataRole.UserRole)

    def is_dir(self, index: QModelIndex) -> bool:
        type_index = self.index(index.row(), COL_TYPE)
        return self.data(type_index, ROLE_SORT) == "folder"

    def sort(
        self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder
    ) -> None:
        self._sort_state = (column, order)
        if self._sorting or self._loading:
            return  # a long folder is sorted when it is all there (_finish_loading)
        self._sorting = True
        try:
            self._do_sort(column, order)
        finally:
            self._sorting = False

    def _do_sort(self, column: int, order: Qt.SortOrder) -> None:
        rows = []
        for i in range(self.rowCount()):
            row = [self.takeItem(i, c) for c in range(NUM_COLUMNS)]
            if all(item is not None for item in row):
                rows.append(row)
        self.removeRows(0, self.rowCount())

        type_col = COL_TYPE

        def sort_key(pair):
            cell = pair[column]
            data = (
                cell.data(ROLE_SORT)
                if cell.data(ROLE_SORT) is not None
                else (cell.text() or "")
            )
            if column == COL_NAME and isinstance(data, str):
                data = _natural_sort_key(data)
            is_dir = pair[type_col].data(ROLE_SORT) == "folder"
            return (0 if is_dir else 1, data)

        descending = order == Qt.SortOrder.DescendingOrder
        if column == COL_NAME:
            # by name the folders stay first, in either order
            folders = [r for r in rows if r[type_col].data(ROLE_SORT) == "folder"]
            files = [r for r in rows if r[type_col].data(ROLE_SORT) != "folder"]
            folders.sort(key=sort_key, reverse=descending)
            files.sort(key=sort_key, reverse=descending)
            rows = folders + files
        else:
            rows.sort(key=sort_key, reverse=descending)

        for row_items in rows:
            self.appendRow(row_items)


class InfoWorker(QThread):
    """Compute file metadata off the GUI thread.

    Each worker runs its own thread and is handed a slice of
    ``(row, filepath)`` pairs; it emits ``info_ready`` with the metadata it has resolved,
    a batch at a time (a signal for each of 20,000 files would flood the main thread's
    event queue). The blocking file reads (mrcfile / PIL / starfile / PDF) therefore never
    stall the UI. ``finished`` is emitted once the slice is exhausted so the owner can
    clean up and re-sort if needed.
    """

    # a list of (filepath, info, n_images, apix), and the epoch
    info_ready = Signal(list, int)
    BATCH = 50
    INTERVAL_S = 0.1

    def __init__(self, tasks: list[tuple[int, str]], epoch: int) -> None:
        super().__init__()
        self._tasks = tasks
        self._epoch = epoch

    def run(self) -> None:
        batch, sent = [], time.monotonic()
        for _row, filepath in self._tasks:
            info, n_images, pixel_size = _get_file_info(filepath)
            if info or n_images or pixel_size:  # nothing to show for the rest
                batch.append((filepath, info, n_images, pixel_size))
            if batch and (
                len(batch) >= self.BATCH or time.monotonic() - sent > self.INTERVAL_S
            ):
                self.info_ready.emit(batch, self._epoch)
                batch, sent = [], time.monotonic()
        if batch:
            self.info_ready.emit(batch, self._epoch)


class DirCountWorker(QThread):
    """Count directory entries off the GUI thread.

    Each worker runs its own thread and is handed a slice of
    ``(row, filepath)`` pairs; it emits ``count_ready`` with the counts it has resolved,
    a batch at a time. Counting is a plain ``readdir`` per directory, so it
    never stalls the UI even for folders holding many subdirectories.
    ``finished`` is emitted once the slice is exhausted so the owner can clean
    up and re-sort if needed.
    """

    count_ready = Signal(list, int)  # a list of (filepath, n_items), and the epoch
    BATCH = 50
    INTERVAL_S = 0.1

    def __init__(
        self,
        tasks: list[tuple[int, str]],
        epoch: int,
        filter_pattern: str,
        use_regex: bool,
    ) -> None:
        super().__init__()
        self._tasks = tasks
        self._epoch = epoch
        self._filter_pattern = filter_pattern
        self._use_regex = use_regex

    def run(self) -> None:
        batch, sent = [], time.monotonic()
        for _row, filepath in self._tasks:
            n_items = _count_dir_items(
                Path(filepath), self._filter_pattern, self._use_regex
            )
            if n_items is not None:
                batch.append((filepath, n_items))
            if batch and (
                len(batch) >= self.BATCH or time.monotonic() - sent > self.INTERVAL_S
            ):
                self.count_ready.emit(batch, self._epoch)
                batch, sent = [], time.monotonic()
        if batch:
            self.count_ready.emit(batch, self._epoch)


class _ModesWorker(QThread):
    """Works out, off the main thread, what can be done with a file: the display modes
    (which read headers and datasets), the label of the slice button, and whether
    ChimeraX is there. ``ready(serial, info)`` at the end."""

    ready = Signal(int, dict)

    def __init__(self, serial: int, path: str, browser, parent=None):
        super().__init__(parent)
        self._serial = serial
        self._path = path
        self._browser = browser

    def run(self) -> None:
        path = self._path
        info = dict(path=path, modes=[], slice_text="Image Slice", chimerax=True)
        try:
            modes = self._browser._display_modes_for(path)
            info["modes"] = modes
            if modes:
                suffix = Path(path).suffix.lower()
                if suffix == ".pdf":
                    info["slice_text"] = "PDF"
                elif suffix == ".eps":
                    info["slice_text"] = "EPS"
                elif "volume" in modes:
                    info["slice_text"] = "2D Slice"
                elif self._browser._is_image_stack(path):
                    info["slice_text"] = "2D Image"
                if "chimerax" in modes:
                    info["chimerax"] = _find_chimerax() is not None
        except Exception:
            info["modes"] = []  # a file that cannot be read gets no buttons
        self.ready.emit(self._serial, info)


# the icon (Font Awesome, drawn by qtawesome) and the category of each action button.
# "viewer" shows the file in a window of the browser; "app" opens a web app or a panel
# of tools, and is outlined in the accent colour.
_ACTION_STYLE = {
    "slice": ("fa5s.image", "viewer"),
    "volume": ("fa5s.cube", "viewer"),
    "3dplot": ("fa5s.project-diagram", "viewer"),
    "chimerax": ("fa5s.atom", "viewer"),
    "stats": ("fa5s.chart-bar", "viewer"),
    "text": ("fa5s.file-alt", "viewer"),
    "gallery": ("fa5s.th", "viewer"),
    "optimiser": ("fa5s.layer-group", "viewer"),
    "2dclasses": ("fa5s.th-large", "viewer"),
    "orthogonal": ("fa5s.cubes", "viewer"),
    "fsc": ("fa5s.chart-line", "viewer"),
    "html": ("fa5s.globe", "viewer"),
    "denovo3D": ("fa5s.dna", "app"),
    "abInitio3D": ("fa5s.shapes", "app"),
    "whereIsMyClass": ("fa5s.map-marker-alt", "app"),
    "helicalProjection": ("fa5s.film", "app"),
    "helicalPitch": ("fa5s.ruler-horizontal", "app"),
    "hill": ("fa5s.wave-square", "app"),
    "hi3d": ("fa5s.circle-notch", "app"),
    "proc3d": ("fa5s.sliders-h", "app"),
    "trueFSC": ("fa5s.chart-line", "app"),
    "images2star": ("fa5s.exchange-alt", "app"),
}


class FolderBrowserWidget(QMainWindow):
    """A folder browser widget for selecting image and volume files.

    Displays a tree view of files with columns for name, size, type,
    info, images, and modification date. Double-click a file to emit
    the ``file_selected`` signal.

    Parameters
    ----------
    start_dir : str or Path, optional
        Initial directory to display. Defaults to the current working directory.

    Signals
    -------
    file_selected(str)
        Emitted when a file is double-clicked. The value is the full file path.
    file_selected_new_window(str)
        Emitted when a file is Shift+double-clicked. The value is the full file path.
    """

    file_selected = Signal(str)
    file_selected_new_window = Signal(str)
    # (path, mode, new_window) where mode is a display-mode string from _DISPLAY_BUTTONS.
    display_requested = Signal(str, str, bool)

    # Maps QPushButton attribute name → display-mode string. Drives button
    # construction, click wiring, per-mode visibility, and alphabetical
    # reordering in ``_reorder_buttons_alphabetically``.
    _DISPLAY_BUTTONS: list[tuple[str, str]] = [
        ("_btn_slice", "slice"),
        ("_btn_volume", "volume"),
        ("_btn_3dplot", "3dplot"),
        ("_btn_chimerax", "chimerax"),
        ("_btn_stats", "stats"),
        ("_btn_text", "text"),
        ("_btn_gallery", "gallery"),
        ("_btn_optimiser", "optimiser"),
        ("_btn_2dclasses", "2dclasses"),
        ("_btn_orthogonal", "orthogonal"),
        ("_btn_proc3d", "proc3d"),
        ("_btn_fsc", "fsc"),
        ("_btn_html", "html"),
        ("_btn_denovo3d", "denovo3D"),
        ("_btn_abinitio3d", "abInitio3D"),
        ("_btn_whereismyclass", "whereIsMyClass"),
        ("_btn_helicalprojection", "helicalProjection"),
        ("_btn_helicalpitch", "helicalPitch"),
        ("_btn_hill", "hill"),
        ("_btn_hi3d", "hi3d"),
        ("_btn_truefsc", "trueFSC"),
        ("_btn_images2star", "images2star"),
    ]

    def __init__(self, start_dir: str | Path | None = None, parent=None):
        super().__init__(parent)

        if start_dir is None:
            start_dir = os.getcwd()
        start_dir = str(Path(start_dir).resolve())

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._menu_bar = self.menuBar()
        self._file_menu = self._menu_bar.addMenu("File")
        self._bookmarks_menu = self._menu_bar.addMenu("Bookmarks")
        self._apps_menu = self._menu_bar.addMenu("Apps")

        self._open_folder_action = QAction("Open Folder…", self)
        self._open_folder_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_folder_action.triggered.connect(self._open_folder)
        self._file_menu.addAction(self._open_folder_action)

        # Terminal is the first app, kept on the Apps menu where every other
        # standalone tool lives rather than buried in the File menu.
        self._open_terminal_action = QAction("Terminal", self)
        self._open_terminal_action.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self._open_terminal_action.triggered.connect(self._open_terminal)
        self._apps_menu.addAction(self._open_terminal_action)

        self._configure_relion_action = QAction("Configure RELION…", self)
        # On macOS, Qt moves an action whose text contains "config" (or
        # "settings", "options", ...) into the application menu as
        # "Preferences…", so it would vanish from this menu.
        self._configure_relion_action.setMenuRole(QAction.MenuRole.NoRole)
        self._configure_relion_action.triggered.connect(self._configure_relion)
        self._apps_menu.addAction(self._configure_relion_action)
        self._relion_processes = {}
        self._relion_timer = QTimer(self)
        self._relion_timer.setInterval(1000)
        self._relion_timer.timeout.connect(self._poll_relion)

        self._app_actions = {}
        for label, key, streamlit_mod, picker in _APP_LAUNCH_TABLE:
            action = QAction(label, self)
            action.setData((key, streamlit_mod, picker))
            action.triggered.connect(self._on_launch_app)
            self._apps_menu.addAction(action)
            self._app_actions[label] = action

        self._bookmark_toggle_action = QAction("Bookmark This Folder", self)
        self._bookmark_toggle_action.setShortcut(QKeySequence("Ctrl+D"))
        self._bookmark_toggle_action.triggered.connect(self._toggle_bookmark)
        # filled each time the menu opens (the model does not exist yet); the
        # window owns the action, so Ctrl+D works with the menu closed
        self._bookmarks_menu.aboutToShow.connect(self._refresh_bookmarks_menu)
        self.addAction(self._bookmark_toggle_action)

        self._recent_menu = self._file_menu.addMenu("Recent Folders")
        self._recent_menu.aboutToShow.connect(self._refresh_recent_menu)
        self._refresh_recent_menu()

        self._file_menu.addSeparator()
        self._quit_action = QAction("Quit", self)
        self._quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self._quit_action.triggered.connect(self._quit)
        self._file_menu.addAction(self._quit_action)

        self._view_menu = self._menu_bar.addMenu("View")

        self._theme_menu = self._view_menu.addMenu("Theme")
        self._theme_actions = {}
        self._theme_action_group = QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        for theme in _THEMES:
            action = QAction(theme, self)
            action.setCheckable(True)
            action.setData(theme)
            action.triggered.connect(self._on_theme_action_triggered)
            self._theme_action_group.addAction(action)
            self._theme_menu.addAction(action)
            self._theme_actions[theme] = action

        self._view_menu.addSeparator()
        self._sidebar_action = QAction("Sidebar", self)
        self._sidebar_action.setCheckable(True)
        self._sidebar_action.setShortcut(QKeySequence("F9"))
        self._sidebar_action.toggled.connect(self._on_sidebar_toggled)
        self._view_menu.addAction(self._sidebar_action)
        self._preview_action = QAction("Preview Pane", self)
        self._preview_action.setCheckable(True)
        self._preview_action.setShortcut(QKeySequence("F3"))
        self._preview_action.toggled.connect(self._on_preview_toggled)
        self._view_menu.addAction(self._preview_action)
        self._location_action = QAction("Edit Location", self)
        self._location_action.setShortcut(QKeySequence("Ctrl+L"))
        self._location_action.triggered.connect(lambda: self._path_edit.edit())
        self._view_menu.addAction(self._location_action)

        self._help_menu = self._menu_bar.addMenu("Help")
        self._home_page_action = QAction("Home Page", self)
        self._home_page_action.triggered.connect(self._open_home_page)
        self._help_menu.addAction(self._home_page_action)
        self._docs_action = QAction("Documentation", self)
        self._docs_action.triggered.connect(self._open_documentation)
        self._help_menu.addAction(self._docs_action)
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(4, 4, 4, 4)
        nav_layout.setSpacing(4)

        self._back_btn = QToolButton()
        self._back_btn.setText("←")
        self._back_btn.setToolTip("Go back")
        self._back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self._back_btn)

        self._up_btn = QToolButton()
        self._up_btn.setText("↑")
        self._up_btn.setToolTip("Go up")
        self._up_btn.clicked.connect(self._go_up)
        nav_layout.addWidget(self._up_btn)

        self._refresh_btn = QToolButton()
        self._refresh_btn.setText("↻")
        self._refresh_btn.setToolTip("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)
        nav_layout.addWidget(self._refresh_btn)

        self._path_edit = PathBar(start_dir)
        self._path_edit.returnPressed.connect(self._go_to_path)
        self._path_edit.pathClicked.connect(self._navigate_to)
        # Stretch factor 1 makes the path box expand to fill all space up to
        # the filter controls at the right end of the row.
        nav_layout.addWidget(self._path_edit, 1)

        self._relion_btn = QPushButton("Open RELION")
        self._relion_btn.setToolTip(
            "Open this project in RELION; setup: Apps → Configure RELION…"
        )
        self._relion_btn.clicked.connect(self._open_relion)
        nav_layout.addWidget(self._relion_btn)

        self._recent_combo = QComboBox()
        self._recent_combo.setPlaceholderText("Recent")
        self._recent_combo.setToolTip("Recent starting folders")
        self._recent_combo.setFixedWidth(80)
        self._recent_combo.setStyleSheet(
            "QComboBox { color: #cccccc; background: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 3px; }"
            "QComboBox QAbstractItemView { color: #cccccc; background: #2d2d2d; }"
        )
        _add_recent_folder(start_dir)
        self._refresh_recent_combo()
        self._recent_combo.activated.connect(self._on_recent_selected)
        self._recent_combo.hide()

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(_THEMES)
        self._theme_combo.setCurrentText(_saved_theme())
        self._theme_combo.setToolTip("Choose the file browser and launch-button theme")
        self._theme_combo.setFixedWidth(90)
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self._theme_combo.hide()

        # Filter controls at the right end of the navigation row, next to the
        # current-folder input box.

        filter_label = QLabel("Filter:")
        nav_layout.addWidget(filter_label)

        self._filter_edit = QLineEdit("*")
        self._filter_edit.setPlaceholderText("e.g. *.mrc, *.tif")
        self._filter_edit.returnPressed.connect(self._apply_filter)
        nav_layout.addWidget(self._filter_edit)

        self._regex_cb = QCheckBox("Regex")
        self._regex_cb.stateChanged.connect(self._apply_filter)
        nav_layout.addWidget(self._regex_cb)

        layout.addLayout(nav_layout)

        self._model = FileBrowserModel(start_dir)

        # Active background info-population threads. Cleared (and quit) whenever
        # the directory is reloaded so superseded workers don't write to the
        # new model.
        self._info_threads: list[QThread] = []

        self._history = [start_dir]
        self._history_index = 0

        self._model.load_finished.connect(self._populate_file_info_async)
        self._tree = QTreeView()
        self._tree.setModel(self._model)
        # all rows are as tall as the first: the view need not measure a row to place it,
        # which is what made a folder of tens of thousands of files slow to scroll
        self._tree.setUniformRowHeights(True)
        self._tree.setRootIndex(QModelIndex())

        self._tree.setSortingEnabled(True)
        self._tree.sortByColumn(COL_NAME, Qt.SortOrder.AscendingOrder)

        # Restore saved sort state.
        self._restore_sort_state()

        self._tree.setSelectionBehavior(QTreeView.SelectionBehavior.SelectItems)
        self._tree.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)

        copy_sc = QShortcut(QKeySequence.StandardKey.Copy, self._tree)
        copy_sc.activated.connect(self._copy_selection)

        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)

        header = self._tree.header()
        header.setStretchLastSection(True)
        header.setSectionsMovable(False)
        for col in range(NUM_COLUMNS):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(COL_NAME, 280)
        header.resizeSection(COL_SIZE, 80)
        header.resizeSection(COL_TYPE, 130)
        header.resizeSection(COL_IMAGES, 70)
        header.resizeSection(COL_INFO, 100)
        header.resizeSection(COL_PIXELSIZE, 90)
        header.resizeSection(COL_MODIFIED, 140)

        # Restore previously saved column widths (persisted between launches).
        # Done before connecting sectionResized so the initial programmatic
        # resizes above don't trigger a redundant save.
        self._restore_col_widths()

        # Persist column widths as the user drags dividers (debounced).
        self._col_save_timer = QTimer(self)
        self._col_save_timer.setSingleShot(True)
        self._col_save_timer.timeout.connect(self._save_col_widths)
        header.sectionResized.connect(lambda *a: self._col_save_timer.start(300))
        header.sortIndicatorChanged.connect(self._save_sort_state)

        self._tree.setAnimated(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setIconSize(QSize(18, 18))
        self._tree.setIndentation(20)
        self._tree.setExpandsOnDoubleClick(False)

        self._tree.setStyleSheet(
            """
            QTreeView {
                font-size: 12px;
                background-color: #2d2d2d;
                color: #cccccc;
                border: none;
            }
            QTreeView::item:selected {
                background-color: #4a6fa5;
            }
            QTreeView::item:hover {
                background-color: #3a3a3a;
            }
            QHeaderView::section {
                padding: 4px;
                font-weight: bold;
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #2d2d2d;
            }
            """
        )

        self.setStyleSheet(
            """
            QWidget {
                background-color: #2d2d2d;
                color: #cccccc;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
            }
            QToolButton {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QToolButton:hover {
                background-color: #4a6fa5;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #4a6fa5;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #5a82c4;
                border: 1px solid #6f9bd6;
            }
            QCheckBox {
                color: #e8e8e8;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #888888;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:unchecked {
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #4a6fa5;
                border: 1px solid #6f9bd6;
                image: none;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #5a82c4;
            }
            """
        )

        self._sidebar = PlacesSidebar()
        self._sidebar.navigate.connect(self._navigate_to)
        self._sidebar.bookmark_removed.connect(self._on_bookmark_removed)
        self._preview = PreviewPane()
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self._sidebar)
        self._splitter.addWidget(self._tree)
        self._splitter.addWidget(self._preview)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([180, 760, 300])
        layout.addWidget(self._splitter, 1)
        self._restore_panes()

        # Bottom bar: contextual display-mode buttons + "new window" toggle.
        # Hidden unless exactly one file (not a directory) is selected.
        self._action_bar = QWidget()
        self._action_bar.setObjectName("actionBar")
        # the buttons wrap to a second row when the window is narrow
        action_layout = FlowLayout(self._action_bar, margin=8, h_spacing=6, v_spacing=6)

        self._btn_slice = QPushButton("Image Slice")
        self._btn_volume = QPushButton("3D Volume")
        self._btn_3dplot = QPushButton("3D Plot")
        self._btn_3dplot.setToolTip("Display 3D coordinates from a .bild file")
        self._btn_chimerax = QPushButton("ChimeraX")
        self._btn_chimerax.setToolTip("Open this file in ChimeraX")
        self._btn_stats = QPushButton("Stats")
        self._btn_text = QPushButton("Text")
        self._btn_text.setToolTip("Open this file as text")
        self._btn_gallery = QPushButton("Gallery")
        self._btn_gallery.setToolTip("Show a lazy thumbnail grid of the stack")
        self._btn_optimiser = QPushButton("XYZ Slice")
        self._btn_optimiser.setToolTip(
            "Show center slices (Z, Y, X) of referenced MRC maps"
        )
        self._btn_2dclasses = QPushButton("2D Classes")
        self._btn_2dclasses.setToolTip("Show 2D class averages sorted by abundance")
        self._btn_orthogonal = QPushButton("Ortho Slice")
        self._btn_orthogonal.setToolTip("Interactive XYZ slice viewer for 3D volumes")
        self._btn_proc3d = QPushButton("Proc3D")
        self._btn_proc3d.setToolTip(
            "Open the Proc3D tools panel to preview and transform "
            "this 3D map (flip hand, clip, resample, ...), comparing the "
            "result side by side with the original, then save a new MRC file"
        )
        self._btn_fsc = QPushButton("FSC")
        self._btn_fsc.setToolTip(
            "Display the FSC curve, or the SSNR-derived W_MAP curve for Class3D, "
            "from model.star. Pick which refinement iterations to overlay when "
            "the job directory holds more than one model.star file."
        )
        self._btn_html = QPushButton("Browser")
        self._btn_html.setToolTip("Open this file in your web browser")
        self._btn_denovo3d = QPushButton("denovo3D")
        self._btn_denovo3d.setToolTip(
            "Open this file in the denovo3D web app for de novo helical indexing"
        )
        self._btn_abinitio3d = QPushButton("AbInitio3D")
        self._btn_abinitio3d.setToolTip(
            "Open this Class2D job in the AbInitio3D web app for an ab initio "
            "3D map from the 2D classes"
        )
        self._btn_whereismyclass = QPushButton("WhereIsMyClass")
        self._btn_whereismyclass.setToolTip(
            "Open this file in the WhereIsMyClass web app for mapping 2D classes"
        )
        self._btn_helicalprojection = QPushButton("HelicalProjection")
        self._btn_helicalprojection.setToolTip(
            "Open this file in the HelicalProjection web app"
        )
        self._btn_helicalpitch = QPushButton("HelicalPitch")
        self._btn_helicalpitch.setToolTip(
            "Open this file in the HelicalPitch web app for determining helical pitch/twist"
        )
        self._btn_hill = QPushButton("HILL")
        self._btn_hill.setToolTip(
            "Open this file in the HILL web app for helical indexing via Fourier layer lines"
        )
        self._btn_hi3d = QPushButton("HI3D")
        self._btn_hi3d.setToolTip(
            "Open this file in the HI3D web app for helical indexing via cylindrical projection"
        )
        self._btn_truefsc = QPushButton("TrueFSC")
        self._btn_truefsc.setToolTip("Compute True FSC curve from the two half-maps")
        self._btn_images2star = QPushButton("Images2Star")
        self._btn_images2star.setToolTip(
            "Open the Images2Star tools panel to preview and transform "
            "this dataset, then save a new RELION STAR/CryoSPARC file"
        )
        self._new_window_cb = QCheckBox("New window")
        self._new_window_cb.setToolTip(
            "<div style='white-space: normal; width: 200px;'>Create a new display window instead of reusing the current one. Most recently clicked/focused display window will be used as the display target if there are two or more display windows</div>"
        )
        for attr, mode in self._DISPLAY_BUTTONS:
            btn = getattr(self, attr)
            btn.setFixedHeight(30)
            btn.setIconSize(QSize(16, 16))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            # the web apps and the panels that open apart are marked from the viewers
            btn.setProperty("category", _ACTION_STYLE.get(mode, ("", "viewer"))[1])
            action_layout.addWidget(btn)
        action_layout.addWidget(self._new_window_cb)
        self._action_bar.hide()

        self._apply_theme(self._theme_combo.currentText())

        for attr, mode in self._DISPLAY_BUTTONS:
            btn = getattr(self, attr)
            btn.clicked.connect(partial(self._emit_display, mode))

        layout.addWidget(self._action_bar)

        # the buttons of the selected file are worked out in a thread (_ModesWorker)
        self._modes_worker: _ModesWorker | None = None
        self._modes_pending: str | None = None
        self._modes_serial = 0
        self._modes_applied = 0
        sel_model = self._tree.selectionModel()
        sel_model.selectionChanged.connect(self._on_selection_changed)

        self._tree.doubleClicked.connect(self._on_double_clicked)
        self._tree.clicked.connect(self._on_clicked)

        # Disabled widgets do not receive hover events, so Qt will not show
        # the ChimeraX "not found" tooltip on its own. This filter surfaces it
        # manually when the cursor is over the (disabled) button.
        self.installEventFilter(self)

        self._refresh_sidebar()
        self._update_preview()

        # Fill Dimension / Images / Pixel Size for the initial directory in
        # the background so the browser opens instantly (these columns start
        # empty and are populated by worker threads).
        self._populate_file_info_async()

    def _restore_panes(self) -> None:
        """Restore which side panes are shown, and how wide, from the last launch."""
        settings = QSettings("helicon", "display")
        for action, key in (
            (self._sidebar_action, "browser_show_sidebar"),
            (self._preview_action, "browser_show_preview"),
        ):
            action.setChecked(settings.value(key, True, type=bool))
        sizes = settings.value("browser_pane_sizes", [], type=list)
        try:
            sizes = [int(x) for x in sizes]
        except (TypeError, ValueError):
            sizes = []
        if len(sizes) == 3 and sizes[1] > 0:
            self._splitter.setSizes(sizes)

    def _save_panes(self) -> None:
        settings = QSettings("helicon", "display")
        settings.setValue("browser_show_sidebar", self._sidebar_action.isChecked())
        settings.setValue("browser_show_preview", self._preview_action.isChecked())
        sizes = self._splitter.sizes()
        # a hidden pane has no width; keep the last one it had
        if all(sizes[i] > 0 for i in (0, 2)):
            settings.setValue("browser_pane_sizes", sizes)

    def _on_sidebar_toggled(self, shown: bool) -> None:
        self._sidebar.setVisible(shown)

    def _on_preview_toggled(self, shown: bool) -> None:
        self._preview.setVisible(shown)
        if shown:
            self._update_preview()

    def _refresh_sidebar(self) -> None:
        self._sidebar.refresh(
            self._model._root_path,
            bookmarks=_get_bookmarks(),
            recent=_get_recent_folders(),
        )

    def _update_preview(self) -> None:
        """Preview the selected file or folder, or else the folder shown."""
        if not self._preview_action.isChecked():
            return
        indexes = self._tree.selectionModel().selectedIndexes()
        rows = sorted({idx.row() for idx in indexes})
        path = None
        if len(rows) == 1:
            path = self._model.file_path(self._model.index(rows[0], 0))
        self._preview.show_path(str(path) if path else self._model._root_path)

    def _set_action_icons(self, colors: dict[str, str]) -> None:
        """Icons on the action buttons, in the colour of the text (white when hovered)
        and of the accent for the apps. None where qtawesome is not installed."""
        try:
            import qtawesome
        except ImportError:
            return
        for attr, mode in self._DISPLAY_BUTTONS:
            name, category = _ACTION_STYLE.get(mode, ("fa5s.file", "viewer"))
            color = colors["app_text"] if category == "app" else colors["text"]
            getattr(self, attr).setIcon(
                qtawesome.icon(
                    name,
                    color=color,
                    color_active="#ffffff",
                    color_disabled=colors["disabled"],
                )
            )

    def _set_nav_icons(self, colors: dict[str, str]) -> None:
        """Icons for back, up and refresh in the colour of the theme (the arrows of the
        text stay where qtawesome is not installed)."""
        try:
            import qtawesome
        except ImportError:
            return
        for button, name in (
            (self._back_btn, "fa5s.arrow-left"),
            (self._up_btn, "fa5s.arrow-up"),
            (self._refresh_btn, "fa5s.sync-alt"),
        ):
            button.setIcon(qtawesome.icon(name, color=colors["text"]))
            button.setText("")

    def _on_theme_changed(self, theme: str) -> None:
        """Persist and immediately apply a newly selected browser theme."""
        _save_theme(theme)
        self._apply_theme(theme)
        try:
            from helicon.commands.display import _refresh_display_theme_windows

            _refresh_display_theme_windows()
        except Exception:
            # Keep the browser usable when auxiliary display dependencies are
            # unavailable or display.py is imported without the full Qt stack.
            pass

    def _on_theme_action_triggered(self, checked: bool = False) -> None:
        """Apply the theme selected from the View menu."""
        action = self.sender()
        if isinstance(action, QAction):
            theme = str(action.data())
            self._theme_combo.setCurrentText(theme)

    def _apply_theme(self, theme: str) -> None:
        """Apply a theme to the file browser and its launch buttons."""
        colors = _THEME_COLORS[_resolved_theme(theme)]
        stylesheet = _browser_stylesheet(colors)
        self.setStyleSheet(stylesheet)
        self._tree.setStyleSheet(stylesheet)
        self._recent_combo.setStyleSheet(stylesheet)
        self._theme_combo.setStyleSheet(stylesheet)
        self._model.set_theme_colors(colors)
        self._sidebar.set_theme(colors)
        self._preview.set_theme(colors)
        self._set_nav_icons(colors)
        self._set_action_icons(colors)
        for name, action in self._theme_actions.items():
            action.setChecked(name == theme)
        for attr, _mode in self._DISPLAY_BUTTONS:
            getattr(self, attr).setStyleSheet("")

    def _on_double_clicked(self, index: QModelIndex) -> None:
        path = self._model.file_path(index)
        if path and not self._model.is_dir(index):
            if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.file_selected_new_window.emit(path)
            else:
                self.file_selected.emit(path)

    def _on_clicked(self, index: QModelIndex) -> None:
        if self._model.is_dir(index):
            path = self._model.file_path(index)
            if path:
                self._navigate_to(path)

    def _display_modes_for(self, path: str) -> list[str]:
        """Return the applicable display modes for a single file.

        Detection order mirrors _get_file_type_label:
        1. Known file types by suffix
        2. Pure-text fallback via UTF-8 decode -> ["general"]
        3. Unknown binary -> []
        """
        from pathlib import Path

        try:
            if Path(path).stat().st_size == 0:
                return []
        except OSError:
            return []

        ext = Path(path).suffix.lower()
        if ext == ".star":
            name = Path(path).name.lower()
            if name.endswith("optimiser.star") or name.endswith("model.star"):
                _is_class2d = any(p.startswith("Class2D") for p in Path(path).parts)
                if _is_class2d:
                    return ["text", "2dclasses"]
                modes = ["text", "optimiser"]
                if name.endswith("model.star"):
                    modes.append("fsc")
                    _is_refine3d = any(
                        p.startswith("Refine3D") for p in Path(path).parts
                    )
                    if _is_refine3d:
                        modes.append("trueFSC")
                return modes
            if name.endswith("data.star"):
                _is_class2d = any(p.startswith("Class2D") for p in Path(path).parts)
                if _is_class2d:
                    # WhereIsMyClass, HelicalPitch and AbInitio3D are helical
                    # web apps and only make sense for helical 2D class averages.
                    modes = ["slice", "gallery", "text", "images2star"]
                    if _folder_is_helical(str(Path(path).parent)):
                        modes.extend(["whereIsMyClass", "helicalPitch", "abInitio3D"])
                    return modes
                _is_class3d_or_refine3d = any(
                    p.startswith("Class3D") or p.startswith("Refine3D")
                    for p in Path(path).parts
                )
                if _is_class3d_or_refine3d:
                    return ["slice", "gallery", "stats", "text", "images2star"]
            if any(name.endswith(s) for s in _METADATA_STAR_SUFFIXES):
                return ["text"]
            return ["slice", "gallery", "text", "images2star"]
        if ext == ".cs":
            # A CryoSPARC dataset that names images is an image stack, the way
            # a RELION data.star is: ``blob/path`` and ``blob/idx`` say where
            # each one lives. Those get the stack actions; a passthrough
            # dataset carries inherited parameters and no images, and a
            # volume's dataset points at a map, so both keep to images2star.
            from helicon.lib import cryosparc_project

            modes = ["images2star"]
            if cryosparc_project.has_image_refs(path):
                modes = ["slice", "gallery"] + modes
                if cryosparc_project.is_helical(path):
                    modes.append("denovo3D")
            return modes
        if ext == ".mrcs":
            modes = ["slice", "gallery"]
            _is_class2d = any(p.startswith("Class2D") for p in Path(path).parts)
            _helical = _folder_is_helical(str(Path(path).parent))
            if _is_class2d and _helical:
                modes.extend(["helicalProjection", "hill"])
            if _helical:
                modes.append("denovo3D")
            return modes
        if ext in _MRC_VOLUME_EXTENSIONS:
            _has_volume = self._mrc_is_volume(path)
            if _has_volume:
                modes = [
                    "slice",
                    "volume",
                    "gallery",
                    "chimerax",
                    "orthogonal",
                    "proc3d",
                ]
            else:
                # Single-slice (nz == 1) or unreadable .mrc/.map files are
                # treated as 2D image stacks, not 3D volumes -- as are the
                # many-image stacks CryoSPARC writes with an .mrc suffix.
                modes = ["slice", "gallery"]
                from helicon.lib import cryosparc_project

                if cryosparc_project.is_class_averages(path):
                    # Sorting by abundance needs the particles dataset that
                    # records each particle's class; offered only when the
                    # companion files are actually there.
                    if cryosparc_project.has_class_abundance(path):
                        modes.append("2dclasses")
                    if cryosparc_project.is_helical(path):
                        modes.extend(["helicalProjection", "hill", "denovo3D"])
                        # it needs the particles dataset as well as the averages
                        if cryosparc_project.has_class_abundance(path):
                            modes.append("abInitio3D")
            if _has_volume:
                _is_class3d_or_refine3d = any(
                    p.startswith("Class3D") or p.startswith("Refine3D")
                    for p in Path(path).parts
                )
                if _is_class3d_or_refine3d and _folder_is_helical(
                    str(Path(path).parent)
                ):
                    modes.append("hi3d")
                else:
                    from helicon.lib import cryosparc_project

                    if cryosparc_project.is_helical(path):
                        modes.append("hi3d")
            return modes
        if ext == ".bild":
            return ["text", "3dplot", "chimerax"]
        if ext == ".pdf":
            return ["slice"]
        if ext == ".eps":
            return ["slice"]
        if ext in _RASTER_IMAGE_EXTENSIONS:
            return ["slice"]
        if ext in (".html", ".htm"):
            return ["html"]
        if ext in _KNOWN_EXTENSIONS:
            return []

        if _is_text_file(path):
            return ["text"]

        return []

    def _mrc_is_volume(self, path: str) -> bool:
        """Whether an MRC-format file holds a 3D volume rather than a stack.

        The header alone cannot say. CryoSPARC writes everything as ``.mrc``,
        so a stack of 100 class averages has nz=100 and looks exactly like a
        map -- one such stack measured 200 x 200 x 200, which no rule about
        dimensions could ever call correctly. Where the file sits in a
        CryoSPARC project, its job type and CryoSPARC's own naming say which
        it is; everywhere else the Z extent is still the best signal there is.
        """
        from helicon.lib import cryosparc_project

        kind = cryosparc_project.mrc_content(path)
        if kind == "volume":
            return True
        if kind == "stack":
            return False
        return self._volume_has_nz_gt1(path)

    def _volume_has_nz_gt1(self, path: str) -> bool:
        """Return True if an MRC-format file has more than one Z slice."""
        try:
            import mrcfile

            # Read only the header: the data block can be gigabytes.
            with mrcfile.open(path, permissive=True, header_only=True) as mrc:
                header = mrc.header
                nz = int(header.nz)
                mz = int(header.mz)
                ispg = int(header.ispg)
            # A volume stack (space group 401-630 with several volumes) is
            # 4D, not a single volume.
            if 401 <= ispg <= 630 and 0 < mz < nz:
                return False
            return nz > 1
        except Exception:
            return False

    def _is_image_stack(self, path: str) -> bool:
        """Return True for files that are stacks of 2D images.

        This covers ``.mrcs`` particle stacks, data ``.star`` files (which
        reference many individual 2D images), and single-slice ``.mrc`` /
        ``.map`` files (nz == 1) that carry one 2D image, plus common raster
        image formats (PNG, JPEG, TIFF, ...). Genuine volumes (nz > 1) are
        not stacks and keep the "2D Slice" label instead.
        """
        from pathlib import Path

        ext = Path(path).suffix.lower()
        if ext == ".mrcs":
            return True
        if ext == ".star":
            name = Path(path).name.lower()
            if any(name.endswith(s) for s in _METADATA_STAR_SUFFIXES):
                return False
            return True
        if ext in _MRC_VOLUME_EXTENSIONS:
            return not self._mrc_is_volume(path)
        if ext in _RASTER_IMAGE_EXTENSIONS:
            return True
        return False

    def _is_volume(self, path: str) -> bool:
        """Return True for genuine 3D volume files (MRC-format files with more
        than one Z slice).

        These expose both a 2D-slice mode and a 3D-volume mode; the slice
        button is labelled "2D Slice" to distinguish it from a 2D image.
        """
        from pathlib import Path

        return Path(path).suffix.lower() in _MRC_VOLUME_EXTENSIONS and (
            self._mrc_is_volume(path)
        )

    def _on_selection_changed(self, selected, deselected) -> None:
        self._update_preview()
        indexes = self._tree.selectionModel().selectedIndexes()
        # selectedIndexes() returns one index per column; collapse to rows.
        rows = sorted({idx.row() for idx in indexes})
        self._current_path = None
        # what is being worked out for an earlier file is dropped
        self._modes_serial += 1
        if len(rows) != 1:
            self._action_bar.hide()
            self._modes_applied = self._modes_serial
            return
        index = self._model.index(rows[0], 0)
        path = self._model.file_path(index)
        if not path or self._model.is_dir(index):
            self._action_bar.hide()
            self._modes_applied = self._modes_serial
            return
        # the buttons depend on what is in the file (and its folder), which can mean
        # reading headers and datasets, on a network disk too: not on the main thread
        self._action_bar.hide()
        self._modes_pending = str(path)
        if self._modes_worker is None or not self._modes_worker.isRunning():
            self._start_modes()

    def _start_modes(self) -> None:
        path, self._modes_pending = self._modes_pending, None
        if path is None:
            return
        worker = _ModesWorker(self._modes_serial, path, self)
        worker.ready.connect(self._on_modes_ready)
        worker.finished.connect(self._on_modes_finished)
        self._modes_worker = worker
        worker.start()

    def _on_modes_finished(self) -> None:
        if self._modes_pending is not None:
            self._start_modes()

    def _on_modes_ready(self, serial: int, info: dict) -> None:
        if serial != self._modes_serial:
            return  # another file was selected meanwhile
        self._apply_modes(info)

    def wait_for_selection(self, timeout_ms: int = 10000) -> None:
        """Wait until the buttons of the selected file are shown (for tests and scripts)."""
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            worker = self._modes_worker
            if worker is not None and worker.isRunning():
                worker.wait(50)
            QApplication.processEvents()
            if (
                self._modes_pending is None
                and (worker is None or not worker.isRunning())
                and self._modes_serial == self._modes_applied
            ):
                return

    def _apply_modes(self, info: dict) -> None:
        """Show the buttons for what the worker found: ``path``, ``modes``, ``slice_text``
        and whether ChimeraX is installed."""
        path, modes = info["path"], info["modes"]
        self._modes_applied = self._modes_serial
        if not modes:
            self._action_bar.hide()
            return
        for attr, mode in self._DISPLAY_BUTTONS:
            getattr(self, attr).setVisible(mode in modes)
        if "chimerax" in modes:
            if not info["chimerax"]:
                self._btn_chimerax.setEnabled(False)
                # Explicit disabled styling so the button is unambiguously
                # greyed-out regardless of the active Qt theme.
                colors = _THEME_COLORS[_resolved_theme(_saved_theme())]
                self._btn_chimerax.setStyleSheet(_disabled_button_stylesheet(colors))
                self._btn_chimerax.setToolTip(
                    "ChimeraX not found. Install it from "
                    "https://www.cgl.ucsf.edu/chimerax/ or add it to your PATH."
                )
            else:
                self._btn_chimerax.setEnabled(True)
                self._btn_chimerax.setStyleSheet("")
                self._btn_chimerax.setToolTip("Open this file in ChimeraX")
        if "stats" in modes:
            self._btn_stats.setEnabled(True)
            self._btn_stats.setStyleSheet("")
            self._btn_stats.setToolTip(
                "Estimate and display per-filament tilt, psi, and rot-angle variance"
            )
        self._btn_slice.setText(info["slice_text"])
        self._current_path = str(path)
        self._reorder_buttons_alphabetically(modes)
        self._action_bar.show()

    def _emit_display(self, mode: str) -> None:
        if not getattr(self, "_current_path", None):
            return
        self.display_requested.emit(
            self._current_path, mode, self._new_window_cb.isChecked()
        )

    def _reorder_buttons_alphabetically(self, modes: list[str]) -> None:
        """Re-order visible display-mode buttons alphabetically by label.

        All display buttons are removed from the layout and only the non-hidden
        ones are re-inserted, sorted by their current text label.  Visibility
        is controlled by ``_on_selection_changed`` (which calls ``setVisible``
        per mode) before this method runs.  Because the button set is driven by
        ``_DISPLAY_BUTTONS``, any new button added there is automatically
        included — no separate mapping to maintain.
        """
        layout = self._action_bar.layout()
        buttons = [getattr(self, attr) for attr, _ in self._DISPLAY_BUTTONS]
        for btn in buttons:
            layout.removeWidget(btn)
        visible = [btn for btn in buttons if not btn.isHidden()]
        visible.sort(key=lambda b: b.text().lower())
        for btn in reversed(visible):
            layout.insertWidget(0, btn)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QToolTip

        # Disabled buttons do not emit hover events, so Qt will not show
        # their tooltips on its own. Surface them manually on mouse-move.
        if event.type() == QEvent.MouseMove:
            for btn in (self._btn_chimerax, self._btn_stats):
                if not btn.isEnabled() and btn.isVisible():
                    local = btn.mapFromGlobal(QCursor.pos())
                    if btn.rect().contains(local):
                        QToolTip.showText(QCursor.pos(), btn.toolTip(), btn)
                        break
            else:
                QToolTip.hideText()
        return False

    def _navigate_to(self, path: str) -> None:
        path = str(Path(path).resolve())
        if path != self._history[self._history_index]:
            # (going to the folder one is in again is a refresh, not a new step back)
            if self._history_index < len(self._history) - 1:
                self._history = self._history[: self._history_index + 1]
            self._history.append(path)
            self._history_index = len(self._history) - 1
        self._model.set_root_path(path)
        sort_col = self._tree.header().sortIndicatorSection()
        sort_order = self._tree.header().sortIndicatorOrder()
        self._model.sort(sort_col, sort_order)
        self._path_edit.setText(path)
        self._refresh_recent_combo()
        self._refresh_recent_menu()
        self._refresh_sidebar()
        self._update_preview()
        # (the columns that need reading a file are filled when the model says that
        # the folder is all there: load_finished)

    def _toggle_bookmark(self) -> None:
        """Bookmark the folder shown, or remove its bookmark if it has one."""
        path = self._model._root_path
        if _is_bookmarked(path):
            _remove_bookmark(path)
        else:
            _add_bookmark(path)
        self._refresh_bookmarks_menu()

    def _refresh_bookmarks_menu(self) -> None:
        """Fill the Bookmarks menu: the toggle, then one entry per bookmark."""
        menu = self._bookmarks_menu
        menu.clear()
        if hasattr(self, "_sidebar"):
            self._refresh_sidebar()
        here = self._model._root_path
        self._bookmark_toggle_action.setText(
            "Remove Bookmark of This Folder"
            if _is_bookmarked(here)
            else "Bookmark This Folder"
        )
        menu.addAction(self._bookmark_toggle_action)
        menu.addSeparator()
        marks = _get_bookmarks()
        if not marks:
            menu.addAction("No bookmarks yet").setEnabled(False)
            return
        for path in marks:
            action = menu.addAction(path)
            if Path(path).is_dir():
                action.triggered.connect(partial(self._on_bookmark_selected, path))
            else:
                action.setText(f"{path}  (not available)")
                action.setEnabled(False)
        menu.addSeparator()
        remove_menu = menu.addMenu("Remove Bookmark")
        for path in marks:
            action = remove_menu.addAction(path)
            action.triggered.connect(partial(self._on_bookmark_removed, path))
        menu.addAction("Manage Bookmarks…").triggered.connect(self._manage_bookmarks)
        menu.addAction("Clear All Bookmarks").triggered.connect(
            self._clear_all_bookmarks
        )

    def _on_bookmark_selected(self, path: str) -> None:
        """Go to a bookmarked folder."""
        if Path(path).is_dir():
            self._navigate_to(path)

    def _manage_bookmarks(self) -> None:
        """Open the dialog to reorder or remove the bookmarks."""
        _BookmarksDialog(self).exec()
        self._refresh_bookmarks_menu()

    def _clear_all_bookmarks(self) -> None:
        if _confirm_clear_bookmarks(self):
            _clear_bookmarks()
            self._refresh_bookmarks_menu()

    def _on_bookmark_removed(self, path: str) -> None:
        _remove_bookmark(path)
        self._refresh_bookmarks_menu()

    def _refresh_recent_combo(self) -> None:
        self._recent_combo.blockSignals(True)
        self._recent_combo.clear()
        self._recent_combo.addItems(_get_recent_folders())
        self._recent_combo.setCurrentIndex(-1)
        self._recent_combo.blockSignals(False)

    def _refresh_recent_menu(self) -> None:
        """Populate the File menu's recent-folder submenu."""
        self._recent_menu.clear()
        recent = _get_recent_folders()
        if not recent:
            action = self._recent_menu.addAction("No recent folders")
            action.setEnabled(False)
            return
        for path in recent:
            action = self._recent_menu.addAction(path)
            action.triggered.connect(partial(self._on_recent_path_selected, path))

    def _on_recent_path_selected(self, path: str) -> None:
        """Navigate to a folder selected from the File menu."""
        if Path(path).is_dir():
            self._navigate_to(path)

    def _on_recent_selected(self, index: int) -> None:
        path = self._recent_combo.itemText(index)
        if path and Path(path).is_dir():
            self._navigate_to(path)

    def _open_folder(self) -> None:
        """Show a folder chooser and navigate to the selected directory."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Open Folder",
            self._model._root_path,
        )
        if path:
            self._navigate_to(path)

    def _open_terminal(self) -> None:
        """Open the host OS terminal in the currently displayed folder."""
        _open_terminal(self._model._root_path)

    def _configure_relion(self) -> None:
        settings = QSettings("helicon", "display")
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure RELION")
        layout = QVBoxLayout(dialog)
        help_text = QLabel(
            "Select a Bash launcher script that loads your RELION module and/or\n"
            "activates its conda environment, then runs exec relion.\n"
            "Clean environment: initialize all required software in the script.\n"
            "User Bash setup: inherit the environment and read login profiles,\n"
            "then source the script in that shell. Switch conda environments as needed.\n\n"
            "With no script, use relion from the current environment (Clean)\n"
            "or from the initialized login shell (User Bash setup).\n"
            "HELICON_RELION_LAUNCHER, when set, overrides this saved choice.\n"
            "Queued jobs also need environment setup in their submission template."
        )
        layout.addWidget(help_text)
        startup_combo = QComboBox()
        startup_combo.addItem("Clean environment (default)", "clean")
        startup_combo.addItem("User Bash setup (interactive login shell)", "user-shell")
        startup_combo.setCurrentIndex(
            max(
                0,
                startup_combo.findData(str(settings.value("relion/startup", "clean"))),
            )
        )
        layout.addWidget(startup_combo)
        row = QHBoxLayout()
        path_edit = QLineEdit(str(settings.value("relion/launcher", "") or ""))
        path_edit.setPlaceholderText("Absolute path to launcher script (optional)")
        row.addWidget(path_edit)
        browse = QPushButton("Browse…")

        def choose_script():
            path, _ = QFileDialog.getOpenFileName(dialog, "RELION launcher script")
            if path:
                path_edit.setText(path)

        browse.clicked.connect(choose_script)
        row.addWidget(browse)
        layout.addLayout(row)
        override = os.environ.get("HELICON_RELION_LAUNCHER", "")
        if override:
            label = QLabel(f"Environment override: {override}")
            label.setTextFormat(Qt.TextFormat.PlainText)
            layout.addWidget(label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )

        def save():
            value = path_edit.text().strip()
            path = Path(value).expanduser()
            if value and (not path.is_absolute() or not path.is_file()):
                QMessageBox.warning(
                    dialog,
                    "RELION launcher",
                    "Select an existing script using an absolute path.",
                )
                return
            settings.setValue("relion/launcher", str(path) if value else "")
            settings.setValue("relion/startup", startup_combo.currentData())
            dialog.accept()

        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _open_relion(self) -> None:
        from helicon.lib.relion_launcher import launch_relion

        project = str(Path(self._model._root_path).resolve())
        existing = self._relion_processes.get(project)
        if existing and existing[0].poll() is None:
            QMessageBox.information(
                self, "RELION", "RELION is already running for this project."
            )
            return
        launcher = os.environ.get("HELICON_RELION_LAUNCHER") or str(
            QSettings("helicon", "display").value("relion/launcher", "") or ""
        )
        startup = str(QSettings("helicon", "display").value("relion/startup", "clean"))
        try:
            process, log = launch_relion(project, launcher, startup=startup)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot open RELION", str(exc))
            return
        self._relion_processes[project] = (process, log)
        self.statusBar().showMessage(f"RELION starting. Launch log: {log}")
        self._relion_timer.start()

    def _poll_relion(self) -> None:
        for project, (process, log) in list(self._relion_processes.items()):
            code = process.poll()
            if code is None:
                continue
            del self._relion_processes[project]
            if code:
                QMessageBox.warning(
                    self,
                    "RELION exited",
                    f"RELION exited with status {code}.\nProject: {project}\n"
                    f"See the launch log: {log}\nCheck Apps → Configure RELION…",
                )
        if not self._relion_processes:
            self._relion_timer.stop()

    def _on_launch_app(self, checked: bool = False) -> None:
        """Launch the standalone tool listed in the Apps menu.

        Shiny web-app tabs reuse the same launch pipeline as the display
        buttons (``_launch_or_reuse_web_app``) but with no input file, so the
        app opens with its own file-picker ready. Streamlit subcommands are
        launched through their command module's ``main``.
        The Proc3D, Images2Star and trueFSC entries pick their inputs first.
        """
        action = self.sender()
        if not isinstance(action, QAction):
            return
        key, streamlit_mod, picker = action.data()
        if picker == "map":
            self._on_pick_proc3d_map()
            return
        if picker == "star":
            self._on_pick_images2star_file()
            return
        if picker == "two_maps":
            self._on_pick_truefsc_maps()
            return
        if streamlit_mod is not None:
            self._launch_streamlit_app(streamlit_mod)
            return
        self._launch_web_app_tab(key)

    def _on_pick_proc3d_map(self) -> None:
        """Open the Proc3D tools panel with its in-panel file selector."""
        try:
            from helicon.commands import display as _display

            _display._open_proc3d_tools(
                None,
                parent=self,
                reuse_window=(
                    None
                    if self._new_window_cb.isChecked()
                    else _display._proc3d.active()
                ),
                tracker=_display._proc3d,
            )
        except Exception:
            import logging

            logging.getLogger("helicon").exception("failed to launch Proc3D")

    def _on_pick_images2star_file(self) -> None:
        """Open the Images2Star tools panel with its in-panel file selector."""
        try:
            from helicon.commands import display as _display

            _display._open_images2star_tools(
                None,
                parent=self,
                reuse_window=(
                    None
                    if self._new_window_cb.isChecked()
                    else _display._images2star.active()
                ),
                tracker=_display._images2star,
            )
        except Exception:
            import logging

            logging.getLogger("helicon").exception("failed to launch Images2Star")

    def _on_pick_truefsc_maps(self) -> None:
        """Open the trueFSC panel with empty input map selectors."""
        try:
            from helicon.commands import display as _display

            _display._launch_truefsc_maps(parent=self)
        except Exception:
            import logging

            logging.getLogger("helicon").exception("failed to launch trueFSC")

    def _launch_web_app_tab(self, tab: str) -> None:
        """Open *tab* in Helicon Lab with no pre-set input file."""
        try:
            from helicon.commands import display as _display

            _display._launch_or_reuse_web_app(
                tab,
                _display._make_bookmark_query(tab, {}),
                new_window=False,
            )
        except Exception:
            import logging

            logging.getLogger("helicon").exception(
                "failed to launch web app tab %r", tab
            )

    @staticmethod
    def _launch_streamlit_app(module_name: str) -> None:
        """Run a streamlit subcommand (e.g. ctfSimulation) in a new process.

        The subcommand ``main`` blocks until the server exits (``streamlit
        run``), so it must run detached from the GUI event loop.
        """
        # Import the command module and invoke ``main`` with a freshly parsed
        # (empty) argument namespace, rather than running its ``__main__``
        # block.  ``runpy.run_module(..., alter_sys=True)`` leaves the module
        # name in ``sys.argv`` and the ``__main__`` block's ``parse_args()``
        # then rejects that unrecognized argument, so the app never starts.
        cmd = [
            sys.executable,
            "-c",
            (
                "import importlib,sys;"
                "m=importlib.import_module(sys.argv[1]);"
                "import argparse;"
                "p=argparse.ArgumentParser();"
                "(m.add_args(p) if hasattr(m,'add_args') else None);"
                "m.main(p.parse_args([]))"
            ),
            module_name,
        ]
        try:
            _spawn_detached(cmd)
        except Exception:
            import logging

            logging.getLogger("helicon").exception(
                "failed to launch streamlit app %r", module_name
            )

    def _quit(self) -> None:
        """Exit the application (File → Quit / Ctrl+Q)."""
        app = QApplication.instance()
        if app is not None:
            app.quit()
        else:
            self.close()

    def _open_home_page(self) -> None:
        """Open the Helicon project home page in the system browser."""
        _open_url("https://jianglab.science.psu.edu/helicon")

    def _open_documentation(self) -> None:
        """Open the Helicon user docs in the system browser."""
        _open_url("https://helicon.readthedocs.io")

    def _go_up(self) -> None:
        current = self._model._root_path
        parent = str(Path(current).parent)
        if parent != current:
            self._navigate_to(parent)

    def _go_back(self) -> None:
        if self._history_index > 0:
            self._history_index -= 1
            path = self._history[self._history_index]
            self._model.set_root_path(path)
            sort_col = self._tree.header().sortIndicatorSection()
            sort_order = self._tree.header().sortIndicatorOrder()
            self._model.sort(sort_col, sort_order)
            self._path_edit.setText(path)
            self._refresh_sidebar()
            self._update_preview()

    def _go_to_path(self) -> None:
        path = self._path_edit.text().strip()
        if path and Path(path).is_dir():
            self._navigate_to(path)

    def _refresh(self) -> None:
        self._sidebar.invalidate()
        self._refresh_sidebar()
        self._model.refresh()

    def _populate_file_info_async(self) -> None:
        """Fill Dimension / Images / Pixel Size and folder counts in the background.

        The model rows are created synchronously with those columns empty so
        the folder opens instantly; this method then resolves the per-file
        metadata and per-folder entry counts off the GUI thread using small
        pools of worker threads. Old workers from a previous directory load
        are dropped (their results are discarded via the epoch guard and they
        self-delete on finish), and every result carries the model epoch so
        stale results are ignored.
        """
        from helicon.lib.relion_launcher import is_relion_project

        self._relion_btn.setVisible(is_relion_project(self._model._root_path))
        # Stop tracking any in-flight workers from a previous load.
        self._info_threads.clear()

        files, dirs = self._model.listed_paths()
        file_tasks = [(-1, path) for path in files]  # (the workers need no row)
        dir_tasks = [(-1, path) for path in dirs]
        if not file_tasks and not dir_tasks:
            return

        epoch = self._model.current_epoch()
        sort_col = self._tree.header().sortIndicatorSection()
        re_sort_needed = sort_col in (COL_SIZE, COL_IMAGES, COL_INFO, COL_PIXELSIZE)
        pending = 0

        def on_info_ready(batch, result_epoch):
            if result_epoch != self._model.current_epoch():
                return  # directory was reloaded; ignore stale result
            self._model.apply_file_infos(batch)

        def on_count_ready(batch, result_epoch):
            if result_epoch != self._model.current_epoch():
                return  # directory was reloaded; ignore stale result
            self._model.apply_dir_counts(batch)

        def on_worker_finished():
            nonlocal pending
            pending -= 1
            if pending == 0 and re_sort_needed:
                self._model.sort(sort_col, self._tree.header().sortIndicatorOrder())

        if file_tasks:
            num_workers = min(4, len(file_tasks))
            chunks = [file_tasks[i::num_workers] for i in range(num_workers)]
            pending += len(chunks)
            for chunk in chunks:
                worker = InfoWorker(chunk, epoch)
                worker.info_ready.connect(on_info_ready)
                # Defer deletion until after any still-queued info_ready events
                # from this worker have been delivered to the GUI thread.
                worker.finished.connect(
                    lambda w=worker: QTimer.singleShot(0, w.deleteLater)
                )
                worker.finished.connect(on_worker_finished)
                self._info_threads.append(worker)
                worker.start()

        if dir_tasks:
            num_workers = min(4, len(dir_tasks))
            chunks = [dir_tasks[i::num_workers] for i in range(num_workers)]
            pending += len(chunks)
            filter_pattern = self._model._filter_pattern
            use_regex = self._model._use_regex
            for chunk in chunks:
                worker = DirCountWorker(chunk, epoch, filter_pattern, use_regex)
                worker.count_ready.connect(on_count_ready)
                worker.finished.connect(
                    lambda w=worker: QTimer.singleShot(0, w.deleteLater)
                )
                worker.finished.connect(on_worker_finished)
                self._info_threads.append(worker)
                worker.start()

    def _apply_filter(self) -> None:
        pattern = self._filter_edit.text().strip()
        if not pattern:
            pattern = "*"
        use_regex = self._regex_cb.isChecked()
        self._model.set_filter(pattern, use_regex)

    def _copy_selection(self) -> None:
        indexes = self._tree.selectionModel().selectedIndexes()
        if not indexes:
            return

        rows = sorted(set(idx.row() for idx in indexes))
        cols = sorted(set(idx.column() for idx in indexes))

        lines = []
        for row in rows:
            cells = []
            for col in cols:
                idx = self._model.index(row, col)
                text = self._model.data(idx, Qt.ItemDataRole.DisplayRole) or ""
                cells.append(text)
            lines.append("\t".join(cells))

        QApplication.clipboard().setText("\n".join(lines))

    def _show_context_menu(self, pos):
        index = self._tree.indexAt(pos)
        if not index.isValid():
            return

        file_path = self._model.file_path(index)
        if not file_path:
            return

        menu = QMenu(self._tree)

        copy_path_action = QAction("Copy Path", self._tree)
        copy_path_action.triggered.connect(
            lambda: QApplication.clipboard().setText(file_path)
        )
        menu.addAction(copy_path_action)

        copy_name_action = QAction("Copy Name", self._tree)
        copy_name_action.triggered.connect(
            lambda: QApplication.clipboard().setText(Path(file_path).name)
        )
        menu.addAction(copy_name_action)

        if Path(file_path).is_dir():
            menu.addSeparator()
            marked = _is_bookmarked(file_path)
            bookmark_action = QAction(
                "Remove Bookmark" if marked else "Bookmark This Folder", self._tree
            )
            bookmark_action.triggered.connect(
                lambda: (_remove_bookmark if marked else _add_bookmark)(file_path)
            )
            menu.addAction(bookmark_action)

        menu.exec_(self._tree.viewport().mapToGlobal(pos))

    def set_root_path(self, path: str | Path) -> None:
        self._navigate_to(str(path))

    def _restore_col_widths(self) -> None:
        """Restore column widths saved from a previous launch.

        Stored as a comma-separated string under the same QSettings group as
        the window geometry (``helicon`` / ``display``). Only columns that
        still exist are applied, so adding/removing columns later is safe.
        """
        try:
            settings = QSettings("helicon", "display")
            raw = settings.value("browser_colwidths")
            if not raw:
                return
            widths = [int(x) for x in str(raw).split(",") if x.strip()]
            header = self._tree.header()
            for col, w in enumerate(widths):
                if col < NUM_COLUMNS and w > 0:
                    header.resizeSection(col, w)
        except (TypeError, ValueError, RuntimeError):
            pass

    def _save_col_widths(self) -> None:
        """Persist current column widths to QSettings."""
        try:
            header = self._tree.header()
            widths = [header.sectionSize(col) for col in range(NUM_COLUMNS)]
            settings = QSettings("helicon", "display")
            settings.setValue("browser_colwidths", ",".join(str(w) for w in widths))
        except RuntimeError:
            pass

    def _restore_sort_state(self) -> None:
        """Restore sort column and order saved from a previous launch."""
        try:
            settings = QSettings("helicon", "display")
            col = settings.value("browser_sort_col")
            order = settings.value("browser_sort_order")
            if col is None or order is None:
                return
            col = int(col)
            order = Qt.SortOrder(int(order))
            if 0 <= col < NUM_COLUMNS:
                self._tree.sortByColumn(col, order)
        except (TypeError, ValueError):
            pass

    def _save_sort_state(self) -> None:
        """Persist current sort column and order to QSettings."""
        try:
            header = self._tree.header()
            settings = QSettings("helicon", "display")
            settings.setValue("browser_sort_col", header.sortIndicatorSection())
            settings.setValue("browser_sort_order", header.sortIndicatorOrder().value)
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        try:
            self._save_col_widths()
            self._save_sort_state()
            self._save_panes()
            self._preview.wait()
        except Exception:
            pass
        super().closeEvent(event)


def _folder_is_helical(folder: str) -> bool:
    """Check if any ``model.star`` in *folder* has ``rlnIsHelix = 1``.

    The answer is cached on disk so repeated file selections within the same
    RELION job folder do not re-scan the STAR header. The cache key includes
    the name, size and modification time of every ``*model.star`` file, so a
    running job that writes a new (helical) ``model.star`` is re-checked
    instead of returning a stale answer.
    """
    try:
        signature = tuple(
            sorted(
                (f.name, f.stat().st_size, f.stat().st_mtime_ns)
                for f in Path(folder).glob("*model.star")
            )
        )
    except Exception:
        signature = ()
    if not signature:
        return False
    return _folder_is_helical_cached(folder, signature)


@cache(
    cache_dir=str(setup_cache_dir() / "file_browser"), expires_after=timedelta(days=7)
)
def _folder_is_helical_cached(folder: str, signature: tuple) -> bool:
    """Scan the ``*model.star`` files of *folder* for ``rlnIsHelix = 1``.

    Parameters
    ----------
    folder : str
        Folder to scan.
    signature : tuple
        ``(name, size, mtime_ns)`` of each ``*model.star`` file; only used
        as part of the cache key.

    Returns
    -------
    bool
        True if any of the files marks the job as helical.
    """
    try:
        folder_path = Path(folder)
        for f in folder_path.glob("*model.star"):
            try:
                if _star_has_helix_col(f):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _star_has_helix_col(star_path: Path) -> bool:
    """Check whether a RELION STAR file has ``rlnIsHelix = 1``.

    Handles two header styles:

    * **Column-definition** (``data.star``) — the header lists column names
      with a ``#N`` index (e.g. ``_rlnIsHelix  #12``).  The first data row
      is then inspected at the indicated column position.
    * **Key-value** (``model.star``) — the header contains
      ``_rlnIsHelix  <value>`` directly, so the value is checked in-place.
    """
    col_index = None
    in_header = True
    with open(star_path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if in_header:
                if stripped.startswith("_rlnIsHelix"):
                    # Key-value format: "_rlnIsHelix  1"
                    tail = stripped.split("_rlnIsHelix", 1)[1].strip()
                    if "#" in tail:
                        # Column-definition format: "_rlnIsHelix  #12"
                        col_index = int(tail.rsplit("#", 1)[-1]) - 1
                    else:
                        return tail == "1"
                elif stripped.startswith("data_"):
                    # Start of a data block — header continues until the
                    # next blank line.
                    continue
                elif stripped == "":
                    # RELION STAR headers: blank line ends the current
                    # block.  Reset so we re-enter header mode when the
                    # next ``data_`` block starts.
                    col_index = None
            elif col_index is not None:
                parts = stripped.split()
                if len(parts) > col_index:
                    return parts[col_index] == "1"
                return False
    return False


def _find_chimerax() -> str | None:
    """Locate the ChimeraX executable.

    Checks common install locations on each platform, then falls back to the
    ``PATH``. Returns the executable path or ``None`` if not found.
    """
    import shutil

    candidates = [
        "ChimeraX",
        "chimerax",
        "/opt/UCSF/ChimeraX/bin/chimerax",
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
        str(Path.home() / "Applications/ChimeraX.app/Contents/MacOS/ChimeraX"),
        "/usr/bin/chimerax",
        "/usr/local/bin/chimerax",
        r"C:\Program Files\ChimeraX\bin\ChimeraX.exe",
        r"C:\Program Files\UCSF\ChimeraX\ChimeraX.exe",
    ]
    for cand in candidates:
        found = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if found:
            return found
    return None


def _open_url(url: str) -> None:
    """Open *url* in the system browser (WSL/Linux/macOS-safe)."""
    try:
        from helicon.lib.shiny import _open_browser

        _open_browser(url)
        return
    except Exception:
        pass
    import webbrowser

    webbrowser.open(url, new=2)
