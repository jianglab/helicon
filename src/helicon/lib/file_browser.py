"""Qt widgets for napari integration."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

from .cache import cache

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
)
from PySide6.QtGui import (
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
    QTimer,
)

__all__ = ["FolderBrowserWidget"]

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
        "accent": "#4a6fa5",
        "pressed": "#5a82c4",
        "accent_border": "#6f9bd6",
        "disabled": "#7a7a7a",
        "disabled_bg": "#2b2b2b",
    },
    "Light": {
        "window": "#f4f4f4",
        "input": "#ffffff",
        "text": "#202020",
        "strong_text": "#202020",
        "border": "#b8b8b8",
        "header_border": "#d2d2d2",
        "hover": "#e8eef7",
        "accent": "#4a78b8",
        "pressed": "#36649f",
        "accent_border": "#2e5d9c",
        "disabled": "#888888",
        "disabled_bg": "#e5e5e5",
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


def _browser_stylesheet(colors: dict[str, str]) -> str:
    """Build the shared stylesheet for the browser and launch buttons."""
    return f"""
        QWidget {{
            background-color: {colors["window"]};
            color: {colors["text"]};
        }}
        QTreeView {{
            font-size: 12px;
            background-color: {colors["window"]};
            color: {colors["text"]};
            border: none;
        }}
        QTreeView::item:selected {{
            background-color: {colors["accent"]};
            color: #ffffff;
        }}
        QTreeView::item:hover {{ background-color: {colors["hover"]}; }}
        QHeaderView::section {{
            padding: 4px;
            font-weight: bold;
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["header_border"]};
        }}
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
        QToolButton, QPushButton {{
            background-color: {colors["input"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 3px;
            padding: 3px 8px;
        }}
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
        QToolButton:hover, QPushButton:hover {{
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


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


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


# Extensions with dedicated handlers/labels.  Everything else falls back
# to known text extension checks or uppercased extensions.
_KNOWN_EXTENSIONS = frozenset(
    {
        ".mrc",
        ".mrcs",
        ".map",
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
    name = Path(filepath).name.lower()
    ext = Path(filepath).suffix.lower()

    if ext == ".star":
        if any(name.endswith(s) for s in _METADATA_STAR_SUFFIXES):
            return "Metadata"
        return "STAR"

    if ext == ".bild":
        return "3D Plot"

    if ext == ".cs":
        return "CryoSPARC"

    if ext in (".html", ".htm"):
        return "Browser"

    if ext == ".pdf":
        return "PDF"

    if ext == ".eps":
        return "EPS"

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

    if ext in (".mrc", ".mrcs", ".map"):
        try:
            import mrcfile

            with mrcfile.open(filepath, header_only=True) as mrc:
                nx = int(mrc.header.nx)
                ny = int(mrc.header.ny)
                nz = int(mrc.header.nz)
                apix = float(mrc.voxel_size.x)
                if apix > 0:
                    pixel_size = f"{apix:.2f} Å"
                if ext == ".mrcs":
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

    elif ext in (".png", ".jpg", ".jpeg"):
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
                                    pixel_size = f"{apix:.2f} Å"
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
                    pixel_size = f"{optics_pixel_size:.2f} Å"
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


COL_NAME = 0
COL_SIZE = 1
COL_TYPE = 2
COL_IMAGES = 3
COL_INFO = 4
COL_PIXELSIZE = 5
COL_MODIFIED = 6
NUM_COLUMNS = 7
ROLE_SORT = Qt.ItemDataRole.UserRole + 1

# Star files that describe pipelines/optimisation rather than image data.
# Must stay in sync with _METADATA_STAR_SUFFIXES in commands/display.py.
_METADATA_STAR_SUFFIXES = (
    "pipeline.star",
    "optimiser.star",
    "model.star",
    "sampling.star",
    "job.star",
    "extractpick.star",
)


class FileBrowserModel(QStandardItemModel):
    def __init__(self, root_path: str, parent=None):
        super().__init__(0, NUM_COLUMNS, parent)
        self.setHorizontalHeaderLabels(
            ["Name", "Size", "Type", "Images", "Dimension", "Pixel Size", "Modified"]
        )
        self._root_path = root_path
        self._file_infos: dict[str, tuple[str, str, str]] = {}
        self._sorting = False
        self._filter_pattern = "*"
        self._use_regex = False
        # Bumped on every directory (re)load; in-flight async info workers from
        # a previous load are discarded via this epoch so stale results never
        # land in the new model.
        self._epoch = 0
        self._load_directory(root_path)

    def _matches_filter(self, name: str) -> bool:
        if not self._filter_pattern:
            return True
        if self._use_regex:
            import re

            try:
                return bool(re.search(self._filter_pattern, name))
            except re.error:
                return True
        else:
            import fnmatch

            return fnmatch.fnmatch(name, self._filter_pattern)

    def _load_directory(self, path: str) -> None:
        self._epoch += 1
        self.removeRows(0, self.rowCount())
        try:
            entries = sorted(
                Path(path).iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            return

        for entry in entries:
            if entry.name.startswith("."):
                continue
            if not entry.is_dir() and not self._matches_filter(entry.name):
                continue
            row = self._make_row(entry)
            self.appendRow(row)

    def set_filter(self, pattern: str, use_regex: bool = False) -> None:
        self._filter_pattern = pattern
        self._use_regex = use_regex
        self._load_directory(self._root_path)

    def _make_row(self, path: Path) -> list[QStandardItem]:
        is_dir = path.is_dir()
        name_item = QStandardItem(path.name + "/" if is_dir else path.name)
        name_item.setData(str(path), Qt.ItemDataRole.UserRole)
        name_item.setEditable(False)

        if is_dir:
            size_item = QStandardItem("")
            size_item.setData(0, ROLE_SORT)
            type_item = QStandardItem("Folder")
            type_item.setData("folder", ROLE_SORT)
            info_item = QStandardItem("")
            info_item.setData("", ROLE_SORT)
            images_item = QStandardItem("")
            images_item.setData(0, ROLE_SORT)
            apix_item = QStandardItem("")
            apix_item.setData("", ROLE_SORT)
            mod_item = QStandardItem("")
            mod_item.setData("", ROLE_SORT)
        else:
            try:
                size_bytes = path.stat().st_size
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
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
            mod_item = QStandardItem(mtime.strftime("%Y-%m-%d %H:%M"))
            mod_item.setData(mtime.isoformat(), ROLE_SORT)

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
        for row in range(self.rowCount()):
            if (
                self.data(self.index(row, COL_NAME), Qt.ItemDataRole.UserRole)
                == filepath
            ):
                return row
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
        if self._sorting:
            return
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
            is_dir = pair[type_col].data(ROLE_SORT) == "folder"
            return (0 if is_dir else 1, data)

        rows.sort(key=sort_key, reverse=(order == Qt.SortOrder.DescendingOrder))

        for row_items in rows:
            self.appendRow(row_items)


class InfoWorker(QThread):
    """Compute file metadata off the GUI thread.

    Each worker runs its own thread and is handed a slice of
    ``(row, filepath)`` pairs; it emits ``info_ready`` per file as it resolves
    the metadata. The blocking file reads (mrcfile / PIL / starfile / PDF)
    therefore never stall the UI. ``finished`` is emitted once the slice is
    exhausted so the owner can clean up and re-sort if needed.
    """

    info_ready = Signal(
        str, str, str, str, int
    )  # filepath, info, n_images, apix, epoch

    def __init__(self, tasks: list[tuple[int, str]], epoch: int) -> None:
        super().__init__()
        self._tasks = tasks
        self._epoch = epoch

    def run(self) -> None:
        for _row, filepath in self._tasks:
            info, n_images, pixel_size = _get_file_info(filepath)
            self.info_ready.emit(filepath, info, n_images, pixel_size, self._epoch)


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
        ("_btn_fsc", "fsc"),
        ("_btn_denovo3d", "denovo3D"),
        ("_btn_whereismyclass", "whereIsMyClass"),
        ("_btn_helicalprojection", "helicalProjection"),
        ("_btn_helicalpitch", "helicalPitch"),
        ("_btn_hill", "hill"),
        ("_btn_hi3d", "hi3d"),
        ("_btn_truefsc", "trueFSC"),
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
        self._view_menu = self._menu_bar.addMenu("View")

        self._open_folder_action = QAction("Open Folder…", self)
        self._open_folder_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_folder_action.triggered.connect(self._open_folder)
        self._file_menu.addAction(self._open_folder_action)

        self._recent_menu = self._file_menu.addMenu("Recent Folders")
        self._recent_menu.aboutToShow.connect(self._refresh_recent_menu)
        self._refresh_recent_menu()

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

        self._path_edit = QLineEdit(start_dir)
        self._path_edit.returnPressed.connect(self._go_to_path)
        nav_layout.addWidget(self._path_edit)

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

        layout.addLayout(nav_layout)

        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(4, 4, 4, 4)
        filter_layout.setSpacing(4)

        filter_label = QLabel("Filter:")
        filter_layout.addWidget(filter_label)

        self._filter_edit = QLineEdit("*")
        self._filter_edit.setPlaceholderText("e.g. *.mrc, *.tif")
        self._filter_edit.returnPressed.connect(self._apply_filter)
        filter_layout.addWidget(self._filter_edit)

        self._regex_cb = QCheckBox("Regex")
        self._regex_cb.stateChanged.connect(self._apply_filter)
        filter_layout.addWidget(self._regex_cb)

        layout.addLayout(filter_layout)

        self._model = FileBrowserModel(start_dir)

        # Active background info-population threads. Cleared (and quit) whenever
        # the directory is reloaded so superseded workers don't write to the
        # new model.
        self._info_threads: list[QThread] = []

        self._history = [start_dir]
        self._history_index = 0

        self._tree = QTreeView()
        self._tree.setModel(self._model)
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
        header.setStretchLastSection(False)
        header.setSectionsMovable(False)
        for col in range(NUM_COLUMNS):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(COL_NAME, 200)
        header.resizeSection(COL_SIZE, 70)
        header.resizeSection(COL_TYPE, 50)
        header.resizeSection(COL_IMAGES, 50)
        header.resizeSection(COL_INFO, 80)
        header.resizeSection(COL_PIXELSIZE, 70)
        header.resizeSection(COL_MODIFIED, 120)

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

        layout.addWidget(self._tree)

        # Bottom bar: contextual display-mode buttons + "new window" toggle.
        # Hidden unless exactly one file (not a directory) is selected.
        self._action_bar = QWidget()
        self._action_bar.setContentsMargins(4, 4, 4, 4)
        action_layout = QHBoxLayout(self._action_bar)
        action_layout.setContentsMargins(4, 4, 4, 4)
        action_layout.setSpacing(4)

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
        self._btn_fsc = QPushButton("FSC")
        self._btn_fsc.setToolTip(
            "Display the FSC curve, or the SSNR-derived W_MAP curve for Class3D, "
            "from model.star. Pick which refinement iterations to overlay when "
            "the job directory holds more than one model.star file."
        )
        self._btn_denovo3d = QPushButton("denovo3D")
        self._btn_denovo3d.setToolTip(
            "Open this file in the denovo3D web app for de novo helical indexing"
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
        self._btn_truefsc = QPushButton("trueFSC")
        self._btn_truefsc.setToolTip("Compute True FSC curve from the two half-maps")
        self._new_window_cb = QCheckBox("New")
        self._new_window_cb.setToolTip(
            "<div style='white-space: normal; width: 200px;'>Create a new display window instead of reusing the current one. Most recently clicked/focused display window will be used as the display target if there are two or more display windows</div>"
        )
        for attr, _mode in self._DISPLAY_BUTTONS:
            btn = getattr(self, attr)
            btn.setFixedHeight(26)
            action_layout.addWidget(btn)
        action_layout.addStretch(1)
        action_layout.addWidget(self._new_window_cb)
        self._action_bar.hide()

        self._apply_theme(self._theme_combo.currentText())

        for attr, mode in self._DISPLAY_BUTTONS:
            btn = getattr(self, attr)
            btn.clicked.connect(partial(self._emit_display, mode))

        layout.addWidget(self._action_bar)

        sel_model = self._tree.selectionModel()
        sel_model.selectionChanged.connect(self._on_selection_changed)

        self._tree.doubleClicked.connect(self._on_double_clicked)
        self._tree.clicked.connect(self._on_clicked)

        # Disabled widgets do not receive hover events, so Qt will not show
        # the ChimeraX "not found" tooltip on its own. This filter surfaces it
        # manually when the cursor is over the (disabled) button.
        self.installEventFilter(self)

        # Fill Dimension / Images / Pixel Size for the initial directory in
        # the background so the browser opens instantly (these columns start
        # empty and are populated by worker threads).
        self._populate_file_info_async()

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
                    modes = ["slice", "gallery", "text", "whereIsMyClass"]
                    if _folder_is_helical(str(Path(path).parent)):
                        modes.append("helicalPitch")
                    return modes
                _is_class3d_or_refine3d = any(
                    p.startswith("Class3D") or p.startswith("Refine3D")
                    for p in Path(path).parts
                )
                if _is_class3d_or_refine3d:
                    return ["slice", "gallery", "stats", "text"]
            if any(name.endswith(s) for s in _METADATA_STAR_SUFFIXES):
                return ["text"]
            return ["slice", "gallery", "text"]
        if ext == ".mrcs":
            modes = ["slice", "gallery"]
            _is_class2d = any(p.startswith("Class2D") for p in Path(path).parts)
            _helical = _folder_is_helical(str(Path(path).parent))
            if _is_class2d and _helical:
                modes.extend(["helicalProjection", "hill"])
            if _helical:
                modes.append("denovo3D")
            return modes
        if ext in (".mrc", ".map"):
            modes = ["slice", "volume", "gallery", "chimerax"]
            if self._volume_has_nz_gt1(path):
                modes.append("orthogonal")
            _is_class3d_or_refine3d = any(
                p.startswith("Class3D") or p.startswith("Refine3D")
                for p in Path(path).parts
            )
            if _is_class3d_or_refine3d and _folder_is_helical(str(Path(path).parent)):
                modes.append("hi3d")
            return modes
        if ext == ".bild":
            return ["text", "3dplot", "chimerax"]
        if ext == ".pdf":
            return ["slice"]
        if ext in _KNOWN_EXTENSIONS:
            return []

        if _is_text_file(path):
            return ["text"]

        return []

    def _volume_has_nz_gt1(self, path: str) -> bool:
        """Return True if a .mrc/.map file has more than one Z slice."""
        try:
            import mrcfile

            with mrcfile.open(path, permissive=True) as mrc:
                return (
                    mrc.data is not None
                    and mrc.data.ndim == 3
                    and mrc.data.shape[0] > 1
                )
        except Exception:
            return False

    def _is_image_stack(self, path: str) -> bool:
        """Return True for files that are stacks of 2D images.

        This covers ``.mrcs`` particle stacks and data ``.star`` files (which
        reference many individual 2D images). Volumes such as ``.mrc`` /
        ``.map`` are not stacks and keep the "Image Slice" label instead.
        """
        from pathlib import Path

        ext = Path(path).suffix.lower()
        if ext == ".mrcs":
            return True
        if ext == ".star":
            name = Path(path).name
            if any(name.endswith(s) for s in _METADATA_STAR_SUFFIXES):
                return False
            return True
        return False

    def _is_volume(self, path: str) -> bool:
        """Return True for 3D volume files (``.mrc`` / ``.map``).

        These expose both a 2D-slice mode and a 3D-volume mode; the slice
        button is labelled "2D Slice" to distinguish it from a 2D image.
        """
        from pathlib import Path

        return Path(path).suffix.lower() in (".mrc", ".map")

    def _on_selection_changed(self, selected, deselected) -> None:
        indexes = self._tree.selectionModel().selectedIndexes()
        # selectedIndexes() returns one index per column; collapse to rows.
        rows = sorted({idx.row() for idx in indexes})
        if len(rows) != 1:
            self._action_bar.hide()
            return
        index = self._model.index(rows[0], 0)
        path = self._model.file_path(index)
        if not path or self._model.is_dir(index):
            self._action_bar.hide()
            return
        modes = self._display_modes_for(str(path))
        if not modes:
            self._action_bar.hide()
            return
        for attr, mode in self._DISPLAY_BUTTONS:
            getattr(self, attr).setVisible(mode in modes)
        if "chimerax" in modes:
            if _find_chimerax() is None:
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
        # Label the slice button by the file/display type.
        if Path(path).suffix.lower() == ".pdf":
            self._btn_slice.setText("PDF")
        elif self._is_image_stack(str(path)):
            self._btn_slice.setText("2D Image")
        elif self._is_volume(str(path)):
            self._btn_slice.setText("2D Slice")
        else:
            self._btn_slice.setText("Image Slice")
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
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]
        self._history.append(path)
        self._history_index = len(self._history) - 1
        self._model.set_root_path(path)
        sort_col = self._tree.header().sortIndicatorSection()
        sort_order = self._tree.header().sortIndicatorOrder()
        self._model.sort(sort_col, sort_order)
        self._path_edit.setText(path)
        _add_recent_folder(path)
        self._refresh_recent_combo()
        self._refresh_recent_menu()
        self._populate_file_info_async()

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
            self._populate_file_info_async()

    def _go_to_path(self) -> None:
        path = self._path_edit.text().strip()
        if path and Path(path).is_dir():
            self._navigate_to(path)

    def _refresh(self) -> None:
        self._model.refresh()
        self._populate_file_info_async()

    def _populate_file_info_async(self) -> None:
        """Fill Dimension / Images / Pixel Size columns in background threads.

        The model rows are created synchronously with those columns empty so
        the folder opens instantly; this method then resolves the per-file
        metadata off the GUI thread using a small pool of worker threads. Old
        workers from a previous directory load are dropped (their results are
        discarded via the epoch guard and they self-delete on finish), and
        every result carries the model epoch so stale results are ignored.
        """
        # Stop tracking any in-flight workers from a previous load.
        self._info_threads.clear()

        tasks = self._model.file_rows()
        if not tasks:
            return

        epoch = self._model.current_epoch()
        num_workers = min(4, len(tasks))
        chunks = [tasks[i::num_workers] for i in range(num_workers)]
        chunks = [c for c in chunks if c]

        pending = len(chunks)
        sort_col = self._tree.header().sortIndicatorSection()
        re_sort_needed = sort_col in (COL_IMAGES, COL_INFO, COL_PIXELSIZE)

        def on_info_ready(filepath, info, n_images, pixel_size, result_epoch):
            if result_epoch != self._model.current_epoch():
                return  # directory was reloaded; ignore stale result
            self._model.apply_file_info(filepath, info, n_images, pixel_size)

        def on_worker_finished():
            nonlocal pending
            pending -= 1
            if pending == 0 and re_sort_needed:
                self._model.sort(sort_col, self._tree.header().sortIndicatorOrder())

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

    def _apply_filter(self) -> None:
        pattern = self._filter_edit.text().strip()
        if not pattern:
            pattern = "*"
        use_regex = self._regex_cb.isChecked()
        self._model.set_filter(pattern, use_regex)
        self._populate_file_info_async()

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
        except Exception:
            pass
        super().closeEvent(event)


@cache(expires_after=timedelta(days=7))
def _folder_is_helical(folder: str) -> bool:
    """Check if any ``model.star`` in *folder* has ``rlnIsHelix = 1``.

    The result is cached for one week so repeated file selections within the
    same RELION job folder do not re-scan the STAR header.
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
