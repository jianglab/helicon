"""Qt widgets for napari integration."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

try:
    from PySide6.QtWidgets import (
        QWidget,
        QTreeView,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QToolButton,
        QHeaderView,
    )
    from PySide6.QtGui import QStandardItemModel, QStandardItem
    from PySide6.QtCore import (
        QModelIndex,
        Signal,
        Qt,
        QDir,
        QFileInfo,
        QThread,
        QObject,
    )
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QWidget,
            QTreeView,
            QVBoxLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QToolButton,
            QHeaderView,
        )
        from PyQt5.QtGui import QStandardItemModel, QStandardItem
        from PyQt5.QtCore import (
            QModelIndex,
            pyqtSignal as Signal,
            Qt,
            QDir,
            QFileInfo,
            QThread,
            QObject,
        )
    except ImportError:
        raise ImportError("Qt widgets require PySide6 or PyQt5")

__all__ = ["FolderBrowserWidget"]

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
}


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _get_file_info(filepath: str) -> tuple[str, str]:
    """Returns (info, n_images) tuple."""
    ext = Path(filepath).suffix.lower()
    info = ""
    n_images = ""

    if ext in (".mrc", ".mrcs", ".map"):
        try:
            import mrcfile

            with mrcfile.open(filepath, header_only=True) as mrc:
                nx = int(mrc.header.nx)
                ny = int(mrc.header.ny)
                nz = int(mrc.header.nz)
                if ext == ".mrcs":
                    n_images = str(nz) if nz > 1 else "1"
                    info = f"{nz}×{nx}×{ny}" if nz > 1 else f"{nx}×{ny}"
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
            import starfile

            df = starfile.read(filepath, always_dict=True)
            for val in df.values():
                if hasattr(val, "shape"):
                    n_images = str(val.shape[0])
                    image_name_col = None
                    for col in val.columns:
                        if "ImageName" in col or "MicrographName" in col:
                            image_name_col = col
                            break
                    if image_name_col:
                        first_name = str(val[image_name_col].iloc[0])
                        if "@" in first_name:
                            img_path = first_name.split("@")[-1]
                        else:
                            img_path = first_name
                        if Path(img_path).is_file():
                            img_ext = Path(img_path).suffix.lower()
                            if img_ext in (".mrc", ".mrcs", ".map"):
                                import mrcfile

                                with mrcfile.open(img_path, header_only=True) as mrc:
                                    nx = int(mrc.header.nx)
                                    ny = int(mrc.header.ny)
                                    nz = int(mrc.header.nz)
                                    if nz == 1:
                                        info = f"{nx}×{ny}"
                                    else:
                                        info = f"{nx}×{ny}×{nz}"
                            else:
                                from PIL import Image

                                with Image.open(img_path) as img:
                                    info = f"{img.width}×{img.height}"
                        else:
                            info = f"{val.shape[0]} rows"
                    else:
                        info = f"{val.shape[0]} rows"
                    break
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

    return info, n_images


COL_NAME = 0
COL_SIZE = 1
COL_TYPE = 2
COL_INFO = 3
COL_IMAGES = 4
COL_MODIFIED = 5
NUM_COLUMNS = 6
ROLE_SORT = Qt.ItemDataRole.UserRole + 1


class FileBrowserModel(QStandardItemModel):
    def __init__(self, root_path: str, parent=None):
        super().__init__(0, NUM_COLUMNS, parent)
        self.setHorizontalHeaderLabels(
            ["Name", "Size", "Type", "Dimension", "Images", "Modified"]
        )
        self._root_path = root_path
        self._file_infos: dict[str, tuple[str, str]] = {}
        self._sorting = False
        self._filter_pattern = "*"
        self._use_regex = False
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
            mod_item = QStandardItem("")
            mod_item.setData("", ROLE_SORT)
        else:
            size_bytes = path.stat().st_size
            size_item = QStandardItem(_format_size(size_bytes))
            size_item.setData(size_bytes, ROLE_SORT)

            ext = path.suffix.lower()
            type_item = QStandardItem(ext.lstrip(".").upper() if ext else "File")
            type_item.setData(ext, ROLE_SORT)

            info, n_images = self._get_or_cache_info(str(path))
            info_item = QStandardItem(info)
            info_item.setData(info, ROLE_SORT)

            images_item = QStandardItem(n_images)
            try:
                images_item.setData(int(n_images) if n_images else 0, ROLE_SORT)
            except ValueError:
                images_item.setData(0, ROLE_SORT)

            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            mod_item = QStandardItem(mtime.strftime("%Y-%m-%d %H:%M"))
            mod_item.setData(mtime.isoformat(), ROLE_SORT)

        name_item.setDragEnabled(False)
        name_item.setDropEnabled(False)

        return [name_item, size_item, type_item, info_item, images_item, mod_item]

    def _get_or_cache_info(self, filepath: str) -> tuple[str, str]:
        if filepath not in self._file_infos:
            self._file_infos[filepath] = _get_file_info(filepath)
        return self._file_infos[filepath]

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


class FolderBrowserWidget(QWidget):
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

    def __init__(self, start_dir: str | Path | None = None, parent=None):
        super().__init__(parent)

        if start_dir is None:
            start_dir = os.getcwd()
        start_dir = str(Path(start_dir).resolve())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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

        layout.addLayout(nav_layout)

        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(4, 4, 4, 4)
        filter_layout.setSpacing(4)

        from PySide6.QtWidgets import QLabel, QCheckBox

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

        self._history = [start_dir]
        self._history_index = 0

        self._tree = QTreeView()
        self._tree.setModel(self._model)
        self._tree.setRootIndex(QModelIndex())

        self._tree.setSortingEnabled(True)
        self._tree.sortByColumn(COL_NAME, Qt.SortOrder.AscendingOrder)

        header = self._tree.header()
        header.setStretchLastSection(False)
        header.setSectionsMovable(False)
        for col in range(NUM_COLUMNS):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(COL_NAME, 200)
        header.resizeSection(COL_SIZE, 70)
        header.resizeSection(COL_TYPE, 50)
        header.resizeSection(COL_INFO, 80)
        header.resizeSection(COL_IMAGES, 50)
        header.resizeSection(COL_MODIFIED, 120)

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
            """
        )

        layout.addWidget(self._tree)

        self._tree.doubleClicked.connect(self._on_double_clicked)
        self._tree.clicked.connect(self._on_clicked)

    def _on_double_clicked(self, index: QModelIndex) -> None:
        path = self._model.file_path(index)
        if path and not self._model.is_dir(index):
            try:
                from PySide6.QtCore import Qt
                from PySide6.QtWidgets import QApplication
            except ImportError:
                from PyQt5.QtCore import Qt
                from PyQt5.QtWidgets import QApplication

            if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.file_selected_new_window.emit(path)
            else:
                self.file_selected.emit(path)

    def _on_clicked(self, index: QModelIndex) -> None:
        if self._model.is_dir(index):
            path = self._model.file_path(index)
            if path:
                self._navigate_to(path)

    def _navigate_to(self, path: str) -> None:
        path = str(Path(path).resolve())
        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]
        self._history.append(path)
        self._history_index = len(self._history) - 1
        self._model.set_root_path(path)
        self._path_edit.setText(path)

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
            self._path_edit.setText(path)

    def _go_to_path(self) -> None:
        path = self._path_edit.text().strip()
        if path and Path(path).is_dir():
            self._navigate_to(path)

    def _refresh(self) -> None:
        self._model.refresh()

    def _apply_filter(self) -> None:
        pattern = self._filter_edit.text().strip()
        if not pattern:
            pattern = "*"
        use_regex = self._regex_cb.isChecked()
        self._model.set_filter(pattern, use_regex)

    def set_root_path(self, path: str | Path) -> None:
        self._navigate_to(str(path))
