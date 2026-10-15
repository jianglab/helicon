"""The recent folders and the bookmarks of the file browser, kept in the user's settings."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)


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
