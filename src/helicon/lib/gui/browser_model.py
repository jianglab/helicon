"""The model of the list of files of the file browser, and the threads that fill it in."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QModelIndex, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QApplication,
)

from helicon.lib.gui.browser_files import (
    _format_item_count,
    _format_modified,
    _format_size,
    _get_file_info,
    _get_file_type_label,
    _count_dir_items,
    _name_matches_filter,
    _natural_sort_key,
)
from helicon.lib.gui.browser_parts import cryosparc_job_info
from helicon.lib.gui.browser_style import _THEME_COLORS, _resolved_theme, _saved_theme


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
        self._aliases: dict[str, str] = {}  # job -> alias, in the folder shown
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
                found, links = [], []
                for e in listing:
                    found.append((e.name, e.is_dir()))
                    if e.is_symlink():
                        links.append((e.name, e.path))
        except OSError:
            self.load_finished.emit()
            return
        # the alias of a RELION job is a link to it in the same folder: one row for the two,
        # the job with its alias, not the job and the link
        self._aliases = {}
        hidden = set()
        if links:
            from helicon.lib.gui.job_alias import alias_target

            names = {name for name, _is_dir in found}
            for name, link in links:
                job = alias_target(link, Path(path).name)
                if job is not None and job in names:
                    self._aliases.setdefault(job, name)
                    hidden.add(name)
        entries = [
            (name, is_dir)
            for name, is_dir in found
            if name not in hidden
            and not name.startswith(".")
            and (is_dir or self._matches_filter(name))
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
        alias = self._aliases.get(path.name) if is_dir else None
        name_item = QStandardItem(
            (f"{path.name}({alias})" if alias else path.name) + ("/" if is_dir else "")
        )
        if alias:
            name_item.setToolTip(f"{path.name}, alias {alias}")
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
