"""Qt widgets for the Images2Star tools panel in ``helicon display``.

The panel is a thin front end over :mod:`helicon.lib.images2star_engine`:
the same ordered plugin dispatch the ``helicon images2star`` CLI runs, so a
GUI transform is byte-for-byte the CLI semantics. File-writing operations
are hidden from the stack (see ``GUI_EXCLUDED_OPERATIONS``).
"""

from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtCore import (
    QAbstractListModel,
    QAbstractTableModel,
    QModelIndex,
    Qt,
    QThread,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QKeySequence,
    QShortcut,
    QTextOption,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

import helicon
from helicon.lib.exceptions import HeliconError, HeliconIOError
from helicon.lib.images2star_engine import (
    apply_options,
    gui_operation_specs,
    parse_operation_value,
    stack_to_namespace,
)

_OUTPUT_FILTER = "RELION STAR (*.star);;CryoSPARC v2 (*.cs);;CSV (*.csv)"

# Compact, color-agnostic styling matching the file browser: the browser
# renders its file table at 12px (via ``QTreeView { font-size: 12px }``) with
# bold, tightly padded column headers, and its action buttons are a fixed
# 26px tall. The theme stylesheet supplies colors; these rules only tune
# typography and padding so the dialog feels as dense as the rest of the
# display app. The table font is set through the same stylesheet mechanism
# the browser uses so both resolve identically regardless of display scale.
# Header sections keep only horizontal padding: the row-number gutter (the
# vertical header) is rendered at the table's row height, and any top/bottom
# padding would shrink its content area below the digit glyph height,
# clipping the numbers (the horizontal header sections size themselves from
# the font, so they need no vertical cushion either).
_COMPACT_QSS = """
    QTableView {
        font-size: 12px;
    }
    QHeaderView::section {
        padding: 0 3px;
        font-weight: bold;
    }
    QSplitter::handle {
        height: 3px;
        background-color: palette(mid);
    }
"""


class _OperationStackModel(QAbstractListModel):
    """List model over ordered ``(option_name, param_text)`` operations.

    Supports internal drag-and-drop reordering (``Qt.DropAction.MoveAction``)
    so long stacks can be rearranged in one gesture instead of many
    Move Up/Down clicks. ``moveRows`` follows Qt's destination convention:
    ``destinationChild`` is the row *before* which the dragged row is placed,
    expressed before the move; moving down therefore lands at
    ``destinationChild - 1`` after removal.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._operations: list[tuple[str, str]] = []
        self._last_moved_row = -1

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._operations)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        name, text = self._operations[index.row()]
        return f"--{name} {text}".rstrip()

    def flags(self, index):
        """Enable dragging items and dropping anywhere in the list."""
        flags = super().flags(index)
        if index.isValid():
            flags |= Qt.ItemFlag.ItemIsDragEnabled
        flags |= Qt.ItemFlag.ItemIsDropEnabled
        return flags

    def supportedDropActions(self):
        """Only internal moves are supported (no external data drops)."""
        return Qt.DropAction.MoveAction

    def operations(self) -> list[tuple[str, str]]:
        """Return a snapshot of ``(option_name, param_text)`` pairs."""
        return list(self._operations)

    def add(self, name: str, text: str) -> None:
        """Append one operation and notify views."""
        row = len(self._operations)
        self.beginInsertRows(QModelIndex(), row, row)
        self._operations.append((name, text))
        self.endInsertRows()

    def remove_row(self, row: int) -> bool:
        """Remove the operation at ``row``; return False if out of range."""
        if not 0 <= row < len(self._operations):
            return False
        self.beginRemoveRows(QModelIndex(), row, row)
        self._operations.pop(row)
        self.endRemoveRows()
        return True

    def move_row(self, row: int, delta: int) -> bool:
        """Move the operation at ``row`` by ``delta`` positions."""
        target = row + delta
        if not (
            0 <= row < len(self._operations) and 0 <= target < len(self._operations)
        ):
            return False
        self.beginMoveRows(
            QModelIndex(),
            row,
            row,
            QModelIndex(),
            target + 1 if delta > 0 else target,
        )
        entry = self._operations.pop(row)
        self._operations.insert(target, entry)
        self.endMoveRows()
        return True

    def moveRows(
        self,
        sourceParent,
        sourceRow,
        count,
        destinationParent,
        destinationChild,
    ):
        """Reorder operations for an internal drag-and-drop move."""
        if sourceParent.isValid() or destinationParent.isValid():
            return False
        if count != 1:
            return False
        n = len(self._operations)
        if not (0 <= sourceRow < n and 0 <= destinationChild <= n):
            return False
        if destinationChild in (sourceRow, sourceRow + 1):
            return False  # no-op: dropped on itself or immediately below
        if not self.beginMoveRows(
            sourceParent,
            sourceRow,
            sourceRow,
            destinationParent,
            destinationChild,
        ):
            return False
        target = (
            destinationChild - 1 if destinationChild > sourceRow else destinationChild
        )
        entry = self._operations.pop(sourceRow)
        self._operations.insert(target, entry)
        self._last_moved_row = target
        self.endMoveRows()
        return True

    def clear(self) -> None:
        """Remove all operations."""
        self.beginResetModel()
        self._operations = []
        self.endResetModel()


class _DataFramePreviewModel(QAbstractTableModel):
    """Virtual model over the entire DataFrame with editable, sortable cells.

    Qt views only request the cells that are visible in the viewport, so no
    per-row materialization happens up front: scrolling through hundreds of
    thousands of rows renders just the on-screen cells, the same
    viewport-culling principle the image gallery uses for particle tiles.

    Sorting keeps the DataFrame itself untouched: ``_order`` maps a view row
    to the corresponding source row, so the underlying data (and therefore
    the saved output) keeps its original order no matter how the preview is
    reordered. Missing values sort last in both directions.

    Cell edits are validated against the column dtype and written back to the
    underlying DataFrame in place (so saving the working dataset includes
    manual corrections); values that cannot be parsed are rejected.
    """

    def __init__(self, data: pd.DataFrame, missing_files=(), parent=None):
        super().__init__(parent)
        self._data = data
        self._order = np.arange(len(data), dtype=int)
        self._missing_files = set(missing_files)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._order)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._data.columns)

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return super().flags(index) | Qt.ItemFlag.ItemIsEditable

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.ForegroundRole:
            if self._missing_files and self._cell_is_missing(index):
                return QColor(self._error_color())
            return None
        if role not in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.EditRole,
        ):
            return None
        value = self._data.iat[self._order[index.row()], index.column()]
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        if isinstance(value, bytes):
            try:
                value = value.decode()
            except UnicodeDecodeError:
                value = repr(value)
        return str(value)

    def _cell_is_missing(self, index) -> bool:
        """Return True when the cell references a file not found on disk."""
        value = self._data.iat[self._order[index.row()], index.column()]
        if value is None:
            return False
        return str(value).split("@")[-1] in self._missing_files

    @staticmethod
    def _error_color() -> str:
        """Return the theme's error color used to flag missing-file cells."""
        from helicon.lib.file_browser import (
            _THEME_COLORS,
            _resolved_theme,
            _saved_theme,
        )

        return _THEME_COLORS[_resolved_theme(_saved_theme())].get("error", "#b3261e")

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        column = self._data.columns[index.column()]
        converted = self._convert_edit(value, self._data[column].dtype)
        if converted is None:
            return False
        self._data.iat[self._order[index.row()], index.column()] = converted
        self.dataChanged.emit(
            index,
            index,
            [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole],
        )
        return True

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        """Sort the preview rows by ``column``, missing values last."""
        if not 0 <= column < len(self._data.columns):
            return
        series = self._data[self._data.columns[column]]
        ascending = order != Qt.SortOrder.DescendingOrder
        self.beginResetModel()
        try:
            self._order = self._sorted_order(series, ascending)
        finally:
            self.endResetModel()

    @staticmethod
    def _sorted_order(series: pd.Series, ascending: bool) -> np.ndarray:
        """Return source-row indices ordered by ``series``.

        Numeric columns (including object columns that are entirely numeric)
        sort numerically so 2 < 10; everything else sorts by string. Missing
        values always sort last, and ties keep their original (stable) order.
        """
        n = len(series)
        missing = series.isna().to_numpy()
        values = series.to_numpy()
        try:
            keys = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(
                dtype=float
            )
        except (TypeError, ValueError):
            keys = np.array(
                [
                    (
                        ""
                        if v is None
                        else (
                            v.decode(errors="replace")
                            if isinstance(v, bytes)
                            else str(v)
                        )
                    )
                    for v in values
                ]
            )
        valid = ~missing
        valid_indices = np.flatnonzero(valid)
        valid_keys = keys[valid]
        if len(valid_keys) > 1:
            try:
                sorted_valid = valid_indices[np.argsort(valid_keys, kind="stable")]
            except TypeError:
                # Mixed types within one column: fall back to string keys.
                sorted_valid = valid_indices[
                    np.argsort([str(k) for k in valid_keys], kind="stable")
                ]
        else:
            sorted_valid = valid_indices
        if not ascending:
            sorted_valid = sorted_valid[::-1]
        return np.concatenate([sorted_valid, np.flatnonzero(missing)])

    @staticmethod
    def _convert_edit(value, dtype) -> object:
        """Convert a user-entered string to the column dtype, or None to reject."""
        if value is None:
            return None
        text = str(value).strip()
        if pd.api.types.is_numeric_dtype(dtype):
            if text == "":
                return float("nan") if pd.api.types.is_float_dtype(dtype) else None
            try:
                number = float(text)
            except ValueError:
                return None
            if pd.api.types.is_integer_dtype(dtype):
                return int(number) if number.is_integer() else None
            return number
        if pd.api.types.is_bool_dtype(dtype):
            lowered = text.lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            return None
        return text

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._data.columns[section])
        return str(section + 1)


class _LoadWorker(QThread):
    """Load an image metadata file into a DataFrame off the UI thread."""

    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, path: str, loader, parent=None):
        super().__init__(parent)
        self._path = path
        self._loader = loader

    def run(self):
        try:
            self.loaded.emit(self._loader(self._path))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _SaveWorker(QThread):
    """Write a DataFrame to a star/cs/csv file off the UI thread."""

    saved = Signal(str)
    failed = Signal(str)

    def __init__(self, data: pd.DataFrame, path: str, saver, parent=None):
        super().__init__(parent)
        self._data = data
        self._path = path
        self._saver = saver

    def run(self):
        try:
            self._saver(self._data, self._path)
            self.saved.emit(self._path)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _CopyableTableView(QTableView):
    """Table view that copies selected cells (TSV, with headers) via Ctrl/Cmd+C
    or the context-menu Copy action.

    Column labels are copied via the header's context menu (single label or
    all labels), since options in the transformations stack are parameterized
    by column names. Header clicks sort the table instead.
    """

    labelCopied = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        header = self.horizontalHeader()
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._show_header_menu)
        header.setSectionsClickable(True)
        header.setToolTip(
            "Click to sort by a column; right-click a column label to copy it"
        )

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.StandardKey.Copy):
            self._copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        copy_action = menu.addAction("Copy")
        selection = self.selectionModel()
        copy_action.setEnabled(selection is not None and selection.hasSelection())
        chosen = menu.exec(event.globalPos())
        if chosen is copy_action:
            self._copy_selection()

    def _copy_selection(self) -> None:
        """Copy the selected cells to the clipboard as tab-separated text."""
        selection = self.selectionModel()
        if selection is None:
            return
        indexes = selection.selectedIndexes()
        if not indexes:
            return
        model = self.model()
        rows = sorted({index.row() for index in indexes})
        cols = sorted({index.column() for index in indexes})

        def _display(row, col) -> str:
            value = model.data(model.index(row, col), Qt.ItemDataRole.DisplayRole)
            return "" if value is None else str(value)

        header = [
            str(model.headerData(col, Qt.Orientation.Horizontal) or "") for col in cols
        ]
        lines = ["\t".join(header)]
        for row in rows:
            lines.append("\t".join(_display(row, col) for col in cols))
        QApplication.clipboard().setText("\n".join(lines))

    def _header_label(self, section: int) -> str:
        """Return the display label of one horizontal header section."""
        model = self.model()
        if model is None:
            return ""
        value = model.headerData(
            section, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
        )
        return "" if value is None else str(value)

    def _copy_header_label(self, section: int) -> None:
        """Copy one column label to the clipboard."""
        label = self._header_label(section)
        if not label:
            return
        QApplication.clipboard().setText(label)
        self.labelCopied.emit(label)

    def _copy_all_header_labels(self) -> None:
        """Copy every column label to the clipboard, one per line."""
        model = self.model()
        if model is None:
            return
        labels = [
            label
            for section in range(model.columnCount())
            if (label := self._header_label(section))
        ]
        if not labels:
            return
        text = "\n".join(labels)
        QApplication.clipboard().setText(text)
        self.labelCopied.emit(text)

    def _header_menu(self, section: int) -> QMenu:
        """Build the header context menu for one section."""
        menu = QMenu(self)
        copy_one = menu.addAction("Copy column label")
        copy_one.setEnabled(bool(self._header_label(section)))
        copy_one.triggered.connect(lambda: self._copy_header_label(section))
        copy_all = menu.addAction("Copy all column labels")
        copy_all.setEnabled(self.model() is not None and self.model().columnCount() > 0)
        copy_all.triggered.connect(self._copy_all_header_labels)
        return menu

    def _show_header_menu(self, pos) -> None:
        """Show the header context menu at the right-clicked section."""
        if self.model() is None:
            return
        header = self.horizontalHeader()
        section = header.logicalIndexAt(pos)
        menu = self._header_menu(section)
        menu.exec(header.viewport().mapToGlobal(pos))


class _DenseEditDelegate(QStyledItemDelegate):
    """Fit the edit editor into the tables' dense rows.

    Rows are sized from the editor's own sizeHint, but the editor still must
    not add its default frame and margins: those would eat the slack and clip
    the cell text at the top and bottom while editing. The editor is given
    the full cell rect, no frame and no margins so its text area matches the
    rendered cell text exactly.
    """

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QLineEdit):
            editor.setFrame(False)
            editor.setContentsMargins(0, 0, 0, 0)
            editor.setTextMargins(0, 0, 0, 0)
        return editor

    def updateEditorGeometry(self, editor, option, index):
        rect = option.rect
        if rect.height() < editor.fontMetrics().height():
            rect.setHeight(editor.fontMetrics().height())
        editor.setGeometry(rect)


def _missing_filename_values(data: pd.DataFrame) -> list[str]:
    """Return referenced image/micrograph paths that do not exist on disk.

    Missing micrograph files never raise (they are normalized with
    ``ignore_bad_micrograph_path=1`` by design), and a tolerant particle
    load leaves unresolved entries at their raw path, so an existence check
    is the single source of truth for what is unavailable.
    """
    seen: set[str] = set()
    missing: list[str] = []
    for column in ("rlnImageName", "rlnMicrographName", "rlnMicrographMovieName"):
        if column not in data:
            continue
        for value in data[column].dropna().unique():
            filename = str(value).split("@")[-1]
            if filename in seen:
                continue
            seen.add(filename)
            if not Path(filename).exists():
                missing.append(filename)
    return missing


def _images2dataframe_tolerant(path: str) -> pd.DataFrame:
    """Load ``path`` into a DataFrame, keeping the table on image errors.

    The strict load fails the moment one referenced image file cannot be
    found (e.g. a shared or not-yet-mounted filesystem), even though the
    star file itself parsed fine. In that case retry with
    ``ignore_bad_particle_path=1`` so every row from the file is still
    shown. Missing image and micrograph files found afterwards are recorded
    in ``data.attrs["load_warnings"]`` (for the status bar) and
    ``data.attrs["missing_files"]`` (to flag the offending table cells).
    """
    try:
        data = helicon.images2dataframe(path, target_convention="relion")
    except HeliconIOError as exc:
        try:
            data = helicon.images2dataframe(
                path,
                target_convention="relion",
                ignore_bad_particle_path=1,
            )
        except Exception:
            raise exc
        data.attrs["load_warnings"] = [f"HeliconIOError: {exc}"]
    missing = _missing_filename_values(data)
    if missing:
        warnings = list(data.attrs.get("load_warnings", []))
        n = len(missing)
        shown = ", ".join(missing[:20])
        if n > 20:
            shown += ", \u2026"
        warnings.append(f"{n} image/micrograph file(s) not found: {shown}")
        data.attrs["load_warnings"] = warnings
        data.attrs["missing_files"] = missing
    return data


class Images2StarDialog(QDialog):
    """Preview and save an image dataset with the images2star engine.

    Phase-1 panel: loads the dataset with ``helicon.images2dataframe``, shows
    a summary plus a virtual preview table over every particle (only the
    visible rows are rendered, so large datasets scroll without
    materializing the full table) and a separate table of optics groups, and
    lets the user build an ordered stack of pure in-memory ``images2star``
    options (``--select``, ``--sortby``, ...). Applying the stack runs the
    exact CLI dispatch loop
    (:func:`~helicon.lib.images2star_engine.apply_options`), so the preview
    matches what ``helicon images2star in out.star <options>`` would produce;
    the current (possibly transformed) dataset is exported via
    ``helicon.dataframe2file``.
    """

    def __init__(self, path: str, parent=None, loader=None, saver=None):
        super().__init__(parent)
        self.setProperty("helicon_theme_window", True)
        self._status_error = False
        self._path = str(Path(path).resolve())
        self._loader = loader or _images2dataframe_tolerant
        self._saver = saver or helicon.dataframe2file
        self._specs = gui_operation_specs()
        self._source_data: pd.DataFrame | None = None
        self._data: pd.DataFrame | None = None
        self._dirty = False
        self._sort_active = False
        self._last_output: str | None = None
        self._workers: list[QThread] = []
        self._load_seq = 0
        self._sized_once = False

        self.setWindowTitle(f"Images2Star - {Path(self._path).name}")
        self.resize(940, 800)

        compact = QFont()
        compact.setPixelSize(12)
        self.setFont(compact)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._title = QLabel(str(Path(self._path)))
        self._title.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        self._table = self._make_table()
        self._optics_table = self._make_table()
        self._particles_label = self._section_label("Particles")
        self._optics_label = self._section_label("Optics groups")

        # Both previews share one vertical splitter so the optics pane can be
        # dragged down to a sliver (most files have a single optics row) or
        # expanded for raw cases with many optics groups.
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setChildrenCollapsible(True)
        self._particles_pane = self._table_pane(self._particles_label, self._table)
        self._optics_pane = self._table_pane(self._optics_label, self._optics_table)
        self._splitter.addWidget(self._particles_pane)
        self._splitter.addWidget(self._optics_pane)
        self._splitter.setStretchFactor(0, 2)
        self._splitter.setStretchFactor(1, 1)
        self._table.horizontalHeader().sortIndicatorChanged.connect(
            self._on_sort_indicator_changed
        )
        self._optics_table.horizontalHeader().sortIndicatorChanged.connect(
            self._on_sort_indicator_changed
        )
        self._table.labelCopied.connect(self._on_label_copied)
        self._optics_table.labelCopied.connect(self._on_label_copied)

        # The previews and the transformations block share one outer vertical
        # splitter so a drag bar separates the optics pane from the options
        # stack (each is also resizable). The transformations block defaults
        # to half of the height its contents would naturally request.
        self._ops_group = self._make_operations_group()
        self._main_splitter = QSplitter(Qt.Orientation.Vertical)
        self._main_splitter.setChildrenCollapsible(True)
        self._main_splitter.addWidget(self._splitter)
        self._main_splitter.addWidget(self._ops_group)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 1)
        layout.addWidget(self._main_splitter, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        self._status = QPlainTextEdit()
        self._status.setReadOnly(True)
        self._status.setPlainText("Loading dataset\u2026")
        self._status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._status.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._status.document().setDocumentMargin(0)
        self._status.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._status.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        line_h = self._status.fontMetrics().lineSpacing()
        self._status.setMinimumHeight(line_h + 2)
        self._status.setMaximumHeight(5 * line_h + 4)
        policy = self._status.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Policy.Fixed)
        self._status.setSizePolicy(policy)
        self._fit_status_height()
        buttons.addWidget(self._status, 1)
        self._btn_save = self._compact_button(QPushButton("Save As\u2026"))
        self._btn_save.setEnabled(False)
        self._btn_save.setToolTip(
            "Write the current dataset to a RELION STAR, CryoSPARC, or CSV file"
        )
        self._btn_save.clicked.connect(self._choose_output)
        self._btn_close = self._compact_button(QPushButton("Close"))
        self._btn_close.clicked.connect(self.reject)
        # Save As and Close share the transformations action row (kept out of
        # the status band so the message can use the full width).
        self._ops_buttons_row.insertWidget(
            self._ops_buttons_row.count() - 1, self._btn_save
        )
        self._ops_buttons_row.insertWidget(
            self._ops_buttons_row.count() - 1, self._btn_close
        )
        layout.addLayout(buttons)

        self._load_worker = _LoadWorker(self._path, self._loader, parent=self)
        self._workers.append(self._load_worker)
        self._load_worker.loaded.connect(self._on_loaded)
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.start()
        self._install_shortcuts()
        self._apply_display_theme()

    def _install_shortcuts(self) -> None:
        """Enable Ctrl+W (close window) and Ctrl+Q (quit the app)."""
        close_sc = QShortcut(QKeySequence("Ctrl+W"), self)
        close_sc.activated.connect(self.close)

        quit_sc = QShortcut(QKeySequence.StandardKey.Quit, self)
        quit_sc.activated.connect(self._quit_app)

    def _quit_app(self) -> None:
        """Close the dialog (letting workers finish) and quit the app."""
        self.close()
        QApplication.quit()

    def load_path(self, path: str) -> None:
        """Reload this panel from a new file, reusing the same window.

        The transformation stack is kept (it is a recipe the user may want to
        apply to another dataset); the preview tables and dirty state are
        reset until the new dataset finishes loading.
        """
        self._path = str(Path(path).resolve())
        self.setWindowTitle(f"Images2Star - {Path(self._path).name}")
        self._source_data = None
        self._data = None
        self._dirty = False
        self._sort_active = False
        self._last_output = None
        self._btn_save.setEnabled(False)
        self._set_status("Loading dataset\u2026")

        # Guard against a slower previous load finishing after this one: only
        # the most recent request may populate the previews.
        self._load_seq += 1
        seq = self._load_seq
        worker = _LoadWorker(self._path, self._loader, parent=self)
        self._workers.append(worker)
        worker.loaded.connect(
            lambda data: self._on_loaded(data) if seq == self._load_seq else None
        )
        worker.failed.connect(
            lambda message: (
                self._on_load_failed(message) if seq == self._load_seq else None
            )
        )
        worker.start()

    def _apply_display_theme(self) -> None:
        """Apply the persisted display theme to the dialog and its children.

        Runs once at construction and is called again by
        :func:`helicon.commands.display._refresh_display_theme_windows` when
        the user switches themes while the dialog is open. The status label's
        error styling is re-rendered afterwards because the error color is
        theme-dependent and would otherwise keep the previous theme's red.
        """
        from PySide6.QtWidgets import QWidget

        from helicon.commands.display import (
            _display_theme_palette,
            _display_theme_stylesheet,
        )

        stylesheet = _display_theme_stylesheet() + _COMPACT_QSS
        palette = _display_theme_palette()
        self.setStyleSheet(stylesheet)
        self.setPalette(palette)
        for child in self.findChildren(QWidget):
            child.setStyleSheet(stylesheet)
            child.setPalette(palette)
        if self._status_error:
            self._status.setStyleSheet(self._status_stylesheet(error=True))

    # ------------------------------------------------------------------
    # Operations stack panel

    @staticmethod
    def _compact_button(button: QPushButton) -> QPushButton:
        """Give a button the browser's fixed 26px action-bar height."""
        button.setFixedHeight(26)
        return button

    def _apply_default_split_sizes(self) -> None:
        """Give the transformations block a compact default height.

        Runs once on first show, when the splitter's real height is known.
        The block defaults to the height its (compact) contents request --
        about half of the taller block the vertical button column used to
        demand -- and the drag bar lets the user expand it.
        """
        total = self._main_splitter.height()
        if total <= 0:
            return
        ops = self._ops_group.sizeHint().height()
        self._main_splitter.setSizes([max(total - ops, 0), ops])

    def showEvent(self, event) -> None:
        """Apply the default split once so user drags are never reset."""
        super().showEvent(event)
        if not self._sized_once:
            self._sized_once = True
            self._apply_default_split_sizes()

    def _make_operations_group(self) -> QGroupBox:
        """Build the ordered option-stack editor (add/apply/reset)."""
        group = QGroupBox("Transformations (images2star options)")
        outer = QVBoxLayout(group)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        entry_row = QHBoxLayout()
        entry_row.setSpacing(4)
        self._ops_combo = QComboBox()
        for name in sorted(self._specs):
            spec = self._specs[name]
            self._ops_combo.addItem(f"--{name}", name)
            self._ops_combo.setItemData(
                self._ops_combo.count() - 1, spec["help"], Qt.ItemDataRole.ToolTipRole
            )
        self._param_edit = QLineEdit()
        self._param_edit.setPlaceholderText("parameter value(s), e.g. rlnDefocusV 5,20")
        self._param_edit.returnPressed.connect(self._add_operation)
        self._btn_add = self._compact_button(QPushButton("Add"))
        self._btn_add.clicked.connect(self._add_operation)
        entry_row.addWidget(self._ops_combo)
        entry_row.addWidget(self._param_edit, 1)
        entry_row.addWidget(self._btn_add)
        outer.addLayout(entry_row)

        stack_row = QHBoxLayout()
        stack_row.setSpacing(4)
        self._stack_model = _OperationStackModel(parent=self)
        self._stack_view = QListView()
        self._stack_view.setModel(self._stack_model)
        self._stack_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._stack_view.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._stack_view.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._stack_view.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._stack_view.setDragDropOverwriteMode(False)
        self._stack_view.setToolTip(
            "Drag rows to reorder; Ctrl+Up / Ctrl+Down or the buttons move "
            "the selected row one step"
        )
        self._stack_model.rowsMoved.connect(self._on_rows_moved)
        self._stack_view.selectionModel().selectionChanged.connect(
            lambda *_: self._update_ops_buttons()
        )
        self._install_move_shortcuts()
        # Ignore the list's height hint so the block can sit at its compact
        # default height and only grow when the splitter is dragged open.
        # A two-row minimum keeps the stack readable at that default.
        self._stack_view.setSizePolicy(
            self._stack_view.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        self._stack_view.setMinimumHeight(
            2 * self._stack_view.fontMetrics().lineSpacing()
        )
        stack_row.addWidget(self._stack_view, 1)
        outer.addLayout(stack_row)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(4)
        self._btn_apply = self._compact_button(QPushButton("Apply"))
        self._btn_apply.setToolTip(
            "Run the stacked options through the images2star engine and "
            "refresh the preview (same semantics as the CLI)"
        )
        self._btn_apply.clicked.connect(self._apply_operations)
        self._btn_remove = self._compact_button(QPushButton("Remove"))
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_up = self._compact_button(QPushButton("Move Up"))
        self._btn_up.clicked.connect(lambda: self._move_selected(-1))
        self._btn_down = self._compact_button(QPushButton("Move Down"))
        self._btn_down.clicked.connect(lambda: self._move_selected(1))
        self._btn_clear = self._compact_button(QPushButton("Clear List"))
        self._btn_clear.clicked.connect(self._clear_stack)
        self._btn_cmd = self._compact_button(QPushButton("Show Command\u2026"))
        self._btn_cmd.setToolTip(
            "Show the equivalent helicon images2star command to copy or "
            "save as a script for batch reruns"
        )
        self._btn_cmd.clicked.connect(self._show_command)
        self._btn_reset = self._compact_button(QPushButton("Reset Data"))
        self._btn_reset.setToolTip(
            "Discard all transforms and restore the dataset as loaded "
            "(also clears the preview sort)"
        )
        self._btn_reset.clicked.connect(self._reset_to_source)
        for button in (
            self._btn_apply,
            self._btn_remove,
            self._btn_up,
            self._btn_down,
            self._btn_clear,
            self._btn_cmd,
            self._btn_reset,
        ):
            buttons_row.addWidget(button)
        buttons_row.addStretch(1)
        # Save As and Close join this action row (added in __init__ once the
        # buttons exist), so the bottom band holds only the status message.
        self._ops_buttons_row = buttons_row
        outer.addLayout(buttons_row)

        self._ops_combo.currentIndexChanged.connect(self._update_param_placeholder)
        self._update_param_placeholder()
        self._update_ops_buttons()
        return group

    def _install_move_shortcuts(self) -> None:
        """Bind Ctrl+Up / Ctrl+Down to move the selected operation."""
        up_sc = QShortcut(QKeySequence("Ctrl+Up"), self._stack_view)
        up_sc.activated.connect(lambda: self._move_selected(-1))
        down_sc = QShortcut(QKeySequence("Ctrl+Down"), self._stack_view)
        down_sc.activated.connect(lambda: self._move_selected(1))

    def _on_rows_moved(self, parent, start, end, dest_parent, row) -> None:
        """Keep the dragged row selected after a drag-and-drop move."""
        target = self._stack_model._last_moved_row
        if target < 0:
            target = row - 1 if row > start else row
        self._stack_view.setCurrentIndex(self._stack_model.index(target, 0))
        self._update_ops_buttons()

    def _update_param_placeholder(self) -> None:
        """Show the selected option's metavar/help as the parameter hint."""
        name = self._ops_combo.currentData()
        if name is None:
            return
        spec = self._specs[name]
        metavar = spec["metavar"]
        if metavar:
            self._param_edit.setPlaceholderText(f"{metavar}")
        self._param_edit.setToolTip(spec["help"] or f"--{name}")

    def _add_operation(self) -> None:
        """Validate the parameter text and append the option to the stack."""
        name = self._ops_combo.currentData()
        if name is None:
            return
        text = self._param_edit.text().strip()
        try:
            parse_operation_value(text, self._specs[name])
        except ValueError as exc:
            self._set_status(f"Cannot add --{name}: {exc}", error=True)
            return
        self._stack_model.add(name, text)
        self._param_edit.clear()
        self._set_status(f"Added --{name}; press Apply to transform the preview")
        self._update_ops_buttons()

    def _remove_selected(self) -> None:
        index = self._stack_view.currentIndex()
        if index.isValid() and self._stack_model.remove_row(index.row()):
            self._update_ops_buttons()

    def _move_selected(self, delta: int) -> None:
        index = self._stack_view.currentIndex()
        if index.isValid() and self._stack_model.move_row(index.row(), delta):
            self._stack_view.setCurrentIndex(
                self._stack_model.index(index.row() + delta, 0)
            )

    def _clear_stack(self) -> None:
        self._stack_model.clear()
        self._set_status("Cleared the operation list; preview unchanged")
        self._update_ops_buttons()

    def _default_output_path(self) -> str:
        """Default export path: the last saved file, else a distinct sibling.

        The fallback is ``<stem>.processed.star`` next to the input, so the
        suggested name can never collide with the input file.

        Returns
        -------
        str
            Path to offer as the default output location.
        """
        if self._last_output:
            return self._last_output
        source = Path(self._path)
        return str(source.with_name(source.stem + ".processed.star"))

    def _command_text(self, output: str | None = None) -> str:
        """Return the equivalent ``helicon images2star`` shell command.

        The stacked operations map 1:1 to CLI options: the user-entered
        parameter text is exactly what ``parse_operation_value`` would feed
        argparse, so shell-quoting each token reproduces the command the CLI
        would run. The output path defaults to
        :meth:`_default_output_path`.

        Parameters
        ----------
        output : str, optional
            Output path to embed; defaults to the remembered save path or
            the input-derived fallback.

        Returns
        -------
        str
            A single-line shell command ready to paste into a terminal.
        """
        if not output:
            output = self._default_output_path()
        parts = ["helicon", "images2star", shlex.quote(self._path), shlex.quote(output)]
        for name, text in self._stack_model.operations():
            parts.append(f"--{name}")
            if text:
                parts.extend(shlex.quote(token) for token in shlex.split(text))
        return " ".join(parts)

    def _show_command(self) -> None:
        """Display the equivalent CLI command for copying or saving."""
        self._build_command_dialog().exec()

    def _build_command_dialog(self) -> QDialog:
        """Return a dialog showing the batch-mode command (caller shows it)."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Equivalent images2star command")
        dlg.resize(760, 300)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        note = QLabel(
            "This is the batch-mode command for the current stack. Edit it, "
            "copy it, or save it as a shell script."
        )
        note.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        edit = QPlainTextEdit()
        edit.setPlainText(self._command_text())
        edit.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        # Wrap long tokens (paths, quoted values) mid-word instead of letting
        # them overflow the dialog horizontally.
        wrap = QTextOption()
        wrap.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        edit.document().setDefaultTextOption(wrap)
        edit.setMinimumHeight(140)
        layout.addWidget(edit)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        buttons.addStretch(1)
        copy_btn = self._compact_button(QPushButton("Copy"))
        copy_btn.setToolTip("Copy the command to the clipboard")
        copy_btn.clicked.connect(lambda: self._copy_command(edit.toPlainText()))
        buttons.addWidget(copy_btn)
        save_btn = self._compact_button(QPushButton("Save Script\u2026"))
        save_btn.setToolTip("Write the command to a shell script file")
        save_btn.clicked.connect(lambda: self._save_command_script(edit.toPlainText()))
        buttons.addWidget(save_btn)
        close_btn = self._compact_button(QPushButton("Close"))
        close_btn.clicked.connect(dlg.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        return dlg

    def _copy_command(self, text: str) -> None:
        """Copy a command string to the clipboard and report it."""
        QApplication.clipboard().setText(text)
        self._set_status("Command copied to clipboard")

    def _save_command_script(self, text: str) -> None:
        """Write the command to a ``.sh`` script chosen by the user."""
        default = Path(self._path).with_name(Path(self._path).stem + ".images2star.sh")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Command Script",
            str(default),
            "Shell script (*.sh);;All files (*)",
        )
        if not path:
            return
        Path(path).write_text(f"#!/bin/sh\n{text}\n", encoding="utf-8")
        self._set_status(f"Saved command to {path}")

    def _update_ops_buttons(self) -> None:
        """Enable Apply when the stack is non-empty; selection ops otherwise.

        Reset Data is available when the dataset is modified (transforms or
        cell edits) *or* the preview is sorted, so a pure sort can be undone
        with the same button without ever marking the data as modified.
        """
        has_ops = self._stack_model.rowCount() > 0
        self._btn_apply.setEnabled(has_ops)
        self._btn_clear.setEnabled(has_ops)
        self._btn_cmd.setEnabled(self._data is not None)
        index = self._stack_view.currentIndex()
        has_selection = index.isValid()
        row = index.row() if has_selection else -1
        self._btn_remove.setEnabled(has_selection)
        self._btn_up.setEnabled(has_selection and row > 0)
        self._btn_down.setEnabled(
            has_selection and row < self._stack_model.rowCount() - 1
        )
        self._btn_reset.setEnabled(
            self._data is not None and (self._dirty or self._sort_active)
        )

    def _on_sort_indicator_changed(self, section: int, order) -> None:
        """Track whether any preview table is sorted so Reset can undo it."""
        self._sort_active = any(
            table.horizontalHeader().sortIndicatorSection() >= 0
            for table in (self._table, self._optics_table)
        )
        self._update_ops_buttons()

    def _apply_operations(self) -> None:
        """Run the stacked options via the engine and refresh the preview."""
        ops = self._stack_model.operations()
        if not ops or self._data is None:
            return
        try:
            stack = [
                (name, parse_operation_value(text, self._specs[name]))
                for name, text in ops
            ]
            args = stack_to_namespace(stack, self._specs)
            append_options = [n for n, s in self._specs.items() if s["append"]]
            data = apply_options(
                self._deep_copy(self._data),
                [name for name, _ in stack],
                args,
                append_options,
            )
        except Exception as exc:
            self._set_status(f"Transform failed: {exc}", error=True)
            return
        self._data = data
        self._set_dirty(True)
        self._refresh_preview(data)
        self._set_status(
            f"Applied {len(stack)} operation(s): {len(data):,} rows \u00d7 "
            f"{len(data.columns)} columns"
        )
        self._update_ops_buttons()

    def _reset_to_source(self) -> None:
        """Discard transforms and restore the original loaded dataset."""
        if self._source_data is None:
            return
        self._stack_model.clear()
        self._data = self._deep_copy(self._source_data)
        self._set_dirty(False)
        self._refresh_preview(self._data)
        self._set_status(self._summary(self._data))
        self._update_ops_buttons()

    def _set_dirty(self, dirty: bool) -> None:
        """Track unsaved transforms and reflect them in the window title."""
        self._dirty = dirty
        base = f"Images2Star - {Path(self._path).name}"
        self.setWindowTitle(base + (" (modified)" if dirty else ""))

    @staticmethod
    def _deep_copy(data: pd.DataFrame) -> pd.DataFrame:
        """Return a deep copy of ``data`` including DataFrame attrs."""
        copy = data.copy(deep=True)
        copy.attrs = {
            key: (value.copy(deep=True) if isinstance(value, pd.DataFrame) else value)
            for key, value in data.attrs.items()
        }
        return copy

    # ------------------------------------------------------------------
    # Load / preview / save

    @staticmethod
    def _section_label(text: str) -> QLabel:
        """Return a bold, selectable section header label (hidden by default)."""
        label = QLabel(text)
        font = QFont(label.font())
        font.setBold(True)
        label.setFont(font)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        label.hide()
        return label

    @staticmethod
    def _make_table() -> _CopyableTableView:
        """Return a cell-editable table hidden until populated."""
        table = _CopyableTableView()
        # The table is created without a parent, so give it the compact font
        # explicitly (the dialog inherits the same font when it is added).
        compact = QFont()
        compact.setPixelSize(12)
        table.setFont(compact)
        # Qt leaves QTableView rows at the style default (~30px) regardless
        # of the font. Size rows from the frameless editor's own sizeHint
        # (one pixel taller, since the cell rect given to the editor is one
        # pixel shorter than the row): the editor then gets exactly the
        # height it needs, so editing never clips the text, whatever the
        # font metrics, style or display scale.
        sample = QLineEdit()
        sample.setFont(table.font())
        sample.setFrame(False)
        sample.setContentsMargins(0, 0, 0, 0)
        sample.setTextMargins(0, 0, 0, 0)
        header = table.verticalHeader()
        header.setMinimumSectionSize(table.fontMetrics().height())
        header.setDefaultSectionSize(sample.sizeHint().height() + 1)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        table.setItemDelegate(_DenseEditDelegate(table))
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        # Keep scrolling through large virtual models cheap: per-pixel
        # scrolling avoids jumping row-by-row through hundreds of thousands
        # of rows, and the Qt 6.9+ item views already size rows lazily.
        table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setWordWrap(False)
        table.setSortingEnabled(True)
        # The QTableView style default minimum (~90px) would stop the splitter
        # from being dragged down to a single-row sliver; ignore the table's
        # height hint so the pane minimum is just its header label (content is
        # clipped while collapsed, not lost).
        table.setSizePolicy(
            table.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Ignored
        )
        table.hide()
        return table

    @staticmethod
    def _table_pane(label: QLabel, table: _CopyableTableView) -> QWidget:
        """Return a header-label + table column kept together in the splitter."""
        pane = QWidget()
        box = QVBoxLayout(pane)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(label)
        box.addWidget(table, 1)
        return pane

    def _on_loaded(self, data: pd.DataFrame) -> None:
        """Populate the particle/optics previews and enable saving."""
        self._source_data = self._deep_copy(data)
        self._data = self._deep_copy(data)
        self._refresh_preview(self._data)
        # Initial split gives particles the bulk and optics a compact strip;
        # later refreshes keep whatever sizes the user dragged to.
        # Use proportional sizes based on available height instead of fixed pixels.
        h = self._splitter.height()
        if h > 0:
            self._splitter.setSizes([int(h * 2 / 3), int(h / 3)])
        self._btn_save.setEnabled(True)
        warnings = list(data.attrs.get("load_warnings", []))
        message = self._summary(self._data)
        if warnings:
            message = f"{message}\n" + "\n".join(warnings)
        self._set_status(message, error=bool(warnings))
        self._update_ops_buttons()

    def _refresh_preview(self, data: pd.DataFrame) -> None:
        """Repopulate the particle/optics preview tables from ``data``."""
        self._table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        missing_files = data.attrs.get("missing_files", ())
        model = _DataFramePreviewModel(data, missing_files=missing_files, parent=self)
        model.dataChanged.connect(self._on_cell_edited)
        self._table.setModel(model)
        self._table.show()
        self._particles_label.show()
        optics = data.attrs.get("optics")
        if optics is not None and len(optics):
            self._optics_table.horizontalHeader().setSortIndicator(
                -1, Qt.SortOrder.AscendingOrder
            )
            optics_model = _DataFramePreviewModel(
                optics, missing_files=missing_files, parent=self
            )
            optics_model.dataChanged.connect(self._on_cell_edited)
            self._optics_table.setModel(optics_model)
            self._optics_table.show()
            self._optics_label.setText("Optics groups")
            self._optics_label.show()
            self._optics_pane.show()
        else:
            self._optics_table.horizontalHeader().setSortIndicator(
                -1, Qt.SortOrder.AscendingOrder
            )
            self._optics_table.setModel(None)
            self._optics_table.hide()
            self._optics_label.hide()
            self._optics_pane.hide()

    def _on_cell_edited(self, top_left, bottom_right, roles=None) -> None:
        """Mark the working dataset dirty after a manual table cell edit."""
        self._set_dirty(True)
        self._update_ops_buttons()
        self._set_status(
            "Cell edited; press Save As\u2026 to export or Reset to discard"
        )

    def _set_status(self, message: str, error: bool = False) -> None:
        """Show a message in the status row, coloring errors with theme red.

        The row is capped at five lines; longer messages (e.g. the list of
        missing files) scroll inside the widget instead of growing the window.
        """
        self._status.setPlainText(message)
        self._status_error = error
        self._status.setStyleSheet(self._status_stylesheet(error))
        self._fit_status_height()

    def _fit_status_height(self) -> None:
        """Size the status row to its content, capped at five rows.

        A taller row would otherwise expand to the layout's shared height;
        this keeps the default single-line band compact while longer
        messages (e.g. the missing-file list) grow up to five rows and then
        scroll inside the widget. The padding shares the extra pixel the
        document needs per line so rows up to five never scroll.
        """
        line_h = self._status.fontMetrics().lineSpacing()
        rows = max(self._status.blockCount(), 1)
        self._status.setFixedHeight(min(rows, 5) * line_h + 4)

    def _status_stylesheet(self, error: bool) -> str:
        """Return the stylesheet for the read-only status row.

        The base rules are repeated here (not left to the group-level QSS)
        because setting a widget-level stylesheet replaces the stylesheet of
        the whole widget, so the compact framing would otherwise be lost as
        soon as the error color is applied.
        """
        from helicon.lib.file_browser import (
            _THEME_COLORS,
            _resolved_theme,
            _saved_theme,
        )

        colors = _THEME_COLORS[_resolved_theme(_saved_theme())]
        parts = ["font-size: 12px;", "border: none;", "background: transparent;"]
        if error:
            parts.insert(0, f"color: {colors.get('error', '#b3261e')};")
        return "QPlainTextEdit { " + " ".join(parts) + " }"

    def _on_load_failed(self, message: str) -> None:
        """Report a failed dataset load and keep the panel usable."""
        self._set_status(f"Failed to load dataset: {message}", error=True)

    def _summary(self, data: pd.DataFrame) -> str:
        """Return a one-line dataset summary for the status label."""
        parts = [f"{len(data):,} rows \u00d7 {len(data.columns)} columns"]
        optics = data.attrs.get("optics")
        if optics is not None and len(optics):
            parts.append(f"{len(optics)} optics group(s)")
        return "  \u2022  ".join(parts)

    def _choose_output(self) -> None:
        """Ask for an output path and start the export."""
        if self._data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Images2Star Output",
            self._default_output_path(),
            _OUTPUT_FILTER,
        )
        if not path:
            return
        self._save_to(path)

    def _save_to(self, path: str) -> None:
        """Save the loaded dataset to ``path`` using the injected saver."""
        self._btn_save.setEnabled(False)
        self._set_status("Saving\u2026")
        worker = _SaveWorker(self._data, path, self._saver, parent=self)
        self._workers.append(worker)
        worker.saved.connect(self._on_saved)
        worker.failed.connect(self._on_save_failed)
        worker.start()

    def _on_saved(self, path: str) -> None:
        """Re-enable saving and report the exported file."""
        self._btn_save.setEnabled(True)
        self._last_output = path
        self._set_status(f"Saved {len(self._data):,} rows to {path}")

    def _on_save_failed(self, message: str) -> None:
        """Re-enable saving and report the export error."""
        self._btn_save.setEnabled(True)
        self._set_status(f"Save failed: {message}", error=True)

    def _on_label_copied(self, text: str) -> None:
        """Report a copied column label in the status bar."""
        if "\n" in text:
            self._set_status(f"Copied {len(text.splitlines())} column labels")
        else:
            self._set_status(f"Copied column label: {text}")

    def closeEvent(self, event) -> None:
        """Give background workers a moment to finish before closing."""
        for worker in self._workers:
            worker.wait(2000)
        super().closeEvent(event)
