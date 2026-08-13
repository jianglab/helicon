"""Qt widgets for the Proc3D tools panel in ``helicon display``.

The panel is a thin front end over :mod:`helicon.lib.proc3d_engine`: the
same ordered plugin dispatch the ``helicon proc3d`` CLI runs, so a GUI
transform is byte-for-byte the CLI semantics. The original volume stays in a
source ortho-slice viewer; every Apply produces a second, side-by-side
result viewer so the transform can be compared against the input at the same
zoom and crosshair position.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QFontDatabase, QKeySequence, QShortcut, QTextOption
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
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import mrcfile

from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget
from helicon.lib.images2star_engine import parse_operation_value
from helicon.lib.gui.images2star_widget import _OperationStackModel
from helicon.lib.proc3d_engine import (
    apply_options,
    gui_operation_specs,
    stack_to_namespace,
)

_MRC_FILTER = "MRC map (*.mrc *.map);;All files (*)"

# Friendlier parameter hints than the generic ``<param>=<val>:...`` metavar
# every proc3d plugin registers. Used as the QLineEdit placeholder so the
# panel reads as a form instead of a raw CLI string.
_PARAM_PLACEHOLDERS = {
    "apix": "pixel size in \u00c5/pixel, e.g. 1.35",
    "clip": "new_nx=..:new_ny=..:new_nz=.. (optional center_x/y/z=..)",
    "denoiseCurvelet": "sigma=..:numScales=..:transform=udct|mct",
    "fft_resample": "new_nx=..:new_ny=..:new_nz=..",
    "flip_hand": "axis to flip: x, y, or z",
    "helical_sym": "twist=..:rise=.. (optional csym=.., new_apix=.., "
    "new_nz=.., new_nxy=.., center_*=..)",
    "z_moving_average": "length=.. (\u00c5) or n_pixel=.., not both",
}


def _load_volume(path: str) -> tuple[np.ndarray, float]:
    """Load a 3D MRC map as ``(float32 data, apix)`` for the panel.

    Mirrors how the display app's orthogonal gallery reads volumes: the
    pixel size falls back to 1.0 when the file does not record one. The data
    is eagerly copied out of the mrcfile so the array stays valid after the
    file handle closes.

    Parameters
    ----------
    path : str
        Path to a 3D MRC/MAP file.

    Returns
    -------
    tuple of (np.ndarray, float)
        Volume with shape ``(nz, ny, nx)`` and pixel size in Angstroms.

    Raises
    ------
    ValueError
        If the file is not a 3D volume.
    """
    with mrcfile.open(path, permissive=True) as mrc:
        data = mrc.data
        if data is None or data.ndim != 3:
            raise ValueError(f"not a 3D volume: {path}")
        if min(data.shape) == 0:
            raise ValueError(f"empty volume (zero-size dimension): {path}")
        apix = float(mrc.voxel_size.x) if mrc.voxel_size.x > 0 else 1.0
    return np.array(data, dtype=np.float32), apix


def _write_volume(data: np.ndarray, apix: float, path: str) -> None:
    """Write a float32 MRC map with the given pixel size (overwrites)."""
    with mrcfile.new(
        path, data=np.ascontiguousarray(data, dtype=np.float32), overwrite=True
    ) as mrc:
        mrc.voxel_size = apix


def _volume_summary(data: np.ndarray, apix: float) -> str:
    """Return a one-line ``nx×ny×nz pixels, apix Å/pixel`` description."""
    nz, ny, nx = data.shape
    return f"{nx}\u00d7{ny}\u00d7{nz} pixels, {apix:.4g} \u00c5/pixel"


class _VolumeLoadWorker(QThread):
    """Load a 3D MRC map off the UI thread."""

    loaded = Signal(object, float)
    failed = Signal(str)

    def __init__(self, path: str, loader, parent=None):
        super().__init__(parent)
        self._path = path
        self._loader = loader

    def run(self):
        try:
            data, apix = self._loader(self._path)
            self.loaded.emit(data, apix)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _VolumeSaveWorker(QThread):
    """Write a volume to an MRC file off the UI thread."""

    saved = Signal(str)
    failed = Signal(str)

    def __init__(self, data: np.ndarray, apix: float, path: str, saver, parent=None):
        super().__init__(parent)
        self._data = data
        self._apix = apix
        self._path = path
        self._saver = saver

    def run(self):
        try:
            self._saver(self._data, self._apix, self._path)
            self.saved.emit(self._path)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class Proc3dDialog(QDialog):
    """Preview, transform, and save a 3D map with the proc3d engine.

    The first row is a file selector: a path field with Browse and Load
    buttons. Launching the panel from a file (button action) pre-fills and
    loads that map; launching it from the Apps menu leaves the selector empty
    so the user picks a map directly inside the panel. Once a 3D map is
    loaded with ``_load_volume``, it appears in an interactive ortho-slice
    viewer (the same ``OrthogonalViewerWidget`` the "Ortho Slice" button
    opens), and the user can build an ordered stack of in-memory ``proc3d``
    options (``--flip_hand``, ``--clip``, ``--apix``, ...). Applying the stack
    runs the exact CLI dispatch loop
    (:func:`~helicon.lib.proc3d_engine.apply_options`); the result appears in
    a second, side-by-side ortho-slice viewer so the transform can be
    compared against the original. The current (possibly transformed) volume
    is exported with ``mrcfile.new`` using the engine's pixel size.

    Parameters
    ----------
    path : str, optional
        Path to a 3D MRC/MAP map to pre-load. When omitted or empty the panel
        opens with an empty file selector so the user can choose a map inside
        it.
    parent : QWidget, optional
        Parent widget for the dialog.
    loader : callable, optional
        Callable ``loader(path) -> (data, apix)`` overriding the default
        :func:`_load_volume`.
    saver : callable, optional
        Callable ``saver(data, apix, path)`` overriding the default
        :func:`_write_volume`.
    """

    def __init__(
        self,
        path: str | None = None,
        parent=None,
        loader=None,
        saver=None,
    ):
        super().__init__(parent)
        self.setProperty("helicon_theme_window", True)
        self._status_error = False
        self._path = str(Path(path).resolve()) if path else ""
        self._loader = loader or _load_volume
        self._saver = saver or _write_volume
        self._specs = gui_operation_specs()
        self._source_volume: np.ndarray | None = None
        self._source_apix = 1.0
        self._volume: np.ndarray | None = None
        self._apix = 1.0
        self._dirty = False
        self._last_output: str | None = None
        self._workers: list[QThread] = []
        self._load_seq = 0
        self._sized_once = False

        self._set_window_title()
        self.resize(1200, 820)

        compact = QFont()
        compact.setPixelSize(12)
        self.setFont(compact)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        file_row = QHBoxLayout()
        file_row.setSpacing(4)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to a 3D MRC/MAP map...")
        self._path_edit.setToolTip(
            "Path to the 3D map to preview and transform (type it, paste it, "
            "or pick it with Browse)"
        )
        self._path_edit.setClearButtonEnabled(True)
        self._path_edit.returnPressed.connect(self._load_from_field)
        self._btn_browse = self._compact_button(QPushButton("Browse..."))
        self._btn_browse.setToolTip("Choose a 3D MRC/MAP map from disk")
        self._btn_browse.clicked.connect(self._browse_for_map)
        file_row.addWidget(self._path_edit, 1)
        file_row.addWidget(self._btn_browse)
        layout.addLayout(file_row)
        if self._path:
            self._path_edit.setText(self._path)

        # Placeholder viewers: the result pane stays hidden until the first
        # Apply so the source never shares screen space with an untransformed
        # copy. Both viewers start on the same array (no data copy).
        self._source_label = self._section_label("Original")
        self._result_label = self._section_label("Result")
        self._source_viewer = OrthogonalViewerWidget(
            np.zeros((1, 1, 1), dtype=np.float32), apix=1.0, name="Original"
        )
        self._result_viewer = OrthogonalViewerWidget(
            np.zeros((1, 1, 1), dtype=np.float32), apix=1.0, name="Result"
        )
        self._source_pane = self._viewer_pane(self._source_label, self._source_viewer)
        self._result_pane = self._viewer_pane(self._result_label, self._result_viewer)
        self._result_pane.hide()

        self._preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._preview_splitter.setChildrenCollapsible(True)
        self._preview_splitter.addWidget(self._source_pane)
        self._preview_splitter.addWidget(self._result_pane)
        self._preview_splitter.setStretchFactor(0, 1)
        self._preview_splitter.setStretchFactor(1, 1)

        self._ops_group = self._make_operations_group()
        self._main_splitter = QSplitter(Qt.Orientation.Vertical)
        self._main_splitter.setChildrenCollapsible(True)
        self._main_splitter.addWidget(self._preview_splitter)
        self._main_splitter.addWidget(self._ops_group)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 1)
        layout.addWidget(self._main_splitter, 1)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        self._status = QPlainTextEdit()
        self._status.setReadOnly(True)
        self._status.setPlainText(
            "Loading volume\u2026"
            if self._path
            else "Select a 3D map (MRC/MAP) above to begin"
        )
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
            "Write the current volume to an MRC/MAP file (keeps the current "
            "pixel size)"
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

        if self._path:
            self._load_worker = _VolumeLoadWorker(self._path, self._loader, parent=self)
            self._workers.append(self._load_worker)
            self._load_worker.loaded.connect(self._on_loaded)
            self._load_worker.failed.connect(self._on_load_failed)
            self._load_worker.start()
        else:
            self._update_ops_buttons()
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

    def _set_window_title(self) -> None:
        """Set the window title from the current map path, or a bare label.

        With no map loaded the title is just "Proc3D"; once a map is chosen
        it becomes "Proc3D - <map-name>".
        """
        name = Path(self._path).name if self._path else ""
        title = "Proc3D" if not name else f"Proc3D - {name}"
        self.setWindowTitle(title)

    def _browse_for_map(self) -> None:
        """Open a file picker and load the chosen 3D MRC/MAP map."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select 3D map",
            self._path or "",
            _MRC_FILTER,
        )
        if not file_name:
            return
        if file_name != self._path:
            self._path_edit.setText(file_name)
            self.load_path(file_name)
        else:
            self._path_edit.setText(file_name)

    def _load_from_field(self) -> None:
        """Load the map whose path is typed into the file selector."""
        path = self._path_edit.text().strip()
        if not path:
            self._set_status("Enter a path to a 3D map first", error=True)
            return
        if path != self._path:
            self.load_path(path)
        else:
            self._set_status(self._summary() or "3D map already loaded")

    def load_path(self, path: str) -> None:
        """Reload this panel from a new file, reusing the same window.

        The transformation stack is kept (it is a recipe the user may want to
        apply to another map); the viewers and dirty state are reset until
        the new volume finishes loading.
        """
        self._path = str(Path(path).resolve())
        self._path_edit.setText(self._path)
        self._set_window_title()
        self._source_volume = None
        self._volume = None
        self._dirty = False
        self._last_output = None
        self._result_pane.hide()
        self._btn_save.setEnabled(False)
        self._set_status("Loading volume\u2026")

        # Guard against a slower previous load finishing after this one: only
        # the most recent request may populate the viewers.
        self._load_seq += 1
        seq = self._load_seq
        worker = _VolumeLoadWorker(self._path, self._loader, parent=self)
        self._workers.append(worker)
        worker.loaded.connect(
            lambda data, apix: (
                self._on_loaded(data, apix) if seq == self._load_seq else None
            )
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
        the user switches themes while the dialog is open. The embedded
        ortho-slice viewers carry their own gallery theming, so they are
        refreshed through ``_apply_gallery_theme``.
        """
        from PySide6.QtWidgets import QWidget

        from helicon.commands.display import (
            _display_theme_palette,
            _display_theme_stylesheet,
        )

        stylesheet = _display_theme_stylesheet()
        palette = _display_theme_palette()
        self.setStyleSheet(stylesheet)
        self.setPalette(palette)
        for child in self.findChildren(QWidget):
            child.setStyleSheet(stylesheet)
            child.setPalette(palette)
        # Refresh the embedded ortho viewers via their own theme hook. We must
        # NOT call ``_apply_gallery_theme(self)`` here: that helper re-invokes
        # ``_apply_display_theme`` on any child that defines it, and since this
        # dialog itself defines it the call would recurse infinitely. The two
        # ``OrthogonalViewerWidget`` instances own their own (safe) theme
        # logic, so call them directly instead.
        for viewer in (self._source_viewer, self._result_viewer):
            viewer._apply_display_theme()
        if self._status_error:
            self._status.setStyleSheet(self._status_stylesheet(error=True))

    # ------------------------------------------------------------------
    # Operations stack panel

    @staticmethod
    def _compact_button(button: QPushButton) -> QPushButton:
        """Give a button the browser's fixed 26px action-bar height."""
        button.setFixedHeight(26)
        # This dialog embeds OrthogonalViewerWidget spinboxes; without
        # auto-default-off, Enter inside a spinbox would activate the nearest
        # QPushButton (Movie/Apply/Close…) instead of committing the value.
        button.setAutoDefault(False)
        return button

    def _apply_default_split_sizes(self) -> None:
        """Apply the default split sizes for both splitters.

        Runs once on first show, when the splitter's real height is known.
        The transformations block defaults to the height its (compact)
        contents request with the option list showing two full rows; the
        preview splitter gives the source and result panes equal widths once
        the result pane is visible.
        """
        total = self._main_splitter.height()
        if total <= 0:
            return
        ops = self._ops_group.sizeHint().height()
        self._main_splitter.setSizes([max(total - ops, 0), ops])
        if not self._result_pane.isHidden():
            w = self._preview_splitter.width()
            if w > 0:
                self._preview_splitter.setSizes([w // 2, w - w // 2])

    def showEvent(self, event) -> None:
        """Apply the default split once so user drags are never reset."""
        super().showEvent(event)
        if not self._sized_once:
            self._sized_once = True
            self._apply_default_split_sizes()

    def _make_operations_group(self) -> QGroupBox:
        """Build the ordered option-stack editor (add/apply/reset)."""
        group = QGroupBox("Transformations (proc3d options)")
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
        self._param_edit.setPlaceholderText(
            "parameter value(s), e.g. --flip_hand x or --clip new_nx=..:new_nz=.."
        )
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
        self._stack_view.setSizePolicy(
            self._stack_view.sizePolicy().horizontalPolicy(),
            QSizePolicy.Policy.Ignored,
        )
        self._stack_view.setMinimumHeight(
            2 * self._stack_view.fontMetrics().lineSpacing()
            + 2 * self._stack_view.frameWidth()
        )
        stack_row.addWidget(self._stack_view, 1)
        outer.addLayout(stack_row)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(4)
        self._btn_apply = self._compact_button(QPushButton("Apply"))
        self._btn_apply.setToolTip(
            "Run the stacked options through the proc3d engine and show the "
            "transformed volume in the Result viewer (same semantics as the CLI)"
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
            "Show the equivalent helicon proc3d command to copy or save as "
            "a script for batch reruns"
        )
        self._btn_cmd.clicked.connect(self._show_command)
        self._btn_reset = self._compact_button(QPushButton("Reset"))
        self._btn_reset.setToolTip(
            "Discard all transforms and restore the original volume as loaded"
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
        """Show the selected option's hint as the parameter placeholder."""
        name = self._ops_combo.currentData()
        if name is None:
            return
        spec = self._specs[name]
        self._param_edit.setPlaceholderText(
            _PARAM_PLACEHOLDERS.get(name, spec["metavar"] or f"--{name} parameter")
        )
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
        self._set_status("Added --%s; press Apply to transform the volume" % name)
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
        """Default export path: the last saved file, else the CLI fallback.

        The fallback is ``<stem>.proc3d.mrc`` next to the input, matching
        what ``helicon proc3d in.mrc`` would write.

        Returns
        -------
        str
            Path to offer as the default output location.
        """
        if self._last_output:
            return self._last_output
        source = Path(self._path)
        if not self._path:
            return str(Path.cwd() / "volume.proc3d.mrc")
        return str(source.with_name(source.stem + ".proc3d.mrc"))

    def _command_text(self, output: str | None = None) -> str:
        """Return the equivalent ``helicon proc3d`` shell command.

        The stacked operations map 1:1 to CLI options: the user-entered
        parameter text is exactly what ``parse_operation_value`` would feed
        argparse, so shell-quoting each token reproduces the command the CLI
        would run. The output path defaults to :meth:`_default_output_path`.

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
        parts = ["helicon", "proc3d", shlex.quote(self._path), shlex.quote(output)]
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
        dlg.setWindowTitle("Equivalent proc3d command")
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
        default = Path(self._path).with_name(Path(self._path).stem + ".proc3d.sh")
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

        Reset is available whenever the working volume differs from the
        original, and Save/Show Command once a volume is loaded.
        """
        has_ops = self._stack_model.rowCount() > 0
        self._btn_apply.setEnabled(has_ops)
        self._btn_clear.setEnabled(has_ops)
        self._btn_cmd.setEnabled(self._volume is not None)
        index = self._stack_view.currentIndex()
        has_selection = index.isValid()
        row = index.row() if has_selection else -1
        self._btn_remove.setEnabled(has_selection)
        self._btn_up.setEnabled(has_selection and row > 0)
        self._btn_down.setEnabled(
            has_selection and row < self._stack_model.rowCount() - 1
        )
        self._btn_reset.setEnabled(self._volume is not None and self._dirty)

    def _apply_operations(self) -> None:
        """Run the stacked options via the engine and refresh the result.

        The whole stack is always applied to the original volume (never
        compounded onto the previous result), so the Result viewer shows
        exactly what ``helicon proc3d <in> <out> <stack>`` would produce and
        stays in sync with the equivalent command shown by "Show Command".
        """
        ops = self._stack_model.operations()
        if not ops or self._volume is None:
            return
        try:
            stack = [
                (name, parse_operation_value(text, self._specs[name]))
                for name, text in ops
            ]
            args = stack_to_namespace(stack, self._specs)
            data, apix = apply_options(
                self._source_volume,
                self._source_apix,
                [name for name, _ in stack],
                args,
            )
        except Exception as exc:
            self._set_status(f"Transform failed: {exc}", error=True)
            return
        self._volume = data
        self._apix = apix
        self._set_dirty(True)
        self._result_viewer.set_volume(data, apix, reset_position=True)
        self._result_label.setText(f"Result \u2014 {_volume_summary(data, apix)}")
        self._result_pane.show()
        if self._sized_once:
            self._apply_default_split_sizes()
        self._set_status(
            f"Applied {len(stack)} operation(s): " f"{_volume_summary(data, apix)}"
        )
        self._update_ops_buttons()

    def _reset_to_source(self) -> None:
        """Discard transforms and restore the original loaded volume."""
        if self._source_volume is None:
            return
        self._stack_model.clear()
        self._volume = self._source_volume
        self._apix = self._source_apix
        self._set_dirty(False)
        self._result_pane.hide()
        self._set_status(self._summary())
        self._update_ops_buttons()

    def _set_dirty(self, dirty: bool) -> None:
        """Track unsaved transforms and reflect them in the window title."""
        self._dirty = dirty
        self._set_window_title()
        base = self.windowTitle()
        self.setWindowTitle(base + (" (modified)" if dirty else ""))

    # ------------------------------------------------------------------
    # Load / preview / save

    @staticmethod
    def _viewer_pane(label: QLabel, viewer: OrthogonalViewerWidget) -> QWidget:
        """Return a header-label + ortho-viewer column kept in the splitter.

        The viewer is wrapped with the same toggleable control panel the
        standalone "Ortho Slice" window uses, so middle-clicking any of its
        slice panels shows/hides the brightness / contrast / gamma controls
        (``OrthogonalViewerWidget.panel_toggle_requested``).
        """
        from helicon.commands.display import _wrap_gallery_with_panel

        container = _wrap_gallery_with_panel(viewer)
        pane = QWidget()
        box = QVBoxLayout(pane)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(label)
        box.addWidget(container, 1)
        return pane

    @staticmethod
    def _section_label(text: str) -> QLabel:
        """Return a bold section label matching the images2star panel."""
        label = QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    def _on_loaded(self, data: np.ndarray, apix: float) -> None:
        """Populate both viewers and enable saving."""
        self._source_volume = data
        self._source_apix = apix
        self._volume = data
        self._apix = apix
        self._source_viewer.set_volume(data, apix, reset_position=True)
        self._source_label.setText(f"Original \u2014 {_volume_summary(data, apix)}")
        self._result_viewer.set_volume(data, apix, reset_position=True)
        self._result_label.setText(f"Result \u2014 {_volume_summary(data, apix)}")
        self._result_pane.hide()
        self._btn_save.setEnabled(True)
        self._set_status(self._summary())
        self._update_ops_buttons()

    def _summary(self) -> str:
        """Return a one-line volume summary for the status label."""
        if self._volume is None:
            return "No volume loaded"
        return _volume_summary(self._volume, self._apix)

    def _set_status(self, message: str, error: bool = False) -> None:
        """Show a message in the status row, coloring errors with theme red."""
        self._status.setPlainText(message)
        self._status_error = error
        self._status.setStyleSheet(self._status_stylesheet(error))
        self._fit_status_height()

    def _fit_status_height(self) -> None:
        """Size the status row to its content, capped at five rows."""
        line_h = self._status.fontMetrics().lineSpacing()
        rows = max(self._status.blockCount(), 1)
        self._status.setFixedHeight(min(rows, 5) * line_h + 4)

    def _status_stylesheet(self, error: bool) -> str:
        """Return the stylesheet for the read-only status row."""
        from helicon.lib.gui.file_browser import (
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
        """Report a failed volume load and keep the panel usable."""
        self._set_status(f"Failed to load volume: {message}", error=True)

    def _choose_output(self) -> None:
        """Ask for an output path and start the export."""
        if self._volume is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Proc3D Output",
            self._default_output_path(),
            _MRC_FILTER,
        )
        if not path:
            return
        self._save_to(path)

    def _save_to(self, path: str) -> None:
        """Save the current volume to ``path`` using the injected saver."""
        self._btn_save.setEnabled(False)
        self._set_status("Saving\u2026")
        worker = _VolumeSaveWorker(
            self._volume, self._apix, path, self._saver, parent=self
        )
        self._workers.append(worker)
        worker.saved.connect(self._on_saved)
        worker.failed.connect(self._on_save_failed)
        worker.start()

    def _on_saved(self, path: str) -> None:
        """Re-enable saving and report the exported file."""
        self._btn_save.setEnabled(True)
        self._last_output = path
        self._set_status(f"Saved {_volume_summary(self._volume, self._apix)} to {path}")

    def _on_save_failed(self, message: str) -> None:
        """Re-enable saving and report the export error."""
        self._btn_save.setEnabled(True)
        self._set_status(f"Save failed: {message}", error=True)

    def closeEvent(self, event) -> None:
        """Give background workers a moment to finish before closing."""
        for worker in self._workers:
            worker.wait(2000)
        super().closeEvent(event)
