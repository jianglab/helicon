#!/usr/bin/env python

"""A file browser for viewing image, map, star, bild, eps, pdf, html, and text files.

This module is the coordinator for the ``helicon display`` command. The
implementation lives in ``helicon.lib.gui``; every moved name is
re-exported here so existing callers that import from
``helicon.commands.display`` keep working unchanged.
"""

from __future__ import annotations

import sys
import warnings

# Suppress harmless mrcfile divide-by-zero warnings when reading headers.
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
    module="mrcfile",
)

# On macOS, set the process name before NSApplication is initialized
# (triggered by PySide6 import below). NSApplication caches the app name
# from getprogname() at first init; after that it can't be changed.
if sys.platform == "darwin":
    try:
        import ctypes
        import ctypes.util

        ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True).setprogname(
            b"Helicon"
        )
    except Exception:
        pass

import argparse
import os
from pathlib import Path

import helicon
from helicon.lib.exceptions import HeliconDependencyError

try:
    from helicon.lib.gui.file_browser import FolderBrowserWidget
except ImportError:
    FolderBrowserWidget = None

# Keep the module importable without the Qt stack (display.py degrades to a
# HeliconDependencyError in main() when PySide6 is missing).  Only
# ``_WindowActivationFilter`` needs a Qt base class at import time.
try:
    from PySide6.QtCore import QObject
except ImportError:  # pragma: no cover - only without the Qt stack
    QObject = None

# ---------------------------------------------------------------------------
# Implementation lives in helicon.lib.gui; re-export every name so
# ``from helicon.commands.display import X`` keeps working for all callers
# (file browser, gallery backends/widgets, proc3d/images2star tools, tests).
# ---------------------------------------------------------------------------
from helicon.lib.gui.file_openers import (  # noqa: E402
    _is_text_file,
    _open_bild,
    _open_eps,
    _open_html,
    _open_pdf,
    _open_text_window,
    _rasterize_eps,
)
from helicon.lib.gui.fsc import (  # noqa: E402
    _FlowContainer,
    _FlowLayout,
    _FscPlotWindow,
    _ITERATION_FILE_RE,
    _distinct_curve_colors,
    _iteration_label,
    _iteration_model_files,
    _model_fsc_curves,
    _open_fsc_plot,
    _read_fsc_curves,
)
from helicon.lib.gui.macos import (  # noqa: E402
    _force_macos_menu_realization,
    _force_x11_platform_under_wslg,
    _macos_activate_and_front,
    _macos_activate_pid,
    _macos_class,
    _macos_frontmost_pid,
    _macos_menu_item_count,
    _macos_msg,
    _macos_native_windows,
    _macos_ns_app,
    _macos_resign_active,
    _macos_sel,
    _prepare_process_identity,
    _qt_argv,
    _set_application_identity,
    _set_macos_app_identity,
    _xcb_platform_available,
)
from helicon.lib.gui.star_parsers import (  # noqa: E402
    _METADATA_STAR_SUFFIXES,
    _find_model_star_from_optimiser,
    _is_metadata_star,
    _is_optimiser_star,
    _parse_class2d_model_star,
    _parse_model_star,
    _parse_optimiser_star,
    _parse_star_image_refs,
)
from helicon.lib.gui.theme import (  # noqa: E402
    _display_plot_theme_colors,
    _display_theme_palette,
    _display_theme_stylesheet,
    _get_display_theme,
    _napari_canvas_background,
    _napari_display_theme,
    _refresh_display_theme_windows,
    _refresh_napari_theme,
)
from helicon.lib.gui.trackers import (  # noqa: E402
    _DisplayTracker,
    _GALLERY_MODES,
    _IMAGES2STAR_MODES,
    _NAPARI_MODES,
    _PLOT_MODES,
    _PROC3D_MODES,
    _TEXT_MODES,
    _TRACKER_FOR,
    _gallery,
    _images2star,
    _install_window_shortcuts,
    _is_alive_viewer,
    _is_alive_widget,
    _napari,
    _plot,
    _proc3d,
    _quit_all_windows,
    _text,
)
from helicon.lib.gui.viewer import (  # noqa: E402
    _LazyStarStack,
    _SliceDirectionWidget,
    _add_welcome_shortcut,
    _auto_contrast,
    _create_napari_viewer,
    _crop_to_content,
    _enable_continuous_auto_contrast,
    _get_qsettings,
    _hide_layer_panels,
    _install_dock_save_hook,
    _install_panel_toggle,
    _install_viewer_save_hook,
    _install_viewer_save_menu,
    _is_wsl,
    _launch_chimerax,
    _load_napari,
    _patch_napari_icon,
    _patch_napari_value_bug,
    _position_default,
    _render_qimage_vector,
    _reset_view,
    _restore_geometry,
    _save_geometry,
    _save_qimage,
    _save_viewport,
    _set_ndisplay,
    _viewer_source_name,
)
from helicon.lib.gui.webapps import (  # noqa: E402
    _WEB_APP_INSTANCES,
    _WebAppState,
    _launch_denovo3d,
    _launch_helicalpitch,
    _launch_helicalprojection,
    _launch_hi3d,
    _launch_hill,
    _launch_or_reuse_web_app,
    _launch_whereismyclass,
    _make_bookmark_query,
    _navigate_web_app,
    _spawn_web_app,
    _terminate_web_apps,
    _web_app_active,
    _web_app_alive,
)

# MRC-format image extensions. RELION/CTFFIND write CTF power spectra as MRC
# maps with a ``.ctf`` suffix (e.g. ``*_PS.ctf``).
_MRC_EXTENSIONS = {".mrc", ".mrcs", ".map", ".ctf"}


def _launch_truefsc(path: str, parent=None) -> None:
    """Compute True FSC from the two half-maps referenced by a model.star file.

    Opens the trueFSC panel with the Map 1 and Map 2 selectors pre-filled and
    starts loading the maps immediately.
    """
    import re
    from pathlib import Path

    model_path = Path(path)
    model_dir = model_path.parent
    model_name = model_path.name

    match = re.match(r"(run_it\d+)_half(\d)_", model_name)
    if match:
        prefix = match.group(1)
        map1 = model_dir / f"{prefix}_half1_class001.mrc"
        map2 = model_dir / f"{prefix}_half2_class001.mrc"
    elif model_name == "run_model.star":
        map1 = model_dir / "run_half1_class001_unfil.mrc"
        map2 = model_dir / "run_half2_class001_unfil.mrc"
    else:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(
            None,
            "trueFSC Error",
            f"Cannot determine half-maps from:\n{model_name}",
        )
        return

    if not map1.exists() or not map2.exists():
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(
            None,
            "trueFSC Error",
            f"Half-maps not found:\n{map1}\n{map2}",
        )
        return

    _launch_truefsc_maps(map1=map1, map2=map2, parent=parent)


def _launch_truefsc_maps(map1=None, map2=None, mask=None, parent=None) -> None:
    """Open the trueFSC panel with Map 1 / Map 2 / optional Mask selectors.

    The information pane holds three file selectors: Map 1, Map 2, and an
    optional Mask. As soon as both half-maps are chosen the two input ortho
    viewers load and the True FSC computation runs in the background; a mask
    given in the selector is used verbatim, otherwise an adaptive mask is
    generated. ``map1`` and ``map2`` pre-fill the selectors when the panel is
    launched from a ``model.star`` action button; launching from the Apps
    menu leaves every selector empty.

    ``map1``, ``map2``, and ``mask`` may be ``str``, ``pathlib.Path``, or
    ``None``.

    Returns
    -------
    QDialog
        The open, non-modal progress dialog.
    """
    import logging
    import os
    import tempfile
    from pathlib import Path

    import numpy as np

    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QDialog,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    from helicon.commands.trueFSC import compute_truefsc
    from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget
    from helicon.lib.gui.proc3d_widget import _load_volume

    def _section_label(text: str) -> QLabel:
        label = QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    def _ortho_pane(label_text: str, viewer: OrthogonalViewerWidget) -> QWidget:
        """Return a bold header plus a toggleable ortho-viewer column."""
        container = _wrap_gallery_with_panel(viewer)
        pane = QWidget()
        box = QVBoxLayout(pane)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(_section_label(label_text))
        box.addWidget(container, 1)
        return pane

    def _output_dir_for(map1_path: str) -> Path:
        """Return a writable output directory next to ``map1_path``."""
        base_dir = Path(map1_path).parent
        if os.access(base_dir, os.W_OK):
            return base_dir
        return Path(tempfile.mkdtemp(prefix="helicon_truefsc_"))

    class LogHandler(logging.Handler):
        def __init__(self, signal):
            super().__init__()
            self._signal = signal

        def emit(self, record):
            msg = self.format(record)
            self._signal.emit(msg)

    class _InputLoader(QThread):
        """Load the chosen maps (and optional mask) off the UI thread."""

        loaded = Signal(object)
        failed = Signal(str)

        def __init__(self, map1, map2, mask, parent=None):
            super().__init__(parent)
            self._map1 = map1
            self._map2 = map2
            self._mask = mask

        def run(self):
            try:
                if self._map1:
                    d1, a1 = _load_volume(self._map1)
                else:
                    d1, a1 = None, None
                if self._map2:
                    d2, a2 = _load_volume(self._map2)
                else:
                    d2, a2 = None, None
                mask_vol = None
                if self._mask:
                    mask_vol, _ = _load_volume(self._mask)
                self.loaded.emit({"map1": (d1, a1), "map2": (d2, a2), "mask": mask_vol})
            except Exception as exc:
                self.failed.emit(f"{type(exc).__name__}: {exc}")

    class _ComputeWorker(QThread):
        """Run ``compute_truefsc`` off the UI thread, streaming log lines."""

        line_received = Signal(str)
        finished = Signal(object)
        error = Signal(str)

        def __init__(self, map1, map2, mask, plot_file, parent=None):
            super().__init__(parent)
            self._map1 = map1
            self._map2 = map2
            self._mask = mask
            self._plot_file = plot_file

        def run(self):
            try:
                self.line_received.emit(f"Map 1: {self._map1}")
                self.line_received.emit(f"Map 2: {self._map2}")
                if self._mask:
                    self.line_received.emit(f"Mask: {self._mask}")
                self.line_received.emit(f"Output: {self._plot_file}")
                self.line_received.emit("")

                handler = LogHandler(self.line_received)
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger = logging.getLogger("helicon.commands.trueFSC")
                logger.addHandler(handler)
                logger.setLevel(logging.DEBUG)
                try:
                    result = compute_truefsc(
                        self._map1,
                        self._map2,
                        self._plot_file,
                        mask_file=[self._mask] if self._mask else None,
                    )
                finally:
                    logger.removeHandler(handler)
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))

    class ProgressDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setStyleSheet(_display_theme_stylesheet())
            self.setWindowTitle("trueFSC")
            self.resize(1200, 900)
            self.setMinimumSize(800, 500)
            # Track background workers via plain Python state so we never
            # touch a C++ QThread wrapper after it has been deleteLater'd.
            self._workers: list[tuple[QThread, list]] = []
            self._seq = 0
            self._map1 = str(Path(map1).resolve()) if map1 else ""
            self._map2 = str(Path(map2).resolve()) if map2 else ""
            self._mask = str(Path(mask).resolve()) if mask else ""

            root = QVBoxLayout(self)
            grid = QGridLayout()
            grid.setSpacing(6)
            root.addLayout(grid, 1)

            placeholder = np.zeros((1, 1, 1), dtype=np.float32)

            # Column 1: the two input maps, shown as soon as they load.
            self._map1_viewer = OrthogonalViewerWidget(
                placeholder, apix=1.0, name="Map 1"
            )
            self._map2_viewer = OrthogonalViewerWidget(
                placeholder, apix=1.0, name="Map 2"
            )
            # Columns 2 and 3: filled in once the computation finishes.
            self._mask_viewer = OrthogonalViewerWidget(
                placeholder, apix=1.0, name="Mask"
            )
            self._masked1_viewer = OrthogonalViewerWidget(
                placeholder, apix=1.0, name="Masked map 1"
            )
            self._masked2_viewer = OrthogonalViewerWidget(
                placeholder, apix=1.0, name="Masked map 2"
            )

            grid.addWidget(_ortho_pane("Map 1", self._map1_viewer), 0, 0)
            grid.addWidget(_ortho_pane("Map 2", self._map2_viewer), 1, 0)
            grid.addWidget(_ortho_pane("Mask", self._mask_viewer), 0, 1)
            grid.addWidget(_ortho_pane("Masked map 1", self._masked1_viewer), 0, 2)
            grid.addWidget(_ortho_pane("Masked map 2", self._masked2_viewer), 1, 2)

            # Column 2, row 2: the input map selectors plus the info log.
            selector_panel = self._build_selector_panel()
            info_box = QVBoxLayout()
            info_box.setContentsMargins(0, 0, 0, 0)
            info_box.setSpacing(2)
            info_box.addWidget(_section_label("Information"))
            self.text_edit = QTextEdit()
            self.text_edit.setReadOnly(True)
            info_box.addWidget(self.text_edit, 1)
            info_widget = QWidget()
            info_col = QVBoxLayout(info_widget)
            info_col.setContentsMargins(0, 0, 0, 0)
            info_col.setSpacing(2)
            info_col.addWidget(selector_panel)
            info_col.addLayout(info_box, 1)
            grid.addWidget(info_widget, 1, 1)

            if self._map1:
                self._map1_edit.setText(self._map1)
            if self._map2:
                self._map2_edit.setText(self._map2)
            if self._mask:
                self._mask_edit.setText(self._mask)

        # ------------------------------------------------------------------
        # Input selectors
        # ------------------------------------------------------------------

        def _compact_button(self, text: str, tooltip: str) -> QPushButton:
            """Give a button the browser's fixed 26px action-bar height."""
            button = QPushButton(text)
            button.setFixedHeight(26)
            button.setAutoDefault(False)
            button.setToolTip(tooltip)
            return button

        def _build_selector_panel(self) -> QWidget:
            """Build the Map 1 / Map 2 / optional Mask file selectors."""
            panel = QWidget()
            box = QVBoxLayout(panel)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(2)
            box.addWidget(_section_label("Input maps"))

            self._map1_edit, row1 = self._selector_row(
                "Map 1", "Choose the half-map 1 3D MRC/MAP map from disk"
            )
            self._map2_edit, row2 = self._selector_row(
                "Map 2", "Choose the half-map 2 3D MRC/MAP map from disk"
            )
            self._mask_edit, row3 = self._selector_row(
                "Mask",
                "Optional mask used verbatim; empty generates an adaptive mask",
            )
            box.addLayout(row1)
            box.addLayout(row2)
            box.addLayout(row3)
            return panel

        def _selector_row(
            self, label_text: str, tooltip: str
        ) -> tuple[QLineEdit, QHBoxLayout]:
            """Return a labeled path field with a Browse button."""
            row = QHBoxLayout()
            row.setSpacing(4)
            label = QLabel(label_text)
            label.setMinimumWidth(48)
            row.addWidget(label)
            edit = QLineEdit()
            edit.setClearButtonEnabled(True)
            edit.setPlaceholderText("Path to a 3D MRC/MAP map...")
            edit.returnPressed.connect(self._on_inputs_changed)
            row.addWidget(edit, 1)
            browse = self._compact_button("Browse...", tooltip)
            browse.clicked.connect(lambda: self._browse_for(edit))
            row.addWidget(browse)
            return edit, row

        def _browse_for(self, edit: QLineEdit) -> None:
            """Open a file picker and immediately load the chosen map."""
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select 3D map",
                edit.text() or self._map1 or str(Path.home()),
                "Maps (*.mrc *.mrcs *.em *.map);;All files (*)",
            )
            if not file_name:
                return
            edit.setText(file_name)
            self._on_inputs_changed()

        # ------------------------------------------------------------------
        # Load-and-compute pipeline
        # ------------------------------------------------------------------

        def _track(self, worker: QThread) -> None:
            """Register a worker so closeEvent can wait for it safely."""
            done = [False]

            def _mark():
                done[0] = True

            worker.finished.connect(_mark)
            self._workers.append((worker, done))

        def _on_inputs_changed(self) -> None:
            """Load each chosen map into its ortho viewer, then (re)compute.

            Every map that is currently specified in the selectors is loaded
            into the matching ortho-slice view right away; the True FSC
            computation restarts only when both half-maps are present.
            """
            self._map1 = self._map1_edit.text().strip()
            self._map2 = self._map2_edit.text().strip()
            self._mask = self._mask_edit.text().strip()
            self._seq += 1
            seq = self._seq
            self.text_edit.clear()
            if self._map1 and self._map2:
                self.text_edit.append(
                    f"Loading {Path(self._map1).name} and {Path(self._map2).name}\u2026"
                )
            else:
                self.text_edit.append(
                    "Choose Map 1 and Map 2 to start the True FSC "
                    "computation. A mask is optional."
                )
            loader = _InputLoader(self._map1, self._map2, self._mask, parent=self)
            self._track(loader)
            loader.loaded.connect(lambda vols, s=seq: self._on_inputs_loaded(vols, s))
            loader.failed.connect(lambda msg, s=seq: self._on_input_failed(msg, s))
            loader.start()

        def _on_inputs_loaded(self, volumes, seq) -> None:
            if seq != self._seq:
                return
            (d1, a1) = volumes["map1"]
            (d2, a2) = volumes["map2"]
            if d1 is not None:
                self._map1_viewer.set_volume(d1, a1, reset_position=True)
            else:
                self._map1_viewer.set_volume(
                    np.zeros((1, 1, 1), dtype=np.float32), 1.0, reset_position=True
                )
            if d2 is not None:
                self._map2_viewer.set_volume(d2, a2, reset_position=True)
            else:
                self._map2_viewer.set_volume(
                    np.zeros((1, 1, 1), dtype=np.float32), 1.0, reset_position=True
                )
            mask_vol = volumes["mask"]
            if mask_vol is not None:
                self._mask_viewer.set_volume(mask_vol, a1, reset_position=True)
            else:
                self._mask_viewer.set_volume(
                    np.zeros((1, 1, 1), dtype=np.float32), 1.0, reset_position=True
                )
            if self._map1 and self._map2:
                self._start_compute(seq)

        def _on_input_failed(self, message: str, seq) -> None:
            if seq != self._seq:
                return
            self.text_edit.clear()
            self.text_edit.append(f"Error loading maps:\n{message}")

        def _start_compute(self, seq) -> None:
            output_dir = _output_dir_for(self._map1)
            plot_file = output_dir / "trueFSC.pdf"
            worker = _ComputeWorker(
                self._map1, self._map2, self._mask, str(plot_file), parent=self
            )
            self._track(worker)
            worker.line_received.connect(self.append_line)
            worker.finished.connect(lambda r, s=seq: self.set_result(r, s))
            worker.error.connect(lambda m, s=seq: self.set_error(m, s))
            worker.finished.connect(worker.deleteLater)
            worker.start()

        # ------------------------------------------------------------------
        # Info log / results
        # ------------------------------------------------------------------

        def append_line(self, line):
            self.text_edit.append(line)
            scrollbar = self.text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def set_result(self, result, seq):
            if seq != self._seq:
                return
            if not result:
                return

            volumes = result.get("volumes") or {}
            for viewer, key in (
                (self._mask_viewer, "mask1_file"),
                (self._masked1_viewer, "masked_map1_file"),
                (self._masked2_viewer, "masked_map2_file"),
            ):
                path = volumes.get(key)
                if path:
                    data, apix = _load_volume(path)
                    viewer.set_volume(data, apix=apix, reset_position=True)

            res = result.get("resolution")

            self.text_edit.clear()
            self.text_edit.append(f"Map 1: {self._map1}")
            self.text_edit.append(f"Map 2: {self._map2}")
            if self._mask:
                self.text_edit.append(f"Mask (input): {self._mask}")
            self.text_edit.append("")
            for label, key in (
                ("Mask", "mask1_file"),
                ("Masked map 1", "masked_map1_file"),
                ("Masked map 2", "masked_map2_file"),
            ):
                path = volumes.get(key)
                if path:
                    self.text_edit.append(f"{label}: {path}")
            plot_file = result.get("plot_file")
            if plot_file:
                self.text_edit.append(f"FSC plot: {plot_file}")
            self.text_edit.append("")
            if result.get("resolution_unmasked"):
                self.text_edit.append(
                    f"Unmasked resolution (0.143): {result['resolution_unmasked']:.2f} A"
                )
            if result.get("resolution_masked"):
                self.text_edit.append(
                    f"Masked resolution (0.143): {result['resolution_masked']:.2f} A"
                )
            if res:
                self.text_edit.append(f"True FSC resolution (0.143): {res:.2f} A")
            else:
                self.text_edit.append("True FSC completed")

            if plot_file:
                viewer = _napari.active()
                if viewer is None:
                    viewer = _create_napari_viewer()
                _open_file(viewer, str(plot_file), mode="slice")
                if Path(plot_file).parent != Path(self._map1).parent:
                    self.text_edit.append(
                        f"\nResults saved to: {Path(plot_file).parent}"
                    )

        def set_error(self, error_msg, seq):
            if seq != self._seq:
                return
            self.text_edit.append(f"\nError: {error_msg}")

        def closeEvent(self, event) -> None:
            # Wait for any still-running background worker so a QThread is
            # never destroyed while executing (crashes otherwise). Only touch
            # workers that are still alive; finished ones are scheduled for
            # deletion.
            for worker, done in list(self._workers):
                if not done[0] and worker.isRunning():
                    worker.quit()
                    worker.wait()
            event.accept()

    dialog = ProgressDialog(parent)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.setWindowModality(Qt.WindowModality.NonModal)
    # Offset the panel so it sits beside the file browser instead of covering it.
    if parent:
        parent_geo = parent.geometry()
        offset_x = max(dialog.width(), parent_geo.width()) // 2
        dialog.move(parent_geo.x() + offset_x, parent_geo.y() + 40)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

    if dialog._map1 and dialog._map2:
        dialog._on_inputs_changed()

    return dialog


class _WindowActivationFilter(QObject):
    """Forward panel focus changes to a display-window tracker.

    Mirrors how the gallery/text/FSC windows report activation so that
    ``tracker.active()`` always points at the most recently focused panel
    (reuse target when the ``New`` checkbox is off). Used by the
    images2star and proc3d tools panels.
    """

    def __init__(self, window, tracker):
        super().__init__(window)
        self._window = window
        self._tracker = tracker
        window.installEventFilter(self)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Type.ActivationChange and obj.isActiveWindow():
            self._tracker.on_activate(self._window)
        return False


def _open_images2star_tools(
    path=None, parent=None, reuse_window=None, tracker=None
) -> None:
    """Open the Images2Star tools panel (preview + save) for a dataset.

    Follows the gallery/text/FSC lifecycle: with ``reuse_window`` the panel
    reloads the new file in place; otherwise a fresh non-modal panel is shown
    (the file browser stays usable) and registered with ``tracker`` so later
    clicks reuse it unless the ``New`` checkbox is checked.

    ``path`` may be ``None`` (or empty) to open the panel with its in-panel
    file selector empty so the user chooses a dataset inside the dialog; pass
    a concrete path (as the button action does) to pre-fill and load it.
    """
    from pathlib import Path

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox

    from helicon.lib.gui.images2star_widget import Images2StarDialog

    try:
        resolved = str(Path(path).resolve()) if path else ""
        if (
            reuse_window is not None
            and _is_alive_widget(reuse_window)
            and isinstance(reuse_window, Images2StarDialog)
        ):
            if resolved:
                reuse_window.load_path(resolved)
            reuse_window.show()
            reuse_window.raise_()
            reuse_window.activateWindow()
            return

        dialog = Images2StarDialog(resolved or None, parent=parent)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setModal(False)
        if tracker is not None:
            tracker.register(dialog)
            dialog.destroyed.connect(lambda *_: tracker.on_close(dialog))
            _WindowActivationFilter(dialog, tracker)
        # Offset the panel at least half of its own width from the parent
        # so the two windows sit side by side instead of fully overlapping
        # the file browser (the dialog already resized itself in __init__).
        if parent:
            parent_geo = parent.geometry()
            offset_x = max(dialog.width(), parent_geo.width()) // 2
            dialog.move(parent_geo.x() + offset_x, parent_geo.y() + 40)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
    except Exception as exc:
        QMessageBox.critical(
            None,
            "Images2Star Error",
            f"Failed to open Images2Star tools:\n{exc}",
        )


def _open_proc3d_tools(path=None, parent=None, reuse_window=None, tracker=None) -> None:
    """Open the Proc3D tools panel (ortho previews + save) for a 3D map.

    Follows the images2star panel lifecycle: with ``reuse_window`` the panel
    reloads the new file in place; otherwise a fresh non-modal panel is shown
    (the file browser stays usable) and registered with ``tracker`` so later
    clicks reuse it unless the ``New`` checkbox is checked.

    ``path`` may be ``None`` (or empty) to open the panel with its in-panel
    file selector empty so the user chooses a 3D map inside the dialog; pass a
    concrete path (as the button action does) to pre-fill and load that map.
    """
    from pathlib import Path

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox

    from helicon.lib.gui.proc3d_widget import Proc3dDialog

    try:
        resolved = str(Path(path).resolve()) if path else ""
        if (
            reuse_window is not None
            and _is_alive_widget(reuse_window)
            and isinstance(reuse_window, Proc3dDialog)
        ):
            if resolved:
                reuse_window.load_path(resolved)
            reuse_window.show()
            reuse_window.raise_()
            reuse_window.activateWindow()
            return

        dialog = Proc3dDialog(resolved or None, parent=parent)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setModal(False)
        if tracker is not None:
            tracker.register(dialog)
            dialog.destroyed.connect(lambda *_: tracker.on_close(dialog))
            _WindowActivationFilter(dialog, tracker)
        # Offset the panel at least half of its own width from the parent
        # so the two windows sit side by side instead of fully overlapping
        # the file browser (the dialog already resized itself in __init__).
        if parent:
            parent_geo = parent.geometry()
            offset_x = max(dialog.width(), parent_geo.width()) // 2
            dialog.move(parent_geo.x() + offset_x, parent_geo.y() + 40)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
    except Exception as exc:
        QMessageBox.critical(
            None,
            "Proc3D Error",
            f"Failed to open Proc3D tools:\n{exc}",
        )


def _open_helical_angle_stats_plot(
    result: dict,
) -> None:
    """Display a generated helical-angle variance plot in napari."""
    plot_file = result.get("plot_file")
    if not plot_file:
        raise ValueError("helical-angle variance calculation returned no plot")

    viewer = _napari.active()
    if viewer is None:
        viewer = _create_napari_viewer()
    _open_file(viewer, str(plot_file), mode="slice")


def _helical_angle_stats_paths(path: str) -> tuple[Path, Path, Path]:
    """Return the output directory, derived STAR path, and plot path."""
    import tempfile

    input_path = Path(path)
    input_dir = input_path.parent
    if os.access(input_dir, os.W_OK):
        output_dir = input_dir
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="helicon_angle_stats_"))
    output_star = output_dir / f"{input_path.stem}.helical_angle_variance.star"
    plot_file = output_dir / f"{input_path.stem}.helical_angle_variance.pdf"
    return output_dir, output_star, plot_file


def _launch_helical_angle_stats(
    path: str,
    parent=None,
) -> None:
    """Compute and display Class3D/Refine3D helical-angle variance statistics."""
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QTextEdit, QVBoxLayout

    input_path = Path(path)
    input_dir = input_path.parent
    output_dir, output_star, plot_file = _helical_angle_stats_paths(path)

    from helicon.plugins.images2star.estimatehelicalanglevariance import (
        estimate_helical_angle_variance_from_star,
    )

    class Worker(QThread):
        line_received = Signal(str)
        finished = Signal(object)
        error = Signal(str)

        def run(self):
            try:
                self.line_received.emit(f"Input: {input_path}")
                self.line_received.emit(f"Output STAR: {output_star}")
                self.line_received.emit(f"Plot: {plot_file}")
                self.line_received.emit("")
                result = estimate_helical_angle_variance_from_star(
                    str(input_path),
                    str(output_star),
                    str(plot_file),
                )
                self.finished.emit(result)
            except Exception as exc:
                self.error.emit(str(exc))

    class ProgressDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setStyleSheet(_display_theme_stylesheet())
            self.setWindowTitle("Helical-angle statistics")
            self.setMinimumSize(550, 300)
            layout = QVBoxLayout(self)

            self.label = QLabel("Calculating helical-angle statistics...")
            layout.addWidget(self.label)

            self.text_edit = QTextEdit()
            self.text_edit.setReadOnly(True)
            layout.addWidget(self.text_edit)

            self.close_btn = QPushButton("Close")
            self.close_btn.setEnabled(False)
            self.close_btn.clicked.connect(self.accept)
            layout.addWidget(self.close_btn)

        def append_line(self, line):
            self.text_edit.append(line)
            scrollbar = self.text_edit.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        def set_result(self, result):
            try:
                _open_helical_angle_stats_plot(result)
            except Exception as exc:
                self.set_error(str(exc))
                return
            self.label.setText("Helical-angle statistics completed")
            if output_dir != input_dir:
                self.text_edit.append(f"\nResults saved to: {output_dir}")
            self.close_btn.setEnabled(True)

        def set_error(self, error_msg):
            self.label.setText("Helical-angle statistics failed")
            self.text_edit.append(f"\nError: {error_msg}")
            self.close_btn.setEnabled(True)

    dialog = ProgressDialog(parent)
    worker = Worker()
    worker.line_received.connect(dialog.append_line)
    worker.finished.connect(dialog.set_result)
    worker.error.connect(dialog.set_error)
    worker.start()
    dialog.exec()


def _wrap_gallery_with_panel(gallery: "ImageGalleryWidget") -> "QWidget":
    """Wrap an ImageGalleryWidget with a left-side _ControlPanel sibling.

    The panel is prepended to the left.  Toggling it grows the parent
    window leftward by ``_ControlPanel.PANEL_WIDTH`` so the gallery
    widget keeps both its width and its screen position unchanged.
    """
    from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

    from helicon.lib.gui.gallery_widget import _ControlPanel

    wc = _ControlPanel.PANEL_WIDTH

    panel = _ControlPanel()
    panel.hide()
    panel.setFixedWidth(wc)
    panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(panel)
    layout.addWidget(gallery, 1)

    def _on_toggle():
        win = container.window()
        visible = not panel.isVisible()
        # Operate on the underlying QWindow's geometry, which is the FRAME
        # rect (title bar + borders included) and is not subject to the
        # client-vs-frame conversion that QWidget.setGeometry does.  This
        # keeps the frame's top-left y pinned across toggles on macOS,
        # Linux, and Windows/WSL -- anchoring the client rect would let the
        # title-bar height accumulate into y on every toggle.
        handle = win.windowHandle()
        if handle is None:
            # Fall back to QWidget geometry (no platform window yet).
            if visible:
                panel.show()
                win.setGeometry(win.x() - wc, win.y(), win.width() + wc, win.height())
            else:
                panel.hide()
                win.setGeometry(
                    win.x() + wc, win.y(), max(win.width() - wc, 1), win.height()
                )
            return
        fg = handle.geometry()
        if visible:
            panel.show()
            handle.setGeometry(fg.x() - wc, fg.y(), fg.width() + wc, fg.height())
        else:
            panel.hide()
            handle.setGeometry(
                fg.x() + wc, fg.y(), max(fg.width() - wc, 1), fg.height()
            )

    gallery.panel_toggle_requested.connect(_on_toggle)

    def _on_brightness(val):
        gallery.set_brightness(val / 100.0)
        panel._brightness_val.setText(f"{gallery._brightness:.2f}")
        _refresh_histogram()

    def _on_contrast(val):
        gallery.set_contrast(val / 100.0)
        panel._contrast_val.setText(f"{gallery._contrast:.2f}")
        _refresh_histogram()

    def _on_gamma(val):
        gallery.set_gamma(val / 100.0)
        panel._gamma_val.setText(f"{gallery._gamma:.2f}")
        _refresh_histogram()

    def _on_log_transform(checked):
        gallery.set_log_transform(checked)
        _refresh_histogram()

    def _refresh_histogram():
        if gallery.has_data() and panel._histogram_chk.isChecked():
            panel._histogram_widget.update_histogram(
                gallery._read_fn,
                gallery._n,
                gallery._brightness,
                gallery._contrast,
                gallery._gamma,
                gallery._log_transform,
            )

    def _on_histogram_toggled(checked):
        panel._histogram_widget.setVisible(checked)
        _refresh_histogram()

    def _on_autocontrast():
        panel._brightness_slider.setValue(0)
        panel._contrast_slider.setValue(100)
        panel._gamma_slider.setValue(100)

    def _on_scope_selected(checked):
        if checked:
            gallery.set_adjust_scope("selected")

    def _on_scope_all(checked):
        if checked:
            gallery.set_adjust_scope("all")

    panel._brightness_slider.valueChanged.connect(_on_brightness)
    panel._contrast_slider.valueChanged.connect(_on_contrast)
    panel._gamma_slider.valueChanged.connect(_on_gamma)
    panel._auto_btn.clicked.connect(_on_autocontrast)
    panel._radio_selected.toggled.connect(_on_scope_selected)
    panel._radio_all.toggled.connect(_on_scope_all)
    panel.log_changed.connect(_on_log_transform)
    panel.histogram_changed.connect(_on_histogram_toggled)

    def _on_show_labels(checked):
        gallery.set_show_labels(checked)

    panel.show_labels_changed.connect(_on_show_labels)

    if hasattr(gallery, "view_changed"):
        gallery.view_changed.connect(_refresh_histogram)

    _refresh_histogram()

    return container


def _open_gallery(
    read_fn, n, img_w, img_h, apix, name, reuse_window=None, tracker=None
) -> None:
    """Show the lazy thumbnail grid for a stack in a standalone window.

    Parameters
    ----------
    read_fn : callable
        ``read_fn(i) -> numpy.ndarray`` returning frame ``i`` lazily.
    n : int
        Number of frames in the stack.
    img_w : int
        Native frame width.
    img_h : int
        Native frame height.
    apix : float
        Pixel size (A/px) for the slice view scale.
    name : str
        Display name (usually the file name).
    tracker : _DisplayTracker, optional
        Tracker to register new windows with for lifecycle management.
    """
    from helicon.lib.gui.gallery_backends import StackGallery

    gallery = StackGallery(
        star_path=name,
        read_fn=read_fn,
        n=n,
        img_w=img_w,
        img_h=img_h,
        apix=apix,
    )
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _open_xyz_slice_gallery(
    star_path: str, reuse_window=None, tracker=None
) -> "QMainWindow | None":
    """Display center slices (Z, Y, X) of MRC files referenced by a star file.

    Delegates to ``Class3dGallery`` (with abundance labels) or
    ``Refine3dGallery`` (without) based on the path.
    """
    from helicon.lib.gui.gallery_backends import Class3dGallery, Refine3dGallery

    name = Path(star_path).name
    is_refine = any(p.startswith("Refine3D") for p in Path(star_path).parts)
    gallery = Refine3dGallery(star_path) if is_refine else Class3dGallery(star_path)
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _open_orthogonal_viewer(
    mrc_path: str, reuse_window=None, tracker=None
) -> "QMainWindow | None":
    """Open an interactive orthogonal slice viewer for a 3D MRC/MAP file."""
    from helicon.lib.gui.gallery_backends import OrthogonalGallery

    gallery = OrthogonalGallery(mrc_path)
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _open_2d_classes_gallery(
    star_path: str, reuse_window=None, tracker=None
) -> "QMainWindow | None":
    """Display 2D class averages from a Class2D model.star.

    Shows one MRC per class (``_rlnReferenceImage``) with abundance labels
    (``_rlnClassDistribution``). Sort-by-abundance and reverse-sort controls
    are provided in the control panel.
    """
    from helicon.lib.gui.gallery_backends import Class2dGallery

    gallery = Class2dGallery(star_path)
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _open_frame_in_slice_view(viewer, read_fn, idx, img_w, img_h, apix, name) -> None:
    """Open a single gallery frame in the normal 2D slice view.

    Reuses the same ``add_image`` call used by the slice branch so behaviour
    is identical: the clicked frame is shown and the camera fits it. Any
    prior gallery/slice layer is removed first to avoid accumulation.

    Parameters
    ----------
    viewer : napari.Viewer
        Target viewer.
    read_fn : callable
        ``read_fn(i) -> numpy.ndarray`` returning frame ``i``.
    idx : int
        Global frame index to display.
    img_w : int
        Native frame width.
    img_h : int
        Native frame height.
    apix : float
        Pixel size (A/px).
    name : str
        Display name.
    """
    try:
        for layer in list(viewer.layers):
            viewer.layers.remove(layer)
    except (AttributeError, RuntimeError):
        pass

    frame = read_fn(idx)
    contrast = _auto_contrast(frame)
    layer = viewer.add_image(
        frame,
        name=name,
        scale=(apix, apix),
        contrast_limits=contrast,
        interpolation2d="linear",
        interpolation3d="linear",
    )
    _enable_continuous_auto_contrast(layer, viewer)
    _SliceDirectionWidget.set_stack_mode(True)
    viewer.dims.ndisplay = 2
    _reset_view(viewer)
    # Reveal the window so a freshly-created (hidden) viewer actually shows
    # the clicked frame; a no-op when the window is already visible.
    try:
        viewer.window._qt_window.show()
        viewer.window._qt_window.raise_()
    except Exception:
        pass


def _open_file(viewer, path: str, mode: str | None = None, reuse_gallery=None) -> None:
    from pathlib import Path

    if viewer is not None:
        # Reset the viewer to the mode expected for THIS file before any
        # layer is added. Otherwise a stale ndisplay (e.g. a 3D volume
        # view left over from the previous file) persists and produces a
        # wrong/empty view for the incoming file. Drop the previous
        # layers so reset_view() below fits only the new file.
        try:
            old_layers = list(viewer.layers)
        except (TypeError, AttributeError):
            old_layers = []
        for layer in old_layers:
            try:
                viewer.layers.remove(layer)
            except Exception:
                pass
        # Reset to the volume default; specific branches below set True when the
        # incoming layer is a true image stack (axis 0 is a frame index, not a
        # spatial axis) so the Z/Y/X axis selector is hidden for those files.
        _SliceDirectionWidget.set_stack_mode(False)
        if mode == "volume":
            _set_ndisplay(viewer, 3)
        else:
            # 2D slice / image stack / general / text / pdf / eps -> 2D.
            _set_ndisplay(viewer, 2)
    else:
        _SliceDirectionWidget.set_stack_mode(False)

    if viewer is not None:
        viewer.title = f"Helicon - {Path(path).name}"
        # Remember the opened file name so the Save-As dialog can pre-fill it
        # (mirrors ImageGalleryWidget._source_name).  napari only populates
        # layer.source.path when a file is loaded via viewer.open(); most
        # helicon openers add layers directly, so we record it here instead.
        viewer._source_name = Path(path).name

    # When a display mode is forced (button bar), the *.mrc / *.map volume
    # branch must show as a 3D volume (ndisplay=3) or a 2D slice stack
    # (ndisplay=2) instead of the auto-chosen default.
    force_ndisplay = None
    if mode == "volume":
        force_ndisplay = 3
    elif mode == "slice":
        force_ndisplay = 2

    ext = Path(path).suffix.lower()

    if ext == ".star" and mode == "optimiser":
        _open_xyz_slice_gallery(path, reuse_window=reuse_gallery, tracker=_gallery)
        return

    if ext == ".star" and mode == "2dclasses":
        _open_2d_classes_gallery(path, reuse_window=reuse_gallery, tracker=_gallery)
        return

    # "text" mode: always open any star file as text, regardless of type.
    if ext == ".star" and mode == "text":
        _open_text_window(path, reuse_window=reuse_gallery)
        return

    if ext == ".star" and _is_metadata_star(path):
        _open_text_window(path, reuse_window=reuse_gallery)
        return

    if ext == ".star":
        import mrcfile

        result = _parse_star_image_refs(path)
        if result is None:
            return
        entries, first_shape, first_apix, n_skipped = result

        if n_skipped:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                None,
                "Missing images",
                f"{n_skipped} image(s) referenced in {Path(path).name} "
                "could not be found on disk and were skipped.",
            )
        if not entries:
            return

        n = len(entries)
        with mrcfile.open(entries[0][1], permissive=True) as mrc:
            frame = mrc.data[entries[0][0]] if mrc.data.ndim >= 3 else mrc.data
            dtype = frame.dtype
            # Contrast limits from the first frame only. Passing explicit
            # contrast_limits stops napari from scanning the entire stack to
            # compute a global range (which would force every referenced
            # image to load). Continuous auto-contrast below keeps each
            # navigated slice contrasted independently, so only the visible
            # frame is read -- the lazy behaviour we want.
            contrast = _auto_contrast(frame)

        stack_shape = (n,) + first_shape
        lazy = _LazyStarStack(entries, stack_shape, dtype)

        name = Path(path).name
        if mode == "gallery":
            _open_gallery(
                read_fn=lazy.__getitem__,
                n=n,
                img_w=first_shape[0],
                img_h=first_shape[1],
                apix=first_apix,
                name=name,
                reuse_window=reuse_gallery,
                tracker=_gallery,
            )
            return

        # This is a true image stack (frame index axis 0), not a 3D volume:
        # hide the Z/Y/X axis selector.
        _SliceDirectionWidget.set_stack_mode(True)
        layer = viewer.add_image(
            lazy,
            name=name,
            scale=(1.0,) + (first_apix,) * len(first_shape),
            contrast_limits=contrast,
            interpolation2d="linear",
            interpolation3d="linear",
        )
        _enable_continuous_auto_contrast(layer, viewer)
        viewer.dims.ndisplay = 2
        # napari defaults the shown slice to the middle frame; start at frame 0.
        step = list(viewer.dims.current_step)
        step[0] = 0
        viewer.dims.current_step = step
        _reset_view(viewer)
        return

    if ext == ".bild":
        _open_bild(viewer, path)
        return

    if ext == ".pdf":
        _open_pdf(viewer, path)
        return

    if ext == ".eps":
        _open_eps(viewer, path)
        return

    if ext in (".html", ".htm"):
        _open_html(viewer, path)
        return

    if _is_text_file(path):
        _open_text_window(path, reuse_window=reuse_gallery)
        return

    if ext in _MRC_EXTENSIONS:
        import struct
        import numpy as np
        import dask.array as da
        from dask import delayed

        # Parse the MRC header directly from raw bytes so we never open
        # the file through mrcfile (which mmap's the *data* section and
        # causes the OS to page the entire file into RSS).
        _MODE_DTYPE = {
            0: np.int8,
            1: np.int16,
            2: np.float32,
            3: np.complex64,
            4: np.int16,
            6: np.uint16,
            12: np.float16,
            13: np.uint8,
            14: np.int32,
            15: np.float64,
        }
        try:
            with open(path, "rb") as _fh:
                _raw_hdr = _fh.read(1024)
            _nx, _ny, _nz = struct.unpack_from("<3i", _raw_hdr, 0)
            _mode = struct.unpack_from("<i", _raw_hdr, 12)[0]
            _nsymbt = struct.unpack_from("<i", _raw_hdr, 92)[0]
            _header_offset = 1024 + _nsymbt
            # cella (floats at offsets 40, 44, 48) / dimensions → pixel size
            _cella_x = struct.unpack_from("<f", _raw_hdr, 40)[0]
            _apix = float(_cella_x) / _nx if _nx else 1.0
        except Exception:
            _nx = _ny = _nz = 0
            _mode = 2
            _apix = 1.0
            _header_offset = 1024
        if _apix <= 0:
            try:
                import mrcfile as _mrcfile_fallback

                with _mrcfile_fallback.open(path, permissive=True) as _m:
                    _apix = float(_m.voxel_size.x)
            except Exception:
                _apix = 1.0
        if _apix <= 0:
            _apix = 1.0

        _dtype = _MODE_DTYPE.get(_mode, np.float32)

        _data_mmap = np.memmap(
            path, dtype=_dtype, mode="r", offset=_header_offset, shape=(_nz, _ny, _nx)
        )

        def _read_raw(i):
            return np.array(_data_mmap[i])

        name = Path(path).name

        # Build a lazily-read 3D array (axis 0 = Z / frame). Planes are read
        # from disk individually, so a 2D view never loads the whole file.
        if _nz >= 1 and _nx > 0 and _ny > 0:
            _vol = da.stack(
                [
                    da.from_delayed(
                        delayed(_read_raw)(i), shape=(_ny, _nx), dtype=_dtype
                    )
                    for i in range(_nz)
                ]
            )
        else:
            _vol = None

        # Gallery: read one plane at a time (lazy) — no full-volume load.
        # Must be checked before the single-plane branch so that a 1-image
        # .mrcs opened via the gallery button still reaches the gallery.
        if mode == "gallery":
            n = int(_vol.shape[0])
            img_h, img_w = int(_vol.shape[1]), int(_vol.shape[2])

            def _gal_read(i):
                return np.asarray(_vol[i])

            _open_gallery(
                read_fn=_gal_read,
                n=n,
                img_w=img_w,
                img_h=img_h,
                apix=_apix,
                name=name,
                reuse_window=reuse_gallery,
                tracker=_gallery,
            )
            return

        # Single 2D plane: one eager read is fine and keeps the 2D path simple.
        if _vol is None or (_vol.ndim == 3 and _vol.shape[0] == 1):
            if viewer is None:
                return
            _single = _read_raw(0) if _vol is not None else None
            if _single is None:
                with mrcfile.open(path, permissive=True) as m:
                    _single = np.asarray(m.data)
            if _single.ndim >= 3:
                _single = _single[0]
            _contrast = _auto_contrast(_single)
            _SliceDirectionWidget.set_stack_mode(False)
            layer = viewer.add_image(
                _single,
                name=name,
                scale=(_apix, _apix),
                contrast_limits=_contrast,
                interpolation2d="linear",
                interpolation3d="linear",
            )
            _enable_continuous_auto_contrast(layer, viewer)
            _lo, _hi = float(_single.min()), float(_single.max())
            if _lo >= _hi:
                _hi = _lo + 1.0
            layer.contrast_limits_range = (_lo, _hi)
            _reset_view(viewer)
            return

        if mode == "orthogonal" and _nz > 1:
            _open_orthogonal_viewer(path, reuse_window=reuse_gallery, tracker=_gallery)
            return

        # Slice / volume view: show the lazy 3D array; napari loads one plane
        # at a time as the user scrolls.
        _SliceDirectionWidget.set_stack_mode(ext == ".mrcs")
        _mid = int(_vol.shape[0]) // 2
        _sample = np.asarray(_vol[_mid])
        contrast = _auto_contrast(_sample)
        layer = viewer.add_image(
            _vol,
            name=name,
            scale=(_apix, _apix, _apix),
            contrast_limits=contrast,
            interpolation2d="linear",
            interpolation3d="linear",
        )
        _enable_continuous_auto_contrast(layer, viewer)
        _lo, _hi = float(_sample.min()), float(_sample.max())
        if _lo >= _hi:
            _hi = _lo + 1.0
        layer.contrast_limits_range = (_lo, _hi)

        if len(viewer.dims.current_step) > 0:
            step = list(viewer.dims.current_step)
            if ext == ".mrcs":
                step[0] = 0
                viewer.dims.current_step = tuple(step)
                viewer.dims.ndisplay = 2
            else:
                step[0] = _mid
                viewer.dims.current_step = tuple(step)
                if force_ndisplay is not None:
                    viewer.dims.ndisplay = force_ndisplay
                else:
                    dims = _vol.shape
                    viewer.dims.ndisplay = (
                        3 if dims[0] * dims[1] * dims[2] < 512**3 else 2
                    )
        _reset_view(viewer)
        return
    else:
        n_before = len(viewer.layers)
        viewer.open(path)
        for layer in viewer.layers[n_before:]:
            _enable_continuous_auto_contrast(layer, viewer)
        _reset_view(viewer)


def _run_standalone() -> None:
    """Open a single file in a fresh napari viewer (separate process).

    Used by the "New display window" feature. Running the second viewer in
    its own process gives it a clean OpenGL context and avoids the VisPy/
    Qt-Wayland segfault that occurs when a second napari canvas is created
    inside an already-running viewer's event loop (fine on macOS, fatal
    under WSLg Wayland).

    Expects ``sys.argv[1]`` = file path and optional ``sys.argv[2]`` = mode.
    """
    _prepare_process_identity()
    _force_x11_platform_under_wslg()
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(_qt_argv())
    _set_application_identity(app)
    napari = _load_napari()

    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

    viewer = _create_napari_viewer(title=f"Helicon - {Path(path).name}")
    _install_panel_toggle(viewer)
    try:
        _open_file(viewer, path, mode=mode)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[helicon] failed to display {path}: {exc}")
    napari.run()


def main(args: argparse.Namespace) -> None:
    """Launch napari with a floating folder browser.

    Opens a napari viewer window alongside a floating folder browser
    window. Double-click any image or volume file in the browser to
    display it in napari. Window positions and sizes are remembered
    between launches.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    if FolderBrowserWidget is None:
        raise HeliconDependencyError(
            "Qt widgets are required for the display command. "
            "Install PySide6: pip install PySide6"
        )

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QWidget, QApplication
    from PySide6.QtGui import QShortcut, QKeySequence

    _prepare_process_identity()
    _force_x11_platform_under_wslg()
    app = QApplication.instance() or QApplication(_qt_argv())
    _set_application_identity(app)

    start_dir = args.folder if args.folder else os.getcwd()

    if app is not None:
        _icon_path = Path(__file__).parent.parent / "resources" / "icon.png"
        if _icon_path.is_file():
            from PySide6.QtGui import QIcon

            app.setWindowIcon(QIcon(str(_icon_path)))

    def _track_viewer(v):
        from PySide6.QtCore import QEvent, QObject

        class _CloseFilter(QObject):
            def __init__(self, viewer, parent=None):
                super().__init__(parent)
                self._viewer = viewer

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Close:
                    _napari.on_close(self._viewer)
                elif event.type() == QEvent.ActivationChange and obj.isActiveWindow():
                    _napari.on_activate(self._viewer)
                return False

        try:
            qt_window = v.window._qt_window
            if qt_window is not None and isinstance(qt_window, QObject):
                qt_window.destroyed.connect(lambda *_: _napari.on_close(v))
                flt = _CloseFilter(v, parent=qt_window)
                qt_window.installEventFilter(flt)
        except Exception:
            pass

    def _ensure_napari_viewer():
        """Return the active napari viewer, creating one on first use."""
        from unittest.mock import MagicMock

        v = _napari.active()
        if v is not None:
            return v
        new_viewer = _create_napari_viewer()
        _track_viewer(new_viewer)
        _add_welcome_shortcut(new_viewer)
        _install_panel_toggle(new_viewer)
        try:
            if not isinstance(new_viewer.window, MagicMock):
                _install_viewer_save_menu(new_viewer)
        except Exception:
            pass
        _install_viewer_save_hook(widget, new_viewer)
        return new_viewer

    def _show_napari_viewer():
        """Ensure a napari viewer exists and reveal it."""
        v = _ensure_napari_viewer()
        try:
            v.window._qt_window.show()
            v.window._qt_window.raise_()
        except Exception:
            pass

    def _spawn_viewer_and_open(path, mode=None):
        """Open ``path`` in a new napari viewer.

        On macOS multiple in-process napari viewers coexist safely, so the
        new window is opened in-process and tracked by the ``_napari``
        tracker.  On every other platform (notably Linux/Wayland), creating
        a second in-process napari viewer segfaults, so the new window is
        spawned in a separate process.
        """
        if sys.platform == "darwin":
            try:
                new_viewer = _create_napari_viewer(title=f"Helicon - {Path(path).name}")
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[helicon] failed to open new display window: {exc}")
                return
            _track_viewer(new_viewer)
            try:
                if not isinstance(new_viewer.window, MagicMock):
                    _install_viewer_save_menu(new_viewer)
            except Exception:
                pass
            try:
                _open_file(new_viewer, path, mode=mode)
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[helicon] failed to display {path}: {exc}")
            return

        import subprocess

        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from helicon.commands.display import _run_standalone; "
                    "_run_standalone()",
                    path,
                    mode or "",
                ],
                start_new_session=True,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[helicon] failed to open new display window: {exc}")

    def _categorize_file(path):
        """Return (tracker, primary_mode) for a file based on its type."""
        modes = widget._display_modes_for(path)
        if not modes:
            return _napari, "slice"
        # For optimiser/model.star files, prefer gallery over text
        if modes[0] == "text":
            for m in modes:
                if m in _GALLERY_MODES:
                    return _TRACKER_FOR[m], m
        return _TRACKER_FOR.get(modes[0], _napari), modes[0]

    def _on_file_selected(path):
        tracker, mode = _categorize_file(path)
        if tracker is _napari:
            _show_napari_viewer()
            _open_file(_napari.active(), path)
        elif tracker is _gallery:
            reuse = tracker.active()
            _open_file(None, path, mode=mode, reuse_gallery=reuse)
        elif tracker is _text:
            _open_text_window(path, reuse_window=tracker.active())
        elif tracker is _plot:
            _open_fsc_plot(path, reuse_window=tracker.active())
        elif tracker is _images2star:
            _open_images2star_tools(
                path,
                parent=widget,
                reuse_window=tracker.active(),
                tracker=tracker,
            )
        elif tracker is _proc3d:
            _open_proc3d_tools(
                path,
                parent=widget,
                reuse_window=tracker.active(),
                tracker=tracker,
            )

    def _on_file_selected_new_window(path):
        tracker, mode = _categorize_file(path)
        if tracker is _napari:
            _spawn_viewer_and_open(path)
        else:
            _on_display_requested(path, mode, new_window=True)

    def _on_display_requested(path, mode, new_window):
        if mode == "chimerax":
            _launch_chimerax(path)
            return
        if mode == "denovo3D":
            _launch_denovo3d(path, new_window=new_window)
            return
        if mode == "whereIsMyClass":
            _launch_whereismyclass(path, new_window=new_window)
            return
        if mode == "helicalProjection":
            _launch_helicalprojection(path, new_window=new_window)
            return
        if mode == "helicalPitch":
            _launch_helicalpitch(path, new_window=new_window)
            return
        if mode == "hill":
            _launch_hill(path, new_window=new_window)
            return
        if mode == "hi3d":
            _launch_hi3d(path, new_window=new_window)
            return
        if mode == "trueFSC":
            _launch_truefsc(path, parent=widget)
            return
        if mode == "stats":
            _launch_helical_angle_stats(
                path,
                parent=widget,
            )
            return
        tracker = _TRACKER_FOR.get(mode)
        if tracker is None:
            return
        if tracker is _napari:
            if (
                new_window
                and _napari.active() is not None
                and _is_alive_viewer(_napari.active())
            ):
                _spawn_viewer_and_open(path, mode=mode)
            else:
                _show_napari_viewer()
                try:
                    _open_file(_napari.active(), path, mode=mode)
                except Exception as exc:
                    import logging

                    logging.getLogger("helicon").warning(
                        "failed to display %s in napari: %s", path, exc
                    )
        else:
            reuse = None if new_window else tracker.active()
            if tracker is _gallery:
                _open_file(None, path, mode=mode, reuse_gallery=reuse)
            elif tracker is _text:
                _open_text_window(path, reuse_window=reuse)
            elif tracker is _plot:
                _open_fsc_plot(path, reuse_window=reuse)
            elif tracker is _images2star:
                _open_images2star_tools(
                    path,
                    parent=widget,
                    reuse_window=reuse,
                    tracker=tracker,
                )
            elif tracker is _proc3d:
                _open_proc3d_tools(
                    path,
                    parent=widget,
                    reuse_window=reuse,
                    tracker=tracker,
                )

    widget = FolderBrowserWidget(start_dir=start_dir)
    widget.file_selected.connect(_on_file_selected)
    widget.file_selected_new_window.connect(_on_file_selected_new_window)
    widget.display_requested.connect(_on_display_requested)
    widget.setWindowFlags(Qt.WindowType.Window)
    widget.setWindowTitle("Helicon - Files")
    widget.show()

    # On macOS the application-name menu item is only realized once the
    # QMainWindow's native menu bar is installed by the event loop. Re-apply
    # the identity after the window is shown so the top-left menu reads
    # "Helicon" instead of the inherited "python3.14". Also nudge the app to
    # become active so macOS renders the (initially blank) native menu bar.
    if sys.platform == "darwin":
        QTimer.singleShot(0, lambda: _set_macos_app_identity("Helicon"))

        menu_state = {
            # App that was frontmost at launch (typically the terminal) —
            # used to reproduce the "click another app" half of the cycle.
            "alternate_pid": _macos_frontmost_pid(),
            "refreshed": False,
        }

        def _realize_macos_menu(attempt: int, full_cycle: bool) -> None:
            """Reproduce the click-away-and-back cycle until the menu exists.

            Launched from a terminal the app is often never granted a real
            activation cycle, so the native File/View menu bar stays blank
            until the user clicks another app and back. Repeat the resign/
            re-activate cycle until NSApp's main menu actually contains the
            File/View items and Helicon is the frontmost application (that is
            what makes macOS display its menu bar), or give up after a short
            cap. The first two attempts do the full cycle; later ones only
            re-activate to avoid flicker in case Qt merely needed a late
            window-activation.
            """
            _force_macos_menu_realization(full_cycle=full_cycle)
            # A full cycle resigns now and re-activates 120 ms later, so its
            # result cannot be judged until after that delayed activation.
            delay = 280 if full_cycle else 30
            QTimer.singleShot(delay, lambda: _realize_macos_menu_check(attempt))

        def _realize_macos_menu_check(attempt: int) -> None:
            """Evaluate the last activation attempt; retry if still needed."""
            count = _macos_menu_item_count(_macos_ns_app())
            frontmost = _macos_frontmost_pid()
            if count >= 2 and frontmost == os.getpid():
                if not menu_state["refreshed"]:
                    # The menu is fully installed and we are frontmost, but the
                    # window server may still be showing the menu bar from
                    # before File/View were installed (or the previous app's).
                    # Repeat the user's manual action: resign, briefly bring
                    # the launch-time frontmost app forward, then come back —
                    # now that the menu exists, the re-activation re-reads it.
                    menu_state["refreshed"] = True
                    _macos_menu_refresh(attempt)
                    return
                # The activation cycle rebuilds the native menu bar, which can
                # reset the application-menu title (observed as "Apple", Qt's
                # placeholder). Re-apply the "Helicon" identity now that the
                # menu exists so the top-left menu reads Helicon.
                _set_macos_app_identity("Helicon")
                return
            if attempt >= 12:
                return
            QTimer.singleShot(
                250,
                lambda: _realize_macos_menu(attempt + 1, attempt < 3),
            )

        def _macos_menu_refresh(attempt: int) -> None:
            """Resign, activate the launch-time frontmost app, then come back."""
            _macos_resign_active()
            alternate = menu_state.get("alternate_pid") or 0
            if alternate and alternate != os.getpid():
                QTimer.singleShot(90, lambda: _macos_activate_pid(alternate))
            QTimer.singleShot(240, _macos_activate_and_front)
            QTimer.singleShot(420, lambda: _realize_macos_menu_check(attempt))

        QTimer.singleShot(150, lambda: _realize_macos_menu(1, True))

    try:
        from unittest.mock import MagicMock

        if not isinstance(widget, MagicMock):
            _install_window_shortcuts(widget)
    except Exception:
        pass

    _install_dock_save_hook(widget)
    _restore_geometry(widget, None)

    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.aboutToQuit.connect(_terminate_web_apps)

    # Also terminate web app servers if the process dies without a clean
    # Qt shutdown (SIGTERM/SIGINT/SIGHUP, exit() in an except block).
    # Handlers clean up then restore default disposition and re-raise.
    # Linux children also get PR_SET_PDEATHSIG from launch_shiny_app so
    # they die if this process is SIGKILL'd and never runs these handlers.
    import atexit
    import signal

    atexit.register(_terminate_web_apps)
    _exit_signals = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, "SIGHUP"):
        _exit_signals.append(signal.SIGHUP)
    for signum in _exit_signals:

        def _handle_signal(sig=signum):
            _terminate_web_apps()
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)

        try:
            signal.signal(signum, _handle_signal)
        except (ValueError, OSError):
            pass

    app.exec()


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments for the display command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to attach arguments to.

    Returns
    -------
    argparse.ArgumentParser
        The parser with arguments added.
    """
    parser.add_argument(
        "folder",
        nargs="?",
        type=str,
        metavar="<folder>",
        default=None,
        help="folder to browse (default: current directory)",
    )

    return parser
