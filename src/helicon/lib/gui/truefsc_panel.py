"""The trueFSC panel of the file browser: the True FSC of the two half-maps of a refinement."""

from __future__ import annotations

import os
import re
from pathlib import Path

from helicon.lib.gui.fsc import _distinct_curve_colors
from helicon.lib.gui.theme import _display_plot_theme_colors, _display_theme_stylesheet
from helicon.lib.gui.trackers import _napari
from helicon.lib.gui.viewer import _create_napari_viewer


def _launch_truefsc(path: str, parent=None) -> None:
    """Compute True FSC from the two half-maps referenced by a model.star file.

    Opens the trueFSC panel with the Map 1 and Map 2 selectors pre-filled and
    starts loading the maps immediately.
    """
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


def _launch_truefsc_maps(
    map1=None,
    map2=None,
    mask=None,
    parent=None,
    apix=0,
    cutoff_res=0,
    one_mask=1,
    mask_fraction_thresh=-1,
    mask_thresh=None,
    mask_mass=0,
    mask_soft=-1,
    refine_mask=1,
) -> None:
    """Open the trueFSC panel with Map 1 / Map 2 selectors and options.

    The information pane holds three tabs. The first, "Input Maps", provides
    the Map 1 and Map 2 file selectors. As soon as both half-maps are chosen
    the two input ortho viewers load and the True FSC computation runs in the
    background. The second tab, "Options", exposes the remaining trueFSC
    command options: pixel size, cutoff resolution, the optional mask map and
    mask threshold behaviour, and the mask slope width. The third, "FSC
    curves", displays the resulting curves in the app and expands over the
    information panel while selected.

    ``map1``, ``map2``, and ``mask`` may be ``str``, ``pathlib.Path``, or
    ``None``. A mask given in the Options tab is used verbatim, otherwise an
    adaptive mask is generated. The remaining keyword arguments seed the
    matching fields on the Options tab; leaving them at their defaults keeps
    the trueFSC auto behaviour.

    Returns
    -------
    QDialog
        The open, non-modal progress dialog.
    """
    from helicon.commands.display import _open_file, _wrap_gallery_with_panel

    import logging
    import tempfile
    from pathlib import Path

    import numpy as np

    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QTabWidget,
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

        def __init__(
            self,
            map1,
            map2,
            mask,
            plot_file,
            options,
            parent=None,
        ):
            super().__init__(parent)
            self._map1 = map1
            self._map2 = map2
            self._mask = mask
            self._plot_file = plot_file
            self._options = options

        def run(self):
            try:
                self.line_received.emit(f"Map 1: {self._map1}")
                self.line_received.emit(f"Map 2: {self._map2}")
                if self._mask:
                    self.line_received.emit(f"Mask: {self._mask}")
                for key, value in self._options.items():
                    if value is not None and value not in ("", -1, 0):
                        self.line_received.emit(f"{key}: {value}")
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
                        **self._options,
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
            self._seed = {
                "apix": apix,
                "cutoff_res": cutoff_res,
                "one_mask": one_mask,
                "mask_fraction_thresh": mask_fraction_thresh,
                "mask_thresh": mask_thresh,
                "mask_mass": mask_mass,
                "mask_soft": mask_soft,
                "refine_mask": refine_mask,
            }

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

            # Column 2, row 2: an "Input Maps" / "Options" / "FSC curves"
            # tab bar plus the info log. The curve tab expands over the info
            # log so the plot can use this entire grid cell.
            self._tabs = QTabWidget()
            selector_panel = self._build_selector_panel()
            options_panel = self._build_options_panel()
            self._tabs.addTab(selector_panel, "Input Maps")
            self._tabs.addTab(options_panel, "Options")
            self._fsc_tab_index = self._tabs.addTab(
                self._build_fsc_panel(), "FSC curves"
            )
            self._config_tabs_height = (
                self._tabs.tabBar().sizeHint().height()
                + max(
                    selector_panel.sizeHint().height(),
                    options_panel.sizeHint().height(),
                )
                + 12
            )
            info_box = QVBoxLayout()
            info_box.setContentsMargins(0, 0, 0, 0)
            info_box.setSpacing(2)
            info_box.addWidget(_section_label("Information"))
            self.text_edit = QTextEdit()
            self.text_edit.setReadOnly(True)
            info_box.addWidget(self.text_edit, 1)
            self._information_panel = QWidget()
            self._information_panel.setLayout(info_box)
            info_widget = QWidget()
            info_col = QVBoxLayout(info_widget)
            info_col.setContentsMargins(0, 0, 0, 0)
            info_col.setSpacing(2)
            info_col.addWidget(self._tabs)
            info_col.addWidget(self._information_panel, 1)
            grid.addWidget(info_widget, 1, 1)
            self._tabs.currentChanged.connect(self._on_tab_changed)
            self._on_tab_changed(self._tabs.currentIndex())

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
            """Build the Map 1 / Map 2 file selectors."""
            panel = QWidget()
            box = QVBoxLayout(panel)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(2)

            self._map1_edit, row1 = self._selector_row(
                "Map 1", "Choose the half-map 1 3D MRC/MAP map from disk"
            )
            self._map2_edit, row2 = self._selector_row(
                "Map 2", "Choose the half-map 2 3D MRC/MAP map from disk"
            )
            box.addLayout(row1)
            box.addLayout(row2)
            return panel

        def _build_options_panel(self) -> QWidget:
            """Build the Options tab exposing the remaining trueFSC options."""
            panel = QWidget()
            box = QVBoxLayout(panel)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(6)

            # Pixel size + cutoff resolution on a single row.
            self._apix_edit, self._cutoff_edit = self._build_options_pair_row(
                box,
                "Pixel size (A)",
                "Pixel size in Angstrom. Empty reads it from the map header",
                str(self._seed["apix"]) if self._seed["apix"] else "",
                "Cutoff res (A)",
                "Starting resolution for phase randomization. Default: FSC=0.8 "
                "of the unmasked maps",
                str(self._seed["cutoff_res"]) if self._seed["cutoff_res"] else "",
            )

            # Mask threshold mode + value.
            self._mask_mode_combo, mask_value_edit = self._build_mask_mode_row(
                box, self._seed["mask_fraction_thresh"]
            )
            self._mask_value_edit = mask_value_edit
            self._mask_mode_combo.currentIndexChanged.connect(
                self._on_mask_mode_changed
            )

            # Mask slope width + refine-slope + common-mask checkboxes on a
            # single row.
            slope_label = QLabel("Slope (A)")
            slope_label.setToolTip(
                "Mask slope width in Angstrom. Empty uses automatic slope"
            )
            self._mask_soft_edit = QLineEdit(
                str(self._seed["mask_soft"]) if self._seed["mask_soft"] > 0 else "",
                maxLength=8,
            )
            self._mask_soft_edit.setToolTip(
                "Mask slope width in Angstrom. Empty uses automatic slope"
            )
            self._mask_soft_edit.setPlaceholderText("auto")
            self._mask_soft_edit.setClearButtonEnabled(True)
            # The slope only needs a few digits; keep it compact so the
            # two checkboxes have room to render their full labels and
            # absorb the spare width (no clipped trailing letter).
            self._mask_soft_edit.setMaximumWidth(110)
            self._refine_check = QCheckBox("Refine slope")
            self._refine_check.setToolTip(
                "Refine the mask slope at the end of the computation"
            )
            self._refine_check.setChecked(bool(self._seed["refine_mask"]))
            self._one_mask_check = QCheckBox("Common mask")
            self._one_mask_check.setToolTip(
                "Use the same mask for both maps (checked) or separate masks"
            )
            self._one_mask_check.setChecked(bool(self._seed["one_mask"]))
            slope_row = QHBoxLayout()
            slope_row.setSpacing(4)
            slope_row.addWidget(slope_label)
            slope_row.addWidget(self._mask_soft_edit)
            slope_row.addWidget(self._refine_check, 1)
            slope_row.addWidget(self._one_mask_check, 1)
            box.addLayout(slope_row)

            # Optional mask map, grouped with the mask threshold options. When a
            # mask file is provided the mask slope below is grayed out.
            mask_row = QHBoxLayout()
            mask_row.setSpacing(4)
            mask_label = QLabel("Mask map")
            mask_label.setMinimumWidth(48)
            mask_row.addWidget(mask_label)
            self._mask_edit = QLineEdit()
            self._mask_edit.setClearButtonEnabled(True)
            self._mask_edit.setPlaceholderText("Path to a mask MRC/MAP map...")
            self._mask_edit.setToolTip(
                "Optional mask used verbatim; empty generates an adaptive mask"
            )
            mask_row.addWidget(self._mask_edit, 1)
            mask_browse = self._compact_button(
                "Browse...",
                "Optional mask used verbatim; empty generates an adaptive mask",
            )
            mask_browse.clicked.connect(lambda: self._browse_for(self._mask_edit))
            mask_row.addWidget(mask_browse)
            box.addLayout(mask_row)
            self._mask_edit.textChanged.connect(self._on_mask_changed)

            for widget in (
                self._apix_edit,
                self._cutoff_edit,
                self._mask_value_edit,
                self._mask_edit,
                self._mask_soft_edit,
                self._mask_mode_combo,
                self._refine_check,
                self._one_mask_check,
            ):
                signal = getattr(widget, "editingFinished", None)
                if signal is not None:
                    signal.connect(self._on_options_changed)
                if isinstance(widget, QCheckBox):
                    widget.toggled.connect(self._on_options_changed)
                if isinstance(widget, QComboBox):
                    widget.currentIndexChanged.connect(self._on_options_changed)
            return panel

        def _build_fsc_panel(self) -> QWidget:
            """Build the interactive, in-app True FSC curve plot."""
            import pyqtgraph as pg

            panel = QWidget()
            box = QVBoxLayout(panel)
            box.setContentsMargins(0, 0, 0, 0)
            self._fsc_placeholder = QLabel(
                "FSC curves will appear here when the computation finishes."
            )
            self._fsc_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(self._fsc_placeholder, 1)

            pg.setConfigOptions(antialias=True)
            self._fsc_plot_widget = pg.PlotWidget()
            self._fsc_plot_widget.setVisible(False)
            box.addWidget(self._fsc_plot_widget, 1)

            plot = self._fsc_plot_widget.getPlotItem()
            self._fsc_plot = plot
            plot.getAxis("bottom").enableAutoSIPrefix(False)
            plot.setLabel("bottom", "Resolution", units="1/Å")
            plot.setLabel("left", "Fourier Shell Correlation")
            plot.showGrid(x=True, y=True, alpha=0.3)
            self._fsc_legend = plot.addLegend()

            top_axis = plot.getAxis("top")
            top_axis.enableAutoSIPrefix(False)
            top_axis.setLabel("Resolution", units="Å")
            top_axis.tickStrings = lambda values, scale, spacing: [
                f"{1.0 / value:.1f}" if value > 0 else "∞" for value in values
            ]
            top_axis.show()

            colors = _display_plot_theme_colors()
            self._fsc_plot_colors = colors
            self._fsc_plot_widget.setBackground(colors["background"])
            for axis_name in ("left", "bottom", "top", "right"):
                axis = plot.getAxis(axis_name)
                axis.setPen(colors["foreground"])
                axis.setTextPen(colors["foreground"])
                axis.setTickPen(colors["foreground"])
            plot.setLabel(
                "bottom",
                "Resolution",
                units="1/Å",
                color=colors["foreground"],
            )
            plot.setLabel(
                "left", "Fourier Shell Correlation", color=colors["foreground"]
            )
            top_axis.setLabel("Resolution", units="Å", color=colors["foreground"])

            threshold_pen = pg.mkPen(
                color=(220, 50, 50), width=1, style=Qt.PenStyle.DashLine
            )
            self._fsc_threshold_line = pg.InfiniteLine(
                pos=0.143, angle=0, pen=threshold_pen
            )
            plot.addItem(self._fsc_threshold_line)
            self._fsc_curve_items = []
            return panel

        def _on_tab_changed(self, index: int) -> None:
            """Give the curve tab the full tab-and-information area."""
            curves_selected = index == self._fsc_tab_index
            self._information_panel.setHidden(curves_selected)
            self._tabs.setMaximumHeight(
                16777215 if curves_selected else self._config_tabs_height
            )

        def _set_fsc_curves(self, curves) -> None:
            """Replace the in-app plot with the curves from ``compute_truefsc``."""
            import pyqtgraph as pg

            for item in self._fsc_curve_items:
                self._fsc_plot.removeItem(item)
            self._fsc_curve_items.clear()

            if not curves:
                self._fsc_plot_widget.setVisible(False)
                self._fsc_placeholder.setText(
                    "FSC curves will appear here when the computation finishes."
                )
                self._fsc_placeholder.setVisible(True)
                return

            colors = _distinct_curve_colors(len(curves))
            xmin = 0.0
            xmax = 0.0
            ymin = 0.0
            ymax = 1.0
            for color, (x, y, label) in zip(colors, curves):
                x = np.asarray(x)
                y = np.asarray(y)
                if not x.size or not y.size:
                    continue
                finite = np.isfinite(x) & np.isfinite(y)
                if not np.any(finite):
                    continue
                x = x[finite]
                y = y[finite]
                xmin = min(xmin, float(np.min(x)))
                xmax = max(xmax, float(np.max(x)))
                ymin = min(ymin, float(np.min(y)))
                ymax = max(ymax, float(np.max(y)))
                pen = pg.mkPen(color=color, width=2)
                self._fsc_curve_items.append(
                    self._fsc_plot.plot(x, y, pen=pen, name=str(label))
                )

            if not self._fsc_curve_items:
                self._fsc_plot_widget.setVisible(False)
                self._fsc_placeholder.setText(
                    "No plottable FSC curve data was returned."
                )
                self._fsc_placeholder.setVisible(True)
                return
            for _sample, legend_label in self._fsc_legend.items:
                legend_label.setText(
                    legend_label.text, color=self._fsc_plot_colors["foreground"]
                )
            self._fsc_plot.setXRange(xmin, xmax, padding=0.02)
            self._fsc_plot.setYRange(ymin, ymax, padding=0.04)
            self._fsc_placeholder.setVisible(False)
            self._fsc_plot_widget.setVisible(True)

        def _build_options_pair_row(
            self,
            box: QVBoxLayout,
            label1: str,
            tooltip1: str,
            value1: str,
            label2: str,
            tooltip2: str,
            value2: str,
        ) -> tuple[QLineEdit, QLineEdit]:
            """Add two labeled line edits side by side and return them."""
            row = QHBoxLayout()
            row.setSpacing(8)
            for index, (label_text, tooltip, value) in enumerate(
                ((label1, tooltip1, value1), (label2, tooltip2, value2))
            ):
                sub = QHBoxLayout()
                sub.setSpacing(4)
                label = QLabel(label_text)
                label.setMinimumWidth(72)
                sub.addWidget(label)
                edit = QLineEdit(value)
                edit.setToolTip(tooltip)
                edit.setPlaceholderText("auto")
                edit.setClearButtonEnabled(True)
                sub.addWidget(edit, 1)
                row.addLayout(sub, 1)
                if index == 0:
                    edit1 = edit
                else:
                    edit2 = edit
            box.addLayout(row)
            return edit1, edit2

        def _build_options_row(
            self, box: QVBoxLayout, label_text: str, tooltip: str, value: str
        ) -> QLineEdit:
            """Add a labeled line edit row to the options tab and return it."""
            row = QHBoxLayout()
            row.setSpacing(4)
            label = QLabel(label_text)
            label.setMinimumWidth(96)
            row.addWidget(label)
            edit = QLineEdit(value)
            edit.setToolTip(tooltip)
            edit.setPlaceholderText("auto")
            edit.setClearButtonEnabled(True)
            row.addWidget(edit, 1)
            box.addLayout(row)
            return edit

        def _build_mask_mode_row(
            self, box: QVBoxLayout, fraction_thresh: float
        ) -> tuple[QComboBox, QLineEdit]:
            """Add the mask-threshold mode selector and its value field."""
            row = QHBoxLayout()
            row.setSpacing(4)
            label = QLabel("Mask threshold")
            label.setMinimumWidth(96)
            row.addWidget(label)
            self._mask_mode_combo = QComboBox()
            self._mask_mode_combo.addItem("Auto", userData="auto")
            self._mask_mode_combo.addItem("Fraction of max", userData="fraction")
            self._mask_mode_combo.addItem("Pixel value", userData="pixel")
            self._mask_mode_combo.addItem("Mass (kDa)", userData="mass")
            row.addWidget(self._mask_mode_combo, 1)
            value_edit = QLineEdit()
            value_edit.setPlaceholderText(
                "Value (auto mode computes it from the map density)"
            )
            value_edit.setClearButtonEnabled(True)
            row.addWidget(value_edit, 1)
            box.addLayout(row)
            self._mask_value_edit = value_edit
            if fraction_thresh > 0:
                self._mask_mode_combo.setCurrentIndex(1)
                value_edit.setText(str(fraction_thresh))
            elif fraction_thresh == 0:
                self._mask_mode_combo.setCurrentIndex(3)
                value_edit.setText(str(self._seed["mask_mass"]))
            elif self._seed["mask_thresh"]:
                self._mask_mode_combo.setCurrentIndex(2)
                thresh = self._seed["mask_thresh"]
                value_edit.setText(
                    str(thresh[0] if isinstance(thresh, list) else thresh)
                )
            self._on_mask_mode_changed()
            return self._mask_mode_combo, self._mask_value_edit

        def _on_mask_mode_changed(self) -> None:
            """Enable/disable the mask-value field in auto mode."""
            mode = self._mask_mode_combo.currentData()
            enabled = mode != "auto"
            self._mask_value_edit.setEnabled(enabled)
            if not enabled:
                self._mask_value_edit.setPlaceholderText(
                    "Auto (computed from the map density)"
                )
            else:
                self._mask_value_edit.setPlaceholderText("Value")

        def _on_mask_changed(self) -> None:
            """Gray out the mask slope when a mask map is provided."""
            has_mask = bool(self._mask_edit.text().strip())
            self._mask_soft_edit.setEnabled(not has_mask)
            tooltip = (
                "Mask used verbatim; slope options are inactive."
                if has_mask
                else "Mask slope width in Angstrom. Empty uses automatic slope"
            )
            self._mask_soft_edit.setToolTip(tooltip)
            self._refine_check.setEnabled(not has_mask)

        def _get_options(self) -> dict:
            """Return the trueFSC options dict parsed from the Options tab."""

            def _float(text_value, default):
                text_value = text_value.strip()
                if not text_value:
                    return default
                try:
                    return float(text_value)
                except ValueError:
                    return default

            apix = _float(self._apix_edit.text(), 0)
            cutoff_res = _float(self._cutoff_edit.text(), 0)
            self._mask = self._mask_edit.text().strip()
            mask_file = [self._mask] if self._mask else None
            if mask_file:
                # A provided mask is used verbatim (mask slope is grayed out).
                mask_soft = -1
            else:
                mask_soft = _float(self._mask_soft_edit.text(), -1)

            mode = self._mask_mode_combo.currentData()
            mask_fraction_thresh = -1
            mask_thresh = None
            mask_mass = 0
            if mode == "fraction":
                mask_fraction_thresh = _float(self._mask_value_edit.text(), 1.0)
            elif mode == "pixel":
                val = _float(self._mask_value_edit.text(), None)
                mask_thresh = [val] if val is not None else None
            elif mode == "mass":
                mask_mass = _float(self._mask_value_edit.text(), 0)

            if mask_soft > 0:
                # A manual slope disables slope refinement (mirrors the CLI).
                refine_mask = 0
            else:
                refine_mask = 1 if self._refine_check.isChecked() else 0

            options = {
                "apix": apix,
                "cutoff_res": cutoff_res,
                "one_mask": 1 if self._one_mask_check.isChecked() else 0,
                "mask_file": mask_file,
                "mask_fraction_thresh": mask_fraction_thresh,
                "mask_thresh": mask_thresh,
                "mask_mass": mask_mass,
                "mask_soft": mask_soft,
                "refine_mask": refine_mask,
            }
            return options

        def _on_options_changed(self) -> None:
            """Recompute True FSC when an option changes and maps are ready."""
            self._mask = self._mask_edit.text().strip()
            if self._map1 and self._map2:
                self._seq += 1
                self._start_compute(self._seq)

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
            d1, a1 = volumes["map1"]
            d2, a2 = volumes["map2"]
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
            self._set_fsc_curves(None)
            self._fsc_placeholder.setText("Computing FSC curves…")
            output_dir = _output_dir_for(self._map1)
            plot_file = output_dir / "trueFSC.pdf"
            options = self._get_options()
            worker = _ComputeWorker(
                self._map1,
                self._map2,
                self._mask,
                str(plot_file),
                options,
                parent=self,
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

            self._set_fsc_curves(result.get("fsc_curves"))

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

            if self._fsc_curve_items:
                self._tabs.setCurrentIndex(self._fsc_tab_index)

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
            self._set_fsc_curves(None)
            self._fsc_placeholder.setText("FSC computation failed; see Information.")
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
