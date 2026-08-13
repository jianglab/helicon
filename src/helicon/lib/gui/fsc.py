"""FSC-curve plot window (pyqtgraph) and the flow layout it uses.

RELION ``model.star`` files are parsed into per-iteration/per-class FSC
or W_MAP curves and plotted with an iteration checkbox strip (and a
class checkbox row for Class3D jobs).
"""

from __future__ import annotations

import colorsys
import re
from pathlib import Path

try:
    from PySide6.QtWidgets import QLayout, QMainWindow, QWidget
    from PySide6.QtCore import QPoint, QRect, QSize
except ImportError:  # pragma: no cover - only without the Qt stack
    QLayout = None
    QMainWindow = None
    QWidget = None
    QPoint = None
    QRect = None
    QSize = None

_FscPlotBase = QMainWindow if QMainWindow is not None else object
_FlowLayoutBase = QLayout if QLayout is not None else object
_FlowContainerBase = QWidget if QWidget is not None else object

from .theme import (
    _display_plot_theme_colors,
    _display_theme_palette,
    _display_theme_stylesheet,
)
from .trackers import _install_window_shortcuts, _is_alive_widget, _plot


class _FlowLayout(_FlowLayoutBase):
    """Minimal flow layout that wraps child widgets to the available width.

    Used for the iteration checkbox strip so that jobs with hundreds of
    refinement iterations keep the window a normal size: the checkboxes wrap
    onto multiple rows inside a scroll area instead of stretching the window
    horizontally.  Height-for-width reporting lets the surrounding scroll
    area show a vertical scrollbar once the wrapped content grows past the
    capped strip height.
    """

    def __init__(self, parent=None, margin=0, spacing=6):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing
        self._item_list = []

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self):
        from PySide6.QtCore import Qt

        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(
            margins.left() + margins.right(), margins.top() + margins.bottom()
        )
        return size

    def _do_layout(self, rect, test_only):
        from PySide6.QtCore import Qt

        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        x = effective.x()
        y = effective.y()
        line_height = 0
        line_items = []
        lines = []
        for item in self._item_list:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            hint = item.sizeHint()
            space_x = self._spacing
            space_y = self._spacing
            next_x = x + hint.width() + space_x
            if next_x - space_x > effective.right() and line_height > 0:
                lines.append((y, line_height, line_items))
                x = effective.x()
                y = y + line_height + space_y
                next_x = x + hint.width() + space_x
                line_height = 0
                line_items = []
            line_items.append((item, x, hint))
            x = next_x
            line_height = max(line_height, hint.height())
        if line_items:
            lines.append((y, line_height, line_items))

        if not test_only:
            for line_y, height, items in lines:
                for item, item_x, hint in items:
                    item_y = line_y + (height - hint.height()) // 2
                    item.setGeometry(QRect(QPoint(item_x, item_y), hint))

        if not lines:
            return margins.top() + margins.bottom()
        last_y, last_height, _items = lines[-1]
        return last_y + last_height - rect.y() + margins.bottom()


class _FlowContainer(_FlowContainerBase):
    """QWidget whose size hint reflects the wrapped height of its flow layout.

    The scroll area uses this size hint to decide whether the wrapped
    checkbox rows overflow the capped strip height and need a scrollbar.
    """

    def __init__(self, flow=None):
        super().__init__()
        self._flow = flow

    def sizeHint(self):
        from PySide6.QtCore import QSize

        if self._flow is None:
            return QSize(400, 40)
        width = self.width() if self.width() > 0 else 400
        return QSize(width, self._flow.heightForWidth(width))

    def minimumSizeHint(self):
        # Keep the scroll strip narrow no matter how many checkboxes it
        # holds.  If this reported the full wrapped size, the scroll area's
        # minimum width would push the whole window wide on real displays
        # and the checkboxes would sit on a single row with empty space.
        # The scroll area (not the window) absorbs the overflow instead.
        from PySide6.QtCore import QSize

        return QSize(200, 50)


def _model_fsc_curves(data, *, use_ssnr_map: bool = False):
    """Extract resolution curves from parsed RELION model STAR data.

    Class3D does not use gold-standard half-set refinement, so its
    ``rlnGoldStandardFsc`` values are zero.  For those jobs, convert
    ``rlnSsnrMap`` to the FSC-equivalent MAP weight
    ``W_MAP = SSNR_MAP / (1 + SSNR_MAP)``.
    """
    import numpy as np

    all_curves = []
    class_sections = [
        (int(key.rsplit("_", 1)[-1]), value)
        for key, value in data.items()
        if re.match(r"^(?:data_)?model_class_\d+$", str(key), re.IGNORECASE)
    ]
    for class_num, df in sorted(class_sections):

        sf_col = None
        ang_col = None
        gold_standard_fsc_col = None
        fallback_fsc_col = None
        ssnr_map_col = None
        for col in df.columns:
            cl = col.lower().lstrip("_")
            if "goldstandardfsc" in cl:
                gold_standard_fsc_col = col
            elif "fouriershellcorrelation" in cl and "phase" not in cl:
                fallback_fsc_col = col
            if cl == "rlnssnrmap":
                ssnr_map_col = col
            if cl == "rlnresolution":
                sf_col = col
            if cl == "rlnangstromresolution":
                ang_col = col

        curve_col = (
            ssnr_map_col if use_ssnr_map else gold_standard_fsc_col or fallback_fsc_col
        )
        if sf_col is None or curve_col is None:
            continue

        spatial_freq = np.asarray(df[sf_col], dtype=np.float64)
        curve = np.asarray(df[curve_col], dtype=np.float64)
        if use_ssnr_map:
            curve = curve / (1.0 + curve)

        order = np.argsort(spatial_freq)
        spatial_freq = spatial_freq[order]
        curve = curve[order]

        angstrom = None
        if ang_col is not None:
            angstrom = np.asarray(df[ang_col], dtype=np.float64)[order]

        all_curves.append((spatial_freq, curve, f"Class {class_num}", angstrom))

    return all_curves


_ITERATION_FILE_RE = re.compile(r"(?:^|_)it(\d+)(?:_(.*))?_model\.star$", re.IGNORECASE)


def _iteration_label(path: str | Path) -> str:
    """Human-readable label for a model.star iteration file.

    Numbered RELION snapshots are labelled with the plain iteration number
    (e.g. ``run_it025_model.star`` becomes ``25``) so a row of many checkboxes
    stays compact; files without an iteration marker keep their file stem
    (e.g. ``run_model.star`` becomes ``run_model``).
    """
    path = Path(path)
    match = _ITERATION_FILE_RE.search(path.name)
    if match:
        return str(int(match.group(1)))
    if path.name.lower() == "run_model.star":
        return "final"
    return path.stem


def _iteration_model_files(star_path: str | Path) -> list[tuple[Path, int | None]]:
    """Return refinement-iteration model.star files next to ``star_path``.

    RELION writes one ``run_itNNN_model.star`` per refinement iteration, so the
    "all iterations" set for an FSC plot is every ``*_model.star`` file in the
    same job directory.  Files are sorted by iteration number, and the selected
    file is always included even when its name carries no iteration marker
    (e.g. the final ``run_model.star``).

    Parameters
    ----------
    star_path : str or Path
        The model.star file selected in the file browser.

    Returns
    -------
    list of (Path, int or None)
        ``(path, iteration_number)`` pairs ordered by iteration number;
        ``iteration_number`` is ``None`` for files without an ``_itNNN_``
        marker.
    """
    path = Path(star_path)
    numbered: dict[int, list[tuple[Path, str | None]]] = {}
    unnumbered: list[Path] = []
    candidates = sorted(path.parent.glob("*model.star"))
    if path not in candidates:
        candidates.append(path)
    for candidate in candidates:
        match = _ITERATION_FILE_RE.search(candidate.name)
        if match:
            numbered.setdefault(int(match.group(1)), []).append(
                (candidate, match.group(2))
            )
        elif candidate.name.lower() == "run_model.star":
            # RELION's final Refine3D model is not tagged with an iteration
            # number, but it belongs at the end of the iteration list.
            unnumbered.append(candidate)

    selected_match = _ITERATION_FILE_RE.search(path.name)
    preferred_variant = selected_match.group(2) if selected_match else None
    matches: list[tuple[Path, int | None]] = []
    for iteration_number in sorted(numbered):
        candidates_for_iteration = numbered[iteration_number]
        selected_candidate = next(
            (
                candidate
                for candidate, _variant in candidates_for_iteration
                if candidate == path
            ),
            None,
        )
        if selected_candidate is not None:
            chosen = selected_candidate
        else:
            chosen = next(
                (
                    candidate
                    for candidate, variant in candidates_for_iteration
                    if variant == preferred_variant
                ),
                candidates_for_iteration[0][0],
            )
        matches.append((chosen, iteration_number))

    for candidate in unnumbered:
        matches.append((candidate, None))
    if not any(candidate == path for candidate, _num in matches):
        matches.append((path, None))
    return matches


def _distinct_curve_colors(count: int) -> list[tuple[int, int, int]]:
    """Generate a color sequence with strong separation between neighbors."""
    if count <= 0:
        return []

    # Start from the existing blue hue, then distribute candidate colors
    # evenly around the hue wheel.  Greedily choosing the candidate farthest
    # from the previous one gives an alternating sequence around the wheel,
    # which keeps adjacent curves as distinct as possible.
    base_hue = 210.0 / 360.0
    candidate_hues = [(base_hue + index / count) % 1.0 for index in range(count)]
    remaining = set(range(1, count))
    order = [0]

    def hue_distance(first: float, second: float) -> float:
        distance = abs(first - second)
        return min(distance, 1.0 - distance)

    while remaining:
        previous_hue = candidate_hues[order[-1]]
        next_index = max(
            remaining,
            key=lambda index: hue_distance(previous_hue, candidate_hues[index]),
        )
        order.append(next_index)
        remaining.remove(next_index)

    return [
        tuple(
            round(channel * 255)
            for channel in colorsys.hsv_to_rgb(candidate_hues[index], 0.85, 0.9)
        )
        for index in order
    ]


def _read_fsc_curves(star_path: str | Path, *, use_ssnr_map: bool):
    """Read FSC curves from a model.star file, or ``None`` on any failure.

    Returns the same tuple format as :func:`_model_fsc_curves`, or ``None`` if
    the file cannot be read or contains no usable resolution/FSC columns.
    Used for sibling iteration files, where a missing/broken file should be
    skipped rather than shown as an error.

    Parameters
    ----------
    star_path : str or Path
        Path of the model.star file to read.
    use_ssnr_map : bool
        Convert ``rlnSsnrMap`` to the FSC-equivalent MAP weight (Class3D).

    Returns
    -------
    list or None
        Curves as returned by ``_model_fsc_curves``, or ``None`` on failure.
    """
    try:
        import starfile

        data = starfile.read(str(star_path), always_dict=True)
    except Exception:
        return None
    if not any(
        re.match(r"^(?:data_)?model_class_\d+$", str(key), re.IGNORECASE)
        for key in data
    ):
        return None
    curves = _model_fsc_curves(data, use_ssnr_map=use_ssnr_map)
    return curves or None


class _FscPlotWindow(_FscPlotBase):
    """FSC plot window with an iteration checkbox bar above the plot.

    One checkbox per refinement-iteration model.star file lets the user
    interactively toggle which iterations are overlaid, with "all" / "none"
    shortcut buttons appended to the end of the checkbox strip.  Each
    displayed curve receives a distinct color, with neighboring curves
    chosen to be as visually separated as possible; iterations also use
    distinct line styles.  Class3D jobs plot the SSNR-derived W_MAP curves,
    show one checkbox per class (all checked by default), and only plot the
    checked classes; Refine3D (and other) jobs plot the gold-standard FSC.

    Parameters
    ----------
    iterations : list of tuple(Path, int or None)
        ``(path, iteration_number)`` pairs as returned by
        ``_iteration_model_files``, restricted to files with readable curves.
    curves_by_path : dict
        Mapping ``Path -> list of curves`` in the format returned by
        ``_model_fsc_curves``.
    default_path : str or Path
        Iteration checked by default (the file selected in the browser).
    is_class3d : bool
        Whether the job is a Class3D refinement (W_MAP instead of FSC).
    parent : QWidget, optional
        Parent widget for the window.
    """

    def __init__(
        self,
        iterations: list[tuple[Path, int | None]],
        curves_by_path: dict,
        default_path: str | Path,
        is_class3d: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.setProperty("helicon_theme_window", True)
        self.setStyleSheet(_display_theme_stylesheet())
        self.setPalette(_display_theme_palette())
        self._is_class3d = bool(is_class3d)
        self._paths: list[Path] = []
        self._curves_by_path: dict = {}
        self._checkboxes = []
        self._class_checkboxes = []
        self._curve_items = []
        self._plot = None
        self._legend = None
        self._threshold_line = None
        self._threshold_label = None
        self._crosshair_lines = []
        self._coord_text = None
        self._bar_layout = None
        self._class_bar_layout = None
        self._class_row = None
        self._build_ui()
        self.set_data(iterations, curves_by_path, default_path)
        self.resize(900, 600)

    def _build_ui(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QCheckBox,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )

        import pyqtgraph as pg

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Control bar: one checkbox per iteration, plus "all" / "none"
        # shortcut buttons appended to the end of the checkbox strip.  The
        # checkboxes live in a wrapping, scrollable strip (set_data() adds
        # them) so jobs with hundreds of iterations never stretch the
        # window; a second row holds one checkbox per class (Class3D jobs
        # only).
        bar = QWidget()
        bar_layout = QVBoxLayout(bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        bar_layout.setSpacing(2)

        iter_row = QWidget()
        self._bar_layout = QGridLayout(iter_row)
        self._bar_layout.setContentsMargins(0, 0, 0, 0)
        self._bar_layout.setHorizontalSpacing(8)
        self._bar_layout.setVerticalSpacing(0)
        sample_checkbox = QCheckBox("it000")
        row_height = sample_checkbox.sizeHint().height()
        # A little extra breathing room between neighbouring checkboxes:
        # roughly a quarter of one label character on top of the flow's
        # base spacing.
        label_space = max(
            1, round(sample_checkbox.fontMetrics().horizontalAdvance("0") * 0.25)
        )
        checkbox_spacing = 6 + label_space

        iter_label = QLabel("Iterations:")
        iter_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        iter_label.setFixedHeight(row_height)
        self._bar_layout.addWidget(iter_label, 0, 0, Qt.AlignmentFlag.AlignTop)

        iter_scroll = QScrollArea()
        iter_scroll.setWidgetResizable(True)
        iter_scroll.setFrameShape(QFrame.Shape.NoFrame)
        iter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        iter_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._iter_flow = _FlowLayout(spacing=checkbox_spacing)
        iter_container = _FlowContainer(self._iter_flow)
        iter_container.setLayout(self._iter_flow)
        iter_scroll.setWidget(iter_container)
        self._iter_scroll = iter_scroll
        self._bar_layout.addWidget(iter_scroll, 0, 1, Qt.AlignmentFlag.AlignTop)
        self._bar_layout.setColumnStretch(1, 1)

        select_all_btn = QPushButton("all")
        button_height = max(row_height, select_all_btn.sizeHint().height())
        button_width = max(
            select_all_btn.sizeHint().width(),
            QPushButton("none").sizeHint().width(),
        )
        iter_scroll.setMaximumHeight(button_height * 2 + checkbox_spacing)
        select_all_btn.setFixedHeight(button_height)
        select_all_btn.setFixedWidth(button_width)
        select_all_btn.setStyleSheet("QPushButton { padding-bottom: 2px; }")
        select_all_btn.setToolTip("Plot every iteration")
        select_all_btn.clicked.connect(self._select_all)
        unselect_all_btn = QPushButton("none")
        unselect_all_btn.setFixedHeight(button_height)
        unselect_all_btn.setFixedWidth(button_width)
        unselect_all_btn.setStyleSheet("QPushButton { padding-bottom: 2px; }")
        unselect_all_btn.setToolTip("Plot no iterations")
        unselect_all_btn.clicked.connect(self._unselect_all)
        self._iter_select_all_btn = select_all_btn
        self._iter_unselect_all_btn = unselect_all_btn
        iter_button_group = QWidget()
        iter_button_layout = QHBoxLayout(iter_button_group)
        iter_button_layout.setContentsMargins(0, 0, 0, 0)
        iter_button_layout.setSpacing(6)
        iter_button_layout.addWidget(select_all_btn)
        iter_button_layout.addWidget(unselect_all_btn)
        self._iter_button_group = iter_button_group
        self._iter_flow.addWidget(iter_button_group)
        bar_layout.addWidget(iter_row)

        class_row = QWidget()
        self._class_bar_layout = QGridLayout(class_row)
        self._class_bar_layout.setContentsMargins(0, 0, 0, 0)
        self._class_bar_layout.setHorizontalSpacing(8)
        self._class_bar_layout.setVerticalSpacing(0)
        class_label = QLabel("Classes:")
        self._class_label = class_label
        class_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        class_label.setFixedHeight(row_height)
        self._class_bar_layout.addWidget(class_label, 0, 0, Qt.AlignmentFlag.AlignTop)
        class_scroll = QScrollArea()
        class_scroll.setWidgetResizable(True)
        class_scroll.setFrameShape(QFrame.Shape.NoFrame)
        class_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        class_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._class_flow = _FlowLayout(spacing=checkbox_spacing)
        class_container = _FlowContainer(self._class_flow)
        class_container.setLayout(self._class_flow)
        class_scroll.setWidget(class_container)
        class_scroll.setMaximumHeight(row_height * 2 + 10)
        self._class_scroll = class_scroll
        self._class_bar_layout.addWidget(class_scroll, 0, 1, Qt.AlignmentFlag.AlignTop)
        self._class_bar_layout.setColumnStretch(1, 1)
        class_select_all_btn = QPushButton("all")
        class_select_all_btn.setFixedHeight(button_height)
        class_select_all_btn.setFixedWidth(button_width)
        class_select_all_btn.setStyleSheet("QPushButton { padding-bottom: 2px; }")
        class_select_all_btn.setToolTip("Check every class")
        class_select_all_btn.clicked.connect(self._select_all_classes)
        class_unselect_all_btn = QPushButton("none")
        class_unselect_all_btn.setFixedHeight(button_height)
        class_unselect_all_btn.setFixedWidth(button_width)
        class_unselect_all_btn.setStyleSheet("QPushButton { padding-bottom: 2px; }")
        class_unselect_all_btn.setToolTip("Uncheck every class")
        class_unselect_all_btn.clicked.connect(self._unselect_all_classes)
        class_button_group = QWidget()
        class_button_layout = QHBoxLayout(class_button_group)
        class_button_layout.setContentsMargins(0, 0, 0, 0)
        class_button_layout.setSpacing(6)
        class_button_layout.addWidget(class_select_all_btn)
        class_button_layout.addWidget(class_unselect_all_btn)
        self._class_select_all_btn = class_select_all_btn
        self._class_unselect_all_btn = class_unselect_all_btn
        self._class_button_group = class_button_group
        class_select_all_btn.setVisible(False)
        class_unselect_all_btn.setVisible(False)
        class_button_group.setVisible(False)
        class_row.setHidden(True)
        self._class_row = class_row
        bar_layout.addWidget(class_row)

        layout.addWidget(bar)

        pg.setConfigOptions(antialias=True)
        plot_widget = pg.PlotWidget()
        layout.addWidget(plot_widget)
        self.plot_widget = plot_widget

        plot = plot_widget.getPlotItem()
        self._plot = plot
        plot.getAxis("bottom").enableAutoSIPrefix(False)
        plot.setLabel("bottom", "Resolution", units="1/Å")
        curve_label = "W_MAP" if self._is_class3d else "FSC"
        plot.setLabel("left", curve_label)
        plot.setYRange(0, 1.05)
        self._legend = plot.addLegend()
        plot.showGrid(x=True, y=True, alpha=0.3)

        top_axis = plot.getAxis("top")
        top_axis.enableAutoSIPrefix(False)
        top_axis.setLabel("Resolution", units="Å")

        def _angstrom_tickStrings(values, scale, spacing):
            return [f"{1.0 / v:.1f}" if v > 0 else "" for v in values]

        top_axis.tickStrings = _angstrom_tickStrings
        top_axis.show()

        threshold_pen = pg.mkPen(
            color=(220, 50, 50), width=1, style=Qt.PenStyle.DashLine
        )
        self._threshold_line = pg.InfiniteLine(pos=0.143, angle=0, pen=threshold_pen)
        plot.addItem(self._threshold_line)
        threshold_label = pg.TextItem("0.143", color=(220, 50, 50), anchor=(0, 0))
        self._threshold_label = threshold_label
        threshold_label.setZValue(5)
        plot.addItem(threshold_label, ignoreBounds=True)

        def _position_threshold_label():
            vr = plot.vb.viewRect()
            threshold_label.setPos(vr.left() + vr.width() * 0.005, 0.143)

        _position_threshold_label()
        plot.vb.sigResized.connect(_position_threshold_label)

        crosshair_pen = pg.mkPen(
            color=(150, 150, 150, 160), width=1, style=Qt.PenStyle.DashLine
        )
        vline = pg.InfiniteLine(angle=90, pen=crosshair_pen, movable=False)
        hline = pg.InfiniteLine(angle=0, pen=crosshair_pen, movable=False)
        self._crosshair_lines = [vline, hline]
        plot.addItem(vline, ignoreBounds=True)
        plot.addItem(hline, ignoreBounds=True)
        coord_text = pg.TextItem(anchor=(0, 1), color=(220, 220, 220))
        self._coord_text = coord_text
        coord_text.setZValue(20)
        plot.addItem(coord_text, ignoreBounds=True)

        def _on_mouse_moved(evt):
            pos = plot.vb.mapSceneToView(evt if hasattr(evt, "x") else evt[0])
            x, y = pos.x(), pos.y()
            if not plot.vb.viewRect().contains(pos):
                coord_text.setVisible(False)
                return
            vline.setValue(x)
            hline.setValue(y)
            vline.setVisible(True)
            hline.setVisible(True)
            ang = f"{1.0 / x:.2f}" if x > 0 else "∞"
            coord_text.setHtml(
                f'<span style="color: {self._plot_theme_colors["tooltip_foreground"]};'
                f' background-color: {self._plot_theme_colors["tooltip_background"]};'
                f' padding: 2px;">'
                f"{x:.4f} 1/Å  ({ang} Å)<br>{curve_label} = {y:.4f}</span>"
            )
            coord_text.setVisible(True)
            vr = plot.vb.viewRect()
            dx = vr.width() * 0.02
            dy = vr.height() * 0.02
            tx = x + dx
            ty = y + dy
            if tx + vr.width() * 0.15 > vr.right():
                tx = x - vr.width() * 0.15 - dx
            coord_text.setPos(tx, ty)

        plot_widget.scene().sigMouseMoved.connect(_on_mouse_moved)

        self.setCentralWidget(central)
        self._apply_display_theme()

    def _apply_display_theme(self) -> None:
        """Apply the saved theme to the pyqtgraph canvas and decorations."""
        if self._plot is None:
            return

        from PySide6.QtCore import Qt

        import pyqtgraph as pg

        colors = _display_plot_theme_colors()
        self._plot_theme_colors = colors
        self.plot_widget.setBackground(colors["background"])

        for axis_name in ("left", "bottom", "top", "right"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(colors["foreground"])
            axis.setTextPen(colors["foreground"])
            axis.setTickPen(colors["foreground"])

        curve_label = "W_MAP" if self._is_class3d else "FSC"
        self._plot.setLabel(
            "bottom", "Resolution", units="1/Å", color=colors["foreground"]
        )
        self._plot.setLabel("left", curve_label, color=colors["foreground"])
        self._plot.getAxis("top").setLabel(
            "Resolution", units="Å", color=colors["foreground"]
        )
        self._plot.showGrid(x=True, y=True, alpha=0.3)

        if self._threshold_line is not None:
            self._threshold_line.setPen(
                pg.mkPen(color=(220, 50, 50), width=1, style=Qt.PenStyle.DashLine)
            )
        if self._threshold_label is not None:
            self._threshold_label.setColor((220, 50, 50))
        for line in self._crosshair_lines:
            line.setPen(
                pg.mkPen(
                    color=colors["crosshair"],
                    width=1,
                    style=Qt.PenStyle.DashLine,
                )
            )
        if self._coord_text is not None:
            self._coord_text.setColor(colors["tooltip_foreground"])

        if self._legend is not None:
            for _sample, label in self._legend.items:
                label.setText(label.text, color=colors["foreground"])

    def set_data(
        self,
        iterations: list[tuple[Path, int | None]],
        curves_by_path: dict,
        default_path: str | Path,
        is_class3d: bool | None = None,
    ) -> None:
        """Replace the iteration set and replot, keeping the same window."""
        from PySide6.QtWidgets import QCheckBox

        if is_class3d is not None:
            self._is_class3d = bool(is_class3d)
            curve_label = "W_MAP" if self._is_class3d else "FSC"
            self._plot.setLabel("left", curve_label)

        # Keep the "all"/"none" buttons at the end of the checkbox strip;
        # they are re-appended after the checkboxes below.
        self._iter_flow.removeWidget(self._iter_button_group)

        for checkbox in self._checkboxes:
            try:
                checkbox.toggled.disconnect(self._rebuild_curves)
            except RuntimeError:
                pass
            self._iter_flow.removeWidget(checkbox)
            checkbox.deleteLater()
        self._checkboxes.clear()

        self._paths = [path for path, _num in iterations]
        self._curves_by_path = dict(curves_by_path)

        for path, _num in iterations:
            checkbox = QCheckBox(_iteration_label(path))
            checkbox.setChecked(False)
            self._iter_flow.addWidget(checkbox)
            self._checkboxes.append(checkbox)

        default = Path(default_path)
        default_index = 0
        for index, path in enumerate(self._paths):
            if path == default:
                default_index = index
                break
        if self._checkboxes:
            self._checkboxes[default_index].setChecked(True)

        for checkbox in self._checkboxes:
            checkbox.toggled.connect(self._rebuild_curves)

        self._iter_flow.addWidget(self._iter_button_group)

        self._rebuild_class_checkboxes()
        self._rebuild_curves()

    def _rebuild_class_checkboxes(self) -> None:
        """Replace the class checkboxes, keeping the same window."""
        from PySide6.QtWidgets import QCheckBox

        # The "all"/"none" buttons trail the class checkboxes while the
        # strip is shown; drop them so the rebuilt checkboxes come first.
        self._class_flow.removeWidget(self._class_button_group)

        for checkbox in self._class_checkboxes:
            try:
                checkbox.toggled.disconnect(self._rebuild_curves)
            except RuntimeError:
                pass
            self._class_flow.removeWidget(checkbox)
            checkbox.deleteLater()
        self._class_checkboxes.clear()

        class_count = max(
            (len(curves) for curves in self._curves_by_path.values()), default=0
        )
        show_classes = self._is_class3d and class_count > 1
        self._class_row.setHidden(not show_classes)
        show_class_buttons = show_classes and class_count > 3
        self._class_select_all_btn.setVisible(show_class_buttons)
        self._class_unselect_all_btn.setVisible(show_class_buttons)
        self._class_button_group.setVisible(show_class_buttons)
        if not show_classes:
            return

        for index in range(class_count):
            checkbox = QCheckBox(str(index + 1))
            checkbox.setChecked(True)
            self._class_flow.addWidget(checkbox)
            self._class_checkboxes.append(checkbox)

        for checkbox in self._class_checkboxes:
            checkbox.toggled.connect(self._rebuild_curves)

        if show_class_buttons:
            self._class_flow.addWidget(self._class_button_group)

    def _checked_paths(self) -> list[Path]:
        return [
            path
            for path, checkbox in zip(self._paths, self._checkboxes)
            if checkbox.isChecked()
        ]

    def _select_all(self) -> None:
        if not self._checkboxes:
            return
        for checkbox in self._checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
        for checkbox in self._checkboxes:
            checkbox.blockSignals(False)
        self._rebuild_curves()

    def _unselect_all(self) -> None:
        if not self._checkboxes:
            return
        for checkbox in self._checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
        for checkbox in self._checkboxes:
            checkbox.blockSignals(False)
        self._rebuild_curves()

    def _select_all_classes(self) -> None:
        if not self._class_checkboxes:
            return
        for checkbox in self._class_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
        for checkbox in self._class_checkboxes:
            checkbox.blockSignals(False)
        self._rebuild_curves()

    def _unselect_all_classes(self) -> None:
        if not self._class_checkboxes:
            return
        for checkbox in self._class_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
        for checkbox in self._class_checkboxes:
            checkbox.blockSignals(False)
        self._rebuild_curves()

    def _checked_class_indices(self) -> list[int] | None:
        """Indices of the checked classes, or None when class filtering is off."""
        if not self._class_checkboxes:
            return None
        return [
            index
            for index, checkbox in enumerate(self._class_checkboxes)
            if checkbox.isChecked()
        ]

    def _rebuild_curves(self) -> None:
        """Clear and re-draw the FSC curves for the checked iterations."""
        from PySide6.QtCore import Qt

        import pyqtgraph as pg

        plot = self._plot
        for item in self._curve_items:
            plot.removeItem(item)
        self._curve_items.clear()

        selected = self._checked_paths()
        checked_classes = self._checked_class_indices()
        curves_by_path = {
            path: (curves if self._is_class3d else curves[:1])
            for path, curves in self._curves_by_path.items()
        }
        curve_count = sum(
            sum(
                checked_classes is None or class_index in checked_classes
                for class_index in range(len(curves_by_path[path]))
            )
            for path in selected
        )
        colors = _distinct_curve_colors(curve_count)
        line_styles = [
            Qt.PenStyle.SolidLine,
            Qt.PenStyle.DashLine,
            Qt.PenStyle.DotLine,
            Qt.PenStyle.DashDotLine,
            Qt.PenStyle.DashDotDotLine,
        ]
        multi = len(self._paths) > 1
        color_index = 0
        for it_index, path in enumerate(self._paths):
            if path not in selected:
                continue
            curves = curves_by_path[path]
            style = line_styles[it_index % len(line_styles)]
            it_label = _iteration_label(path)
            for class_index, (spatial_freq, fsc, label, _angstrom) in enumerate(curves):
                if checked_classes is not None and class_index not in checked_classes:
                    continue
                color = colors[color_index]
                color_index += 1
                pen = pg.mkPen(color=color, width=2, style=style)
                if not self._is_class3d:
                    curve_name = it_label
                else:
                    curve_name = f"{it_label} · {label}" if multi else label
                self._curve_items.append(
                    plot.plot(spatial_freq, fsc, pen=pen, name=curve_name)
                )
        self._apply_display_theme()

    def closeEvent(self, event):
        _plot.on_close(self)
        super().closeEvent(event)

    def changeEvent(self, event):
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            _plot.on_activate(self)
        super().changeEvent(event)


def _open_fsc_plot(star_path: str, reuse_window=None) -> None:
    """Display the FSC curves from RELION model.star files using pyqtgraph.

    Reads ``data_model_class_N`` sections and plots resolution vs. FSC for
    each class.  Class3D jobs use the FSC-equivalent MAP weight calculated
    from ``rlnSsnrMap``; other jobs use the gold-standard FSC column.  A
    horizontal line at 0.143 marks the FSC threshold.

    When the job directory contains more than one ``*_model.star`` file (one
    per refinement iteration), a checkbox bar at the top of the plot lets the
    user interactively choose which iterations to overlay, with the file
    selected in the browser checked by default and "all" / "none" shortcut
    buttons at the end of the checkbox strip.  Class3D jobs get an extra row
    of checkboxes, one per class (all checked by default), that filter which
    classes are drawn.
    """
    try:
        import pyqtgraph  # noqa: F401 - availability probe
        from PySide6.QtWidgets import QMessageBox
    except ImportError:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(
            None,
            "Missing dependency",
            "pyqtgraph is required to display FSC curves.\n"
            "Install it with: pip install pyqtgraph",
        )
        return

    try:
        import starfile

        data = starfile.read(star_path, always_dict=True)
    except Exception:
        QMessageBox.warning(
            None,
            "Error",
            f"Failed to read {Path(star_path).name}.\n"
            "Make sure it is a valid RELION model.star file.",
        )
        return

    if not any(
        re.match(r"^(?:data_)?model_class_\d+$", str(key), re.IGNORECASE)
        for key in data
    ):
        QMessageBox.information(
            None,
            "No FSC data",
            f"No data_model_class_N sections found in {Path(star_path).name}.",
        )
        return

    is_class3d = any(
        part.lower().startswith("class3d") for part in Path(star_path).parts
    )
    all_curves = _model_fsc_curves(data, use_ssnr_map=is_class3d)

    if not all_curves:
        curve_column = "rlnSsnrMap" if is_class3d else "FSC"
        QMessageBox.warning(
            None,
            "Column error",
            f"Could not find resolution/{curve_column} columns in any class section.",
        )
        return

    # Read curves for every iteration file in the job directory.  The file
    # selected in the browser is already parsed; sibling files that cannot be
    # read or have no usable FSC columns are skipped silently.
    iterations = _iteration_model_files(star_path)
    primary = Path(star_path)
    curves_by_path = {}
    for path, _num in iterations:
        curves = (
            all_curves
            if path == primary
            else _read_fsc_curves(str(path), use_ssnr_map=is_class3d)
        )
        if curves:
            curves_by_path[path] = curves

    plot_iterations = [
        (path, num) for path, num in iterations if path in curves_by_path
    ]
    if not plot_iterations:
        QMessageBox.warning(
            None,
            "Column error",
            "Could not find resolution/FSC columns in any selected file.",
        )
        return

    name = Path(star_path).name
    window_title = (
        name
        if len(plot_iterations) == 1
        else f"{name} ({len(plot_iterations)} iterations)"
    )

    if (
        reuse_window is not None
        and _is_alive_widget(reuse_window)
        and isinstance(reuse_window, _FscPlotWindow)
    ):
        reuse_window.set_data(
            plot_iterations,
            curves_by_path,
            default_path=primary,
            is_class3d=is_class3d,
        )
        reuse_window.setWindowTitle(f"FSC — {window_title}")
        reuse_window.show()
        reuse_window.raise_()
        return

    win = _FscPlotWindow(
        plot_iterations,
        curves_by_path,
        default_path=primary,
        is_class3d=is_class3d,
    )
    win.setWindowTitle(f"FSC — {window_title}")
    _plot.register(win)
    _install_window_shortcuts(win)
    win.show()
