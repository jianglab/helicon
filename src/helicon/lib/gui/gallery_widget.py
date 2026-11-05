"""
This module provides :class:`ImageGalleryWidget`, a Qt widget that displays a
grid of image thumbnails for a stack (e.g. a ``.mrcs`` particle stack or a
data ``.star`` file). Only the images currently visible in the viewport are
read from the (lazy) data source.

The viewport-culling math is kept framework-agnostic in :class:`GalleryPanel`
so it can be unit-tested without a Qt application.
"""

from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

import math

import numpy as np

from helicon.lib.gui.caches import LRUCache
from helicon.lib.gui.gallery_style import (  # noqa: F401 (also what callers and tests take from here)
    _apply_gallery_theme,
    _button_icon,
    _gallery_qss,
    _gallery_theme_colors,
)


def _tile_contrast(frame: np.ndarray) -> tuple[float, float]:
    """Black and white points of a gallery tile.

    The 0.5th and 99.5th percentiles: only outlier pixels are clipped. A
    median +/- 3 MAD window, which suits a micrograph, is too narrow for a
    class average: most of its pixels are flat background outside the mask,
    so the window clipped the particle itself to white.

    Parameters
    ----------
    frame : numpy.ndarray
        2D image data.

    Returns
    -------
    tuple[float, float]
        (black, white), with white > black.
    """
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return 0.0, 1.0
    black, white = (float(v) for v in np.percentile(finite, (0.5, 99.5)))
    if black >= white:
        black, white = float(finite.min()), float(finite.max())
        if black >= white:
            white = black + 1.0
    return black, white


class GalleryPanel:
    """
    Given the widget viewport size, a zoom ``scale``, and the per-image rendered
    size, compute which rows/columns are visible and where vertical scrolling
    starts. Pure Python — no Qt, OpenGL, or napari dependencies.

    Parameters
    ----------
    n_img : int
        Total number of images in the stack.
    img_w : int
        Native width of a single image, in pixels.
    img_h : int
        Native height of a single image, in pixels.
    min_sep : int, optional
        Minimum gap (in pixels) between adjacent thumbnails. Defaults to ``2``.
    """

    def __init__(self, n_img: int, img_w: int, img_h: int, min_sep: int = 2):
        self.n_img = int(n_img)
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.min_sep = int(min_sep)

        # Populated by update_panel_params().
        self.rendered_w = self.img_w + self.min_sep
        self.rendered_h = self.img_h + self.min_sep
        self.visiblecols = 1
        self.visiblerows = 1
        self.xoffset = 0
        self.rowstart = 0
        self.height = 0
        self.scroll_x = 0
        self.scroll_y = 0
        self.max_y = 0

    def update_panel_params(
        self,
        view_w: int,
        view_h: int,
        scale: float,
        scroll_x: int = 0,
        scroll_y: int = 0,
    ) -> None:
        """Recompute layout parameters from the current view and scroll state.

        This method is a pure calculator: it does NOT mutate ``scroll_y``.
        The caller owns the authoritative scroll offset and passes it in each
        call; the returned ``rowstart`` is derived from it. Clamping is the
        caller's responsibility (see :meth:`clamp_scroll_y`), so repeated
        calls with consistent inputs never shift the scroll position.

        Parameters
        ----------
        view_w : int
            Viewport width in pixels.
        view_h : int
            Viewport height in pixels.
        scale : float
            Zoom factor applied to native image size.
        scroll_x : int, optional
            Horizontal scroll offset in pixels (positive = scrolled right).
        scroll_y : int, optional
            Vertical scroll offset in pixels (positive = scrolled down).
        """
        self.rendered_w = int(round(self.img_w * scale)) + self.min_sep
        self.rendered_h = int(round(self.img_h * scale)) + self.min_sep

        self.visiblecols = max(0, int(view_w) // self.rendered_w)
        self.visiblerows = max(0, int(view_h) // self.rendered_h)

        # Continuous vertical anchor. Tiles are positioned as
        #   ty(r) = r * rendered_h + scroll_y
        # (row 0 sits at scroll_y, row 1 one tile lower, etc.). This is a
        # strictly monotonic function of scroll_y, so the grid scrolls
        # smoothly instead of snapping by a whole row each time an integer
        # row boundary is crossed (the buggy ``scroll_y % rendered_h`` term).
        self.xoffset = int(scroll_x % self.rendered_w)
        self.rowstart = max(0, (-int(scroll_y)) // self.rendered_h)
        self.height = self.visiblerows * self.rendered_h

        if self.visiblecols > 0:
            total_rows = max(1, int(math.ceil(self.n_img / self.visiblecols)))
            # Furthest-down scroll keeps the bottom of the last row aligned
            # with the bottom of the viewport. view_h - total_rows*rendered_h
            # is <= 0 exactly when the stack overflows the viewport, and the
            # min(0, ...) keeps the top from scrolling past the start.
            self.max_y = min(0, int(view_h) - total_rows * self.rendered_h)
        else:
            self.max_y = 0

        self.scroll_x = int(scroll_x)

    def clamp_scroll_y(self, scroll_y: int) -> int:
        """Clamp a desired vertical scroll offset into ``[max_y, 0]``.

        Parameters
        ----------
        scroll_y : int
            Desired vertical scroll offset (negative = scrolled down).

        Returns
        -------
        int
            The clamped offset, guaranteed within the valid range for the
            layout most recently computed by :meth:`update_panel_params`.
        """
        return int(max(self.max_y, min(0, scroll_y)))

    def visible_row_col(
        self,
        view_w: int,
        view_h: int,
        scale: float,
        scroll_x: int = 0,
        scroll_y: int = 0,
    ):
        """Return the visible grid window, or ``None`` if scale is too large.

        Parameters
        ----------
        view_w : int
            Viewport width in pixels.
        view_h : int
            Viewport height in pixels.
        scale : float
            Zoom factor.
        scroll_x : int, optional
            Horizontal scroll offset.
        scroll_y : int, optional
            Vertical scroll offset.

        Returns
        -------
        list or None
            ``[rowstart, visiblerows, visiblecols]`` when at least one column
            fits, otherwise ``None`` (the caller should clamp ``scale``).
        """
        self.update_panel_params(view_w, view_h, scale, scroll_x, scroll_y)
        if self.visiblecols <= 0:
            return None
        return [self.rowstart, self.visiblerows, self.visiblecols]


class _MultiSelectComboBox(QComboBox):
    """Drop-down list of check boxes: any number of its items can be chosen.

    Clicking an item toggles it and leaves the list open; the closed box
    shows the chosen items joined by commas.

    Signals
    -------
    selection_changed(list)
        The checked item texts, in the order of the list, after each toggle.
    """

    selection_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModel(QStandardItemModel(self))
        # the plain delegate draws the check boxes in every style
        self.setItemDelegate(QStyledItemDelegate(self))
        self.view().viewport().installEventFilter(self)

    def set_items(self, items: list[str], checked: list[str]) -> None:
        """Replace the items, checking those in ``checked``. Emits nothing."""
        model = self.model()
        model.clear()
        for text in items:
            item = QStandardItem(text)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if text in checked else Qt.Unchecked)
            model.appendRow(item)
        self._update_tooltip()
        self.update()

    def checked_items(self) -> list[str]:
        """The texts of the checked items, in the order of the list."""
        model = self.model()
        return [
            model.item(r).text()
            for r in range(model.rowCount())
            if model.item(r).checkState() == Qt.Checked
        ]

    def toggle_item(self, row: int) -> None:
        """Toggle the check box of item ``row`` and emit the new selection."""
        item = self.model().item(row)
        if item is None:
            return
        state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
        item.setCheckState(state)
        self._update_tooltip()
        self.update()
        self.selection_changed.emit(self.checked_items())

    def display_text(self) -> str:
        """What the closed box shows."""
        return ", ".join(self.checked_items()) or "(none)"

    def _update_tooltip(self) -> None:
        self.setToolTip("\n".join(self.checked_items()))

    def eventFilter(self, obj, event):
        # a click on an item toggles it; swallowing the release keeps the
        # list open, so several items can be chosen in one go
        if obj is self.view().viewport() and event.type() == QEvent.MouseButtonRelease:
            index = self.view().indexAt(event.position().toPoint())
            if index.isValid():
                self.toggle_item(index.row())
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        # the arrow keys of a closed box would pick a single item
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.showPopup()
            return
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        # scrolling over the closed box would pick a single item
        event.ignore()

    def paintEvent(self, event) -> None:
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        width = (
            self.style()
            .subControlRect(QStyle.CC_ComboBox, opt, QStyle.SC_ComboBoxEditField, self)
            .width()
        )
        opt.currentText = self.fontMetrics().elidedText(
            self.display_text(), Qt.ElideRight, max(width, 0)
        )
        opt.currentIcon = QIcon()
        painter.drawComplexControl(QStyle.CC_ComboBox, opt)
        painter.drawControl(QStyle.CE_ComboBoxLabel, opt)


class _ControlPanel(QWidget):
    """Collapsible side panel with brightness / contrast / gamma controls.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    PANEL_WIDTH = 180
    z_thickness_changed = Signal(float)
    sort_column_changed = Signal(str)
    sort_reverse_changed = Signal(bool)
    log_changed = Signal(bool)
    histogram_changed = Signal(bool)
    show_labels_changed = Signal(bool)
    label_columns_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        colors = _gallery_theme_colors()
        self.setFixedWidth(self.PANEL_WIDTH)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(colors["panel"]))
        self.setPalette(palette)
        self.setStyleSheet(_gallery_qss(colors))

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # the look of the labels, sliders, buttons and boxes is the stylesheet of the
        # panel (_gallery_qss), so a change of theme reaches all of them
        lbl_style = slider_style = btn_style = radio_style = chk_style = ""

        # --- Brightness ---
        root.addWidget(self._label("Brightness", lbl_style))
        self._brightness_slider = self._make_slider(-100, 100, 0, slider_style)
        self._brightness_val = QLabel("0", parent=self)
        self._brightness_val.setStyleSheet(lbl_style)
        self._brightness_val.setAlignment(Qt.AlignCenter)
        row_b = QHBoxLayout()
        row_b.addWidget(self._brightness_slider)
        row_b.addWidget(self._brightness_val)
        root.addLayout(row_b)

        # --- Contrast ---
        root.addWidget(self._label("Contrast", lbl_style))
        self._contrast_slider = self._make_slider(1, 300, 100, slider_style)
        self._contrast_val = QLabel("1.00", parent=self)
        self._contrast_val.setStyleSheet(lbl_style)
        self._contrast_val.setAlignment(Qt.AlignCenter)
        row_c = QHBoxLayout()
        row_c.addWidget(self._contrast_slider)
        row_c.addWidget(self._contrast_val)
        root.addLayout(row_c)

        # --- Gamma ---
        root.addWidget(self._label("Gamma", lbl_style))
        self._gamma_slider = self._make_slider(1, 300, 100, slider_style)
        self._gamma_val = QLabel("1.00", parent=self)
        self._gamma_val.setStyleSheet(lbl_style)
        self._gamma_val.setAlignment(Qt.AlignCenter)
        row_g = QHBoxLayout()
        row_g.addWidget(self._gamma_slider)
        row_g.addWidget(self._gamma_val)
        root.addLayout(row_g)

        # --- Autocontrast ---
        self._auto_btn = QPushButton("Autocontrast")
        self._auto_btn.setStyleSheet(btn_style)
        icon = _button_icon("fa5s.magic", colors)
        if icon is not None:
            self._auto_btn.setIcon(icon)
        root.addWidget(self._auto_btn)

        # --- Histogram ---
        self._histogram_chk = QCheckBox("Show Histogram")
        self._histogram_chk.setChecked(True)
        self._histogram_chk.setStyleSheet(chk_style)
        self._histogram_chk.toggled.connect(self.histogram_changed)
        root.addWidget(self._histogram_chk)

        self._histogram_widget = _HistogramWidget()
        root.addWidget(self._histogram_widget)

        # --- Log transform ---
        self._log_chk = QCheckBox("Log Transform")
        self._log_chk.setChecked(False)
        self._log_chk.setStyleSheet(chk_style)
        self._log_chk.toggled.connect(self.log_changed)
        root.addWidget(self._log_chk)

        # --- Scope ---
        root.addWidget(self._label("Scope", lbl_style))
        self._scope_group = QButtonGroup(self)
        self._radio_selected = QRadioButton("Selected image")
        self._radio_selected.setStyleSheet(radio_style)
        self._radio_all = QRadioButton("All visible")
        self._radio_all.setStyleSheet(radio_style)
        self._radio_all.setChecked(True)
        self._scope_group.addButton(self._radio_selected, 0)
        self._scope_group.addButton(self._radio_all, 1)
        root.addWidget(self._radio_selected)
        root.addWidget(self._radio_all)

        root.addSpacing(6)
        self._show_labels_chk = QCheckBox("Show Labels")
        self._show_labels_chk.setChecked(True)
        self._show_labels_chk.setStyleSheet(chk_style)
        self._show_labels_chk.toggled.connect(self.show_labels_changed)
        root.addWidget(self._show_labels_chk)

        # which columns of the star file the labels show (2D classes only)
        self._label_columns_label = self._label("Label columns", lbl_style)
        self._label_columns_label.setVisible(False)
        root.addWidget(self._label_columns_label)
        self._label_columns_combo = _MultiSelectComboBox()
        self._label_columns_combo.setVisible(False)
        self._label_columns_combo.selection_changed.connect(self.label_columns_changed)
        root.addWidget(self._label_columns_combo)

        root.addSpacing(6)
        self._z_thickness_sep = QFrame()
        self._z_thickness_sep.setFrameShape(QFrame.HLine)
        self._z_thickness_sep.setStyleSheet(f"color: {colors['border']};")
        self._z_thickness_sep.setVisible(False)
        root.addWidget(self._z_thickness_sep)

        self._z_thickness_label = self._label("Z Thickness (Å)", lbl_style)
        self._z_thickness_spin = QDoubleSpinBox()
        self._z_thickness_spin.setRange(0.0, 99999.0)
        self._z_thickness_spin.setDecimals(1)
        self._z_thickness_spin.setSingleStep(1.0)
        self._z_thickness_spin.setValue(0.0)
        self._z_thickness_spin.setSuffix(" Å")
        self._z_thickness_label.setVisible(False)
        self._z_thickness_spin.setVisible(False)
        self._z_thickness_spin.valueChanged.connect(self.z_thickness_changed)
        root.addWidget(self._z_thickness_label)
        root.addWidget(self._z_thickness_spin)

        root.addSpacing(6)
        self._sort_sep = QFrame()
        self._sort_sep.setFrameShape(QFrame.HLine)
        self._sort_sep.setStyleSheet(f"color: {colors['border']};")
        self._sort_sep.setVisible(False)
        root.addWidget(self._sort_sep)

        combo_style = ""
        self._sort_column_combo = QComboBox()
        self._sort_column_combo.setStyleSheet(combo_style)
        self._sort_column_combo.setVisible(False)
        self._sort_column_combo.currentTextChanged.connect(self.sort_column_changed)
        self._sort_label = self._label("Sort by", lbl_style)
        self._sort_label.setVisible(False)
        root.addWidget(self._sort_label)
        root.addWidget(self._sort_column_combo)

        self._sort_reverse_chk = QCheckBox("Reverse Sort")
        self._sort_reverse_chk.setStyleSheet(chk_style)
        self._sort_reverse_chk.setChecked(True)
        self._sort_reverse_chk.setVisible(False)
        self._sort_reverse_chk.toggled.connect(self.sort_reverse_changed)
        root.addWidget(self._sort_reverse_chk)

        root.addStretch(1)

    def _apply_display_theme(self) -> None:
        """Apply the current theme to the gallery adjustment controls."""
        colors = _gallery_theme_colors()
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(colors["panel"]))
        self.setPalette(palette)
        self.setStyleSheet(_gallery_qss(colors))
        for child in self.findChildren(QWidget):
            child.setStyleSheet("")
        icon = _button_icon("fa5s.magic", colors)
        if icon is not None:
            self._auto_btn.setIcon(icon)
        self.update()

    def show_z_thickness(self, visible: bool = True, maximum: float = 99999.0) -> None:
        self._z_thickness_sep.setVisible(visible)
        self._z_thickness_label.setVisible(visible)
        self._z_thickness_spin.setVisible(visible)
        self._z_thickness_spin.setMaximum(maximum)
        if visible:
            self._z_thickness_spin.setValue(0.0)

    def show_sort_ui(
        self, visible: bool = True, columns: list[str] | None = None
    ) -> None:
        self._sort_label.setVisible(visible)
        self._sort_sep.setVisible(visible)
        self._sort_column_combo.setVisible(visible)
        self._sort_reverse_chk.setVisible(visible)
        if visible and columns is not None:
            current = self._sort_column_combo.currentText()
            self._sort_column_combo.blockSignals(True)
            self._sort_column_combo.clear()
            self._sort_column_combo.addItems(columns)
            if current in columns:
                self._sort_column_combo.setCurrentText(current)
            self._sort_column_combo.blockSignals(False)

    def show_label_columns_ui(
        self,
        visible: bool = True,
        columns: list[str] | None = None,
        checked: list[str] | None = None,
    ) -> None:
        """Show the multi-select list of the columns the labels show.

        Parameters
        ----------
        visible : bool
            Whether to show the list.
        columns : list of str, optional
            The columns to choose from, in the order of the list.
        checked : list of str, optional
            The columns chosen at first.
        """
        self._label_columns_label.setVisible(visible)
        self._label_columns_combo.setVisible(visible)
        if visible and columns is not None:
            self._label_columns_combo.set_items(columns, checked or [])

    def show_log_ui(self, visible: bool = True) -> None:
        self._log_chk.setVisible(visible)

    def show_histogram_ui(self, visible: bool = True) -> None:
        self._histogram_chk.setVisible(visible)
        self._histogram_widget.setVisible(visible)

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _label(text: str, style: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    @staticmethod
    def _make_slider(lo: int, hi: int, val: int, style: str) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.setStyleSheet(style)
        return s


class _HistogramWidget(QWidget):
    """Histogram of pixel values with BCG transfer curve overlay.

    Samples a few images from the stack, bins pixel values into 256 buckets,
    and draws them as gray bars.  A white curve shows the current
    brightness/contrast/gamma mapping from normalized [0, 1] → display [0, 1].
    The black- and white-point handles can be dragged horizontally to change
    brightness and contrast; dragging the unclipped part of the curve
    vertically changes gamma.
    """

    HIST_HEIGHT = 100
    HANDLE_RADIUS = 5.0
    HIT_RADIUS = 7.0
    MIN_CONTRAST = 0.01
    MAX_CONTRAST = 3.0
    MIN_GAMMA = 0.01
    MAX_GAMMA = 3.0

    bcg_changed = Signal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.HIST_HEIGHT)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(
            self.backgroundRole(), QColor(_gallery_theme_colors()["background"])
        )
        self.setPalette(palette)
        self._bins: np.ndarray | None = None
        self._brightness = 0.0
        self._contrast = 1.0
        self._gamma = 1.0
        self._log_transform = False
        self._drag_target: str | None = None
        self._gamma_drag_base: float | None = None
        self.setMouseTracking(True)
        self.setToolTip(
            "Drag the end handles left/right for brightness and contrast; "
            "drag the curve up/down for gamma."
        )

    def _apply_display_theme(self) -> None:
        """Apply the current theme to the histogram canvas."""
        colors = _gallery_theme_colors()
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(colors["background"]))
        self.setPalette(palette)
        self.update()

    def update_histogram(
        self,
        read_fn,
        n: int,
        brightness: float = 0.0,
        contrast: float = 1.0,
        gamma: float = 1.0,
        log_transform: bool = False,
    ) -> None:
        if read_fn is None or n <= 0:
            self._bins = None
            self.update()
            return
        self._brightness = brightness
        self._contrast = contrast
        self._gamma = gamma
        self._log_transform = log_transform
        step = max(1, n // 10)
        samples = []
        for i in range(0, n, step):
            try:
                frame = read_fn(i).astype(np.float64)
                if log_transform:
                    frame = np.log1p(frame - frame.min())
                samples.append(frame.ravel())
            except Exception:
                pass
        if not samples:
            self._bins = None
            self.update()
            return
        all_px = np.concatenate(samples)
        lo, hi = float(np.min(all_px)), float(np.max(all_px))
        if hi - lo < 1e-12:
            self._bins = np.zeros(256, dtype=np.float64)
            self.update()
            return
        normed = np.clip((all_px - lo) / (hi - lo), 0.0, 1.0) * 255.0
        self._bins, _ = np.histogram(normed, bins=256, range=(0, 255))
        self.update()

    def set_bcg(self, brightness: float, contrast: float, gamma: float) -> None:
        """Update only the transfer curve, retaining the cached histogram."""
        self._brightness = float(brightness)
        self._contrast = float(contrast)
        self._gamma = float(gamma)
        self.update()

    def _plot_rect(self) -> QRectF:
        margin = self.HANDLE_RADIUS + 1.0
        return QRectF(
            margin,
            margin,
            max(1.0, self.width() - 2.0 * margin),
            max(1.0, self.height() - 2.0 * margin),
        )

    def _linear_transfer(self, x: float | np.ndarray) -> float | np.ndarray:
        """Return the transfer value before gamma and output clipping."""
        return (x + self._brightness - 0.5) * self._contrast + 0.5

    def _endpoint_values(self) -> tuple[float, float]:
        """Return normalized input positions of the black and white points."""
        contrast = max(self.MIN_CONTRAST, self._contrast)
        half_width = 0.5 / contrast
        center = 0.5 - self._brightness
        return center - half_width, center + half_width

    def _endpoint_screen_position(self, target: str) -> QPointF | None:
        black, white = self._endpoint_values()
        x = black if target == "black" else white
        if not 0.0 <= x <= 1.0:
            return None
        rect = self._plot_rect()
        y = rect.bottom() if target == "black" else rect.top()
        return QPointF(rect.left() + x * rect.width(), y)

    def _hit_target(self, pos: QPointF) -> str | None:
        """Identify a draggable handle or the active part of the curve."""
        for target in ("black", "white"):
            point = self._endpoint_screen_position(target)
            if (
                point is not None
                and math.hypot(pos.x() - point.x(), pos.y() - point.y())
                <= self.HIT_RADIUS
            ):
                return target

        rect = self._plot_rect()
        if not rect.adjusted(
            -self.HIT_RADIUS,
            -self.HIT_RADIUS,
            self.HIT_RADIUS,
            self.HIT_RADIUS,
        ).contains(pos):
            return None
        x = float(np.clip((pos.x() - rect.left()) / rect.width(), 0.0, 1.0))
        base = float(self._linear_transfer(x))
        # Gamma cannot move the clipped horizontal sections of the curve.
        if not 1e-6 < base < 1.0 - 1e-6:
            return None
        curve_y = float(self._transfer(np.asarray([x]))[0])
        screen_y = rect.bottom() - curve_y * rect.height()
        if abs(pos.y() - screen_y) <= self.HIT_RADIUS:
            return "gamma"
        return None

    def _emit_endpoint_drag(self, target: str, screen_x: float) -> None:
        """Convert a dragged black/white point back to brightness/contrast."""
        rect = self._plot_rect()
        dragged = float(np.clip((screen_x - rect.left()) / rect.width(), 0.0, 1.0))
        black, white = self._endpoint_values()
        min_gap = 1.0 / self.MAX_CONTRAST
        if target == "black":
            black = min(dragged, white - min_gap)
        else:
            white = max(dragged, black + min_gap)
        contrast = float(
            np.clip(
                1.0 / max(white - black, min_gap),
                self.MIN_CONTRAST,
                self.MAX_CONTRAST,
            )
        )
        brightness = float(np.clip(0.5 - (black + white) / 2.0, -1.0, 1.0))
        self.bcg_changed.emit(brightness, contrast, self._gamma)

    def _emit_gamma_drag(self, screen_y: float) -> None:
        """Choose gamma so the grabbed curve point follows the mouse vertically."""
        if self._gamma_drag_base is None:
            return
        rect = self._plot_rect()
        output = float(
            np.clip((rect.bottom() - screen_y) / rect.height(), 1e-6, 1.0 - 1e-6)
        )
        gamma = math.log(self._gamma_drag_base) / math.log(output)
        gamma = float(np.clip(gamma, self.MIN_GAMMA, self.MAX_GAMMA))
        self.bcg_changed.emit(self._brightness, self._contrast, gamma)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        self._drag_target = self._hit_target(event.position())
        if self._drag_target == "gamma":
            rect = self._plot_rect()
            x = float(
                np.clip((event.position().x() - rect.left()) / rect.width(), 0.0, 1.0)
            )
            self._gamma_drag_base = float(
                np.clip(self._linear_transfer(x), 1e-6, 1.0 - 1e-6)
            )
        else:
            self._gamma_drag_base = None
        if self._drag_target is not None:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_target in ("black", "white"):
            self._emit_endpoint_drag(self._drag_target, event.position().x())
            event.accept()
            return
        if self._drag_target == "gamma":
            self._emit_gamma_drag(event.position().y())
            event.accept()
            return

        target = self._hit_target(event.position())
        if target in ("black", "white"):
            self.setCursor(Qt.SizeHorCursor)
        elif target == "gamma":
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.unsetCursor()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drag_target is not None:
            self._drag_target = None
            self._gamma_drag_base = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        if self._drag_target is None:
            self.unsetCursor()
        super().leaveEvent(event)

    def _transfer(self, x: np.ndarray) -> np.ndarray:
        """Apply the BCG transfer function to normalized [0, 1] values."""
        y = self._linear_transfer(x)
        if self._gamma != 1.0:
            y = np.clip(y, 0.0, 1.0) ** (1.0 / self._gamma)
        return np.clip(y, 0.0, 1.0)

    def paintEvent(self, event) -> None:
        from PySide6.QtCore import QPointF

        painter = QPainter(self)
        colors = _gallery_theme_colors()
        painter.fillRect(self.rect(), QColor(colors["background"]))
        if self._bins is None:
            painter.end()
            return
        rect = self._plot_rect()
        margin = int(rect.left())
        draw_w = rect.width()
        draw_h = rect.height()
        max_bin = max(1, int(self._bins.max()))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(colors["muted"]))
        for i in range(256):
            bx = margin + int(i * draw_w / 255)
            bw = max(1, int(draw_w / 256))
            bh = int(self._bins[i] * draw_h / max_bin)
            if bh > 0:
                painter.drawRect(bx, int(rect.bottom() - bh), bw, bh)

        x_vals = np.linspace(0.0, 1.0, 256)
        y_vals = self._transfer(x_vals)
        pts = QPolygonF()
        for i in range(256):
            px = rect.left() + i * draw_w / 255.0
            py = rect.bottom() - y_vals[i] * draw_h
            pts.append(QPointF(px, py))
        painter.setPen(QColor(colors["text"]))
        painter.setBrush(Qt.NoBrush)
        painter.drawPolyline(pts)

        painter.setPen(QPen(QColor(colors["text"]), 1.5))
        painter.setBrush(QColor(colors["accent"]))
        for target in ("black", "white"):
            point = self._endpoint_screen_position(target)
            if point is not None:
                painter.drawEllipse(point, self.HANDLE_RADIUS, self.HANDLE_RADIUS)

        painter.end()


class ImageGalleryWidget(QWidget):
    """Lazy thumbnail grid for an image stack, rendered with ``QPainter``.

    Only the images intersecting the current viewport are read (via the
    supplied ``read_fn``), so very large stacks cost nothing until scrolled
    into view. Wheel-zoom changes the zoom factor and re-flows the grid;
    left-drag pans; a click (without drag) on a thumbnail emits
    :attr:`image_activated` with the global image index.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    """

    image_activated = Signal(int)
    panel_toggle_requested = Signal()

    # Smallest zoom: the longer side of a tile stays at least this many
    # screen pixels, so zooming out never asks for thousands of tiles.
    MIN_TILE_PX = 16
    # Bounds of the raw-frame and rendered-pixmap caches.
    FRAME_CACHE_ITEMS = 4096
    FRAME_CACHE_BYTES = 512 * 2**20
    THUMB_CACHE_ITEMS = 4096
    THUMB_CACHE_BYTES = 256 * 2**20

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setContentsMargins(0, 0, 0, 0)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(
            self.backgroundRole(), QColor(_gallery_theme_colors()["background"])
        )
        self.setPalette(palette)

        self._read_fn = None
        self._n = 0
        self._img_w = 0
        self._img_h = 0
        self._dtype = None
        self._labels: list[str] | None = None
        self._source_name: str | None = None
        self._scale = 1.0
        self._scroll_y = 0
        self._panel = GalleryPanel(0, 1, 1)
        self._coords: dict[int, QRect] = {}
        # Raw frames (as read) and rendered pixmaps are cached separately:
        # a brightness/contrast/gamma change only re-renders the pixmaps,
        # it does not re-read the frames. Both caches are bounded.
        self._frame_cache = LRUCache(
            max_items=self.FRAME_CACHE_ITEMS, max_bytes=self.FRAME_CACHE_BYTES
        )
        self._thumb_cache = LRUCache(
            max_items=self.THUMB_CACHE_ITEMS, max_bytes=self.THUMB_CACHE_BYTES
        )
        self._drag_last = None
        self._dragged = False
        self._sb_width = 12
        self._scrubbing = False
        self._zooming = False

        # --- adjustment state (driven externally via setters) ---
        self._brightness = 0.0
        self._contrast = 1.0
        self._gamma = 1.0
        self._log_transform = False
        self._show_labels = True
        self._adjust_scope = "all"
        self._selected_idx: int | None = None

    def _apply_display_theme(self) -> None:
        """Apply the current theme to the thumbnail canvas."""
        colors = _gallery_theme_colors()
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(colors["background"]))
        self.setPalette(palette)
        self.update()

    # --------------------------------------------------------------- setters
    def set_brightness(self, val: float) -> None:
        self._brightness = val
        self._thumb_cache.clear()
        self.update()

    def set_contrast(self, val: float) -> None:
        self._contrast = val
        self._thumb_cache.clear()
        self.update()

    def set_gamma(self, val: float) -> None:
        self._gamma = val
        self._thumb_cache.clear()
        self.update()

    def set_adjust_scope(self, scope: str) -> None:
        self._adjust_scope = scope
        self._thumb_cache.clear()
        self.update()

    def set_log_transform(self, val: bool) -> None:
        self._log_transform = val
        self._thumb_cache.clear()
        self.update()

    def set_show_labels(self, val: bool) -> None:
        self._show_labels = val
        self.update()

    def reset_adjustments(self) -> None:
        self._brightness = 0.0
        self._contrast = 1.0
        self._gamma = 1.0
        self._thumb_cache.clear()
        self.update()

    def set_selected_idx(self, idx: int | None) -> None:
        self._selected_idx = idx
        self._thumb_cache.clear()
        self.update()

    # ------------------------------------------------------------------ data
    def set_data(
        self,
        read_fn,
        n: int,
        img_w: int,
        img_h: int,
        dtype,
        labels: list[str] | None = None,
        source_name: str | None = None,
    ) -> None:
        """Bind a lazy data source and trigger a repaint.

        Parameters
        ----------
        read_fn : callable
            ``read_fn(i) -> numpy.ndarray`` returning the 2D frame at index
            ``i``. Must perform I/O lazily; only visible indices are requested.
        n : int
            Total number of images in the stack.
        img_w : int
            Native width of a single image.
        img_h : int
            Native height of a single image.
        dtype : numpy.dtype or type
            Data type of the frames (used only for bookkeeping).
        source_name : str, optional
            Original source file name (without expecting a specific suffix).
            Used to pre-fill the save dialog with a matching base name.
        """
        self._read_fn = read_fn
        self._n = int(n)
        self._img_w = int(img_w)
        self._img_h = int(img_h)
        self._dtype = dtype
        self._labels = labels
        self._source_name = source_name
        self._panel = GalleryPanel(self._n, self._img_w, self._img_h)
        self._coords = {}
        self._frame_cache.clear()
        self._thumb_cache.clear()
        self._scroll_y = 0
        self._selected_idx = None

        self.update()

    def invalidate_frames(self) -> None:
        """Drop the cached frames and pixmaps and repaint.

        Call this when ``read_fn`` starts returning different data for the
        same indices (re-sorting, a new slab thickness, ...). Display
        adjustments only need the pixmap cache cleared.
        """
        self._frame_cache.clear()
        self._thumb_cache.clear()
        self.update()

    def has_data(self) -> bool:
        """Return ``True`` if a data source is bound with at least one image."""
        return self._read_fn is not None and self._n > 0

    # -------------------------------------------------------------- rendering
    def _to_thumb(
        self,
        frame: np.ndarray,
        brightness: float | None = None,
        contrast: float | None = None,
        gamma: float | None = None,
    ) -> QPixmap:
        """Normalise a frame to an 8-bit grayscale ``QPixmap`` thumbnail.

        Parameters
        ----------
        frame : numpy.ndarray
            2D image data.
        brightness : float, optional
            Additive offset in ``[-1, 1]`` applied *after* normalisation.
            ``None`` uses the instance attribute.
        contrast : float, optional
            Multiplicative contrast centred at 1.0.  ``None`` uses the
            instance attribute.
        gamma : float, optional
            Gamma-correction value (1.0 = no correction).  ``None`` uses
            the instance attribute.

        Returns
        -------
        QPixmap
            Grayscale thumbnail.
        """
        if self._log_transform:
            # Work in float: ``frame - frame.min()`` and ``log1p`` overflow or
            # wrap around for integer frames.
            frame = np.asarray(frame, dtype=np.float64)
            frame = np.log1p(frame - frame.min())
        black, white = _tile_contrast(frame)
        arr = frame.astype(np.float64)
        arr = np.clip((arr - black) / max(1e-9, white - black), 0.0, 1.0)

        b = brightness if brightness is not None else self._brightness
        c = contrast if contrast is not None else self._contrast
        g = gamma if gamma is not None else self._gamma

        # brightness: additive offset
        arr = arr + b
        # contrast: multiplicative around midpoint
        arr = (arr - 0.5) * c + 0.5
        # gamma: power-law correction
        if g != 1.0:
            arr = np.clip(arr, 0.0, 1.0) ** (1.0 / g)
        arr = np.clip(arr, 0.0, 1.0)

        arr = (arr * 255.0).astype(np.uint8)
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())

    def _canvas_width(self, with_sb: bool = True) -> int:
        """Usable width for tiles (viewport minus the vertical scrollbar).

        Parameters
        ----------
        with_sb : bool, optional
            When ``True`` (default) a vertical scrollbar is assumed present
            and its width is reserved on the right. Pass ``False`` when the
            scrollbar is not shown so the gallery can use (and center in) the
            full widget width.
        """
        sb = self._sb_width if with_sb else 0
        return max(1, self.width() - sb)

    def _needs_scrollbar(self) -> bool:
        """Return ``True`` if the stack overflows the viewport vertically."""
        panel = GalleryPanel(self._n, self._img_w, self._img_h, self._panel.min_sep)
        panel.visible_row_col(
            self._canvas_width(with_sb=False),
            self.height(),
            self._scale,
            0,
            self._scroll_y,
        )
        return panel.max_y < 0

    def _fit_scale_for_width(self, canvas_w: int) -> float:
        """Largest zoom scale that keeps at least one column visible.

        Wide frames (e.g. a 3710 px movie micrograph) do not fit a single
        column at the default scale of 1.0; this clamps the scale so the
        first paint is not a blank canvas.
        """
        available = max(1, canvas_w - self._panel.min_sep)
        return max(1e-3, min(20.0, available / max(1, self._img_w)))

    def _min_zoom_scale(self) -> float:
        """Smallest zoom scale: the longer tile side stays ``MIN_TILE_PX``."""
        longest = max(1, self._img_w, self._img_h)
        return max(1e-3, self.MIN_TILE_PX / longest)

    def _clamp_zoom_scale(self, scale: float) -> float:
        """Clamp a requested zoom scale to the allowed range.

        The lower bound keeps tiles at least ``MIN_TILE_PX`` on screen (a
        paint then reads a bounded number of frames); the upper bound keeps
        one column visible and never exceeds 20x. Fitting one column wins
        over the lower bound.
        """
        scale = max(scale, self._min_zoom_scale())
        if self._img_w > 0:
            available = self._canvas_width() - self._panel.min_sep
            max_fit = available / max(1, self._img_w)
            scale = min(scale, max(max_fit, 1e-3))
        return min(scale, 20.0)

    def _frame(self, i: int) -> np.ndarray:
        """Return frame ``i``, reading it through the bounded frame cache."""
        frame = self._frame_cache.get(i)
        if frame is None:
            frame = self._read_fn(i)
            self._frame_cache[i] = frame
        return frame

    def _max_scroll(self, view_h: int) -> int:
        """Maximum vertical scroll offset (negative) for the current layout."""
        self._panel.visible_row_col(
            self._canvas_width(), view_h, self._scale, 0, self._scroll_y
        )
        return int(self._panel.max_y)

    def _clamp_scroll(self) -> None:
        """Clamp the authoritative ``_scroll_y`` using the current layout."""
        self._panel.visible_row_col(
            self._canvas_width(), self.height(), self._scale, 0, self._scroll_y
        )
        self._scroll_y = self._panel.clamp_scroll_y(self._scroll_y)

    def _apply_zoom(self, factor: float, cx: int, cy: int) -> None:
        """Zoom by ``factor`` about the canvas point ``(cx, cy)``."""
        if self._img_w <= 0:
            return
        new_scale = self._clamp_zoom_scale(self._scale * factor)
        if new_scale == self._scale:
            return

        old_h = int(round(self._img_h * self._scale)) + self._panel.min_sep
        frac_row = (cy - self._scroll_y) / old_h if old_h > 0 else 0.0

        self._scale = new_scale
        self._clamp_scroll()

        new_h = int(round(self._img_h * self._scale)) + self._panel.min_sep
        self._scroll_y = cy - frac_row * new_h
        self._clamp_scroll()
        self.update()

    def _scrollbar_rect(self) -> QRect:
        """Track rectangle of the vertical scrollbar (right edge)."""
        w = self.width()
        h = self.height()
        return QRect(w - self._sb_width, 0, self._sb_width, h)

    def _scrollbar_thumb_rect(self) -> QRect | None:
        """Thumb rectangle of the vertical scrollbar, or ``None`` if not needed."""
        view_h = self.height()
        max_scroll = self._max_scroll(view_h)
        if max_scroll == 0:
            return None
        track = self._scrollbar_rect()
        total_rows = int(np.ceil(self._n / max(1, self._panel.visiblecols)))
        visible_rows = max(1, self._panel.visiblerows)
        frac = min(1.0, visible_rows / total_rows)
        thumb_h = max(20, int(track.height() * frac))
        frac_scrolled = (-self._scroll_y) / (-max_scroll) if max_scroll < 0 else 0.0
        frac_scrolled = min(1.0, max(0.0, frac_scrolled))
        y = int(track.y() + frac_scrolled * (track.height() - thumb_h))
        return QRect(track.x() + 2, y, self._sb_width - 4, thumb_h)

    def paintEvent(self, event) -> None:
        """Draw only the visible tiles, reading each lazily on first paint."""
        # Rebuild the visible-tile map from scratch each paint so it never
        # retains stale rects from a previous scroll/zoom position.  _save_as
        # and the click hit-test both rely on it matching exactly what is on
        # screen.
        self._coords = {}
        painter = QPainter(self)
        colors = _gallery_theme_colors()
        painter.fillRect(self.rect(), QColor(colors["background"]))

        if not self.has_data():
            painter.end()
            return

        size = self.size()
        needs_sb = self._needs_scrollbar()
        canvas_w = self._canvas_width(with_sb=needs_sb)
        res = self._panel.visible_row_col(
            canvas_w, size.height(), self._scale, 0, self._scroll_y
        )
        if res is None:
            # No column fits at the current zoom: shrink until at least one
            # column is visible so wide single images are drawn, not blanked.
            # Reserve the scrollbar width so the fit stays valid even when a
            # vertical scrollbar appears after shrinking.
            self._scale = self._fit_scale_for_width(self._canvas_width(with_sb=True))
            needs_sb = self._needs_scrollbar()
            canvas_w = self._canvas_width(with_sb=needs_sb)
            res = self._panel.visible_row_col(
                canvas_w, size.height(), self._scale, 0, self._scroll_y
            )
        if res is None:
            painter.end()
            return
        rowstart, visiblerows, visiblecols = res

        self._scroll_y = self._panel.clamp_scroll_y(self._scroll_y)

        block_w = visiblecols * self._panel.rendered_w
        x_start = max(0, (canvas_w - block_w) // 2)

        label_font = painter.font()
        label_font.setPixelSize(max(9, min(13, int(self._panel.rendered_h * 0.28))))
        painter.setFont(label_font)
        label_color = QColor(colors["text"])
        label_bg = QColor(colors["label_background"])

        for r in range(rowstart, rowstart + visiblerows + 1):
            row_count = min(visiblecols, max(0, self._n - r * visiblecols))
            if row_count <= 0:
                break
            row_x_start = x_start + (block_w - row_count * self._panel.rendered_w) // 2

            draw_w = self._panel.rendered_w - self._panel.min_sep
            draw_h = self._panel.rendered_h - self._panel.min_sep
            for c in range(row_count):
                i = r * visiblecols + c
                tx = row_x_start + c * self._panel.rendered_w
                ty = r * self._panel.rendered_h + self._scroll_y
                if i not in self._thumb_cache:
                    # In "selected" mode, only apply adjustments to the
                    # selected thumbnail; others use identity values.
                    if self._adjust_scope == "selected" and i != self._selected_idx:
                        self._thumb_cache[i] = self._to_thumb(
                            self._frame(i),
                            brightness=0.0,
                            contrast=1.0,
                            gamma=1.0,
                        )
                    else:
                        self._thumb_cache[i] = self._to_thumb(self._frame(i))
                thumb = self._thumb_cache[i]
                painter.drawPixmap(tx, ty, draw_w, draw_h, thumb)
                self._coords[i] = QRect(tx, ty, draw_w, draw_h)

                if self._show_labels:
                    text = (
                        self._labels[i]
                        if self._labels and i < len(self._labels)
                        else str(i)
                    )
                    trect = QRect(tx, ty, draw_w, draw_h)
                    metrics = painter.fontMetrics()
                    # a label may hold several lines (one per chosen column)
                    lines = text.split("\n") if text else [""]
                    tw = max(metrics.horizontalAdvance(line) for line in lines)
                    th = metrics.height()
                    pad = 2
                    painter.save()
                    painter.setClipRect(trect)
                    painter.fillRect(
                        tx,
                        ty,
                        min(tw + 2 * pad, trect.width()),
                        th * len(lines) + 2 * pad,
                        label_bg,
                    )
                    painter.setPen(label_color)
                    for k, line in enumerate(lines):
                        painter.drawText(
                            tx + pad, ty + pad + k * th + metrics.ascent(), line
                        )
                    painter.restore()

        if needs_sb:
            track = self._scrollbar_rect()
            painter.fillRect(track, QColor(colors["scrollbar"]))
            thumb_rect = self._scrollbar_thumb_rect()
            if thumb_rect is not None:
                painter.fillRect(thumb_rect, QColor(colors["scrollbar_thumb"]))

    # ------------------------------------------------------------- interaction
    def wheelEvent(self, event) -> None:
        """Scroll vertically (or zoom when the Ctrl modifier is held)."""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 1.0 / 1.1
            self._scale = self._clamp_zoom_scale(self._scale * factor)
            self.update()
            event.accept()
            return

        if self._max_scroll(self.height()) == 0:
            event.accept()
            return
        self._scroll_y = self._scroll_y + event.angleDelta().y()
        self._clamp_scroll()
        self.update()
        event.accept()

    def _hit_scrollbar(self, pos) -> bool:
        """Return True if ``pos`` is over the scrollbar track or thumb."""
        return self._scrollbar_rect().contains(pos)

    def contextMenuEvent(self, event) -> None:
        """Ignore right-click so it drives zoom instead of a context menu."""
        event.ignore()

    def _show_save_menu(self, pos):
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        menu.addAction("Save Canvas As…", self._save_as)
        menu.exec(self.mapToGlobal(pos))

    def _save_as(self):
        from PySide6.QtCore import QRect

        from helicon.commands.display import _save_qimage

        # Grab the full widget, then copy out the tight bounding box of the
        # drawn thumbnail rects so the saved image contains the real content
        # and not the surrounding gray padding or the scrollbar.  We grab the
        # whole widget (QWidget.grab(rect) is unreliable on HiDPI builds and
        # returns the full widget) and crop in device pixels afterwards.
        full = self.grab()
        if self._coords:
            min_x = max(0, min(r.x() for r in self._coords.values()))
            min_y = max(0, min(r.y() for r in self._coords.values()))
            max_x = min(
                self.width(), max(r.x() + r.width() for r in self._coords.values())
            )
            max_y = min(
                self.height(), max(r.y() + r.height() for r in self._coords.values())
            )
            dpr = full.devicePixelRatio()
            cx = int(round(min_x * dpr))
            cy = int(round(min_y * dpr))
            cw = int(round((max_x - min_x) * dpr))
            ch = int(round((max_y - min_y) * dpr))
            qimg = full.copy(QRect(cx, cy, cw, ch))
        else:
            qimg = full
        _save_qimage(qimg, self, default_name=self._source_name)

    def mousePressEvent(self, event) -> None:
        """Begin vertical drag, cursor zoom, scrollbar scrub, or toggle panel."""
        if event.button() == Qt.MiddleButton:
            self.panel_toggle_requested.emit()
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self._drag_last = event.pos()
            self._dragged = False
            self._scrubbing = self._hit_scrollbar(event.pos())
        elif event.button() == Qt.RightButton:
            self._drag_last = event.pos()
            self._dragged = False
            self._zooming = True

    def mouseMoveEvent(self, event) -> None:
        """Pan, zoom-to-cursor, or scrub the scrollbar thumb on drag."""
        if self._drag_last is None or not event.buttons():
            return
        dy = event.y() - self._drag_last.y()

        if self._zooming:
            factor = float(np.exp(-dy * 0.005))
            anchor = self._drag_last
            self._apply_zoom(factor, anchor.x(), anchor.y())
            self._drag_last = event.pos()
            if abs(dy) > 1:
                self._dragged = True
            return

        if self._scrubbing:
            max_scroll = self._max_scroll(self.height())
            if max_scroll < 0:
                track = self._scrollbar_rect()
                thumb = self._scrollbar_thumb_rect()
                if thumb is not None:
                    usable = track.height() - thumb.height()
                    num = event.y() - track.y() - thumb.height() // 2
                    frac = min(1.0, max(0.0, num / max(1, usable)))
                    self._scroll_y = max_scroll * frac
                self._drag_last = event.pos()
                self.update()
                return

        self._scroll_y -= dy
        self._clamp_scroll()
        self._drag_last = event.pos()
        if abs(dy) > 3:
            self._dragged = True
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        """Emit the clicked image index if the press was not a drag/scrub."""
        if (
            event.button() == Qt.LeftButton
            and not self._dragged
            and not self._scrubbing
        ):
            for i, rect in self._coords.items():
                if rect.contains(event.pos()):
                    self._selected_idx = i
                    self._thumb_cache.clear()
                    self.image_activated.emit(i)
                    self.update()
                    break
        if event.button() == Qt.RightButton and not self._dragged:
            self._show_save_menu(event.pos())
        self._drag_last = None
        self._scrubbing = False
        self._zooming = False


# the orthogonal viewer moved to its own module; its names are still importable from here
from helicon.lib.gui.orthogonal_viewer import (  # noqa: E402,F401
    OrthogonalViewerWidget,
    _AXIS_COLORS,
    _BCGPanel,
    _COLOR_X,
    _COLOR_Y,
    _COLOR_Z,
    _ControlBar,
    _DARK_BG,
    _SliceView,
)
