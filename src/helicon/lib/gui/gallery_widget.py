"""
This module provides :class:`ImageGalleryWidget`, a Qt widget that displays a
grid of image thumbnails for a stack (e.g. a ``.mrcs`` particle stack or a
data ``.star`` file). Only the images currently visible in the viewport are
read from the (lazy) data source.

The viewport-culling math is kept framework-agnostic in :class:`GalleryPanel`
so it can be unit-tested without a Qt application.
"""

from PySide6.QtCore import QTimer, QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import math

import numpy as np


def _gallery_theme_colors() -> dict[str, str]:
    """Return colors for custom-painted gallery widgets."""
    from helicon.lib.gui.file_browser import (
        _THEME_COLORS,
        _resolved_theme,
        _saved_theme,
    )

    resolved = _resolved_theme(_saved_theme())
    colors = _THEME_COLORS[resolved]
    if resolved == "Light":
        return {
            "background": "#ffffff",
            "panel": colors["window"],
            "text": colors["text"],
            "input": colors["input"],
            "border": colors["border"],
            "accent": colors["accent"],
            "muted": "#707070",
            # QColor does not parse CSS rgba() strings reliably in Qt6;
            # use 8-digit ARGB hex so the painter preserves the alpha.
            "label_background": "#d2ffffff",
            "scrollbar": "#d0d0d0",
            "scrollbar_thumb": "#a0a0a0",
        }
    return {
        "background": "#2d2d2d",
        "panel": "#3a3a3a",
        "text": "#e8e8e8",
        "input": "#555555",
        "border": "#777777",
        "accent": "#5a82c4",
        "muted": "#aaaaaa",
        "label_background": "#8c000000",
        "scrollbar": "#1f1f1f",
        "scrollbar_thumb": "#5a5a5a",
    }


def _gallery_qss(colors: dict[str, str]) -> str:
    """Return a theme-aware stylesheet for gallery controls."""
    return f"""
        QWidget {{
            background-color: {colors["panel"]};
            color: {colors["text"]};
        }}
        QLabel, QCheckBox, QRadioButton {{
            color: {colors["text"]};
        }}
        QSlider::groove:horizontal {{
            background: {colors["border"]};
            height: 4px;
        }}
        QSlider::handle:horizontal {{
            background: {colors["text"]};
            width: 12px;
            margin: -4px 0;
            border-radius: 6px;
        }}
        QPushButton, QComboBox, QSpinBox, QDoubleSpinBox {{
            color: {colors["text"]};
            background: {colors["input"]};
            border: 1px solid {colors["border"]};
            border-radius: 3px;
            padding: 3px;
        }}
        QPushButton:hover {{
            background: {colors["accent"]};
        }}
        QComboBox QAbstractItemView {{
            color: {colors["text"]};
            background: {colors["input"]};
        }}
    """


def _apply_gallery_theme(widget: QWidget) -> None:
    """Refresh gallery child widgets after the shared theme changes."""
    for child in [widget, *widget.findChildren(QWidget)]:
        apply_theme = getattr(child, "_apply_display_theme", None)
        if apply_theme is not None:
            apply_theme()
        else:
            colors = _gallery_theme_colors()
            palette = child.palette()
            palette.setColor(child.backgroundRole(), QColor(colors["panel"]))
            child.setPalette(palette)
            child.update()


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

        lbl_style = f"color: {colors['text']}; font-size: 11px;"
        slider_style = (
            f"QSlider::groove:horizontal {{ background: {colors['border']}; height: 4px; }}"
            f"QSlider::handle:horizontal {{ background: {colors['text']}; width: 12px; "
            "margin: -4px 0; border-radius: 6px; }"
        )
        btn_style = (
            f"QPushButton {{ color: {colors['text']}; background: {colors['input']}; "
            f"border: 1px solid {colors['border']}; "
            "border-radius: 3px; padding: 4px; }"
            f"QPushButton:hover {{ background: {colors['border']}; }}"
        )
        radio_style = f"QRadioButton {{ color: {colors['text']}; font-size: 11px; }}"
        chk_style = f"QCheckBox {{ color: {colors['text']}; font-size: 11px; }}"

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
        self._z_thickness_spin.setStyleSheet(
            f"QDoubleSpinBox {{ color: {colors['text']}; background: {colors['input']}; "
            f"border: 1px solid {colors['border']}; "
            "border-radius: 3px; padding: 3px; }"
        )
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

        combo_style = (
            f"QComboBox {{ color: {colors['text']}; background: {colors['input']}; "
            f"border: 1px solid {colors['border']}; "
            "border-radius: 3px; padding: 3px; font-size: 11px; }"
            "QComboBox::drop-down { border: none; }"
            f"QComboBox QAbstractItemView {{ color: {colors['text']}; "
            f"background: {colors['panel']}; }}"
        )
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
    """

    HIST_HEIGHT = 100

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

    def _transfer(self, x: np.ndarray) -> np.ndarray:
        """Apply the BCG transfer function to normalized [0, 1] values."""
        y = x + self._brightness
        y = (y - 0.5) * self._contrast + 0.5
        if self._gamma != 1.0:
            y = np.clip(y, 0.0, 1.0) ** (1.0 / self._gamma)
        return np.clip(y, 0.0, 1.0)

    def paintEvent(self, event) -> None:
        from PySide6.QtCore import QPointF

        painter = QPainter(self)
        colors = _gallery_theme_colors()
        painter.fillRect(self.rect(), QColor(colors["background"]))
        if self._bins is None or self._bins.sum() == 0:
            painter.end()
            return
        w = self.width()
        h = self.height()
        margin = 4
        draw_w = w - 2 * margin
        draw_h = h - 2 * margin
        max_bin = max(1, int(self._bins.max()))

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(colors["muted"]))
        for i in range(256):
            bx = margin + int(i * draw_w / 255)
            bw = max(1, int(draw_w / 256))
            bh = int(self._bins[i] * draw_h / max_bin)
            if bh > 0:
                painter.drawRect(bx, margin + draw_h - bh, bw, bh)

        x_vals = np.linspace(0.0, 1.0, 256)
        y_vals = self._transfer(x_vals)
        pts = QPolygonF()
        for i in range(256):
            px = margin + i * draw_w / 255.0
            py = margin + draw_h - y_vals[i] * draw_h
            pts.append(QPointF(px, py))
        painter.setPen(QColor(colors["text"]))
        painter.setBrush(Qt.NoBrush)
        painter.drawPolyline(pts)

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        from PySide6.QtWidgets import QSizePolicy

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
        self._thumb_cache: dict[int, QPixmap] = {}
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
        self._thumb_cache = {}
        self._scroll_y = 0
        self._selected_idx = None

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
        from helicon.commands.display import _auto_contrast

        if self._log_transform:
            frame = np.log1p(frame - frame.min())
        black, white = _auto_contrast(frame)
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
        new_scale = self._scale * factor
        available = self._canvas_width() - self._panel.min_sep
        max_fit = available / max(1, self._img_w)
        new_scale = min(max(new_scale, 1e-3), max(max_fit, 1e-3), 20.0)
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
                            self._read_fn(i),
                            brightness=0.0,
                            contrast=1.0,
                            gamma=1.0,
                        )
                    else:
                        self._thumb_cache[i] = self._to_thumb(self._read_fn(i))
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
                    tw = metrics.horizontalAdvance(text)
                    th = metrics.height()
                    pad = 2
                    painter.save()
                    painter.setClipRect(trect)
                    painter.fillRect(
                        tx, ty, min(tw + 2 * pad, trect.width()), th + 2 * pad, label_bg
                    )
                    painter.setPen(label_color)
                    painter.drawText(tx + pad, ty + pad + metrics.ascent(), text)
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
            new_scale = self._scale * factor
            if self._img_w > 0:
                available = self._canvas_width() - self._panel.min_sep
                max_fit = available / max(1, self._img_w)
                new_scale = min(new_scale, max(max_fit, 1e-3))
            new_scale = min(new_scale, 20.0)
            self._scale = new_scale
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


_DARK_BG = QColor("#2d2d2d")
_COLOR_X = QColor(255, 0, 0)
_COLOR_Y = QColor(0, 255, 0)
_COLOR_Z = QColor(0, 100, 255)
_AXIS_COLORS = {"x": _COLOR_X, "y": _COLOR_Y, "z": _COLOR_Z}


class _SliceView(QWidget):
    """Single 2D slice view with pan, zoom, cross-hair, and marker lines.

    Signals
    -------
    clicked(float, float)
        Emitted with data-space coordinates when the user clicks.
    panned(int, int)
        Emitted with pixel deltas when the user drags.
    zoomed(float)
        Emitted with zoom factor when the user scrolls.
    """

    clicked = Signal(float, float)
    panned = Signal(int, int)
    zoomed = Signal(float)
    middle_clicked = Signal()

    def __init__(self, parent=None, axis_label: str = ""):
        super().__init__(parent)
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(
            self.backgroundRole(), QColor(_gallery_theme_colors()["background"])
        )
        self.setPalette(pal)

        self._image: np.ndarray | None = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._crosshair: tuple[float, float] | None = None
        self._crosshair_color_x = QColor(255, 255, 0)
        self._crosshair_color_y = QColor(255, 255, 0)
        # (position_data_units, color, "h"|"v")
        self._markers: list[tuple[float, QColor, str]] = []
        self._brightness = 0.0
        self._contrast = 1.0
        self._gamma = 1.0
        self._log_transform = False
        self._drag_last: QPoint | None = None
        self._border_color = QColor(255, 255, 255)
        self._axis_label = axis_label
        # h = horizontal arrow, v = vertical arrow (e.g. Z panel: h="x", v="y").
        self._axes_h = ""
        self._axes_v = ""

    def _apply_display_theme(self) -> None:
        """Apply the current theme to this custom-painted slice view."""
        colors = _gallery_theme_colors()
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(colors["background"]))
        self.setPalette(pal)
        self._border_color = QColor(colors["text"])
        self.update()

    def set_image(self, data: np.ndarray | None) -> None:
        self._image = data
        self.update()

    def set_border_color(self, color: QColor) -> None:
        self._border_color = color

    def set_axes(self, h_label: str, v_label: str) -> None:
        """Set the in-plane axis names drawn as a mini axes icon.

        Parameters
        ----------
        h_label : str
            Axis name for the horizontal arrow (e.g. ``"x"`` for the Z panel).
        v_label : str
            Axis name for the vertical arrow (e.g. ``"y"`` for the Z panel).
        """
        self._axes_h = h_label
        self._axes_v = v_label
        self.update()

    def set_crosshair(
        self,
        x: float,
        y: float,
        color_x: QColor | None = None,
        color_y: QColor | None = None,
    ) -> None:
        self._crosshair = (x, y)
        if color_x is not None:
            self._crosshair_color_x = color_x
        if color_y is not None:
            self._crosshair_color_y = color_y
        self.update()

    def set_markers(self, markers: list[tuple[float, QColor, str]]) -> None:
        self._markers = markers
        self.update()

    def set_bcg(self, brightness: float, contrast: float, gamma: float) -> None:
        self._brightness = brightness
        self._contrast = contrast
        self._gamma = gamma
        self.update()

    def set_log_transform(self, val: bool) -> None:
        self._log_transform = val
        self.update()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.05, zoom)
        self.update()

    def set_pan(self, x: float, y: float) -> None:
        self._pan_x = x
        self._pan_y = y
        self.update()

    def _apply_bcg(self, data: np.ndarray) -> np.ndarray:
        """Apply brightness/contrast/gamma to raw slice data."""
        arr = data.astype(np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-9:
            arr = np.zeros_like(arr)
        else:
            arr = (arr - lo) / (hi - lo)
        if self._log_transform:
            arr = np.log1p(arr)
        arr = arr + self._brightness
        arr = (arr - 0.5) * self._contrast + 0.5
        if self._gamma != 1.0:
            arr = np.clip(arr, 0.0, 1.0) ** (1.0 / self._gamma)
        return np.clip(arr, 0.0, 1.0)

    def _data_to_screen(self, dx: float, dy: float) -> tuple[float, float]:
        w, h = self.width(), self.height()
        if self._image is None:
            return 0.0, 0.0
        ih, iw = self._image.shape[:2]
        scale = min(w / max(iw, 1), h / max(ih, 1)) * self._zoom
        sx = (w - iw * scale) / 2 + self._pan_x
        sy = (h - ih * scale) / 2 + self._pan_y
        return sx + dx * scale, sy + dy * scale

    def _screen_to_data(self, sx: float, sy: float) -> tuple[float, float]:
        w, h = self.width(), self.height()
        if self._image is None:
            return 0.0, 0.0
        ih, iw = self._image.shape[:2]
        scale = min(w / max(iw, 1), h / max(ih, 1)) * self._zoom
        ix = (w - iw * scale) / 2 + self._pan_x
        iy = (h - ih * scale) / 2 + self._pan_y
        return (sx - ix) / max(scale, 1e-9), (sy - iy) / max(scale, 1e-9)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        colors = _gallery_theme_colors()
        painter.fillRect(self.rect(), QColor(colors["background"]))

        if self._image is None or self._image.size == 0:
            painter.end()
            return

        w, h = self.width(), self.height()
        ih, iw = self._image.shape[:2]
        scale = min(w / max(iw, 1), h / max(ih, 1)) * self._zoom
        ox = (w - iw * scale) / 2 + self._pan_x
        oy = (h - ih * scale) / 2 + self._pan_y

        arr = self._apply_bcg(self._image)
        # QImage requires a contiguous, stable buffer. This is especially
        # important for the transposed X slice, which is a NumPy view.
        gray = np.ascontiguousarray((arr * 255).astype(np.uint8))
        qimg = QImage(gray.data, iw, ih, gray.strides[0], QImage.Format_Grayscale8)
        qimg = qimg.copy()
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(
            QRectF(ox, oy, iw * scale, ih * scale), qimg, QRectF(0, 0, iw, ih)
        )

        for mpos, mcolor, morient in self._markers:
            painter.setPen(QPen(mcolor, 1))
            if morient == "v":
                sx, _ = self._data_to_screen(mpos, 0)
                painter.drawLine(QPointF(sx, 0), QPointF(sx, h))
            else:
                _, sy = self._data_to_screen(0, mpos)
                painter.drawLine(QPointF(0, sy), QPointF(w, sy))

        if self._crosshair is not None:
            cx, cy = self._crosshair
            sx, sy = self._data_to_screen(cx, cy)
            pen = QPen(self._crosshair_color_x, 0.5, Qt.DashLine)
            pen.setDashPattern([8, 12])
            painter.setPen(pen)
            painter.drawLine(QPointF(sx, 0), QPointF(sx, h))
            pen = QPen(self._crosshair_color_y, 0.5, Qt.DashLine)
            pen.setDashPattern([8, 12])
            painter.setPen(pen)
            painter.drawLine(QPointF(0, sy), QPointF(w, sy))

        painter.setPen(QPen(self._border_color, 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        if self._axis_label:
            label_color = _AXIS_COLORS.get(
                self._axis_label.lower(), QColor(colors["text"])
            )
            painter.setPen(label_color)
            painter.setFont(QFont("Arial", 14, QFont.Bold))
            painter.drawText(6, 18, self._axis_label)

        self._draw_axes_icon(painter)

        painter.end()

    def _draw_axes_icon(self, painter: QPainter) -> None:
        """Draw a mini axes icon (horizontal + vertical labeled arrows).

        Placed at the top-left, just below the plane label. Each arrow is
        colored by the standard per-axis color (X red, Y green, Z blue) so it
        matches the crosshair colors drawn elsewhere in the panel. Both arrows
        share a common tail at the top-left corner so the icon reads as a
        corner origin instead of a centered cross.
        """
        if not self._axes_h or not self._axes_v:
            return
        colors = _gallery_theme_colors()
        h_color = _AXIS_COLORS.get(self._axes_h, QColor(colors["text"]))
        v_color = _AXIS_COLORS.get(self._axes_v, QColor(colors["text"]))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        pen = QPen()
        pen.setWidthF(1.5)

        # Common tail at the top-left corner (an "L" shape).
        ox, oy = 10, 30  # origin
        h_len, v_len = 22, 22
        pen.setColor(h_color)
        painter.setPen(pen)
        # Horizontal arrow pointing right from the origin.
        hx2 = ox + h_len
        painter.drawLine(QPointF(ox, oy), QPointF(hx2, oy))
        painter.drawLine(QPointF(hx2, oy), QPointF(hx2 - 5, oy - 4))
        painter.drawLine(QPointF(hx2, oy), QPointF(hx2 - 5, oy + 4))
        painter.setPen(h_color)
        painter.drawText(QPointF(hx2 + 4, oy + 4), self._axes_h)

        # Vertical arrow pointing down from the same origin.
        vy2 = oy + v_len
        pen.setColor(v_color)
        painter.setPen(pen)
        painter.drawLine(QPointF(ox, oy), QPointF(ox, vy2))
        painter.drawLine(QPointF(ox, vy2), QPointF(ox - 4, vy2 - 5))
        painter.drawLine(QPointF(ox, vy2), QPointF(ox + 4, vy2 - 5))
        painter.setPen(v_color)
        painter.drawText(QPointF(ox + 5, vy2 + 2), self._axes_v)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self.middle_clicked.emit()
        elif event.button() == Qt.LeftButton:
            dx, dy = self._screen_to_data(event.pos().x(), event.pos().y())
            self.clicked.emit(dx, dy)
            self._drag_last = event.pos()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_last is not None:
            dpx = event.pos().x() - self._drag_last.x()
            dpy = event.pos().y() - self._drag_last.y()
            self._drag_last = event.pos()
            self.panned.emit(dpx, dpy)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_last = None

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 1 / 1.1
        self.zoomed.emit(factor)


class _ControlBar(QWidget):
    """Right-side control bar: position sliders, movie, zoom."""

    position_changed = Signal(int, int, int)
    movie_toggled = Signal(bool)
    zoom_changed = Signal(float)
    reset_view_requested = Signal()
    middle_clicked = Signal()

    def __init__(self, nx: int, ny: int, nz: int, parent=None):
        super().__init__(parent)
        colors = _gallery_theme_colors()
        self.setMinimumWidth(170)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(colors["panel"]))
        self.setPalette(pal)
        self.setStyleSheet(_gallery_qss(colors))

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        lbl = f"QLabel {{ color: {colors['text']}; font-size: 11px; }}"
        sld = (
            f"QSlider::groove:horizontal {{ background: {colors['border']}; height: 4px; }}"
            f"QSlider::handle:horizontal {{ background: {colors['text']}; width: 12px; "
            "margin: -4px 0; border-radius: 3px; }"
        )
        val_lbl = f"QLabel {{ color: {colors['text']}; font-size: 10px; }}"
        btn = (
            f"QPushButton {{ color: {colors['text']}; background: {colors['input']}; "
            f"border: 1px solid {colors['border']}; "
            "border-radius: 3px; padding: 4px; font-size: 11px; }"
            f"QPushButton:hover {{ background: {colors['accent']}; }}"
        )

        def _label(text):
            l = QLabel(text)
            l.setStyleSheet(lbl)
            return l

        def _slider(rng):
            s = QSlider(Qt.Horizontal)
            s.setRange(0, max(0, rng - 1))
            s.setStyleSheet(sld)
            return s

        def _spinbox(rng, val):
            sp = QSpinBox()
            sp.setRange(0, max(0, rng - 1))
            sp.setValue(val)
            # Compact numeric entry; the slider handles stepped scrolling.
            sp.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            sp.setAlignment(Qt.AlignCenter)
            sp.setFixedWidth(54)
            return sp

        def _make_axis_row(label_text, rng, val):
            sl = _slider(rng)
            sp = _spinbox(rng, val)
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addWidget(_label(label_text))
            row.addWidget(sl, 1)
            row.addWidget(sp)
            return sl, sp, row

        self._x_slider, self._x_spin, x_row = _make_axis_row("X", nx, 0)
        root.addLayout(x_row)

        self._y_slider, self._y_spin, y_row = _make_axis_row("Y", ny, 0)
        root.addLayout(y_row)

        self._z_slider, self._z_spin, z_row = _make_axis_row("Z", nz, 0)
        root.addLayout(z_row)

        self._zoom_slider = QSlider(Qt.Horizontal)
        self._zoom_slider.setRange(0, 1000)
        self._zoom_slider.setValue(500)
        self._zoom_slider.setStyleSheet(sld)
        self._zoom_val = _label("1.00x")
        self._zoom_val.setStyleSheet(val_lbl)
        row_zm = QHBoxLayout()
        row_zm.setSpacing(4)
        row_zm.addWidget(_label("Zoom"))
        row_zm.addWidget(self._zoom_slider, 1)
        row_zm.addWidget(self._zoom_val)
        root.addLayout(row_zm)

        row_btns = QHBoxLayout()
        self._movie_btn = QPushButton("Movie")
        self._movie_btn.setCheckable(True)
        self._movie_btn.setStyleSheet(btn)
        # Auto-default-off: otherwise Enter in a sibling spinbox toggles the
        # movie button (QPushButton auto-defaults inside a QDialog parent).
        self._movie_btn.setAutoDefault(False)
        self._movie_btn.clicked.connect(self.movie_toggled)
        row_btns.addWidget(self._movie_btn)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setStyleSheet(btn)
        self._reset_btn.setAutoDefault(False)
        self._reset_btn.clicked.connect(self.reset_view_requested)
        row_btns.addWidget(self._reset_btn)
        root.addLayout(row_btns)

        root.addStretch(1)

        # Slider<->spinbox mirror: each side blocks the other's signals
        # while updating so one user action emits position_changed once.
        self._sliders = (self._x_slider, self._y_slider, self._z_slider)
        self._spins = (self._x_spin, self._y_spin, self._z_spin)
        for axis, slider in enumerate(self._sliders):
            slider.valueChanged.connect(lambda v, a=axis: self._on_slider_changed(a, v))
        for axis, spin in enumerate(self._spins):
            spin.valueChanged.connect(lambda v, a=axis: self._on_spin_changed(a, v))
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider)

    def _apply_display_theme(self) -> None:
        """Apply the current theme to the orthogonal-view controls."""
        colors = _gallery_theme_colors()
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(colors["panel"]))
        self.setPalette(pal)
        self.setStyleSheet(_gallery_qss(colors))
        for child in self.findChildren(QWidget):
            child.setStyleSheet("")
        self.update()

    @staticmethod
    def _slider_to_zoom(v: int) -> float:
        return 10.0 ** (2.0 * v / 1000.0 - 1.0)

    @staticmethod
    def _zoom_to_slider(zoom: float) -> int:
        import math

        return int(round((math.log10(zoom) + 1.0) / 2.0 * 1000.0))

    def _on_zoom_slider(self, v: int) -> None:
        zoom = self._slider_to_zoom(v)
        self._zoom_val.setText(f"{zoom:.2f}x")
        self.zoom_changed.emit(zoom)

    def _on_slider_changed(self, axis: int, v: int) -> None:
        """Mirror a slider move into the matching spinbox, then emit position."""
        self._spins[axis].blockSignals(True)
        self._spins[axis].setValue(v)
        self._spins[axis].blockSignals(False)
        self.position_changed.emit(*[s.value() for s in self._sliders])

    def _on_spin_changed(self, axis: int, v: int) -> None:
        """Mirror a spinbox entry into the matching slider, then emit position."""
        self._sliders[axis].blockSignals(True)
        self._sliders[axis].setValue(v)
        self._sliders[axis].blockSignals(False)
        self.position_changed.emit(*[s.value() for s in self._sliders])

    def set_position(self, x: int, y: int, z: int) -> None:
        for slider, spin, val in [
            (self._x_slider, self._x_spin, x),
            (self._y_slider, self._y_spin, y),
            (self._z_slider, self._z_spin, z),
        ]:
            slider.blockSignals(True)
            slider.setValue(val)
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
            slider.blockSignals(False)

    def set_dimensions(self, nx: int, ny: int, nz: int) -> None:
        """Retarget the position sliders and spinboxes to a new volume size.

        Used when the displayed volume is replaced (e.g. after a transform
        that changes its dimensions). ``QSlider.setRange`` and
        ``QSpinBox.setRange`` clamp the current value into the new range, so
        the crosshair position stays where it was when the new volume still
        contains that location.

        Parameters
        ----------
        nx : int
            New X dimension.
        ny : int
            New Y dimension.
        nz : int
            New Z dimension.
        """
        for slider, spin, rng in [
            (self._x_slider, self._x_spin, nx),
            (self._y_slider, self._y_spin, ny),
            (self._z_slider, self._z_spin, nz),
        ]:
            slider.blockSignals(True)
            slider.setRange(0, max(0, rng - 1))
            spin.blockSignals(True)
            spin.setRange(0, max(0, rng - 1))
            spin.blockSignals(False)
            slider.blockSignals(False)

    def set_zoom(self, zoom: float) -> None:
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(self._zoom_to_slider(zoom))
        self._zoom_val.setText(f"{zoom:.2f}x")
        self._zoom_slider.blockSignals(False)

    def set_movie_playing(self, playing: bool) -> None:
        self._movie_btn.blockSignals(True)
        self._movie_btn.setChecked(playing)
        self._movie_btn.blockSignals(False)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self.middle_clicked.emit()
        else:
            super().mousePressEvent(event)


class _BCGPanel(QWidget):
    """Collapsible left-side panel with brightness / contrast / gamma controls."""

    bcg_changed = Signal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        colors = _gallery_theme_colors()
        self.setFixedWidth(180)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(colors["panel"]))
        self.setPalette(pal)
        self.setStyleSheet(_gallery_qss(colors))

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        lbl = f"QLabel {{ color: {colors['text']}; font-size: 11px; }}"
        sld = (
            f"QSlider::groove:horizontal {{ background: {colors['border']}; height: 4px; }}"
            f"QSlider::handle:horizontal {{ background: {colors['text']}; width: 12px; "
            "margin: -4px 0; border-radius: 3px; }"
        )
        val_lbl = f"QLabel {{ color: {colors['text']}; font-size: 10px; }}"

        def _label(text):
            return QLabel(text)

        def _slider(rng):
            s = QSlider(Qt.Horizontal)
            s.setRange(0, max(0, rng - 1))
            s.setStyleSheet(sld)
            return s

        root.addWidget(_label("Brightness"))
        self._b_slider = _slider(200)
        self._b_slider.setValue(100)
        self._b_val = _label("0.00")
        self._b_val.setStyleSheet(val_lbl)
        row_b = QHBoxLayout()
        row_b.addWidget(self._b_slider, 1)
        row_b.addWidget(self._b_val)
        root.addLayout(row_b)

        root.addWidget(_label("Contrast"))
        self._c_slider = _slider(300)
        self._c_slider.setValue(100)
        self._c_val = _label("1.00")
        self._c_val.setStyleSheet(val_lbl)
        row_c = QHBoxLayout()
        row_c.addWidget(self._c_slider, 1)
        row_c.addWidget(self._c_val)
        root.addLayout(row_c)

        root.addWidget(_label("Gamma"))
        self._g_slider = _slider(300)
        self._g_slider.setValue(100)
        self._g_val = _label("1.00")
        self._g_val.setStyleSheet(val_lbl)
        row_g = QHBoxLayout()
        row_g.addWidget(self._g_slider, 1)
        row_g.addWidget(self._g_val)
        root.addLayout(row_g)

        root.addStretch(1)

        self._b_slider.valueChanged.connect(self._emit)
        self._c_slider.valueChanged.connect(self._emit)
        self._g_slider.valueChanged.connect(self._emit)

    def _apply_display_theme(self) -> None:
        """Apply the current theme to the BCG controls."""
        colors = _gallery_theme_colors()
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(colors["panel"]))
        self.setPalette(pal)
        self.setStyleSheet(_gallery_qss(colors))
        for child in self.findChildren(QWidget):
            child.setStyleSheet("")
        self.update()

    def _emit(self) -> None:
        self.bcg_changed.emit(*self.get_bcg())

    def get_bcg(self) -> tuple[float, float, float]:
        return (
            self._b_slider.value() / 100.0 - 1.0,
            self._c_slider.value() / 100.0,
            self._g_slider.value() / 100.0,
        )


class OrthogonalViewerWidget(QWidget):
    """Interactive three-panel orthogonal slice viewer for 3D volumes.

    Displays Z, X, and Y slices in that order. The Z panel has X horizontal
    and Y vertical axes; the X panel has Z horizontal and Y vertical axes;
    and the Y panel has X horizontal and Z vertical axes.

    Parameters
    ----------
    volume : np.ndarray
        3D array with shape ``(nz, ny, nx)``.
    apix : float
        Pixel size in Angstroms.
    name : str
        Display name for the window title.
    """

    panel_toggle_requested = Signal()
    view_changed = Signal()

    def __init__(self, volume: np.ndarray, apix: float = 1.0, name: str = ""):
        super().__init__()
        self._volume = volume
        self._apix = apix
        self._name = name
        nz, ny, nx = volume.shape
        self._nx, self._ny, self._nz = nx, ny, nz
        self._pos = [nx // 2, ny // 2, nz // 2]
        self._movie_timer = QTimer(self)
        self._movie_timer.timeout.connect(self._advance_movie)
        self._movie_axis = 2

        self._brightness = 0.0
        self._contrast = 1.0
        self._gamma = 1.0
        self._log_transform = False
        self._adjust_scope = "all"
        self._selected_panel_idx = -1

        self._setup_ui()
        self._sync_views(source_idx=-1)

    def set_volume(
        self,
        volume: np.ndarray,
        apix: float | None = None,
        reset_position: bool = False,
    ) -> None:
        """Replace the displayed volume, keeping the current view state.

        Used by host panels that transform a volume in place (e.g. the
        Proc3D tools panel's result preview). The three slice panels keep
        their zoom and pan, the crosshair position is clamped into the new
        dimensions, and the control-bar sliders are retargeted to the new
        size. ``apix`` is only updated when explicitly passed.

        Parameters
        ----------
        volume : np.ndarray
            Replacement 3D array with shape ``(nz, ny, nx)``.
        apix : float, optional
            New pixel size in Angstroms; keeps the current value when None.
        reset_position : bool, optional
            When True, the crosshair position is reset to the volume center
            ``(nx//2, ny//2, nz//2)`` instead of being clamped into the new
            dimensions. Use this when the previous position is meaningless
            (e.g. replacing a placeholder volume on first load). Defaults to
            False.
        """
        self._volume = volume
        if min(volume.shape) == 0:
            raise ValueError(
                f"invalid volume shape {volume.shape}: all dimensions must be > 0"
            )
        if apix is not None:
            self._apix = apix
        nz, ny, nx = volume.shape
        self._nx, self._ny, self._nz = nx, ny, nz
        if reset_position:
            self._pos = [nx // 2, ny // 2, nz // 2]
        else:
            for i, dim in enumerate((self._nx, self._ny, self._nz)):
                self._pos[i] = int(np.clip(self._pos[i], 0, dim - 1))
        self._ctrl.set_dimensions(nx, ny, nz)
        self._sync_views(source_idx=-1)

    def _apply_display_theme(self) -> None:
        """Apply the current theme to all orthogonal-view components."""
        colors = _gallery_theme_colors()
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(colors["panel"]))
        self.setPalette(palette)
        for view in (self._xy_view, self._xz_view, self._yz_view, self._ctrl):
            view._apply_display_theme()
        self.update()

    def _setup_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._xy_view = _SliceView(axis_label="Z")
        self._xz_view = _SliceView(axis_label="X")
        self._yz_view = _SliceView(axis_label="Y")

        self._xy_view.set_border_color(QColor(0, 200, 0))
        self._xz_view.set_border_color(QColor(200, 0, 0))
        self._yz_view.set_border_color(QColor(0, 100, 255))

        # Per-panel in-plane axes: Z panel → x horizontal / y vertical,
        # X panel → z horizontal / y vertical, Y panel → x horizontal / z vertical.
        self._xy_view.set_axes("x", "y")
        self._xz_view.set_axes("z", "y")
        self._yz_view.set_axes("x", "z")

        self._ctrl = _ControlBar(self._nx, self._ny, self._nz)

        # Arrange the three orthogonal panels in Z, X, Y order.
        layout.addWidget(self._xy_view, 0, 0)
        layout.addWidget(self._xz_view, 0, 1)
        layout.addWidget(self._yz_view, 1, 0)
        layout.addWidget(self._ctrl, 1, 1)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)

        self._xy_view.clicked.connect(lambda x, y: self._on_click(0, x, y))
        self._xz_view.clicked.connect(lambda x, y: self._on_click(1, x, y))
        self._yz_view.clicked.connect(lambda x, y: self._on_click(2, x, y))

        self._xy_view.panned.connect(lambda dx, dy: self._on_pan(0, dx, dy))
        self._xz_view.panned.connect(lambda dx, dy: self._on_pan(1, dx, dy))
        self._yz_view.panned.connect(lambda dx, dy: self._on_pan(2, dx, dy))

        self._xy_view.zoomed.connect(lambda f: self._on_zoom(f))
        self._xz_view.zoomed.connect(lambda f: self._on_zoom(f))
        self._yz_view.zoomed.connect(lambda f: self._on_zoom(f))

        self._xy_view.middle_clicked.connect(self.panel_toggle_requested.emit)
        self._xz_view.middle_clicked.connect(self.panel_toggle_requested.emit)
        self._yz_view.middle_clicked.connect(self.panel_toggle_requested.emit)
        self._ctrl.middle_clicked.connect(self.panel_toggle_requested.emit)

        self._ctrl.position_changed.connect(self._on_slider_position)
        self._ctrl.movie_toggled.connect(self._on_movie_toggle)
        self._ctrl.zoom_changed.connect(self._on_ctrl_zoom)
        self._ctrl.reset_view_requested.connect(self._on_reset_view)

    def _get_slice(self, axis: int, idx: int) -> np.ndarray:
        """axis 0=Z (XY plane), 1=Y (XZ plane), 2=X (YZ plane)."""
        if axis == 0:
            return self._volume[idx]
        elif axis == 1:
            return self._volume[:, idx, :]
        else:
            return self._volume[:, :, idx]

    def _sync_views(self, source_idx: int = -1) -> None:
        x, y, z = self._pos

        # Z panel: X horizontal, Y vertical.
        self._xy_view.set_image(self._get_slice(0, z))
        self._xy_view.set_crosshair(x, y, _COLOR_X, _COLOR_Y)

        # X panel: Z horizontal, Y vertical. The native (Z, Y) slice is
        # transposed so that its columns are Z and rows are Y.
        self._xz_view.set_image(self._get_slice(2, x).T)
        self._xz_view.set_crosshair(z, y, _COLOR_Z, _COLOR_Y)

        # Y panel: X horizontal, Z vertical.
        self._yz_view.set_image(self._get_slice(1, y))
        self._yz_view.set_crosshair(x, z, _COLOR_X, _COLOR_Z)

        self._ctrl.set_position(x, y, z)
        self.view_changed.emit()

    _MOVIE_AXES = [2, 0, 1]

    def _on_click(self, panel_idx: int, dx: float, dy: float) -> None:
        self._selected_panel_idx = panel_idx
        self._movie_axis = self._MOVIE_AXES[panel_idx]
        if panel_idx == 0:
            self._pos[0] = int(np.clip(round(dx), 0, self._nx - 1))
            self._pos[1] = int(np.clip(round(dy), 0, self._ny - 1))
        elif panel_idx == 1:
            self._pos[2] = int(np.clip(round(dx), 0, self._nz - 1))
            self._pos[1] = int(np.clip(round(dy), 0, self._ny - 1))
        else:
            self._pos[0] = int(np.clip(round(dx), 0, self._nx - 1))
            self._pos[2] = int(np.clip(round(dy), 0, self._nz - 1))
        self._sync_views(source_idx=panel_idx)

    def _on_pan(self, panel_idx: int, dpx: int, dpy: int) -> None:
        """Linked panning: dragging in one view shifts the other two to keep
        corresponding regions visible, matching IMOD's XYZ window behavior."""
        views = [self._xy_view, self._xz_view, self._yz_view]
        view = views[panel_idx]
        if view._image is None:
            return
        ih, iw = view._image.shape[:2]
        w, h = view.width(), view.height()
        scale = min(w / max(iw, 1), h / max(ih, 1)) * view._zoom
        if scale < 1e-9:
            return

        view._pan_x += dpx
        view._pan_y += dpy

        if panel_idx == 0:
            self._xz_view._pan_x += dpx
            self._yz_view._pan_y += dpy
        elif panel_idx == 1:
            self._xy_view._pan_x += dpx
            self._yz_view._pan_y += dpy
        else:
            self._xy_view._pan_y += dpx
            self._xz_view._pan_y += dpy

        for v in views:
            v.update()

    def _on_zoom(self, factor: float) -> None:
        for v in [self._xy_view, self._xz_view, self._yz_view]:
            v.set_zoom(v._zoom * factor)
        self._ctrl.set_zoom(self._xy_view._zoom)

    def _on_ctrl_zoom(self, zoom: float) -> None:
        for v in [self._xy_view, self._xz_view, self._yz_view]:
            v.set_zoom(zoom)

    # ---- adapter methods for _wrap_gallery_with_panel compatibility ----

    def has_data(self) -> bool:
        return self._volume is not None

    @property
    def _read_fn(self):
        def _sample(i: int):
            if self._adjust_scope == "selected" and self._selected_panel_idx >= 0:
                idx = self._selected_panel_idx
            else:
                idx = i % 3
            views = [self._xy_view, self._xz_view, self._yz_view]
            return (
                views[idx]._image
                if views[idx]._image is not None
                else np.zeros((1, 1), dtype=np.float32)
            )

        return _sample

    @property
    def _n(self):
        if self._adjust_scope == "selected" and self._selected_panel_idx >= 0:
            return 3
        return 9

    def _propagate_bcg(self):
        views = [self._xy_view, self._xz_view, self._yz_view]
        if self._adjust_scope == "selected" and self._selected_panel_idx >= 0:
            targets = [views[self._selected_panel_idx]]
        else:
            targets = views
        for v in targets:
            v._brightness = self._brightness
            v._contrast = self._contrast
            v._gamma = self._gamma
            v._log_transform = self._log_transform
            v.update()

    def set_brightness(self, val: float) -> None:
        self._brightness = val
        self._propagate_bcg()

    def set_contrast(self, val: float) -> None:
        self._contrast = val
        self._propagate_bcg()

    def set_gamma(self, val: float) -> None:
        self._gamma = val
        self._propagate_bcg()

    def set_log_transform(self, val: bool) -> None:
        self._log_transform = val
        self._propagate_bcg()

    def reset_adjustments(self) -> None:
        self._brightness = 0.0
        self._contrast = 1.0
        self._gamma = 1.0
        self._log_transform = False
        self._propagate_bcg()

    def set_adjust_scope(self, scope: str) -> None:
        self._adjust_scope = scope

    def set_selected_idx(self, idx) -> None:
        pass

    def _on_slider_position(self, x: int, y: int, z: int) -> None:
        self._pos = [x, y, z]
        self._sync_views(source_idx=-1)

    def _on_movie_toggle(self, playing: bool) -> None:
        if playing:
            self._movie_timer.start(100)
        else:
            self._movie_timer.stop()

    def _on_reset_view(self) -> None:
        self._pos = [self._nx // 2, self._ny // 2, self._nz // 2]
        for v in [self._xy_view, self._xz_view, self._yz_view]:
            v._zoom = 1.0
            v._pan_x = 0.0
            v._pan_y = 0.0
        self._ctrl.set_zoom(1.0)
        self._sync_views(source_idx=-1)

    def _advance_movie(self) -> None:
        axis = self._movie_axis
        sizes = [self._nx, self._ny, self._nz]
        self._pos[axis] = (self._pos[axis] + 1) % sizes[axis]
        self._sync_views(source_idx=-1)
