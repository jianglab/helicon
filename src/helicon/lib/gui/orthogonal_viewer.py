"""The orthogonal slice viewer: three views of a 3D map and their controls."""

from __future__ import annotations

from PySide6.QtCore import QTimer, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import math

import numpy as np

from helicon.lib.gui.gallery_style import (
    _button_icon,
    _gallery_qss,
    _gallery_theme_colors,
)


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
    hovered(float, float)
        Emitted with data-space coordinates while the cursor is over the image.
    panned(int, int)
        Emitted with pixel deltas when the user drags.
    zoomed(float)
        Emitted with zoom factor when the user scrolls.
    """

    clicked = Signal(float, float)
    hovered = Signal(float, float)
    panned = Signal(int, int)
    zoomed = Signal(float)
    middle_clicked = Signal()

    def __init__(self, parent=None, axis_label: str = ""):
        super().__init__(parent)
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
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
        self._hover_text = ""
        self._hover_pos: QPointF | None = None

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
        self.clear_hover_tip()
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
        self.clear_hover_tip()
        self.update()

    def set_pan(self, x: float, y: float) -> None:
        self._pan_x = x
        self._pan_y = y
        self.clear_hover_tip()
        self.update()

    def set_hover_tip(self, text: str) -> None:
        """Set the text painted next to the current hover position."""
        self._hover_text = text
        self.update()

    def clear_hover_tip(self) -> None:
        """Hide any voxel tooltip currently painted over the slice."""
        if self._hover_text or self._hover_pos is not None:
            self._hover_text = ""
            self._hover_pos = None
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

    def _data_position_at(self, sx: float, sy: float) -> tuple[float, float] | None:
        """Return the data position under a screen point, excluding padding."""
        if self._image is None or self._image.size == 0:
            return None
        dx, dy = self._screen_to_data(sx, sy)
        ih, iw = self._image.shape[:2]
        if 0.0 <= dx < iw and 0.0 <= dy < ih:
            return dx, dy
        return None

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
        self._draw_hover_tip(painter)

        painter.end()

    def _draw_hover_tip(self, painter: QPainter) -> None:
        """Paint a themed voxel readout near, but not under, the cursor."""
        if not self._hover_text or self._hover_pos is None:
            return

        colors = _gallery_theme_colors()
        painter.save()
        painter.setFont(QFont("Arial", 9))
        metrics = painter.fontMetrics()
        lines = self._hover_text.splitlines()
        padding = 5.0
        offset = 12.0
        margin = 3.0
        text_width = max(metrics.horizontalAdvance(line) for line in lines)
        box_width = text_width + padding * 2
        box_height = metrics.height() * len(lines) + padding * 2

        x = self._hover_pos.x() + offset
        y = self._hover_pos.y() + offset
        if x + box_width > self.width() - margin:
            x = self._hover_pos.x() - offset - box_width
        if y + box_height > self.height() - margin:
            y = self._hover_pos.y() - offset - box_height
        x = max(margin, min(x, max(margin, self.width() - box_width - margin)))
        y = max(margin, min(y, max(margin, self.height() - box_height - margin)))

        box = QRectF(x, y, box_width, box_height)
        painter.setPen(QPen(QColor(colors["border"]), 1))
        painter.setBrush(QColor(colors["label_background"]))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(QColor(colors["text"]))
        baseline = y + padding + metrics.ascent()
        for line in lines:
            painter.drawText(QPointF(x + padding, baseline), line)
            baseline += metrics.height()
        painter.restore()

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
        self.clear_hover_tip()
        if event.button() == Qt.MiddleButton:
            self.middle_clicked.emit()
        elif event.button() == Qt.LeftButton:
            dx, dy = self._screen_to_data(event.pos().x(), event.pos().y())
            self.clicked.emit(dx, dy)
            self._drag_last = event.pos()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_last is not None:
            self.clear_hover_tip()
            dpx = event.pos().x() - self._drag_last.x()
            dpy = event.pos().y() - self._drag_last.y()
            self._drag_last = event.pos()
            self.panned.emit(dpx, dpy)
            return

        pos = event.position()
        data_pos = self._data_position_at(pos.x(), pos.y())
        if data_pos is None:
            self.clear_hover_tip()
            return
        self._hover_pos = QPointF(pos)
        self.hovered.emit(*data_pos)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_last = None

    def wheelEvent(self, event) -> None:
        self.clear_hover_tip()
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 1 / 1.1
        self.zoomed.emit(factor)

    def leaveEvent(self, event) -> None:
        self.clear_hover_tip()
        super().leaveEvent(event)


class _ControlBar(QWidget):
    """Right-side control bar: position sliders, movie, zoom."""

    position_changed = Signal(int, int, int)
    movie_toggled = Signal(bool)
    zoom_changed = Signal(float)
    reset_view_requested = Signal()
    middle_clicked = Signal()

    def __init__(self, nx: int, ny: int, nz: int, apix: float = 1.0, parent=None):
        super().__init__(parent)
        colors = _gallery_theme_colors()
        self.setMinimumWidth(170)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(colors["panel"]))
        self.setPalette(pal)
        self.setStyleSheet(_gallery_qss(colors))
        # Pixel size and dimensions back the world-coordinate readout.
        self._apix = float(apix)
        self._nx, self._ny, self._nz = nx, ny, nz

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # styled by the stylesheet of the bar (_gallery_qss), not widget by widget
        lbl = sld = val_lbl = btn = ""

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
        self._zoom_spin = QDoubleSpinBox()
        self._zoom_spin.setDecimals(2)
        self._zoom_spin.setRange(0.05, 100.0)
        self._zoom_spin.setValue(1.00)
        self._zoom_spin.setSingleStep(0.1)
        self._zoom_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._zoom_spin.setAlignment(Qt.AlignCenter)
        self._zoom_spin.setFixedWidth(54)
        self._zoom_spin.setAccelerated(True)
        row_zm = QHBoxLayout()
        row_zm.setSpacing(4)
        row_zm.addWidget(_label("Zoom"))
        row_zm.addWidget(self._zoom_slider, 1)
        row_zm.addWidget(self._zoom_spin)
        root.addLayout(row_zm)

        row_btns = QHBoxLayout()
        self._movie_btn = QPushButton("Movie")
        self._movie_btn.setCheckable(True)
        self._movie_btn.setStyleSheet(btn)
        # Auto-default-off: otherwise Enter in a sibling spinbox toggles the
        # movie button (QPushButton auto-defaults inside a QDialog parent).
        self._movie_btn.setAutoDefault(False)
        self._movie_btn.clicked.connect(self.movie_toggled)
        self._movie_btn.toggled.connect(lambda _on: self._set_icons())
        row_btns.addWidget(self._movie_btn)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setStyleSheet(btn)
        self._reset_btn.setAutoDefault(False)
        self._reset_btn.clicked.connect(self.reset_view_requested)
        row_btns.addWidget(self._reset_btn)
        root.addLayout(row_btns)
        self._set_icons()

        root.addStretch(1)

        self._voxel_label = _label("")
        self._voxel_label.setStyleSheet(val_lbl)
        self._voxel_label.setWordWrap(True)
        self._voxel_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self._voxel_label)

        # Slider<->spinbox mirror: each side blocks the other's signals
        # while updating so one user action emits position_changed once.
        self._sliders = (self._x_slider, self._y_slider, self._z_slider)
        self._spins = (self._x_spin, self._y_spin, self._z_spin)
        for axis, slider in enumerate(self._sliders):
            slider.valueChanged.connect(lambda v, a=axis: self._on_slider_changed(a, v))
        for axis, spin in enumerate(self._spins):
            spin.valueChanged.connect(lambda v, a=axis: self._on_spin_changed(a, v))
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider)
        self._zoom_spin.valueChanged.connect(self._on_zoom_spin)

    def _set_icons(self) -> None:
        """Icons on the movie and reset buttons (a pause while the movie plays)."""
        for button, name in (
            (
                self._movie_btn,
                "fa5s.pause" if self._movie_btn.isChecked() else "fa5s.play",
            ),
            (self._reset_btn, "fa5s.undo"),
        ):
            icon = _button_icon(name)
            if icon is not None:
                button.setIcon(icon)

    def _apply_display_theme(self) -> None:
        """Apply the current theme to the orthogonal-view controls."""
        colors = _gallery_theme_colors()
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(colors["panel"]))
        self.setPalette(pal)
        self.setStyleSheet(_gallery_qss(colors))
        for child in self.findChildren(QWidget):
            child.setStyleSheet("")
        self._set_icons()
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
        self._zoom_spin.blockSignals(True)
        self._zoom_spin.setValue(zoom)
        self._zoom_spin.blockSignals(False)
        self.zoom_changed.emit(zoom)

    def _on_zoom_spin(self, zoom: float) -> None:
        zoom = max(0.05, zoom)
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(self._zoom_to_slider(zoom))
        self._zoom_slider.blockSignals(False)
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
        self._nx, self._ny, self._nz = nx, ny, nz
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
        self._zoom_spin.blockSignals(True)
        self._zoom_spin.setValue(max(0.05, zoom))
        self._zoom_spin.blockSignals(False)
        self._zoom_slider.blockSignals(False)

    def set_voxel_value(self, x: int, y: int, z: int, val: float) -> None:
        """Show the voxel index, value and world coordinates in the panel.

        The world coordinate is the offset from the map center
        (nx//2, ny//2, nz//2) in Angstroms, i.e. (idx - center) * apix.
        """
        cx, cy, cz = self._nx // 2, self._ny // 2, self._nz // 2
        wx = (int(x) - cx) * self._apix
        wy = (int(y) - cy) * self._apix
        wz = (int(z) - cz) * self._apix
        self._voxel_label.setText(
            f"val={float(val):.6g} at x,y,z={int(x)},{int(y)},{int(z)} ({wx:.3g},{wy:.3g},{wz:.3g} Å | {self._apix:.4g} Å/pixel)"
        )

    def set_apix(self, apix: float) -> None:
        """Update the pixel size used for the world-coordinate readout."""
        self._apix = float(apix)

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

        sld = val_lbl = ""

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
        self._ctrl.set_apix(self._apix)
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
        self._ctrl.set_apix(self._apix)

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

        self._xy_view.hovered.connect(lambda x, y: self._on_hover(0, x, y))
        self._xz_view.hovered.connect(lambda x, y: self._on_hover(1, x, y))
        self._yz_view.hovered.connect(lambda x, y: self._on_hover(2, x, y))

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
        self._update_voxel_label()

    def _voxel_at_panel_position(
        self, panel_idx: int, dx: float, dy: float
    ) -> tuple[int, int, int, float] | None:
        """Map an in-plane cursor position to a raw volume voxel."""
        u, v = math.floor(dx), math.floor(dy)
        current_x, current_y, current_z = self._pos
        if panel_idx == 0:
            x, y, z = u, v, current_z
        elif panel_idx == 1:
            x, y, z = current_x, v, u
        elif panel_idx == 2:
            x, y, z = u, current_y, v
        else:
            return None
        if not (0 <= x < self._nx and 0 <= y < self._ny and 0 <= z < self._nz):
            return None
        return x, y, z, float(self._volume[z, y, x])

    def _format_hover_tip(self, x: int, y: int, z: int, val: float) -> str:
        """Format raw voxel, index, and center-relative world coordinates."""
        cx, cy, cz = self._nx // 2, self._ny // 2, self._nz // 2
        wx = (x - cx) * self._apix
        wy = (y - cy) * self._apix
        wz = (z - cz) * self._apix
        return (
            f"val={val:.6g}\n"
            f"x,y,z={x},{y},{z}\n"
            f"world={wx:.3g},{wy:.3g},{wz:.3g} Å"
        )

    def _on_hover(self, panel_idx: int, dx: float, dy: float) -> None:
        """Update one slice's tooltip without changing the selected voxel."""
        voxel = self._voxel_at_panel_position(panel_idx, dx, dy)
        views = (self._xy_view, self._xz_view, self._yz_view)
        if voxel is None:
            if 0 <= panel_idx < len(views):
                views[panel_idx].clear_hover_tip()
            return
        x, y, z, val = voxel
        views[panel_idx].set_hover_tip(self._format_hover_tip(x, y, z, val))

    # (horizontal, vertical) volume axis of each panel: Z panel shows (x, y),
    # X panel (z, y), Y panel (x, z) -- the same order as ``set_axes``.
    _PANEL_AXES = (("x", "y"), ("z", "y"), ("x", "z"))

    @staticmethod
    def _view_scale(view) -> float:
        """Screen pixels per voxel of ``view`` (0 if it has no image)."""
        if view._image is None:
            return 0.0
        ih, iw = view._image.shape[:2]
        w, h = view.width(), view.height()
        return min(w / max(iw, 1), h / max(ih, 1)) * view._zoom

    def _on_pan(self, panel_idx: int, dpx: int, dpy: int) -> None:
        """Linked panning: dragging in one view shifts the other two to keep
        corresponding regions visible, matching IMOD's XYZ window behavior.

        A drag moves the dragged panel's two volume axes; every other panel
        showing one of those axes pans along it by the same distance in
        voxels (converted to that panel's screen scale).
        """
        views = [self._xy_view, self._xz_view, self._yz_view]
        view = views[panel_idx]
        scale = self._view_scale(view)
        if scale < 1e-9:
            return

        view._pan_x += dpx
        view._pan_y += dpy

        # Shift in voxels along each axis the dragged panel shows.
        h_axis, v_axis = self._PANEL_AXES[panel_idx]
        shift = {h_axis: dpx / scale, v_axis: dpy / scale}
        for other_idx, other in enumerate(views):
            if other_idx == panel_idx:
                continue
            other_scale = self._view_scale(other) or scale
            oh_axis, ov_axis = self._PANEL_AXES[other_idx]
            if oh_axis in shift:
                other._pan_x += shift[oh_axis] * other_scale
            if ov_axis in shift:
                other._pan_y += shift[ov_axis] * other_scale

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

    def _update_voxel_label(self) -> None:
        """Refresh the voxel value/world-coordinate readout from self._pos."""
        if self._volume is None:
            return
        x, y, z = self._pos
        val = float(self._volume[z, y, x])
        self._ctrl.set_voxel_value(x, y, z, val)

    def _on_slider_position(self, x: int, y: int, z: int) -> None:
        self._pos = [x, y, z]
        self._sync_views(source_idx=-1)
        self._update_voxel_label()

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
        self._update_voxel_label()

    def _advance_movie(self) -> None:
        axis = self._movie_axis
        sizes = [self._nx, self._ny, self._nz]
        self._pos[axis] = (self._pos[axis] + 1) % sizes[axis]
        self._sync_views(source_idx=-1)
        self._update_voxel_label()
