"""
This module provides :class:`ImageGalleryWidget`, a Qt widget that displays a
grid of image thumbnails for a stack (e.g. a ``.mrcs`` particle stack or a
data ``.star`` file). Only the images currently visible in the viewport are
read from the (lazy) data source.

The viewport-culling math is kept framework-agnostic in :class:`GalleryPanel`
so it can be unit-tested without a Qt application.
"""

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

import math

import numpy as np


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        # Fill the host dock in both directions so the grid centers in the
        # dock's actual (not preferred) width; otherwise the dock leaves the
        # widget at its preferred size and left-aligned within the dock.
        from PySide6.QtWidgets import QSizePolicy

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setContentsMargins(0, 0, 0, 0)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor("#2d2d2d"))
        self.setPalette(palette)

        self._read_fn = None
        self._n = 0
        self._img_w = 0
        self._img_h = 0
        self._dtype = None
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

    # ------------------------------------------------------------------ data
    def set_data(self, read_fn, n: int, img_w: int, img_h: int, dtype) -> None:
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
        """
        self._read_fn = read_fn
        self._n = int(n)
        self._img_w = int(img_w)
        self._img_h = int(img_h)
        self._dtype = dtype
        self._panel = GalleryPanel(self._n, self._img_w, self._img_h)
        self._coords = {}
        self._thumb_cache = {}
        self._scroll_y = 0

        self.update()

        self.update()

    def has_data(self) -> bool:
        """Return ``True`` if a data source is bound with at least one image."""
        return self._read_fn is not None and self._n > 0

    # -------------------------------------------------------------- rendering
    def _to_thumb(self, frame: np.ndarray) -> QPixmap:
        """Normalise a frame to an 8-bit grayscale ``QPixmap`` thumbnail.

        Parameters
        ----------
        frame : numpy.ndarray
            2D image data.

        Returns
        -------
        QPixmap
            Grayscale thumbnail.
        """
        from helicon.commands.display import _auto_contrast

        black, white = _auto_contrast(frame)
        arr = frame.astype(np.float64)
        arr = np.clip((arr - black) / max(1e-9, white - black), 0.0, 1.0)
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
        """Return ``True`` if the stack overflows the viewport vertically.

        A scrollbar should only be shown (and its width reserved) when there
        is actually something to scroll, so the gallery stays centered in the
        full width instead of leaving dead space on the right.
        """
        panel = GalleryPanel(self._n, self._img_w, self._img_h, self._panel.min_sep)
        panel.visible_row_col(
            self._canvas_width(with_sb=False),
            self.height(),
            self._scale,
            0,
            self._scroll_y,
        )
        return panel.max_y < 0

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
        """Zoom by ``factor`` about the canvas point ``(cx, cy)``.

        The grid is scaled about the cursor so the image content currently
        under the pointer stays put (zoom-to-cursor), instead of always
        growing from the top-left. ``factor > 1`` zooms in; ``< 1`` zooms out.

        Parameters
        ----------
        factor : float
            Multiplicative zoom factor applied to the current scale.
        cx, cy : int
            Cursor position, in widget pixels, to keep stationary.
        """
        if self._img_w <= 0:
            return
        new_scale = self._scale * factor
        # Cap so at least one column fits the canvas, and never absurd.
        available = self._canvas_width() - self._panel.min_sep
        max_fit = available / max(1, self._img_w)
        new_scale = min(max(new_scale, 1e-3), max(max_fit, 1e-3), 20.0)
        if new_scale == self._scale:
            return

        # Fractional row under the cursor before the zoom.
        old_h = int(round(self._img_h * self._scale)) + self._panel.min_sep
        frac_row = (cy - self._scroll_y) / old_h if old_h > 0 else 0.0

        self._scale = new_scale
        self._clamp_scroll()

        # Re-anchor scroll_y so the same fractional row sits under the cursor.
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
        # Thumb height proportional to visible/total content.
        total_rows = int(np.ceil(self._n / max(1, self._panel.visiblecols)))
        visible_rows = max(1, self._panel.visiblerows)
        frac = min(1.0, visible_rows / total_rows)
        thumb_h = max(20, int(track.height() * frac))
        # Position from current scroll fraction.
        frac_scrolled = (-self._scroll_y) / (-max_scroll) if max_scroll < 0 else 0.0
        frac_scrolled = min(1.0, max(0.0, frac_scrolled))
        y = int(track.y() + frac_scrolled * (track.height() - thumb_h))
        return QRect(track.x() + 2, y, self._sb_width - 4, thumb_h)

    def paintEvent(self, event) -> None:
        """Draw only the visible tiles, reading each lazily on first paint."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#2d2d2d"))

        if not self.has_data():
            return

        size = self.size()
        # Only reserve the scrollbar width (and draw it) when the stack
        # actually overflows vertically, so the grid centers in the full
        # width instead of drifting left with dead space on the right.
        needs_sb = self._needs_scrollbar()
        canvas_w = self._canvas_width(with_sb=needs_sb)
        res = self._panel.visible_row_col(
            canvas_w, size.height(), self._scale, 0, self._scroll_y
        )
        if res is None:
            return
        rowstart, visiblerows, visiblecols = res

        # Clamp the authoritative scroll offset once, using the layout just
        # computed. Mid-scroll this is a no-op, so the grid never jumps.
        self._scroll_y = self._panel.clamp_scroll_y(self._scroll_y)

        # Center the column block horizontally within the canvas.
        block_w = visiblecols * self._panel.rendered_w
        x_start = max(0, (canvas_w - block_w) // 2)

        # Vertical anchor: ty(r) = r * rendered_h + scroll_y (continuous in
        # scroll_y, so the grid scrolls smoothly). rowstart is the first row
        # whose top may be >= scroll_y; draw one extra row so the bottom of
        # the viewport is always filled when the top row is partially above.
        # Index labels: small, legible in the top-left of every thumbnail.
        label_font = painter.font()
        label_font.setPixelSize(max(9, min(13, int(self._panel.rendered_h * 0.28))))
        painter.setFont(label_font)
        label_color = QColor("#e8e8e8")
        label_bg = QColor(0, 0, 0, 140)

        for r in range(rowstart, rowstart + visiblerows + 1):
            # Items actually present in this row (the last row may be partial).
            # Center each row independently so a short trailing row does not
            # hang off to the left of the grid.
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
                    self._thumb_cache[i] = self._to_thumb(self._read_fn(i))
                thumb = self._thumb_cache[i]
                painter.drawPixmap(tx, ty, draw_w, draw_h, thumb)
                self._coords[i] = QRect(tx, ty, draw_w, draw_h)

                # Label text sized to the rendered thumbnail.
                text = str(i)
                trect = QRect(tx, ty, thumb.width(), thumb.height())
                metrics = painter.fontMetrics()
                tw = metrics.horizontalAdvance(text)
                th = metrics.height()
                pad = 2
                # Clip the label background to the thumbnail so it never spills
                # into neighbouring tiles.
                painter.save()
                painter.setClipRect(trect)
                painter.fillRect(
                    tx, ty, min(tw + 2 * pad, trect.width()), th + 2 * pad, label_bg
                )
                painter.setPen(label_color)
                painter.drawText(tx + pad, ty + pad + metrics.ascent(), text)
                painter.restore()

        # Vertical scrollbar (only when the stack overflows the viewport).
        if needs_sb:
            track = self._scrollbar_rect()
            painter.fillRect(track, QColor("#1f1f1f"))
            thumb_rect = self._scrollbar_thumb_rect()
            if thumb_rect is not None:
                painter.fillRect(thumb_rect, QColor("#5a5a5a"))

    # ------------------------------------------------------------- interaction
    def wheelEvent(self, event) -> None:
        """Scroll vertically (or zoom when the Ctrl modifier is held)."""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 1.0 / 1.1
            new_scale = self._scale * factor
            # Cap scale so at least one column fits the canvas width.
            if self._img_w > 0:
                available = self._canvas_width() - self._panel.min_sep
                max_fit = available / max(1, self._img_w)
                new_scale = min(new_scale, max(max_fit, 1e-3))
            new_scale = min(new_scale, 20.0)
            self._scale = new_scale
            self.update()
            event.accept()
            return

        # Plain wheel: scroll vertically through the stack. Wheel-down (delta
        # < 0) advances to later images (more negative scroll_y).
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
        menu.addAction("Save Viewport As…", self._save_as)
        menu.exec(self.mapToGlobal(pos))

    def _save_as(self):
        qimg = self.grab()
        from helicon.commands.display import _save_qimage

        _save_qimage(qimg, self)

    def mousePressEvent(self, event) -> None:
        """Begin vertical drag, cursor zoom, or scrollbar scrub."""
        if event.button() == Qt.LeftButton:
            self._drag_last = event.pos()
            self._dragged = False
            self._scrubbing = self._hit_scrollbar(event.pos())
        elif event.button() == Qt.RightButton:
            # Right-drag zooms about the press point (no keyboard needed).
            self._drag_last = event.pos()
            self._dragged = False
            self._zooming = True

    def mouseMoveEvent(self, event) -> None:
        """Pan, zoom-to-cursor, or scrub the scrollbar thumb on drag."""
        if self._drag_last is None or not event.buttons():
            return
        dy = event.y() - self._drag_last.y()

        if self._zooming:
            # Drag up zooms in, down zooms out; exponential for smoothness.
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
                    self.image_activated.emit(i)
                    break
        if event.button() == Qt.RightButton and not self._dragged:
            self._show_save_menu(event.pos())
        self._drag_last = None
        self._scrubbing = False
        self._zooming = False
