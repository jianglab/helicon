#!/usr/bin/env python

"""A file browser for viewing image, map, star, bild, eps, pdf, html, and text files"""

from __future__ import annotations

import sys

# On macOS, set the process name before NSApplication is initialized
# (triggered by PySide6 import below). NSApplication caches the app name
# from getprogname() at first init; after that it can't be changed.
if sys.platform == "darwin":
    try:
        import ctypes
        import ctypes.util

        ctypes.CDLL(ctypes.util.find_library("c")).setprogname(b"helicon")
        from AppKit import NSApplication, NSBundle

        bundle = NSBundle.mainBundle()
        if bundle is not None:
            bundle.infoDictionary().setObject_forKey_("helicon", "CFBundleName")
        NSApplication.sharedApplication().setActivationPolicy_(0)
    except Exception:
        pass

import argparse
import os
from pathlib import Path

import helicon
from helicon.lib.exceptions import HeliconDependencyError

try:
    from helicon.lib.napari_widgets import FolderBrowserWidget
except ImportError:
    FolderBrowserWidget = None


def _get_qsettings():
    from PySide6.QtCore import QSettings

    return QSettings("helicon", "display")


def _is_wsl():
    """Return True if running inside Windows Subsystem for Linux."""
    import platform

    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _supports_position_restore():
    """Return True if the platform reliably supports QWidget.move().

    macOS and native Linux compositors honour move() requests.
    WSL and Windows do not (the compositor overrides positions),
    so callers should fall back to automatic placement there.
    """
    import platform

    system = platform.system()
    if system == "Darwin":
        return True
    if system == "Linux" and not _is_wsl():
        return True
    return False


def _restore_geometry(dock, viewer):
    """Restore saved window sizes and reposition after the compositor places them."""
    from PySide6.QtCore import QTimer

    def _apply(attempt=0):
        try:
            qt_win = viewer.window._qt_window
        except AttributeError:
            return
        if not qt_win.isVisible() and attempt < 10:
            QTimer.singleShot(50, lambda: _apply(attempt + 1))
            return

        settings = _get_qsettings()

        viewer_ba = settings.value("viewer_ba")
        if viewer_ba is not None:
            try:
                qt_win.restoreGeometry(viewer_ba)
            except AttributeError:
                pass

        dock_geo = _read_rect(settings, "dock")
        if (
            dock_geo is not None
            and _supports_position_restore()
            and _on_screen(*dock_geo)
        ):
            x, y, w, h = dock_geo
            dock.setGeometry(x, y, w, h)
        else:
            _position_default(dock, viewer)
            if dock_geo is not None:
                _, _, w, h = dock_geo
                # Keep the default x/y but restore the saved width/height as
                # the outer frame size (consistent with _write_rect).
                dock.setGeometry(dock.x(), dock.y(), w, h)
        dock.show()

    QTimer.singleShot(0, _apply)


def _read_rect(settings, prefix):
    """Read saved x, y, width, height from QSettings.

    Returns
    -------
    tuple[int, int, int, int] or None
        The (x, y, width, height) tuple, or None if any value is missing.
    """
    x = settings.value(f"{prefix}_x")
    y = settings.value(f"{prefix}_y")
    w = settings.value(f"{prefix}_width")
    h = settings.value(f"{prefix}_height")
    if None in (x, y, w, h):
        return None
    try:
        return int(x), int(y), int(w), int(h)
    except (TypeError, ValueError):
        return None


def _on_screen(x, y, w, h):
    """Check if a window rectangle intersects any connected screen."""
    from PySide6.QtGui import QGuiApplication

    win_center_x = x + w // 2
    win_center_y = y + h // 2
    for screen in QGuiApplication.screens():
        geo = screen.geometry()
        if geo.contains(win_center_x, win_center_y):
            return True
    return False


def _position_default(dock, viewer):
    try:
        napari_rect = viewer.window._qt_window.frameGeometry()
        dock.adjustSize()
        dock_rect = dock.frameGeometry()
        w = max(dock_rect.width(), 300)
        x = napari_rect.x() - w - 10
        y = napari_rect.y()
        h = napari_rect.height()
        dock.setGeometry(x, y, w, h)
    except AttributeError:
        pass


def _install_save_hook(dock, viewer):
    """Install an event filter to save geometry when the viewer window closes."""
    from PySide6.QtCore import QEvent, QObject

    class _CloseFilter(QObject):
        def __init__(self, dock, viewer, parent=None):
            super().__init__(parent)
            self._dock = dock
            self._viewer = viewer

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Close:
                _save_geometry(self._dock, self._viewer)
            return False

    flt = _CloseFilter(dock, viewer, parent=viewer.window._qt_window)
    viewer.window._qt_window.installEventFilter(flt)
    return flt


def _save_geometry(dock, viewer):
    settings = _get_qsettings()
    qt_win = viewer.window._qt_window

    cached_ba = getattr(viewer, "_display_only_ba", None)
    viewer_ba = cached_ba if cached_ba is not None else qt_win.saveGeometry()
    settings.setValue("viewer_ba", viewer_ba)

    _write_rect(settings, "dock", dock)

    save_cols = getattr(dock, "_save_col_widths", None)
    if callable(save_cols):
        save_cols()


def _write_rect(settings, prefix, widget):
    try:
        geo = widget.frameGeometry()
        settings.setValue(f"{prefix}_x", geo.x())
        settings.setValue(f"{prefix}_y", geo.y())
        settings.setValue(f"{prefix}_width", geo.width())
        settings.setValue(f"{prefix}_height", geo.height())
    except RuntimeError:
        pass


_MRC_EXTENSIONS = {".mrc", ".mrcs", ".map"}

# Star files that describe pipelines/optimisation rather than image data;
# opened as text, not as image/volume stacks.
_METADATA_STAR_SUFFIXES = (
    "pipeline.star",
    "optimiser.star",
    "model.star",
    "sampling.star",
    "job.star",
    "extractpick.star",
)


class _LazyStarStack:
    """Lazy array that reads individual images from star file references on demand."""

    def __init__(self, entries: list[tuple[int, str, float]], shape: tuple, dtype):
        self._entries = entries
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self._cache: dict[int, object] = {}

    def __getitem__(self, key):
        import numpy as np

        if isinstance(key, int):
            key = key % self.shape[0]
            if key not in self._cache:
                self._cache[key] = self._read(key)
            return self._cache[key]

        if isinstance(key, slice):
            indices = range(*key.indices(self.shape[0]))
            return np.stack([self._read(i) for i in indices])

        if isinstance(key, tuple):
            if len(key) == 1:
                return self[key[0]]
            return np.stack(
                [self._read(i)[key[1:]] for i in range(*key[0].indices(self.shape[0]))]
            )

        raise TypeError(f"unsupported key type: {type(key)}")

    def _read(self, idx: int):
        import mrcfile

        frame_idx_0based, mrc_path, apix = self._entries[idx]
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            data = mrc.data
            if data.ndim == 2:
                return data
            return data[frame_idx_0based]

    @property
    def nbytes(self):
        return self.shape[0] * self.shape[1] * self.shape[2] * 4


class _SliceDirectionWidget:
    """Replaces axis labels with a Z/Y/X dropdown at the right end of each slider.

    For 3D volumes displayed as 2D slices the dropdown lets the user swap
    which spatial axis (Z/Y/X) the visible slider navigates through. For a
    *true* image stack (a list of independent 2D frames — ``.mrcs`` particle
    stacks, ``_data.star`` reference stacks, multi-page PDFs) the axis
    selector is meaningless because axis 0 is a frame index, not a spatial
    axis. ``set_stack_mode(True)`` hides every combo for that case.
    """

    _combos: list = []
    # True while the active layer is an image stack (non-spatial axis 0):
    # the axis-direction combos are hidden so the Z/Y/X selector does not
    # appear for stacks where it would be nonsensical.
    _stack_mode: bool = False

    def __init__(self, viewer):
        self._viewer = viewer

    @classmethod
    def set_stack_mode(cls, is_stack: bool) -> None:
        """Show or hide all axis-direction combos.

        Called by ``_open_file`` immediately before ``viewer.add_image(...)`` so
        that newly-built sliders pick up the right state and any pre-existing
        combos (reused across files) are updated in place.
        """
        cls._stack_mode = bool(is_stack)
        alive = []
        for combo in cls._combos:
            try:
                if cls._stack_mode:
                    combo.hide()
                    combo.setFixedWidth(0)
                else:
                    combo.setFixedWidth(60)
                    combo.show()
                alive.append(combo)
            except RuntimeError:
                continue
        cls._combos = alive

    def inject(self):
        from PySide6.QtWidgets import QComboBox

        from napari._qt.widgets.qt_dims import QtDimSliderWidget

        # Start fresh: combos from a previous viewer are tied to C++ widgets
        # that no longer exist, so never carry them into a new viewer.
        _SliceDirectionWidget._combos = []

        _orig_init = QtDimSliderWidget.__init__

        def _patched_init(self_slider, parent, axis):
            _orig_init(self_slider, parent, axis)

            self_slider.axis_label.hide()
            self_slider.axis_label.setFixedWidth(0)
            self_slider.totslice_label.hide()
            self_slider.totslice_label.setFixedWidth(0)
            sep = self_slider.findChild(type(self_slider).__mro__[0])
            from PySide6.QtWidgets import QFrame

            for child in self_slider.findChildren(QFrame):
                child.hide()
                child.setFixedWidth(0)

            combo = QComboBox(self_slider)
            combo.addItems(["Z", "Y", "X"])
            combo.setFixedSize(60, 22)
            combo.setStyleSheet(
                """
                QComboBox {
                    background-color: #3c3c3c;
                    color: #cccccc;
                    border: 1px solid #555555;
                    border-radius: 3px;
                    padding: 1px 4px 1px 4px;
                    font-size: 11px;
                }
                QComboBox::drop-down {
                    border: none;
                    width: 16px;
                }
                QComboBox QAbstractItemView {
                    background-color: #3c3c3c;
                    color: #cccccc;
                    selection-background-color: #4a6fa5;
                }
            """
            )

            # Suppress the popup entirely — cycle through items on click instead
            combo.showPopup = lambda: None

            def _mousePressEvent(event):
                idx = (combo.currentIndex() + 1) % combo.count()
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
                combo.currentIndexChanged.emit(idx)

            combo.mousePressEvent = _mousePressEvent

            if _SliceDirectionWidget._stack_mode:
                combo.hide()
                combo.setFixedWidth(0)

            _SliceDirectionWidget._combos.append(combo)

            def _on_change(idx):
                alive = []
                for c in _SliceDirectionWidget._combos:
                    if c is combo:
                        alive.append(c)
                        continue
                    # A combo from a previous viewer has a destroyed C++ object
                    # and raises RuntimeError on access; skip and drop it.
                    try:
                        c.blockSignals(True)
                        c.setCurrentIndex(idx)
                        c.blockSignals(False)
                        alive.append(c)
                    except RuntimeError:
                        continue
                # Drop combos whose C++ object has been destroyed so the
                # list never accumulates stale references across viewers.
                _SliceDirectionWidget._combos = alive
                ndim = self_slider.dims.ndim
                if ndim >= 3:
                    orders = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
                    step = [int(self_slider.dims.nsteps[d]) // 2 for d in range(ndim)]
                    self_slider.dims.current_step = tuple(step)
                    self_slider.dims.order = orders[idx]
                    self_slider.qt_dims._update_slider()
                    self_slider.qt_dims._update_range()

            combo.currentIndexChanged.connect(_on_change)
            self_slider.layout().addWidget(combo)

        if not getattr(QtDimSliderWidget, "_patched", False):
            QtDimSliderWidget.__init__ = _patched_init
            QtDimSliderWidget._patched = True


def _add_welcome_shortcut(viewer):
    """Add custom shortcut to the napari welcome screen."""
    try:
        from napari._qt.widgets.qt_welcome import QtWelcomeWidget, QtShortcutLabel
    except ImportError:
        return

    def _inject():
        qt_window = viewer.window._qt_window
        welcome_widgets = qt_window.findChildren(QtWelcomeWidget)
        if not welcome_widgets:
            return

        welcome = welcome_widgets[0]
        from PySide6.QtWidgets import QFormLayout

        def find_form_layout(widget):
            for child in widget.children():
                if isinstance(child, QFormLayout):
                    return child
                result = find_form_layout(child)
                if result:
                    return result
            return None

        form_layout = find_form_layout(welcome)
        if form_layout is None:
            return

        shortcut_label = QtShortcutLabel("Mid mouse button click")
        description_label = QtShortcutLabel("Toggle left side control panel")

        form_layout.addRow(shortcut_label, description_label)

    from PySide6.QtCore import QTimer

    QTimer.singleShot(0, _inject)


def _auto_contrast(data):
    """Compute contrast limits using robust statistics.

    Black point = max(median - 3*MAD, 1st percentile)
    White point = min(median + 3*MAD, 99th percentile)

    Parameters
    ----------
    data : numpy.ndarray
        Image data.

    Returns
    -------
    tuple[float, float]
        (black_point, white_point) contrast limits.
    """
    import numpy as np

    flat = data.ravel()
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med))) * 1.4826
    p1 = float(np.percentile(flat, 1))
    p99 = float(np.percentile(flat, 99))

    black = max(med - 3 * mad, p1)
    white = min(med + 3 * mad, p99)

    if black >= white:
        black = float(flat.min())
        white = float(flat.max())
        if black == white:
            white = black + 1.0

    return black, white


def _enable_continuous_auto_contrast(layer, viewer) -> None:
    """Enable continuous auto-contrast and sync napari's UI toggle.

    napari only binds the 'continuous' button's ``toggled`` signal to
    ``layer._keep_auto_contrast`` (one direction). Setting the attribute
    programmatically therefore never lights up the button, leaving the UI
    out of sync with the actual (continuous) behavior. We set the attribute
    directly so the contrast behavior is guaranteed even without a GUI, then
    check the button as a virtual click so the 'continuous' label highlights.
    The UI sync is best-effort: skipped when no Qt controls exist (headless
    runs or mocked tests).
    """
    layer._keep_auto_contrast = True
    try:
        controls = viewer.window._qt_viewer.controls.widgets[layer]
        auto_btn = controls._contrast_limits_control.auto_scale_bar._auto_btn
        auto_btn.setChecked(True)
    except (AttributeError, KeyError):
        pass


def _save_qimage(qimage, parent=None, default_name=None):
    """Open a Save-As dialog and write a QImage to the chosen file.

    Supported formats: PNG, TIFF, PDF, SVG.  The dialog filter list
    determines the format from the chosen extension.

    Parameters
    ----------
    qimage : PySide6.QtGui.QImage
        The image to save.
    parent : QWidget, optional
        Parent widget for the dialog (centre-on-parent).
    default_name : str, optional
        Suggested base filename (without extension).  The dialog pre-fills
        it with a ``.png`` suffix, replacing any existing suffix on the
        caller's source name.
    """
    from PySide6.QtWidgets import QFileDialog
    from PySide6.QtGui import QPixmap

    if isinstance(qimage, QPixmap):
        qimage = qimage.toImage()

    if default_name and isinstance(default_name, (str, bytes)):
        stem = (
            Path(default_name).stem if "." in Path(default_name).name else default_name
        )
        suggested = stem + ".png"
    else:
        suggested = ""

    filt = "Images (*.png *.tiff *.tif);;PDF (*.pdf);;SVG (*.svg)"
    dlg = QFileDialog(parent, "Save Image")
    dlg.setAcceptMode(QFileDialog.AcceptSave)
    dlg.setNameFilter(filt)
    # selectFile() reliably pre-fills the file name on native dialogs
    # (macOS/Windows); passing it only via the static getSaveFileName()
    # "directory" argument is ignored by the native save dialog.
    if suggested:
        dlg.selectFile(suggested)
    if not dlg.exec():
        return
    selected = dlg.selectedFiles()
    if not selected:
        return
    path = selected[0]
    ext = path.rsplit(".", 1)[-1].lower()
    if ext in ("pdf", "svg"):
        _render_qimage_vector(qimage, path, ext)
    else:
        qimage.save(path)


def _render_qimage_vector(qimage, path, fmt):
    """Render a QImage to a vector file (PDF or SVG).

    Parameters
    ----------
    qimage : PySide6.QtGui.QImage
        Source image.
    path : str
        Destination file path.
    fmt : str
        ``"pdf"`` or ``"svg"``.
    """
    from PySide6.QtGui import QPainter
    from PySide6.QtCore import QMarginsF

    w, h = qimage.width(), qimage.height()
    if fmt == "pdf":
        from PySide6.QtGui import QPdfWriter, QPageSize
        from PySide6.QtCore import QSizeF

        writer = QPdfWriter(path)
        writer.setResolution(72)
        writer.setPageSize(
            QPageSize(QSizeF(w, h), QPageSize.Unit.Point),
        )
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        painter = QPainter(writer)
    else:
        from PySide6.QtSvg import QSvgGenerator
        from PySide6.QtCore import QSize, QRectF

        gen = QSvgGenerator()
        gen.setFileName(path)
        gen.setSize(QSize(w, h))
        gen.setViewBox(QRectF(0, 0, w, h))
        painter = QPainter(gen)

    from PySide6.QtCore import QRect

    painter.drawImage(QRect(0, 0, w, h), qimage)
    painter.end()


def _install_viewer_save_menu(viewer):
    """Register a right-click save menu on the napari canvas.

    napari's vispy canvas consumes mouse events at the vispy level before Qt
    event filters ever see them.  The camera's ``viewbox_mouse_event`` is
    connected as a **bound method** in ``EventEmitter._callbacks``, so
    patching the class or instance method *after* connection has no effect —
    the original function pointer is frozen at connect time.

    The fix is to **disconnect** the camera's original callbacks from the
    viewbox's mouse events and **connect** new filtered wrappers that:

    * Block right-button events from reaching the camera (preventing zoom).
    * Show a "Save Viewport As…" context menu on right-click press.
    * Forward all other events to the original camera handler unchanged.
    """
    from unittest.mock import MagicMock

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMenu

    try:
        if isinstance(viewer.window, MagicMock):
            return
        qv = viewer.window._qt_viewer
        canvas = getattr(qv, "canvas", None)
        if canvas is None:
            return

        view = getattr(canvas, "view", None)
        if view is None:
            return
        camera = getattr(view, "camera", None)
        if camera is None:
            return

        original_handler = getattr(camera, "viewbox_mouse_event", None)
        if original_handler is None:
            return

        _mouse_names = ("mouse_press", "mouse_move", "mouse_release")
        for name in _mouse_names:
            emitter = getattr(view.events, name, None)
            if emitter is not None:
                try:
                    emitter.disconnect(original_handler)
                except Exception:
                    pass

        _right_active = [False]

        def _show_save_menu():
            gpos = canvas.native.cursor().pos()
            menu = QMenu()
            menu.addAction("Save Viewport As…")
            menu.triggered.connect(
                lambda action: (
                    _save_viewport(viewer)
                    if "Save" in (action.text() if action else "")
                    else None
                )
            )
            menu.exec(gpos)

        def _filtered_mouse_event(event):
            etype = getattr(event, "type", None)

            if etype == "mouse_press" and getattr(event, "button", None) == 2:
                _right_active[0] = True
                event.handled = True
                QTimer.singleShot(0, _show_save_menu)
                return

            if _right_active[0]:
                event.handled = True
                if etype == "mouse_release":
                    _right_active[0] = False
                return

            original_handler(event)

        for name in _mouse_names:
            emitter = getattr(view.events, name, None)
            if emitter is not None:
                emitter.connect(_filtered_mouse_event)

    except Exception:
        pass


def _crop_to_content(arr):
    """Crop an RGB/RGBA screenshot to the tight bounding box of the data.

    napari's ``screenshot(canvas_only=True)`` still contains the uniformly
    filled canvas background (letterboxing when the data does not fill the
    canvas).  The background is assumed to be a single solid colour, sampled
    from the top-left corner; every pixel that differs from it is content,
    and the result is cropped to that region's bounding box.

    Parameters
    ----------
    arr : numpy.ndarray
        ``H x W x (3 or 4)`` uint8 image.

    Returns
    -------
    numpy.ndarray
        Cropped image, or ``arr`` unchanged if it has no colour channel or
        no non-background pixels.
    """
    import numpy as np

    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    h, w = arr.shape[:2]
    bg = arr[0, 0, :3].astype(int)
    rgb = arr[:, :, :3].astype(int)
    # Allow a small tolerance for anti-aliased edges against the background.
    diff = np.any(np.abs(rgb - bg) > 8, axis=2)
    if not diff.any():
        return arr
    rows = np.where(diff.any(axis=1))[0]
    cols = np.where(diff.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return arr[y0:y1, x0:x1]


def _viewer_source_name(viewer) -> str | None:
    """Best-effort source file name for a napari viewer's current content.

    napari records the path a layer was loaded from in ``layer.source.path``
    (image/points/etc.).  Returns the first such path's name, or ``None`` if
    no layer carries a source path (e.g. generated/empty viewers).
    """
    try:
        for layer in viewer.layers:
            src = getattr(layer, "source", None)
            path = getattr(src, "path", None)
            if path:
                return Path(path).name
    except Exception:
        pass
    return None


def _save_viewport(viewer, parent=None):
    """Capture the napari canvas and save it via the file dialog.

    Parameters
    ----------
    viewer : napari.Viewer
        The viewer whose canvas to capture.
    parent : QWidget, optional
        Parent for the save dialog.
    """
    import numpy as np
    from PySide6.QtGui import QImage

    try:
        arr = np.ascontiguousarray(viewer.screenshot(canvas_only=True, flash=False))
    except Exception:
        return
    # Drop the empty canvas padding around the data (mirrors the gallery
    # save, which crops to the drawn thumbnail bounding box).
    arr = _crop_to_content(arr)
    # _crop_to_content returns a (possibly non-contiguous) view; QImage needs
    # contiguous memory, so copy before wrapping.
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
    qimg.ndarray = arr
    # Prefer the name recorded by _open_file (covers openers that add layers
    # directly); fall back to napari's own layer.source.path for files opened
    # through viewer.open().
    source = getattr(viewer, "_source_name", None) or _viewer_source_name(viewer)
    _save_qimage(qimg, parent, default_name=source)


def _launch_chimerax(path: str) -> None:
    """Open ``path`` in an external ChimeraX process.

    ChimeraX is launched detached so it runs independently of the helicon
    process. A clear message is printed if ChimeraX cannot be found.
    """
    import subprocess

    from helicon.lib.napari_widgets import _find_chimerax

    exe = _find_chimerax()
    if exe is None:
        print(
            "[helicon] ChimeraX not found. Install it from "
            "https://www.cgl.ucsf.edu/chimerax/ or put it on your PATH."
        )
        return
    try:
        subprocess.Popen([exe, path])
        print(f"[helicon] launched ChimeraX with {path}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[helicon] failed to launch ChimeraX: {exc}")


def _hide_layer_panels(viewer) -> None:
    """Hide the left-side layer list and layer controls dock widgets.

    The display command shows the floating folder browser as the primary
    navigation surface, so the napari layer panel is hidden by default in
    both the main and any new display window. A middle-click on the canvas
    toggles it back.
    """
    try:
        from unittest.mock import MagicMock

        if isinstance(viewer.window, MagicMock):
            return
        qv = viewer.window._qt_viewer
        qv.dockLayerList.hide()
        qv.dockLayerControls.hide()
    except Exception:
        pass


def _install_panel_toggle(viewer) -> None:
    """Install a middle-click handler that toggles the layer panel.

    Middle-clicking the canvas shows/hides the left-side layer list and
    layer controls. The display area keeps its size and screen position:
    the window grows leftward when the panel opens and shrinks from the
    left when it closes, so the canvas is never resized.

    Three sources of unwanted zoom had to be defeated:

    * A Qt event filter consumes ``MouseButtonPress`` (triggers panel toggle),
      ``MouseButtonRelease`` (clears the shared ``_middle_held`` flag), and
      **``MouseButtonDblClick``** (napari binds ALL double-clicks to
      ``double_click_to_zoom`` which multiplies ``viewer.camera.zoom * 2`` —
      this was the root cause of the progressive zoom on rapid middle clicks).

    * A wrap around the camera's ``viewbox_mouse_event`` suppresses camera
      pan/zoom when the ``button`` or ``buttons`` field contains a middle value.
      VisPy and Qt number the middle button differently (3 vs 4), so all
      of ``(2, 3, 4)`` are checked.  The viewbox stores the connection as
      ``(camera, "viewbox_mouse_event")`` — a weak ref + method name — so
      patching the instance attribute (not the class) is honoured on each emit.

    * The ``_middle_held`` flag bridges two layers: the Qt event filter sets it
      on middle press; the camera wrap reads it to suppress ``mouse_wheel``
      events that have ``buttons=[]`` (no button reported).  These micro-scroll
      events are generated by the physical scroll wheel mechanism as it
      settles — they carry neither the middle button flag nor arrive through the
      Qt event filter, so only the flag can suppress them.

    napari swaps the camera instance when the 2D/3D display mode changes, so
    the wrap is re-applied whenever ``dims.ndisplay`` changes.

    References
    ----------
    VisPy button map (``_qt.py``): 1=left, 2=right, 3=middle
    Qt enum values: 1=left, 2=right, 4=middle
    napari double-click zoom: ``_viewer_mouse_bindings.py:double_click_to_zoom``
    """
    try:
        from unittest.mock import MagicMock

        from PySide6.QtCore import QEvent, QObject, Qt

        if isinstance(viewer.window, MagicMock):
            return
        qv = viewer.window._qt_viewer
        layer_list = qv.dockLayerList
        layer_controls = qv.dockLayerControls

        # Shared state: the event filter sets this on press/release, and the
        # camera wrap reads it to suppress micro-scroll events that arrive as
        # a side-effect of pressing the physical scroll wheel.
        _middle_held = [False]

        class _MiddleClickFilter(QObject):
            def __init__(
                self, layer_list, layer_controls, qt_viewer, viewer, parent=None
            ):
                super().__init__(parent)
                self._layer_list = layer_list
                self._layer_controls = layer_controls
                self._qt_viewer = qt_viewer
                self._viewer = viewer
                self._saved_dock_w = 0
                self._toggling = False
                self._pending_timer = None
                self._panel_shown = False

            def _toggle_panel(self):
                if self._toggling:
                    return
                if self._pending_timer is not None:
                    self._pending_timer.stop()
                    self._pending_timer = None
                self._toggling = True
                try:
                    self._do_toggle()
                finally:
                    self._toggling = False

            def _apply_geometry(self, handle, win, x, y, w, h):
                if handle:
                    handle.setGeometry(x, y, w, h)
                else:
                    win.setGeometry(x, y, w, h)

            def _cache_display_rect(self, handle, win):
                self._viewer._display_only_ba = win.saveGeometry()

            def _do_toggle(self):
                win = self._qt_viewer.window()
                if win is None:
                    return
                from PySide6.QtCore import QTimer
                from PySide6.QtWidgets import QStyle

                handle = win.windowHandle()
                style = self._layer_list.style()
                grip = max(
                    style.pixelMetric(QStyle.PixelMetric.PM_DockWidgetSeparatorExtent),
                    style.pixelMetric(QStyle.PixelMetric.PM_DockWidgetHandleExtent),
                    0,
                )
                hiding = self._layer_list.isVisible()
                base = handle.geometry() if handle else win.geometry()
                if hiding:
                    dock_w = self._layer_list.width() + grip
                    self._saved_dock_w = dock_w
                    self._layer_list.hide()
                    self._layer_controls.hide()
                    self._panel_shown = False
                    new_x = base.x() + dock_w
                    new_w = max(base.width() - dock_w, 1)
                    self._apply_geometry(
                        handle, win, new_x, base.y(), new_w, base.height()
                    )
                    self._cache_display_rect(handle, win)
                else:
                    self._cache_display_rect(handle, win)
                    if self._saved_dock_w > 0:
                        dock_w = self._saved_dock_w
                        new_x = base.x() - dock_w
                        new_w = base.width() + dock_w
                        self._apply_geometry(
                            handle, win, new_x, base.y(), new_w, base.height()
                        )
                        self._layer_list.show()
                        self._layer_controls.show()
                        self._panel_shown = True
                    else:
                        # Docks were hidden since cold start and have never
                        # been laid out, so width() is not yet valid. Resize the
                        # window first using a sizeHint estimate so the docks
                        # land in the new left-hand space (no canvas-overlap
                        # flash), then correct on the next turn once Qt has laid
                        # out the freshly-shown docks and width() is real.
                        est = self._layer_list.sizeHint().width() + grip
                        self._saved_dock_w = est
                        new_x = base.x() - est
                        new_w = base.width() + est
                        self._apply_geometry(
                            handle, win, new_x, base.y(), new_w, base.height()
                        )
                        self._layer_list.show()
                        self._layer_controls.show()
                        self._panel_shown = True
                        self._pending_timer = QTimer()
                        self._pending_timer.setSingleShot(True)
                        self._pending_timer.timeout.connect(
                            lambda: self._finish_first_show(handle, win, grip, est)
                        )
                        self._pending_timer.start(0)

            def _finish_first_show(self, handle, win, grip, est):
                self._pending_timer = None
                dock_w = self._layer_list.width() + grip
                self._saved_dock_w = dock_w
                # The window was already shifted left by ``est`` in the
                # current turn; correct only the remaining difference.
                delta = dock_w - est
                if delta == 0:
                    return
                base = handle.geometry() if handle else win.geometry()
                new_x = base.x() - delta
                new_w = base.width() + delta
                self._apply_geometry(handle, win, new_x, base.y(), new_w, base.height())

            def eventFilter(self, obj, event):
                etype = event.type()
                if etype in (
                    QEvent.MouseButtonPress,
                    QEvent.MouseButtonRelease,
                    QEvent.MouseButtonDblClick,
                ):
                    if event.button() == Qt.MouseButton.MiddleButton:
                        if etype == QEvent.MouseButtonPress:
                            _middle_held[0] = True
                            self._toggle_panel()
                            return True
                        elif etype == QEvent.MouseButtonRelease:
                            _middle_held[0] = False
                        elif etype == QEvent.MouseButtonDblClick:
                            return True
                return False

        canvas_native = getattr(getattr(qv, "canvas", None), "native", None)
        if isinstance(canvas_native, QObject):
            mf = _MiddleClickFilter(
                layer_list,
                layer_controls,
                qv,
                viewer,
                parent=canvas_native,
            )
            canvas_native.installEventFilter(mf)
        else:
            mf = _MiddleClickFilter(layer_list, layer_controls, qv, viewer, parent=qv)
            qv.installEventFilter(mf)
        viewer._panel_toggle = mf

        view = getattr(getattr(qv, "canvas", None), "view", None)
        if view is None:
            return

        # VisPy reports middle as 3 (its own enum), Qt reports it as 4,
        # and some builds pass through the raw Qt value.  Suppress all
        # three, including when middle is held during mouse_wheel/move.
        middle_values = (2, 3, 4)

        def _wrap_camera() -> None:
            camera = getattr(view, "camera", None)
            if camera is None or getattr(camera, "_panel_toggle_wrapped", False):
                return
            original = camera.viewbox_mouse_event

            def _wrapped_viewbox_mouse_event(event) -> None:
                btn = getattr(event, "button", None)
                btns = getattr(event, "buttons", None) or []
                is_middle = (
                    btn in middle_values
                    or any(b in middle_values for b in btns)
                    or (
                        getattr(event, "type", None) == "mouse_wheel"
                        and _middle_held[0]
                    )
                )
                if is_middle:
                    try:
                        event.handled = True
                    except Exception:
                        pass
                    return
                original(event)

            camera.viewbox_mouse_event = _wrapped_viewbox_mouse_event
            camera._panel_toggle_wrapped = True

        _wrap_camera()
        try:
            viewer.dims.events.ndisplay.connect(lambda *a, **k: _wrap_camera())
        except Exception:
            pass
    except Exception:
        pass


def _is_text_file(path: str) -> bool:
    """Check if a file is a text file by attempting UTF-8 decode.

    Called only after known-type suffix checks have failed, so this is the
    fallback that distinguishes pure-text files from unknown binary.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def _open_text(viewer, path: str) -> None:
    """Open a text file as an overlay in the main viewer canvas."""
    from pathlib import Path

    try:
        from PySide6.QtWidgets import QTextEdit
        from PySide6.QtGui import QFont
        from PySide6.QtCore import Qt
    except ImportError:
        return

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return

    qt_window = viewer.window._qt_window
    central = qt_window.centralWidget()

    if hasattr(qt_window, "_text_overlay") and qt_window._text_overlay.isVisible():
        qt_window._text_overlay.hide()

    if not hasattr(qt_window, "_text_overlay"):
        overlay = QTextEdit(central)
        overlay.setReadOnly(True)
        overlay.setFont(QFont("Courier New", 12))
        overlay.setLineWrapMode(QTextEdit.NoWrap)
        overlay.setStyleSheet(
            "background-color: #2d2d2d; color: #cccccc; border: none;"
        )
        overlay.hide()
        qt_window._text_overlay = overlay
    else:
        overlay = qt_window._text_overlay

    overlay.setPlainText(content)
    overlay.setGeometry(central.rect())
    overlay.show()
    overlay.raise_()
    overlay.setFocus()

    name = Path(path).name
    overlay.setWindowTitle(f"helicon - {name}")


def _open_html(viewer, path: str) -> None:
    import webbrowser

    webbrowser.open(f"file://{path}")


def _open_pdf(viewer, path: str) -> None:
    """Open a PDF file, rendering pages as images in napari."""
    from pathlib import Path

    try:
        from PySide6.QtPdf import QPdfDocument
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QImage
    except ImportError:
        return
    import numpy as np

    doc = QPdfDocument()
    doc.load(path)
    n_pages = doc.pageCount()
    if n_pages == 0:
        return

    dpi = 150
    pages = []
    for i in range(n_pages):
        pt_size = doc.pagePointSize(i)
        w_px = int(pt_size.width() * dpi / 72)
        h_px = int(pt_size.height() * dpi / 72)
        img = doc.render(i, QSize(w_px, h_px))
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        ptr = img.bits()
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(
            img.height(), img.width(), 4
        )
        rgb = arr[:, :, 2::-1].astype(np.float32)  # BGRA -> RGB float
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        # Composite onto white background
        composite = alpha * rgb + (1.0 - alpha) * 255.0
        pages.append(composite)

    data = np.stack(pages) if len(pages) > 1 else pages[0]
    name = Path(path).name
    contrast = _auto_contrast(data)
    # Multi-page PDF is a stack of 2D pages, not a 3D volume: hide the
    # axis selector so it does not appear for the page-frame axis.
    _SliceDirectionWidget.set_stack_mode(len(pages) > 1)
    layer = viewer.add_image(
        data,
        name=name,
        contrast_limits=contrast,
        interpolation2d="linear",
        interpolation3d="linear",
    )
    _enable_continuous_auto_contrast(layer, viewer)
    layer.contrast_limits_range = (float(data.min()), float(data.max()))
    _reset_view(viewer)


def _open_eps(viewer, path: str) -> None:
    """Open an EPS (PostScript) file by rasterizing it with Ghostscript.

    Qt has no PostScript interpreter, so EPS cannot be read by QPdfDocument
    or QImageReader. We shell out to ``gs`` to render the EPS to a PNG and
    then display that image, reusing the same white-background compositing
    and contrast logic as the PDF viewer.
    """
    import shutil
    import subprocess
    import tempfile

    import numpy as np
    from PySide6.QtGui import QImage

    gs = shutil.which("gs") or shutil.which("ghostscript")
    if gs is None:
        print(
            "[helicon] Ghostscript (gs) is required to display EPS files but "
            "was not found on your PATH."
        )
        return

    tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_png.close()
    out_path = tmp_png.name

    try:
        subprocess.run(
            [
                gs,
                "-dQUIET",
                "-dNOPAUSE",
                "-dBATCH",
                "-dSAFER",
                "-dEPSCrop",
                "-sDEVICE=png16m",
                "-r150",
                f"-sOutputFile={out_path}",
                path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        img = QImage(out_path)
        if img.isNull():
            print(f"[helicon] failed to render EPS: {path}")
            return
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        ptr = img.bits()
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(
            img.height(), img.width(), 4
        )
        rgb = arr[:, :, 2::-1].astype(np.float32)  # BGRA -> RGB float
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        # Composite onto white background
        composite = alpha * rgb + (1.0 - alpha) * 255.0

        from pathlib import Path as _Path

        name = _Path(path).name
        contrast = _auto_contrast(composite)
        layer = viewer.add_image(
            composite,
            name=name,
            contrast_limits=contrast,
            interpolation2d="linear",
            interpolation3d="linear",
        )
        _enable_continuous_auto_contrast(layer, viewer)
        layer.contrast_limits_range = (
            float(composite.min()),
            float(composite.max()),
        )
        _reset_view(viewer)
    except (subprocess.CalledProcessError, OSError) as exc:  # pragma: no cover
        print(f"[helicon] failed to display EPS {path}: {exc}")
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def _capped_cylinder_mesh(p1, p2, radius, segments=24):
    """Vertices + triangle indices for a capped cylinder between ``p1`` and
    ``p2`` (ChimeraX BILD semantics: capped by default).

    Ported from ChimeraX ``shape.cylinder_geometry``: the side is a tube and
    each flat end is closed with a triangle fan so the rod looks solid.
    """
    import numpy as np

    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    axis = p2 - p1
    length = float(np.linalg.norm(axis))
    if length == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    a = axis / length
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    bot = p1 + ring
    top = p2 + ring
    verts = np.vstack([bot, top, p1, p2])
    n_bot, n_top = 2 * segments, segments
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, n_top + j])
        faces.append([i, n_top + j, n_top + i])
        faces.append([n_bot, j, i])
        faces.append([n_bot + 1, n_top + i, n_top + j])
    return verts, np.array(faces, int)


def _sphere_mesh(center, radius, rings=16, segments=24):
    """Vertices + triangle indices for a UV sphere (always closed)."""
    import numpy as np

    center = np.asarray(center, float)
    verts = [center]
    for i in range(1, rings):
        phi = np.pi * i / rings
        r = radius * np.sin(phi)
        y = radius * np.cos(phi)
        theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        for t in theta:
            verts.append(center + np.array([r * np.cos(t), y, r * np.sin(t)]))
    verts.append(center + np.array([0.0, radius, 0.0]))
    verts = np.array(verts)
    n = len(verts)
    pole_s, pole_n = 0, n - 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([pole_s, j + 1, i + 1])
    for k in range(rings - 2):
        base = 1 + k * segments
        nxt = base + segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([base + i, base + j, nxt + j])
            faces.append([base + i, nxt + j, nxt + i])
    last_base = 1 + (rings - 2) * segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([pole_n, last_base + j, last_base + i])
    return verts, np.array(faces, int)


def _open_bild(viewer, path: str) -> None:
    from pathlib import Path
    import numpy as np

    # Each primitive becomes a capped 3D surface (ChimeraX semantics: cylinders
    # are solid/capped unless the ``open`` keyword is given). We render every
    # primitive as its own surface layer so per-object colors are preserved.
    meshes = []  # list of (vertices, faces, color)
    current_color = [1.0, 1.0, 1.0]

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(".color"):
                parts = line.split()
                current_color = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif line.startswith(".cylinder"):
                parts = line.split()
                x1, y1, z1 = float(parts[1]), float(parts[2]), float(parts[3])
                x2, y2, z2 = float(parts[4]), float(parts[5]), float(parts[6])
                r = float(parts[7])
                # ``open`` keyword (8th field) leaves the cylinder uncapped.
                capped = not (len(parts) > 8 and parts[8] == "open")
                verts, faces = _capped_cylinder_mesh((x1, y1, z1), (x2, y2, z2), r)
                if not capped:
                    # Drop the two end-cap fans (last 2 * segments triangles);
                    # approximate by rebuilding an uncapped tube.
                    n = len(verts)
                    segs = (n - 4) // 2
                    side_faces = faces[: 2 * segs]
                    verts = verts[: 2 * segs]
                    faces = side_faces
                meshes.append((verts, faces, current_color))
            elif line.startswith(".sphere"):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r = float(parts[4])
                verts, faces = _sphere_mesh((x, y, z), r)
                meshes.append((verts, faces, current_color))

    if not meshes:
        return

    # Merge every primitive into a single surface layer. A BILD file can hold
    # thousands of objects (e.g. an angular-distribution plot); one layer per
    # object would create thousands of layers and stall napari's 3D view.
    all_verts = []
    all_faces = []
    all_colors = []
    offset = 0
    for verts, faces, color in meshes:
        if len(verts) == 0 or len(faces) == 0:
            continue
        all_verts.append(verts)
        all_faces.append(faces + offset)
        all_colors.append(np.tile(np.asarray(color, float), (len(verts), 1)))
        offset += len(verts)

    if not all_verts:
        return

    vertices = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    vertex_colors = np.vstack(all_colors)
    name = Path(path).name
    viewer.add_surface(
        (vertices, faces),
        name=name,
        vertex_colors=vertex_colors,
        shading="smooth",
    )
    viewer.dims.ndisplay = 3
    _reset_view(viewer)


def _patch_napari_value_bug() -> None:
    """Patch napari 0.8.0 bugs that crash the UI in 3D display mode.

    Two independent napari 0.8.0 bugs are triggered by switching a volume
    layer to 3D (``viewer.dims.ndisplay = 3``):

    1. ``ScalarFieldBase._get_value_3d`` computes a 2D ``slice_shape`` but
       indexes it with a 3D ``dims_displayed`` (``slice_shape`` is
       ``(256, 256)`` while ``level0_shape`` is ``(256, 256, 256)``),
       raising ``ValueError`` from ``_get_value_ray``. The ``StatusChecker``
       thread calls ``get_value`` on every mouse move, so this crashes the
       UI thread.

    2. ``Image._update_thumbnail`` receives a 1D thumbnail (the napari
       projection collapses the 2D slice to 1D) while ``zoom_factor`` is
       2D, so ``scipy.ndimage.zoom`` raises ``RuntimeError``. This fires
       during the re-slice that ``ndisplay = 3`` triggers.

    In both cases the failure is non-essential (cursor readout / layer
    thumbnail), so we patch the class methods once and degrade gracefully
    instead of crashing.

    Per-instance attribute patching did not take effect because the
    ``StatusChecker`` thread resolves the class method rather than the
    monkey-patched instance attribute, so we patch the classes directly.
    """
    try:
        from napari.layers._scalar_field.scalar_field import ScalarFieldBase
    except Exception:
        ScalarFieldBase = None

    try:
        from napari.layers.image.image import Image as _NapariImage
    except Exception:
        _NapariImage = None

    if ScalarFieldBase is not None and not getattr(
        ScalarFieldBase, "_helicon_value_patched", False
    ):
        _orig_get_value_3d = ScalarFieldBase._get_value_3d

        def _safe_get_value_3d(
            self, start_point=None, end_point=None, dims_displayed=None
        ):
            try:
                return _orig_get_value_3d(
                    self,
                    start_point=start_point,
                    end_point=end_point,
                    dims_displayed=dims_displayed,
                )
            except (ValueError, IndexError, RuntimeError):
                return None

        ScalarFieldBase._get_value_3d = _safe_get_value_3d
        ScalarFieldBase._helicon_value_patched = True

    if _NapariImage is not None and not getattr(
        _NapariImage, "_helicon_thumb_patched", False
    ):
        _orig_update_thumbnail = _NapariImage._update_thumbnail

        def _safe_update_thumbnail(self):
            try:
                return _orig_update_thumbnail(self)
            except (ValueError, IndexError, RuntimeError):
                return None

        _NapariImage._update_thumbnail = _safe_update_thumbnail
        _NapariImage._helicon_thumb_patched = True


def _is_metadata_star(path: str) -> bool:
    """Star files that describe pipelines/optimisation rather than image data.

    These should be opened as text, not as an image/volume stack.
    """
    from pathlib import Path

    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in _METADATA_STAR_SUFFIXES)


def _is_optimiser_star(path: str) -> bool:
    """Return True for RELION optimiser.star files.

    These contain references to MRC files whose center slices can be
    displayed in a gallery view.
    """
    from pathlib import Path

    return Path(path).name.lower().endswith("optimiser.star")


def _parse_optimiser_star(optimiser_path: str) -> list[str] | None:
    """Parse a RELION optimiser.star file and extract referenced MRC file paths.

    The optimiser.star file references model.star files, which in turn
    contain the actual MRC file paths in the ``_rlnReferenceImage`` column.

    Parameters
    ----------
    optimiser_path : str
        Path to the optimiser.star file.

    Returns
    -------
    list of str or None
        List of resolved MRC file paths, or None if parsing fails.
    """
    from pathlib import Path

    star_dir = Path(optimiser_path).parent
    model_star_paths: list[str] = []

    try:
        with open(optimiser_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("_rlnModelStarFile"):
                    parts = s.split()
                    if len(parts) >= 2:
                        model_rel = parts[-1]
                        for ancestor in [star_dir] + list(star_dir.parents):
                            candidate = ancestor / model_rel
                            if candidate.is_file():
                                model_star_paths.append(str(candidate))
                                break
    except Exception:
        return None

    if not model_star_paths:
        return None

    mrc_paths: list[str] = []
    for model_path in model_star_paths:
        result = _parse_model_star(model_path)
        if result:
            for p in result:
                if p not in mrc_paths:
                    mrc_paths.append(p)

    return mrc_paths if mrc_paths else None


def _parse_model_star(model_path: str) -> list[str] | None:
    """Extract referenced MRC file paths from a RELION model.star.

    Reads the ``data_model_classes`` section and resolves the
    ``_rlnReferenceImage`` column to absolute MRC paths.
    """
    from pathlib import Path

    model_dir = Path(model_path).parent
    in_loop = False
    in_data_model_classes = False
    col_names: list[str] = []
    ref_image_col_idx = -1
    mrc_paths: list[str] = []

    try:
        with open(model_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("data_"):
                    in_data_model_classes = "model_classes" in s
                    in_loop = False
                    col_names = []
                    ref_image_col_idx = -1
                    continue

                if s == "loop_":
                    in_loop = True
                    col_names = []
                    ref_image_col_idx = -1
                    continue

                if in_loop and s.startswith("_"):
                    col_names.append(s.split()[0])
                    if "referenceimage" in s.lower():
                        ref_image_col_idx = len(col_names) - 1
                    continue

                if not in_data_model_classes or ref_image_col_idx < 0:
                    continue

                if not in_loop:
                    continue

                parts = s.split()
                if ref_image_col_idx >= len(parts):
                    continue

                mrc_rel = parts[ref_image_col_idx]

                resolved = None
                for ancestor in [model_dir] + list(model_dir.parents):
                    candidate = ancestor / mrc_rel
                    if candidate.is_file():
                        resolved = str(candidate)
                        break

                if resolved and resolved not in mrc_paths:
                    mrc_paths.append(resolved)
    except Exception:
        return None

    return mrc_paths if mrc_paths else None


def _parse_star_image_refs(
    star_path: str,
) -> tuple[list[tuple[int, str, float]], tuple, float] | None:
    """Parse a .star file line-by-line and build lazy image-stack entries.

    Extracts only the ImageName/MicrographName column instead of loading the
    entire file into a pandas DataFrame (which blocks for minutes on large
    *data.star files with millions of particles). Resolved MRC paths are
    cached because most particles reference frames from the same file.

    Parameters
    ----------
    star_path : str
        Path to the .star file.

    Returns
    -------
    tuple of (entries, first_shape, first_apix) or None
        * ``entries``: list of ``(frame_idx_0based, mrc_path, 0.0)`` tuples.
        * ``first_shape``: ``(nx, ny)`` or ``(nx, ny, nz)`` of the first image.
        * ``first_apix``: pixel size in Angstroms (fallback 1.0).
        Returns None if no image references could be resolved.
    """
    from pathlib import Path

    import mrcfile

    star_dir = Path(star_path).parent

    col_names: list[str] = []
    image_col_idx = -1
    in_loop = False
    in_data = False
    entries: list[tuple[int, str, float]] = []
    first_shape: tuple | None = None
    first_apix = 1.0
    resolved_cache: dict[str, str | None] = {}

    def _resolve(img_rel: str) -> str | None:
        if img_rel in resolved_cache:
            return resolved_cache[img_rel]
        resolved = None
        for ancestor in [star_dir] + list(star_dir.parents):
            candidate = ancestor / img_rel
            if candidate.is_file():
                resolved = str(candidate)
                break
        resolved_cache[img_rel] = resolved
        return resolved

    try:
        with open(star_path) as f:
            for line in f:
                raw = line.rstrip("\n\r")
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue

                if s.startswith("data_") or s == "loop_":
                    in_loop = s == "loop_"
                    in_data = False
                    if in_loop:
                        col_names = []
                        image_col_idx = -1
                    continue

                if in_loop and s.startswith("_"):
                    col_names.append(s.split()[0])
                    if image_col_idx < 0:
                        cl = s.lower()
                        if "imagename" in cl or "micrographname" in cl:
                            image_col_idx = len(col_names) - 1
                    continue

                if image_col_idx < 0:
                    continue

                if not in_data:
                    in_data = True

                parts = raw.split()
                if image_col_idx >= len(parts):
                    continue
                image_ref = parts[image_col_idx]

                if "@" in image_ref:
                    idx_str, img_rel = image_ref.split("@", 1)
                else:
                    idx_str, img_rel = "1", image_ref

                try:
                    frame_idx = int(idx_str) - 1
                except ValueError:
                    continue

                resolved_path = _resolve(img_rel)
                if resolved_path is None:
                    continue

                entries.append((frame_idx, resolved_path, 0.0))

                if first_shape is None:
                    with mrcfile.open(resolved_path, header_only=True) as mrc:
                        nx = int(mrc.header.nx)
                        ny = int(mrc.header.ny)
                        nz = int(mrc.header.nz)
                        first_shape = (
                            (nx, ny)
                            if nz == 1 or Path(resolved_path).suffix.lower() == ".mrcs"
                            else (nx, ny, nz)
                        )
                        first_apix = float(mrc.voxel_size.x)
                        if first_apix <= 0:
                            first_apix = 1.0
    except Exception:
        return None

    if not entries or first_shape is None:
        return None

    return entries, first_shape, first_apix


def _set_ndisplay(viewer, value: int) -> None:
    """Set the viewer's 2D/3D display dimension, tolerating empty dims.

    Setting ``dims.ndisplay`` before any layer exists (or on a mock
    viewer in tests) is harmless; guard so a transient state never
    raises and aborts the open.
    """
    try:
        viewer.dims.ndisplay = value
    except (AttributeError, RuntimeError, ValueError):
        pass


def _reset_view(viewer) -> None:
    """Recenter/zoom the camera to fit the current layers ("home" view).

    Called after a new file's layer is added so the incoming file is
    framed correctly rather than left at the previous file's camera pose.
    Guarded so a mock viewer in tests (or a transient state) never
    raises and aborts the open.
    """
    try:
        viewer.reset_view()
    except (AttributeError, RuntimeError, ValueError):
        pass


# Gallery window lifecycle — mirrors _viewers / _active_viewer for napari.
# Gallery windows are plain QMainWindow objects kept alive here until closed.
_galleries: list = []
_active_gallery: list = [None]  # last-focused gallery window


def _gallery_is_alive(w) -> bool:
    try:
        return w is not None and w.isVisible()
    except Exception:
        return False


def _on_gallery_closing(w) -> None:
    if w in _galleries:
        _galleries.remove(w)
    if _active_gallery[0] is w:
        _active_gallery[0] = _galleries[-1] if _galleries else None


def _wrap_gallery_with_panel(gallery: "ImageGalleryWidget") -> "QWidget":
    """Wrap an ImageGalleryWidget with a left-side _ControlPanel sibling.

    The panel is prepended to the left.  Toggling it grows the parent
    window leftward by ``_ControlPanel.PANEL_WIDTH`` so the gallery
    widget keeps both its width and its screen position unchanged.
    """
    from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

    from helicon.lib.image_gallery import _ControlPanel

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

    def _on_contrast(val):
        gallery.set_contrast(val / 100.0)
        panel._contrast_val.setText(f"{gallery._contrast:.2f}")

    def _on_gamma(val):
        gallery.set_gamma(val / 100.0)
        panel._gamma_val.setText(f"{gallery._gamma:.2f}")

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

    return container


def _open_gallery(read_fn, n, img_w, img_h, apix, name, reuse_window=None) -> None:
    """Show the lazy thumbnail grid for a stack in a standalone window.

    The gallery is a self-contained :class:`PySide6.QtWidgets.QMainWindow`
    with an :class:`ImageGalleryWidget`; it does not depend on a napari viewer.

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
    """
    from PySide6.QtWidgets import QMainWindow

    from helicon.lib.image_gallery import ImageGalleryWidget

    widget = ImageGalleryWidget()
    widget.set_data(read_fn, n, img_w, img_h, None, source_name=name)

    container = _wrap_gallery_with_panel(widget)

    if reuse_window is not None:
        reuse_window.setWindowTitle(f"helicon - {name}")
        reuse_window.setCentralWidget(container)
        reuse_window.show()
        reuse_window.raise_()
        return reuse_window

    class _GalleryWindow(QMainWindow):
        def closeEvent(self, event):
            _on_gallery_closing(self)
            super().closeEvent(event)

    window = _GalleryWindow()
    window.setWindowTitle(f"helicon - {name}")
    window.setCentralWidget(container)
    tile = 128 + widget._panel.min_sep
    window.resize(5 * tile + widget._sb_width, 5 * tile)
    _galleries.append(window)
    _active_gallery[0] = window
    window.show()
    return window


def _open_xyz_slice_gallery(star_path: str, reuse_window=None) -> "QMainWindow | None":
    """Display center slices (Z, Y, X) of MRC files referenced by a star file.

    Works for both ``*optimiser.star`` (which indirects through model.star)
    and ``*model.star`` (which directly lists MRC references). Each MRC file
    gets one row of three center slices.
    """
    from pathlib import Path

    import mrcfile
    import numpy as np
    from PySide6.QtWidgets import QMainWindow

    from helicon.lib.image_gallery import ImageGalleryWidget

    name = Path(star_path).name
    if name.endswith("model.star"):
        mrc_paths = _parse_model_star(star_path)
    else:
        mrc_paths = _parse_optimiser_star(star_path)
    if not mrc_paths:
        return None

    slices_per_mrc = 3
    total_images = len(mrc_paths) * slices_per_mrc
    img_w = img_h = 0

    def _read_slice(i: int) -> np.ndarray:
        nonlocal img_w, img_h
        mrc_idx = i // slices_per_mrc
        slice_idx = i % slices_per_mrc

        with mrcfile.open(mrc_paths[mrc_idx], permissive=True) as mrc:
            data = mrc.data
            nz, ny, nx = data.shape
            z_center = nz // 2
            y_center = ny // 2
            x_center = nx // 2

            if slice_idx == 0:
                sl = data[z_center, :, :]
            elif slice_idx == 1:
                sl = data[:, y_center, :]
            else:
                sl = data[:, :, x_center]

            if img_w == 0:
                img_h, img_w = sl.shape[:2]

            return sl.astype(np.float32)

    _read_slice(0)

    axis_labels = ["Z", "Y", "X"]
    labels = [
        f"{mrc_idx + 1}-{axis_labels[slice_idx]}"
        for mrc_idx in range(len(mrc_paths))
        for slice_idx in range(slices_per_mrc)
    ]

    widget = ImageGalleryWidget()
    widget.set_data(
        _read_slice,
        total_images,
        img_w,
        img_h,
        None,
        labels=labels,
        source_name=Path(star_path).name,
    )

    container = _wrap_gallery_with_panel(widget)

    if reuse_window is not None:
        reuse_window.setWindowTitle(f"helicon - {Path(star_path).name}")
        reuse_window.setCentralWidget(container)
        reuse_window.show()
        reuse_window.raise_()
        return reuse_window

    class _GalleryWindow(QMainWindow):
        def closeEvent(self, event):
            _on_gallery_closing(self)
            super().closeEvent(event)

    window = _GalleryWindow()
    window.setWindowTitle(f"helicon - {Path(star_path).name}")
    window.setCentralWidget(container)
    tile = 128 + widget._panel.min_sep
    window.resize(5 * tile + widget._sb_width, 5 * tile)
    _galleries.append(window)
    _active_gallery[0] = window
    window.show()
    return window


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
        qt_window = viewer.window._qt_window
        if hasattr(qt_window, "_text_overlay") and qt_window._text_overlay.isVisible():
            qt_window._text_overlay.hide()

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
        viewer.title = f"helicon - {Path(path).name}"
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
        _open_xyz_slice_gallery(path, reuse_window=reuse_gallery)
        return

    # "metadata" mode: always open any star file as text, regardless of type.
    if ext == ".star" and mode == "metadata":
        _open_text(viewer, path)
        _reset_view(viewer)
        return

    if ext == ".star" and _is_metadata_star(path) and mode != "general":
        _open_text(viewer, path)
        _reset_view(viewer)
        return

    if ext == ".star" and mode != "general":
        import mrcfile

        result = _parse_star_image_refs(path)
        if result is None:
            return
        entries, first_shape, first_apix = result

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
        _open_text(viewer, path)
        return

    if ext in _MRC_EXTENSIONS and mode != "general":
        import mrcfile
        import numpy as np
        import dask.array as da
        from dask import delayed

        # Read only the header so opening a file never eagerly pulls the whole
        # volume/stack into memory. 2D views (slice and gallery) are read one
        # plane at a time on demand via a lazily-evaluated dask array.
        try:
            with mrcfile.open(path, permissive=True) as _mrc:
                _hdr = _mrc.header
                _nx = int(_hdr.nx)
                _ny = int(_hdr.ny)
                _nz = int(_hdr.nz)
                _mode = int(_hdr.mode)
                try:
                    _apix = float(_hdr.cella.x) / _nx if _nx else 1.0
                except (AttributeError, ZeroDivisionError):
                    _apix = 1.0
        except Exception:
            _nx = _ny = _nz = 0
            _mode = 2
            _apix = 1.0
        if _apix <= 0:
            try:
                with mrcfile.open(path, permissive=True) as _m:
                    _apix = float(_m.voxel_size.x)
            except Exception:
                _apix = 1.0
        if _apix <= 0:
            _apix = 1.0

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
        _dtype = _MODE_DTYPE.get(_mode, np.float32)

        def _read_raw(i):
            with mrcfile.open(path, permissive=True) as m:
                return np.asarray(m.data[i])

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

        # Single 2D plane: one eager read is fine and keeps the 2D path simple.
        if _vol is None or (_vol.ndim == 3 and _vol.shape[0] == 1):
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
            layer.contrast_limits_range = (
                float(_single.min()),
                float(_single.max()),
            )
            _reset_view(viewer)
            return

        # Gallery: read one plane at a time (lazy) — no full-volume load.
        if mode == "gallery":
            n = int(_vol.shape[0])
            img_w, img_h = int(_vol.shape[1]), int(_vol.shape[2])

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
            )
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
        layer.contrast_limits_range = (
            float(_sample.min()),
            float(_sample.max()),
        )

        if len(viewer.dims.current_step) > 0:
            step = list(viewer.dims.current_step)
            if ext == ".mrcs":
                step[0] = 0
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
    import os

    import napari

    _patch_napari_value_bug()

    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

    viewer = napari.Viewer(title=os.path.basename(path))
    _hide_layer_panels(viewer)
    _install_panel_toggle(viewer)
    try:
        if not isinstance(viewer.window, MagicMock):
            _SliceDirectionWidget(viewer).inject()
    except Exception:
        pass
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
    if not helicon.has_napari():
        raise HeliconDependencyError(
            "napari is required for the display command. "
            'Install it with: pip install "helicon[napari]"'
        )

    if FolderBrowserWidget is None:
        raise HeliconDependencyError(
            "Qt widgets are required for the display command. "
            "Install PySide6: pip install PySide6"
        )

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QWidget, QApplication
    from PySide6.QtGui import QShortcut, QKeySequence

    app = QApplication.instance() or QApplication(sys.argv)

    import napari

    _patch_napari_value_bug()

    start_dir = args.folder if args.folder else os.getcwd()

    viewer = napari.Viewer(title="helicon")
    _add_welcome_shortcut(viewer)

    if sys.platform == "darwin":
        try:
            from AppKit import NSApplication

            nsapp = NSApplication.sharedApplication()
            nsapp.activateIgnoringOtherApps_(True)
            menu = nsapp.mainMenu()
            if menu and menu.numberOfItems() > 0:
                menu.itemAtIndex_(0).submenu().setTitle_("helicon")
        except Exception:
            pass

    app = QApplication.instance()
    if app is not None:
        app.setApplicationName("helicon")

    _viewers = [viewer]
    _active_viewer = [viewer]

    def _on_focus_changed(old, new):
        from PySide6.QtWidgets import QWidget

        for v in _viewers:
            try:
                if v.window._qt_window.isAncestorOf(new) or v.window._qt_window == new:
                    _active_viewer[0] = v
                    break
            except Exception:
                pass
        for gw in _galleries:
            try:
                if gw.isAncestorOf(new) or gw == new:
                    _active_gallery[0] = gw
                    break
            except Exception:
                pass

    if app is not None:
        app.focusChanged.connect(_on_focus_changed)

    def _on_viewer_closing(closed_viewer):
        if closed_viewer in _viewers:
            _viewers.remove(closed_viewer)
        if _active_viewer[0] is closed_viewer:
            _active_viewer[0] = _viewers[0] if _viewers else None

    def _track_viewer(v):
        from PySide6.QtCore import QEvent, QObject

        class _CloseFilter(QObject):
            def __init__(self, viewer, parent=None):
                super().__init__(parent)
                self._viewer = viewer

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Close:
                    _on_viewer_closing(self._viewer)
                return False

        try:
            qt_window = v.window._qt_window
            if qt_window is not None and isinstance(qt_window, QObject):
                # Connect to destroyed: napari may tear the window down without
                # delivering a QEvent.Close, so this reliably tracks real closes.
                qt_window.destroyed.connect(lambda *_: _on_viewer_closing(v))
                flt = _CloseFilter(v, parent=qt_window)
                qt_window.installEventFilter(flt)
        except Exception:
            pass

    _track_viewer(viewer)
    try:
        from unittest.mock import MagicMock

        if not isinstance(viewer.window, MagicMock):
            _install_viewer_save_menu(viewer)
            slice_widget = _SliceDirectionWidget(viewer)
            slice_widget.inject()
    except Exception:
        pass

    _hide_layer_panels(viewer)
    _install_panel_toggle(viewer)

    # Hide the main napari window at startup so only the file browser shows;
    # revealed when the first file is opened into it.
    def _show_main_viewer():
        try:
            viewer.window._qt_window.show()
            viewer.window._qt_window.raise_()
        except Exception:
            pass

    def _first_visible_viewer():
        for v in _viewers:
            if _viewer_is_alive(v) and v.window._qt_window.isVisible():
                return v
        return None

    def _viewer_is_alive(v):
        try:
            w = v.window._qt_window
            if w is None:
                return False
            w.isVisible()
            return True
        except Exception:
            return False

    def _recreate_main_viewer():
        # The main viewer was closed/destroyed; build a fresh in-process viewer
        # to stand in for it. Safe on every platform because no other in-process
        # canvas is live (the segfault only hits a *concurrent* 2nd viewer).
        new_viewer = napari.Viewer(title="helicon display")
        _viewers.append(new_viewer)
        _active_viewer[0] = new_viewer
        _track_viewer(new_viewer)
        _hide_layer_panels(new_viewer)
        _install_panel_toggle(new_viewer)
        try:
            if not isinstance(new_viewer.window, MagicMock):
                _install_viewer_save_menu(new_viewer)
                _SliceDirectionWidget(new_viewer).inject()
        except Exception:
            pass
        try:
            viewer.window._qt_window.hide()
        except Exception:
            pass
        return new_viewer

    try:
        from unittest.mock import MagicMock

        if not isinstance(viewer.window, MagicMock):
            viewer.window._qt_window.hide()
    except Exception:
        pass

    def _on_file_selected(path):
        target = _active_viewer[0]
        if target is None or not _viewer_is_alive(target):
            target = _recreate_main_viewer()
        if target is viewer or target is _active_viewer[0]:
            _show_main_viewer()
        _open_file(target, path)

    def _spawn_viewer_and_open(path, mode=None):
        """Open ``path`` in a new napari viewer.

        On macOS multiple in-process napari viewers coexist safely, so the
        new window is opened in-process and tracked by the focus logic
        (``_viewers`` / ``_active_viewer``), giving the same behaviour as
        the main viewer.

        On every other platform (notably Linux/Wayland), creating a second
        in-process napari viewer segfaults (VisPy's QOpenGLWidget crashes
        in glFlush of the 2nd canvas). There the new window is spawned in a
        separate process with its own isolated OpenGL context, so a crash
        there can never take down the main helicon process.
        """
        if sys.platform == "darwin":
            try:
                new_viewer = napari.Viewer(title=f"helicon - {os.path.basename(path)}")
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[helicon] failed to open new display window: {exc}")
                return
            _viewers.append(new_viewer)
            _active_viewer[0] = new_viewer
            _track_viewer(new_viewer)
            _hide_layer_panels(new_viewer)
            _install_panel_toggle(new_viewer)
            try:
                if not isinstance(new_viewer.window, MagicMock):
                    _install_viewer_save_menu(new_viewer)
                    _SliceDirectionWidget(new_viewer).inject()
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

    def _on_file_selected_new_window(path):
        _spawn_viewer_and_open(path)

    def _on_display_requested(path, mode, new_window):
        if mode == "chimerax":
            _launch_chimerax(path)
            return
        if mode in ("gallery", "optimiser"):
            reuse = None
            if not new_window:
                ag = _active_gallery[0]
                if ag is not None and _gallery_is_alive(ag):
                    reuse = ag

            _open_file(None, path, mode=mode, reuse_gallery=reuse)
            return
        # The "new window" checkbox only matters once a window is already
        # visible: it then forces a second window. Otherwise the file opens in
        # the existing viewer, revealing it if it was hidden at startup. A dead
        # or missing viewer is recreated so the buttons always respond.
        if new_window and _first_visible_viewer() is not None:
            _spawn_viewer_and_open(path, mode=mode)
        else:
            target = _active_viewer[0]
            if target is None or not _viewer_is_alive(target):
                target = _recreate_main_viewer()
            _show_main_viewer()
            _open_file(target, path, mode=mode)

    widget = FolderBrowserWidget(start_dir=start_dir)
    widget.file_selected.connect(_on_file_selected)
    widget.file_selected_new_window.connect(_on_file_selected_new_window)
    widget.display_requested.connect(_on_display_requested)
    widget.setWindowFlags(Qt.WindowType.Window)
    widget.setWindowTitle("helicon - Files")
    widget.show()

    try:
        from unittest.mock import MagicMock

        if not isinstance(widget, MagicMock):
            quit_sc = QShortcut(QKeySequence.StandardKey.Quit, widget)
            quit_sc.activated.connect(lambda: (viewer.close(), widget.close()))
    except Exception:
        pass

    _restore_geometry(widget, viewer)
    _install_save_hook(widget, viewer)

    napari.run()


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
