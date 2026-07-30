#!/usr/bin/env python

"""A file browser for viewing image, map, star, bild, eps, pdf, html, and text files"""

from __future__ import annotations

import sys
import time
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
    from helicon.lib.file_browser import FolderBrowserWidget
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


def _restore_geometry(dock, viewer):
    """Restore saved window sizes and reposition after the compositor places them."""
    from PySide6.QtCore import QTimer

    def _apply(attempt=0):
        settings = _get_qsettings()

        # Restore the viewer geometry if a viewer exists.
        if viewer is not None:
            try:
                qt_win = viewer.window._qt_window
            except AttributeError:
                qt_win = None
            if qt_win is not None and not qt_win.isVisible() and attempt < 10:
                QTimer.singleShot(50, lambda: _apply(attempt + 1))
                return
            if qt_win is not None:
                viewer_ba = settings.value("viewer_ba")
                if viewer_ba is not None:
                    try:
                        qt_win.restoreGeometry(viewer_ba)
                    except AttributeError:
                        pass

        # Restore the dock (browser) geometry independently.
        dock_ba = settings.value("dock_ba")
        if dock_ba is not None:
            try:
                dock.restoreGeometry(dock_ba)
            except (AttributeError, TypeError):
                pass
        elif viewer is not None:
            _position_default(dock, viewer)
        dock.show()

    QTimer.singleShot(0, _apply)


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


def _install_dock_save_hook(dock):
    """Install an event filter to save dock geometry when it closes independently."""
    from PySide6.QtCore import QEvent, QObject

    class _DockCloseFilter(QObject):
        """Saves only dock geometry when the dock closes independently."""

        def __init__(self, dock, parent=None):
            super().__init__(parent)
            self._dock = dock

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Close:
                settings = _get_qsettings()
                settings.setValue("dock_ba", self._dock.saveGeometry())
                save_cols = getattr(self._dock, "_save_col_widths", None)
                if callable(save_cols):
                    save_cols()
            return False

    dock_flt = _DockCloseFilter(dock, parent=dock)
    dock.installEventFilter(dock_flt)
    return dock_flt


def _install_viewer_save_hook(dock, viewer):
    """Install an event filter to save both viewer and dock geometry on viewer close."""
    from PySide6.QtCore import QEvent, QObject

    class _ViewerCloseFilter(QObject):
        """Saves both viewer and dock geometry when the viewer window closes."""

        def __init__(self, dock, viewer, parent=None):
            super().__init__(parent)
            self._dock = dock
            self._viewer = viewer

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Close:
                _save_geometry(self._dock, self._viewer)
            return False

    flt = _ViewerCloseFilter(dock, viewer, parent=viewer.window._qt_window)
    viewer.window._qt_window.installEventFilter(flt)
    return flt


def _save_geometry(dock, viewer):
    settings = _get_qsettings()
    qt_win = viewer.window._qt_window

    cached_ba = getattr(viewer, "_display_only_ba", None)
    viewer_ba = cached_ba if cached_ba is not None else qt_win.saveGeometry()
    settings.setValue("viewer_ba", viewer_ba)

    settings.setValue("dock_ba", dock.saveGeometry())

    save_cols = getattr(dock, "_save_col_widths", None)
    if callable(save_cols):
        save_cols()


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
    * Show a "Save Canvas As…" context menu on right-click press.
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

        def _show_save_menu():
            gpos = canvas.native.cursor().pos()
            menu = QMenu()
            menu.addAction("Save Canvas As…")
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
                QTimer.singleShot(0, _show_save_menu)

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

    from helicon.lib.file_browser import _find_chimerax

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


def _make_bookmark_query(tab_name: str, params: dict) -> dict:
    """Build a query_params dict that produces a Shiny bookmark URL.

    Returns a dict suitable for ``launch_shiny_app(query_params=...)`` that
    produces: ``?_inputs_&helicon_tab="TabName"&_values_&p=...``
    """
    import json

    return {
        "_inputs_": "",
        "helicon_tab": f'"{tab_name}"',
        "_values_": "",
        "p": json.dumps(params),
    }


def _launch_denovo3d(path: str) -> None:
    """Open a .mrcs file in the Helicon Lab Denovo3D tab via bookmark URL."""
    from pathlib import Path

    from PySide6.QtWidgets import QMessageBox

    from helicon.lib.shiny import launch_shiny_app

    app_file = "helicon.webApps.app:app"
    try:
        launch_shiny_app(
            app_file,
            block=False,
            query_params=_make_bookmark_query(
                "Denovo3D",
                {
                    "input_mode_images": "url",
                    "url_images": str(Path(path).resolve()),
                },
            ),
        )
    except Exception as exc:
        QMessageBox.critical(
            None,
            "Denovo3D Launch Error",
            f"Failed to launch Denovo3D:\n{exc}",
        )


def _launch_whereismyclass(path: str) -> None:
    """Open a star/cs file in the WhereIsMyClass tab via bookmark URL."""
    from pathlib import Path

    from PySide6.QtWidgets import QMessageBox

    from helicon.lib.shiny import launch_shiny_app

    app_file = "helicon.webApps.app:app"
    try:
        launch_shiny_app(
            app_file,
            block=False,
            query_params=_make_bookmark_query(
                "WhereIsMyClass",
                {
                    "input_mode": "url",
                    "url_star": str(Path(path).resolve()),
                },
            ),
        )
    except Exception as exc:
        QMessageBox.critical(
            None,
            "WhereIsMyClass Launch Error",
            f"Failed to launch WhereIsMyClass:\n{exc}",
        )


def _launch_helicalprojection(path: str) -> None:
    """Open a file in the HelicalProjection tab via bookmark URL."""
    from pathlib import Path

    from PySide6.QtWidgets import QMessageBox

    from helicon.lib.shiny import launch_shiny_app

    app_file = "helicon.webApps.app:app"
    try:
        launch_shiny_app(
            app_file,
            block=False,
            query_params=_make_bookmark_query(
                "HelicalProjection",
                {
                    "mode_images": "url",
                    "url_images": str(Path(path).resolve()),
                },
            ),
        )
    except Exception as exc:
        QMessageBox.critical(
            None,
            "HelicalProjection Launch Error",
            f"Failed to launch HelicalProjection:\n{exc}",
        )


def _launch_helicalpitch(path: str) -> None:
    """Open a file in the HelicalPitch tab via bookmark URL.

    Derives the companion file (star↔mrcs) from the given path when it
    exists on disk, so both params and class images are loaded together.
    """
    import re
    from pathlib import Path

    from PySide6.QtWidgets import QMessageBox

    from helicon.lib.shiny import launch_shiny_app

    app_file = "helicon.webApps.app:app"
    try:
        file_path = Path(path).resolve()
        suffix = file_path.suffix.lower()

        bookmark = {
            "mode_params": "url",
            "mode_classes": "url",
        }

        if suffix in (".star", ".cs"):
            bookmark["url_params"] = str(file_path)
            iter_match = re.search(r"run_it(\d+)", file_path.name)
            if iter_match:
                mrcs_file = (
                    file_path.parent / f"run_it{iter_match.group(1)}_classes.mrcs"
                )
                if mrcs_file.exists():
                    bookmark["url_classes"] = str(mrcs_file)
        else:
            bookmark["url_classes"] = str(file_path)
            iter_match = re.search(r"run_it(\d+)", file_path.name)
            if iter_match:
                star_file = file_path.parent / f"run_it{iter_match.group(1)}_data.star"
                if star_file.exists():
                    bookmark["url_params"] = str(star_file)

        launch_shiny_app(
            app_file,
            block=False,
            query_params=_make_bookmark_query("HelicalPitch", bookmark),
        )
    except Exception as exc:
        QMessageBox.critical(
            None,
            "HelicalPitch Launch Error",
            f"Failed to launch HelicalPitch:\n{exc}",
        )


def _launch_hill(path: str) -> None:
    """Open a file in the HILL tab of the consolidated Helicon Lab web app.

    Launches the unified Shiny app and selects the HILL tab with the given file.
    """
    from PySide6.QtWidgets import QMessageBox
    from helicon.lib.shiny import launch_shiny_app

    try:
        params = _make_bookmark_query("HILL", {})
        params["input_mode"] = "2"
        params["img_file_url"] = path
        launch_shiny_app("helicon.webApps.app:app", block=False, query_params=params)
    except Exception as exc:
        QMessageBox.critical(
            None,
            "HILL Launch Error",
            f"Failed to launch HILL:\n{exc}",
        )


def _launch_hi3d(path: str) -> None:
    """Open a file in the HI3D tab of the consolidated Helicon Lab web app.

    Launches the unified Shiny app and selects the HI3D tab with the given file.
    """
    from PySide6.QtWidgets import QMessageBox
    from helicon.lib.shiny import launch_shiny_app

    try:
        params = _make_bookmark_query("HI3D", {})
        params["img_file_url"] = path
        launch_shiny_app("helicon.webApps.app:app", block=False, query_params=params)
    except Exception as exc:
        QMessageBox.critical(
            None,
            "HI3D Launch Error",
            f"Failed to launch HI3D:\n{exc}",
        )


def _launch_truefsc(path: str, parent=None) -> None:
    """Compute True FSC from the two half-maps referenced by a model.star file."""
    import logging
    import os
    import re
    import tempfile
    from pathlib import Path

    from PySide6.QtCore import QThread, Signal
    from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QTextEdit, QVBoxLayout

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

    if os.access(model_dir, os.W_OK):
        output_dir = model_dir
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="helicon_truefsc_"))

    plot_file = output_dir / "trueFSC.pdf"

    from helicon.commands.trueFSC import compute_truefsc

    class LogHandler(logging.Handler):
        def __init__(self, signal):
            super().__init__()
            self._signal = signal

        def emit(self, record):
            msg = self.format(record)
            self._signal.emit(msg)

    class Worker(QThread):
        line_received = Signal(str)
        finished = Signal(object)
        error = Signal(str)

        def __init__(self):
            super().__init__()

        def run(self):
            try:
                self.line_received.emit(f"Map 1: {map1}")
                self.line_received.emit(f"Map 2: {map2}")
                self.line_received.emit(f"Output: {plot_file}")
                self.line_received.emit("")

                handler = LogHandler(self.line_received)
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger = logging.getLogger("helicon.commands.trueFSC")
                logger.addHandler(handler)
                logger.setLevel(logging.DEBUG)
                try:
                    result = compute_truefsc(
                        str(map1),
                        str(map2),
                        str(plot_file),
                    )
                finally:
                    logger.removeHandler(handler)
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))

    class ProgressDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("trueFSC")
            self.setMinimumSize(500, 300)
            layout = QVBoxLayout(self)

            self.label = QLabel("Running trueFSC...")
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
            if result and result.get("plot_file"):
                self.label.setText(
                    f"True FSC completed - Resolution: {result['resolution']:.2f} A"
                )
                viewer = _napari.active()
                if viewer is None:
                    viewer = _create_napari_viewer()
                _open_file(viewer, str(result["plot_file"]), mode="slice")
            else:
                self.label.setText("trueFSC completed")
            if output_dir != model_dir:
                self.text_edit.append(f"\nResults saved to: {output_dir}")
            self.close_btn.setEnabled(True)

        def set_error(self, error_msg):
            self.label.setText("trueFSC failed")
            self.text_edit.append(f"\nError: {error_msg}")
            self.close_btn.setEnabled(True)

    dialog = ProgressDialog(parent)
    worker = Worker()
    worker.line_received.connect(dialog.append_line)
    worker.finished.connect(dialog.set_result)
    worker.error.connect(dialog.set_error)
    worker.start()
    dialog.exec()


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

        # VisPy reports middle as 3 (its own enum), Qt reports it as 4.
        # VisPy right button is 2 — do NOT include it here.
        middle_values = (3, 4)

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


def _create_napari_viewer(title="helicon display"):
    """Create a new napari viewer with standard helicon customizations.

    Sets up the ``_SliceDirectionWidget`` and hides the layer panels so
    the viewer matches the default helicon look.  Raises
    ``HeliconDependencyError`` if napari or OpenGL is unavailable.
    """
    from unittest.mock import MagicMock

    import napari

    try:
        new_viewer = napari.Viewer(title=title)
    except Exception as exc:
        raise HeliconDependencyError(
            f"Failed to create the napari viewer: {exc}\n"
            "This can happen when no OpenGL-accelerated display is "
            "available.\nTry setting QT_QPA_PLATFORM=offscreen or "
            "updating your GPU drivers."
        ) from exc
    _napari.register(new_viewer)
    _hide_layer_panels(new_viewer)
    try:
        if not isinstance(new_viewer.window, MagicMock):
            _SliceDirectionWidget(new_viewer).inject()
    except Exception:
        pass
    return new_viewer


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


def _open_text_window(path, reuse_window=None):
    """Open a text file in a standalone text window."""
    from pathlib import Path

    try:
        from PySide6.QtWidgets import QMainWindow, QTextEdit
        from PySide6.QtGui import QFont
    except ImportError:
        return

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return

    if reuse_window is not None:
        try:
            if reuse_window.isVisible():
                reuse_window._text_edit.setPlainText(content)
                reuse_window.setWindowTitle(f"helicon - {Path(path).name}")
                reuse_window.show()
                reuse_window.raise_()
                return reuse_window
        except Exception:
            pass

    class _TextWindow(QMainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)
            from PySide6.QtWidgets import (
                QWidget,
                QVBoxLayout,
                QHBoxLayout,
                QLineEdit,
                QToolButton,
            )
            from PySide6.QtGui import QShortcut, QKeySequence, QTextCursor
            from PySide6.QtCore import Qt

            self.setWindowTitle(f"helicon - {Path(path).name}")
            self.resize(700, 500)

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            te = QTextEdit(self)
            te.setReadOnly(True)
            te.setFont(QFont("Courier New", 12))
            te.setLineWrapMode(QTextEdit.WidgetWidth)
            te.setStyleSheet("background-color: #2d2d2d; color: #cccccc; border: none;")
            self._text_edit = te
            layout.addWidget(te, 1)

            find_bar = QWidget()
            find_bar.setStyleSheet(
                "background-color: #3c3c3c; border-top: 1px solid #555;"
            )
            find_layout = QHBoxLayout(find_bar)
            find_layout.setContentsMargins(6, 4, 6, 4)
            find_layout.setSpacing(6)

            find_input = QLineEdit()
            find_input.setPlaceholderText("Find…")
            find_input.setStyleSheet(
                "background-color: #2d2d2d; color: #cccccc; "
                "border: 1px solid #555; border-radius: 3px; padding: 3px;"
            )
            find_input.returnPressed.connect(self._find_next)
            self._find_input = find_input

            close_btn = QToolButton()
            close_btn.setText("✕")
            close_btn.setToolTip("Close find bar")
            close_btn.clicked.connect(lambda: self._toggle_find_bar(False))
            close_btn.setStyleSheet(
                "QToolButton { background: transparent; border: none; "
                "color: #cccccc; padding: 2px 6px; }"
                "QToolButton:hover { color: #ffffff; }"
            )

            find_layout.addWidget(find_input, 1)
            find_layout.addWidget(close_btn)
            self._find_bar = find_bar
            self._find_bar_visible = False
            find_bar.hide()
            layout.addWidget(find_bar)

            self.setCentralWidget(central)

            find_sc = QShortcut(QKeySequence.StandardKey.Find, self)
            find_sc.activated.connect(lambda: self._toggle_find_bar(True))

            esc_sc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
            esc_sc.activated.connect(self._close_find_bar)

            wrap_sc = QShortcut(QKeySequence("Ctrl+Shift+W"), self)
            wrap_sc.activated.connect(self._toggle_wrap)

            _install_window_shortcuts(self)

        def _toggle_find_bar(self, show: bool) -> None:
            if show:
                self._find_bar.show()
                self._find_input.setFocus()
                self._find_input.selectAll()
                self._find_bar_visible = True
            else:
                self._find_bar.hide()
                self._find_bar_visible = False
                self._text_edit.setFocus()

        def _close_find_bar(self) -> None:
            if self._find_bar_visible:
                self._toggle_find_bar(False)

        def _find_next(self) -> None:
            text = self._find_input.text()
            if not text:
                return
            from PySide6.QtGui import QTextCursor, QTextDocument

            found = self._text_edit.find(
                text, QTextDocument.FindFlag.FindCaseSensitively
            )
            if not found:
                cursor = self._text_edit.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                self._text_edit.setTextCursor(cursor)
                self._text_edit.find(text, QTextDocument.FindFlag.FindCaseSensitively)

        def _toggle_wrap(self) -> None:
            current = self._text_edit.lineWrapMode()
            if current == QTextEdit.NoWrap:
                self._text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
                self.statusBar().showMessage("Word wrap: ON", 2000)
            else:
                self._text_edit.setLineWrapMode(QTextEdit.NoWrap)
                self.statusBar().showMessage("Word wrap: OFF", 2000)

        def closeEvent(self, event):
            _text.on_close(self)
            super().closeEvent(event)

        def changeEvent(self, event):
            from PySide6.QtCore import QEvent

            if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
                _text.on_activate(self)
            super().changeEvent(event)

    win = _TextWindow()
    win._text_edit.setPlainText(content)
    _text.register(win)
    win.show()
    return win


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
    max_w, max_h = 0, 0
    for i in range(n_pages):
        pt_size = doc.pagePointSize(i)
        w_px = int(pt_size.width() * dpi / 72)
        h_px = int(pt_size.height() * dpi / 72)
        max_w = max(max_w, w_px)
        max_h = max(max_h, h_px)

    for i in range(n_pages):
        pt_size = doc.pagePointSize(i)
        w_px = int(pt_size.width() * dpi / 72)
        h_px = int(pt_size.height() * dpi / 72)
        scale = min(max_w / w_px, max_h / h_px)
        rw = int(w_px * scale)
        rh = int(h_px * scale)
        img = doc.render(i, QSize(rw, rh))
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        ptr = img.bits()
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(
            img.height(), img.width(), 4
        )
        rgb = arr[:, :, 2::-1].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        composite = alpha * rgb + (1.0 - alpha) * 255.0
        canvas = np.full((max_h, max_w, 3), 255.0, dtype=np.float32)
        y0 = (max_h - rh) // 2
        x0 = (max_w - rw) // 2
        canvas[y0 : y0 + rh, x0 : x0 + rw] = composite
        pages.append(canvas)

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
    _hide_layer_panels(viewer)
    if len(pages) > 1:
        step = list(viewer.dims.current_step)
        step[0] = 0
        viewer.dims.current_step = step


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


def _parse_class2d_model_star(
    model_path: str,
) -> tuple[list[tuple[str, int]], list[float]] | None:
    """Extract MRC references and class distributions from a RELION model.star.

    Reads the ``data_model_classes`` section and resolves
    ``_rlnReferenceImage`` (which may be ``idx@path.mrcs`` for Class2D)
    and ``_rlnClassDistribution`` (abundance) columns.

    Returns
    -------
    tuple of (entries, distributions) or None
        * ``entries``: list of ``(mrc_path, frame_idx)`` tuples.
        * ``distributions``: list of class distribution values (0-1).
    """
    from pathlib import Path

    model_dir = Path(model_path).parent
    in_loop = False
    in_data_model_classes = False
    col_names: list[str] = []
    ref_image_col_idx = -1
    class_dist_col_idx = -1
    entries: list[tuple[str, int]] = []
    distributions: list[float] = []

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
                    class_dist_col_idx = -1
                    continue

                if s == "loop_":
                    in_loop = True
                    col_names = []
                    ref_image_col_idx = -1
                    class_dist_col_idx = -1
                    continue

                if in_loop and s.startswith("_"):
                    col_name = s.split()[0]
                    col_names.append(col_name)
                    if "referenceimage" in col_name.lower():
                        ref_image_col_idx = len(col_names) - 1
                    elif "classdistribution" in col_name.lower():
                        class_dist_col_idx = len(col_names) - 1
                    continue

                if not in_data_model_classes:
                    continue

                if not in_loop:
                    continue

                parts = s.split()
                if ref_image_col_idx < 0 or ref_image_col_idx >= len(parts):
                    continue

                ref_raw = parts[ref_image_col_idx]
                frame_idx = 0
                if "@" in ref_raw:
                    idx_str, file_part = ref_raw.split("@", 1)
                    frame_idx = int(idx_str) - 1
                else:
                    file_part = ref_raw

                resolved = None
                for ancestor in [model_dir] + list(model_dir.parents):
                    candidate = ancestor / file_part
                    if candidate.is_file():
                        resolved = str(candidate)
                        break

                if not resolved:
                    continue

                dist = 0.0
                if class_dist_col_idx >= 0 and class_dist_col_idx < len(parts):
                    try:
                        dist = float(parts[class_dist_col_idx])
                    except ValueError:
                        dist = 0.0

                entries.append((resolved, frame_idx))
                distributions.append(dist)
    except Exception:
        return None

    if not entries:
        return None

    return entries, distributions


def _parse_star_image_refs(
    star_path: str,
) -> tuple[list[tuple[int, str, float]], tuple, float, int] | None:
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
    tuple of (entries, first_shape, first_apix, n_skipped) or None
        * ``entries``: list of ``(frame_idx_0based, mrc_path, 0.0)`` tuples.
        * ``first_shape``: ``(nx, ny)`` or ``(nx, ny, nz)`` of the first image.
        * ``first_apix``: pixel size in Angstroms (fallback 1.0).
        * ``n_skipped``: number of data lines whose binary images could not
          be found on disk.
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
    n_skipped = 0
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
                    n_skipped += 1
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
        if n_skipped:
            return [], first_shape or (0, 0), first_apix, n_skipped
        return None

    return entries, first_shape, first_apix, n_skipped


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


class _DisplayTracker:
    """Track alive windows of a single display category.

    Each tracked window carries a ``time.monotonic()`` timestamp that
    records when it was created or last activated (brought to front).
    ``active()`` returns the window with the most recent timestamp.

    Timeline of approaches tried (for the record):

    1. ``owns_fn`` + ``app.focusChanged`` → ``on_focus()``
       Problem: gallery windows display non-focusable image widgets, so
       clicking them never changes the application *focus widget* — the
       file browser's QTreeView keeps focus.  ``on_focus`` never matched.

    2. ``changeEvent`` with ``QEvent.Type.WindowActivate``
       Problem: this event type is never delivered on macOS.

    3. ``changeEvent`` with ``QEvent.Type.ActivationChange`` + ``isActiveWindow()``
       Works on all platforms.  When a window is brought to front Qt sends
       ``ActivationChange``, and ``isActiveWindow()`` tells us whether
       this window is now the active one.
    """

    def __init__(self, is_alive):
        self._windows: dict = {}  # window → monotonic timestamp
        self._is_alive = is_alive

    def alive(self) -> list:
        """Prune dead windows and return the living ones (newest first)."""
        self._windows = {w: t for w, t in self._windows.items() if self._is_alive(w)}
        return sorted(self._windows, key=self._windows.get, reverse=True)

    def active(self):
        """Return the most-recent alive window, or ``None``."""
        alive = self.alive()
        return alive[0] if alive else None

    def register(self, window):
        """Track a new window with current timestamp."""
        self._windows[window] = time.monotonic()

    def on_activate(self, window):
        """Update the timestamp when *window* is activated (brought to front)."""
        if window in self._windows:
            self._windows[window] = time.monotonic()

    def on_close(self, window):
        """Remove a closed window from tracking."""
        self._windows.pop(window, None)


# ---------------------------------------------------------------------------
# Alive-check helpers used by the trackers
# ---------------------------------------------------------------------------


def _is_alive_viewer(v):
    try:
        w = v.window._qt_window
        return w is not None and w.isVisible()
    except Exception:
        return False


def _is_alive_widget(w):
    try:
        return w is not None and w.isVisible()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-category trackers (module-level so gallery_backends can reference them)
# ---------------------------------------------------------------------------

_napari = _DisplayTracker(_is_alive_viewer)
_gallery = _DisplayTracker(_is_alive_widget)
_plot = _DisplayTracker(_is_alive_widget)
_text = _DisplayTracker(_is_alive_widget)

_NAPARI_MODES = {"slice", "volume", "stats", "3dplot"}
_GALLERY_MODES = {"gallery", "optimiser", "2dclasses", "orthogonal"}
_TEXT_MODES = {"text"}
_PLOT_MODES = {"fsc"}


def _quit_all_windows():
    """Close every tracked window and the file browser, then quit."""
    from PySide6.QtWidgets import QApplication

    for tracker in (_napari, _gallery, _text, _plot):
        for w in list(tracker.alive()):
            try:
                w.close()
            except Exception:
                pass
    for w in QApplication.topLevelWidgets():
        try:
            w.close()
        except Exception:
            pass


def _install_window_shortcuts(window):
    """Install Ctrl+W (close) and Ctrl+Q (quit) on *window*."""
    from PySide6.QtGui import QShortcut, QKeySequence

    close_sc = QShortcut(QKeySequence("Ctrl+W"), window)
    close_sc.activated.connect(window.close)

    quit_sc = QShortcut(QKeySequence.StandardKey.Quit, window)
    quit_sc.activated.connect(_quit_all_windows)


_TRACKER_FOR: dict[str, _DisplayTracker] = {}
for _m in _NAPARI_MODES:
    _TRACKER_FOR[_m] = _napari
for _m in _GALLERY_MODES:
    _TRACKER_FOR[_m] = _gallery
for _m in _TEXT_MODES:
    _TRACKER_FOR[_m] = _text
for _m in _PLOT_MODES:
    _TRACKER_FOR[_m] = _plot


def _wrap_gallery_with_panel(gallery: "ImageGalleryWidget") -> "QWidget":
    """Wrap an ImageGalleryWidget with a left-side _ControlPanel sibling.

    The panel is prepended to the left.  Toggling it grows the parent
    window leftward by ``_ControlPanel.PANEL_WIDTH`` so the gallery
    widget keeps both its width and its screen position unchanged.
    """
    from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

    from helicon.lib.gallery_widget import _ControlPanel

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
    from helicon.lib.gallery_backends import StackGallery

    gallery = StackGallery(
        star_path=name,
        read_fn=read_fn,
        n=n,
        img_w=img_w,
        img_h=img_h,
        apix=apix,
    )
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _open_fsc_plot(star_path: str, reuse_window=None) -> None:
    """Display the FSC curve from a RELION model.star file using pyqtgraph.

    Reads ``data_model_class_N`` sections and plots resolution vs. FSC
    for each class.  A horizontal line at FSC = 0.143 marks the
    gold-standard threshold.
    """
    import numpy as np

    try:
        import pyqtgraph as pg
        from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
        from PySide6.QtCore import Qt
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
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(
            None,
            "Error",
            f"Failed to read {Path(star_path).name}.\n"
            "Make sure it is a valid RELION model.star file.",
        )
        return

    class_sections = {k: v for k, v in data.items() if k.startswith("model_class_")}
    if not class_sections:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(
            None,
            "No FSC data",
            f"No data_model_class_N sections found in {Path(star_path).name}.",
        )
        return

    all_curves = []
    for key in sorted(class_sections, key=lambda k: int(k.split("_")[-1])):
        df = class_sections[key]
        class_num = key.split("_")[-1]

        sf_col = None
        ang_col = None
        fsc_col = None
        for col in df.columns:
            cl = col.lower()
            if "goldstandardfsc" in cl:
                fsc_col = col
            elif "fouriershellcorrelation" in cl and "phase" not in cl:
                fsc_col = col
            if cl in ("_rlnresolution", "rlnresolution"):
                sf_col = col
            if cl in ("_rlnangstromresolution", "rlnangstromresolution"):
                ang_col = col

        if sf_col is None or fsc_col is None:
            continue

        spatial_freq = np.asarray(df[sf_col], dtype=np.float64)
        fsc = np.asarray(df[fsc_col], dtype=np.float64)

        order = np.argsort(spatial_freq)
        spatial_freq = spatial_freq[order]
        fsc = fsc[order]

        angstrom = None
        if ang_col is not None:
            angstrom = np.asarray(df[ang_col], dtype=np.float64)[order]

        all_curves.append((spatial_freq, fsc, f"Class {class_num}", angstrom))

    if not all_curves:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(
            None,
            "Column error",
            "Could not find resolution/FSC columns in any class section.",
        )
        return

    name = Path(star_path).name

    central = QWidget()
    layout = QVBoxLayout(central)
    layout.setContentsMargins(0, 0, 0, 0)

    pg.setConfigOptions(antialias=True)
    plot_widget = pg.PlotWidget()
    layout.addWidget(plot_widget)

    plot = plot_widget.getPlotItem()
    plot.getAxis("bottom").enableAutoSIPrefix(False)
    plot.setLabel("bottom", "Resolution", units="1/Å")
    plot.setLabel("left", "FSC")
    plot.setYRange(0, 1.05)
    plot.addLegend()
    plot.showGrid(x=True, y=True, alpha=0.3)

    top_axis = plot.getAxis("top")
    top_axis.enableAutoSIPrefix(False)
    top_axis.setLabel("Resolution", units="Å")

    def _angstrom_tickStrings(values, scale, spacing):
        return [f"{1.0 / v:.1f}" if v > 0 else "" for v in values]

    top_axis.tickStrings = _angstrom_tickStrings
    top_axis.show()

    colors = [
        (0, 120, 215),
        (220, 120, 0),
        (0, 170, 80),
        (180, 0, 180),
        (200, 50, 50),
        (100, 100, 100),
    ]

    for i, (spatial_freq, fsc, label, _) in enumerate(all_curves):
        color = colors[i % len(colors)]
        pen = pg.mkPen(color=color, width=2)
        plot.plot(spatial_freq, fsc, pen=pen, name=label)

    threshold_pen = pg.mkPen(color=(220, 50, 50), width=1, style=Qt.PenStyle.DashLine)
    plot.addItem(
        pg.InfiniteLine(
            pos=0.143,
            angle=0,
            pen=threshold_pen,
        )
    )
    threshold_label = pg.TextItem("0.143", color=(220, 50, 50), anchor=(0, 0))
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
    plot.addItem(vline, ignoreBounds=True)
    plot.addItem(hline, ignoreBounds=True)
    coord_text = pg.TextItem(anchor=(0, 1), color=(220, 220, 220))
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
            f'<span style="color: #dcdcdc; background-color: rgba(0,0,0,180);'
            f' padding: 2px;">'
            f"{x:.4f} 1/Å  ({ang} Å)<br>FSC = {y:.4f}</span>"
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

    if reuse_window is not None and _is_alive_widget(reuse_window):
        reuse_window.setWindowTitle(f"FSC — {name}")
        reuse_window.setCentralWidget(central)
        reuse_window.resize(700, 450)
        reuse_window.show()
        reuse_window.raise_()
        return

    class _FscWindow(QMainWindow):
        def closeEvent(self, event):
            _plot.on_close(self)
            super().closeEvent(event)

        def changeEvent(self, event):
            from PySide6.QtCore import QEvent

            if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
                _plot.on_activate(self)
            super().changeEvent(event)

    win = _FscWindow()
    win.setWindowTitle(f"FSC — {name}")
    win.setCentralWidget(central)
    win.resize(700, 450)
    _plot.register(win)
    _install_window_shortcuts(win)
    win.show()


def _open_xyz_slice_gallery(
    star_path: str, reuse_window=None, tracker=None
) -> "QMainWindow | None":
    """Display center slices (Z, Y, X) of MRC files referenced by a star file.

    Delegates to ``Class3dGallery`` (with abundance labels) or
    ``Refine3dGallery`` (without) based on the path.
    """
    from helicon.lib.gallery_backends import Class3dGallery, Refine3dGallery

    name = Path(star_path).name
    is_refine = any(p.startswith("Refine3D") for p in Path(star_path).parts)
    gallery = Refine3dGallery(star_path) if is_refine else Class3dGallery(star_path)
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _open_orthogonal_viewer(
    mrc_path: str, reuse_window=None, tracker=None
) -> "QMainWindow | None":
    """Open an interactive orthogonal slice viewer for a 3D MRC/MAP file."""
    from helicon.lib.gallery_backends import OrthogonalGallery

    gallery = OrthogonalGallery(mrc_path)
    return gallery.open(reuse_window=reuse_window, tracker=tracker)


def _find_model_star_from_optimiser(optimiser_path: str) -> str | None:
    """Find the model.star referenced by an optimiser.star file."""
    from pathlib import Path

    star_dir = Path(optimiser_path).parent

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
                                return str(candidate)
    except Exception:
        pass
    return None


def _open_2d_classes_gallery(
    star_path: str, reuse_window=None, tracker=None
) -> "QMainWindow | None":
    """Display 2D class averages from a Class2D model.star.

    Shows one MRC per class (``_rlnReferenceImage``) with abundance labels
    (``_rlnClassDistribution``). Sort-by-abundance and reverse-sort controls
    are provided in the control panel.
    """
    from helicon.lib.gallery_backends import Class2dGallery

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
    import os

    import napari

    _patch_napari_value_bug()

    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

    viewer = _create_napari_viewer(title=Path(path).name)
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
        from pathlib import Path

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
                new_viewer = _create_napari_viewer(title=f"helicon - {Path(path).name}")
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
            _launch_denovo3d(path)
            return
        if mode == "whereIsMyClass":
            _launch_whereismyclass(path)
            return
        if mode == "helicalProjection":
            _launch_helicalprojection(path)
            return
        if mode == "helicalPitch":
            _launch_helicalpitch(path)
            return
        if mode == "hill":
            _launch_hill(path)
            return
        if mode == "hi3d":
            _launch_hi3d(path)
            return
        if mode == "trueFSC":
            _launch_truefsc(path, parent=widget)
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
            _install_window_shortcuts(widget)
    except Exception:
        pass

    _install_dock_save_hook(widget)
    _restore_geometry(widget, None)

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
