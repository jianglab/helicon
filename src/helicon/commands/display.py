#!/usr/bin/env python

"""Open a napari viewer with a floating folder browser for viewing images, volumes, PDFs, and text files."""

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

import helicon
from helicon.lib.exceptions import HeliconDependencyError

try:
    from helicon.lib.napari_widgets import FolderBrowserWidget
except ImportError:
    FolderBrowserWidget = None


def _get_qsettings():
    from PySide6.QtCore import QSettings

    return QSettings("helicon", "display")


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

        viewer_geo = _read_rect(settings, "viewer")
        if viewer_geo is not None:
            x, y, w, h = viewer_geo
            try:
                qt_win.resize(w, h)
                qt_win.move(x, y)
            except AttributeError:
                pass

        dock_geo = _read_rect(settings, "dock")
        _position_default(dock, viewer)
        if dock_geo is not None:
            _, _, w, h = dock_geo
            dock.resize(w, h)
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
        napari_rect = viewer.window._qt_window.geometry()
        dock.adjustSize()
        dock_rect = dock.geometry()
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
    _write_rect(settings, "dock", dock)
    _write_rect(settings, "viewer", viewer.window._qt_window)
    # Persist file-browser column widths (backup for the widget closeEvent).
    save_cols = getattr(dock, "_save_col_widths", None)
    if callable(save_cols):
        save_cols()


def _write_rect(settings, prefix, widget):
    try:
        geo = widget.geometry()
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
    """Replaces axis labels with a Z/Y/X dropdown at the right end of each slider."""

    _combos: list = []

    def __init__(self, viewer):
        self._viewer = viewer

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


def _is_text_file(path: str) -> bool:
    """Check if a file is a text file by attempting UTF-8 decode."""
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
        overlay.setFont(QFont("Courier", 12))
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
    overlay.setWindowTitle(name)


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
    layer = viewer.add_image(
        data,
        name=name,
        contrast_limits=contrast,
    )
    _enable_continuous_auto_contrast(layer, viewer)
    layer.contrast_limits_range = (float(data.min()), float(data.max()))


def _open_bild(viewer, path: str) -> None:
    from pathlib import Path
    import numpy as np

    paths = []
    colors = []
    widths = []
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
                w = float(parts[7])
                paths.append([[x1, y1, z1], [x2, y2, z2]])
                colors.append(current_color)
                widths.append(w)
            elif line.startswith(".sphere"):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r = float(parts[4])
                paths.append([[x - r, y, z], [x + r, y, z]])
                colors.append(current_color)
                widths.append(r * 2)

    if not paths:
        return

    edge_colors = np.array(colors)
    name = Path(path).name
    layer = viewer.add_shapes(
        paths,
        shape_type="path",
        name=name,
        edge_width=widths,
        edge_color=edge_colors,
    )
    viewer.dims.ndisplay = 3


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


def _open_file(viewer, path: str) -> None:
    from pathlib import Path

    qt_window = viewer.window._qt_window
    if hasattr(qt_window, "_text_overlay") and qt_window._text_overlay.isVisible():
        qt_window._text_overlay.hide()

    ext = Path(path).suffix.lower()

    if ext == ".star" and _is_metadata_star(path):
        _open_text(viewer, path)
        return

    if ext == ".star":
        import starfile
        import mrcfile

        df = starfile.read(path, always_dict=True)
        star_dir = Path(path).parent

        image_name_col = None
        image_df = None
        for block_name, val in df.items():
            if block_name == "optics":
                continue
            for col in val.columns:
                if "ImageName" in col or "MicrographName" in col:
                    image_name_col = col
                    image_df = val
                    break
            if image_name_col:
                break

        if image_name_col is None:
            return

        entries = []
        first_apix = 1.0
        first_shape = None

        for name in image_df[image_name_col]:
            name = str(name)
            if "@" in name:
                idx_str, mrc_path = name.split("@", 1)
            else:
                idx_str, mrc_path = "1", name

            mrc_path_resolved = None
            for ancestor in [star_dir] + list(star_dir.parents):
                candidate = ancestor / mrc_path
                if candidate.is_file():
                    mrc_path_resolved = candidate
                    break

            if mrc_path_resolved is None:
                continue

            frame_idx = int(idx_str) - 1
            entries.append((frame_idx, str(mrc_path_resolved), 0.0))

            if first_shape is None:
                with mrcfile.open(str(mrc_path_resolved), header_only=True) as mrc:
                    nx = int(mrc.header.nx)
                    ny = int(mrc.header.ny)
                    nz = int(mrc.header.nz)
                    first_shape = (
                        (nx, ny)
                        if nz == 1 or mrc_path_resolved.suffix.lower() == ".mrcs"
                        else (nx, ny, nz)
                    )
                    first_apix = float(mrc.voxel_size.x)
                    if first_apix <= 0:
                        first_apix = 1.0

        if not entries:
            return

        n = len(entries)
        with mrcfile.open(entries[0][1], permissive=True) as mrc:
            frame = mrc.data[entries[0][0]] if mrc.data.ndim >= 3 else mrc.data
            dtype = frame.dtype

        stack_shape = (n,) + first_shape
        lazy = _LazyStarStack(entries, stack_shape, dtype)

        # Continuous auto-contrast: napari recomputes contrast limits from the
        # currently displayed slice on every navigation, so each frame is
        # contrasted independently and only the visible frame is read (lazy).
        # We deliberately do NOT pass contrast_limits -- setting it flips
        # napari's _keep_auto_contrast off and disables continuous mode.
        name = Path(path).name
        layer = viewer.add_image(
            lazy,
            name=name,
            scale=(1.0,) + (first_apix,) * len(first_shape),
        )
        _enable_continuous_auto_contrast(layer, viewer)
        viewer.dims.ndisplay = 2
        # napari defaults the shown slice to the middle frame; start at frame 0.
        step = list(viewer.dims.current_step)
        step[0] = 0
        viewer.dims.current_step = step
        return

    if ext == ".bild":
        _open_bild(viewer, path)
        return

    if ext == ".pdf":
        _open_pdf(viewer, path)
        return

    if ext in (".html", ".htm"):
        _open_html(viewer, path)
        return

    if _is_text_file(path):
        _open_text(viewer, path)
        return

    if ext in _MRC_EXTENSIONS:
        import mrcfile
        import numpy as np

        with mrcfile.open(path) as mrc:
            data = np.array(mrc.data)
            apix = float(mrc.voxel_size.x)
            if apix <= 0:
                try:
                    apix = float(mrc.header.cella.x) / int(mrc.header.nx)
                except (AttributeError, ZeroDivisionError):
                    apix = 1.0
            if apix <= 0:
                apix = 1.0

            if ext == ".mrcs":
                pass
            elif data.ndim >= 3 and int(mrc.header.nz) > 1:
                data, _ = helicon.change_map_axes_order(data, mrc.header)
            else:
                if data.ndim >= 3:
                    data = data[0]

        name = Path(path).name
        contrast = _auto_contrast(data)
        layer = viewer.add_image(
            data,
            name=name,
            scale=(apix, apix) if data.ndim == 2 else (apix, apix, apix),
            contrast_limits=contrast,
        )
        _enable_continuous_auto_contrast(layer, viewer)
        layer.contrast_limits_range = (float(data.min()), float(data.max()))

        if data.ndim >= 3 and len(viewer.dims.current_step) > 0:
            if ext == ".mrcs":
                step = list(viewer.dims.current_step)
                step[0] = 0
                viewer.dims.current_step = tuple(step)
                viewer.dims.ndisplay = 2
            else:
                step = list(viewer.dims.current_step)
                step[0] = data.shape[0] // 2
                viewer.dims.current_step = tuple(step)
                dims = data.shape
                viewer.dims.ndisplay = 3 if dims[0] * dims[1] * dims[2] < 512**3 else 2
    else:
        n_before = len(viewer.layers)
        viewer.open(path)
        for layer in viewer.layers[n_before:]:
            _enable_continuous_auto_contrast(layer, viewer)


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

    import napari

    _patch_napari_value_bug()

    start_dir = args.folder if args.folder else os.getcwd()

    viewer = napari.Viewer(title="helicon display")
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
                flt = _CloseFilter(v, parent=qt_window)
                qt_window.installEventFilter(flt)
        except Exception:
            pass

    _track_viewer(viewer)

    try:
        from unittest.mock import MagicMock

        if not isinstance(viewer.window, MagicMock):
            slice_widget = _SliceDirectionWidget(viewer)
            slice_widget.inject()
    except Exception:
        pass

    try:
        from unittest.mock import MagicMock

        if not isinstance(viewer.window, MagicMock):
            qv = viewer.window._qt_viewer
            layer_list = qv.dockLayerList
            layer_controls = qv.dockLayerControls
            layer_list.hide()
            layer_controls.hide()

            from PySide6.QtCore import QEvent, QObject

            class _MiddleClickFilter(QObject):
                def __init__(self, layer_list, layer_controls, parent=None):
                    super().__init__(parent)
                    self._layer_list = layer_list
                    self._layer_controls = layer_controls

                def eventFilter(self, obj, event):
                    if (
                        event.type() == QEvent.MouseButtonPress
                        and event.button() == Qt.MouseButton.MiddleButton
                    ):
                        if self._layer_list.isVisible():
                            self._layer_list.hide()
                            self._layer_controls.hide()
                        else:
                            self._layer_list.show()
                            self._layer_controls.show()
                        # Consume so the camera does not also pan in 3D.
                        return True
                    return False

            qt_viewer = viewer.window._qt_viewer

            # 3D: VisPy canvas consumes the middle button for camera panning,
            # which stops propagation to the ancestor qt_viewer filter. Install
            # on the canvas native QWidget (the leaf that gets the event first)
            # so the toggle fires in both 2D and 3D.
            canvas_native = getattr(getattr(qt_viewer, "canvas", None), "native", None)
            if isinstance(canvas_native, QObject):
                mf = _MiddleClickFilter(
                    layer_list, layer_controls, parent=canvas_native
                )
                canvas_native.installEventFilter(mf)
            else:
                mf = _MiddleClickFilter(layer_list, layer_controls, parent=qt_viewer)
                qt_viewer.installEventFilter(mf)
    except Exception:
        pass

    def _on_file_selected(path):
        if _active_viewer[0] is not None:
            _open_file(_active_viewer[0], path)

    def _on_file_selected_new_window(path):
        new_viewer = napari.Viewer(title=os.path.basename(path))
        _viewers.append(new_viewer)
        _active_viewer[0] = new_viewer
        _track_viewer(new_viewer)
        _open_file(new_viewer, path)

    widget = FolderBrowserWidget(start_dir=start_dir)
    widget.file_selected.connect(_on_file_selected)
    widget.file_selected_new_window.connect(_on_file_selected_new_window)
    widget.setWindowFlags(Qt.WindowType.Window)
    widget.setWindowTitle("helicon — Files")
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
