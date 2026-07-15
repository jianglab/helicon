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
    try:
        from PySide6.QtCore import QSettings
    except ImportError:
        from PyQt5.QtCore import QSettings
    return QSettings("helicon", "display")


def _restore_geometry(dock, viewer):
    """Restore saved window geometry after Qt finishes initial layout."""
    try:
        from PySide6.QtCore import QTimer
    except ImportError:
        from PyQt5.QtCore import QTimer

    def _apply():
        settings = _get_qsettings()
        dock_geo = _read_rect(settings, "dock")
        if dock_geo is not None and _on_screen(*dock_geo):
            x, y, w, h = dock_geo
            dock.setGeometry(x, y, w, h)
            dock.show()
        else:
            _position_default(dock, viewer)

        viewer_geo = _read_rect(settings, "viewer")
        if viewer_geo is not None and _on_screen(*viewer_geo):
            x, y, w, h = viewer_geo
            try:
                viewer.window._qt_window.setGeometry(x, y, w, h)
            except AttributeError:
                pass

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
    try:
        from PySide6.QtGui import QGuiApplication
    except ImportError:
        from PyQt5.QtGui import QGuiApplication

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
    try:
        from PySide6.QtCore import QEvent, QObject
    except ImportError:
        from PyQt5.QtCore import QEvent, QObject

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


class _SliceDirectionWidget:
    """Replaces axis labels with a Z/Y/X dropdown at the right end of each slider."""

    _combos: list = []

    def __init__(self, viewer):
        self._viewer = viewer

    def inject(self):
        try:
            from PySide6.QtWidgets import QComboBox
        except ImportError:
            from PyQt5.QtWidgets import QComboBox

        from napari._qt.widgets.qt_dims import QtDimSliderWidget

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

            _SliceDirectionWidget._combos.append(combo)

            def _on_change(idx):
                for c in _SliceDirectionWidget._combos:
                    if c is not combo:
                        c.blockSignals(True)
                        c.setCurrentIndex(idx)
                        c.blockSignals(False)
                ndim = self_slider.dims.ndim
                if ndim >= 3:
                    orders = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
                    self_slider.dims.order = orders[idx]

            combo.currentIndexChanged.connect(_on_change)
            self_slider.layout().addWidget(combo)

        if not getattr(QtDimSliderWidget, "_patched", False):
            QtDimSliderWidget.__init__ = _patched_init
            QtDimSliderWidget._patched = True


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
    layer.contrast_limits_range = (float(data.min()), float(data.max()))


def _open_file(viewer, path: str) -> None:
    from pathlib import Path

    qt_window = viewer.window._qt_window
    if hasattr(qt_window, "_text_overlay") and qt_window._text_overlay.isVisible():
        qt_window._text_overlay.hide()

    ext = Path(path).suffix.lower()

    if ext == ".star":
        import starfile

        df = starfile.read(path, always_dict=True)
        image_name_col = None
        for val in df.values():
            for col in val.columns:
                if "ImageName" in col or "MicrographName" in col:
                    image_name_col = col
                    image_df = val
                    break
            if image_name_col:
                break
        if image_name_col is None:
            return
        first_name = str(image_df[image_name_col].iloc[0])
        if "@" in first_name:
            img_path = first_name.split("@")[-1]
        else:
            img_path = first_name
        if not Path(img_path).is_file():
            return
        _open_file(viewer, img_path)
        viewer.layers[-1].name = Path(path).name
        return

    if ext == ".pdf":
        _open_pdf(viewer, path)
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
        layer.contrast_limits_range = (float(data.min()), float(data.max()))
    else:
        viewer.open(path)


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
            "Install PySide6 or PyQt5: pip install PySide6"
        )

    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import QWidget, QApplication
    except ImportError:
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtWidgets import QWidget, QApplication

    import napari

    start_dir = args.folder if args.folder else os.getcwd()

    viewer = napari.Viewer(title="helicon display")

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
        try:
            from PySide6.QtWidgets import QWidget
        except ImportError:
            from PyQt5.QtWidgets import QWidget
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
        try:
            from PySide6.QtCore import QEvent, QObject
        except ImportError:
            from PyQt5.QtCore import QEvent, QObject

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
