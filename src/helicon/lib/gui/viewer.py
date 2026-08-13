"""napari viewer creation, customizations, and save helpers.

Everything that instantiates or customizes a napari ``Viewer`` for the
``helicon display`` command: theme/geometry persistence, the Z/Y/X slice
direction widget, auto-contrast, the save-as menu, the layer-panel
middle-click toggle, and the icon patches that keep the Helicon brand
intact.
"""

from __future__ import annotations

from pathlib import Path

from helicon.lib.exceptions import HeliconDependencyError

from .theme import (
    _display_theme_stylesheet,
    _napari_canvas_background,
    _napari_display_theme,
)
from .trackers import _napari


def _load_napari():
    """Import napari only when a napari-backed display is requested."""
    import napari

    _patch_napari_value_bug()
    _patch_napari_icon()
    return napari


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
            combo.setStyleSheet(_display_theme_stylesheet())

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

    from helicon.lib.gui.file_browser import _find_chimerax

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


def _create_napari_viewer(title="Helicon display"):
    """Create a new napari viewer with standard helicon customizations.

    Sets up the ``_SliceDirectionWidget`` and hides the layer panels so
    the viewer matches the default helicon look.  Raises
    ``HeliconDependencyError`` if napari or OpenGL is unavailable.
    """
    from unittest.mock import MagicMock

    napari = _load_napari()

    try:
        new_viewer = napari.Viewer(title=title)
    except Exception as exc:
        raise HeliconDependencyError(
            f"Failed to create the napari viewer: {exc}\n"
            "This can happen when no OpenGL-accelerated display is "
            "available.\nTry setting QT_QPA_PLATFORM=offscreen or "
            "updating your GPU drivers."
        ) from exc
    try:
        new_viewer.theme = _napari_display_theme()
        new_viewer.background_color = _napari_canvas_background()
    except Exception:
        pass
    _napari.register(new_viewer)
    _hide_layer_panels(new_viewer)
    try:
        if not isinstance(new_viewer.window, MagicMock):
            _SliceDirectionWidget(new_viewer).inject()
    except Exception:
        pass
    return new_viewer


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


def _patch_napari_icon() -> None:
    """Point napari's window icon at the Helicon icon.

    napari's ``_QtMainWindow`` calls ``QApplication.setWindowIcon`` with the
    napari logo both when a viewer is created and whenever the theme changes
    (``_update_logo``). On macOS that call also swaps the Dock icon away from
    Helicon. ``_QtMainWindow._get_window_icon`` intentionally reads a
    ``_window_icon`` attribute when present (napari's documented extension
    point for custom window icons), so we install the Helicon icon there:
    every napari ``setWindowIcon`` call then installs the Helicon logo, and
    no napari code path can replace it afterwards.

    napari renders window icons through ``QSvgRenderer``, so the resource is
    an SVG embedding a downscaled copy of the Helicon PNG.
    """
    try:
        from napari._qt.qt_main_window import (
            _QtMainWindow as _NapariMainWindow,
        )
    except Exception:
        try:
            # napari < 0.5 exported the (then public) QtMainWindow name.
            from napari._qt.qt_main_window import (
                QtMainWindow as _NapariMainWindow,
            )
        except Exception:
            return
    if getattr(_NapariMainWindow, "_helicon_icon_patched", False):
        return
    svg_path = Path(__file__).parents[2] / "resources" / "icon.svg"
    if not svg_path.is_file():
        return
    try:
        orig_init = _NapariMainWindow.__init__

        def _init_with_helicon_icon(self, *args, **kwargs):
            self._window_icon = str(svg_path)
            return orig_init(self, *args, **kwargs)

        _NapariMainWindow.__init__ = _init_with_helicon_icon
        _NapariMainWindow._helicon_icon_patched = True
    except Exception:
        pass


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
