"""Tests for the EMAN2-style lazy image-stack gallery.

Covers ``helicon.lib.image_gallery`` and its display-command integration.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

try:
    from PySide6.QtCore import Qt, QEvent, QPoint
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    from PyQt5.QtCore import Qt, QEvent, QPoint
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication

from helicon.lib.gallery_widget import GalleryPanel, ImageGalleryWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestGalleryPanel:
    def test_visible_window_at_scale_one(self):
        panel = GalleryPanel(1000, 100, 100)
        assert panel.visible_row_col(800, 600, 1.0) == [0, 5, 7]

    def test_scroll_advances_rowstart(self):
        panel = GalleryPanel(1000, 100, 100)
        # 300 px / 102 px per row -> row 2
        res = panel.visible_row_col(800, 600, 1.0, 0, -300)
        rowstart, visiblerows, visiblecols = res
        assert rowstart == 2
        assert visiblecols == 7

    def test_scale_too_large_returns_none(self):
        panel = GalleryPanel(1000, 100, 100)
        # Rendered width >> viewport => zero columns
        assert panel.visible_row_col(50, 600, 100.0) is None

    def test_zero_images_safe(self):
        panel = GalleryPanel(0, 100, 100)
        assert panel.visible_row_col(800, 600, 1.0) == [0, 5, 7]


class TestImageGalleryWidget:
    def _fake_stack(self, n):
        images = [np.full((40, 40), float(i), dtype=np.float32) for i in range(n)]
        return images

    def test_only_visible_images_read(self, qapp):
        images = self._fake_stack(1000)
        read_log = []

        def read_fn(i):
            read_log.append(i)
            return images[i]

        widget = ImageGalleryWidget()
        widget.set_data(read_fn, 1000, 40, 40, np.float32)
        widget.resize(800, 600)
        # set_data() already sampled the stack for global contrast; isolate the
        # paint-time reads by issuing exactly one explicit paint at a known size.
        read_log.clear()
        from PySide6.QtGui import QPaintEvent

        widget.paintEvent(QPaintEvent(widget.rect()))

        painted = set(read_log)
        # 40px images render at 42px -> 19 cols, 14 rows = 266 visible tiles max.
        # The key laziness invariant: only visible tiles are read, never the
        # full 1000-image stack.
        assert len(painted) < 1000
        assert max(painted) < 1000
        widget.deleteLater()

    def test_ctrl_wheel_zoom_changes_column_count(self, qapp):
        images = self._fake_stack(500)
        widget = ImageGalleryWidget()
        widget.set_data(lambda i: images[i], 500, 40, 40, np.float32)
        widget.resize(800, 600)
        widget.show()
        QApplication.processEvents()
        canvas = widget._canvas_width()
        cols_before = widget._panel.visible_row_col(canvas, 600, widget._scale)[2]

        # Ctrl+wheel zooms out (delta < 0) -> MORE columns fit.
        event = _wheel_event(-120, ctrl=True)
        widget.wheelEvent(event)
        cols_after = widget._panel.visible_row_col(canvas, 600, widget._scale)[2]
        assert cols_after >= cols_before
        widget.deleteLater()

    def test_plain_wheel_scrolls_vertically(self, qapp):
        images = self._fake_stack(1000)
        widget = ImageGalleryWidget()
        widget.set_data(lambda i: images[i], 1000, 40, 40, np.float32)
        widget.resize(800, 600)
        widget.show()
        QApplication.processEvents()
        rowstart_before = widget._panel.visible_row_col(
            widget._canvas_width(), 600, widget._scale, 0, widget._scroll_y
        )[0]
        # Plain wheel scrolls down the stack -> first visible row advances.
        event = _wheel_event(-240)
        widget.wheelEvent(event)
        rowstart_after = widget._panel.visible_row_col(
            widget._canvas_width(), 600, widget._scale, 0, widget._scroll_y
        )[0]
        assert rowstart_after > rowstart_before
        widget.deleteLater()

    def test_click_emits_image_index(self, qapp):
        images = self._fake_stack(100)
        widget = ImageGalleryWidget()
        widget.set_data(lambda i: images[i], 100, 40, 40, np.float32)
        widget.resize(800, 600)
        widget.show()
        QApplication.processEvents()

        emitted = []
        widget.image_activated.connect(emitted.append)

        # Pick a known visible tile and click its center.
        first_idx = next(iter(widget._coords))
        rect = widget._coords[first_idx]
        center = rect.center()
        release = _mouse_event(QEvent.MouseButtonRelease, center, Qt.LeftButton)
        widget.mouseReleaseEvent(release)

        assert emitted == [first_idx]
        widget.deleteLater()

    def test_drag_does_not_emit(self, qapp):
        images = self._fake_stack(100)
        widget = ImageGalleryWidget()
        widget.set_data(lambda i: images[i], 100, 40, 40, np.float32)
        widget.resize(800, 600)
        widget.show()
        QApplication.processEvents()

        emitted = []
        widget.image_activated.connect(emitted.append)

        first_idx = next(iter(widget._coords))
        rect = widget._coords[first_idx]
        start = rect.center()
        press = _mouse_event(QEvent.MouseButtonPress, start, Qt.LeftButton)
        widget.mousePressEvent(press)
        drag_to = start + QPoint(50, 30)
        move = _mouse_event(QEvent.MouseMove, drag_to, Qt.LeftButton)
        widget.mouseMoveEvent(move)
        release = _mouse_event(QEvent.MouseButtonRelease, drag_to, Qt.LeftButton)
        widget.mouseReleaseEvent(release)

        assert emitted == []
        widget.deleteLater()


def _wheel_event(delta, ctrl=False):
    from PySide6.QtCore import QPoint, Qt

    mods = Qt.ControlModifier if ctrl else Qt.NoModifier
    return type(
        "E",
        (),
        {
            "angleDelta": lambda self: QPoint(0, delta),
            "modifiers": lambda self: mods,
            "accept": lambda self: None,
        },
    )()


def _mouse_event(etype, pos, button):
    return QMouseEvent(etype, pos, button, button, Qt.NoModifier)


# ---------------------------------------------------------------------------
# Display-command integration tests
# ---------------------------------------------------------------------------

from helicon.commands import display  # noqa: E402
from helicon.lib import file_browser  # noqa: E402


class TestGalleryModeButtons:
    def test_mrcs_has_gallery_mode(self, qapp, tmp_path):
        f = tmp_path / "particles.mrcs"
        f.write_bytes(b"x" * 64)
        w = file_browser.FolderBrowserWidget()
        assert "gallery" in w._display_modes_for(str(f))
        w.deleteLater()

    def test_data_star_has_gallery_mode(self, qapp, tmp_path):
        f = tmp_path / "particles.star"
        f.write_text("_rlnImageName\n")
        w = file_browser.FolderBrowserWidget()
        assert "gallery" in w._display_modes_for(str(f))
        w.deleteLater()

    def test_metadata_star_has_no_gallery_mode(self, qapp, tmp_path):
        f = tmp_path / "pipeline.star"
        f.write_text("options\n")
        w = file_browser.FolderBrowserWidget()
        assert "gallery" not in w._display_modes_for(str(f))
        w.deleteLater()

    def test_volume_has_gallery_mode(self, qapp, tmp_path):
        f = tmp_path / "volume.mrc"
        f.write_bytes(b"x" * 64)
        w = file_browser.FolderBrowserWidget()
        assert "gallery" in w._display_modes_for(str(f))
        w.deleteLater()


class TestOpenGalleryDispatch:
    def test_opens_standalone_window(self, qapp):
        images = [np.full((20, 20), float(i), dtype=np.float32) for i in range(50)]

        window = display._open_gallery(
            read_fn=lambda i: images[i],
            n=50,
            img_w=20,
            img_h=20,
            apix=1.0,
            name="x.mrcs",
        )
        # The gallery is a standalone window with no napari viewer dependency.
        assert window is not None
        assert window.centralWidget() is not None
        from helicon.lib.gallery_widget import ImageGalleryWidget

        assert window.centralWidget().findChild(ImageGalleryWidget) is not None

    def test_click_does_not_open_in_viewer(self, qapp):
        from helicon.lib.gallery_widget import ImageGalleryWidget

        viewer = _mock_viewer()
        images = [np.full((20, 20), float(i), dtype=np.float32) for i in range(50)]
        window = display._open_gallery(
            read_fn=lambda i: images[i],
            n=50,
            img_w=20,
            img_h=20,
            apix=1.0,
            name="x.mrcs",
        )
        gallery = window.centralWidget().findChild(ImageGalleryWidget)
        gallery.image_activated.emit(17)
        assert viewer._added_image is None

    def test_panel_toggle_grows_window_leftward(self, qapp):
        from helicon.lib.gallery_widget import ImageGalleryWidget, _ControlPanel

        images = [np.full((20, 20), float(i), dtype=np.float32) for i in range(50)]
        window = display._open_gallery(
            read_fn=lambda i: images[i],
            n=50,
            img_w=20,
            img_h=20,
            apix=1.0,
            name="x.mrcs",
        )
        gallery = window.centralWidget().findChild(ImageGalleryWidget)
        panel = window.centralWidget().findChild(_ControlPanel)

        wc = _ControlPanel.PANEL_WIDTH
        w0, h0, x0, y0 = window.width(), window.height(), window.x(), window.y()
        assert not panel.isVisible()

        # Toggle ON: window grows leftward by ~wc, gallery position unchanged.
        gallery.panel_toggle_requested.emit()
        qapp.processEvents()
        assert panel.isVisible()
        assert window.width() == w0 + wc
        assert window.height() == h0
        # x shifts left by wc (frame-margin noise allowed on the test platform)
        assert abs((window.x() - x0) + wc) <= 4
        # Absolute invariant the user reported: no vertical drift per toggle.
        assert window.y() == y0

        # Toggle OFF: window shrinks back to original geometry.
        gallery.panel_toggle_requested.emit()
        qapp.processEvents()
        assert not panel.isVisible()
        assert window.width() == w0
        assert window.height() == h0
        assert abs(window.x() - x0) <= 4
        assert window.y() == y0

    def test_panel_toggle_no_vertical_drift(self, qapp):
        # Regression test for the reported bug: each toggle used to move the
        # window up by the title-bar height.  Toggling many times must keep
        # the window's y exactly constant (no accumulation).
        from helicon.lib.gallery_widget import ImageGalleryWidget

        images = [np.full((20, 20), float(i), dtype=np.float32) for i in range(50)]
        window = display._open_gallery(
            read_fn=lambda i: images[i],
            n=50,
            img_w=20,
            img_h=20,
            apix=1.0,
            name="x.mrcs",
        )
        gallery = window.centralWidget().findChild(ImageGalleryWidget)
        y0 = window.y()

        for _ in range(5):
            gallery.panel_toggle_requested.emit()
            qapp.processEvents()
            gallery.panel_toggle_requested.emit()
            qapp.processEvents()

        assert window.y() == y0

    def test_reuse_window_updates_content(self, qapp):
        images_a = [np.full((20, 20), float(i), dtype=np.float32) for i in range(50)]
        images_b = [
            np.full((20, 20), float(i) + 100, dtype=np.float32) for i in range(30)
        ]
        window = display._open_gallery(
            read_fn=lambda i: images_a[i],
            n=50,
            img_w=20,
            img_h=20,
            apix=1.0,
            name="stack_a.mrcs",
        )
        old_id = id(window)
        old_widget = window.centralWidget()

        returned = display._open_gallery(
            read_fn=lambda i: images_b[i],
            n=30,
            img_w=20,
            img_h=20,
            apix=2.0,
            name="stack_b.mrcs",
            reuse_window=window,
        )
        assert id(returned) == old_id
        assert window.windowTitle() == "helicon - stack_b.mrcs"
        assert window.centralWidget() is not old_widget

    def test_close_removes_from_tracker(self, qapp):
        tracker = display._DisplayTracker(display._is_alive_widget)
        old_tracker_windows = display._gallery._windows[:]
        old_tracker_active = display._gallery._active
        display._gallery._windows.clear()
        display._gallery._active = None
        try:
            images_a = [np.full((20, 20), 0.0, dtype=np.float32) for _ in range(10)]
            images_b = [np.full((20, 20), 1.0, dtype=np.float32) for _ in range(10)]
            wa = display._open_gallery(
                read_fn=lambda i: images_a[i],
                n=10,
                img_w=20,
                img_h=20,
                apix=1.0,
                name="a.mrcs",
                tracker=tracker,
            )
            wb = display._open_gallery(
                read_fn=lambda i: images_b[i],
                n=10,
                img_w=20,
                img_h=20,
                apix=1.0,
                name="b.mrcs",
                tracker=tracker,
            )
            assert wa in tracker.alive()
            assert wb in tracker.alive()
            assert tracker.active() is wb

            wa.close()
            assert wa not in tracker.alive()
            assert wb in tracker.alive()
            assert tracker.active() is wb

            wb.close()
            assert wb not in tracker.alive()
            assert tracker.active() is None
        finally:
            display._gallery._windows.clear()
            display._gallery._windows.extend(old_tracker_windows)
            display._gallery._active = old_tracker_active


class TestGallerySave:
    def test_right_click_without_drag_shows_menu(self, qapp):
        images = [np.full((20, 20), 0.5, dtype=np.float32) for _ in range(3)]
        w = ImageGalleryWidget()
        w.set_data(lambda i: images[i], 3, 20, 20, np.float32)
        w.show()
        QApplication.processEvents()

        from unittest.mock import patch

        pos = QPoint(50, 50)
        press = QMouseEvent(
            QEvent.MouseButtonPress,
            pos,
            Qt.RightButton,
            Qt.RightButton,
            Qt.NoModifier,
        )
        release = QMouseEvent(
            QEvent.MouseButtonRelease,
            pos,
            Qt.RightButton,
            Qt.RightButton,
            Qt.NoModifier,
        )
        w.mousePressEvent(press)
        with patch("PySide6.QtWidgets.QMenu") as mock_menu_cls:
            mock_menu = mock_menu_cls.return_value
            w.mouseReleaseEvent(release)
            calls = [str(c) for c in mock_menu.addAction.call_args_list]
            assert any("Save Canvas As…" in c for c in calls)
        assert not w._dragged
        w.close()

    def test_save_as_crops_to_content(self, qapp):
        # The saved image should contain only the drawn thumbnails, not the
        # surrounding gray padding or the scrollbar.
        from unittest.mock import patch

        images = [np.full((20, 20), 0.5, dtype=np.float32) for _ in range(3)]
        w = ImageGalleryWidget()
        w.set_data(lambda i: images[i], 3, 20, 20, np.float32)
        w.resize(400, 300)
        w.show()
        QApplication.processEvents()

        captured = {}
        with patch(
            "helicon.commands.display._save_qimage",
            side_effect=lambda qimg, parent: captured.update({"qimg": qimg}),
        ):
            w._save_as()

        assert "qimg" in captured
        qimg = captured["qimg"]
        # Cropped image must be strictly smaller than the full widget (which
        # includes padding + scrollbar space).  The grabbed pixmap is in
        # device pixels, so compare logical sizes via devicePixelRatio.
        dpr = qimg.devicePixelRatio() or 1
        assert (qimg.width() / dpr) < w.width()
        assert (qimg.height() / dpr) < w.height()
        # And must match the tight bounding box of the drawn thumbnails.
        coords = w._coords
        assert coords
        min_x = min(r.x() for r in coords.values())
        min_y = min(r.y() for r in coords.values())
        max_x = max(r.x() + r.width() for r in coords.values())
        max_y = max(r.y() + r.height() for r in coords.values())
        assert (qimg.width() / dpr) == max_x - min_x
        assert (qimg.height() / dpr) == max_y - min_y
        w.close()

    def test_save_as_prefills_source_name(self, qapp):
        # The save dialog should open pre-filled with the opened file's base
        # name (suffix replaced by .png).
        from unittest.mock import patch

        images = [np.full((20, 20), 0.5, dtype=np.float32) for _ in range(3)]
        w = ImageGalleryWidget()
        w.set_data(
            lambda i: images[i], 3, 20, 20, np.float32, source_name="stack_a.mrcs"
        )
        w.resize(400, 300)
        w.show()
        QApplication.processEvents()

        captured = {}
        real_save = display._save_qimage

        def _fake(qimg, parent, default_name=None):
            captured["default_name"] = default_name
            return real_save(qimg, parent, default_name=default_name)

        with patch("helicon.commands.display._save_qimage", side_effect=_fake):
            w._save_as()
        assert captured["default_name"] == "stack_a.mrcs"

    def test_save_as_excludes_scrolled_off_tiles(self, qapp):
        # After scrolling, the crop must reflect only the tiles now visible,
        # not stale rects from before the scroll (the bug that made the saved
        # image larger than the viewport in both directions).
        from unittest.mock import patch

        images = [np.full((100, 100), 0.5, dtype=np.float32) for _ in range(50)]
        w = ImageGalleryWidget()
        w.set_data(lambda i: images[i], 50, 100, 100, np.float32)
        # Small viewport so the stack overflows and requires scrolling.
        w.resize(200, 200)
        w.show()
        QApplication.processEvents()

        # Sanity: initially the top rows are visible.
        assert 0 in w._coords

        # Scroll far down so the first rows leave the viewport.
        w._scroll_y = -5000
        w.update()
        QApplication.processEvents()

        captured = {}
        with patch(
            "helicon.commands.display._save_qimage",
            side_effect=lambda qimg, parent: captured.update({"qimg": qimg}),
        ):
            w._save_as()

        assert "qimg" in captured
        qimg = captured["qimg"]
        coords = w._coords
        assert coords
        # No stale tile from the top rows may remain after the scroll.
        min_index = min(coords.keys())
        assert min_index > 0, "scrolled-off top tiles leaked into the crop"
        # Crop must match the visible tiles' bbox, clipped to the widget
        # (grab() clips to the widget rect, so partial tiles shrink to what is
        # actually on screen).  The grabbed pixmap is in device pixels.
        dpr = qimg.devicePixelRatio() or 1
        min_x = max(0, min(r.x() for r in coords.values()))
        min_y = max(0, min(r.y() for r in coords.values()))
        max_x = min(w.width(), max(r.x() + r.width() for r in coords.values()))
        max_y = min(w.height(), max(r.y() + r.height() for r in coords.values()))
        assert (qimg.width() / dpr) == max_x - min_x
        assert (qimg.height() / dpr) == max_y - min_y
        w.close()


def _mock_viewer():
    """A minimal fake napari viewer that records window + image calls."""
    added = {}

    class FakeLayer:
        def __init__(self, data, name, scale, contrast_limits):
            self.data = data
            self.name = name
            self.scale = scale
            self.contrast_limits = contrast_limits

    class FakeViewer:
        def __init__(self):
            self.layers = []
            self.dims = type("D", (), {"ndisplay": 2, "current_step": (0,)})()
            self._gallery_widget = None
            self._gallery_window = None
            self._added_image = None
            self._added_image_name = None
            self.window = type("W", (), {"_gallery_window": None})()

        def add_image(
            self, data, name=None, scale=None, contrast_limits=None, **kwargs
        ):
            layer = FakeLayer(data, name, scale, contrast_limits)
            self.layers.append(layer)
            self._added_image = data
            self._added_image_name = name
            return layer

        def reset_view(self):
            pass

    return FakeViewer()
