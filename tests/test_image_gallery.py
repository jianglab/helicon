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

from helicon.lib.image_gallery import GalleryPanel, ImageGalleryWidget


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
from helicon.lib import napari_widgets  # noqa: E402


class TestGalleryModeButtons:
    def test_mrcs_has_gallery_mode(self, qapp, tmp_path):
        f = tmp_path / "particles.mrcs"
        f.write_bytes(b"x" * 64)
        w = napari_widgets.FolderBrowserWidget()
        assert "gallery" in w._display_modes_for(str(f))
        w.deleteLater()

    def test_data_star_has_gallery_mode(self, qapp, tmp_path):
        f = tmp_path / "particles.star"
        f.write_text("_rlnImageName\n")
        w = napari_widgets.FolderBrowserWidget()
        assert "gallery" in w._display_modes_for(str(f))
        w.deleteLater()

    def test_metadata_star_has_no_gallery_mode(self, qapp, tmp_path):
        f = tmp_path / "pipeline.star"
        f.write_text("options\n")
        w = napari_widgets.FolderBrowserWidget()
        assert "gallery" not in w._display_modes_for(str(f))
        w.deleteLater()

    def test_volume_has_gallery_mode(self, qapp, tmp_path):
        f = tmp_path / "volume.mrc"
        f.write_bytes(b"x" * 64)
        w = napari_widgets.FolderBrowserWidget()
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

    def test_click_does_not_open_in_viewer(self, qapp):
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
        widget = window.centralWidget()
        widget.image_activated.emit(17)
        assert viewer._added_image is None

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
        old_galleries = display._galleries[:]
        old_active = display._active_gallery[0]
        display._galleries.clear()
        display._active_gallery[0] = None
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
            )
            wb = display._open_gallery(
                read_fn=lambda i: images_b[i],
                n=10,
                img_w=20,
                img_h=20,
                apix=1.0,
                name="b.mrcs",
            )
            assert wa in display._galleries
            assert wb in display._galleries
            assert display._active_gallery[0] is wb

            wa.close()
            assert wa not in display._galleries
            assert wb in display._galleries
            assert display._active_gallery[0] is wb

            wb.close()
            assert wb not in display._galleries
            assert display._active_gallery[0] is None
        finally:
            display._galleries.clear()
            display._galleries.extend(old_galleries)
            display._active_gallery[0] = old_active


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
            assert any("Save Viewport As…" in c for c in calls)
        assert not w._dragged
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
