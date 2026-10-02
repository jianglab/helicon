"""Regression tests for the display GUI fixes (workers, caches, I/O, panning)."""

import os
import signal
import sys
import threading
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import mrcfile
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication

from helicon.commands import display
from helicon.lib.gui import file_browser, file_openers, workers
from helicon.lib.gui.caches import LRUCache
from helicon.lib.gui.gallery_widget import ImageGalleryWidget, OrthogonalViewerWidget
from helicon.lib.gui.images2star_widget import (
    Images2StarDialog,
    _DataFramePreviewModel,
)
from helicon.lib.gui.proc3d_widget import Proc3dDialog
from helicon.lib.gui.viewer import _LazyStarStack


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _flush_deletes(qapp):
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _write_mrc(path, data):
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
    return str(path)


class TestDialogCloseWithRunningWorker:
    """Closing a tool dialog must not destroy a still-running worker thread."""

    @staticmethod
    def _close_while_loading(qapp, make_dialog, close):
        release = threading.Event()

        def slow_loader(path):
            release.wait(10)
            raise RuntimeError("cancelled")

        dialog = make_dialog(slow_loader)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        worker = dialog._load_worker
        assert worker.isRunning()
        close(dialog)
        _flush_deletes(qapp)
        # The dialog is gone but the running worker was parked, not destroyed.
        assert worker in workers.parked_threads()
        assert worker.isRunning()
        release.set()
        assert worker.wait(5000)
        _flush_deletes(qapp)
        assert worker not in workers.parked_threads()

    def test_images2star_reject(self, qapp):
        self._close_while_loading(
            qapp,
            lambda loader: Images2StarDialog("dummy.star", loader=loader),
            lambda d: d.reject(),
        )

    def test_proc3d_reject(self, qapp):
        self._close_while_loading(
            qapp,
            lambda loader: Proc3dDialog("dummy.mrc", loader=loader),
            lambda d: d.reject(),
        )

    def test_proc3d_close(self, qapp):
        self._close_while_loading(
            qapp,
            lambda loader: Proc3dDialog("dummy.mrc", loader=loader),
            lambda d: d.close(),
        )

    def test_finished_worker_is_not_parked(self, qapp):
        dialog = Images2StarDialog("dummy.star", loader=lambda p: pd.DataFrame())
        dialog._load_worker.wait()
        worker = dialog._load_worker
        dialog.reject()
        assert worker not in workers.parked_threads()


class TestConvertEditBool:
    def test_bool_words(self):
        dtype = np.dtype(bool)
        assert _DataFramePreviewModel._convert_edit("true", dtype) is True
        assert _DataFramePreviewModel._convert_edit("No", dtype) is False
        assert _DataFramePreviewModel._convert_edit("1", dtype) is True
        assert _DataFramePreviewModel._convert_edit("0", dtype) is False
        assert _DataFramePreviewModel._convert_edit("maybe", dtype) is None

    def test_numeric_unchanged(self):
        assert _DataFramePreviewModel._convert_edit("3", np.dtype(int)) == 3
        assert _DataFramePreviewModel._convert_edit("2.5", np.dtype(float)) == 2.5


class TestVolumeHeaderOnly:
    def test_nz_from_header(self, tmp_path):
        vol = _write_mrc(tmp_path / "vol.mrc", np.zeros((4, 5, 6)))
        img = _write_mrc(tmp_path / "img.mrc", np.zeros((1, 5, 6)))
        real_open = mrcfile.open
        calls = []

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return real_open(*args, **kwargs)

        with patch("mrcfile.open", spy):
            browser = file_browser.FolderBrowserWidget
            assert browser._volume_has_nz_gt1(None, vol) is True
            assert browser._volume_has_nz_gt1(None, img) is False
        assert calls and all(c.get("header_only") for c in calls)

    def test_unreadable_file(self, tmp_path):
        bad = tmp_path / "bad.mrc"
        bad.write_bytes(b"not an mrc")
        browser = file_browser.FolderBrowserWidget
        assert browser._volume_has_nz_gt1(None, str(bad)) is False


class TestRowForFilepath:
    def test_lookup_follows_row_changes(self, qapp, tmp_path):
        for name in ("b.txt", "a.txt", "c.txt"):
            (tmp_path / name).write_text("x")
        model = file_browser.FileBrowserModel(str(tmp_path))
        paths = [
            model.data(model.index(r, file_browser.COL_NAME), Qt.ItemDataRole.UserRole)
            for r in range(model.rowCount())
        ]
        for row, path in enumerate(paths):
            assert model._row_for_filepath(path) == row
        model.sort(file_browser.COL_NAME, Qt.SortOrder.DescendingOrder)
        for path in paths:
            row = model._row_for_filepath(path)
            assert (
                model.data(
                    model.index(row, file_browser.COL_NAME), Qt.ItemDataRole.UserRole
                )
                == path
            )
        model.removeRows(0, 1)
        assert model._row_for_filepath("/no/such/file") == -1
        assert model.rowCount() == 2


class TestFolderIsHelicalCacheKey:
    def test_new_model_star_is_seen(self, tmp_path):
        folder = tmp_path / "Class3D" / "job001"
        folder.mkdir(parents=True)
        assert file_browser._folder_is_helical(str(folder)) is False
        star = folder / "run_it001_model.star"
        star.write_text("data_model_general\n\n_rlnIsHelix 0\n")
        assert file_browser._folder_is_helical(str(folder)) is False
        star.write_text("data_model_general\n\n_rlnIsHelix   1\n")
        os.utime(star, ns=(star.stat().st_atime_ns, star.stat().st_mtime_ns + 10**9))
        assert file_browser._folder_is_helical(str(folder)) is True


class TestBildOpenCylinder:
    def test_open_cylinder_faces_in_range(self, tmp_path):
        bild = tmp_path / "x.bild"
        bild.write_text(
            ".color 1 0 0\n"
            ".cylinder 0 0 0 0 0 10 1 open\n"
            ".cylinder 5 0 0 5 0 10 1\n"
        )
        viewer = MagicMock()
        with patch.object(file_openers, "_reset_view"):
            file_openers._open_bild(viewer, str(bild))
        (vertices, faces) = viewer.add_surface.call_args[0][0]
        segs = 24
        n_open = 2 * segs
        n_capped = 2 * segs + 2
        assert len(vertices) == n_open + n_capped
        assert len(faces) == 2 * segs + 4 * segs
        assert faces.min() >= 0 and faces.max() < len(vertices)
        # The open tube uses only its own ring vertices.
        assert faces[: 2 * segs].max() < n_open


class TestOpenHtmlUri:
    def test_uses_file_uri(self, tmp_path):
        page = tmp_path / "a b.html"
        page.write_text("<html></html>")
        with patch("webbrowser.open") as opener:
            file_openers._open_html(None, str(page))
        uri = opener.call_args[0][0]
        assert uri == page.resolve().as_uri()
        assert "%20" in uri


class TestLRUCache:
    def test_count_limit(self):
        cache = LRUCache(max_items=2, max_bytes=10**9)
        cache[1] = np.zeros(1)
        cache[2] = np.zeros(1)
        _ = cache[1]
        cache[3] = np.zeros(1)
        assert 1 in cache and 3 in cache and 2 not in cache

    def test_byte_limit(self):
        cache = LRUCache(max_items=100, max_bytes=100)
        for i in range(5):
            cache[i] = np.zeros(5, dtype=np.float64)  # 40 bytes each
        assert len(cache) == 2
        assert cache.nbytes == 80
        cache.clear()
        assert len(cache) == 0 and cache.nbytes == 0


class TestGalleryCachesAndZoom:
    @staticmethod
    def _gallery(n=100, w=64, h=64, dtype=np.float32):
        reads = []

        def read(i):
            reads.append(i)
            return (np.arange(w * h).reshape(h, w) % 251).astype(dtype)

        gallery = ImageGalleryWidget()
        gallery.resize(400, 400)
        gallery.set_data(read, n, w, h, dtype)
        return gallery, reads

    def test_min_zoom(self, qapp):
        gallery, _ = self._gallery()
        for _ in range(200):
            gallery._apply_zoom(1 / 1.1, 0, 0)
        assert gallery._scale * 64 >= gallery.MIN_TILE_PX - 1e-9

    def test_adjustment_does_not_reread_frames(self, qapp):
        gallery, reads = self._gallery()
        gallery.grab()
        n_reads = len(reads)
        assert n_reads > 0
        gallery.set_brightness(0.2)
        gallery.set_contrast(1.5)
        gallery.grab()
        assert len(reads) == n_reads
        gallery.invalidate_frames()
        gallery.grab()
        assert len(reads) > n_reads

    def test_thumb_cache_is_bounded(self, qapp):
        gallery, _ = self._gallery()
        assert isinstance(gallery._thumb_cache, LRUCache)
        assert isinstance(gallery._frame_cache, LRUCache)
        assert gallery._thumb_cache.max_items <= 10000

    def test_log_transform_integer_frame(self, qapp):
        gallery = ImageGalleryWidget()
        gallery._log_transform = True
        frame = np.array([[-30000, 30000], [0, 100]], dtype=np.int16)
        pixmap = gallery._to_thumb(frame)
        image = pixmap.toImage()
        # The brightest input pixel must render brightest (no wrap-around).
        assert image.pixelColor(1, 0).value() > image.pixelColor(0, 0).value()

    def test_histogram_not_reread_on_slider(self, qapp):
        gallery, reads = self._gallery()
        container = display._wrap_gallery_with_panel(gallery)
        from helicon.lib.gui.gallery_widget import _ControlPanel

        panel = container.findChild(_ControlPanel)
        panel._histogram_chk.setChecked(True)
        qapp.processEvents()
        n_reads = len(reads)
        panel._brightness_slider.setValue(30)
        panel._contrast_slider.setValue(150)
        assert len(reads) == n_reads
        assert panel._histogram_widget._brightness == pytest.approx(0.3)


class TestLazyStarStack:
    def test_shape_order_and_tuple_keys(self, tmp_path):
        stack = np.arange(3 * 4 * 6, dtype=np.float32).reshape(3, 4, 6)
        path = _write_mrc(tmp_path / "s.mrcs", stack)
        entries = [(i, path, 1.0) for i in range(3)]
        lazy = _LazyStarStack(entries, (3, 4, 6), np.float32)
        np.testing.assert_array_equal(lazy[1], stack[1])
        np.testing.assert_array_equal(lazy[np.int64(2)], stack[2])
        np.testing.assert_array_equal(lazy[(1, slice(0, 2))], stack[1, 0:2])
        np.testing.assert_array_equal(lazy[(1,)], stack[1])
        np.testing.assert_array_equal(
            lazy[(slice(0, 2), slice(None), 3)], stack[0:2, :, 3]
        )

    def test_open_image_ref_stack_shape(self, tmp_path):
        stack = np.zeros((2, 4, 6), dtype=np.float32)  # ny=4, nx=6
        path = _write_mrc(tmp_path / "s.mrcs", stack)
        entries = [(i, path, 1.0) for i in range(2)]
        viewer = MagicMock()
        viewer.dims.current_step = (1, 0, 0)
        with (
            patch.object(display, "_enable_continuous_auto_contrast"),
            patch.object(display, "_reset_view"),
            patch.object(display, "_SliceDirectionWidget"),
        ):
            display._open_image_ref_stack(viewer, entries, (6, 4), 1.0, "s", "slice")
        layer_data = viewer.add_image.call_args[0][0]
        assert layer_data.shape == (2, 4, 6)
        assert layer_data[0].shape == (4, 6)


class TestClass2dReadImage:
    def test_reads_one_frame(self, tmp_path):
        from helicon.lib.gui.gallery_backends import Class2dGallery

        stack = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        path = _write_mrc(tmp_path / "classes.mrcs", stack)
        gallery = Class2dGallery(str(tmp_path / "run_model.star"))
        gallery._entries = [(path, i) for i in range(3)]
        gallery._order = [2, 0, 1]
        np.testing.assert_array_equal(gallery._read_image(0), stack[2])
        assert gallery._read_image(1).dtype == np.float32


class TestOrthoLinkedPan:
    @staticmethod
    def _viewer(qapp):
        widget = OrthogonalViewerWidget(np.zeros((8, 8, 8), dtype=np.float32))
        widget.resize(600, 600)
        for v in (widget._xy_view, widget._xz_view, widget._yz_view):
            v.resize(200, 200)
            v._pan_x = v._pan_y = 0.0
        return widget

    def test_z_panel_drag(self, qapp):
        w = self._viewer(qapp)
        w._on_pan(0, 10, 20)  # Z panel: x horizontal, y vertical
        assert (w._xy_view._pan_x, w._xy_view._pan_y) == (10, 20)
        # X panel (z, y): only y moves, vertically.
        assert w._xz_view._pan_x == 0 and w._xz_view._pan_y == pytest.approx(20)
        # Y panel (x, z): only x moves, horizontally.
        assert w._yz_view._pan_x == pytest.approx(10) and w._yz_view._pan_y == 0

    def test_x_panel_drag(self, qapp):
        w = self._viewer(qapp)
        w._on_pan(1, 10, 20)  # X panel: z horizontal, y vertical
        assert w._xy_view._pan_x == 0 and w._xy_view._pan_y == pytest.approx(20)
        assert w._yz_view._pan_x == 0 and w._yz_view._pan_y == pytest.approx(10)

    def test_y_panel_drag(self, qapp):
        w = self._viewer(qapp)
        w._on_pan(2, 10, 20)  # Y panel: x horizontal, z vertical
        assert w._xy_view._pan_x == pytest.approx(10) and w._xy_view._pan_y == 0
        assert w._xz_view._pan_x == pytest.approx(20) and w._xz_view._pan_y == 0


class TestExitSignalHandler:
    def test_handler_accepts_signum_and_frame(self):
        handler = display._make_exit_signal_handler(signal.SIGTERM)
        with (
            patch.object(display, "_terminate_web_apps") as terminate,
            patch("signal.signal") as set_handler,
            patch("os.kill") as kill,
        ):
            handler(signal.SIGTERM, None)
        terminate.assert_called_once()
        set_handler.assert_called_once_with(signal.SIGTERM, signal.SIG_DFL)
        kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
