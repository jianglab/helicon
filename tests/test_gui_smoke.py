"""The windows of the file browser draw something: a blank or broken window is caught.

Not a pixel comparison (fonts and platforms differ): the picture of the window is taken
offscreen and must have content, with a few of the things that should be in it.
"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from helicon.lib.gui import file_browser as fb
from helicon.lib.gui import gallery_widget as gw


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _pixels(widget) -> np.ndarray:
    """The picture of a window as an ``(h, w, 3)`` array (through a PNG: the memory of a
    QImage is not safe to read from Python)."""
    import io

    from PIL import Image
    from PySide6.QtCore import QBuffer, QIODevice

    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    widget.grab().toImage().save(buffer, "PNG")
    return np.asarray(Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGB"))


def _has_content(widget, min_colours=12) -> bool:
    px = _pixels(widget).reshape(-1, 3)
    return len(np.unique(px, axis=0)) >= min_colours and px.std() > 5


def _pump(app, rounds=30):
    import time

    for _ in range(rounds):
        app.processEvents()
        time.sleep(0.01)


class TestSmoke:
    def test_the_browser_draws_its_list_sidebar_and_preview(self, qapp, tmp_path):
        import mrcfile

        (tmp_path / "Class2D" / "job001").mkdir(parents=True)
        (tmp_path / "default_pipeline.star").write_text("")
        with mrcfile.new(
            tmp_path / "Class2D" / "job001" / "c.mrcs", overwrite=True
        ) as m:
            m.set_data(
                np.random.default_rng(0).normal(size=(6, 24, 24)).astype(np.float32)
            )
        w = fb.FolderBrowserWidget(start_dir=str(tmp_path / "Class2D" / "job001"))
        w.resize(1100, 600)
        w.show()
        _pump(qapp)
        w._tree.setCurrentIndex(w._model.index(0, 0))
        w.wait_for_selection()
        w._preview.wait()
        _pump(qapp)
        try:
            assert _has_content(w)
            # the sidebar lists the project, the preview shows the stack as a picture
            assert w._sidebar.topLevelItemCount() >= 2
            assert w._preview._pixmap is not None and not w._preview._picture.isHidden()
            assert not w._action_bar.isHidden()
        finally:
            w._preview.wait()
            w.close()

    def test_the_gallery_draws_its_tiles(self, qapp):
        rng = np.random.default_rng(1)
        frames = [rng.normal(size=(32, 32)).astype(np.float32) for _ in range(12)]
        widget = gw.ImageGalleryWidget()
        widget.set_data(
            lambda i: frames[i], 12, 32, 32, None, labels=[str(i) for i in range(12)]
        )
        widget.resize(400, 300)
        widget.show()
        _pump(qapp)
        assert _has_content(widget, min_colours=50)

    def test_the_orthogonal_viewer_draws_three_views_and_its_controls(self, qapp):
        z, y, x = np.mgrid[-8:8, -8:8, -8:8]
        volume = np.exp(-(x**2 + y**2 + z**2) / 20).astype(np.float32)
        widget = gw.OrthogonalViewerWidget(volume, apix=1.0, name="v")
        widget.resize(700, 400)
        widget.show()
        _pump(qapp)
        assert _has_content(widget, min_colours=50)
