import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _make_volume(nz=4, ny=5, nx=6):
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                vol[i, j, k] = i * 100 + j * 10 + k
    return vol


class TestVoxelValueDisplay:
    def test_click_panel_z_updates_voxel_label(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        x, y = 2, 1
        widget._on_click(0, x, y)  # Z panel: horizontal=x, vertical=y
        z = widget._nz // 2
        val = widget._volume[z, y, x]
        assert widget._ctrl._voxel_label.text() == (
            f"x,y,z={x},{y},{z} val={float(val):.6g}"
        )

    def test_click_panel_x_updates_voxel_label(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        z_click, y_click = 0, 3
        widget._on_click(1, z_click, y_click)  # X panel: horizontal=z, vertical=y
        x = widget._nx // 2
        val = widget._volume[z_click, y_click, x]
        assert widget._ctrl._voxel_label.text() == (
            f"x,y,z={x},{y_click},{z_click} val={float(val):.6g}"
        )

    def test_click_panel_y_updates_voxel_label(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        x_click, z_click = 5, 1
        widget._on_click(2, x_click, z_click)  # Y panel: horizontal=x, vertical=z
        y = widget._ny // 2
        val = widget._volume[z_click, y, x_click]
        assert widget._ctrl._voxel_label.text() == (
            f"x,y,z={x_click},{y},{z_click} val={float(val):.6g}"
        )


class TestZoomInputBox:
    def test_zoom_spinbox_present(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        assert isinstance(widget._ctrl._zoom_spin, QDoubleSpinBox)
        assert abs(widget._ctrl._zoom_spin.value() - 1.0) < 1e-6

    def test_zoom_spinbox_drives_views_and_slider(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        widget._ctrl._zoom_spin.setValue(4.0)
        assert abs(widget._ctrl._zoom_spin.value() - 4.0) < 1e-6
        # The slider should reflect the typed value.
        assert (
            abs(widget._ctrl._slider_to_zoom(widget._ctrl._zoom_slider.value()) - 4.0)
            < 1e-3
        )
        # All three slice views should zoom in lockstep.
        for view in (widget._xy_view, widget._xz_view, widget._yz_view):
            assert abs(view._zoom - 4.0) < 1e-6

    def test_zoom_slider_updates_spinbox(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        widget._ctrl.set_zoom(2.0)
        assert abs(widget._ctrl._zoom_spin.value() - 2.0) < 1e-6
        widget._ctrl._zoom_slider.setValue(widget._ctrl._zoom_to_slider(3.0))
        # The spinbox rounds to 2 decimals, so allow for rounding error.
        assert abs(widget._ctrl._zoom_spin.value() - 3.0) < 5e-2

    def test_zoom_outside_spinbox_range_clamps(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        widget._ctrl._zoom_spin.setValue(0.01)
        assert widget._ctrl._zoom_spin.value() >= 0.05
