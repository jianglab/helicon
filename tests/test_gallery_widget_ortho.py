import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDoubleSpinBox

from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget, _SliceView


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


def _expected_control_text(widget, x, y, z):
    val = float(widget._volume[z, y, x])
    cx, cy, cz = widget._nx // 2, widget._ny // 2, widget._nz // 2
    wx = (x - cx) * widget._apix
    wy = (y - cy) * widget._apix
    wz = (z - cz) * widget._apix
    return (
        f"val={val:.6g} at x,y,z={x},{y},{z} "
        f"({wx:.3g},{wy:.3g},{wz:.3g} Å | {widget._apix:.4g} Å/pixel)"
    )


class TestVoxelValueDisplay:
    def test_click_panel_z_updates_voxel_label(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        x, y = 2, 1
        widget._on_click(0, x, y)  # Z panel: horizontal=x, vertical=y
        z = widget._nz // 2
        assert widget._ctrl._voxel_label.text() == _expected_control_text(
            widget, x, y, z
        )

    def test_click_panel_x_updates_voxel_label(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        z_click, y_click = 0, 3
        widget._on_click(1, z_click, y_click)  # X panel: horizontal=z, vertical=y
        x = widget._nx // 2
        assert widget._ctrl._voxel_label.text() == _expected_control_text(
            widget, x, y_click, z_click
        )

    def test_click_panel_y_updates_voxel_label(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        x_click, z_click = 5, 1
        widget._on_click(2, x_click, z_click)  # Y panel: horizontal=x, vertical=z
        y = widget._ny // 2
        assert widget._ctrl._voxel_label.text() == _expected_control_text(
            widget, x_click, y, z_click
        )

    def test_slider_position_updates_voxel_label(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        x, y, z = 4, 0, 3
        widget._on_slider_position(x, y, z)
        assert widget._pos == [x, y, z]
        assert widget._ctrl._voxel_label.text() == _expected_control_text(
            widget, x, y, z
        )

    def test_apix_scales_world_coordinates(self, qapp):
        apix = 2.5
        widget = OrthogonalViewerWidget(_make_volume(), apix=apix)
        assert widget._ctrl._apix == apix
        x, y = widget._nx - 1, 0  # corner: world offset = (nx-1 - cx) * apix
        widget._on_click(0, x, y)
        z = widget._pos[2]
        assert widget._ctrl._voxel_label.text() == _expected_control_text(
            widget, x, y, z
        )


class TestVoxelHoverTip:
    @pytest.mark.parametrize(
        ("panel_idx", "dx", "dy", "expected_xyz"),
        [
            (0, 2.9, 1.8, (2, 1, 2)),  # Z: x, y, current z
            (1, 0.9, 3.2, (3, 3, 0)),  # X: current x, y, z
            (2, 5.8, 1.9, (5, 2, 1)),  # Y: x, current y, z
        ],
    )
    def test_hover_maps_each_panel_to_raw_voxel(
        self, qapp, panel_idx, dx, dy, expected_xyz
    ):
        widget = OrthogonalViewerWidget(_make_volume(), apix=2.0)
        original_pos = list(widget._pos)
        original_selected = widget._selected_panel_idx
        view = (widget._xy_view, widget._xz_view, widget._yz_view)[panel_idx]

        voxel = widget._voxel_at_panel_position(panel_idx, dx, dy)
        assert voxel is not None
        x, y, z, val = voxel
        assert (x, y, z) == expected_xyz
        assert val == float(widget._volume[z, y, x])

        widget._on_hover(panel_idx, dx, dy)
        assert view._hover_text == widget._format_hover_tip(x, y, z, val)
        assert widget._pos == original_pos
        assert widget._selected_panel_idx == original_selected

    def test_hover_tip_contains_raw_value_indices_and_world_position(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume(), apix=2.5)
        widget._brightness = 0.75
        widget._contrast = 2.0
        widget._gamma = 1.5
        widget._log_transform = True

        widget._on_hover(0, 5.2, 0.4)
        assert widget._xy_view._hover_text == (
            "val=205\n"
            "x,y,z=5,0,2\n"
            "world=5,-5,0 Å"
        )

    @pytest.mark.parametrize(
        ("value", "rendered"), [(np.nan, "nan"), (np.inf, "inf")]
    )
    def test_hover_formats_non_finite_raw_values(self, qapp, value, rendered):
        volume = _make_volume()
        volume[2, 1, 2] = value
        widget = OrthogonalViewerWidget(volume)
        widget._on_hover(0, 2.1, 1.1)
        assert widget._xy_view._hover_text.splitlines()[0] == f"val={rendered}"

    def test_hover_rejects_out_of_bounds_positions(self, qapp):
        widget = OrthogonalViewerWidget(_make_volume())
        widget._xy_view._hover_text = "stale"
        widget._on_hover(0, widget._nx, 1.0)
        assert widget._xy_view._hover_text == ""
        assert widget._voxel_at_panel_position(99, 1.0, 1.0) is None

    def test_slice_screen_mapping_excludes_letterbox_and_tracks_pan_zoom(self, qapp):
        view = _SliceView()
        view.resize(200, 100)
        view.set_image(np.zeros((10, 10), dtype=np.float32))
        assert view.hasMouseTracking()
        assert view._data_position_at(49.9, 50.0) is None
        assert view._data_position_at(50.0, 0.0) == pytest.approx((0.0, 0.0))
        assert view._data_position_at(149.9, 99.9) == pytest.approx((9.99, 9.99))
        assert view._data_position_at(150.0, 50.0) is None

        view.set_zoom(2.0)
        view.set_pan(10.0, -5.0)
        sx, sy = view._data_to_screen(3.25, 4.5)
        assert view._data_position_at(sx, sy) == pytest.approx((3.25, 4.5))

    def test_hover_clears_when_view_content_or_transform_changes(self, qapp):
        view = _SliceView()
        view.set_image(np.zeros((4, 4), dtype=np.float32))
        view._hover_text = "tip"
        view.set_zoom(2.0)
        assert view._hover_text == ""
        view._hover_text = "tip"
        view.set_pan(1.0, 1.0)
        assert view._hover_text == ""
        view._hover_text = "tip"
        view.set_image(np.ones((4, 4), dtype=np.float32))
        assert view._hover_text == ""


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
