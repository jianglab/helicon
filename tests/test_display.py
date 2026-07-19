import argparse
import os
import sys
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock
import helicon
from helicon.commands import display
from helicon.lib.exceptions import HeliconDependencyError

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestDisplayArgs(object):
    def test_add_args_has_folder_argument(self):
        parser = argparse.ArgumentParser()
        display.add_args(parser)
        actions = {a.dest for a in parser._actions}
        assert "folder" in actions

    def test_add_args_folder_is_optional(self):
        parser = argparse.ArgumentParser()
        display.add_args(parser)
        args = parser.parse_args([])
        assert args.folder is None

    def test_add_args_folder_accepts_path(self):
        parser = argparse.ArgumentParser()
        display.add_args(parser)
        args = parser.parse_args(["/some/path"])
        assert args.folder == "/some/path"

    def test_add_args_folder_accepts_relative_path(self):
        parser = argparse.ArgumentParser()
        display.add_args(parser)
        args = parser.parse_args(["./data"])
        assert args.folder == "./data"


class TestDisplayMain(object):
    @patch("helicon.has_napari", return_value=False)
    def test_main_raises_error_when_napari_not_installed(self, mock_has_napari):
        parser = argparse.ArgumentParser()
        display.add_args(parser)
        args = parser.parse_args([])
        with pytest.raises(HeliconDependencyError):
            display.main(args)

    @patch("helicon.has_napari", return_value=True)
    @patch.object(display, "FolderBrowserWidget")
    def test_main_raises_error_when_qt_not_available(
        self, mock_widget_class, mock_has_napari
    ):
        original = display.FolderBrowserWidget
        display.FolderBrowserWidget = None
        try:
            parser = argparse.ArgumentParser()
            display.add_args(parser)
            args = parser.parse_args([])
            with pytest.raises(HeliconDependencyError):
                display.main(args)
        finally:
            display.FolderBrowserWidget = original

    @patch.object(display, "_install_save_hook")
    @patch.object(display, "_restore_geometry")
    @patch("helicon.has_napari", return_value=True)
    @patch.object(display, "FolderBrowserWidget")
    def test_main_creates_viewer_with_dock(
        self,
        mock_widget_class,
        mock_has_napari,
        mock_restore,
        mock_save_hook,
    ):
        mock_napari = MagicMock()
        mock_viewer = MagicMock()
        mock_napari.Viewer.return_value = mock_viewer
        mock_widget = MagicMock()
        mock_widget_class.return_value = mock_widget

        sys.modules["napari"] = mock_napari
        try:
            parser = argparse.ArgumentParser()
            display.add_args(parser)
            args = parser.parse_args([])
            display.main(args)

            mock_napari.Viewer.assert_called_once_with(title="helicon")
            mock_widget_class.assert_called_once_with(start_dir=os.getcwd())
            mock_widget.setWindowFlags.assert_called_once()
            mock_widget.show.assert_called_once()
            mock_napari.run.assert_called_once()
        finally:
            del sys.modules["napari"]

    @patch("helicon.has_napari", return_value=True)
    @patch.object(display, "FolderBrowserWidget")
    def test_main_restores_and_saves_geometry(self, mock_widget_class, mock_has_napari):
        mock_napari = MagicMock()
        mock_viewer = MagicMock()
        mock_napari.Viewer.return_value = mock_viewer
        mock_widget = MagicMock()
        mock_widget_class.return_value = mock_widget

        sys.modules["napari"] = mock_napari
        try:
            parser = argparse.ArgumentParser()
            display.add_args(parser)
            args = parser.parse_args([])

            with (
                patch.object(display, "_restore_geometry") as mock_restore,
                patch.object(display, "_install_save_hook") as mock_save,
            ):
                display.main(args)

                mock_restore.assert_called_once_with(mock_widget, mock_viewer)
                mock_save.assert_called_once_with(mock_widget, mock_viewer)
        finally:
            del sys.modules["napari"]

    @patch.object(display, "_install_save_hook")
    @patch.object(display, "_restore_geometry")
    @patch("helicon.has_napari", return_value=True)
    @patch.object(display, "FolderBrowserWidget")
    def test_main_uses_provided_folder(
        self,
        mock_widget_class,
        mock_has_napari,
        mock_restore,
        mock_save_hook,
    ):
        mock_napari = MagicMock()
        mock_viewer = MagicMock()
        mock_napari.Viewer.return_value = mock_viewer
        mock_widget = MagicMock()
        mock_widget_class.return_value = mock_widget

        sys.modules["napari"] = mock_napari
        try:
            parser = argparse.ArgumentParser()
            display.add_args(parser)
            args = parser.parse_args(["/custom/path"])
            display.main(args)

            mock_widget_class.assert_called_once_with(start_dir="/custom/path")
        finally:
            del sys.modules["napari"]

    @patch.object(display, "_install_save_hook")
    @patch.object(display, "_restore_geometry")
    @patch("helicon.has_napari", return_value=True)
    @patch.object(display, "FolderBrowserWidget")
    def test_main_connects_file_selected_signal(
        self,
        mock_widget_class,
        mock_has_napari,
        mock_restore,
        mock_save_hook,
    ):
        mock_napari = MagicMock()
        mock_viewer = MagicMock()
        mock_napari.Viewer.return_value = mock_viewer
        mock_widget = MagicMock()
        mock_widget_class.return_value = mock_widget

        sys.modules["napari"] = mock_napari
        try:
            parser = argparse.ArgumentParser()
            display.add_args(parser)
            args = parser.parse_args([])
            display.main(args)

            assert mock_widget.file_selected.connect.called
            connect_call = mock_widget.file_selected.connect
            assert connect_call.call_count == 1
        finally:
            del sys.modules["napari"]

    def test_main_has_docstring(self):
        assert display.main.__doc__ is not None
        assert "napari" in display.main.__doc__.lower()


class TestGeometryPersistence(object):
    @patch.object(display, "_supports_position_restore", return_value=False)
    @patch.object(display, "_position_default")
    @patch.object(display, "_read_rect")
    @patch.object(display, "_get_qsettings")
    def test_restore_geometry_falls_back_when_position_not_supported(
        self,
        mock_get_settings,
        mock_read_rect,
        mock_position_default,
        mock_pos_supported,
    ):
        """On unsupported platforms (WSL), dock uses _position_default."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_read_rect.return_value = (100, 200, 300, 400)

        mock_dock = MagicMock()
        mock_viewer = MagicMock()

        with patch("PySide6.QtCore.QTimer") as mock_timer:
            display._restore_geometry(mock_dock, mock_viewer)
            captured_fn = mock_timer.singleShot.call_args[0][1]
            captured_fn()

        mock_position_default.assert_called_once_with(mock_dock, mock_viewer)
        # _position_default runs first, then the saved width/height are applied
        # via setGeometry (outer frame rect). x()/y() are MagicMocks here, so we
        # only assert the trailing width/height match the saved (300, 400).
        assert mock_dock.setGeometry.called
        assert mock_dock.setGeometry.call_args_list[-1].args[-2:] == (300, 400)
        mock_dock.show.assert_called_once()

    @patch.object(display, "_on_screen", return_value=True)
    @patch.object(display, "_supports_position_restore", return_value=True)
    @patch.object(display, "_read_rect")
    @patch.object(display, "_get_qsettings")
    def test_restore_geometry_restores_dock_position_when_supported(
        self,
        mock_get_settings,
        mock_read_rect,
        mock_pos_supported,
        mock_on_screen,
    ):
        """On macOS/Linux, dock position is fully restored from saved values."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_read_rect.return_value = (100, 200, 300, 400)

        mock_dock = MagicMock()
        mock_viewer = MagicMock()

        with patch("PySide6.QtCore.QTimer") as mock_timer:
            display._restore_geometry(mock_dock, mock_viewer)
            captured_fn = mock_timer.singleShot.call_args[0][1]
            captured_fn()

        mock_dock.setGeometry.assert_called_once_with(100, 200, 300, 400)
        mock_dock.show.assert_called_once()

    @patch.object(display, "_on_screen", return_value=False)
    @patch.object(display, "_supports_position_restore", return_value=True)
    @patch.object(display, "_position_default")
    @patch.object(display, "_read_rect")
    @patch.object(display, "_get_qsettings")
    def test_restore_geometry_falls_back_when_dock_off_screen(
        self,
        mock_get_settings,
        mock_read_rect,
        mock_position_default,
        mock_pos_supported,
        mock_on_screen,
    ):
        """When saved position is off-screen, fall back to _position_default."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_read_rect.return_value = (5000, 5000, 300, 400)

        mock_dock = MagicMock()
        mock_viewer = MagicMock()

        with patch("PySide6.QtCore.QTimer") as mock_timer:
            display._restore_geometry(mock_dock, mock_viewer)
            captured_fn = mock_timer.singleShot.call_args[0][1]
            captured_fn()

        mock_position_default.assert_called_once_with(mock_dock, mock_viewer)
        assert mock_dock.setGeometry.called
        assert mock_dock.setGeometry.call_args_list[-1].args[-2:] == (300, 400)
        mock_dock.show.assert_called_once()

    @patch.object(display, "_read_rect", return_value=None)
    @patch.object(display, "_get_qsettings")
    def test_restore_geometry_applies_saved_viewer(
        self, mock_get_settings, mock_read_rect
    ):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_settings.value.side_effect = lambda key: (
            b"fake-geometry-data" if key == "viewer_ba" else None
        )

        mock_dock = MagicMock()
        mock_dock.geometry.return_value = MagicMock(
            x=lambda: 0, y=lambda: 0, width=lambda: 250, height=lambda: 800
        )
        mock_dock.frameGeometry.return_value = MagicMock(
            x=lambda: 0, y=lambda: 0, width=lambda: 250, height=lambda: 800
        )
        mock_viewer = MagicMock()

        with patch("PySide6.QtCore.QTimer") as mock_timer:
            display._restore_geometry(mock_dock, mock_viewer)
            captured_fn = mock_timer.singleShot.call_args[0][1]
            captured_fn()

        mock_viewer.window._qt_window.restoreGeometry.assert_called_once_with(
            b"fake-geometry-data"
        )

    @patch.object(display, "_position_default")
    @patch.object(display, "_on_screen", return_value=True)
    @patch.object(display, "_read_rect", return_value=None)
    @patch.object(display, "_get_qsettings")
    def test_restore_geometry_calls_position_default_when_no_saved_state(
        self, mock_get_settings, mock_read_rect, mock_on_screen, mock_position
    ):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_dock = MagicMock()
        mock_viewer = MagicMock()

        with patch("PySide6.QtCore.QTimer") as mock_timer:
            display._restore_geometry(mock_dock, mock_viewer)
            captured_fn = mock_timer.singleShot.call_args[0][1]
            captured_fn()

        mock_position.assert_called_once_with(mock_dock, mock_viewer)

    def test_position_default_places_dock_left_of_viewer(self):
        mock_dock = MagicMock()
        mock_dock.frameGeometry.return_value = MagicMock(
            x=lambda: 0, y=lambda: 0, width=lambda: 250, height=lambda: 800
        )

        mock_viewer = MagicMock()
        mock_viewer.window._qt_window.frameGeometry.return_value = MagicMock(
            x=lambda: 500, y=lambda: 100, width=lambda: 1000, height=lambda: 800
        )

        display._position_default(mock_dock, mock_viewer)

        mock_dock.adjustSize.assert_called_once()
        mock_dock.setGeometry.assert_called_once_with(190, 100, 300, 800)

    def test_position_default_uses_minimum_width(self):
        mock_dock = MagicMock()
        mock_dock.frameGeometry.return_value = MagicMock(
            x=lambda: 0, y=lambda: 0, width=lambda: 100, height=lambda: 600
        )

        mock_viewer = MagicMock()
        mock_viewer.window._qt_window.frameGeometry.return_value = MagicMock(
            x=lambda: 500, y=lambda: 100, width=lambda: 1000, height=lambda: 800
        )

        display._position_default(mock_dock, mock_viewer)

        args = mock_dock.setGeometry.call_args[0]
        assert args[2] == 300  # width clamped to minimum 300

    @patch.object(display, "_on_screen", return_value=True)
    @patch.object(display, "_read_rect")
    @patch.object(display, "_get_qsettings")
    def test_restore_geometry_handles_missing_qt_window(
        self, mock_get_settings, mock_read_rect, mock_on_screen
    ):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_read_rect.side_effect = lambda s, p: (
            (100, 200, 300, 400) if p == "viewer" else None
        )

        mock_dock = MagicMock()
        mock_viewer = MagicMock()
        del mock_viewer.window._qt_window

        with patch("PySide6.QtCore.QTimer") as mock_timer:
            display._restore_geometry(mock_dock, mock_viewer)
            captured_fn = mock_timer.singleShot.call_args[0][1]
            captured_fn()

    def test_read_rect_returns_tuple_when_all_values_present(self):
        mock_settings = MagicMock()
        mock_settings.value.side_effect = lambda key: {
            "dock_x": 100,
            "dock_y": 200,
            "dock_width": 300,
            "dock_height": 400,
        }[key]

        result = display._read_rect(mock_settings, "dock")
        assert result == (100, 200, 300, 400)

    def test_read_rect_returns_none_when_value_missing(self):
        mock_settings = MagicMock()
        mock_settings.value.side_effect = lambda key: {
            "dock_x": 100,
            "dock_y": 200,
            "dock_width": None,
            "dock_height": 400,
        }[key]

        result = display._read_rect(mock_settings, "dock")
        assert result is None

    def test_read_rect_returns_none_when_value_not_int(self):
        mock_settings = MagicMock()
        mock_settings.value.side_effect = lambda key: {
            "dock_x": 100,
            "dock_y": "bad",
            "dock_width": 300,
            "dock_height": 400,
        }[key]

        result = display._read_rect(mock_settings, "dock")
        assert result is None

    def test_on_screen_returns_true_when_center_on_screen(self):
        mock_screen = MagicMock()
        mock_screen.geometry.return_value.contains.return_value = True
        with patch("PySide6.QtGui.QGuiApplication") as mock_qapp:
            mock_qapp.screens.return_value = [mock_screen]
            assert display._on_screen(100, 200, 300, 400) is True

    def test_on_screen_returns_false_when_center_off_screen(self):
        mock_screen = MagicMock()
        mock_screen.geometry.return_value.contains.return_value = False
        with patch("PySide6.QtGui.QGuiApplication") as mock_qapp:
            mock_qapp.screens.return_value = [mock_screen]
            assert display._on_screen(5000, 5000, 300, 400) is False

    @patch("platform.system", return_value="Darwin")
    def test_supports_position_restore_true_on_macos(self, _mock_sys):
        assert display._supports_position_restore() is True

    @patch.object(display, "_is_wsl", return_value=False)
    @patch("platform.system", return_value="Linux")
    def test_supports_position_restore_true_on_native_linux(self, _mock_sys, _mock_wsl):
        assert display._supports_position_restore() is True

    @patch.object(display, "_is_wsl", return_value=True)
    @patch("platform.system", return_value="Linux")
    def test_supports_position_restore_false_on_wsl(self, _mock_sys, _mock_wsl):
        assert display._supports_position_restore() is False

    @patch("platform.system", return_value="Windows")
    def test_supports_position_restore_false_on_windows(self, _mock_sys):
        assert display._supports_position_restore() is False

    @patch("platform.system", return_value="Linux")
    def test_is_wsl_true_when_microsoft_in_proc_version(self, _mock_sys):
        mock_file = MagicMock()
        mock_file.__enter__ = lambda s: s
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = "Linux version 5.10.16.3-microsoft-standard-WSL2"
        with patch("builtins.open", return_value=mock_file):
            assert display._is_wsl() is True

    @patch("platform.system", return_value="Linux")
    def test_is_wsl_false_on_native_linux(self, _mock_sys):
        mock_file = MagicMock()
        mock_file.__enter__ = lambda s: s
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = "Linux version 6.1.0-25-amd64"
        with patch("builtins.open", return_value=mock_file):
            assert display._is_wsl() is False

    @patch("platform.system", return_value="Darwin")
    def test_is_wsl_false_on_macos(self, _mock_sys):
        assert display._is_wsl() is False

    @patch("PySide6.QtCore.QObject")
    @patch("PySide6.QtCore.QEvent")
    @patch.object(display, "_save_geometry")
    def test_install_save_hook_returns_filter(
        self, mock_save, mock_qevent, mock_qobject
    ):
        mock_dock = MagicMock()
        mock_viewer = MagicMock()

        result = display._install_save_hook(mock_dock, mock_viewer)

        mock_viewer.window._qt_window.installEventFilter.assert_called_once_with(result)
        assert hasattr(result, "eventFilter")

    @patch.object(display, "_write_rect")
    @patch.object(display, "_get_qsettings")
    def test_save_geometry_saves_dock_and_viewer(
        self, mock_get_settings, mock_write_rect
    ):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_dock = MagicMock()
        mock_dock.geometry.return_value = MagicMock(
            x=lambda: 100, y=lambda: 200, width=lambda: 300, height=lambda: 400
        )

        def _make_viewer(display_ba=None):
            mock_viewer = MagicMock()
            mock_viewer._display_only_ba = display_ba
            win = mock_viewer.window._qt_window
            win.saveGeometry.return_value = b"saved-geo"
            return mock_viewer

        mock_write_rect.reset_mock()
        display._save_geometry(mock_dock, _make_viewer())
        assert mock_write_rect.call_count == 1
        mock_write_rect.assert_called_once_with(mock_settings, "dock", mock_dock)
        mock_settings.setValue.assert_any_call("viewer_ba", b"saved-geo")

        mock_write_rect.reset_mock()
        mock_settings.reset_mock()
        viewer_with_cache = _make_viewer(display_ba=b"display-only-ba")
        display._save_geometry(mock_dock, viewer_with_cache)
        mock_settings.setValue.assert_any_call("viewer_ba", b"display-only-ba")

    @patch.object(display, "_get_qsettings")
    def test_write_rect_saves_individual_values(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_widget = MagicMock()
        mock_widget.frameGeometry.return_value = MagicMock(
            x=lambda: 100, y=lambda: 200, width=lambda: 300, height=lambda: 400
        )

        display._write_rect(mock_settings, "dock", mock_widget)

        mock_settings.setValue.assert_any_call("dock_x", 100)
        mock_settings.setValue.assert_any_call("dock_y", 200)
        mock_settings.setValue.assert_any_call("dock_width", 300)
        mock_settings.setValue.assert_any_call("dock_height", 400)

    @patch.object(display, "_get_qsettings")
    def test_write_rect_handles_deleted_c_object(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_widget = MagicMock()
        mock_widget.frameGeometry.side_effect = RuntimeError(
            "Internal C++ object already deleted"
        )

        display._write_rect(mock_settings, "dock", mock_widget)

        mock_settings.setValue.assert_not_called()


class TestOpenFile(object):
    def test_open_mrc_file_adds_image(self):
        import numpy as np

        mock_viewer = MagicMock()
        mock_mrc = MagicMock()
        test_data = np.zeros((3, 4, 5), dtype=np.float32)
        mock_mrc.data = test_data
        mock_mrc.header.cella.x = 10.0
        mock_mrc.header.nx = 5
        mock_mrc.header.ny = 4
        mock_mrc.header.nz = 3
        mock_mrc.header.mode = 2
        mock_mrc.header.mapc = 1
        mock_mrc.header.mapr = 2
        mock_mrc.header.maps = 3

        mock_mrcfile = MagicMock()
        mock_mrcfile.open.return_value.__enter__ = MagicMock(return_value=mock_mrc)
        mock_mrcfile.open.return_value.__exit__ = MagicMock(return_value=False)

        old_mrc = sys.modules.get("mrcfile")
        sys.modules["mrcfile"] = mock_mrcfile
        try:
            display._open_file(mock_viewer, "/path/to/test.mrc")
        finally:
            if old_mrc is not None:
                sys.modules["mrcfile"] = old_mrc
            else:
                del sys.modules["mrcfile"]

        mock_viewer.add_image.assert_called_once()
        call_args, call_kwargs = mock_viewer.add_image.call_args
        # 2D views are opened lazily: add_image receives a dask array that
        # reconstructs the full volume plane-by-plane, never the eager array.
        assert hasattr(call_args[0], "compute")
        np.testing.assert_array_equal(call_args[0].compute(), test_data)
        assert call_args[0].shape == (3, 4, 5)
        assert call_kwargs["name"] == "test.mrc"
        assert call_kwargs["scale"] == (2.0, 2.0, 2.0)
        assert "contrast_limits" in call_kwargs
        assert len(call_kwargs["contrast_limits"]) == 2

    def test_ndisplay_resets_for_new_file(self):
        import numpy as np

        # A fake viewer that records dims.ndisplay and supports layer removal,
        # so we can assert the 2D/3D mode is reset per file.
        class FakeDims:
            ndisplay = 2
            current_step = (0,)

        class FakeLayer:
            def __init__(self, name):
                self.name = name
                self.visible = True

        class FakeViewer:
            def __init__(self):
                self.layers = []
                self.dims = FakeDims()
                self.window = MagicMock()
                self.window._qt_window = MagicMock()
                self.reset_calls = 0

            def add_image(self, data, **kwargs):
                layer = FakeLayer(kwargs.get("name", "layer"))
                self.layers.append(layer)
                return layer

            def reset_view(self):
                self.reset_calls += 1

        # Mock mrcfile so the volume open doesn't hit real file I/O.
        mock_mrc = MagicMock()
        test_data = np.zeros((4, 8, 8), dtype=np.float32)
        mock_mrc.data = test_data
        mock_mrc.voxel_size.x = 1.5
        mock_mrc.header.nz = 4
        mock_mrc.header.mapc = 1
        mock_mrc.header.mapr = 2
        mock_mrc.header.maps = 3
        mock_mrcfile = MagicMock()
        mock_mrcfile.open.return_value.__enter__ = MagicMock(return_value=mock_mrc)
        mock_mrcfile.open.return_value.__exit__ = MagicMock(return_value=False)
        # helicon.change_map_axes_order is only called for ndim>=3 & nz>1.
        with (
            patch.dict(sys.modules, {"mrcfile": mock_mrcfile}),
            patch.object(
                display.helicon, "change_map_axes_order", lambda d, h: (d, None)
            ),
        ):
            # First open a 3D volume (ndisplay forced to 3)...
            vol_viewer = FakeViewer()
            display._open_file(vol_viewer, "/d/map.mrc", mode="volume")
            assert vol_viewer.dims.ndisplay == 3

            # ...then a 2D image stack. The stale 3D mode must not
            # persist: ndisplay is reset to 2 and only the new file shows
            # (previous layers are removed so reset_view fits just the new file).
            img_viewer = FakeViewer()
            img_viewer.dims.ndisplay = 3  # simulate leftover state
            old_layer = FakeLayer("old.mrc")
            img_viewer.layers.append(old_layer)
            display._open_file(img_viewer, "/d/stack.mrcs", mode="slice")
            assert img_viewer.dims.ndisplay == 2
            assert len(img_viewer.layers) == 1  # old removed, new added
            assert img_viewer.layers[0].name == "stack.mrcs"
            assert img_viewer.reset_calls == 1  # reset_view called once

    def test_open_tif_file_uses_viewer_open(self):
        mock_viewer = MagicMock()
        display._open_file(mock_viewer, "/path/to/test.tif")
        mock_viewer.open.assert_called_once_with("/path/to/test.tif")

    def test_open_data_star_uses_lazy_stack(self):
        import numpy as np

        first_frame = np.full((4, 4), 1.0, dtype=np.float32)

        class _FakeMRC:
            data = first_frame

            class header:
                ndim = 2

            class voxel_size:
                x = 2.0

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        mock_viewer = MagicMock()
        mock_viewer.dims.current_step = [0]

        fake_entries = [(0, "/x.mrcs", 0.0), (1, "/x.mrcs", 0.0), (2, "/x.mrcs", 0.0)]
        fake_shape = (4, 4)
        fake_apix = 2.0

        mock_mrcfile = MagicMock()
        mock_mrcfile.open.return_value = _FakeMRC()

        with (
            patch.object(
                display,
                "_parse_star_image_refs",
                return_value=(fake_entries, fake_shape, fake_apix),
            ),
            patch.dict(sys.modules, {"mrcfile": mock_mrcfile}),
        ):
            display._open_file(mock_viewer, "/path/to/data.star", mode="slice")

        mock_viewer.add_image.assert_called_once()
        args, kwargs = mock_viewer.add_image.call_args
        assert type(args[0]).__name__ == "_LazyStarStack"
        assert "contrast_limits" in kwargs
        assert kwargs["scale"] == (1.0, 2.0, 2.0)

    def test_open_png_file_uses_viewer_open(self):
        mock_viewer = MagicMock()
        display._open_file(mock_viewer, "/path/to/test.png")
        mock_viewer.open.assert_called_once_with("/path/to/test.png")

    def test_open_eps_rasterizes_with_ghostscript(self, qapp, tmp_path):
        import shutil

        from PySide6.QtGui import QImage

        # Build a real PNG that the gs mock will "produce" as its output.
        src_png = tmp_path / "rendered.png"
        img = QImage(16, 16, QImage.Format.Format_ARGB32)
        img.fill(0xFFFFFFFF)
        img.save(str(src_png))

        def fake_run(args, **kwargs):
            # gs is asked to write to -sOutputFile=<out>; copy our PNG there.
            out = next(a for a in args if a.startswith("-sOutputFile=")).split("=", 1)[
                1
            ]
            shutil.copy(str(src_png), out)
            return type("R", (), {"returncode": 0})()

        with (
            patch("subprocess.run", fake_run),
            patch("shutil.which", return_value="/usr/bin/gs"),
        ):
            mock_viewer = MagicMock()
            display._open_eps(mock_viewer, "/d/fig.eps")

        mock_viewer.add_image.assert_called_once()
        args, kwargs = mock_viewer.add_image.call_args
        assert kwargs["name"] == "fig.eps"

    def test_open_eps_without_ghostscript_prints_message(self):
        with (
            patch("shutil.which", return_value=None),
            patch.object(display, "print") as mock_print,
        ):
            mock_viewer = MagicMock()
            display._open_eps(mock_viewer, "/d/fig.eps")
        mock_viewer.add_image.assert_not_called()
        assert any("Ghostscript" in str(c.args) for c in mock_print.call_args_list)

    def test_extractpick_star_opens_as_text(self):
        mock_viewer = MagicMock()
        mock_viewer.window._qt_window._text_overlay = MagicMock()
        mock_viewer.window._qt_window._text_overlay.isVisible.return_value = False
        mock_viewer.window._qt_window.centralWidget.return_value = MagicMock()
        mock_viewer.window._qt_window.centralWidget.return_value.rect.return_value = (
            MagicMock()
        )

        with patch("builtins.open", MagicMock(return_value=MagicMock())):
            with patch.object(display, "_is_text_file", return_value=True):
                display._open_file(mock_viewer, "/path/to/extractpick.star")

        mock_viewer.open.assert_not_called()
        mock_viewer.add_image.assert_not_called()

    def test_metadata_mode_opens_any_star_as_text(self):
        mock_viewer = MagicMock()
        mock_viewer.window._qt_window._text_overlay = MagicMock()
        mock_viewer.window._qt_window._text_overlay.isVisible.return_value = False
        mock_viewer.window._qt_window.centralWidget.return_value = MagicMock()
        mock_viewer.window._qt_window.centralWidget.return_value.rect.return_value = (
            MagicMock()
        )

        with patch("builtins.open", MagicMock(return_value=MagicMock())):
            with patch.object(display, "_is_text_file", return_value=True):
                display._open_file(
                    mock_viewer, "/path/to/particles.star", mode="metadata"
                )

        mock_viewer.open.assert_not_called()
        mock_viewer.add_image.assert_not_called()


class TestAutoContrast(object):
    def test_returns_tuple_of_two_floats(self):
        import numpy as np

        data = np.random.rand(100, 100).astype(np.float32)
        result = display._auto_contrast(data)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_black_less_than_white(self):
        import numpy as np

        data = np.random.rand(100, 100).astype(np.float32)
        black, white = display._auto_contrast(data)
        assert black < white

    def test_uniform_data_returns_valid_range(self):
        import numpy as np

        data = np.ones((50, 50), dtype=np.float32) * 42.0
        black, white = display._auto_contrast(data)
        assert black < white

    def test_normal_distribution_uses_mad_bounds(self):
        import numpy as np

        np.random.seed(42)
        data = np.random.normal(100, 10, size=(1000, 1000)).astype(np.float32)
        black, white = display._auto_contrast(data)

        med = np.median(data)
        mad = np.median(np.abs(data - med)) * 1.4826
        p1 = np.percentile(data, 1)
        p99 = np.percentile(data, 99)

        expected_black = max(med - 3 * mad, p1)
        expected_white = min(med + 3 * mad, p99)

        assert abs(black - expected_black) < 0.01
        assert abs(white - expected_white) < 0.01

    def test_outlier_resistant_with_extreme_values(self):
        import numpy as np

        np.random.seed(42)
        data = np.random.normal(100, 10, size=(1000, 1000)).astype(np.float32)
        data[0, 0] = -10000
        data[999, 999] = 10000

        black, white = display._auto_contrast(data)

        assert black > -1000
        assert white < 1000

    def test_uses_percentile_when_mad_too_restrictive(self):
        import numpy as np

        data = np.zeros((100, 100), dtype=np.float32)
        data[0, 0] = 1000
        data[99, 99] = -1000

        black, white = display._auto_contrast(data)

        assert black == -1000.0
        assert white == 1000.0


class TestFolderBrowser(object):
    def test_format_size_bytes(self):
        from helicon.lib.napari_widgets import _format_size

        assert _format_size(512) == "512 B"

    def test_format_size_kilobytes(self):
        from helicon.lib.napari_widgets import _format_size

        assert _format_size(2048) == "2.0 KB"

    def test_format_size_megabytes(self):
        from helicon.lib.napari_widgets import _format_size

        assert _format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_format_size_gigabytes(self):
        from helicon.lib.napari_widgets import _format_size

        assert _format_size(2 * 1024 * 1024 * 1024) == "2.00 GB"

    def test_file_browser_model_has_six_columns(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, NUM_COLUMNS

        (tmp_path / "test.txt").write_text("hello")
        model = FileBrowserModel(str(tmp_path))
        assert model.columnCount() == NUM_COLUMNS

    def test_file_browser_model_headers(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel

        model = FileBrowserModel(str(tmp_path))
        headers = [model.headerData(c, Qt.Orientation.Horizontal) for c in range(7)]
        assert headers == [
            "Name",
            "Size",
            "Type",
            "Images",
            "Dimension",
            "Pixel Size",
            "Modified",
        ]

    def test_file_browser_model_lists_files(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_NAME

        (tmp_path / "aaa.txt").write_text("a")
        (tmp_path / "bbb.txt").write_text("b")
        model = FileBrowserModel(str(tmp_path))
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "aaa.txt" in names
        assert "bbb.txt" in names

    def test_file_browser_model_lists_dirs_first(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_NAME, COL_TYPE

        (tmp_path / "adir").mkdir()
        (tmp_path / "file.txt").write_text("x")
        model = FileBrowserModel(str(tmp_path))
        first_type = model.item(0, COL_TYPE).text()
        last_type = model.item(model.rowCount() - 1, COL_TYPE).text()
        assert first_type == "Folder"
        assert last_type != "Folder"

    def test_file_browser_model_shows_size(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_SIZE

        (tmp_path / "data.bin").write_bytes(b"x" * 2048)
        model = FileBrowserModel(str(tmp_path))
        sizes = [model.item(r, COL_SIZE).text() for r in range(model.rowCount())]
        assert "2.0 KB" in sizes

    def test_file_browser_model_shows_date(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_MODIFIED

        (tmp_path / "recent.txt").write_text("new")
        model = FileBrowserModel(str(tmp_path))
        dates = [model.item(r, COL_MODIFIED).text() for r in range(model.rowCount())]
        assert any("202" in d for d in dates)

    def test_file_browser_model_sort_by_size(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_SIZE

        (tmp_path / "small.txt").write_text("s")
        (tmp_path / "big.txt").write_text("x" * 10000)
        (tmp_path / "adir").mkdir()
        model = FileBrowserModel(str(tmp_path))
        model.sort(COL_SIZE, Qt.SortOrder.DescendingOrder)
        sizes = [model.item(r, COL_SIZE).text() for r in range(model.rowCount())]
        assert sizes[0] == "9.8 KB"
        assert sizes[-1] == ""

    def test_file_browser_model_sort_dirs_first(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_TYPE

        (tmp_path / "adir").mkdir()
        (tmp_path / "zfile.txt").write_text("z")
        model = FileBrowserModel(str(tmp_path))
        model.sort(0, Qt.SortOrder.AscendingOrder)
        assert model.item(0, COL_TYPE).text() == "Folder"

    def test_file_browser_model_set_root_path(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_NAME

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "inner.txt").write_text("i")
        model = FileBrowserModel(str(tmp_path))
        model.set_root_path(str(sub))
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "inner.txt" in names

    def test_file_browser_model_file_path(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel

        (tmp_path / "hello.txt").write_text("hi")
        model = FileBrowserModel(str(tmp_path))
        path = model.file_path(model.index(0, 0))
        assert path is not None
        assert "hello.txt" in path

    def test_file_browser_model_is_dir(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel

        (tmp_path / "mydir").mkdir()
        (tmp_path / "file.txt").write_text("f")
        model = FileBrowserModel(str(tmp_path))
        for r in range(model.rowCount()):
            idx = model.index(r, 0)
            name_idx = model.index(r, 0)
            name = model.item(r, 0).text()
            if name == "mydir":
                assert model.is_dir(idx) is True
            elif name == "file.txt":
                assert model.is_dir(idx) is False

    def test_folder_browser_emits_signal(self, tmp_path, qapp):
        from helicon.lib.napari_widgets import FolderBrowserWidget

        (tmp_path / "click.txt").write_text("c")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        results = []
        widget.file_selected.connect(lambda p: results.append(p))
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        index = widget._model.index(0, 0)
        widget._tree.doubleClicked.emit(index)
        assert len(results) == 1
        assert "click.txt" in results[0]

    def test_folder_browser_go_up(self, tmp_path, qapp):
        from helicon.lib.napari_widgets import FolderBrowserWidget

        sub = tmp_path / "sub"
        sub.mkdir()
        widget = FolderBrowserWidget(start_dir=str(sub))
        widget._go_up()
        assert widget._model._root_path == str(tmp_path)

    def test_folder_browser_go_back(self, tmp_path, qapp):
        from helicon.lib.napari_widgets import FolderBrowserWidget

        sub = tmp_path / "sub"
        sub.mkdir()
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        widget._navigate_to(str(sub))
        widget._go_back()
        assert widget._model._root_path == str(tmp_path)

    def test_folder_browser_shift_double_click_emits_new_window_signal(
        self, tmp_path, qapp
    ):
        from helicon.lib.napari_widgets import FolderBrowserWidget

        (tmp_path / "image.mrc").write_bytes(b"\x00" * 100)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        regular_results = []
        new_window_results = []
        widget.file_selected.connect(lambda p: regular_results.append(p))
        widget.file_selected_new_window.connect(lambda p: new_window_results.append(p))
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        from unittest.mock import patch

        with patch.object(
            QApplication,
            "keyboardModifiers",
            return_value=Qt.KeyboardModifier.ShiftModifier,
        ):
            index = widget._model.index(0, 0)
            widget._tree.doubleClicked.emit(index)
        assert len(regular_results) == 0
        assert len(new_window_results) == 1
        assert "image.mrc" in new_window_results[0]

    def test_is_image_stack_classification(self):
        from helicon.lib.napari_widgets import FolderBrowserWidget

        assert FolderBrowserWidget._is_image_stack(None, "/d/particles.mrcs")
        assert FolderBrowserWidget._is_image_stack(None, "/d/data.star")
        # Metadata star files are not image stacks.
        assert not FolderBrowserWidget._is_image_stack(None, "/d/run1_optimiser.star")
        # Volumes keep the "Image Slice" label, so are not image stacks.
        assert not FolderBrowserWidget._is_image_stack(None, "/d/map.mrc")
        assert not FolderBrowserWidget._is_image_stack(None, "/d/map.map")

    def test_slice_button_label_for_image_stack(self, tmp_path, qapp):
        from helicon.lib.napari_widgets import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        (tmp_path / "particles.mrcs").write_bytes(b"\x00" * 1024)
        (tmp_path / "volume.mrc").write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        def select(name):
            idx = None
            for r in range(widget._model.rowCount()):
                if widget._model.file_path(widget._model.index(r, 0)).endswith(name):
                    idx = widget._model.index(r, 0)
                    break
            assert idx is not None
            widget._tree.selectionModel().select(
                idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
            )

        select("particles.mrcs")
        assert widget._btn_slice.text() == "2D Image"

        select("volume.mrc")
        assert widget._btn_slice.text() == "2D Slice"

    def test_chimerax_button_shown_for_volumes(self, tmp_path, qapp):
        from helicon.lib.napari_widgets import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        (tmp_path / "volume.mrc").write_bytes(b"\x00" * 1024)
        (tmp_path / "particles.mrcs").write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        def select(name):
            idx = None
            for r in range(widget._model.rowCount()):
                if widget._model.file_path(widget._model.index(r, 0)).endswith(name):
                    idx = widget._model.index(r, 0)
                    break
            assert idx is not None
            widget._tree.selectionModel().select(
                idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
            )

        select("volume.mrc")
        assert not widget._btn_chimerax.isHidden()
        assert widget._btn_chimerax.text() == "ChimeraX"

        # Image stacks must not offer the ChimeraX button.
        select("particles.mrcs")
        assert widget._btn_chimerax.isHidden()

    def test_chimerax_button_shown_for_bild(self, tmp_path, qapp):
        from helicon.lib.napari_widgets import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        (tmp_path / "plot.bild").write_bytes(b"\x00" * 64)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        idx = widget._model.index(0, 0)
        widget._tree.selectionModel().select(
            idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
        )
        # bild files get a ChimeraX button (renders cylinders/spheres natively)
        # alongside the general image display.
        assert not widget._btn_chimerax.isHidden()
        assert widget._btn_chimerax.text() == "ChimeraX"
        assert not widget._btn_general.isHidden()

    def test_launch_chimerax_invokes_executable(self, tmp_path):
        import subprocess

        from helicon.lib import napari_widgets

        called = {}

        def fake_find():
            return "/fake/ChimeraX"

        def fake_popen(args):
            called["args"] = list(args)
            return object()

        with (
            patch.object(napari_widgets, "_find_chimerax", fake_find),
            patch.object(subprocess, "Popen", fake_popen),
        ):
            display._launch_chimerax("/d/map.mrc")

        assert called["args"] == ["/fake/ChimeraX", "/d/map.mrc"]

    def test_chimerax_button_disabled_when_not_installed(self, tmp_path, qapp):
        from helicon.lib import napari_widgets
        from helicon.lib.napari_widgets import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        with patch.object(napari_widgets, "_find_chimerax", lambda: None):
            (tmp_path / "volume.mrc").write_bytes(b"\x00" * 1024)
            widget = FolderBrowserWidget(start_dir=str(tmp_path))
            idx = widget._model.index(0, 0)
            widget._tree.selectionModel().select(
                idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
            )
            assert not widget._btn_chimerax.isHidden()
            assert not widget._btn_chimerax.isEnabled()
            assert "not found" in widget._btn_chimerax.toolTip().lower()

    def test_chimerax_button_enabled_when_installed(self, tmp_path, qapp):
        from helicon.lib import napari_widgets
        from helicon.lib.napari_widgets import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        with patch.object(napari_widgets, "_find_chimerax", lambda: "/x/ChimeraX"):
            (tmp_path / "volume.mrc").write_bytes(b"\x00" * 1024)
            widget = FolderBrowserWidget(start_dir=str(tmp_path))
            idx = widget._model.index(0, 0)
            widget._tree.selectionModel().select(
                idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
            )
            assert widget._btn_chimerax.isEnabled()
            assert "Open this file" in widget._btn_chimerax.toolTip()

    def test_stats_button_for_data_star_only(self, tmp_path, qapp):
        from helicon.lib.napari_widgets import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        (tmp_path / "data.star").write_text("_data_\n")
        (tmp_path / "particles.mrcs").write_bytes(b"\x00" * 1024)
        (tmp_path / "volume.mrc").write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        def select(name):
            idx = None
            for r in range(widget._model.rowCount()):
                if widget._model.file_path(widget._model.index(r, 0)).endswith(name):
                    idx = widget._model.index(r, 0)
                    break
            assert idx is not None
            widget._tree.selectionModel().select(
                idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
            )

        # Shown and disabled (not implemented) for data.star.
        select("data.star")
        assert not widget._btn_stats.isHidden()
        assert not widget._btn_stats.isEnabled()
        assert "not implemented" in widget._btn_stats.toolTip().lower()

        # Not shown for image stacks or volumes.
        select("particles.mrcs")
        assert widget._btn_stats.isHidden()
        select("volume.mrc")
        assert widget._btn_stats.isHidden()

    def test_eps_label_and_mode(self, tmp_path, qapp):
        from helicon.lib.napari_widgets import FolderBrowserWidget

        (tmp_path / "fig.eps").write_bytes(b"\x00" * 16)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        from PySide6.QtCore import QItemSelectionModel

        idx = None
        for r in range(widget._model.rowCount()):
            if widget._model.file_path(widget._model.index(r, 0)).endswith("fig.eps"):
                idx = widget._model.index(r, 0)
                break
        assert idx is not None
        widget._tree.selectionModel().select(
            idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
        )
        assert widget._btn_general.text() == "EPS"
        # EPS is a single general image (no slice/volume/chimerax modes).
        assert widget._display_modes_for(str(tmp_path / "fig.eps")) == ["general"]

    def test_file_browser_model_filter_wildcard(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_NAME

        (tmp_path / "aaa.mrc").write_bytes(b"x")
        (tmp_path / "bbb.txt").write_text("y")
        (tmp_path / "ccc.tif").write_bytes(b"z")
        model = FileBrowserModel(str(tmp_path))
        model.set_filter("*.mrc")
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "aaa.mrc" in names
        assert "bbb.txt" not in names
        assert "ccc.tif" not in names

    def test_file_browser_model_filter_regex(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_NAME

        (tmp_path / "image_001.mrc").write_bytes(b"x")
        (tmp_path / "image_002.mrc").write_bytes(b"y")
        (tmp_path / "data.txt").write_text("z")
        model = FileBrowserModel(str(tmp_path))
        model.set_filter(r"image_\d+\.mrc", use_regex=True)
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "image_001.mrc" in names
        assert "image_002.mrc" in names
        assert "data.txt" not in names

    def test_file_browser_model_filter_empty_shows_all(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_NAME

        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.mrc").write_bytes(b"b")
        model = FileBrowserModel(str(tmp_path))
        model.set_filter("")
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "a.txt" in names
        assert "b.mrc" in names

    def test_file_browser_model_filter_dirs_always_shown(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel, COL_NAME

        (tmp_path / "mydir").mkdir()
        (tmp_path / "file.mrc").write_bytes(b"x")
        (tmp_path / "file.txt").write_text("y")
        model = FileBrowserModel(str(tmp_path))
        model.set_filter("*.mrc")
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "mydir/" in names
        assert "file.mrc" in names
        assert "file.txt" not in names


class TestAsyncFileInfo(object):
    def test_async_columns_start_empty(self, tmp_path, qapp):
        from helicon.lib.napari_widgets import (
            COL_IMAGES,
            COL_INFO,
            COL_PIXELSIZE,
            FolderBrowserWidget,
        )

        (tmp_path / "a.mrc").write_bytes(b"\x00" * 1024)
        (tmp_path / "b.mrc").write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        # Dimension / Images / Pixel Size are intentionally empty until the
        # background worker fills them, so the folder opens without blocking.
        for row in range(widget._model.rowCount()):
            assert widget._model.item(row, COL_INFO).text() == ""
            assert widget._model.item(row, COL_IMAGES).text() == ""
            assert widget._model.item(row, COL_PIXELSIZE).text() == ""

    def test_apply_file_info_updates_text_sort_and_cache(self, tmp_path):
        from helicon.lib.napari_widgets import (
            COL_INFO,
            COL_PIXELSIZE,
            FileBrowserModel,
            ROLE_SORT,
        )

        (tmp_path / "a.mrc").write_bytes(b"\x00" * 1024)
        model = FileBrowserModel(str(tmp_path))
        row = 0
        filepath = model.file_path(model.index(row, 0))
        model.apply_file_info(filepath, "100x100", "1", "1.50 Å")
        assert model.item(row, COL_INFO).text() == "100x100"
        assert model.item(row, COL_INFO).data(ROLE_SORT) == "100x100"
        assert model.item(row, COL_PIXELSIZE).data(ROLE_SORT) == 1.5
        # Result is cached so subsequent synchronous reads reuse it.
        assert filepath in model._file_infos
        assert model._file_infos[filepath] == ("100x100", "1", "1.50 Å")

    def test_epoch_bumps_on_reload(self, tmp_path):
        from helicon.lib.napari_widgets import FileBrowserModel

        (tmp_path / "a.mrc").write_bytes(b"\x00" * 1024)
        model = FileBrowserModel(str(tmp_path))
        first = model.current_epoch()
        model.set_filter("*")  # triggers a directory reload -> epoch bump
        assert model.current_epoch() == first + 1

    def test_populate_fills_columns_async(self, tmp_path, qapp):
        from PySide6.QtWidgets import QApplication

        from helicon.lib import napari_widgets
        from helicon.lib.napari_widgets import (
            COL_IMAGES,
            COL_INFO,
            COL_PIXELSIZE,
            FolderBrowserWidget,
        )

        # Use fake metadata so the test does not depend on real file parsing.
        def fake_info(filepath):
            name = filepath.rsplit("/", 1)[-1]
            n = int("".join(c for c in name if c.isdigit()) or "1")
            return f"{n*10}x{n*10}", str(n), f"{n}.00 Å"

        with patch.object(napari_widgets, "_get_file_info", fake_info):
            for i in range(1, 6):
                (tmp_path / f"img_{i}.mrc").write_bytes(b"\x00" * 512)
            widget = FolderBrowserWidget(start_dir=str(tmp_path))
            widget._populate_file_info_async()
            # Drive the event loop so the queued info_ready/finished signals
            # from the worker threads are processed (blocking wait() would
            # deadlock, since finished -> quit needs the main thread).
            import time

            deadline = time.time() + 10
            while any(t.isRunning() for t in widget._info_threads) and (
                time.time() < deadline
            ):
                QApplication.processEvents()
            # Drain once more: a worker's final info_ready can still be queued
            # after it reports not-running. The real app's live event loop
            # covers this; here we flush explicitly.
            QApplication.processEvents()

        for row in range(widget._model.rowCount()):
            info = widget._model.item(row, COL_INFO).text()
            assert info != ""  # each row now has resolved dimensions
            assert widget._model.item(row, COL_IMAGES).text() != ""
            assert widget._model.item(row, COL_PIXELSIZE).text() != ""

    def test_initial_directory_populates_on_construction(self, tmp_path, qapp):
        from PySide6.QtWidgets import QApplication

        from helicon.lib import napari_widgets
        from helicon.lib.napari_widgets import (
            COL_IMAGES,
            COL_INFO,
            COL_PIXELSIZE,
            FolderBrowserWidget,
        )

        # The initial directory shown by the widget must be populated in the
        # background without any navigation/refresh; otherwise the columns
        # stay empty (the bug that motivated async loading).
        def fake_info(filepath):
            name = filepath.rsplit("/", 1)[-1]
            n = int("".join(c for c in name if c.isdigit()) or "1")
            return f"{n*10}x{n*10}", str(n), f"{n}.00 Å"

        with patch.object(napari_widgets, "_get_file_info", fake_info):
            for i in range(1, 4):
                (tmp_path / f"img_{i}.mrc").write_bytes(b"\x00" * 512)
            widget = FolderBrowserWidget(start_dir=str(tmp_path))
            import time

            deadline = time.time() + 10
            while any(t.isRunning() for t in widget._info_threads) and (
                time.time() < deadline
            ):
                QApplication.processEvents()
            QApplication.processEvents()

        for row in range(widget._model.rowCount()):
            assert widget._model.item(row, COL_INFO).text() != ""
            assert widget._model.item(row, COL_IMAGES).text() != ""
            assert widget._model.item(row, COL_PIXELSIZE).text() != ""


class TestSaveQimage:
    def test_saves_png_to_chosen_path(self, qapp, tmp_path):
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QFileDialog

        qimg = QImage(10, 10, QImage.Format.Format_RGB32)
        qimg.fill(0xFF0000FF)
        out = str(tmp_path / "out.png")

        def _exec(self):
            return QFileDialog.Accepted

        def _selected_files(self):
            return [out]

        with (
            patch.object(QFileDialog, "exec", _exec),
            patch.object(QFileDialog, "selectedFiles", _selected_files),
        ):
            display._save_qimage(qimg)
        assert os.path.exists(out)

    def test_cancel_dialog_saves_nothing(self, qapp):
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QFileDialog

        qimg = QImage(4, 4, QImage.Format.Format_RGB32)
        qimg.fill(0)

        def _exec(self):
            return QFileDialog.Rejected

        with patch.object(QFileDialog, "exec", _exec):
            display._save_qimage(qimg)

    def test_prefills_default_name_replacing_suffix(self, qapp, tmp_path):
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QFileDialog

        qimg = QImage(4, 4, QImage.Format.Format_RGB32)
        qimg.fill(0)
        captured = {}
        out = str(tmp_path / "ignored.png")

        def _select_file(self, name):
            captured["name"] = name

        def _exec(self):
            return QFileDialog.Accepted

        def _selected_files(self):
            return [out]

        with (
            patch.object(QFileDialog, "selectFile", _select_file),
            patch.object(QFileDialog, "exec", _exec),
            patch.object(QFileDialog, "selectedFiles", _selected_files),
        ):
            display._save_qimage(qimg, default_name="run1_optimiser.star")
        assert captured["name"] == "run1_optimiser.png"

    def test_prefills_default_name_without_suffix(self, qapp, tmp_path):
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QFileDialog

        qimg = QImage(4, 4, QImage.Format.Format_RGB32)
        qimg.fill(0)
        captured = {}
        out = str(tmp_path / "ignored.png")

        def _select_file(self, name):
            captured["name"] = name

        def _exec(self):
            return QFileDialog.Accepted

        def _selected_files(self):
            return [out]

        with (
            patch.object(QFileDialog, "selectFile", _select_file),
            patch.object(QFileDialog, "exec", _exec),
            patch.object(QFileDialog, "selectedFiles", _selected_files),
        ):
            display._save_qimage(qimg, default_name="volume")
        assert captured["name"] == "volume.png"

    def test_no_default_name_leaves_field_blank(self, qapp, tmp_path):
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QFileDialog

        qimg = QImage(4, 4, QImage.Format.Format_RGB32)
        qimg.fill(0)
        captured = {}
        out = str(tmp_path / "ignored.png")

        def _select_file(self, name):
            captured["name"] = name

        def _exec(self):
            return QFileDialog.Accepted

        def _selected_files(self):
            return [out]

        with (
            patch.object(QFileDialog, "selectFile", _select_file),
            patch.object(QFileDialog, "exec", _exec),
            patch.object(QFileDialog, "selectedFiles", _selected_files),
        ):
            display._save_qimage(qimg)
        assert "name" not in captured

    def test_viewer_source_name_reads_layer_path(self):
        class _Src:
            path = "/data/run1_model.star"

        class _Layer:
            source = _Src()

        class _Viewer:
            layers = [_Layer()]

        assert display._viewer_source_name(_Viewer()) == "run1_model.star"

    def test_viewer_source_name_returns_none_without_path(self):
        class _Layer:
            source = None

        class _Viewer:
            layers = [_Layer()]

        assert display._viewer_source_name(_Viewer()) is None


class TestRenderQimageVector:
    def test_pdf_render_creates_file(self, qapp, tmp_path):
        from PySide6.QtGui import QImage

        qimg = QImage(20, 20, QImage.Format.Format_RGB32)
        qimg.fill(0xFFFFFFFF)
        out = str(tmp_path / "out.pdf")
        display._render_qimage_vector(qimg, out, "pdf")
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_svg_render_creates_file(self, qapp, tmp_path):
        from PySide6.QtGui import QImage

        qimg = QImage(20, 20, QImage.Format.Format_RGB32)
        qimg.fill(0xFFFFFFFF)
        out = str(tmp_path / "out.svg")
        display._render_qimage_vector(qimg, out, "svg")
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


class TestInstallViewerSaveMenu:
    def test_installs_right_click_filter(self, qapp):
        from PySide6.QtWidgets import QWidget

        class FakeWindow(QWidget):
            pass

        mock_view = MagicMock()
        mock_camera = MagicMock()
        mock_camera.viewbox_mouse_event = MagicMock()
        mock_view.camera = mock_camera

        qv = MagicMock()
        qv.canvas.view = mock_view

        mock_viewer = MagicMock()
        mock_viewer.window = FakeWindow()
        mock_viewer.window._qt_viewer = qv
        display._install_viewer_save_menu(mock_viewer)

        for name in ("mouse_press", "mouse_move", "mouse_release"):
            emitter = getattr(mock_view.events, name)
            emitter.disconnect.assert_called_once_with(mock_camera.viewbox_mouse_event)
            emitter.connect.assert_called()

    def test_returns_early_on_bad_viewer(self, qapp):
        mock_viewer = MagicMock()
        type(mock_viewer.window).isinstance = lambda self, t: False
        display._install_viewer_save_menu(mock_viewer)


class TestSaveViewport:
    def test_captures_screenshot_and_saves(self, qapp, tmp_path):
        import numpy as np
        from PySide6.QtWidgets import QFileDialog

        mock_viewer = MagicMock()
        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        mock_viewer.screenshot.return_value = arr
        out = str(tmp_path / "shot.png")

        def _exec(self):
            return QFileDialog.Accepted

        def _selected_files(self):
            return [out]

        with (
            patch.object(QFileDialog, "exec", _exec),
            patch.object(QFileDialog, "selectedFiles", _selected_files),
        ):
            display._save_viewport(mock_viewer)
        mock_viewer.screenshot.assert_called_once_with(
            canvas_only=True,
            flash=False,
        )
        assert os.path.exists(out)

    def test_returns_early_on_screenshot_error(self, qapp):
        mock_viewer = MagicMock()
        mock_viewer.screenshot.side_effect = RuntimeError("fail")
        display._save_viewport(mock_viewer)

    def test_save_viewport_prefills_recorded_source_name(self, qapp, tmp_path):
        # The viewer records the opened file name on _source_name; the save
        # dialog must open pre-filled with that base name + .png, mirroring
        # the gallery widget behaviour.
        import numpy as np
        from PySide6.QtWidgets import QFileDialog

        mock_viewer = MagicMock()
        mock_viewer._source_name = "run1_model.star"
        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        mock_viewer.screenshot.return_value = arr
        out = str(tmp_path / "shot.png")
        captured = {}

        def _exec(self):
            return QFileDialog.Accepted

        def _selected_files(self):
            return [out]

        def _select_file(self, name):
            captured["name"] = name

        with (
            patch.object(QFileDialog, "exec", _exec),
            patch.object(QFileDialog, "selectedFiles", _selected_files),
            patch.object(QFileDialog, "selectFile", _select_file),
        ):
            display._save_viewport(mock_viewer)
        assert captured["name"] == "run1_model.png"

    def test_crops_canvas_padding_around_content(self, qapp, tmp_path):
        # The data is a small white block centred in a large black canvas;
        # the saved image must be cropped to the white block only.
        import numpy as np
        from PySide6.QtWidgets import QFileDialog

        mock_viewer = MagicMock()
        h = w = 100
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[40:60, 30:70, :3] = 255  # white content block
        arr[:, :, 3] = 255
        mock_viewer.screenshot.return_value = arr
        out = str(tmp_path / "shot.png")

        def _exec(self):
            return QFileDialog.Accepted

        def _selected_files(self):
            return [out]

        with (
            patch.object(QFileDialog, "exec", _exec),
            patch.object(QFileDialog, "selectedFiles", _selected_files),
        ):
            display._save_viewport(mock_viewer)
        from PySide6.QtGui import QImage

        saved = QImage(out)
        assert saved.height() == 20  # content height (60 - 40)
        assert saved.width() == 40  # content width (70 - 30)

    def test_crop_to_content_returns_full_image_when_no_content(self):
        import numpy as np

        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        cropped = display._crop_to_content(arr)
        assert cropped.shape == arr.shape

    def test_crop_to_content_returns_full_image_for_gray_content(self):
        # A content block that is the same colour as the background corner is
        # not cropped away (it differs nowhere meaningful, so the whole image
        # is returned rather than an empty crop).
        import numpy as np

        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        arr[2:8, 2:8, :3] = 0  # identical to background
        cropped = display._crop_to_content(arr)
        assert cropped.shape == arr.shape


class TestPanelToggle:
    def test_middle_click_toggles_panel_and_skips_camera(self, qapp):
        # Middle-click must toggle the layer panel but NOT reach the camera's
        # viewbox_mouse_event (otherwise it also pans/zooms the object).
        # VisPy reports the middle button as integer 2 (not the Qt enum value).
        napari = pytest.importorskip("napari")
        from vispy.app.canvas import MouseEvent

        viewer = napari.Viewer(show=False)
        try:
            camera = viewer.window._qt_viewer.canvas.view.camera
            # Spy on the REAL camera handler first; _install_panel_toggle will
            # capture this spy as its "original" and only call it for non-middle
            # events.
            cam_calls = []

            def _spy(ev):
                cam_calls.append(ev.type)

            camera.viewbox_mouse_event = _spy

            display._install_panel_toggle(viewer)
            wrapped = viewer.window._qt_viewer.canvas.view.camera.viewbox_mouse_event

            ev_mid = MouseEvent(type="mouse_press", button=2, pos=(5, 5))
            wrapped(ev_mid)
            assert cam_calls == []  # camera never saw the middle press

            ev_left = MouseEvent(type="mouse_press", button=1, pos=(5, 5))
            wrapped(ev_left)
            assert cam_calls == ["mouse_press"]  # left press still reaches camera
        finally:
            viewer.close()
