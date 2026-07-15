import argparse
import os
import sys
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

            mock_napari.Viewer.assert_called_once_with(title="helicon display")
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
    @patch.object(display, "_on_screen", return_value=True)
    @patch.object(display, "_read_rect")
    @patch.object(display, "_get_qsettings")
    def test_restore_geometry_applies_saved_dock(
        self, mock_get_settings, mock_read_rect, mock_on_screen
    ):
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

    @patch.object(display, "_on_screen", return_value=True)
    @patch.object(display, "_read_rect")
    @patch.object(display, "_get_qsettings")
    def test_restore_geometry_applies_saved_viewer(
        self, mock_get_settings, mock_read_rect, mock_on_screen
    ):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_read_rect.side_effect = lambda s, p: (
            (500, 600, 700, 800) if p == "viewer" else None
        )

        mock_dock = MagicMock()
        mock_dock.geometry.return_value = MagicMock(
            x=lambda: 0, y=lambda: 0, width=lambda: 250, height=lambda: 800
        )
        mock_viewer = MagicMock()
        mock_viewer.window._qt_window.geometry.return_value = MagicMock(
            x=lambda: 1200, y=lambda: 100, width=lambda: 1000, height=lambda: 800
        )

        with patch("PySide6.QtCore.QTimer") as mock_timer:
            display._restore_geometry(mock_dock, mock_viewer)
            captured_fn = mock_timer.singleShot.call_args[0][1]
            captured_fn()

        mock_viewer.window._qt_window.setGeometry.assert_called_once_with(
            500, 600, 700, 800
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

        mock_dock.restoreGeometry.assert_not_called()
        mock_position.assert_called_once_with(mock_dock, mock_viewer)

    def test_position_default_places_dock_left_of_viewer(self):
        mock_dock = MagicMock()
        mock_dock.geometry.return_value = MagicMock(
            x=lambda: 0, y=lambda: 0, width=lambda: 250, height=lambda: 800
        )

        mock_viewer = MagicMock()
        mock_viewer.window._qt_window.geometry.return_value = MagicMock(
            x=lambda: 500, y=lambda: 100, width=lambda: 1000, height=lambda: 800
        )

        display._position_default(mock_dock, mock_viewer)

        mock_dock.adjustSize.assert_called_once()
        mock_dock.setGeometry.assert_called_once_with(190, 100, 300, 800)

    def test_position_default_uses_minimum_width(self):
        mock_dock = MagicMock()
        mock_dock.geometry.return_value = MagicMock(
            x=lambda: 0, y=lambda: 0, width=lambda: 100, height=lambda: 600
        )

        mock_viewer = MagicMock()
        mock_viewer.window._qt_window.geometry.return_value = MagicMock(
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
        mock_viewer = MagicMock()
        mock_viewer.window._qt_window.geometry.return_value = MagicMock(
            x=lambda: 500, y=lambda: 600, width=lambda: 700, height=lambda: 800
        )

        display._save_geometry(mock_dock, mock_viewer)

        assert mock_write_rect.call_count == 2
        mock_write_rect.assert_any_call(mock_settings, "dock", mock_dock)
        mock_write_rect.assert_any_call(
            mock_settings, "viewer", mock_viewer.window._qt_window
        )

    @patch.object(display, "_get_qsettings")
    def test_write_rect_saves_individual_values(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_widget = MagicMock()
        mock_widget.geometry.return_value = MagicMock(
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
        mock_widget.geometry.side_effect = RuntimeError(
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
        mock_mrc.voxel_size.x = 2.0
        mock_mrc.header.mapc = 1
        mock_mrc.header.mapr = 2
        mock_mrc.header.maps = 3
        mock_mrc.header.nz = 3

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
        np.testing.assert_array_equal(call_args[0], test_data)
        assert call_kwargs["name"] == "test.mrc"
        assert call_kwargs["scale"] == (2.0, 2.0, 2.0)
        assert "contrast_limits" in call_kwargs
        assert len(call_kwargs["contrast_limits"]) == 2

    def test_open_tif_file_uses_viewer_open(self):
        mock_viewer = MagicMock()
        display._open_file(mock_viewer, "/path/to/test.tif")
        mock_viewer.open.assert_called_once_with("/path/to/test.tif")

    def test_open_png_file_uses_viewer_open(self):
        mock_viewer = MagicMock()
        display._open_file(mock_viewer, "/path/to/test.png")
        mock_viewer.open.assert_called_once_with("/path/to/test.png")


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
        headers = [model.headerData(c, Qt.Orientation.Horizontal) for c in range(6)]
        assert headers == ["Name", "Size", "Type", "Dimension", "Images", "Modified"]

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
