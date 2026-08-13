import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import helicon
from helicon.commands import display
from helicon.lib.exceptions import HeliconDependencyError

try:
    from PySide6.QtCore import Qt, QModelIndex
    from PySide6.QtWidgets import QAbstractItemView, QApplication
except ImportError:
    from PyQt5.QtCore import Qt, QModelIndex
    from PyQt5.QtWidgets import QAbstractItemView, QApplication

try:
    from helicon.lib.gui.images2star_widget import Images2StarDialog
except ImportError:
    Images2StarDialog = None


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestDisplayArgs(object):
    def test_application_identity_is_helicon(self, qapp):
        display._set_application_identity(qapp)
        assert qapp.applicationName() == "Helicon"
        assert qapp.applicationDisplayName() == "Helicon"

    def test_qt_argv_uses_helicon_as_program_name(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["python3.14", "display", "/tmp"])
        assert display._qt_argv() == ["Helicon", "display", "/tmp"]

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

    def test_stats_uses_napari_window_category(self):
        assert "stats" in display._NAPARI_MODES
        assert "stats" not in display._PLOT_MODES

    def test_helical_angle_stats_plot_opens_pdf_in_active_napari(self):
        viewer = MagicMock()
        result = {"plot_file": "/job/run_data.helical_angle_variance.pdf"}

        with (
            patch.object(display._napari, "active", return_value=viewer),
            patch.object(display, "_create_napari_viewer") as create_viewer,
            patch.object(display, "_open_file") as open_file,
        ):
            display._open_helical_angle_stats_plot(result)

        create_viewer.assert_not_called()
        open_file.assert_called_once_with(
            viewer,
            "/job/run_data.helical_angle_variance.pdf",
            mode="slice",
        )

    def test_helical_angle_stats_plot_creates_napari_when_needed(self):
        viewer = MagicMock()
        result = {"plot_file": "/job/run_data.helical_angle_variance.pdf"}

        with (
            patch.object(display._napari, "active", return_value=None),
            patch.object(display, "_create_napari_viewer", return_value=viewer),
            patch.object(display, "_open_file") as open_file,
        ):
            display._open_helical_angle_stats_plot(result)

        open_file.assert_called_once_with(
            viewer,
            "/job/run_data.helical_angle_variance.pdf",
            mode="slice",
        )

    def test_napari_theme_maps_saved_theme(self, qapp):
        from PySide6.QtCore import QSettings

        settings = QSettings("helicon", "display")
        try:
            for saved, expected in (
                ("Dark", "dark"),
                ("Light", "light"),
                ("System", "system"),
            ):
                settings.setValue("theme", saved)
                assert display._napari_display_theme() == expected
        finally:
            settings.remove("theme")

    def test_refresh_napari_theme_updates_open_viewers(self, monkeypatch, qapp):
        from PySide6.QtCore import QSettings

        settings = QSettings("helicon", "display")
        viewer = MagicMock()
        monkeypatch.setattr(display._napari, "alive", lambda: [viewer])
        settings.setValue("theme", "Light")
        try:
            display._refresh_napari_theme()
            assert viewer.theme == "light"
            assert viewer.background_color == "#ffffff"
        finally:
            settings.remove("theme")

    def test_napari_canvas_background_maps_saved_theme(self, qapp):
        from PySide6.QtCore import QSettings

        settings = QSettings("helicon", "display")
        try:
            settings.setValue("theme", "Dark")
            assert display._napari_canvas_background() == "#202020"
            settings.setValue("theme", "Light")
            assert display._napari_canvas_background() == "#ffffff"
        finally:
            settings.remove("theme")

    def test_helical_angle_stats_paths_use_writable_job_directory(self, tmp_path):
        input_star = tmp_path / "run_data.star"
        with patch.object(display.os, "access", return_value=True):
            output_dir, output_star, plot_file = display._helical_angle_stats_paths(
                str(input_star)
            )

        assert output_dir == tmp_path
        assert output_star == tmp_path / "run_data.helical_angle_variance.star"
        assert plot_file == tmp_path / "run_data.helical_angle_variance.pdf"

    def test_helical_angle_stats_paths_fall_back_to_temporary_directory(self, tmp_path):
        input_star = tmp_path / "job" / "run_data.star"
        fallback = tmp_path / "temporary-results"
        with (
            patch.object(display.os, "access", return_value=False),
            patch("tempfile.mkdtemp", return_value=str(fallback)),
        ):
            output_dir, output_star, plot_file = display._helical_angle_stats_paths(
                str(input_star)
            )

        assert output_dir == fallback
        assert output_star == fallback / "run_data.helical_angle_variance.star"
        assert plot_file == fallback / "run_data.helical_angle_variance.pdf"

    def test_model_fsc_curves_uses_ssnr_map_for_class3d(self):
        import numpy as np
        import pandas as pd

        data = {
            "model_class_1": pd.DataFrame(
                {
                    "rlnResolution": [0.3, 0.1, 0.2],
                    "rlnAngstromResolution": [3.0, 10.0, 5.0],
                    "rlnGoldStandardFsc": [0.0, 0.0, 0.0],
                    "rlnSsnrMap": [3.0, 0.0, 1.0],
                }
            )
        }

        curves = display._model_fsc_curves(data, use_ssnr_map=True)

        assert len(curves) == 1
        spatial_freq, w_map, label, angstrom = curves[0]
        np.testing.assert_allclose(spatial_freq, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(w_map, [0.0, 0.5, 0.75])
        np.testing.assert_allclose(angstrom, [10.0, 5.0, 3.0])
        assert label == "Class 1"

    def test_model_fsc_curves_keeps_gold_standard_fsc_for_refine3d(self):
        import numpy as np
        import pandas as pd

        data = {
            "model_class_1": pd.DataFrame(
                {
                    "_rlnResolution": [0.2, 0.1],
                    "_rlnGoldStandardFsc": [0.25, 0.75],
                    "_rlnSsnrMap": [9.0, 9.0],
                }
            )
        }

        curves = display._model_fsc_curves(data, use_ssnr_map=False)

        spatial_freq, fsc, _, _ = curves[0]
        np.testing.assert_allclose(spatial_freq, [0.1, 0.2])
        np.testing.assert_allclose(fsc, [0.75, 0.25])


class TestFscIterations(object):
    def test_iteration_model_files_sorts_by_number(self, tmp_path):
        (tmp_path / "run_it003_model.star").touch()
        (tmp_path / "run_it001_model.star").touch()
        (tmp_path / "run_it002_model.star").touch()
        (tmp_path / "run_it004_half2_model.star").touch()
        (tmp_path / "run_it004_half1_model.star").touch()
        (tmp_path / "run_model.star").touch()
        (tmp_path / "run_data.star").touch()
        (tmp_path / "run_it001_optimiser.star").touch()

        files = display._iteration_model_files(tmp_path / "run_it004_half2_model.star")

        assert [(p.name, n) for p, n in files] == [
            ("run_it001_model.star", 1),
            ("run_it002_model.star", 2),
            ("run_it003_model.star", 3),
            ("run_it004_half2_model.star", 4),
            ("run_model.star", None),
        ]

    def test_iteration_model_files_collapses_half_files_to_one_iteration(
        self, tmp_path
    ):
        (tmp_path / "run_it001_half1_model.star").touch()
        (tmp_path / "run_it001_half2_model.star").touch()
        (tmp_path / "run_it002_half1_model.star").touch()
        (tmp_path / "run_it002_half2_model.star").touch()
        (tmp_path / "run_model.star").touch()

        files = display._iteration_model_files(tmp_path / "run_it001_half2_model.star")

        assert [(p.name, n) for p, n in files] == [
            ("run_it001_half2_model.star", 1),
            ("run_it002_half2_model.star", 2),
            ("run_model.star", None),
        ]

    def test_iteration_model_files_ignores_unrelated_star_files(self, tmp_path):
        (tmp_path / "run_it001_model.star").touch()
        (tmp_path / "run_data.star").touch()
        (tmp_path / "run_optimiser.star").touch()
        (tmp_path / "other.star").touch()

        files = display._iteration_model_files(tmp_path / "run_it001_model.star")

        assert [p.name for p, _n in files] == ["run_it001_model.star"]

    def test_iteration_model_files_includes_unmarked_selected_file(self, tmp_path):
        (tmp_path / "run_it001_model.star").touch()
        (tmp_path / "run_model.star").touch()

        files = display._iteration_model_files(tmp_path / "run_model.star")

        assert [(p.name, n) for p, n in files] == [
            ("run_it001_model.star", 1),
            ("run_model.star", None),
        ]

    def test_iteration_model_files_returns_selected_when_alone(self, tmp_path):
        (tmp_path / "run_model.star").touch()

        files = display._iteration_model_files(tmp_path / "run_model.star")

        assert [(p.name, n) for p, n in files] == [("run_model.star", None)]

    def test_iteration_label_formats_numbered_and_final_files(self):
        assert display._iteration_label("run_it025_model.star") == "25"
        assert display._iteration_label("run_ct1_it001_model.star") == "1"
        assert display._iteration_label("run_it016_half2_model.star") == "16"
        assert display._iteration_label("run_model.star") == "final"

    def test_read_fsc_curves_returns_none_for_unreadable_file(self, tmp_path):
        assert (
            display._read_fsc_curves(tmp_path / "missing.star", use_ssnr_map=False)
            is None
        )

    def test_read_fsc_curves_parses_sibling_file(self, tmp_path):
        import numpy as np
        import pandas as pd
        import starfile

        path = tmp_path / "run_it001_model.star"
        starfile.write(
            {
                "model_class_1": pd.DataFrame(
                    {
                        "rlnResolution": [0.1, 0.2],
                        "rlnGoldStandardFsc": [0.8, 0.2],
                    }
                )
            },
            str(path),
            overwrite=True,
        )

        curves = display._read_fsc_curves(path, use_ssnr_map=False)

        assert curves is not None
        assert len(curves) == 1
        spatial_freq, fsc, label, _angstrom = curves[0]
        np.testing.assert_allclose(spatial_freq, [0.1, 0.2])
        np.testing.assert_allclose(fsc, [0.8, 0.2])
        assert label == "Class 1"

    def test_fsc_window_defaults_to_selected_iteration(self, qapp, tmp_path):
        iterations = [
            (tmp_path / "run_it001_model.star", 1),
            (tmp_path / "run_it002_model.star", 2),
            (tmp_path / "run_it003_model.star", 3),
        ]
        curves = {
            path: [(np.array([0.1, 0.2]), np.array([0.8, 0.2]), "Class 1", None)]
            for path, _num in iterations
        }
        window = display._FscPlotWindow(
            iterations,
            curves,
            default_path=iterations[1][0],
            is_class3d=False,
        )
        try:
            assert [checkbox.isChecked() for checkbox in window._checkboxes] == [
                False,
                True,
                False,
            ]
            assert window._checked_paths() == [iterations[1][0]]
        finally:
            window.close()

    def test_fsc_plot_canvas_follows_saved_theme(self, qapp, tmp_path):
        from PySide6.QtCore import QSettings

        path = tmp_path / "run_it001_model.star"
        curves = {
            path: [
                (
                    np.array([0.1, 0.2]),
                    np.array([0.8, 0.2]),
                    "Class 1",
                    None,
                )
            ]
        }
        settings = QSettings("helicon", "display")
        settings.setValue("theme", "Light")
        window = display._FscPlotWindow(
            [(path, 1)],
            curves,
            default_path=path,
            is_class3d=False,
        )
        try:
            assert window.plot_widget.backgroundBrush().color().name() == "#ffffff"
            assert (
                window.plot_widget.getPlotItem()
                .getAxis("bottom")
                .textPen()
                .color()
                .name()
                == "#202020"
            )
        finally:
            window.close()
            settings.remove("theme")

    def test_fsc_controls_follow_saved_theme(self, qapp, tmp_path):
        from PySide6.QtCore import QSettings

        path = tmp_path / "run_it001_model.star"
        curves = {
            path: [
                (
                    np.array([0.1, 0.2]),
                    np.array([0.8, 0.2]),
                    "Class 1",
                    None,
                )
            ]
        }
        settings = QSettings("helicon", "display")
        settings.setValue("theme", "Dark")
        window = display._FscPlotWindow(
            [(path, 1)],
            curves,
            default_path=path,
            is_class3d=False,
        )
        try:
            assert window.palette().window().color().name() == "#2d2d2d"
            assert (
                window._iter_select_all_btn.palette().button().color().name()
                == "#3c3c3c"
            )
        finally:
            window.close()
            settings.remove("theme")

    def test_fsc_button_width_fits_all_and_none_labels(self, qapp, tmp_path):
        path = tmp_path / "run_it001_model.star"
        curves = {
            path: [
                (
                    np.array([0.1, 0.2]),
                    np.array([0.8, 0.2]),
                    "Class 1",
                    None,
                )
            ]
        }
        window = display._FscPlotWindow(
            [(path, 1)],
            curves,
            default_path=path,
            is_class3d=False,
        )
        try:
            assert (
                window._iter_select_all_btn.width()
                >= window._iter_unselect_all_btn.fontMetrics().horizontalAdvance("none")
            )
        finally:
            window.close()

    def test_fsc_window_falls_back_to_first_when_selected_missing(self, qapp, tmp_path):
        iterations = [(tmp_path / "run_it001_model.star", 1)]
        curves = {
            path: [(np.array([0.1, 0.2]), np.array([0.8, 0.2]), "Class 1", None)]
            for path, _num in iterations
        }
        window = display._FscPlotWindow(
            iterations,
            curves,
            default_path=tmp_path / "missing.star",
            is_class3d=False,
        )
        try:
            assert window._checked_paths() == [iterations[0][0]]
        finally:
            window.close()

    def test_fsc_window_select_and_unselect_all(self, qapp, tmp_path):
        iterations = [
            (tmp_path / "run_it001_model.star", 1),
            (tmp_path / "run_it002_model.star", 2),
            (tmp_path / "run_it003_model.star", 3),
        ]
        curves = {
            path: [(np.array([0.1, 0.2]), np.array([0.8, 0.2]), "Class 1", None)]
            for path, _num in iterations
        }
        window = display._FscPlotWindow(
            iterations,
            curves,
            default_path=iterations[0][0],
            is_class3d=False,
        )
        try:
            window._select_all()
            assert [checkbox.isChecked() for checkbox in window._checkboxes] == [
                True,
                True,
                True,
            ]
            assert window._checked_paths() == [p for p, _n in iterations]

            window._unselect_all()
            assert [checkbox.isChecked() for checkbox in window._checkboxes] == [
                False,
                False,
                False,
            ]
            assert window._checked_paths() == []
        finally:
            window.close()

    def test_fsc_window_rebuilds_curves_on_toggle(self, qapp, tmp_path):
        it1 = tmp_path / "run_it001_model.star"
        it2 = tmp_path / "run_it002_model.star"
        iterations = [(it1, 1), (it2, 2)]
        curves = {
            path: [(np.array([0.1, 0.2]), np.array([0.8, 0.2]), "Class 1", None)]
            for path, _num in iterations
        }
        window = display._FscPlotWindow(
            iterations, curves, default_path=it1, is_class3d=False
        )
        try:
            legend = window.plot_widget.getPlotItem().legend

            def _labels():
                return [item[1].item.toPlainText() for item in legend.items]

            # Default: only the selected iteration is plotted.
            assert len(window._curve_items) == 1
            assert _labels() == ["1"]

            window._select_all()
            assert len(window._curve_items) == 2
            assert _labels() == ["1", "2"]

            window._checkboxes[0].setChecked(False)
            assert len(window._curve_items) == 1
            assert _labels() == ["2"]

            window._select_all()
            assert len(window._curve_items) == 2
            assert _labels() == ["1", "2"]
        finally:
            window.close()

    def test_refine3d_plots_one_curve_per_iteration(self, qapp, tmp_path):
        it1 = tmp_path / "run_it001_model.star"
        it2 = tmp_path / "run_it002_model.star"
        iterations = [(it1, 1), (it2, 2)]
        curves = {
            path: [
                (
                    np.array([0.1, 0.2]),
                    np.array([0.8, 0.2]),
                    "Half 1",
                    None,
                ),
                (
                    np.array([0.1, 0.2]),
                    np.array([0.7, 0.3]),
                    "Half 2",
                    None,
                ),
            ]
            for path, _num in iterations
        }
        window = display._FscPlotWindow(
            iterations, curves, default_path=it1, is_class3d=False
        )
        try:
            # Refine3D stores two half-set curves, but the plot should show
            # only one curve for each selected iteration.
            assert len(window._curve_items) == 1
            window._select_all()
            assert len(window._curve_items) == 2
        finally:
            window.close()

    def test_fsc_window_many_iterations_stays_narrow_and_scrolls(self, qapp, tmp_path):
        iterations = [
            (tmp_path / f"run_it{i:03d}_model.star", i) for i in range(1, 301)
        ]
        curves = {
            path: [(np.array([0.1, 0.2]), np.array([0.8, 0.2]), "Class 1", None)]
            for path, _num in iterations
        }
        window = display._FscPlotWindow(
            iterations,
            curves,
            default_path=iterations[0][0],
            is_class3d=False,
        )
        window.show()
        qapp.processEvents()
        try:
            assert len(window._checkboxes) == 300
            # The wrapping scroll strip must keep the window a normal width
            # instead of stretching it to fit every iteration checkbox.
            assert window.width() < 1000
            assert window._iter_scroll.maximumHeight() < 80
            assert window._iter_scroll.verticalScrollBar().maximum() > 0
            # The "Iterations:" label text lines up with the checkbox row.
            from PySide6.QtCore import QPoint

            iter_label = window._bar_layout.itemAtPosition(0, 0).widget()
            label_center = iter_label.mapTo(
                window, QPoint(0, iter_label.height() // 2)
            ).y()
            checkbox_center = (
                window._checkboxes[0]
                .mapTo(window, QPoint(0, window._checkboxes[0].height() // 2))
                .y()
            )
            assert abs(label_center - checkbox_center) <= 2
            # The checkboxes must wrap onto multiple rows inside the strip,
            # and the strip must not be wider than its viewport (no
            # off-screen clipping with the horizontal scrollbar disabled).
            wrapped_rows = {
                checkbox.geometry().y()
                for checkbox in window._checkboxes
                if checkbox.geometry().height() > 0
            }
            assert len(wrapped_rows) > 1
            assert window._iter_scroll.widget().width() <= (
                window._iter_scroll.viewport().width() + 1
            )
            # The "all"/"none" shortcut buttons trail the checkbox strip.
            assert window._iter_select_all_btn.text() == "all"
            assert window._iter_unselect_all_btn.text() == "none"
            assert window._iter_select_all_btn.height() >= 25
            assert window._iter_unselect_all_btn.height() >= 25
            assert (
                window._iter_select_all_btn.height() >= window._checkboxes[0].height()
            )
            assert (
                window._iter_unselect_all_btn.height() >= window._checkboxes[0].height()
            )
            assert (
                window._iter_flow.itemAt(window._iter_flow.count() - 1).widget()
                is window._iter_button_group
            )
            assert (
                window._iter_select_all_btn.parentWidget() is window._iter_button_group
            )
            assert (
                window._iter_unselect_all_btn.parentWidget()
                is window._iter_button_group
            )

            window._select_all()
            assert len(window._checked_paths()) == 300
            window._unselect_all()
            assert window._checked_paths() == []
        finally:
            window.close()

    def test_fsc_window_two_iteration_rows_do_not_show_scrollbar(self, qapp, tmp_path):
        iterations = [(tmp_path / f"run_it{i:03d}_model.star", i) for i in range(1, 5)]
        curves = {
            path: [(np.array([0.1, 0.2]), np.array([0.8, 0.2]), "Class 1", None)]
            for path, _num in iterations
        }
        window = display._FscPlotWindow(
            iterations,
            curves,
            default_path=iterations[0][0],
            is_class3d=False,
        )
        window.show()
        qapp.processEvents()
        try:
            assert window._iter_scroll.verticalScrollBar().maximum() == 0
            assert window._iter_scroll.height() >= (
                window._checkboxes[0].height() * 2 + window._iter_flow._spacing
            )
        finally:
            window.close()

    def test_fsc_window_has_larger_default_size(self, qapp, tmp_path):
        model = tmp_path / "run_it001_model.star"
        curves = {
            model: [(np.array([0.1, 0.2]), np.array([0.8, 0.2]), "Class 1", None)]
        }
        window = display._FscPlotWindow(
            [(model, 1)],
            curves,
            default_path=model,
            is_class3d=False,
        )
        try:
            assert window.size().width() == 900
            assert window.size().height() == 600
        finally:
            window.close()

    def test_fsc_window_many_classes_wraps_and_scrolls(self, qapp, tmp_path):
        it1 = tmp_path / "run_it001_model.star"
        iterations = [(it1, 1)]
        curves = {
            it1: [
                (
                    np.array([0.1, 0.2]),
                    np.array([0.8, 0.2]),
                    f"Class {index + 1}",
                    None,
                )
                for index in range(40)
            ]
        }
        window = display._FscPlotWindow(
            iterations, curves, default_path=it1, is_class3d=True
        )
        window.show()
        qapp.processEvents()
        try:
            assert len(window._class_checkboxes) == 40
            # Many classes must wrap onto multiple rows instead of forcing
            # the window wide on a single row.
            assert window.width() < 1000
            assert window._class_scroll.verticalScrollBar().maximum() > 0
            wrapped_rows = {
                checkbox.geometry().y()
                for checkbox in window._class_checkboxes
                if checkbox.geometry().height() > 0
            }
            assert len(wrapped_rows) > 1
            assert window._class_scroll.widget().width() <= (
                window._class_scroll.viewport().width() + 1
            )
            assert window._checked_class_indices() == list(range(40))

            # Many classes get Select All / Unselect All buttons, trailing
            # the class checkboxes, with short "all" / "none" labels.
            assert window._class_select_all_btn.text() == "all"
            assert window._class_unselect_all_btn.text() == "none"
            assert window._class_select_all_btn.height() >= 25
            assert window._class_unselect_all_btn.height() >= 25
            assert (
                window._class_select_all_btn.height()
                >= window._class_checkboxes[0].height()
            )
            assert (
                window._class_unselect_all_btn.height()
                >= window._class_checkboxes[0].height()
            )
            assert window._class_select_all_btn.isVisible()
            assert window._class_unselect_all_btn.isVisible()
            assert (
                window._class_flow.itemAt(window._class_flow.count() - 1).widget()
                is window._class_button_group
            )
            assert (
                window._class_select_all_btn.parentWidget()
                is window._class_button_group
            )
            assert (
                window._class_unselect_all_btn.parentWidget()
                is window._class_button_group
            )
            from PySide6.QtCore import QPoint

            def _center_y(widget):
                return widget.mapTo(window, QPoint(0, widget.height() // 2)).y()

            assert _center_y(window._class_select_all_btn) == _center_y(
                window._class_unselect_all_btn
            )
            assert (
                abs(
                    _center_y(window._class_select_all_btn)
                    - _center_y(window._class_checkboxes[-1])
                )
                <= 2
            )

            window._unselect_all_classes()
            assert window._checked_class_indices() == []
            window._select_all_classes()
            assert window._checked_class_indices() == list(range(40))
            assert len(window._curve_items) == 40
        finally:
            window.close()

    def test_fsc_window_hides_class_row_for_non_class3d(self, qapp, tmp_path):
        it1 = tmp_path / "run_it001_model.star"
        iterations = [(it1, 1)]
        curves = {
            it1: [
                (np.array([0.1, 0.2]), np.array([0.8, 0.2]), "Class 1", None),
                (np.array([0.1, 0.2]), np.array([0.7, 0.3]), "Class 2", None),
            ]
        }
        window = display._FscPlotWindow(
            iterations, curves, default_path=it1, is_class3d=False
        )
        try:
            assert window._class_checkboxes == []
            assert window._class_row.isHidden()
            assert not window._class_select_all_btn.isVisible()
            assert not window._class_unselect_all_btn.isVisible()
        finally:
            window.close()

    def test_fsc_window_class3d_class_checkboxes_filter_curves(self, qapp, tmp_path):
        it1 = tmp_path / "run_it001_model.star"
        it2 = tmp_path / "run_it002_model.star"
        iterations = [(it1, 1), (it2, 2)]
        curves = {
            path: [
                (np.array([0.1, 0.2]), np.array([0.8, 0.2]), "Class 1", None),
                (np.array([0.1, 0.2]), np.array([0.7, 0.3]), "Class 2", None),
            ]
            for path, _num in iterations
        }
        window = display._FscPlotWindow(
            iterations, curves, default_path=it1, is_class3d=True
        )
        window.show()
        qapp.processEvents()
        try:
            legend = window.plot_widget.getPlotItem().legend

            def _labels():
                return [item[1].item.toPlainText() for item in legend.items]

            from PySide6.QtCore import QPoint

            def _center_y(widget):
                return widget.mapTo(window, QPoint(0, widget.height() // 2)).y()

            # One checkbox per class, all checked by default, row visible.
            assert [checkbox.text() for checkbox in window._class_checkboxes] == [
                "1",
                "2",
            ]
            assert all(checkbox.isChecked() for checkbox in window._class_checkboxes)
            assert not window._class_row.isHidden()
            assert window._checked_class_indices() == [0, 1]
            # The "Classes:" label text lines up with the checkbox row instead
            # of being centered against the taller scroll strip.
            class_label = window._class_bar_layout.itemAtPosition(0, 0).widget()
            assert (
                abs(_center_y(class_label) - _center_y(window._class_checkboxes[0]))
                <= 2
            )
            # The label and controls occupy two top-aligned columns, so the
            # label remains stable when the window's vertical size changes.
            for height in (350, 700, 500):
                window.resize(700, height)
                qapp.processEvents()
                assert class_label.geometry().top() == 0
                assert window._class_scroll.geometry().top() == 0
            # Few classes: no Select All / Unselect All buttons needed.
            assert not window._class_select_all_btn.isVisible()
            assert not window._class_unselect_all_btn.isVisible()

            # Default: selected iteration only, both classes.
            assert len(window._curve_items) == 2
            assert _labels() == ["1 · Class 1", "1 · Class 2"]

            window._select_all()
            assert len(window._curve_items) == 4
            assert _labels() == [
                "1 · Class 1",
                "1 · Class 2",
                "2 · Class 1",
                "2 · Class 2",
            ]

            # Uncheck Class 2: only Class 1 curves across both iterations.
            window._class_checkboxes[1].setChecked(False)
            assert window._checked_class_indices() == [0]
            assert len(window._curve_items) == 2
            assert _labels() == ["1 · Class 1", "2 · Class 1"]
        finally:
            window.close()

    @staticmethod
    def _write_model_star(path, fsc_values):
        import pandas as pd
        import starfile

        starfile.write(
            {
                "model_class_1": pd.DataFrame(
                    {
                        "rlnResolution": [0.1, 0.2],
                        "rlnGoldStandardFsc": fsc_values,
                    }
                )
            },
            str(path),
            overwrite=True,
        )

    @patch.object(display, "_install_window_shortcuts")
    def test_open_fsc_plot_plots_all_picked_iterations(
        self, mock_shortcuts, qapp, tmp_path
    ):
        it1 = tmp_path / "run_it001_model.star"
        it2 = tmp_path / "run_it002_model.star"
        self._write_model_star(it1, [0.8, 0.2])
        self._write_model_star(it2, [0.9, 0.1])

        display._open_fsc_plot(str(it1))
        try:
            windows = list(display._plot.alive())
            assert len(windows) == 1
            window = windows[0]
            assert window.windowTitle() == "FSC — run_it001_model.star (2 iterations)"
            assert [checkbox.isChecked() for checkbox in window._checkboxes] == [
                True,
                False,
            ]
            legend = window.plot_widget.getPlotItem().legend
            labels = [item[1].item.toPlainText() for item in legend.items]
            assert labels == ["1"]
            assert len(window._curve_items) == 1

            # Toggle the second iteration on, then play with the buttons.
            window._checkboxes[1].setChecked(True)
            labels = [item[1].item.toPlainText() for item in legend.items]
            assert labels == ["1", "2"]
            assert len(window._curve_items) == 2

            window._unselect_all()
            assert len(window._curve_items) == 0
            window._select_all()
            assert len(window._curve_items) == 2
        finally:
            for window in list(display._plot.alive()):
                window.close()

    @patch.object(display, "_install_window_shortcuts")
    def test_open_fsc_plot_single_iteration_uses_iteration_label(
        self, mock_shortcuts, qapp, tmp_path
    ):
        model = tmp_path / "run_it001_model.star"
        self._write_model_star(model, [0.8, 0.2])

        display._open_fsc_plot(str(model))
        try:
            windows = list(display._plot.alive())
            assert len(windows) == 1
            window = windows[0]
            assert window.windowTitle() == "FSC — run_it001_model.star"
            assert [checkbox.isChecked() for checkbox in window._checkboxes] == [True]
            legend = window.plot_widget.getPlotItem().legend
            labels = [item[1].item.toPlainText() for item in legend.items]
            assert labels == ["1"]
        finally:
            for window in list(display._plot.alive()):
                window.close()

    @patch.object(display, "_install_window_shortcuts")
    def test_open_fsc_plot_reuses_active_window(self, mock_shortcuts, qapp, tmp_path):
        job_a = tmp_path / "jobA"
        job_b = tmp_path / "jobB"
        job_a.mkdir()
        job_b.mkdir()
        model_a = job_a / "run_it001_model.star"
        model_b1 = job_b / "run_it001_model.star"
        model_b2 = job_b / "run_it002_model.star"
        self._write_model_star(model_a, [0.8, 0.2])
        self._write_model_star(model_b1, [0.8, 0.2])
        self._write_model_star(model_b2, [0.9, 0.1])

        display._open_fsc_plot(str(model_a))
        try:
            windows = list(display._plot.alive())
            assert len(windows) == 1
            first = windows[0]
            assert first.windowTitle() == "FSC — run_it001_model.star"
            assert len(first._checkboxes) == 1

            display._open_fsc_plot(str(model_b1), reuse_window=first)
            assert list(display._plot.alive()) == [first]
            assert first.windowTitle() == "FSC — run_it001_model.star (2 iterations)"
            assert len(first._checkboxes) == 2
            assert [checkbox.isChecked() for checkbox in first._checkboxes] == [
                True,
                False,
            ]
        finally:
            for window in list(display._plot.alive()):
                window.close()

    @patch.object(display, "_install_window_shortcuts")
    def test_open_fsc_plot_class3d_uses_wmap_curves(
        self, mock_shortcuts, qapp, tmp_path
    ):
        import pandas as pd
        import starfile

        job = tmp_path / "Class3D" / "job001"
        job.mkdir(parents=True)
        it1 = job / "run_it001_model.star"
        it2 = job / "run_it002_model.star"
        for path, ssnr in ((it1, [2.0, 0.5]), (it2, [3.0, 1.0])):
            starfile.write(
                {
                    "model_class_1": pd.DataFrame(
                        {
                            "rlnResolution": [0.1, 0.2],
                            "rlnSsnrMap": ssnr,
                        }
                    ),
                    "model_class_2": pd.DataFrame(
                        {
                            "rlnResolution": [0.1, 0.2],
                            "rlnSsnrMap": [v * 0.8 for v in ssnr],
                        }
                    ),
                },
                str(path),
                overwrite=True,
            )

        display._open_fsc_plot(str(it1))
        try:
            windows = list(display._plot.alive())
            assert len(windows) == 1
            window = windows[0]
            plot = window.plot_widget.getPlotItem()
            assert plot.axes["left"]["item"].labelText == "W_MAP"

            # One checkbox per class, all checked; default plots only the
            # selected iteration with W_MAP = SSNR/(1+SSNR).
            assert [checkbox.text() for checkbox in window._class_checkboxes] == [
                "1",
                "2",
            ]
            assert len(window._curve_items) == 2
            np.testing.assert_allclose(
                window._curve_items[0].yData, [2.0 / 3.0, 0.5 / 1.5]
            )
            np.testing.assert_allclose(
                window._curve_items[1].yData, [1.6 / 2.6, 0.4 / 1.4]
            )

            window._checkboxes[1].setChecked(True)
            assert len(window._curve_items) == 4
            np.testing.assert_allclose(
                window._curve_items[2].yData, [3.0 / 4.0, 1.0 / 2.0]
            )
            labels = [item[1].item.toPlainText() for item in plot.legend.items]
            assert labels == [
                "1 · Class 1",
                "1 · Class 2",
                "2 · Class 1",
                "2 · Class 2",
            ]

            # Uncheck Class 2: only Class 1 curves remain.
            window._class_checkboxes[1].setChecked(False)
            assert len(window._curve_items) == 2
            labels = [item[1].item.toPlainText() for item in plot.legend.items]
            assert labels == ["1 · Class 1", "2 · Class 1"]
        finally:
            for window in list(display._plot.alive()):
                window.close()

    @patch.object(display, "_install_window_shortcuts")
    def test_open_fsc_plot_reuse_switches_job_type(
        self, mock_shortcuts, qapp, tmp_path
    ):
        import pandas as pd
        import starfile

        refine_dir = tmp_path / "Refine3D" / "job001"
        class3d_dir = tmp_path / "Class3D" / "job002"
        refine_dir.mkdir(parents=True)
        class3d_dir.mkdir(parents=True)
        refine_model = refine_dir / "run_it001_model.star"
        class3d_model = class3d_dir / "run_it001_model.star"

        starfile.write(
            {
                "model_class_1": pd.DataFrame(
                    {
                        "rlnResolution": [0.1, 0.2],
                        "rlnGoldStandardFsc": [0.8, 0.2],
                    }
                )
            },
            str(refine_model),
            overwrite=True,
        )
        starfile.write(
            {
                "model_class_1": pd.DataFrame(
                    {
                        "rlnResolution": [0.1, 0.2],
                        "rlnSsnrMap": [2.0, 0.5],
                    }
                ),
                "model_class_2": pd.DataFrame(
                    {
                        "rlnResolution": [0.1, 0.2],
                        "rlnSsnrMap": [1.6, 0.4],
                    }
                ),
            },
            str(class3d_model),
            overwrite=True,
        )

        display._open_fsc_plot(str(refine_model))
        try:
            windows = list(display._plot.alive())
            assert len(windows) == 1
            first = windows[0]
            plot = first.plot_widget.getPlotItem()
            assert plot.axes["left"]["item"].labelText == "FSC"
            assert first._class_checkboxes == []

            # Reuse the window for a Class3D job: label, class checkboxes and
            # curves all switch to the new job.
            display._open_fsc_plot(str(class3d_model), reuse_window=first)
            assert list(display._plot.alive()) == [first]
            assert plot.axes["left"]["item"].labelText == "W_MAP"
            assert [checkbox.text() for checkbox in first._class_checkboxes] == [
                "1",
                "2",
            ]
            assert not first._class_row.isHidden()
            assert len(first._curve_items) == 2
            np.testing.assert_allclose(
                first._curve_items[0].yData, [2.0 / 3.0, 0.5 / 1.5]
            )
            np.testing.assert_allclose(
                first._curve_items[1].yData, [1.6 / 2.6, 0.4 / 1.4]
            )
        finally:
            for window in list(display._plot.alive()):
                window.close()


class TestInstallWindowShortcuts(object):
    def _quit_shortcuts(self, window):
        from PySide6.QtGui import QKeySequence, QShortcut

        quit_key = QKeySequence(QKeySequence.StandardKey.Quit)
        return [sc for sc in window.findChildren(QShortcut) if sc.key() == quit_key]

    def test_installs_close_and_quit_shortcuts_when_none_bound(self, qapp):
        from PySide6.QtGui import QKeySequence, QShortcut
        from PySide6.QtWidgets import QMainWindow

        win = QMainWindow()
        display._install_window_shortcuts(win)

        keys = [sc.key() for sc in win.findChildren(QShortcut)]
        assert QKeySequence("Ctrl+W") in keys
        assert QKeySequence(QKeySequence.StandardKey.Quit) in keys

    def test_skips_quit_shortcut_when_action_already_binds_it(self, qapp):
        from PySide6.QtGui import QAction, QKeySequence
        from PySide6.QtWidgets import QMainWindow

        win = QMainWindow()
        action = QAction("Quit", win)
        action.setShortcut(QKeySequence.StandardKey.Quit)
        win.menuBar().addMenu("File").addAction(action)

        display._install_window_shortcuts(win)
        assert self._quit_shortcuts(win) == []

    def test_browser_window_gets_no_duplicate_quit_shortcut(self, tmp_path, qapp):
        from PySide6.QtGui import QKeySequence, QShortcut

        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        # The browser owns its File -> Quit action (Ctrl+Q); the generic
        # window shortcuts must not add a second binding for it.
        display._install_window_shortcuts(widget)
        assert self._quit_shortcuts(widget) == []
        # Ctrl+W (close) is still installed for the browser window.
        keys = [sc.key() for sc in widget.findChildren(QShortcut)]
        assert QKeySequence("Ctrl+W") in keys


class TestDisplayMain(object):
    def test_has_napari_does_not_import_napari(self, monkeypatch):
        import importlib.util

        monkeypatch.delitem(sys.modules, "napari", raising=False)
        with patch.object(importlib.util, "find_spec", return_value=None):
            assert helicon.has_napari() is False
        assert "napari" not in sys.modules

    @patch.object(display, "_prepare_process_identity")
    @patch.object(display, "_set_application_identity")
    @patch.object(display, "_install_dock_save_hook")
    @patch.object(display, "_restore_geometry")
    @patch.object(display, "FolderBrowserWidget")
    @patch("PySide6.QtWidgets.QApplication.exec", return_value=0)
    def test_main_starts_without_importing_napari(
        self,
        mock_app_exec,
        mock_widget_class,
        mock_restore,
        mock_dock_save_hook,
        mock_set_identity,
        mock_prepare_identity,
    ):
        mock_widget = MagicMock()
        mock_widget_class.return_value = mock_widget
        parser = argparse.ArgumentParser()
        display.add_args(parser)
        args = parser.parse_args([])
        with patch.dict(sys.modules, {"napari": None}):
            display.main(args)
        mock_app_exec.assert_called_once()
        mock_widget_class.assert_called_once_with(start_dir=os.getcwd())

    @patch.object(display, "_prepare_process_identity")
    @patch.object(display, "_set_application_identity")
    @patch.object(display, "_install_dock_save_hook")
    @patch.object(display, "_restore_geometry")
    @patch.object(display, "FolderBrowserWidget")
    @patch("PySide6.QtWidgets.QApplication.exec", return_value=0)
    def test_main_skips_macos_menu_tricks_on_other_platforms(
        self,
        mock_app_exec,
        mock_widget_class,
        mock_restore,
        mock_dock_save_hook,
        mock_set_identity,
        mock_prepare_identity,
        monkeypatch,
    ):
        """Non-mac platforms must never run the macOS menu/activation code."""
        monkeypatch.setattr(display.sys, "platform", "linux")
        mock_widget = MagicMock()
        mock_widget_class.return_value = mock_widget
        parser = argparse.ArgumentParser()
        display.add_args(parser)
        args = parser.parse_args([])
        with (
            patch.object(display, "_set_macos_app_identity") as mock_identity,
            patch.object(display, "_force_macos_menu_realization") as mock_force,
        ):
            with patch.dict(sys.modules, {"napari": None}):
                display.main(args)
        mock_identity.assert_not_called()
        mock_force.assert_not_called()
        mock_app_exec.assert_called_once()

    def test_force_macos_menu_realization_noop_on_other_platforms(self, monkeypatch):
        """The activation helper must be inert when not on macOS."""
        monkeypatch.setattr(display.sys, "platform", "linux")
        with (
            patch.object(display, "_macos_ns_app") as mock_ns_app,
            patch.object(display, "_macos_resign_active") as mock_resign,
            patch.object(display, "_macos_activate_and_front") as mock_front,
        ):
            result = display._force_macos_menu_realization(full_cycle=True)
        assert result is None
        mock_ns_app.assert_not_called()
        mock_resign.assert_not_called()
        mock_front.assert_not_called()

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

    @patch.object(display, "_install_dock_save_hook")
    @patch.object(display, "_restore_geometry")
    @patch("helicon.has_napari", return_value=True)
    @patch.object(display, "FolderBrowserWidget")
    @patch("PySide6.QtWidgets.QApplication.exec", return_value=0)
    def test_main_creates_dock_without_viewer(
        self,
        mock_app_exec,
        mock_widget_class,
        mock_has_napari,
        mock_restore,
        mock_dock_save_hook,
    ):
        mock_napari = MagicMock()
        mock_widget = MagicMock()
        mock_widget_class.return_value = mock_widget

        sys.modules["napari"] = mock_napari
        try:
            parser = argparse.ArgumentParser()
            display.add_args(parser)
            args = parser.parse_args([])
            display.main(args)

            mock_napari.Viewer.assert_not_called()
            mock_widget_class.assert_called_once_with(start_dir=os.getcwd())
            mock_widget.setWindowFlags.assert_called_once()
            mock_widget.show.assert_called_once()
            mock_app_exec.assert_called_once()
        finally:
            del sys.modules["napari"]

    @patch("helicon.has_napari", return_value=True)
    @patch.object(display, "FolderBrowserWidget")
    @patch("PySide6.QtWidgets.QApplication.exec", return_value=0)
    def test_main_restores_and_saves_geometry(
        self, mock_app_exec, mock_widget_class, mock_has_napari
    ):
        mock_napari = MagicMock()
        mock_widget = MagicMock()
        mock_widget_class.return_value = mock_widget

        sys.modules["napari"] = mock_napari
        try:
            parser = argparse.ArgumentParser()
            display.add_args(parser)
            args = parser.parse_args([])

            with (
                patch.object(display, "_restore_geometry") as mock_restore,
                patch.object(display, "_install_dock_save_hook") as mock_save,
            ):
                display.main(args)

                mock_restore.assert_called_once_with(mock_widget, None)
                mock_save.assert_called_once_with(mock_widget)
        finally:
            del sys.modules["napari"]

    @patch.object(display, "_install_dock_save_hook")
    @patch.object(display, "_restore_geometry")
    @patch("helicon.has_napari", return_value=True)
    @patch.object(display, "FolderBrowserWidget")
    @patch("PySide6.QtWidgets.QApplication.exec", return_value=0)
    def test_main_uses_provided_folder(
        self,
        mock_app_exec,
        mock_widget_class,
        mock_has_napari,
        mock_restore,
        mock_dock_save_hook,
    ):
        mock_napari = MagicMock()
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

    @patch.object(display, "_install_dock_save_hook")
    @patch.object(display, "_restore_geometry")
    @patch("helicon.has_napari", return_value=True)
    @patch.object(display, "FolderBrowserWidget")
    @patch("PySide6.QtWidgets.QApplication.exec", return_value=0)
    def test_main_connects_file_selected_signal(
        self,
        mock_app_exec,
        mock_widget_class,
        mock_has_napari,
        mock_restore,
        mock_dock_save_hook,
    ):
        mock_napari = MagicMock()
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


class TestTextWindowEditing(object):
    """The text window is an editor: editable buffer, Save As…, prompts."""

    def _open(self, tmp_path, name="notes.txt", content="hello"):
        p = tmp_path / name
        p.write_text(content)
        win = display._open_text_window(str(p))
        assert win is not None
        return p, win

    def _append_text(self, win, text):
        from PySide6.QtGui import QTextCursor

        cursor = win._text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        win._text_edit.setTextCursor(cursor)
        win._text_edit.insertPlainText(text)

    def test_text_window_is_editable(self, qapp, tmp_path):
        _, win = self._open(tmp_path)
        try:
            assert win._text_edit.isReadOnly() is False
        finally:
            win.close()

    def test_edit_marks_title_modified(self, qapp, tmp_path):
        _, win = self._open(tmp_path)
        try:
            self._append_text(win, " more")
            assert win._text_edit.document().isModified()
            assert " *" in win.windowTitle()
        finally:
            win._text_edit.document().setModified(False)
            win.close()

    def test_save_as_writes_new_file_and_keeps_original(self, qapp, tmp_path):
        p, win = self._open(tmp_path, content="hello")
        try:
            target = tmp_path / "notes_edited.txt"
            with patch(
                "PySide6.QtWidgets.QFileDialog.getSaveFileName",
                return_value=(str(target), "All Files (*)"),
            ):
                win._save_as()
            assert target.read_text() == "hello"
            assert p.read_text() == "hello"
            assert not win._text_edit.document().isModified()
            assert "*" not in win.windowTitle()
            assert win._source_path == str(target)
        finally:
            win.close()

    def test_close_prompts_and_can_save(self, qapp, tmp_path):
        from PySide6.QtWidgets import QMessageBox

        p, win = self._open(tmp_path, content="hello")
        try:
            self._append_text(win, " world")
            target = tmp_path / "saved.txt"
            with (
                patch.object(
                    QMessageBox,
                    "question",
                    return_value=QMessageBox.StandardButton.Save,
                ),
                patch(
                    "PySide6.QtWidgets.QFileDialog.getSaveFileName",
                    return_value=(str(target), "All Files (*)"),
                ),
            ):
                win.close()
            assert not win.isVisible()
            assert target.read_text() == "hello world"
            assert p.read_text() == "hello"
        finally:
            if win.isVisible():
                win.close()

    def test_close_cancel_keeps_window(self, qapp, tmp_path):
        from PySide6.QtWidgets import QMessageBox

        p, win = self._open(tmp_path, content="hello")
        try:
            self._append_text(win, " world")
            with patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.StandardButton.Cancel,
            ):
                win.close()
            assert win.isVisible()
            assert "hello world" in win._text_edit.toPlainText()
        finally:
            win._text_edit.document().setModified(False)
            win.close()

    def test_close_discard_keeps_original(self, qapp, tmp_path):
        from PySide6.QtWidgets import QMessageBox

        p, win = self._open(tmp_path, content="hello")
        try:
            self._append_text(win, " world")
            with patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.StandardButton.Discard,
            ):
                win.close()
            assert not win.isVisible()
            assert p.read_text() == "hello"
        finally:
            if win.isVisible():
                win.close()

    def test_reuse_prompts_before_replacing_unsaved(self, qapp, tmp_path):
        from PySide6.QtWidgets import QMessageBox

        _, win = self._open(tmp_path, name="a.txt", content="aaa")
        try:
            self._append_text(win, " (edited)")
            p2 = tmp_path / "b.txt"
            p2.write_text("bbb")
            with patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.StandardButton.Discard,
            ):
                display._open_text_window(str(p2), reuse_window=win)
            assert win._text_edit.toPlainText() == "bbb"
            assert win._source_path == str(p2)
            assert "b.txt" in win.windowTitle()
            assert "*" not in win.windowTitle()
        finally:
            win.close()

    def test_reuse_cancel_keeps_current_buffer(self, qapp, tmp_path):
        from PySide6.QtWidgets import QMessageBox

        _, win = self._open(tmp_path, name="a.txt", content="aaa")
        try:
            self._append_text(win, " (edited)")
            p2 = tmp_path / "b.txt"
            p2.write_text("bbb")
            with patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.StandardButton.Cancel,
            ):
                display._open_text_window(str(p2), reuse_window=win)
            assert "aaa (edited)" in win._text_edit.toPlainText()
        finally:
            win._text_edit.document().setModified(False)
            win.close()


class TestWslgPlatform(object):
    """Under WSLg, prefer the xcb (X11) Qt platform so window icons reach
    the Windows taskbar (Wayland cannot carry icons, so WSLg falls back to
    the Tux placeholder)."""

    @staticmethod
    def _patch_xcb_available(monkeypatch, value=True):
        monkeypatch.setattr(
            "helicon.lib.gui.macos._xcb_platform_available", lambda: value
        )

    def test_wslg_env_forces_xcb(self, monkeypatch):
        self._patch_xcb_available(monkeypatch)
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        display._force_x11_platform_under_wslg()
        assert os.environ["QT_QPA_PLATFORM"] == "xcb"
        # PyOpenGL must match Qt's GLX context (WSLg sets WAYLAND_DISPLAY, so
        # PyOpenGL would otherwise default to its EGL backend and napari's
        # canvas fails with "Attempt to retrieve context when no valid
        # context").
        assert os.environ["PYOPENGL_PLATFORM"] == "glx"

    def test_wsl_interop_env_forces_xcb(self, monkeypatch):
        self._patch_xcb_available(monkeypatch)
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
        monkeypatch.setenv("WSL_INTEROP", "/run/WSL/1_interop")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        display._force_x11_platform_under_wslg()
        assert os.environ["QT_QPA_PLATFORM"] == "xcb"
        assert os.environ["PYOPENGL_PLATFORM"] == "glx"

    def test_wslg_env_keeps_wayland_when_xcb_unavailable(self, monkeypatch):
        self._patch_xcb_available(monkeypatch, value=False)
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        display._force_x11_platform_under_wslg()
        assert "QT_QPA_PLATFORM" not in os.environ
        # PyOpenGL is left alone so it stays on EGL, matching native Wayland.
        assert "PYOPENGL_PLATFORM" not in os.environ

    def test_warns_when_xcb_unavailable_under_wslg(self, monkeypatch, capsys):
        # The silent Wayland fallback leaves the Tux taskbar icon with no
        # explanation; the warning must name the missing system library so
        # the user knows how to restore the icon.
        self._patch_xcb_available(monkeypatch, value=False)
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        display._force_x11_platform_under_wslg()
        assert "QT_QPA_PLATFORM" not in os.environ
        err = capsys.readouterr().err
        assert "libxcb-cursor0" in err
        assert "apt install" in err

    def test_explicit_platform_is_respected(self, monkeypatch):
        self._patch_xcb_available(monkeypatch)
        monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        display._force_x11_platform_under_wslg()
        assert os.environ["QT_QPA_PLATFORM"] == "offscreen"
        assert "PYOPENGL_PLATFORM" not in os.environ

    def test_explicit_xcb_platform_also_forces_glx(self, monkeypatch):
        self._patch_xcb_available(monkeypatch)
        monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
        monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        display._force_x11_platform_under_wslg()
        assert os.environ["QT_QPA_PLATFORM"] == "xcb"
        assert os.environ["PYOPENGL_PLATFORM"] == "glx"

    def test_user_pyopengl_platform_wins(self, monkeypatch):
        self._patch_xcb_available(monkeypatch)
        monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        display._force_x11_platform_under_wslg()
        assert os.environ["QT_QPA_PLATFORM"] == "xcb"
        # setdefault must never override an explicit user value.
        assert os.environ["PYOPENGL_PLATFORM"] == "egl"

    def test_noop_outside_wsl(self, monkeypatch):
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        monkeypatch.delenv("WSL_INTEROP", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("DISPLAY", ":0")
        display._force_x11_platform_under_wslg()
        assert "QT_QPA_PLATFORM" not in os.environ
        assert "PYOPENGL_PLATFORM" not in os.environ

    def test_noop_without_wayland(self, monkeypatch):
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        display._force_x11_platform_under_wslg()
        assert "QT_QPA_PLATFORM" not in os.environ
        assert "PYOPENGL_PLATFORM" not in os.environ

    def test_noop_without_x_server(self, monkeypatch):
        monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
        monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.delenv("DISPLAY", raising=False)
        display._force_x11_platform_under_wslg()
        assert "QT_QPA_PLATFORM" not in os.environ
        assert "PYOPENGL_PLATFORM" not in os.environ

    def test_xcb_plugin_probe_returns_bool(self):
        assert isinstance(display._xcb_platform_available(), bool)


class TestNapariDockIcon(object):
    """napari replaces the app icon; the monkey-patch keeps the Helicon icon."""

    @staticmethod
    def _fake_qt_main_window():
        """A stand-in for ``napari._qt.qt_main_window._QtMainWindow``."""
        return type("_QtMainWindow", (), {"__init__": lambda self, *a, **k: None})

    @staticmethod
    def _fake_module(qt_main_window):
        return MagicMock(QtMainWindow=qt_main_window, _QtMainWindow=qt_main_window)

    def test_patch_targets_private_main_window_class(self, qapp):
        """napari >= 0.5 calls the class _QtMainWindow; it must be patched."""
        qt_main_window = self._fake_qt_main_window()
        with patch.dict(
            sys.modules,
            {"napari._qt.qt_main_window": MagicMock(_QtMainWindow=qt_main_window)},
        ):
            display._patch_napari_icon()
        assert getattr(qt_main_window, "_helicon_icon_patched", False)
        instance = object.__new__(qt_main_window)
        qt_main_window.__init__(instance, viewer=None, window=None)
        assert str(instance._window_icon).endswith("resources/icon.svg")

    def test_load_napari_patches_icon(self, qapp):
        fake_napari = MagicMock()
        with (
            patch.dict(sys.modules, {"napari": fake_napari}),
            patch("helicon.lib.gui.viewer._patch_napari_value_bug") as mock_value,
            patch("helicon.lib.gui.viewer._patch_napari_icon") as mock_icon,
        ):
            result = display._load_napari()
        assert result is fake_napari
        mock_value.assert_called_once_with()
        mock_icon.assert_called_once_with()

    def test_patch_installs_helicon_window_icon(self, qapp):
        qt_main_window = self._fake_qt_main_window()
        with patch.dict(
            sys.modules,
            {"napari._qt.qt_main_window": self._fake_module(qt_main_window)},
        ):
            display._patch_napari_icon()
        assert getattr(qt_main_window, "_helicon_icon_patched", False)
        instance = object.__new__(qt_main_window)
        qt_main_window.__init__(instance, viewer=None, window=None)
        assert str(instance._window_icon).endswith("resources/icon.svg")

    def test_patch_is_idempotent(self, qapp):
        qt_main_window = self._fake_qt_main_window()
        with patch.dict(
            sys.modules,
            {"napari._qt.qt_main_window": self._fake_module(qt_main_window)},
        ):
            display._patch_napari_icon()
            wrapped = qt_main_window.__init__
            display._patch_napari_icon()
        assert qt_main_window.__init__ is wrapped

    def test_patch_noop_when_qt_main_window_unavailable(self, qapp):
        class _MissingQtMainWindow:
            @property
            def QtMainWindow(self):
                raise ImportError("napari qt_main_window unavailable")

        with patch.dict(
            sys.modules,
            {"napari._qt.qt_main_window": _MissingQtMainWindow()},
        ):
            display._patch_napari_icon()

    def test_patch_noop_when_icon_svg_missing(self, qapp):
        class _FakePath:
            def __init__(self, *args, **kwargs):
                pass

            @property
            def parent(self):
                return self

            @property
            def parents(self):
                return [self, self, self, self]

            def __truediv__(self, other):
                return self

            def is_file(self):
                return False

        qt_main_window = self._fake_qt_main_window()
        with (
            patch.dict(
                sys.modules,
                {"napari._qt.qt_main_window": self._fake_module(qt_main_window)},
            ),
            patch("helicon.lib.gui.viewer.Path", _FakePath),
        ):
            display._patch_napari_icon()
        assert not getattr(qt_main_window, "_helicon_icon_patched", False)


class TestGeometryPersistence(object):
    @patch("helicon.lib.gui.viewer._position_default")
    @patch("helicon.lib.gui.viewer._get_qsettings")
    def test_restore_geometry_restores_dock_from_ba(
        self,
        mock_get_settings,
        mock_position_default,
    ):
        """Dock geometry is restored from saved QByteArray."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_settings.value.side_effect = lambda key: (
            b"fake-dock-geo" if key == "dock_ba" else None
        )

        mock_dock = MagicMock()
        mock_viewer = MagicMock()

        with patch("PySide6.QtCore.QTimer") as mock_timer:
            display._restore_geometry(mock_dock, mock_viewer)
            captured_fn = mock_timer.singleShot.call_args[0][1]
            captured_fn()

        mock_dock.restoreGeometry.assert_called_once_with(b"fake-dock-geo")
        mock_position_default.assert_not_called()
        mock_dock.show.assert_called_once()

    @patch("helicon.lib.gui.viewer._position_default")
    @patch("helicon.lib.gui.viewer._get_qsettings")
    def test_restore_geometry_calls_position_default_when_no_saved_dock(
        self, mock_get_settings, mock_position_default
    ):
        """When no dock_ba is saved and viewer exists, fall back to _position_default."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_settings.value.return_value = None

        mock_dock = MagicMock()
        mock_viewer = MagicMock()

        with patch("PySide6.QtCore.QTimer") as mock_timer:
            display._restore_geometry(mock_dock, mock_viewer)
            captured_fn = mock_timer.singleShot.call_args[0][1]
            captured_fn()

        mock_position_default.assert_called_once_with(mock_dock, mock_viewer)
        mock_dock.show.assert_called_once()

    @patch("helicon.lib.gui.viewer._position_default")
    @patch("helicon.lib.gui.viewer._get_qsettings")
    def test_restore_geometry_applies_saved_viewer(self, mock_get_settings, mock_pd):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_settings.value.side_effect = lambda key: (
            b"fake-geometry-data" if key == "viewer_ba" else None
        )

        mock_dock = MagicMock()
        mock_viewer = MagicMock()

        with patch("PySide6.QtCore.QTimer") as mock_timer:
            display._restore_geometry(mock_dock, mock_viewer)
            captured_fn = mock_timer.singleShot.call_args[0][1]
            captured_fn()

        mock_viewer.window._qt_window.restoreGeometry.assert_called_once_with(
            b"fake-geometry-data"
        )

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

    @patch("PySide6.QtCore.QObject", side_effect=lambda *a, **kw: MagicMock())
    @patch("PySide6.QtCore.QEvent")
    @patch.object(display, "_save_geometry")
    def test_install_viewer_save_hook_returns_filter(
        self, mock_save, mock_qevent, mock_qobject
    ):
        mock_dock = MagicMock()
        mock_viewer = MagicMock()

        result = display._install_viewer_save_hook(mock_dock, mock_viewer)

        mock_viewer.window._qt_window.installEventFilter.assert_called_once_with(result)
        assert hasattr(result, "eventFilter")

    @patch("PySide6.QtCore.QObject", side_effect=lambda *a, **kw: MagicMock())
    @patch("PySide6.QtCore.QEvent")
    @patch.object(display, "_get_qsettings")
    def test_install_dock_save_hook_saves_geometry(
        self, mock_get_settings, mock_qevent, mock_qobject
    ):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_dock = MagicMock()
        mock_dock.saveGeometry.return_value = b"dock-geo"

        result = display._install_dock_save_hook(mock_dock)

        mock_dock.installEventFilter.assert_called_once()
        assert hasattr(result, "eventFilter")

    @patch("helicon.lib.gui.viewer._get_qsettings")
    def test_save_geometry_saves_dock_and_viewer(self, mock_get_settings):
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        mock_dock = MagicMock()
        mock_dock.saveGeometry.return_value = b"dock-ba"

        def _make_viewer(display_ba=None):
            mock_viewer = MagicMock()
            mock_viewer._display_only_ba = display_ba
            win = mock_viewer.window._qt_window
            win.saveGeometry.return_value = b"saved-geo"
            return mock_viewer

        mock_settings.reset_mock()
        display._save_geometry(mock_dock, _make_viewer())
        mock_settings.setValue.assert_any_call("viewer_ba", b"saved-geo")
        mock_settings.setValue.assert_any_call("dock_ba", b"dock-ba")

        mock_settings.reset_mock()
        viewer_with_cache = _make_viewer(display_ba=b"display-only-ba")
        display._save_geometry(mock_dock, viewer_with_cache)
        mock_settings.setValue.assert_any_call("viewer_ba", b"display-only-ba")
        mock_settings.setValue.assert_any_call("dock_ba", b"dock-ba")


class TestOpenFile(object):
    def test_open_mrc_file_adds_image(self, tmp_path):
        import numpy as np
        import mrcfile as _real_mrcfile

        test_data = np.zeros((3, 4, 5), dtype=np.float32)
        p = tmp_path / "test.mrc"
        with _real_mrcfile.new(str(p), overwrite=True) as mrc:
            mrc.set_data(test_data)
            mrc.voxel_size = (2.0, 2.0, 2.0)

        mock_viewer = MagicMock()
        display._open_file(mock_viewer, str(p))

        mock_viewer.add_image.assert_called_once()
        call_args, call_kwargs = mock_viewer.add_image.call_args
        assert hasattr(call_args[0], "compute")
        np.testing.assert_array_equal(call_args[0].compute(), test_data)
        assert call_args[0].shape == (3, 4, 5)
        assert call_kwargs["name"] == "test.mrc"
        assert call_kwargs["scale"] == (2.0, 2.0, 2.0)
        assert "contrast_limits" in call_kwargs
        assert len(call_kwargs["contrast_limits"]) == 2

    def test_ndisplay_resets_for_new_file(self, tmp_path):
        import numpy as np
        import mrcfile as _real_mrcfile

        # A fake viewer that records dims.ndisplay and supports layer removal,
        # so we can assert the 2D/3D mode is reset per file.
        class FakeDims:
            def __init__(self):
                self.ndisplay = 2
                self.current_step = (0, 0, 0)

        class FakeLayer:
            def __init__(self, name="layer"):
                self.name = name

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

        # Create real temp MRC files.
        vol_path = tmp_path / "map.mrc"
        with _real_mrcfile.new(str(vol_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((4, 8, 8), dtype=np.float32))
            mrc.voxel_size = (1.5, 1.5, 1.5)

        stack_path = tmp_path / "stack.mrcs"
        with _real_mrcfile.new(str(stack_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((3, 8, 8), dtype=np.float32))
            mrc.voxel_size = (2.0, 2.0, 2.0)

        with patch.object(
            display.helicon, "change_map_axes_order", lambda d, h: (d, None)
        ):
            # First open a 3D volume (ndisplay forced to 3)...
            vol_viewer = FakeViewer()
            display._open_file(vol_viewer, str(vol_path), mode="volume")
            assert vol_viewer.dims.ndisplay == 3

            # ...then a 2D image stack. The stale 3D mode must not
            # persist: ndisplay is reset to 2 and only the new file shows
            # (previous layers are removed so reset_view fits just the new file).
            img_viewer = FakeViewer()
            img_viewer.dims.ndisplay = 3  # simulate leftover state
            old_layer = FakeLayer("old.mrc")
            img_viewer.layers.append(old_layer)
            display._open_file(img_viewer, str(stack_path), mode="slice")
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
                return_value=(fake_entries, fake_shape, fake_apix, 0),
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

    def test_open_eps_falls_back_to_quicklook_on_macos(
        self, qapp, tmp_path, monkeypatch
    ):
        import shutil

        from PySide6.QtGui import QImage

        monkeypatch.setattr(sys, "platform", "darwin")
        src_png = tmp_path / "rendered.png"
        img = QImage(16, 16, QImage.Format.Format_ARGB32)
        img.fill(0xFFFFFFFF)
        img.save(str(src_png))

        def fake_which(name):
            if name in ("gs", "ghostscript"):
                return None
            if name == "qlmanage":
                return "/usr/bin/qlmanage"
            return None

        def fake_run(args, **kwargs):
            # qlmanage writes <filename>.png into the -o directory.
            out_dir = args[args.index("-o") + 1]
            shutil.copy(str(src_png), str(Path(out_dir) / "fig.eps.png"))
            return type("R", (), {"returncode": 0})()

        with (
            patch("shutil.which", side_effect=fake_which),
            patch("subprocess.run", fake_run),
            patch("helicon.lib.gui.file_openers.print") as mock_print,
        ):
            mock_viewer = MagicMock()
            display._open_eps(mock_viewer, "/d/fig.eps")

        mock_viewer.add_image.assert_called_once()
        mock_print.assert_not_called()
        args, kwargs = mock_viewer.add_image.call_args
        assert kwargs["name"] == "fig.eps"

    def test_open_eps_without_ghostscript_prints_message(self):
        with (
            patch("shutil.which", return_value=None),
            patch("helicon.lib.gui.file_openers.print") as mock_print,
        ):
            mock_viewer = MagicMock()
            display._open_eps(mock_viewer, "/d/fig.eps")
        mock_viewer.add_image.assert_not_called()
        assert any("Ghostscript" in str(c.args) for c in mock_print.call_args_list)

    def test_extractpick_star_opens_as_text(self):
        mock_viewer = MagicMock()

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = "_rlnCoordinateX 1\n"

        with patch("builtins.open", MagicMock(return_value=mock_file)):
            with patch.object(display, "_is_text_file", return_value=True):
                display._open_file(mock_viewer, "/path/to/extractpick.star")

        mock_viewer.open.assert_not_called()
        mock_viewer.add_image.assert_not_called()

    def test_metadata_mode_opens_any_star_as_text(self):
        mock_viewer = MagicMock()

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read.return_value = "_rlnCoordinateX 1\n"

        with patch("builtins.open", MagicMock(return_value=mock_file)):
            with patch.object(display, "_is_text_file", return_value=True):
                display._open_file(mock_viewer, "/path/to/particles.star", mode="text")

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
    def test_text_display_window_uses_saved_theme(self, tmp_path, qapp):
        from PySide6.QtCore import QSettings

        text_path = tmp_path / "example.txt"
        text_path.write_text("hello")
        settings = QSettings("helicon", "display")
        settings.setValue("theme", "Light")
        try:
            assert "background-color: #f4f4f4" in display._display_theme_stylesheet()
            palette = display._display_theme_palette()
            assert palette.window().color().name() == "#f4f4f4"
        finally:
            settings.remove("theme")

    def test_launched_qt_display_styles_follow_saved_theme(self, qapp):
        from PySide6.QtCore import QSettings

        settings = QSettings("helicon", "display")
        settings.setValue("theme", "Light")
        try:
            stylesheet = display._display_theme_stylesheet()
            assert "background-color: #f4f4f4" in stylesheet
            assert "color: #202020" in stylesheet
        finally:
            settings.remove("theme")

    def test_browser_theme_is_persisted(self, tmp_path, qapp):
        from PySide6.QtCore import QSettings
        from helicon.lib.gui.file_browser import FolderBrowserWidget, _saved_theme

        settings = QSettings("helicon", "display")
        settings.remove("theme")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        assert widget._theme_combo.currentText() == "System"

        widget._theme_combo.setCurrentText("Light")
        assert _saved_theme() == "Light"

        widget.close()
        restored = FolderBrowserWidget(start_dir=str(tmp_path))
        assert restored._theme_combo.currentText() == "Light"
        settings.remove("theme")

    def test_browser_has_file_and_view_menus(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        assert [action.text() for action in widget._menu_bar.actions()] == [
            "File",
            "Apps",
            "View",
            "Help",
        ]
        assert widget._file_menu.title() == "File"
        assert widget._view_menu.title() == "View"
        assert widget._apps_menu.title() == "Apps"
        assert widget._help_menu.title() == "Help"
        assert widget._open_folder_action.text() == "Open Folder…"
        assert widget._quit_action.text() == "Quit"
        assert widget._docs_action.text() == "Documentation"
        assert widget._home_page_action.text() == "Home Page"
        assert widget._theme_menu.title() == "Theme"
        assert set(widget._theme_actions) == {"Dark", "Light", "System"}
        file_labels = [a.text() for a in widget._file_menu.actions()]
        assert "Quit" in file_labels
        help_labels = [a.text() for a in widget._help_menu.actions()]
        assert help_labels[0] == "Home Page"
        assert help_labels[1] == "Documentation"
        # Terminal leads the Apps menu, followed by each standalone tool.
        assert widget._open_terminal_action in widget._apps_menu.actions()
        assert widget._apps_menu.actions()[0] is widget._open_terminal_action
        app_labels = [a.text() for a in widget._apps_menu.actions()]
        assert app_labels[0] == "Terminal"
        assert "WhereIsMyClass" in app_labels
        assert "CTF Simulation" in app_labels
        assert "ProCart" in app_labels
        assert "Terminal" not in file_labels

    def test_quit_action_quits_application(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(qapp, "quit") as mock_quit:
            widget._quit_action.trigger()
        mock_quit.assert_called_once()

    def test_documentation_action_opens_readthedocs(self, tmp_path, qapp):
        from helicon.lib.gui import file_browser
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(file_browser, "_open_url") as mock_open:
            widget._docs_action.trigger()
        mock_open.assert_called_once_with("https://helicon.readthedocs.io")

    def test_home_page_action_opens_homepage(self, tmp_path, qapp):
        from helicon.lib.gui import file_browser
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(file_browser, "_open_url") as mock_open:
            widget._home_page_action.trigger()
        mock_open.assert_called_once_with("https://jianglab.science.psu.edu/helicon")

    def test_theme_menu_action_persists_selection(self, tmp_path, qapp):
        from PySide6.QtCore import QSettings
        from helicon.lib.gui.file_browser import FolderBrowserWidget, _saved_theme

        settings = QSettings("helicon", "display")
        settings.remove("theme")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        try:
            widget._theme_actions["Light"].trigger()
            assert _saved_theme() == "Light"
            assert widget._theme_actions["Light"].isChecked()
            assert not widget._theme_actions["Dark"].isChecked()
        finally:
            settings.remove("theme")

    def test_recent_folders_are_available_from_file_menu(self, tmp_path, qapp):
        from PySide6.QtCore import QSettings
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        recent = tmp_path / "recent"
        recent.mkdir()
        settings = QSettings("helicon", "display")
        settings.setValue("recent_folders", [str(recent)])
        try:
            widget = FolderBrowserWidget(start_dir=str(tmp_path))
            actions = widget._recent_menu.actions()
            assert [action.text() for action in actions] == [str(recent)]
        finally:
            settings.remove("recent_folders")

    def test_open_folder_action_navigates_to_selected_directory(self, tmp_path, qapp):
        from PySide6.QtWidgets import QFileDialog
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        selected = tmp_path / "selected"
        selected.mkdir()
        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        with patch.object(
            QFileDialog,
            "getExistingDirectory",
            return_value=str(selected),
        ):
            widget._open_folder_action.trigger()

        assert widget._model._root_path == str(selected)

    def test_open_terminal_launches_in_current_folder(self, tmp_path, qapp):
        from helicon.lib import terminal
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        # The action must exist in the Apps menu (as its first item) and
        # launch the host terminal rooted at the currently shown folder.
        assert widget._open_terminal_action.text() == "Terminal"
        assert widget._open_terminal_action in widget._apps_menu.actions()
        with patch.object(terminal, "_spawn_detached", return_value=True) as mock_spawn:
            widget._open_terminal_action.trigger()
        mock_spawn.assert_called_once()
        cmd = mock_spawn.call_args[0][0]
        # The target folder appears as a CLI arg (``--working-directory=...``)
        # or embedded in the macOS `do script` line / fallback shell command,
        # so check the args and the injected command text.
        payload = list(cmd)
        assert any(str(tmp_path) in arg for arg in payload)

    def test_apps_menu_web_app_launch_reuses_action_button_pipeline(
        self, tmp_path, qapp
    ):
        from unittest.mock import patch

        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        action = widget._app_actions["WhereIsMyClass"]
        with patch.object(widget, "_launch_web_app_tab") as mock_launch:
            action.trigger()
        mock_launch.assert_called_once_with("WhereIsMyClass")

        # The web-app tab launch goes through display's shared pipeline with
        # an empty bookmark (no input file pre-set).
        with (
            patch("helicon.commands.display._launch_or_reuse_web_app") as mock_reuse,
            patch(
                "helicon.commands.display._make_bookmark_query",
                return_value={"p": "{}"},
            ) as mock_query,
        ):
            widget._launch_web_app_tab("HILL")
        mock_query.assert_called_once_with("HILL", {})
        mock_reuse.assert_called_once_with("HILL", {"p": "{}"}, new_window=False)

    def test_apps_menu_streamlit_launch_spawns_detached(self, tmp_path, qapp):
        from unittest.mock import patch

        from helicon.lib.gui import file_browser
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        action = widget._app_actions["CTF Simulation"]
        with patch.object(
            file_browser, "_spawn_detached", return_value=True
        ) as mock_spawn:
            with patch.object(file_browser, "sys") as mock_sys:
                mock_sys.executable = "/path/to/python"
                action.trigger()
        mock_spawn.assert_called_once()
        cmd = mock_spawn.call_args[0][0]
        assert cmd[0] == "/path/to/python"
        assert "helicon.commands.ctfSimulation" in cmd

    def test_apps_menu_images2star_opens_with_empty_selector(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        action = widget._app_actions["Images2Star"]
        assert action.data() == (None, None, "star")

        # The menu action routes to the in-panel selector picker.
        with patch.object(widget, "_on_pick_images2star_file") as mock_pick:
            action.trigger()
        mock_pick.assert_called_once_with()

        # The picker opens the tools panel with no pre-set input file,
        # reusing the tracker's active window unless "New" is checked.
        with (
            patch("helicon.commands.display._open_images2star_tools") as mock_open,
            patch(
                "helicon.commands.display._images2star.active",
                return_value="active-win",
            ),
        ):
            widget._on_pick_images2star_file()
        mock_open.assert_called_once_with(
            None,
            parent=widget,
            reuse_window="active-win",
            tracker=display._images2star,
        )

        widget._new_window_cb.setChecked(True)
        with patch("helicon.commands.display._open_images2star_tools") as mock_open:
            widget._on_pick_images2star_file()
        mock_open.assert_called_once_with(
            None, parent=widget, reuse_window=None, tracker=display._images2star
        )

    def test_open_terminal_darwin_uses_native_terminal(self):
        from unittest.mock import patch

        from helicon.lib import terminal

        with patch.object(terminal, "platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            with patch.object(
                terminal, "_spawn_detached", return_value=True
            ) as mock_spawn:
                terminal._open_terminal("/some/folder")
        # macOS drives Terminal.app via AppleScript `do script`, but only a
        # single `source` line (the env snapshot) is sent -- LaunchServices
        # drops a subprocess env, and multi-line exports would echo noisily.
        mock_spawn.assert_called_once()
        cmd = mock_spawn.call_args[0][0]
        assert cmd[0] == "osascript"
        assert cmd[1] == "-e"
        applescript = cmd[2]
        assert applescript.startswith('tell application "Terminal" to do script ')
        inner = applescript.split(" do script ", 1)[1]
        assert "cd /some/folder" in inner
        assert "source" in inner
        assert terminal._env_file().name in inner
        # Only env snapshot handling; no subprocess env is passed for the GUI.
        assert mock_spawn.call_args[1] == {}

    def test_open_terminal_wsl_prefers_windows_terminal(self):
        from unittest.mock import patch

        from helicon.lib import terminal

        with (
            patch.object(terminal, "platform") as mock_platform,
            patch.object(terminal, "_is_wsl", return_value=True),
            patch.object(terminal, "shutil") as mock_shutil,
            patch.object(terminal, "_spawn_detached", return_value=True) as mock_spawn,
        ):
            mock_platform.system.return_value = "Linux"
            mock_shutil.which.side_effect = lambda c: c if c == "wt.exe" else None
            terminal._open_terminal("/some/folder")
        mock_spawn.assert_called_once_with(
            ["wt.exe", "wsl", "--cd", "/some/folder"],
            env=terminal._terminal_env(),
        )

    def test_open_terminal_wsl_falls_back_to_wsl_exe(self):
        from unittest.mock import patch

        from helicon.lib import terminal

        with (
            patch.object(terminal, "platform") as mock_platform,
            patch.object(terminal, "_is_wsl", return_value=True),
            patch.object(terminal, "shutil") as mock_shutil,
            patch.object(terminal, "_spawn_detached", return_value=True) as mock_spawn,
        ):
            mock_platform.system.return_value = "Linux"
            mock_shutil.which.side_effect = lambda c: c if c == "wsl.exe" else None
            terminal._open_terminal("/some/folder")
        mock_spawn.assert_called_once_with(
            ["wsl.exe", "--cd", "/some/folder"],
            env=terminal._terminal_env(),
        )

    def test_open_terminal_wsl_without_interop_uses_x_terminals(self):
        from unittest.mock import patch

        from helicon.lib import terminal

        with (
            patch.object(terminal, "platform") as mock_platform,
            patch.object(terminal, "_is_wsl", return_value=True),
            patch.object(terminal, "shutil") as mock_shutil,
            patch.object(terminal, "_gnome_terminal_usable", return_value=True),
            patch.object(terminal, "_spawn_detached", return_value=True) as mock_spawn,
        ):
            mock_platform.system.return_value = "Linux"
            mock_shutil.which.side_effect = lambda c: (
                c if c == "gnome-terminal" else None
            )
            terminal._open_terminal("/some/folder")
        mock_spawn.assert_called_once_with(
            ["gnome-terminal", "--working-directory=/some/folder"],
            check_early_exit=True,
            env=terminal._terminal_env(),
        )

    def test_open_terminal_linux_uses_xterm_e_flag(self):
        from unittest.mock import patch

        from helicon.lib import terminal

        with (
            patch.object(terminal, "platform") as mock_platform,
            patch.object(terminal, "_is_wsl", return_value=False),
            patch.object(terminal, "shutil") as mock_shutil,
            patch.object(terminal, "_gnome_terminal_usable", return_value=False),
            patch.object(terminal, "_spawn_detached", return_value=True) as mock_spawn,
        ):
            mock_platform.system.return_value = "Linux"
            mock_shutil.which.side_effect = lambda c: c if c == "xterm" else None
            terminal._open_terminal("/some/folder")
        cmd = mock_spawn.call_args[0][0]
        assert cmd[0] == "xterm"
        assert cmd[1] == "-e"
        shell = os.environ.get("SHELL") or "/bin/sh"
        assert cmd[-1] == (
            f"cd /some/folder && {terminal._source_env_command()} ; exec {shell}"
        )
        assert mock_spawn.call_args[1].get("check_early_exit") is True
        assert mock_spawn.call_args[1]["env"] == terminal._terminal_env()

    def test_open_terminal_linux_skips_gnome_without_session(self):
        """SSH/X11 hosts often have gnome-terminal installed but no D-Bus service."""
        from unittest.mock import patch

        from helicon.lib import terminal

        with (
            patch.object(terminal, "platform") as mock_platform,
            patch.object(terminal, "_is_wsl", return_value=False),
            patch.object(terminal, "shutil") as mock_shutil,
            patch.object(terminal, "_gnome_terminal_usable", return_value=False),
            patch.object(terminal, "_spawn_detached", return_value=True) as mock_spawn,
        ):
            mock_platform.system.return_value = "Linux"
            available = {"gnome-terminal", "lxterminal", "xterm"}
            mock_shutil.which.side_effect = lambda c: c if c in available else None
            terminal._open_terminal("/some/folder")
        mock_spawn.assert_called_once_with(
            ["lxterminal", "--working-directory=/some/folder"],
            check_early_exit=True,
            env=terminal._terminal_env(),
        )

    def test_open_terminal_linux_falls_through_on_early_exit(self):
        from unittest.mock import patch

        from helicon.lib import terminal

        def spawn_side_effect(cmd, check_early_exit=False, env=None):
            return cmd[0] != "lxterminal"

        with (
            patch.object(terminal, "platform") as mock_platform,
            patch.object(terminal, "_is_wsl", return_value=False),
            patch.object(terminal, "shutil") as mock_shutil,
            patch.object(terminal, "_gnome_terminal_usable", return_value=False),
            patch.object(
                terminal, "_spawn_detached", side_effect=spawn_side_effect
            ) as mock_spawn,
        ):
            mock_platform.system.return_value = "Linux"
            available = {"lxterminal", "xterm"}
            mock_shutil.which.side_effect = lambda c: c if c in available else None
            terminal._open_terminal("/some/folder")
        assert mock_spawn.call_count == 2
        assert mock_spawn.call_args_list[0][0][0][0] == "lxterminal"
        assert mock_spawn.call_args_list[1][0][0][0] == "xterm"

    def test_open_terminal_linux_honours_terminal_env(self):
        from unittest.mock import patch

        from helicon.lib import terminal

        expected_env = None
        with (
            patch.object(terminal, "platform") as mock_platform,
            patch.object(terminal, "_is_wsl", return_value=False),
            patch.object(terminal, "shutil") as mock_shutil,
            patch.object(terminal, "_gnome_terminal_usable", return_value=False),
            patch.object(terminal, "_spawn_detached", return_value=True) as mock_spawn,
            patch.dict(os.environ, {"TERMINAL": "kitty"}, clear=False),
        ):
            mock_platform.system.return_value = "Linux"
            mock_shutil.which.side_effect = lambda c: (
                c if c in {"kitty", "xterm"} else None
            )
            terminal._open_terminal("/some/folder")
            expected_env = terminal._terminal_env()
        mock_spawn.assert_called_once_with(
            ["kitty", "--working-directory=/some/folder"],
            check_early_exit=True,
            env=expected_env,
        )

    def test_terminal_env_prepends_interpreter_bin_to_path(self):
        from helicon.lib import terminal

        with (
            patch.object(terminal, "sys") as mock_sys,
            patch.dict(
                os.environ,
                {"PATH": "/usr/bin:/bin", "PYTHONPATH": "/some/py"},
                clear=True,
            ),
        ):
            mock_sys.prefix = "/envs/helicon"
            mock_sys.base_prefix = "/base"
            env = terminal._terminal_env()

        bin_dir = "/envs/helicon/bin"
        assert env["PATH"] == f"{bin_dir}:/usr/bin:/bin"
        assert env["VIRTUAL_ENV"] == "/envs/helicon"
        assert env["CONDA_PREFIX"] == "/envs/helicon"
        assert env["CONDA_DEFAULT_ENV"] == "helicon"

    def test_terminal_env_within_base_prefix_does_not_sync_helpers(self):
        from helicon.lib import terminal

        with (
            patch.object(terminal, "sys") as mock_sys,
            patch.dict(
                os.environ,
                {"PATH": "/usr/bin"},
                clear=True,
            ),
        ):
            mock_sys.prefix = "/usr"
            mock_sys.base_prefix = "/usr"
            env = terminal._terminal_env()

        assert env["PATH"] == "/usr/bin"
        assert "VIRTUAL_ENV" not in env
        assert "CONDA_PREFIX" not in env
        assert "CONDA_DEFAULT_ENV" not in env

    def test_gnome_terminal_usable_requires_session_or_owner(self):
        from unittest.mock import patch

        from helicon.lib import terminal

        with (
            patch.object(
                terminal.shutil, "which", return_value="/usr/bin/gnome-terminal"
            ),
            patch.object(terminal, "_dbus_name_has_owner", return_value=False),
            patch.dict(
                os.environ,
                {"XDG_CURRENT_DESKTOP": "", "DESKTOP_SESSION": ""},
                clear=False,
            ),
        ):
            assert terminal._gnome_terminal_usable() is False

        with (
            patch.object(
                terminal.shutil, "which", return_value="/usr/bin/gnome-terminal"
            ),
            patch.object(terminal, "_dbus_name_has_owner", return_value=False),
            patch.dict(
                os.environ,
                {"XDG_CURRENT_DESKTOP": "GNOME", "DESKTOP_SESSION": "gnome"},
                clear=False,
            ),
        ):
            assert terminal._gnome_terminal_usable() is True

        with (
            patch.object(
                terminal.shutil, "which", return_value="/usr/bin/gnome-terminal"
            ),
            patch.object(terminal, "_dbus_name_has_owner", return_value=True),
            patch.dict(
                os.environ,
                {"XDG_CURRENT_DESKTOP": "", "DESKTOP_SESSION": ""},
                clear=False,
            ),
        ):
            assert terminal._gnome_terminal_usable() is True

    def test_invalid_browser_theme_falls_back_to_system(self, qapp):
        from PySide6.QtCore import QSettings
        from helicon.lib.gui.file_browser import _saved_theme

        settings = QSettings("helicon", "display")
        settings.setValue("theme", "not-a-theme")
        assert _saved_theme() == "System"
        settings.remove("theme")

    def test_format_size_bytes(self):
        from helicon.lib.gui.file_browser import _format_size

        assert _format_size(512) == "512 B"

    def test_format_size_kilobytes(self):
        from helicon.lib.gui.file_browser import _format_size

        assert _format_size(2048) == "2.0 KB"

    def test_format_size_megabytes(self):
        from helicon.lib.gui.file_browser import _format_size

        assert _format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_format_size_gigabytes(self):
        from helicon.lib.gui.file_browser import _format_size

        assert _format_size(2 * 1024 * 1024 * 1024) == "2.00 GB"

    def test_file_browser_model_has_six_columns(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel, NUM_COLUMNS

        (tmp_path / "test.txt").write_text("hello")
        model = FileBrowserModel(str(tmp_path))
        assert model.columnCount() == NUM_COLUMNS

    def test_file_browser_model_headers(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel

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
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_NAME

        (tmp_path / "aaa.txt").write_text("a")
        (tmp_path / "bbb.txt").write_text("b")
        model = FileBrowserModel(str(tmp_path))
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "aaa.txt" in names
        assert "bbb.txt" in names

    def test_file_browser_model_lists_dirs_first(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_NAME, COL_TYPE

        (tmp_path / "adir").mkdir()
        (tmp_path / "file.txt").write_text("x")
        model = FileBrowserModel(str(tmp_path))
        first_type = model.item(0, COL_TYPE).text()
        last_type = model.item(model.rowCount() - 1, COL_TYPE).text()
        assert first_type == "Folder"
        assert last_type != "Folder"

    def test_file_browser_model_shows_size(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_SIZE

        (tmp_path / "data.bin").write_bytes(b"x" * 2048)
        model = FileBrowserModel(str(tmp_path))
        sizes = [model.item(r, COL_SIZE).text() for r in range(model.rowCount())]
        assert "2.0 KB" in sizes

    def test_file_browser_model_shows_date(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_MODIFIED

        (tmp_path / "recent.txt").write_text("new")
        model = FileBrowserModel(str(tmp_path))
        dates = [model.item(r, COL_MODIFIED).text() for r in range(model.rowCount())]
        assert any("202" in d for d in dates)

    def test_file_browser_model_sort_by_size(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_SIZE

        (tmp_path / "small.txt").write_text("s")
        (tmp_path / "big.txt").write_text("x" * 10000)
        (tmp_path / "adir").mkdir()
        model = FileBrowserModel(str(tmp_path))
        model.sort(COL_SIZE, Qt.SortOrder.DescendingOrder)
        sizes = [model.item(r, COL_SIZE).text() for r in range(model.rowCount())]
        assert sizes[0] == "9.8 KB"
        assert sizes[-1] == ""

    def test_file_browser_model_sort_dirs_first(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_TYPE

        (tmp_path / "adir").mkdir()
        (tmp_path / "zfile.txt").write_text("z")
        model = FileBrowserModel(str(tmp_path))
        model.sort(0, Qt.SortOrder.AscendingOrder)
        assert model.item(0, COL_TYPE).text() == "Folder"

    def test_file_browser_model_set_root_path(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_NAME

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "inner.txt").write_text("i")
        model = FileBrowserModel(str(tmp_path))
        model.set_root_path(str(sub))
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "inner.txt" in names

    def test_file_browser_model_file_path(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel

        (tmp_path / "hello.txt").write_text("hi")
        model = FileBrowserModel(str(tmp_path))
        path = model.file_path(model.index(0, 0))
        assert path is not None
        assert "hello.txt" in path

    def test_file_browser_model_is_dir(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel

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

    def test_count_dir_items_matches_listing_rules(self, tmp_path):
        from helicon.lib.gui.file_browser import _count_dir_items

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.txt").write_text("a")
        (sub / "b.txt").write_text("b")
        (sub / "data.mrc").write_bytes(b"x" * 64)
        (sub / "nested").mkdir()
        (sub / ".hidden.txt").write_text("h")

        # No filter: hidden entries are skipped, everything else counts.
        assert _count_dir_items(sub, "*", False) == 4
        # With a filter: non-directory entries must match it.
        assert _count_dir_items(sub, "*.txt", False) == 3
        assert _count_dir_items(sub, "*.mrc", False) == 2
        assert _count_dir_items(sub, r".*\.mrc", True) == 2

    def test_count_dir_items_unreadable_returns_none(self, tmp_path):
        from helicon.lib.gui.file_browser import _count_dir_items

        # A regular file is not a directory; iterdir() raises NotADirectoryError.
        f = tmp_path / "plain.txt"
        f.write_text("x")
        assert _count_dir_items(f, "*", False) is None

    def test_file_browser_model_dir_rows(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel

        (tmp_path / "adir").mkdir()
        (tmp_path / "bfile.txt").write_text("x")
        model = FileBrowserModel(str(tmp_path))
        dirs = model.dir_rows()
        assert len(dirs) == 1
        row, path = dirs[0]
        assert Path(path).name == "adir"
        assert model.is_dir(model.index(row, 0))

    def test_apply_dir_count_sets_size_cell(self, tmp_path):
        from helicon.lib.gui.file_browser import (
            FileBrowserModel,
            COL_NAME,
            COL_SIZE,
            ROLE_SORT,
        )

        sub = tmp_path / "sub"
        sub.mkdir()
        model = FileBrowserModel(str(tmp_path))
        row = model._row_for_filepath(str(sub))
        assert row >= 0

        model.apply_dir_count(str(sub), 3)
        assert model.item(row, COL_SIZE).text() == "3 items"
        assert model.item(row, COL_SIZE).data(ROLE_SORT) == 3

        model.apply_dir_count(str(sub), 1)
        assert model.item(row, COL_SIZE).text() == "1 item"

    def test_apply_dir_count_ignores_unknown_path(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_SIZE

        (tmp_path / "sub").mkdir()
        model = FileBrowserModel(str(tmp_path))
        model.apply_dir_count(str(tmp_path / "does-not-exist"), 5)
        model.apply_dir_count(str(tmp_path / "sub"), -1)
        sizes = [model.item(r, COL_SIZE).text() for r in range(model.rowCount())]
        assert sizes == [""]

    def test_file_browser_model_sorts_dirs_by_item_count(self, tmp_path):
        from helicon.lib.gui.file_browser import (
            FileBrowserModel,
            COL_NAME,
            COL_SIZE,
        )

        few = tmp_path / "few"
        many = tmp_path / "many"
        few.mkdir()
        many.mkdir()
        (many / "1.txt").write_text("1")
        (many / "2.txt").write_text("2")
        model = FileBrowserModel(str(tmp_path))
        model.apply_dir_count(str(few), 1)
        model.apply_dir_count(str(many), 2)
        model.sort(COL_SIZE, Qt.SortOrder.AscendingOrder)
        assert model.item(0, COL_NAME).text() == "few/"
        assert model.item(1, COL_NAME).text() == "many/"

    def test_folder_browser_emits_signal(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

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
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        sub = tmp_path / "sub"
        sub.mkdir()
        widget = FolderBrowserWidget(start_dir=str(sub))
        widget._go_up()
        assert widget._model._root_path == str(tmp_path)

    def test_folder_browser_go_back(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        sub = tmp_path / "sub"
        sub.mkdir()
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        widget._navigate_to(str(sub))
        widget._go_back()
        assert widget._model._root_path == str(tmp_path)

    def test_folder_browser_shift_double_click_emits_new_window_signal(
        self, tmp_path, qapp
    ):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

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

    def test_is_image_stack_classification(self, tmp_path, qapp):
        import numpy as np
        import mrcfile as _real_mrcfile
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        assert widget._is_image_stack("/d/particles.mrcs")
        assert widget._is_image_stack("/d/data.star")
        # Metadata star files are not image stacks.
        assert not widget._is_image_stack("/d/run1_optimiser.star")
        assert not widget._is_image_stack("/d/20170629_00049_frameImage.star")
        # Genuine volumes (nz > 1) keep the "2D Slice" label, so are not
        # image stacks.
        vol_path = tmp_path / "map.mrc"
        with _real_mrcfile.new(str(vol_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((4, 8, 8), dtype=np.float32))
        assert not widget._is_image_stack(str(vol_path))
        # Single-slice (nz == 1) mrc files are 2D image stacks.
        single_path = tmp_path / "single.mrc"
        with _real_mrcfile.new(str(single_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((1, 8, 8), dtype=np.float32))
        assert widget._is_image_stack(str(single_path))

    def test_frameimage_star_opens_as_text(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import (
            FolderBrowserWidget,
            _get_file_type_label,
        )

        star_file = tmp_path / "20170629_00049_frameImage.star"
        star_file.write_text("dummy")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        assert widget._display_modes_for(str(star_file)) == ["text"]
        assert not widget._is_image_stack(str(star_file))
        assert _get_file_type_label(str(star_file)) == "Metadata"

        # AutoPick result files are metadata too, not image stacks.
        autopick_file = tmp_path / "20170629_00049_frameImage_autopick.star"
        autopick_file.write_text("dummy")
        assert widget._display_modes_for(str(autopick_file)) == ["text"]
        assert not widget._is_image_stack(str(autopick_file))
        assert _get_file_type_label(str(autopick_file)) == "Metadata"

    def test_slice_button_label_for_image_stack(self, tmp_path, qapp):
        import numpy as np
        import mrcfile as _real_mrcfile
        from helicon.lib.gui.file_browser import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        (tmp_path / "particles.mrcs").write_bytes(b"\x00" * 1024)
        vol_path = tmp_path / "volume.mrc"
        with _real_mrcfile.new(str(vol_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((4, 8, 8), dtype=np.float32))
            mrc.voxel_size = (1.5, 1.5, 1.5)
        single_path = tmp_path / "single.mrc"
        with _real_mrcfile.new(str(single_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((1, 8, 8), dtype=np.float32))
            mrc.voxel_size = (2.0, 2.0, 2.0)
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

        # A single-slice (nz == 1) mrc is a 2D image, not a volume.
        select("single.mrc")
        assert widget._btn_slice.text() == "2D Image"

    def test_pdf_button_label(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        pdf_file = tmp_path / "figure.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\nplaceholder")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        index = widget._model.index(0, 0)
        widget._tree.selectionModel().select(
            index, QItemSelectionModel.Select | QItemSelectionModel.Clear
        )

        assert widget._btn_slice.text() == "PDF"

    def test_chimerax_button_shown_for_volumes(self, tmp_path, qapp):
        import numpy as np
        import mrcfile as _real_mrcfile
        from helicon.lib.gui.file_browser import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        vol_path = tmp_path / "volume.mrc"
        with _real_mrcfile.new(str(vol_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((4, 8, 8), dtype=np.float32))
            mrc.voxel_size = (1.5, 1.5, 1.5)
        single_path = tmp_path / "single.mrc"
        with _real_mrcfile.new(str(single_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((1, 8, 8), dtype=np.float32))
            mrc.voxel_size = (2.0, 2.0, 2.0)
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

        # Single-slice (nz == 1) mrc files must not offer ChimeraX.
        select("single.mrc")
        assert widget._btn_chimerax.isHidden()

        # Image stacks must not offer the ChimeraX button.
        select("particles.mrcs")
        assert widget._btn_chimerax.isHidden()

    def test_chimerax_button_shown_for_bild(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        (tmp_path / "plot.bild").write_bytes(b"\x00" * 64)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        idx = widget._model.index(0, 0)
        widget._tree.selectionModel().select(
            idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
        )
        # bild files get a ChimeraX button (renders cylinders/spheres natively).
        assert not widget._btn_chimerax.isHidden()
        assert widget._btn_chimerax.text() == "ChimeraX"

    def test_launch_chimerax_invokes_executable(self, tmp_path):
        import subprocess

        from helicon.lib.gui import file_browser

        called = {}

        def fake_find():
            return "/fake/ChimeraX"

        def fake_popen(args):
            called["args"] = list(args)
            return object()

        with (
            patch.object(file_browser, "_find_chimerax", fake_find),
            patch.object(subprocess, "Popen", fake_popen),
        ):
            display._launch_chimerax("/d/map.mrc")

        assert called["args"] == ["/fake/ChimeraX", "/d/map.mrc"]

    def test_chimerax_button_disabled_when_not_installed(self, tmp_path, qapp):
        import numpy as np
        import mrcfile as _real_mrcfile
        from helicon.lib.gui import file_browser
        from helicon.lib.gui.file_browser import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        with patch.object(file_browser, "_find_chimerax", lambda: None):
            vol_path = tmp_path / "volume.mrc"
            with _real_mrcfile.new(str(vol_path), overwrite=True) as mrc:
                mrc.set_data(np.zeros((4, 8, 8), dtype=np.float32))
                mrc.voxel_size = (1.5, 1.5, 1.5)
            widget = FolderBrowserWidget(start_dir=str(tmp_path))
            idx = widget._model.index(0, 0)
            widget._tree.selectionModel().select(
                idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
            )
            assert not widget._btn_chimerax.isHidden()
            assert not widget._btn_chimerax.isEnabled()
            assert "not found" in widget._btn_chimerax.toolTip().lower()

    def test_chimerax_button_enabled_when_installed(self, tmp_path, qapp):
        import numpy as np
        import mrcfile as _real_mrcfile
        from helicon.lib.gui import file_browser
        from helicon.lib.gui.file_browser import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        with patch.object(file_browser, "_find_chimerax", lambda: "/x/ChimeraX"):
            vol_path = tmp_path / "volume.mrc"
            with _real_mrcfile.new(str(vol_path), overwrite=True) as mrc:
                mrc.set_data(np.zeros((4, 8, 8), dtype=np.float32))
                mrc.voxel_size = (1.5, 1.5, 1.5)
            widget = FolderBrowserWidget(start_dir=str(tmp_path))
            idx = widget._model.index(0, 0)
            widget._tree.selectionModel().select(
                idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
            )
            assert widget._btn_chimerax.isEnabled()
            assert "Open this file" in widget._btn_chimerax.toolTip()

    def test_stats_button_for_class3d_and_refine3d_data_star_only(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        class3d_dir = tmp_path / "Class3D" / "job001"
        refine3d_dir = tmp_path / "Refine3D" / "job002"
        class2d_dir = tmp_path / "Class2D" / "job003"
        for directory in (class3d_dir, refine3d_dir, class2d_dir):
            directory.mkdir(parents=True)
        class3d_star = class3d_dir / "run_data.star"
        refine3d_star = refine3d_dir / "run_data.star"
        class2d_star = class2d_dir / "run_data.star"
        generic_star = tmp_path / "data.star"
        for star_path in (
            class3d_star,
            refine3d_star,
            class2d_star,
            generic_star,
        ):
            star_path.write_text("_data_\n")
        (tmp_path / "particles.mrcs").write_bytes(b"\x00" * 1024)
        (tmp_path / "volume.mrc").write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        def select(path):
            widget.set_root_path(path.parent)
            idx = None
            for r in range(widget._model.rowCount()):
                if widget._model.file_path(widget._model.index(r, 0)) == str(path):
                    idx = widget._model.index(r, 0)
                    break
            assert idx is not None
            widget._tree.selectionModel().select(
                idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
            )

        for star_path in (class3d_star, refine3d_star):
            select(star_path)
            assert not widget._btn_stats.isHidden()
            assert widget._btn_stats.isEnabled()
            assert "variance" in widget._btn_stats.toolTip().lower()

        for star_path in (class2d_star, generic_star):
            select(star_path)
            assert widget._btn_stats.isHidden()

        select(tmp_path / "particles.mrcs")
        assert widget._btn_stats.isHidden()
        select(tmp_path / "volume.mrc")
        assert widget._btn_stats.isHidden()

    def test_eps_has_slice_mode_and_eps_button(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

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
        # EPS rasterizes to a 2D image, so it reuses the slice mode (button
        # labelled "EPS", mirroring the "PDF" label for .pdf files).
        assert widget._display_modes_for(str(tmp_path / "fig.eps")) == ["slice"]
        assert not widget._btn_slice.isHidden()
        assert widget._btn_slice.text() == "EPS"

    def test_pdf_has_image_slice_mode(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        pdf_file = tmp_path / "figure.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\nplaceholder")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        assert widget._display_modes_for(str(pdf_file)) == ["slice"]

    def test_display_modes_class2d_model_star(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        job_dir = tmp_path / "Class2D" / "job001"
        job_dir.mkdir(parents=True)
        star_file = job_dir / "model.star"
        star_file.write_text("dummy")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        assert widget._display_modes_for(str(star_file)) == [
            "text",
            "2dclasses",
        ]

    def test_display_modes_class2d_data_star_helical_gated(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        # Non-helical Class2D job: helical web apps (WhereIsMyClass,
        # HelicalPitch) must not be offered.
        plain_dir = tmp_path / "Class2D" / "job001"
        plain_dir.mkdir(parents=True)
        (plain_dir / "run_it100_model.star").write_text(
            "data_images\n\n_rlnIsHelix  0\n"
        )
        plain_star = plain_dir / "run_it100_data.star"
        plain_star.write_text("dummy")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        assert widget._display_modes_for(str(plain_star)) == [
            "slice",
            "gallery",
            "text",
            "images2star",
        ]

        # Helical Class2D job: the helical web apps are offered.
        helical_dir = tmp_path / "Class2D" / "job002"
        helical_dir.mkdir(parents=True)
        (helical_dir / "run_it100_model.star").write_text(
            "data_images\n\n_rlnIsHelix  1\n"
        )
        helical_star = helical_dir / "run_it100_data.star"
        helical_star.write_text("dummy")
        assert widget._display_modes_for(str(helical_star)) == [
            "slice",
            "gallery",
            "text",
            "images2star",
            "whereIsMyClass",
            "helicalPitch",
        ]

    def test_display_modes_class3d_optimiser_star(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        job_dir = tmp_path / "Class3D" / "job001"
        job_dir.mkdir(parents=True)
        star_file = job_dir / "optimiser.star"
        star_file.write_text("dummy")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        assert widget._display_modes_for(str(star_file)) == [
            "text",
            "optimiser",
        ]

    def test_display_modes_refine3d_model_star(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        job_dir = tmp_path / "Refine3D" / "job001"
        job_dir.mkdir(parents=True)
        star_file = job_dir / "model.star"
        star_file.write_text("dummy")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        assert widget._display_modes_for(str(star_file)) == [
            "text",
            "optimiser",
            "fsc",
            "trueFSC",
        ]

    def test_ps_ctf_treated_as_mrc(self, tmp_path, qapp):
        import numpy as np
        import mrcfile as _real_mrcfile
        from helicon.lib.gui.file_browser import (
            FolderBrowserWidget,
            _get_file_type_label,
        )

        # RELION writes CTF power spectra as single-slice MRC maps with a
        # ``.ctf`` suffix; they must behave exactly like a 2D MRC stack.
        ctf_file = tmp_path / "20170629_00049_frameImage_PS.ctf"
        with _real_mrcfile.new(str(ctf_file), overwrite=True) as mrc:
            mrc.set_data(np.zeros((1, 8, 8), dtype=np.float32))
        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        assert widget._display_modes_for(str(ctf_file)) == ["slice", "gallery"]
        assert widget._is_image_stack(str(ctf_file))
        assert not widget._is_volume(str(ctf_file))
        assert _get_file_type_label(str(ctf_file)) == "MRC"

    def test_raster_images_have_slice_button(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import (
            FolderBrowserWidget,
            _get_file_type_label,
        )

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        for suffix in (
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".gif",
            ".webp",
        ):
            img_file = tmp_path / f"image{suffix}"
            img_file.write_bytes(b"\x00" * 64)
            assert widget._display_modes_for(str(img_file)) == ["slice"]
            assert widget._is_image_stack(str(img_file))
            assert _get_file_type_label(str(img_file)) == suffix.lstrip(".").upper()

    def test_html_has_browser_button(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        html_file = tmp_path / "report.html"
        html_file.write_text("<html><body>hi</body></html>")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        assert widget._display_modes_for(str(html_file)) == ["html"]

        idx = None
        for r in range(widget._model.rowCount()):
            if widget._model.file_path(widget._model.index(r, 0)) == str(html_file):
                idx = widget._model.index(r, 0)
                break
        assert idx is not None
        widget._tree.selectionModel().select(
            idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
        )
        assert not widget._btn_html.isHidden()
        assert widget._btn_html.isEnabled()
        assert widget._btn_html.text() == "Browser"

    def test_file_browser_model_filter_wildcard(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_NAME

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
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_NAME

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
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_NAME

        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.mrc").write_bytes(b"b")
        model = FileBrowserModel(str(tmp_path))
        model.set_filter("")
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "a.txt" in names
        assert "b.mrc" in names

    def test_file_browser_model_filter_dirs_always_shown(self, tmp_path):
        from helicon.lib.gui.file_browser import FileBrowserModel, COL_NAME

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
        from helicon.lib.gui.file_browser import (
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
        from helicon.lib.gui.file_browser import (
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
        from helicon.lib.gui.file_browser import FileBrowserModel

        (tmp_path / "a.mrc").write_bytes(b"\x00" * 1024)
        model = FileBrowserModel(str(tmp_path))
        first = model.current_epoch()
        model.set_filter("*")  # triggers a directory reload -> epoch bump
        assert model.current_epoch() == first + 1

    def test_populate_fills_columns_async(self, tmp_path, qapp):
        from PySide6.QtWidgets import QApplication

        from helicon.lib.gui import file_browser
        from helicon.lib.gui.file_browser import (
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

        with patch.object(file_browser, "_get_file_info", fake_info):
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

        from helicon.lib.gui import file_browser
        from helicon.lib.gui.file_browser import (
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

        with patch.object(file_browser, "_get_file_info", fake_info):
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


class TestFolderCounts:
    def test_format_item_count(self):
        from helicon.lib.gui.file_browser import _format_item_count

        assert _format_item_count(0) == "0 items"
        assert _format_item_count(1) == "1 item"
        assert _format_item_count(42) == "42 items"

    def test_count_excludes_hidden_entries(self, tmp_path):
        from helicon.lib.gui.file_browser import _count_folder_entries

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.mrc").write_bytes(b"\x00")
        (sub / ".hidden").write_bytes(b"\x00")
        assert _count_folder_entries(str(sub)) == 1
        assert _count_folder_entries(str(tmp_path / "missing")) == 0

    def test_populate_folder_counts_async(self, tmp_path, qapp):
        from PySide6.QtWidgets import QApplication

        from helicon.lib.gui.file_browser import (
            COL_SIZE,
            FolderBrowserWidget,
            ROLE_SORT,
        )

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.mrc").write_bytes(b"\x00" * 512)
        (sub / "b.mrc").write_bytes(b"\x00" * 512)
        (tmp_path / "empty").mkdir()

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        widget._populate_file_info_async()
        import time

        deadline = time.time() + 10
        while any(t.isRunning() for t in widget._info_threads) and (
            time.time() < deadline
        ):
            QApplication.processEvents()
        QApplication.processEvents()

        model = widget._model
        for row, folderpath in model.dir_rows():
            assert model.item(row, COL_SIZE).text() != ""
        sub_row = model._row_for_filepath(str(sub))
        assert model.item(sub_row, COL_SIZE).text() == "2 items"
        assert model.item(sub_row, COL_SIZE).data(ROLE_SORT) == 2
        empty_row = model._row_for_filepath(str(tmp_path / "empty"))
        assert model.item(empty_row, COL_SIZE).text() == "0 items"
        assert model.item(empty_row, COL_SIZE).data(ROLE_SORT) == 0

    def test_initial_folder_counts_on_construction(self, tmp_path, qapp):
        from PySide6.QtWidgets import QApplication

        from helicon.lib.gui.file_browser import COL_SIZE, FolderBrowserWidget

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.mrc").write_bytes(b"\x00" * 512)

        # The initial directory must get its folder counts filled in the
        # background without any explicit refresh/navigation call.
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        import time

        deadline = time.time() + 10
        while any(t.isRunning() for t in widget._info_threads) and (
            time.time() < deadline
        ):
            QApplication.processEvents()
        QApplication.processEvents()

        sub_row = widget._model._row_for_filepath(str(sub))
        assert widget._model.item(sub_row, COL_SIZE).text() == "1 item"

    def test_filter_reload_restores_cached_counts(self, tmp_path, qapp):
        from PySide6.QtWidgets import QApplication

        from helicon.lib.gui.file_browser import COL_SIZE, FolderBrowserWidget

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.mrc").write_bytes(b"\x00" * 512)

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        import time

        deadline = time.time() + 10
        while any(t.isRunning() for t in widget._info_threads) and (
            time.time() < deadline
        ):
            QApplication.processEvents()
        QApplication.processEvents()
        assert widget._model._folder_counts[str(sub)] == 1

        # Filter reload rebuilds the rows; the count must come back from the
        # cache instead of staying blank.
        widget._apply_filter()
        deadline = time.time() + 10
        while any(t.isRunning() for t in widget._info_threads) and (
            time.time() < deadline
        ):
            QApplication.processEvents()
        QApplication.processEvents()

        sub_row = widget._model._row_for_filepath(str(sub))
        assert widget._model.item(sub_row, COL_SIZE).text() == "1 item"
        assert widget._model._folder_counts[str(sub)] == 1


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
        assert Path(out).exists()

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
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_svg_render_creates_file(self, qapp, tmp_path):
        from PySide6.QtGui import QImage

        qimg = QImage(20, 20, QImage.Format.Format_RGB32)
        qimg.fill(0xFFFFFFFF)
        out = str(tmp_path / "out.svg")
        display._render_qimage_vector(qimg, out, "svg")
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0


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
        assert Path(out).exists()

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
        # VisPy reports the middle button as integer 3 (its own enum; Qt uses 4).
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

            ev_mid = MouseEvent(type="mouse_press", button=3, pos=(5, 5))
            wrapped(ev_mid)
            assert cam_calls == []  # camera never saw the middle press

            ev_left = MouseEvent(type="mouse_press", button=1, pos=(5, 5))
            wrapped(ev_left)
            assert cam_calls == ["mouse_press"]  # left press still reaches camera
        finally:
            viewer.close()


class TestOrthogonalViewer:

    def test_display_modes_mrc_with_nz_gt1_includes_orthogonal(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        mrc_path = tmp_path / "volume.mrc"
        mrc_path.write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(widget, "_volume_has_nz_gt1", return_value=True):
            modes = widget._display_modes_for(str(mrc_path))
        assert "orthogonal" in modes

    def test_display_modes_mrc_with_nz_eq1_no_orthogonal(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        mrc_path = tmp_path / "volume.mrc"
        mrc_path.write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(widget, "_volume_has_nz_gt1", return_value=False):
            modes = widget._display_modes_for(str(mrc_path))
        # A single-slice (nz == 1) mrc is a 2D stack: no volume, no
        # ChimeraX, no orthogonal viewer.
        assert modes == ["slice", "gallery"]

    def test_display_modes_mrc_with_nz_gt1_includes_volume_and_chimerax(
        self, tmp_path, qapp
    ):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        mrc_path = tmp_path / "volume.mrc"
        mrc_path.write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(widget, "_volume_has_nz_gt1", return_value=True):
            modes = widget._display_modes_for(str(mrc_path))
        assert modes == [
            "slice",
            "volume",
            "gallery",
            "chimerax",
            "orthogonal",
            "proc3d",
        ]

    def test_display_modes_real_single_slice_mrc_is_2d_stack(self, tmp_path, qapp):
        import numpy as np
        import mrcfile as _real_mrcfile
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        mrc_path = tmp_path / "20170629_00049_frameImage_PS.mrc"
        with _real_mrcfile.new(str(mrc_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((1, 8, 8), dtype=np.float32))
            mrc.voxel_size = (2.0, 2.0, 2.0)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        assert widget._display_modes_for(str(mrc_path)) == ["slice", "gallery"]

    def test_display_modes_map_with_nz_gt1_includes_orthogonal(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        map_path = tmp_path / "volume.map"
        map_path.write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(widget, "_volume_has_nz_gt1", return_value=True):
            modes = widget._display_modes_for(str(map_path))
        assert "orthogonal" in modes

    def test_display_modes_non_mrc_no_orthogonal(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        star_path = tmp_path / "particles.star"
        star_path.write_text("data_\n\nloop_\n_rlnImageName\n")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        modes = widget._display_modes_for(str(star_path))
        assert "orthogonal" not in modes

    def test_open_file_orthogonal_mode_calls_viewer(self, tmp_path):
        import numpy as np
        import mrcfile as _real_mrcfile

        mock_viewer = MagicMock()
        vol_path = tmp_path / "volume.mrc"
        with _real_mrcfile.new(str(vol_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((4, 8, 8), dtype=np.float32))
            mrc.voxel_size = (1.5, 1.5, 1.5)

        with patch("helicon.commands.display._open_orthogonal_viewer") as mock_open_ov:
            display._open_file(mock_viewer, str(vol_path), mode="orthogonal")
            mock_open_ov.assert_called_once()

    def test_open_file_orthogonal_mode_nz_eq1_skips(self, tmp_path):
        import numpy as np
        import mrcfile as _real_mrcfile

        mock_viewer = MagicMock()
        vol_path = tmp_path / "volume.mrc"
        with _real_mrcfile.new(str(vol_path), overwrite=True) as mrc:
            mrc.set_data(np.zeros((1, 8, 8), dtype=np.float32))
            mrc.voxel_size = (1.5, 1.5, 1.5)

        with patch("helicon.commands.display._open_orthogonal_viewer") as mock_open_ov:
            display._open_file(mock_viewer, str(vol_path), mode="orthogonal")
            mock_open_ov.assert_not_called()
            mock_viewer.add_image.assert_called_once()

    def test_sliceview_instantiates(self, qapp):
        from helicon.lib.gui.gallery_widget import _SliceView

        view = _SliceView()
        assert view is not None
        assert view._zoom == 1.0
        assert view._brightness == 0.0
        assert view._contrast == 1.0
        assert view._gamma == 1.0
        assert view._log_transform is False
        view.deleteLater()

    def test_controlbar_instantiates(self, qapp):
        from helicon.lib.gui.gallery_widget import _ControlBar, _BCGPanel

        bar = _ControlBar(64, 48, 32)
        assert bar is not None
        bar.deleteLater()

        bcg = _BCGPanel()
        b, c, g = bcg.get_bcg()
        assert b == pytest.approx(0.0)
        assert c == pytest.approx(1.0)
        assert g == pytest.approx(1.0)
        bcg.deleteLater()

    def test_orthogonal_viewer_widget_instantiates(self, qapp):
        import numpy as np
        from PySide6.QtCore import QSettings
        from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget

        vol = np.random.rand(8, 10, 12).astype(np.float32)
        settings = QSettings("helicon", "display")
        old_theme = settings.value("theme", None)
        try:
            settings.setValue("theme", "Light")
            w = OrthogonalViewerWidget(vol, apix=2.0, name="test.mrc")
            assert w is not None
            assert w._nx == 12
            assert w._ny == 10
            assert w._nz == 8
            assert w._pos == [6, 5, 4]
            assert (
                w._xy_view.palette().color(w._xy_view.backgroundRole()).name()
                == "#ffffff"
            )
            assert w._ctrl.palette().color(w._ctrl.backgroundRole()).name() != "#333333"
            settings.setValue("theme", "Dark")
            w._apply_display_theme()
            assert (
                w._xy_view.palette().color(w._xy_view.backgroundRole()).name()
                == "#2d2d2d"
            )
            w.deleteLater()
        finally:
            if old_theme is None:
                settings.remove("theme")
            else:
                settings.setValue("theme", old_theme)

    def test_orthogonal_viewer_get_slice(self, qapp):
        import numpy as np
        from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget

        vol = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        w = OrthogonalViewerWidget(vol, apix=1.0, name="test")
        sl = w._get_slice(0, 1)
        np.testing.assert_array_equal(sl, vol[1])
        sl = w._get_slice(1, 2)
        np.testing.assert_array_equal(sl, vol[:, 2, :])
        sl = w._get_slice(2, 3)
        np.testing.assert_array_equal(sl, vol[:, :, 3])
        w.deleteLater()

    def test_orthogonal_viewer_panel_order_is_z_x_y(self, qapp):
        import numpy as np
        from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget

        w = OrthogonalViewerWidget(np.zeros((8, 8, 8), dtype=np.float32))
        layout = w.layout()
        positions = {}
        for i in range(layout.count()):
            item = layout.itemAt(i)
            row, column, row_span, column_span = layout.getItemPosition(i)
            positions[item.widget()] = (row, column, row_span, column_span)

        assert positions[w._xy_view][:2] == (0, 0)  # Z
        assert positions[w._xz_view][:2] == (0, 1)  # X
        assert positions[w._yz_view][:2] == (1, 0)  # Y
        assert w._xy_view._axis_label == "Z"
        assert w._xz_view._axis_label == "X"
        assert w._yz_view._axis_label == "Y"
        w.deleteLater()

    def test_orthogonal_viewer_click_updates_position(self, qapp):
        import numpy as np
        from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget

        vol = np.zeros((8, 8, 8), dtype=np.float32)
        w = OrthogonalViewerWidget(vol, apix=1.0, name="test")
        w._on_click(0, 5.0, 3.0)
        assert w._pos[0] == 5
        assert w._pos[1] == 3
        w._on_click(1, 2.0, 7.0)
        assert w._pos[2] == 2
        assert w._pos[1] == 7
        w._on_click(2, 6.0, 1.0)
        assert w._pos[0] == 6
        assert w._pos[2] == 1
        w.deleteLater()

    def test_orthogonal_viewer_slider_updates_position(self, qapp):
        import numpy as np
        from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget

        vol = np.zeros((8, 8, 8), dtype=np.float32)
        w = OrthogonalViewerWidget(vol, apix=1.0, name="test")
        w._on_slider_position(3, 4, 5)
        assert w._pos == [3, 4, 5]
        w.deleteLater()

    def test_controlbar_spinbox_sync(self, qapp):
        from helicon.lib.gui.gallery_widget import _ControlBar

        bar = _ControlBar(10, 12, 14)
        bar._x_slider.setValue(4)
        assert bar._x_spin.value() == 4
        bar._y_spin.setValue(7)
        assert bar._y_slider.value() == 7
        bar.deleteLater()

    def test_orthogonal_viewer_reset_position_center(self, qapp):
        import numpy as np
        from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget

        vol1 = np.zeros((4, 4, 4), dtype=np.float32)
        vol2 = np.zeros((10, 12, 14), dtype=np.float32)
        w = OrthogonalViewerWidget(vol1, apix=1.0, name="test")
        assert w._pos == [2, 2, 2]
        w.set_volume(vol2, reset_position=True)
        assert w._pos == [7, 6, 5]
        w.deleteLater()

    def test_orthogonal_viewer_set_volume_rejects_degenerate(self, qapp):
        import numpy as np
        import pytest
        from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget

        w = OrthogonalViewerWidget(np.zeros((8, 8, 8), dtype=np.float32))
        with pytest.raises(ValueError):
            w.set_volume(np.zeros((0, 8, 8), dtype=np.float32), reset_position=True)
        w.deleteLater()

    def test_orthogonal_viewer_axes_labels(self, qapp):
        import numpy as np
        from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget

        w = OrthogonalViewerWidget(np.zeros((8, 8, 8), dtype=np.float32))
        assert w._xy_view._axes_h == "x"
        assert w._xy_view._axes_v == "y"
        assert w._xz_view._axes_h == "z"
        assert w._xz_view._axes_v == "y"
        assert w._yz_view._axes_h == "x"
        assert w._yz_view._axes_v == "z"
        w.deleteLater()


class TestTrueFscPanel(object):

    def test_empty_launch_leaves_selectors_blank(self, qapp):
        from helicon.commands import display

        dialog = display._launch_truefsc_maps(parent=None)
        qapp.processEvents()
        assert dialog._map1_edit.text() == ""
        assert dialog._map2_edit.text() == ""
        assert dialog._mask_edit.text() == ""
        assert dialog._workers == []
        dialog.close()
        qapp.processEvents()

    def test_prefilled_launch_populates_map_selectors(
        self, qapp, tmp_path, monkeypatch
    ):
        import mrcfile
        import numpy as np
        from PySide6.QtCore import QThread

        from helicon.commands import display

        half1 = tmp_path / "half1.mrc"
        half2 = tmp_path / "half2.mrc"
        for path in (half1, half2):
            with mrcfile.new(str(path), overwrite=True) as m:
                m.set_data(np.zeros((4, 4, 4), dtype=np.float32))
                m.voxel_size = 1.35

        monkeypatch.setattr(QThread, "start", lambda self: None)
        dialog = display._launch_truefsc_maps(
            map1=str(half1), map2=str(half2), parent=None
        )
        qapp.processEvents()
        assert dialog._map1_edit.text() == str(half1.resolve())
        assert dialog._map2_edit.text() == str(half2.resolve())
        assert dialog._mask_edit.text() == ""
        assert dialog._seq == 1
        dialog.close()
        qapp.processEvents()


class TestDisplayButtonSorting:

    def test_reorder_sorts_visible_buttons_alphabetically(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        modes = ["hi3d", "slice", "volume", "gallery", "chimerax", "orthogonal"]
        for attr, mode in widget._DISPLAY_BUTTONS:
            getattr(widget, attr).setVisible(mode in modes)
        widget._reorder_buttons_alphabetically(modes)

        layout = widget._action_bar.layout()
        display_btn_set = {
            id(getattr(widget, attr)) for attr, _ in widget._DISPLAY_BUTTONS
        }
        visible_labels = [
            layout.itemAt(i).widget().text()
            for i in range(layout.count())
            if layout.itemAt(i).widget() is not None
            and id(layout.itemAt(i).widget()) in display_btn_set
            and not layout.itemAt(i).widget().isHidden()
        ]
        assert visible_labels == sorted(visible_labels, key=str.lower)

    def test_reorder_covers_all_display_buttons(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        all_modes = [mode for _, mode in widget._DISPLAY_BUTTONS]
        for attr, mode in widget._DISPLAY_BUTTONS:
            getattr(widget, attr).setVisible(mode in all_modes)
        widget._reorder_buttons_alphabetically(all_modes)

        layout = widget._action_bar.layout()
        display_btn_set = {
            id(getattr(widget, attr)) for attr, _ in widget._DISPLAY_BUTTONS
        }
        visible_labels = [
            layout.itemAt(i).widget().text()
            for i in range(layout.count())
            if layout.itemAt(i).widget() is not None
            and id(layout.itemAt(i).widget()) in display_btn_set
            and not layout.itemAt(i).widget().isHidden()
        ]
        assert len(visible_labels) == len(widget._DISPLAY_BUTTONS)
        assert visible_labels == sorted(visible_labels, key=str.lower)

    def test_reorder_hides_inactive_buttons(self, tmp_path, qapp):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        modes = ["slice", "volume"]
        for attr, mode in widget._DISPLAY_BUTTONS:
            getattr(widget, attr).setVisible(mode in modes)
        widget._reorder_buttons_alphabetically(modes)

        layout = widget._action_bar.layout()
        display_btn_set = {
            id(getattr(widget, attr)) for attr, _ in widget._DISPLAY_BUTTONS
        }
        visible_labels = [
            layout.itemAt(i).widget().text()
            for i in range(layout.count())
            if layout.itemAt(i).widget() is not None
            and id(layout.itemAt(i).widget()) in display_btn_set
            and not layout.itemAt(i).widget().isHidden()
        ]
        assert visible_labels == ["3D Volume", "Image Slice"]


class TestReexecMacosDisplay:
    """macOS re-exec of ``helicon display`` under a Helicon-named binary."""

    def test_skips_when_not_macos(self, monkeypatch):
        import helicon.helicon as cli

        monkeypatch.setattr(cli.sys, "platform", "linux")
        monkeypatch.setattr(
            cli.os,
            "execve",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not exec")),
        )
        assert cli._maybe_reexec_macos_display() is None

    def test_skips_after_identity_reexec(self, monkeypatch):
        import helicon.helicon as cli

        monkeypatch.setattr(cli.sys, "platform", "darwin")
        monkeypatch.setenv("HELICON_MACOS_IDENTITY", "1")
        monkeypatch.setattr(
            cli.os,
            "execve",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not exec")),
        )
        assert cli._maybe_reexec_macos_display() is None

    def test_skips_unless_display_in_argv(self, monkeypatch):
        import helicon.helicon as cli

        monkeypatch.setattr(cli.sys, "platform", "darwin")
        monkeypatch.delenv("HELICON_MACOS_IDENTITY", raising=False)
        monkeypatch.setattr(cli.sys, "argv", ["helicon", "proc3d"])
        monkeypatch.setattr(
            cli.os,
            "execve",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not exec")),
        )
        assert cli._maybe_reexec_macos_display() is None

    def test_reexec_copies_interpreter_as_Helicon(self, monkeypatch, tmp_path):
        import helicon.helicon as cli

        monkeypatch.setattr(cli.sys, "platform", "darwin")
        monkeypatch.delenv("HELICON_MACOS_IDENTITY", raising=False)
        monkeypatch.setattr(cli.sys, "argv", ["helicon", "display", "--folder", "."])
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

        python_bin = tmp_path / "python3.14"
        python_bin.write_bytes(b"\x0a\x0b\x0c\x0d" * 16)
        python_bin.chmod(0o755)
        monkeypatch.setattr(cli.sys, "executable", str(python_bin))
        monkeypatch.setattr(cli.sys, "prefix", str(tmp_path / "env"))

        captured = {}

        def fake_execve(path, cmd, env):
            captured["path"] = path
            captured["cmd"] = list(cmd)
            captured["env"] = dict(env)

        monkeypatch.setattr(cli.os, "execve", fake_execve)

        cli._maybe_reexec_macos_display()

        helicon_copy = tmp_path / "helicon_bin" / "Helicon"
        assert helicon_copy.is_file()
        assert helicon_copy.read_bytes() == python_bin.read_bytes()
        assert captured["path"] == str(helicon_copy)
        assert captured["cmd"][0] == str(helicon_copy)
        # argv[1] is the original script that the interpreter should run.
        assert captured["cmd"][1] == "helicon"
        assert captured["cmd"][2:] == ["display", "--folder", "."]
        assert captured["env"]["HELICON_MACOS_IDENTITY"] == "1"
        assert captured["env"]["PYTHONHOME"] == str((tmp_path / "env").resolve())

    def test_reexec_replaces_legacy_symlink_with_real_copy(self, monkeypatch, tmp_path):
        """A leftover ``Helicon`` symlink (from older launches) must be replaced
        by a real executable so the Dock does not resolve back to ``python``."""
        import helicon.helicon as cli

        monkeypatch.setattr(cli.sys, "platform", "darwin")
        monkeypatch.delenv("HELICON_MACOS_IDENTITY", raising=False)
        monkeypatch.setattr(cli.sys, "argv", ["helicon", "display"])
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

        python_bin = tmp_path / "python3.14"
        python_bin.write_bytes(b"\x0a\x0b\x0c\x0d" * 16)
        python_bin.chmod(0o755)
        monkeypatch.setattr(cli.sys, "executable", str(python_bin))
        monkeypatch.setattr(cli.sys, "prefix", str(tmp_path / "env"))

        bin_dir = tmp_path / "helicon_bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        helicon_symlink = bin_dir / "Helicon"
        helicon_symlink.symlink_to(python_bin)

        captured = {}

        def fake_execve(path, cmd, env):
            captured["path"] = path
            captured["env"] = dict(env)

        monkeypatch.setattr(cli.os, "execve", fake_execve)

        cli._maybe_reexec_macos_display()

        assert helicon_symlink.is_symlink() is False
        assert helicon_symlink.is_file()
        assert helicon_symlink.read_bytes() == python_bin.read_bytes()
        assert captured["path"] == str(helicon_symlink)
        assert captured["env"]["HELICON_MACOS_IDENTITY"] == "1"

    def test_reexec_does_not_loop_after_success(self, monkeypatch, tmp_path):
        """The identity guard is set on the re-exec env so the child returns early."""
        import helicon.helicon as cli

        monkeypatch.setattr(cli.sys, "platform", "darwin")
        monkeypatch.delenv("HELICON_MACOS_IDENTITY", raising=False)
        monkeypatch.setattr(cli.sys, "argv", ["helicon", "display"])
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

        python_bin = tmp_path / "python3.14"
        python_bin.write_bytes(b"x" * 32)
        python_bin.chmod(0o755)
        monkeypatch.setattr(cli.sys, "executable", str(python_bin))
        monkeypatch.setattr(cli.sys, "prefix", str(tmp_path / "env"))

        captured = {}

        def fake_execve(path, cmd, env):
            captured["env"] = dict(env)

        monkeypatch.setattr(cli.os, "execve", fake_execve)
        cli._maybe_reexec_macos_display()
        assert captured["env"]["HELICON_MACOS_IDENTITY"] == "1"


class TestImages2StarDialog(object):
    """Phase-0 Images2Star panel: threaded load, preview, and save."""

    @staticmethod
    def _df_with_optics():
        data = pd.DataFrame(
            {"rlnImageName": ["a.mrc", "b.mrc"], "rlnAngleRot": [1.0, 2.0]}
        )
        optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnPixelSize": [1.2]})
        data.attrs["optics"] = optics
        data.attrs["convention"] = "relion"
        return data

    @staticmethod
    def _open(path, **kwargs):
        if Images2StarDialog is None:
            pytest.skip("PySide6 required for the Images2Star panel")
        return Images2StarDialog(path, **kwargs)

    @staticmethod
    def _pump(qapp):
        qapp.processEvents()

    def test_loads_via_injected_loader_and_shows_preview(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            assert dialog._data is not None
            assert len(dialog._data) == 2
            assert "rows" in dialog._status.toPlainText()
            assert "optics" in dialog._status.toPlainText()
            assert dialog._btn_save.isEnabled()
            model = dialog._table.model()
            assert model.rowCount() == 2
            assert model.columnCount() == 2
            assert dialog._particles_label.text() == "Particles"
            assert not dialog._particles_label.isHidden()
            optics_model = dialog._optics_table.model()
            assert optics_model is not None
            assert optics_model.rowCount() == 1
            assert optics_model.columnCount() == 2
            assert dialog._optics_label.text() == "Optics groups"
            assert not dialog._optics_label.isHidden()
            assert not dialog._optics_table.isHidden()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_no_optics_hides_optics_table(self, qapp):
        data = pd.DataFrame({"rlnImageName": ["a.mrc"]})
        dialog = self._open("dummy.star", loader=lambda p: data)
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            assert dialog._data is not None
            assert dialog._table.model() is not None
            assert dialog._optics_table.model() is None
            assert dialog._optics_label.isHidden()
            assert dialog._optics_table.isHidden()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_edit_mode_does_not_clip_dense_row_text(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            table = dialog._table
            index = table.model().index(0, 1)
            table.edit(index)
            self._pump(qapp)
            editor = table.indexWidget(index)
            assert editor is not None
            # Rows are sized to the frameless editor's sizeHint (plus one,
            # since the cell rect is one pixel shorter than the row); the
            # editor must keep at least 3px of slack over the font or the
            # cell text is clipped top and bottom while editing.
            assert editor.contentsRect().height() >= editor.fontMetrics().height() + 3
            assert editor.height() <= table.verticalHeader().sectionSize(0)
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_load_path_reuses_window_with_new_dataset(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)
            first = dialog._data

            # Reuse: reload a different file into the same window.
            dialog.load_path("other.star")
            dialog._workers[-1].wait()
            self._pump(qapp)

            assert dialog._data is not first
            assert "other.star" in dialog.windowTitle()
            assert dialog._btn_save.isEnabled()
            assert dialog._table.model().rowCount() == 2

            # The transformation stack survives the reload.
            dialog._stack_model.add("select", "rlnImageName")
            dialog.load_path("third.star")
            dialog._workers[-1].wait()
            self._pump(qapp)
            assert dialog._stack_model.rowCount() == 1
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_opens_with_empty_selector_when_no_path(self, qapp):
        dialog = self._open(None, loader=lambda p: self._df_with_optics())
        try:
            assert dialog._path == ""
            assert not hasattr(dialog, "_load_worker")
            assert dialog._workers == []
            assert dialog._path_edit.text() == ""
            assert dialog._path_edit.placeholderText()
            assert not dialog._btn_save.isEnabled()
            assert dialog.windowTitle() == "Images2Star"
            assert "Select a dataset file" in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_load_from_field_loads_typed_path(self, qapp):
        dialog = self._open(None, loader=lambda p: self._df_with_optics())
        try:
            dialog._path_edit.setText("typed.star")
            dialog._load_from_field()
            dialog._workers[-1].wait()
            self._pump(qapp)

            assert dialog._path.endswith("typed.star")
            assert dialog._path_edit.text() == dialog._path
            assert dialog._data is not None
            assert dialog._btn_save.isEnabled()
            assert "typed.star" in dialog.windowTitle()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_load_from_field_reports_empty_entry(self, qapp):
        dialog = self._open(None, loader=lambda p: self._df_with_optics())
        try:
            dialog._path_edit.setText("")
            dialog._load_from_field()
            assert dialog._path == ""
            assert "Enter a path to a dataset file" in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_default_output_falls_back_when_no_path(self, qapp):
        dialog = self._open(None, loader=lambda p: self._df_with_optics())
        try:
            assert dialog._default_output_path() == str(
                Path.cwd() / "dataset.processed.star"
            )
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_dialog_styles_follow_saved_theme(self, qapp):
        from PySide6.QtCore import QSettings

        settings = QSettings("helicon", "display")
        try:
            for theme, window, button in (
                ("Dark", "#2d2d2d", "#3c3c3c"),
                ("Light", "#f4f4f4", "#ffffff"),
            ):
                settings.setValue("theme", theme)
                dialog = self._open(
                    "dummy.star", loader=lambda p: self._df_with_optics()
                )
                try:
                    assert dialog.palette().window().color().name() == window
                    assert dialog._btn_save.palette().button().color().name() == button
                    assert f"background-color: {window}" in dialog.styleSheet()
                    assert dialog._status.palette().windowText().color().name() == (
                        "#cccccc" if theme == "Dark" else "#202020"
                    )
                finally:
                    dialog.close()
                    dialog.deleteLater()
                    self._pump(qapp)
        finally:
            settings.remove("theme")

    def test_open_dialog_refreshes_when_theme_changes(self, qapp):
        from PySide6.QtCore import QSettings

        settings = QSettings("helicon", "display")
        settings.setValue("theme", "Light")
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            assert dialog.palette().window().color().name() == "#f4f4f4"

            settings.setValue("theme", "Dark")
            display._refresh_display_theme_windows()

            assert dialog.palette().window().color().name() == "#2d2d2d"
            assert "background-color: #2d2d2d" in dialog.styleSheet()
            assert "background-color: #3c3c3c" in dialog._btn_save.styleSheet()
            assert "color: #cccccc" in dialog._btn_save.styleSheet()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)
            settings.remove("theme")

    def test_large_dataset_preview_is_virtual_over_all_rows(self, qapp):
        n = 2000
        data = pd.DataFrame(
            {
                "rlnImageName": [f"frame_{i:04d}.mrc" for i in range(n)],
                "rlnAngleRot": np.linspace(0.0, 180.0, n),
            }
        )
        dialog = self._open("dummy.star", loader=lambda p: data)
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            model = dialog._table.model()
            assert model.rowCount() == n
            assert model.columnCount() == 2
            # Rows beyond the old 500-row cap are reachable on demand.
            late_index = model.index(1234, 0)
            assert model.data(late_index) == "frame_1234.mrc"
            assert dialog._particles_label.text() == "Particles"
            assert "capped" not in dialog._status.toPlainText()
            assert f"{n:,} rows" in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_preview_table_uses_virtual_scroll_settings(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            table = dialog._table
            optics_table = dialog._optics_table
            assert table.verticalScrollMode() == (
                QAbstractItemView.ScrollMode.ScrollPerPixel
            )
            assert table.horizontalScrollMode() == (
                QAbstractItemView.ScrollMode.ScrollPerPixel
            )
            assert not table.wordWrap()
            assert optics_table.verticalScrollMode() == (
                QAbstractItemView.ScrollMode.ScrollPerPixel
            )
            assert optics_table.horizontalScrollMode() == (
                QAbstractItemView.ScrollMode.ScrollPerPixel
            )
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_preview_tables_use_compact_font_and_rows(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            # Matches the file browser's compactness: 12px font, and rows
            # sized to the frameless editor's sizeHint plus one pixel (the
            # cell rect is one pixel shorter than the row) so editing never
            # clips the text, instead of the style default.
            from PySide6.QtWidgets import QLineEdit

            sample = QLineEdit()
            sample.setFont(dialog._table.font())
            sample.setFrame(False)
            sample.setContentsMargins(0, 0, 0, 0)
            sample.setTextMargins(0, 0, 0, 0)
            for table in (dialog._table, dialog._optics_table):
                assert table.fontInfo().pixelSize() == 12
                vh = table.verticalHeader()
                assert vh.defaultSectionSize() == sample.sizeHint().height() + 1
                assert vh.minimumSectionSize() <= vh.fontMetrics().height()
            assert (
                dialog._table.rowHeight(0)
                == dialog._table.verticalHeader().defaultSectionSize()
            )
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_preview_splitter_drag_and_optics_collapse(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            splitter = dialog._splitter
            assert splitter.count() == 2
            optics_pane = splitter.widget(1)
            assert not optics_pane.isHidden()

            # Dragging the handle down to a sliver must not be blocked by the
            # table's style-default minimum height (~90px).
            splitter.setSizes([splitter.height() - 30, 30])
            self._pump(qapp)
            assert splitter.sizes()[1] <= 60
            dragged_sizes = list(splitter.sizes())

            # A dataset without optics collapses the optics pane entirely.
            plain = pd.DataFrame({"rlnImageName": ["a.mrc"], "rlnAngleRot": [1.0]})
            dialog._refresh_preview(plain)
            self._pump(qapp)
            assert optics_pane.isHidden()

            # Restoring optics brings the pane back at the dragged size.
            dialog._refresh_preview(self._df_with_optics())
            self._pump(qapp)
            assert not optics_pane.isHidden()
            assert splitter.sizes() == dragged_sizes
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_table_sorts_by_column_ascending(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            dialog._table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
            model = dialog._table.model()
            # rlnDefocusU = [2.0, 1.0, 2.0, 1.0] -> 1.0, 1.0, 2.0, 2.0 with
            # ties keeping the original (stable) row order.
            assert [model.data(model.index(r, 1)) for r in range(4)] == [
                "1.0",
                "1.0",
                "2.0",
                "2.0",
            ]
            # The working dataset keeps its original order.
            assert dialog._data["rlnDefocusU"].tolist() == [2.0, 1.0, 2.0, 1.0]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_table_sorts_by_column_descending(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            dialog._table.sortByColumn(1, Qt.SortOrder.DescendingOrder)
            model = dialog._table.model()
            assert [model.data(model.index(r, 1)) for r in range(4)] == [
                "2.0",
                "2.0",
                "1.0",
                "1.0",
            ]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_table_sorts_numerically_not_lexicographically(self, qapp):
        data = pd.DataFrame(
            {
                "rlnDefocusU": [10.0, 2.0, 100.0, 1.0],
            }
        )
        dialog = self._open("dummy.star", loader=lambda p: data)
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            dialog._table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
            model = dialog._table.model()
            assert [model.data(model.index(r, 0)) for r in range(4)] == [
                "1.0",
                "2.0",
                "10.0",
                "100.0",
            ]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_table_sort_keeps_missing_values_last(self, qapp):
        data = pd.DataFrame({"rlnDefocusU": [3.0, None, 1.0, 2.0]})
        dialog = self._open("dummy.star", loader=lambda p: data)
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            dialog._table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
            model = dialog._table.model()
            assert [model.data(model.index(r, 0)) for r in range(4)] == [
                "1.0",
                "2.0",
                "3.0",
                "",
            ]

            dialog._table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
            assert [model.data(model.index(r, 0)) for r in range(4)] == [
                "3.0",
                "2.0",
                "1.0",
                "",
            ]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_edit_after_sort_writes_correct_source_row(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            dialog._table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
            model = dialog._table.model()
            # View row 0 is source row 1 (rlnDefocusU == 1.0).
            index = model.index(0, 1)
            assert model.setData(index, "7.5")

            assert dialog._data["rlnDefocusU"].tolist()[1] == 7.5
            assert model.data(index) == "7.5"
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_optics_table_sorts_too(self, qapp):
        optics = pd.DataFrame(
            {
                "rlnOpticsGroup": [2, 1, 3],
                "rlnPixelSize": [1.2, 0.9, 1.5],
            }
        )
        data = pd.DataFrame({"rlnImageName": ["a.mrc", "b.mrc"]})
        data.attrs["optics"] = optics
        dialog = self._open("dummy.star", loader=lambda p: data)
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            dialog._optics_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)
            model = dialog._optics_table.model()
            assert [model.data(model.index(r, 1)) for r in range(3)] == [
                "1.5",
                "1.2",
                "0.9",
            ]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_refresh_preview_clears_sort_indicator(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            dialog._table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
            assert dialog._table.horizontalHeader().sortIndicatorSection() == 1

            dialog._refresh_preview(dialog._data)
            assert dialog._table.horizontalHeader().sortIndicatorSection() == -1
            assert dialog._optics_table.horizontalHeader().sortIndicatorSection() == -1
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_many_optics_groups_preview_is_virtual(self, qapp):
        n = 2000
        optics = pd.DataFrame(
            {
                "rlnOpticsGroup": range(1, n + 1),
                "rlnVoltage": np.linspace(300.0, 300.0, n),
                "rlnPixelSize": np.linspace(1.0, 0.5, n),
            }
        )
        data = pd.DataFrame({"rlnImageName": ["a.mrc", "b.mrc"]})
        data.attrs["optics"] = optics
        dialog = self._open("dummy.star", loader=lambda p: data)
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            model = dialog._optics_table.model()
            assert model.rowCount() == n
            assert model.columnCount() == 3
            # Late rows render on demand, exactly like the particles table.
            late_index = model.index(1234, 0)
            assert model.data(late_index) == "1235"
            assert dialog._optics_label.text() == "Optics groups"
            assert not dialog._optics_table.isHidden()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_labels_are_selectable(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            for label in (
                dialog._status,
                dialog._particles_label,
                dialog._optics_label,
            ):
                flags = label.textInteractionFlags()
                assert flags & Qt.TextInteractionFlag.TextSelectableByMouse
                assert flags & Qt.TextInteractionFlag.TextSelectableByKeyboard
            # The file selector line edit is fully selectable/editable by its
            # nature (replaced the old read-only title label).
            assert dialog._path_edit.isReadOnly() is False
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_save_and_close_share_transformations_action_row(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)
            dialog.resize(940, 760)
            dialog.show()
            self._pump(qapp)

            # Save As and Close sit in the transformations action row, not in
            # the status band, so the status message gets the full width.
            assert dialog._status.parent() is dialog
            ops_group = dialog._ops_group
            assert dialog._btn_save.parent() is ops_group
            assert dialog._btn_close.parent() is ops_group
            # The status band is as wide as the window (no buttons flanking it).
            assert (
                dialog._status.geometry().width()
                >= dialog.width() - dialog.layout().contentsMargins().left() * 2
            )
            assert dialog._btn_save.isVisible()
            assert dialog._btn_close.isVisible()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_section_headers_are_bold(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            for label in (dialog._particles_label, dialog._optics_label):
                assert label.font().bold()
                assert label.text() in ("Particles", "Optics groups")
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_table_copy_puts_tsv_on_clipboard(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            dialog._table.selectRow(0)
            dialog._table._copy_selection()
            copied = qapp.clipboard().text()
            assert "rlnImageName" in copied
            assert "rlnAngleRot" in copied
            assert "a.mrc" in copied
            assert "1.0" in copied

            dialog._optics_table.selectRow(0)
            dialog._optics_table._copy_selection()
            copied = qapp.clipboard().text()
            assert "rlnOpticsGroup" in copied
            assert "1" in copied
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_copy_shortcut_triggered_by_keypress(self, qapp):
        from PySide6.QtTest import QTest

        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            dialog._table.selectRow(0)
            QTest.keyClick(
                dialog._table,
                Qt.Key_C,
                Qt.KeyboardModifier.ControlModifier,
            )
            copied = qapp.clipboard().text()
            assert "rlnImageName" in copied
            assert "a.mrc" in copied
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_save_via_injected_saver(self, tmp_path, qapp):
        captured = {}

        def saver(data, path):
            captured["data"] = data
            captured["path"] = path

        dialog = self._open(
            "dummy.star",
            loader=lambda p: self._df_with_optics(),
            saver=saver,
        )
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            out = str(tmp_path / "out.star")
            dialog._save_to(out)
            dialog._workers[-1].wait()
            self._pump(qapp)

            assert captured["data"] is dialog._data
            assert captured["path"] == out
            assert "Saved" in dialog._status.toPlainText()
            assert "out.star" in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_load_failure_reported(self, qapp):
        def loader(path):
            raise RuntimeError("boom")

        dialog = self._open("missing.star", loader=loader)
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            assert dialog._data is None
            assert "Failed to load dataset" in dialog._status.toPlainText()
            assert "RuntimeError" in dialog._status.toPlainText()
            assert not dialog._btn_save.isEnabled()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_default_loader_and_saver_end_to_end(self, tmp_path, qapp):
        import mrcfile

        for name in ("a.mrc", "b.mrc"):
            mrcfile.new(
                str(tmp_path / name),
                data=np.zeros((2, 2), dtype=np.float32),
                overwrite=True,
            )
        star = tmp_path / "images.star"
        star.write_text(
            "data_images\n\nloop_\n"
            "_rlnImageName #1\n_rlnAngleRot #2\n"
            "a.mrc 1.5\nb.mrc 2.5\n"
        )
        dialog = self._open(str(star))
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            assert dialog._data is not None
            assert len(dialog._data) == 2

            out = tmp_path / "out.star"
            dialog._save_to(str(out))
            dialog._workers[-1].wait()
            self._pump(qapp)

            assert out.is_file()
            assert "Saved" in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    # ------------------------------------------------------------------
    # Phase 1: ordered operation stack

    @staticmethod
    def _df_for_ops():
        """Dataset with values that select/sort can discriminate."""
        data = pd.DataFrame(
            {
                "rlnMicrographName": [
                    "/d/A.mrc",
                    "/d/A.mrc",
                    "/d/B.mrc",
                    "/d/B.mrc",
                ],
                "rlnDefocusU": [2.0, 1.0, 2.0, 1.0],
                "rlnDefocusV": [10.0, 20.0, 5.0, 15.0],
            }
        )
        optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnPixelSize": [1.2]})
        data.attrs["optics"] = optics
        data.attrs["convention"] = "relion"
        return data

    def _open_ops(self, qapp, **kwargs):
        dialog = self._open("dummy.star", loader=lambda p: self._df_for_ops(), **kwargs)
        dialog._load_worker.wait()
        self._pump(qapp)
        return dialog

    @staticmethod
    def _select_op(dialog, name):
        index = dialog._ops_combo.findData(name)
        assert index >= 0, f"operation {name} missing from combo"
        dialog._ops_combo.setCurrentIndex(index)

    def test_operations_combo_excludes_file_writing_options(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            names = {
                dialog._ops_combo.itemData(i) for i in range(dialog._ops_combo.count())
            }
            assert {"select", "sortby", "setParm"} <= names
            for excluded in (
                "process",
                "createStack",
                "splitByMicrograph",
                "extractHelices",
                "path",
                "sets",
            ):
                assert excluded not in names
            assert not dialog._btn_apply.isEnabled()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_add_operation_validates_and_appends(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "select")
            dialog._param_edit.setText("rlnDefocusV 5,20")
            dialog._btn_add.click()

            assert dialog._stack_model.rowCount() == 1
            assert dialog._stack_model.operations() == [("select", "rlnDefocusV 5,20")]
            assert dialog._btn_apply.isEnabled()

            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusU")
            dialog._btn_add.click()
            assert dialog._stack_model.operations() == [
                ("select", "rlnDefocusV 5,20"),
                ("sortby", "rlnDefocusU"),
            ]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_invalid_parameter_rejected_on_add(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "select")
            dialog._param_edit.setText("rlnDefocusV")  # select needs two tokens
            dialog._btn_add.click()

            assert dialog._stack_model.rowCount() == 0
            assert "Cannot add --select" in dialog._status.toPlainText()
            assert not dialog._btn_apply.isEnabled()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_apply_runs_engine_and_refreshes_preview(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "select")
            dialog._param_edit.setText("rlnDefocusV 5,20")
            dialog._btn_add.click()
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusU")
            dialog._btn_add.click()

            dialog._btn_apply.click()

            assert len(dialog._data) == 2
            assert dialog._data["rlnDefocusU"].tolist() == [1.0, 2.0]
            assert dialog._data["rlnDefocusV"].tolist() == [20.0, 5.0]
            model = dialog._table.model()
            assert model.rowCount() == 2
            assert model.columnCount() == 3
            assert "2 operation(s)" in dialog._status.toPlainText()
            assert "modified" in dialog.windowTitle()
            assert dialog._optics_table.model().rowCount() == 1
            assert dialog._btn_reset.isEnabled()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_append_option_applied_in_order(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusU")
            dialog._btn_add.click()
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusV")
            dialog._btn_add.click()

            dialog._btn_apply.click()

            # The last sort is primary (stable sort keeps the first's order).
            assert dialog._data["rlnDefocusV"].tolist() == [5.0, 10.0, 15.0, 20.0]
            assert dialog._data["rlnDefocusU"].tolist() == [2.0, 2.0, 1.0, 1.0]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_move_and_remove_reorder_stack(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusV")
            dialog._btn_add.click()
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusU")
            dialog._btn_add.click()

            dialog._stack_view.setCurrentIndex(dialog._stack_model.index(1, 0))
            dialog._btn_up.click()
            assert dialog._stack_model.operations() == [
                ("sortby", "rlnDefocusU"),
                ("sortby", "rlnDefocusV"),
            ]

            dialog._stack_view.setCurrentIndex(dialog._stack_model.index(0, 0))
            dialog._btn_remove.click()
            assert dialog._stack_model.operations() == [("sortby", "rlnDefocusV")]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_stack_view_configured_for_drag_reorder(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            view = dialog._stack_view
            assert view.dragDropMode() == QAbstractItemView.DragDropMode.InternalMove
            assert view.defaultDropAction() == Qt.DropAction.MoveAction
            assert not view.dragDropOverwriteMode()

            model = dialog._stack_model
            assert model.supportedDropActions() == Qt.DropAction.MoveAction
            for name, text in (
                ("keepParm", "rlnAngleRot"),
                ("select", "rlnDefocusV"),
                ("sortby", "rlnDefocusU"),
                ("renameParm", "rlnPixelSize"),
            ):
                model.add(name, text)
            for row in range(model.rowCount()):
                flags = model.flags(model.index(row, 0))
                assert flags & Qt.ItemFlag.ItemIsDragEnabled
                assert flags & Qt.ItemFlag.ItemIsDropEnabled
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_stack_model_move_rows_reorders_like_drag(self, qapp):
        """moveRows uses Qt's destination-before-row convention."""
        dialog = self._open_ops(qapp)
        try:
            model = dialog._stack_model
            for name, text in (
                ("keepParm", "rlnAngleRot"),
                ("select", "rlnDefocusV"),
                ("sortby", "rlnDefocusU"),
                ("renameParm", "rlnPixelSize"),
            ):
                model.add(name, text)

            def names():
                return [n for n, _ in model.operations()]

            def move(source_row, destination_child):
                assert model.moveRows(
                    QModelIndex(), source_row, 1, QModelIndex(), destination_child
                )
                return names()

            # First row dragged below the third (drop before original row 3).
            assert move(0, 3) == [
                "select",
                "sortby",
                "keepParm",
                "renameParm",
            ]
            # Last row dragged to the top (drop before row 0).
            assert move(3, 0) == [
                "renameParm",
                "select",
                "sortby",
                "keepParm",
            ]
            # Second row dragged below the last (drop at the end).
            assert move(1, 4) == [
                "renameParm",
                "sortby",
                "keepParm",
                "select",
            ]
            # Third row dragged above the second (drop before row 1).
            assert move(2, 1) == [
                "renameParm",
                "keepParm",
                "sortby",
                "select",
            ]
            # Dropping on itself or directly below it is a no-op.
            assert not model.moveRows(QModelIndex(), 0, 1, QModelIndex(), 0)
            assert not model.moveRows(QModelIndex(), 0, 1, QModelIndex(), 1)
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_ctrl_up_down_shortcuts_move_selection(self, qapp):
        from PySide6.QtGui import QKeySequence, QShortcut

        dialog = self._open_ops(qapp)
        try:
            model = dialog._stack_model
            model.add("sortby", "rlnDefocusV")
            model.add("sortby", "rlnDefocusU")
            model.add("select", "rlnMicrographName")

            shortcuts = dialog._stack_view.findChildren(QShortcut)
            up_sc = next(sc for sc in shortcuts if sc.key() == QKeySequence("Ctrl+Up"))
            down_sc = next(
                sc for sc in shortcuts if sc.key() == QKeySequence("Ctrl+Down")
            )

            dialog.show()
            dialog._stack_view.setCurrentIndex(model.index(2, 0))
            self._pump(qapp)

            # Drive the shortcut signal directly (like the Ctrl+Q test): a
            # synthesized key press never reliably reaches a window-scoped
            # shortcut when the window is not the active window.
            up_sc.activated.emit()
            self._pump(qapp)
            assert model.operations() == [
                ("sortby", "rlnDefocusV"),
                ("select", "rlnMicrographName"),
                ("sortby", "rlnDefocusU"),
            ]
            assert dialog._stack_view.currentIndex().row() == 1

            down_sc.activated.emit()
            self._pump(qapp)
            assert model.operations() == [
                ("sortby", "rlnDefocusV"),
                ("sortby", "rlnDefocusU"),
                ("select", "rlnMicrographName"),
            ]
            assert dialog._stack_view.currentIndex().row() == 2
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_command_text_maps_stack_to_cli(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            for name, text in (
                ("select", "rlnDefocusV 5,20"),
                ("sortby", "rlnDefocusU"),
                ("renameParm", "rlnPixelSize rlnPixelSizeNew"),
            ):
                self._select_op(dialog, name)
                dialog._param_edit.setText(text)
                dialog._btn_add.click()

            command = dialog._command_text()
            source = str(Path("dummy.star").resolve())
            fallback = str(Path("dummy.processed.star").resolve())
            expected = (
                "helicon images2star "
                f"{source} {fallback} "
                "--select rlnDefocusV 5,20 --sortby rlnDefocusU "
                "--renameParm rlnPixelSize rlnPixelSizeNew"
            )
            assert command == expected
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_command_text_quotes_paths_and_values(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "select")
            dialog._param_edit.setText("rlnMicrographName 'd ir/file 1.mrc'")
            dialog._btn_add.click()

            command = dialog._command_text()
            source = str(Path("dummy.star").resolve())
            fallback = str(Path("dummy.processed.star").resolve())
            assert source in command
            assert fallback in command
            assert command.endswith("--select rlnMicrographName 'd ir/file 1.mrc'")
            # Round-trips through the shell.
            import shlex

            tokens = shlex.split(command)
            assert tokens[0] == "helicon"
            assert tokens[1] == "images2star"
            assert tokens[2] == source
            assert tokens[3] == fallback
            assert tokens[4:] == [
                "--select",
                "rlnMicrographName",
                "d ir/file 1.mrc",
            ]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_command_uses_last_saved_output_path(self, qapp):
        captured = {}

        def saver(data, path):
            captured["data"] = data
            captured["path"] = path

        dialog = self._open_ops(qapp, saver=saver)
        try:
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusU")
            dialog._btn_add.click()

            dialog._save_to("/out/target.star")
            dialog._workers[-1].wait()
            self._pump(qapp)

            command = dialog._command_text()
            assert "/out/target.star" in command
            assert ".processed.star" not in command
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_save_default_path_differs_from_input(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            source = str(Path("dummy.star").resolve())
            assert dialog._default_output_path() == str(
                Path("dummy.processed.star").resolve()
            )
            assert dialog._default_output_path() != source
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_save_default_path_remembers_last_output(self, qapp):
        dialog = self._open_ops(qapp, saver=lambda data, path: None)
        try:
            dialog._save_to("/out/target.star")
            dialog._workers[-1].wait()
            self._pump(qapp)

            assert dialog._default_output_path() == "/out/target.star"
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_choose_output_suggests_distinct_default(self, qapp):
        from unittest.mock import patch

        dialog = self._open_ops(qapp)
        try:
            with patch(
                "PySide6.QtWidgets.QFileDialog.getSaveFileName",
                return_value=("", ""),
            ) as m:
                dialog._choose_output()

            suggested = m.call_args.args[2]
            assert suggested == str(Path("dummy.processed.star").resolve())
            assert suggested != str(Path("dummy.star").resolve())
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_command_dialog_has_editable_selectable_command(self, qapp):
        from PySide6.QtGui import QFontDatabase
        from PySide6.QtWidgets import QPlainTextEdit

        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusU")
            dialog._btn_add.click()

            cmd_dlg = dialog._build_command_dialog()
            try:
                edits = cmd_dlg.findChildren(QPlainTextEdit)
                assert len(edits) == 1
                edit = edits[0]
                text = edit.toPlainText()
                assert text.startswith("helicon images2star ")
                assert "--sortby rlnDefocusU" in text
                assert edit.isReadOnly() is False
                assert edit.lineWrapMode() == QPlainTextEdit.LineWrapMode.WidgetWidth
                from PySide6.QtGui import QTextOption

                assert (
                    edit.document().defaultTextOption().wrapMode()
                    == QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere
                )
                font = edit.font()
                assert (
                    font.family()
                    == QFontDatabase.systemFont(
                        QFontDatabase.SystemFont.FixedFont
                    ).family()
                )
                assert cmd_dlg.windowTitle() == "Equivalent images2star command"
            finally:
                cmd_dlg.close()
                cmd_dlg.deleteLater()
                self._pump(qapp)
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_command_copy_puts_text_on_clipboard(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusU")
            dialog._btn_add.click()

            command = dialog._command_text()
            dialog._copy_command(command)
            assert qapp.clipboard().text() == command
            assert "Command copied to clipboard" in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_reset_restores_source_dataset(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "select")
            dialog._param_edit.setText("rlnDefocusV 5,20")
            dialog._btn_add.click()
            dialog._btn_apply.click()
            assert len(dialog._data) == 2

            dialog._table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
            assert dialog._table.horizontalHeader().sortIndicatorSection() == 1

            dialog._btn_reset.click()

            assert len(dialog._data) == 4
            assert dialog._stack_model.rowCount() == 0
            assert "modified" not in dialog.windowTitle()
            assert dialog._table.model().rowCount() == 4
            assert not dialog._btn_reset.isEnabled()
            # Reset also discards the preview sort: the indicator is cleared
            # and the table returns to the natural (unsorted) row order.
            assert dialog._table.horizontalHeader().sortIndicatorSection() == -1
            model = dialog._table.model()
            assert [model.data(model.index(r, 0)) for r in range(4)] == [
                "/d/A.mrc",
                "/d/A.mrc",
                "/d/B.mrc",
                "/d/B.mrc",
            ]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_pure_sort_is_resettable_without_modified_title(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            assert not dialog._btn_reset.isEnabled()
            assert "modified" not in dialog.windowTitle()

            dialog._table.sortByColumn(1, Qt.SortOrder.DescendingOrder)

            # A pure sort enables Reset Data but never marks the data dirty.
            assert dialog._btn_reset.isEnabled()
            assert "modified" not in dialog.windowTitle()
            model = dialog._table.model()
            assert [model.data(model.index(r, 0)) for r in range(4)] == [
                "/d/B.mrc",
                "/d/A.mrc",
                "/d/B.mrc",
                "/d/A.mrc",
            ]

            dialog._btn_reset.click()

            assert not dialog._btn_reset.isEnabled()
            assert "modified" not in dialog.windowTitle()
            assert dialog._table.horizontalHeader().sortIndicatorSection() == -1
            model = dialog._table.model()
            assert [model.data(model.index(r, 0)) for r in range(4)] == [
                "/d/A.mrc",
                "/d/A.mrc",
                "/d/B.mrc",
                "/d/B.mrc",
            ]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_optics_sort_is_resettable_with_same_button(self, qapp):
        optics = pd.DataFrame(
            {
                "rlnOpticsGroup": [2, 1, 3],
                "rlnPixelSize": [1.2, 0.9, 1.5],
            }
        )
        data = pd.DataFrame({"rlnImageName": ["a.mrc", "b.mrc"]})
        data.attrs["optics"] = optics
        dialog = self._open("dummy.star", loader=lambda p: data)
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            dialog._optics_table.sortByColumn(1, Qt.SortOrder.DescendingOrder)
            assert dialog._btn_reset.isEnabled()
            assert "modified" not in dialog.windowTitle()

            dialog._btn_reset.click()

            assert not dialog._btn_reset.isEnabled()
            assert dialog._optics_table.horizontalHeader().sortIndicatorSection() == -1
            model = dialog._optics_table.model()
            assert [model.data(model.index(r, 0)) for r in range(3)] == [
                "2",
                "1",
                "3",
            ]
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_save_writes_transformed_data(self, tmp_path, qapp):
        captured = {}

        def saver(data, path):
            captured["data"] = data
            captured["path"] = path

        dialog = self._open_ops(qapp, saver=saver)
        try:
            self._select_op(dialog, "select")
            dialog._param_edit.setText("rlnDefocusV 5,20")
            dialog._btn_add.click()
            dialog._btn_apply.click()

            out = str(tmp_path / "filtered.star")
            dialog._save_to(out)
            dialog._workers[-1].wait()
            self._pump(qapp)

            assert len(captured["data"]) == 2
            assert captured["path"] == out
            assert "Saved" in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_transform_failure_reported_in_status(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnNoSuchColumn")
            dialog._btn_add.click()

            dialog._btn_apply.click()

            assert len(dialog._data) == 4  # unchanged
            assert "Transform failed" in dialog._status.toPlainText()
            assert "rlnNoSuchColumn" in dialog._status.toPlainText()
            assert not dialog._btn_reset.isEnabled()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_status_error_message_renders_red(self, qapp):
        from PySide6.QtCore import QSettings

        settings = QSettings("helicon", "display")
        settings.setValue("theme", "Light")
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnNoSuchColumn")
            dialog._btn_add.click()

            dialog._btn_apply.click()

            assert "Transform failed" in dialog._status.toPlainText()
            assert "color: #b3261e" in dialog._status.styleSheet()

            # A subsequent info message clears the error styling.
            dialog._set_status("back to normal")
            assert "color:" not in dialog._status.styleSheet()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)
            settings.remove("theme")

    def test_status_error_uses_dark_theme_red(self, qapp):
        from PySide6.QtCore import QSettings

        settings = QSettings("helicon", "display")
        settings.setValue("theme", "Dark")
        dialog = self._open_ops(qapp)
        try:
            dialog._set_status("boom", error=True)
            assert "color: #ff6b6b" in dialog._status.styleSheet()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)
            settings.remove("theme")

    def test_error_style_reapplied_after_theme_switch(self, qapp):
        from PySide6.QtCore import QSettings

        settings = QSettings("helicon", "display")
        settings.setValue("theme", "Light")
        dialog = self._open_ops(qapp)
        try:
            dialog._set_status("boom", error=True)
            assert "color: #b3261e" in dialog._status.styleSheet()

            settings.setValue("theme", "Dark")
            dialog._apply_display_theme()
            assert "color: #ff6b6b" in dialog._status.styleSheet()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)
            settings.remove("theme")

    def test_add_parm_missing_column_shown_in_status(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            self._select_op(dialog, "addParm")
            dialog._param_edit.setText("rlnNoSuchColumn 1.5")
            dialog._btn_add.click()

            dialog._btn_apply.click()

            assert len(dialog._data) == 4  # unchanged
            assert "Transform failed" in dialog._status.toPlainText()
            assert "rlnNoSuchColumn" in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_edit_particle_cell_updates_data_and_marks_dirty(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            model = dialog._table.model()
            index = model.index(0, 1)  # rlnDefocusU, row 0
            assert model.setData(index, "2.5")

            assert dialog._data["rlnDefocusU"].tolist()[0] == 2.5
            assert model.data(index) == "2.5"
            assert "modified" in dialog.windowTitle()
            assert dialog._btn_reset.isEnabled()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_edit_optics_cell_updates_attrs(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            model = dialog._optics_table.model()
            index = model.index(0, 1)  # rlnPixelSize
            assert model.setData(index, "1.35")

            assert dialog._data.attrs["optics"]["rlnPixelSize"].tolist() == [1.35]
            assert "modified" in dialog.windowTitle()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_edit_rejects_invalid_value(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            model = dialog._table.model()
            index = model.index(0, 1)  # rlnDefocusU is float
            assert not model.setData(index, "not-a-number")

            assert dialog._data["rlnDefocusU"].tolist()[0] == 2.0
            assert "modified" not in dialog.windowTitle()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_manual_edit_survives_apply(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            model = dialog._table.model()
            assert model.setData(model.index(0, 1), "9.0")

            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusU")
            dialog._btn_add.click()
            dialog._btn_apply.click()

            # The edited 9.0 survives the transform (apply runs on current data).
            assert dialog._data["rlnDefocusU"].tolist()[3] == 9.0
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_preview_tables_allow_editing(self, qapp):
        dialog = self._open_ops(qapp)
        try:
            triggers = (
                QAbstractItemView.EditTrigger.DoubleClicked
                | QAbstractItemView.EditTrigger.EditKeyPressed
            )
            assert dialog._table.editTriggers() == triggers
            assert dialog._optics_table.editTriggers() == triggers
            assert dialog._table.model().flags(dialog._table.model().index(0, 0)) & (
                Qt.ItemFlag.ItemIsEditable
            )
            assert dialog._optics_table.model().flags(
                dialog._optics_table.model().index(0, 0)
            ) & (Qt.ItemFlag.ItemIsEditable)
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_ops_end_to_end_with_real_files(self, tmp_path, qapp):
        """Default loader + stack apply + real save produce a valid star."""
        import starfile

        data = self._df_for_ops()
        src = tmp_path / "images.star"
        helicon.dataframe2file(data, str(src))

        dialog = self._open(str(src))
        try:
            dialog._load_worker.wait()
            self._pump(qapp)
            assert len(dialog._data) == 4
            assert dialog._optics_table.model().rowCount() == 1

            self._select_op(dialog, "select")
            dialog._param_edit.setText("rlnDefocusV 5,20")
            dialog._btn_add.click()
            self._select_op(dialog, "sortby")
            dialog._param_edit.setText("rlnDefocusU")
            dialog._btn_add.click()
            dialog._btn_apply.click()

            out = tmp_path / "filtered.star"
            dialog._save_to(str(out))
            dialog._workers[-1].wait()
            self._pump(qapp)

            written = starfile.read(str(out))
            if isinstance(written, dict):
                block = next((v for k, v in written.items() if k != "optics"), None)
                assert block is not None, list(written)
                particles = block
            else:
                particles = written
            assert len(particles) == 2
            assert particles["rlnDefocusU"].tolist() == [1.0, 2.0]
            assert particles["rlnDefocusV"].tolist() == [20.0, 5.0]
            assert "Saved" in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_ctrl_w_shortcut_closes_dialog(self, qapp):
        from PySide6.QtGui import QKeySequence, QShortcut

        dialog = self._open_ops(qapp)
        try:
            shortcuts = dialog.findChildren(QShortcut)
            close_sc = next(
                sc for sc in shortcuts if sc.key() == QKeySequence("Ctrl+W")
            )
            assert close_sc.isEnabled()

            dialog.show()
            self._pump(qapp)
            assert dialog.isVisible()

            close_sc.activated.emit()
            self._pump(qapp)
            assert not dialog.isVisible()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_ctrl_q_shortcut_quits_app(self, qapp):
        from PySide6.QtGui import QKeySequence, QShortcut

        dialog = self._open_ops(qapp)
        try:
            shortcuts = dialog.findChildren(QShortcut)
            quit_sc = next(
                sc
                for sc in shortcuts
                if sc.key() == QKeySequence(QKeySequence.StandardKey.Quit)
            )
            assert quit_sc.isEnabled()

            with patch("PySide6.QtWidgets.QApplication.quit") as mock_quit:
                dialog.show()
                self._pump(qapp)
                # StandardKey.Quit is Cmd+Q on macOS and Ctrl+Q elsewhere, so
                # drive the shortcut signal directly rather than a key press.
                quit_sc.activated.emit()
                self._pump(qapp)

            mock_quit.assert_called_once()
            assert not dialog.isVisible()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_header_double_click_does_not_copy_column_label(self, qapp):
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest

        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            qapp.clipboard().setText("sentinel")
            header = dialog._table.horizontalHeader()
            x = header.sectionViewportPosition(0) + header.sectionSize(0) // 2
            QTest.mouseDClick(
                header.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                QPoint(x, header.height() // 2),
            )

            # Double-clicking a header only sorts; the label is copied via
            # the right-click context menu instead.
            assert qapp.clipboard().text() == "sentinel"
            assert "rlnImageName" not in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_header_menu_actions_copy_label_or_all_labels(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            menu = dialog._table._header_menu(1)
            actions = menu.actions()
            assert len(actions) == 2

            actions[0].trigger()  # Copy column label
            assert qapp.clipboard().text() == "rlnAngleRot"
            assert "rlnAngleRot" in dialog._status.toPlainText()

            actions[1].trigger()  # Copy all column labels
            assert qapp.clipboard().text().splitlines() == [
                "rlnImageName",
                "rlnAngleRot",
            ]
            assert "2 column labels" in dialog._status.toPlainText()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_header_label_copy_works_on_optics_table(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            dialog._optics_table._copy_header_label(1)
            assert qapp.clipboard().text() == "rlnPixelSize"
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_tolerant_loader_keeps_table_and_records_warnings(self, tmp_path):
        from helicon.lib.exceptions import HeliconIOError
        from helicon.lib.gui.images2star_widget import _images2dataframe_tolerant

        present = tmp_path / "present.mrc"
        present.write_bytes(b"x")
        missing = tmp_path / "missing.mrc"

        data = pd.DataFrame(
            {
                "rlnImageName": [f"1@{missing}", f"2@{present}"],
                "rlnAngleRot": [1.0, 2.0],
            }
        )
        data.attrs["convention"] = "relion"

        with patch(
            "helicon.images2dataframe",
            side_effect=[
                HeliconIOError(f"cannot find image {missing} in file x.star"),
                data,
            ],
        ):
            result = _images2dataframe_tolerant("x.star")

        # The table from the star file is still returned...
        assert len(result) == 2
        assert result["rlnAngleRot"].tolist() == [1.0, 2.0]
        # ...with the resolution failure recorded beside it.
        assert len(result.attrs["load_warnings"]) == 2
        assert f"cannot find image {missing}" in result.attrs["load_warnings"][0]
        assert str(missing) in result.attrs["missing_files"]
        assert str(present) not in result.attrs["missing_files"]

    def test_tolerant_loader_reraises_fatal_errors(self):
        from helicon.lib.exceptions import HeliconIOError
        from helicon.lib.gui.images2star_widget import _images2dataframe_tolerant

        with patch(
            "helicon.images2dataframe",
            side_effect=[
                HeliconIOError("cannot find image missing.mrc in file x.star"),
                HeliconIOError("cannot parse x.star"),
            ],
        ):
            with pytest.raises(HeliconIOError, match="cannot find image missing.mrc"):
                _images2dataframe_tolerant("x.star")

    def test_tolerant_loader_also_reports_missing_micrographs(self, tmp_path):
        from helicon.lib.gui.images2star_widget import _images2dataframe_tolerant

        present = tmp_path / "present.mrc"
        present.write_bytes(b"x")
        missing_mg = tmp_path / "missing_frames.mrcs"

        data = pd.DataFrame(
            {
                "rlnImageName": [f"1@{present}", f"2@{present}"],
                "rlnMicrographName": [str(missing_mg), str(missing_mg)],
            }
        )
        data.attrs["convention"] = "relion"

        # Strict load succeeds (particles resolve) but the micrograph file is
        # silently missing; it still must be reported.
        with patch("helicon.images2dataframe", return_value=data):
            result = _images2dataframe_tolerant("x.star")

        assert result.attrs["missing_files"] == [str(missing_mg)]
        assert len(result.attrs["load_warnings"]) == 1
        assert "micrograph" in result.attrs["load_warnings"][0].lower()

    def test_missing_file_cells_are_colored_red(self, qapp, tmp_path):
        from helicon.lib.gui.images2star_widget import _DataFramePreviewModel

        present = tmp_path / "present.mrc"
        present.write_bytes(b"x")
        missing = tmp_path / "missing.mrc"

        data = pd.DataFrame(
            {
                "rlnImageName": [
                    f"1@{present}",
                    f"1@{missing}",
                ],
                "rlnAngleRot": [1.0, 2.0],
            }
        )
        model = _DataFramePreviewModel(
            data, missing_files=[str(missing)], parent=qapp.activeWindow()
        )
        try:
            ok_index = model.index(0, 0)
            bad_index = model.index(1, 0)
            num_index = model.index(1, 1)
            # Only the cell referencing the missing file is flagged.
            assert model.data(ok_index, Qt.ItemDataRole.ForegroundRole) is None
            assert model.data(num_index, Qt.ItemDataRole.ForegroundRole) is None
            color = model.data(bad_index, Qt.ItemDataRole.ForegroundRole)
            assert color is not None
            assert color.isValid()
        finally:
            model.deleteLater()
            self._pump(qapp)

    def test_status_row_caps_at_five_lines_and_scrolls(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            status = dialog._status
            line_h = status.fontMetrics().lineSpacing()

            dialog._set_status("a single line")
            self._pump(qapp)
            assert status.height() < 2 * line_h

            dialog._set_status("\n".join(f"row {i}" for i in range(12)))
            self._pump(qapp)
            assert status.height() == 5 * line_h + 4
            assert status.verticalScrollBar().maximum() > 0
            assert status.isReadOnly()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_transformations_block_in_draggable_splitter(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            dialog.show()
            self._pump(qapp)

            # A drag bar separates the previews from the transformations
            # block, and the block defaults to its compact height (about
            # half of the tall block the vertical button column needed).
            assert dialog._main_splitter.count() == 2
            assert dialog._main_splitter.handle(1) is not None
            ops_default = dialog._main_splitter.sizes()[1]
            assert ops_default <= 2 * dialog._ops_group.sizeHint().height()

            # Dragging the handle open must not be undone by a re-show
            # (small per-pixel handle redistribution is fine).
            h = dialog._main_splitter.height()
            dialog._main_splitter.setSizes([int(h * 0.4), int(h * 0.6)])
            dialog.show()
            self._pump(qapp)
            preview, ops = dialog._main_splitter.sizes()
            assert abs(preview - int(h * 0.4)) <= 3
            assert abs(ops - int(h * 0.6)) <= 3
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_default_ops_block_fits_two_stack_rows(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            dialog.show()
            self._pump(qapp)

            # The transformations block defaults tall enough for the option
            # list to show two full argument rows (frame included) instead of
            # one row with the second clipped by the list frame.
            stack = dialog._stack_view
            row_h = stack.fontMetrics().lineSpacing()
            assert stack.minimumHeight() == 2 * row_h + 2 * stack.frameWidth()
            assert stack.viewport().height() >= 2 * row_h
            assert dialog._ops_group.height() == dialog._ops_group.sizeHint().height()
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_optics_pane_defaults_to_one_and_half_rows(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            dialog._load_worker.wait()
            dialog.show()
            self._pump(qapp)

            # Optics gets a compact strip showing ~1.5 data rows (label +
            # column headers + 1.5 rows); the particles table gets the rest.
            expected = dialog._optics_default_pane_height()
            preview, optics = dialog._splitter.sizes()
            assert abs(optics - expected) <= 2
            # The two panes plus the splitter handle fill the splitter, so the
            # particles pane keeps everything except the optics strip and the
            # few pixels taken by the handle.
            assert preview >= dialog._splitter.height() - expected - 5
            row_h = dialog._optics_table.verticalHeader().defaultSectionSize()
            visible = (
                dialog._optics_table.height()
                - dialog._optics_table.horizontalHeader().height()
            )
            assert visible >= 1.5 * row_h - 1
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_preview_default_applies_when_load_finishes_before_show(self, qapp):
        dialog = self._open("dummy.star", loader=lambda p: self._df_with_optics())
        try:
            # Let the load complete before the window is ever shown so the
            # optics default must survive into the first show.
            dialog._load_worker.wait()
            self._pump(qapp)
            dialog.show()
            self._pump(qapp)
            expected = dialog._optics_default_pane_height()
            assert abs(dialog._splitter.sizes()[1] - expected) <= 2
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)

    def test_load_warnings_show_as_error_beside_the_table(self, qapp):
        data = self._df_with_optics()
        data.attrs["load_warnings"] = ["HeliconIOError: cannot find image missing.mrc"]
        dialog = self._open("dummy.star", loader=lambda p: data)
        try:
            dialog._load_worker.wait()
            self._pump(qapp)

            # The table is still populated despite the warnings...
            assert dialog._table.model().rowCount() == 2
            # ...and the summary plus the error message are both shown.
            status = dialog._status.toPlainText()
            assert "2 rows" in status
            assert "cannot find image missing.mrc" in status
            assert dialog._status_error is True
        finally:
            dialog.close()
            dialog.deleteLater()
            self._pump(qapp)
