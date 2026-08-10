import argparse
import os
import sys
from pathlib import Path

import numpy as np
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
            patch.object(display, "_macos_menu_debug_report") as mock_debug,
            patch.object(display, "_force_macos_menu_realization") as mock_force,
        ):
            with patch.dict(sys.modules, {"napari": None}):
                display.main(args)
        mock_identity.assert_not_called()
        mock_debug.assert_not_called()
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


class TestGeometryPersistence(object):
    @patch.object(display, "_position_default")
    @patch.object(display, "_get_qsettings")
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

    @patch.object(display, "_position_default")
    @patch.object(display, "_get_qsettings")
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

    @patch.object(display, "_position_default")
    @patch.object(display, "_get_qsettings")
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

    @patch.object(display, "_get_qsettings")
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
        from helicon.lib.file_browser import FolderBrowserWidget, _saved_theme

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
        from helicon.lib.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        assert [action.text() for action in widget._menu_bar.actions()] == [
            "File",
            "View",
        ]
        assert widget._file_menu.title() == "File"
        assert widget._view_menu.title() == "View"
        assert widget._open_folder_action.text() == "Open Folder…"
        assert widget._theme_menu.title() == "Theme"
        assert set(widget._theme_actions) == {"Dark", "Light", "System"}

    def test_theme_menu_action_persists_selection(self, tmp_path, qapp):
        from PySide6.QtCore import QSettings
        from helicon.lib.file_browser import FolderBrowserWidget, _saved_theme

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
        from helicon.lib.file_browser import FolderBrowserWidget

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
        from helicon.lib.file_browser import FolderBrowserWidget

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

    def test_invalid_browser_theme_falls_back_to_system(self, qapp):
        from PySide6.QtCore import QSettings
        from helicon.lib.file_browser import _saved_theme

        settings = QSettings("helicon", "display")
        settings.setValue("theme", "not-a-theme")
        assert _saved_theme() == "System"
        settings.remove("theme")

    def test_format_size_bytes(self):
        from helicon.lib.file_browser import _format_size

        assert _format_size(512) == "512 B"

    def test_format_size_kilobytes(self):
        from helicon.lib.file_browser import _format_size

        assert _format_size(2048) == "2.0 KB"

    def test_format_size_megabytes(self):
        from helicon.lib.file_browser import _format_size

        assert _format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_format_size_gigabytes(self):
        from helicon.lib.file_browser import _format_size

        assert _format_size(2 * 1024 * 1024 * 1024) == "2.00 GB"

    def test_file_browser_model_has_six_columns(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel, NUM_COLUMNS

        (tmp_path / "test.txt").write_text("hello")
        model = FileBrowserModel(str(tmp_path))
        assert model.columnCount() == NUM_COLUMNS

    def test_file_browser_model_headers(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel

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
        from helicon.lib.file_browser import FileBrowserModel, COL_NAME

        (tmp_path / "aaa.txt").write_text("a")
        (tmp_path / "bbb.txt").write_text("b")
        model = FileBrowserModel(str(tmp_path))
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "aaa.txt" in names
        assert "bbb.txt" in names

    def test_file_browser_model_lists_dirs_first(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel, COL_NAME, COL_TYPE

        (tmp_path / "adir").mkdir()
        (tmp_path / "file.txt").write_text("x")
        model = FileBrowserModel(str(tmp_path))
        first_type = model.item(0, COL_TYPE).text()
        last_type = model.item(model.rowCount() - 1, COL_TYPE).text()
        assert first_type == "Folder"
        assert last_type != "Folder"

    def test_file_browser_model_shows_size(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel, COL_SIZE

        (tmp_path / "data.bin").write_bytes(b"x" * 2048)
        model = FileBrowserModel(str(tmp_path))
        sizes = [model.item(r, COL_SIZE).text() for r in range(model.rowCount())]
        assert "2.0 KB" in sizes

    def test_file_browser_model_shows_date(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel, COL_MODIFIED

        (tmp_path / "recent.txt").write_text("new")
        model = FileBrowserModel(str(tmp_path))
        dates = [model.item(r, COL_MODIFIED).text() for r in range(model.rowCount())]
        assert any("202" in d for d in dates)

    def test_file_browser_model_sort_by_size(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel, COL_SIZE

        (tmp_path / "small.txt").write_text("s")
        (tmp_path / "big.txt").write_text("x" * 10000)
        (tmp_path / "adir").mkdir()
        model = FileBrowserModel(str(tmp_path))
        model.sort(COL_SIZE, Qt.SortOrder.DescendingOrder)
        sizes = [model.item(r, COL_SIZE).text() for r in range(model.rowCount())]
        assert sizes[0] == "9.8 KB"
        assert sizes[-1] == ""

    def test_file_browser_model_sort_dirs_first(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel, COL_TYPE

        (tmp_path / "adir").mkdir()
        (tmp_path / "zfile.txt").write_text("z")
        model = FileBrowserModel(str(tmp_path))
        model.sort(0, Qt.SortOrder.AscendingOrder)
        assert model.item(0, COL_TYPE).text() == "Folder"

    def test_file_browser_model_set_root_path(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel, COL_NAME

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "inner.txt").write_text("i")
        model = FileBrowserModel(str(tmp_path))
        model.set_root_path(str(sub))
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "inner.txt" in names

    def test_file_browser_model_file_path(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel

        (tmp_path / "hello.txt").write_text("hi")
        model = FileBrowserModel(str(tmp_path))
        path = model.file_path(model.index(0, 0))
        assert path is not None
        assert "hello.txt" in path

    def test_file_browser_model_is_dir(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel

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
        from helicon.lib.file_browser import FolderBrowserWidget

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
        from helicon.lib.file_browser import FolderBrowserWidget

        sub = tmp_path / "sub"
        sub.mkdir()
        widget = FolderBrowserWidget(start_dir=str(sub))
        widget._go_up()
        assert widget._model._root_path == str(tmp_path)

    def test_folder_browser_go_back(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget

        sub = tmp_path / "sub"
        sub.mkdir()
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        widget._navigate_to(str(sub))
        widget._go_back()
        assert widget._model._root_path == str(tmp_path)

    def test_folder_browser_shift_double_click_emits_new_window_signal(
        self, tmp_path, qapp
    ):
        from helicon.lib.file_browser import FolderBrowserWidget

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
        from helicon.lib.file_browser import FolderBrowserWidget

        assert FolderBrowserWidget._is_image_stack(None, "/d/particles.mrcs")
        assert FolderBrowserWidget._is_image_stack(None, "/d/data.star")
        # Metadata star files are not image stacks.
        assert not FolderBrowserWidget._is_image_stack(None, "/d/run1_optimiser.star")
        # Volumes keep the "Image Slice" label, so are not image stacks.
        assert not FolderBrowserWidget._is_image_stack(None, "/d/map.mrc")
        assert not FolderBrowserWidget._is_image_stack(None, "/d/map.map")

    def test_slice_button_label_for_image_stack(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget
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

    def test_pdf_button_label(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget
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
        from helicon.lib.file_browser import FolderBrowserWidget
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
        from helicon.lib.file_browser import FolderBrowserWidget
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

        from helicon.lib import file_browser

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
        from helicon.lib import file_browser
        from helicon.lib.file_browser import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        with patch.object(file_browser, "_find_chimerax", lambda: None):
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
        from helicon.lib import file_browser
        from helicon.lib.file_browser import FolderBrowserWidget
        from PySide6.QtCore import QItemSelectionModel

        with patch.object(file_browser, "_find_chimerax", lambda: "/x/ChimeraX"):
            (tmp_path / "volume.mrc").write_bytes(b"\x00" * 1024)
            widget = FolderBrowserWidget(start_dir=str(tmp_path))
            idx = widget._model.index(0, 0)
            widget._tree.selectionModel().select(
                idx, QItemSelectionModel.Select | QItemSelectionModel.Clear
            )
            assert widget._btn_chimerax.isEnabled()
            assert "Open this file" in widget._btn_chimerax.toolTip()

    def test_stats_button_for_class3d_and_refine3d_data_star_only(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget
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

    def test_eps_label_and_mode(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget

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
        # EPS is a known type with no display modes (no button shown).
        assert widget._display_modes_for(str(tmp_path / "fig.eps")) == []

    def test_pdf_has_image_slice_mode(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget

        pdf_file = tmp_path / "figure.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\nplaceholder")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))

        assert widget._display_modes_for(str(pdf_file)) == ["slice"]

    def test_display_modes_class2d_model_star(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget

        job_dir = tmp_path / "Class2D" / "job001"
        job_dir.mkdir(parents=True)
        star_file = job_dir / "model.star"
        star_file.write_text("dummy")
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        assert widget._display_modes_for(str(star_file)) == [
            "text",
            "2dclasses",
        ]

    def test_display_modes_class3d_optimiser_star(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget

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
        from helicon.lib.file_browser import FolderBrowserWidget

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

    def test_file_browser_model_filter_wildcard(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel, COL_NAME

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
        from helicon.lib.file_browser import FileBrowserModel, COL_NAME

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
        from helicon.lib.file_browser import FileBrowserModel, COL_NAME

        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.mrc").write_bytes(b"b")
        model = FileBrowserModel(str(tmp_path))
        model.set_filter("")
        names = [model.item(r, COL_NAME).text() for r in range(model.rowCount())]
        assert "a.txt" in names
        assert "b.mrc" in names

    def test_file_browser_model_filter_dirs_always_shown(self, tmp_path):
        from helicon.lib.file_browser import FileBrowserModel, COL_NAME

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
        from helicon.lib.file_browser import (
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
        from helicon.lib.file_browser import (
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
        from helicon.lib.file_browser import FileBrowserModel

        (tmp_path / "a.mrc").write_bytes(b"\x00" * 1024)
        model = FileBrowserModel(str(tmp_path))
        first = model.current_epoch()
        model.set_filter("*")  # triggers a directory reload -> epoch bump
        assert model.current_epoch() == first + 1

    def test_populate_fills_columns_async(self, tmp_path, qapp):
        from PySide6.QtWidgets import QApplication

        from helicon.lib import file_browser
        from helicon.lib.file_browser import (
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

        from helicon.lib import file_browser
        from helicon.lib.file_browser import (
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


class TestOrthogonalViewer:

    def test_display_modes_mrc_with_nz_gt1_includes_orthogonal(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget

        mrc_path = tmp_path / "volume.mrc"
        mrc_path.write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(widget, "_volume_has_nz_gt1", return_value=True):
            modes = widget._display_modes_for(str(mrc_path))
        assert "orthogonal" in modes

    def test_display_modes_mrc_with_nz_eq1_no_orthogonal(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget

        mrc_path = tmp_path / "volume.mrc"
        mrc_path.write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(widget, "_volume_has_nz_gt1", return_value=False):
            modes = widget._display_modes_for(str(mrc_path))
        assert "orthogonal" not in modes

    def test_display_modes_map_with_nz_gt1_includes_orthogonal(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget

        map_path = tmp_path / "volume.map"
        map_path.write_bytes(b"\x00" * 1024)
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(widget, "_volume_has_nz_gt1", return_value=True):
            modes = widget._display_modes_for(str(map_path))
        assert "orthogonal" in modes

    def test_display_modes_non_mrc_no_orthogonal(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget

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
        from helicon.lib.gallery_widget import _SliceView

        view = _SliceView()
        assert view is not None
        assert view._zoom == 1.0
        assert view._brightness == 0.0
        assert view._contrast == 1.0
        assert view._gamma == 1.0
        assert view._log_transform is False
        view.deleteLater()

    def test_controlbar_instantiates(self, qapp):
        from helicon.lib.gallery_widget import _ControlBar, _BCGPanel

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
        from helicon.lib.gallery_widget import OrthogonalViewerWidget

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
        from helicon.lib.gallery_widget import OrthogonalViewerWidget

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
        from helicon.lib.gallery_widget import OrthogonalViewerWidget

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
        from helicon.lib.gallery_widget import OrthogonalViewerWidget

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
        from helicon.lib.gallery_widget import OrthogonalViewerWidget

        vol = np.zeros((8, 8, 8), dtype=np.float32)
        w = OrthogonalViewerWidget(vol, apix=1.0, name="test")
        w._on_slider_position(3, 4, 5)
        assert w._pos == [3, 4, 5]
        w.deleteLater()


class TestDisplayButtonSorting:

    def test_reorder_sorts_visible_buttons_alphabetically(self, tmp_path, qapp):
        from helicon.lib.file_browser import FolderBrowserWidget

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
        from helicon.lib.file_browser import FolderBrowserWidget

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
        from helicon.lib.file_browser import FolderBrowserWidget

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
