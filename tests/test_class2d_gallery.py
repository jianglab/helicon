"""The 2D Classes gallery of a RELION Class2D model.star: sort modes and label columns."""

import sys

import mrcfile
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from helicon.lib.gui.gallery_backends import Class2dGallery  # noqa: E402
from helicon.lib.gui.star_parsers import (  # noqa: E402
    _parse_class2d_model_classes,
    _parse_class2d_model_star,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def _fresh_gallery_settings():
    """Each test starts from the defaults: the run's settings folder is shared."""

    def clear():
        settings = QSettings("helicon", "display")
        for key in ("class2d_helical", "sort_column", "sort_reverse", "label_columns"):
            settings.remove(key)
        settings.sync()

    clear()
    yield
    clear()


# Class 1..5. Class 6 refers to a missing stack and must be dropped.
_HELICAL_ROWS = [
    # dist, helical type, probability
    (0.30, 2, 0.90),
    (0.10, 1, 0.60),
    (0.25, 1, 0.95),
    (0.05, 2, 0.99),
    (0.30, 1, 0.60),
]


def _write_model(tmp_path, helical: bool) -> str:
    job = tmp_path / "Class2D" / "job010"
    job.mkdir(parents=True)
    stack = np.stack(
        [np.full((8, 8), k, dtype=np.float32) for k in range(len(_HELICAL_ROWS))]
    )
    with mrcfile.new(str(job / "run_it025_classes.mrcs"), overwrite=True) as mrc:
        mrc.set_data(stack)
        mrc.voxel_size = 2.0

    cols = ["_rlnReferenceImage", "_rlnClassDistribution", "_rlnAccuracyRotations"]
    if helical:
        cols += ["_rlnHelicalType", "_rlnHelicalTypeProbability"]
    lines = [
        "data_model_general",
        "_rlnNrClasses 6",
        "",
        "data_model_classes",
        "loop_",
    ]
    lines += [f"{c} #{i + 1}" for i, c in enumerate(cols)]
    for k, (dist, htype, prob) in enumerate(_HELICAL_ROWS):
        row = [
            f"{k + 1:06d}@Class2D/job010/run_it025_classes.mrcs",
            f"{dist:.6f}",
            "1.25",
        ]
        if helical:
            row += [str(htype), f"{prob:.6f}"]
        lines.append("  ".join(row))
    missing = ["000006@Class2D/job010/missing.mrcs", "0.0", "1.0"]
    if helical:
        missing += ["1", "0.5"]
    lines.append("  ".join(missing))
    path = job / "run_it025_model.star"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _gallery(tmp_path, helical: bool) -> Class2dGallery:
    gallery = Class2dGallery(_write_model(tmp_path, helical))
    gallery._parse()
    assert gallery._parsed
    return gallery


def _class_order(gallery) -> list[int]:
    """Class numbers (1-based) in gallery order."""
    return [i + 1 for i in gallery._order]


class TestParseClassesTable:
    def test_keeps_every_column_of_the_found_classes(self, tmp_path):
        entries, table = _parse_class2d_model_classes(_write_model(tmp_path, True))
        assert len(entries) == 5
        assert [frame for _, frame in entries] == [0, 1, 2, 3, 4]
        assert set(table) == {
            "rlnReferenceImage",
            "rlnClassDistribution",
            "rlnAccuracyRotations",
            "rlnHelicalType",
            "rlnHelicalTypeProbability",
        }
        assert all(len(v) == 5 for v in table.values())
        assert table["rlnHelicalType"] == ["2", "1", "1", "2", "1"]

    def test_distributions_unchanged(self, tmp_path):
        entries, dists = _parse_class2d_model_star(_write_model(tmp_path, False))
        assert len(entries) == 5
        assert dists == pytest.approx([d for d, _, _ in _HELICAL_ROWS])


class TestSortModes:
    def test_helical_type_is_the_default_when_present(self, tmp_path):
        gallery = _gallery(tmp_path, helical=True)
        assert gallery._sort_columns == ["Helical Type", "Abundance", "Name", "None"]
        assert gallery._sort_column == "Helical Type"
        # type 1 first; within it most probable first, ties by abundance
        assert _class_order(gallery) == [3, 5, 2, 4, 1]

    def test_reverse_flips_the_types_only(self, tmp_path):
        gallery = _gallery(tmp_path, helical=True)
        gallery._sort_reverse = True
        gallery._apply_sort()
        assert _class_order(gallery) == [4, 1, 3, 5, 2]

    def test_text_helical_types(self, tmp_path):
        gallery = _gallery(tmp_path, helical=True)
        gallery._table["rlnHelicalType"] = ["b", "a", "a", "b", "a"]
        gallery._apply_sort()
        assert _class_order(gallery) == [3, 5, 2, 4, 1]

    def test_no_helical_type_option_without_the_column(self, tmp_path):
        gallery = _gallery(tmp_path, helical=False)
        assert gallery._sort_columns == ["Abundance", "Name", "None"]
        assert gallery._sort_column == "Abundance"
        assert gallery._sort_reverse is True
        assert [gallery._dists[i] for i in gallery._order] == sorted(
            gallery._dists, reverse=True
        )

    def test_no_sort_keeps_the_file_order(self, tmp_path):
        gallery = _gallery(tmp_path, helical=False)
        gallery._sort_column = "None"
        gallery._sort_reverse = False
        gallery._apply_sort()
        assert _class_order(gallery) == [1, 2, 3, 4, 5]
        gallery._sort_reverse = True
        gallery._apply_sort()
        assert _class_order(gallery) == [5, 4, 3, 2, 1]

    def test_saved_sort_is_restored_per_kind_of_file(self, tmp_path):
        helical = _gallery(tmp_path / "a", helical=True)
        helical._sort_column = "None"
        helical._sort_reverse = True
        helical._save_settings()

        # a plain file keeps its own default ...
        assert _gallery(tmp_path / "b", helical=False)._sort_column == "Abundance"
        # ... and a helical one gets back what was chosen
        again = _gallery(tmp_path / "c", helical=True)
        assert again._sort_column == "None"
        assert again._sort_reverse is True
        assert _class_order(again) == [5, 4, 3, 2, 1]


class TestLabels:
    def test_default_labels(self, tmp_path):
        gallery = _gallery(tmp_path, helical=False)
        gallery._sort_column = "None"
        gallery._sort_reverse = False
        gallery._apply_sort()
        assert gallery._make_labels()[0] == "Class 1\nClassDistribution: 30.0%"

    def test_helical_defaults_are_the_type_and_abundance(self, tmp_path):
        gallery = _gallery(tmp_path, helical=True)
        assert gallery._label_columns == ["rlnHelicalType", "rlnClassDistribution"]
        # first tile: class 3
        assert gallery._labels[0] == "HelicalType: 1\nClassDistribution: 25.0%"

    def test_menu_offers_every_column_but_the_image(self, tmp_path):
        gallery = _gallery(tmp_path, helical=True)
        # the helical columns come first, after the class number
        assert gallery._available_label_columns() == [
            "Class",
            "rlnHelicalType",
            "rlnHelicalTypeProbability",
            "rlnClassDistribution",
            "rlnAccuracyRotations",
        ]

    def test_lines_follow_the_menu_order(self, tmp_path):
        gallery = _gallery(tmp_path / "a", helical=True)
        gallery._label_columns = ["rlnClassDistribution", "Class"]
        gallery._save_settings()
        again = _gallery(tmp_path / "b", helical=True)
        assert again._label_columns == ["Class", "rlnClassDistribution"]

    def test_chosen_columns_only(self, tmp_path):
        gallery = _gallery(tmp_path, helical=False)
        gallery._label_columns = ["rlnAccuracyRotations"]
        assert set(gallery._make_labels()) == {"AccuracyRotations: 1.25"}
        gallery._label_columns = []
        assert gallery._make_labels() == [""] * 5

    def test_tooltip_lists_every_column(self, tmp_path):
        gallery = _gallery(tmp_path, helical=True)
        # first tile: class 3, whatever the labels show
        gallery._label_columns = []
        assert gallery._make_tooltips()[0] == (
            "Class 3\nHelicalType: 1\nHelicalTypeProbability: 0.95\n"
            "ClassDistribution: 25.0%\nAccuracyRotations: 1.25"
        )
        assert len(gallery._tooltips) == 5

    def test_choice_is_saved(self, tmp_path):
        gallery = _gallery(tmp_path / "a", helical=False)
        gallery._label_columns = ["Class", "rlnAccuracyRotations"]
        gallery._save_settings()
        again = _gallery(tmp_path / "b", helical=False)
        assert again._label_columns == ["Class", "rlnAccuracyRotations"]


class TestPanel:
    @staticmethod
    def _open(tmp_path, helical):
        from helicon.lib.gui.gallery_widget import ImageGalleryWidget, _ControlPanel

        gallery = Class2dGallery(_write_model(tmp_path, helical))
        window = gallery.open()
        assert window is not None
        panel = window.findChild(_ControlPanel)
        widget = window.findChild(ImageGalleryWidget)
        return gallery, window, panel, widget

    def test_label_menu_changes_the_labels(self, qapp, tmp_path):
        gallery, window, panel, widget = self._open(tmp_path, helical=True)
        try:
            combo = panel._label_columns_combo
            assert not combo.isHidden()
            assert combo.checked_items() == gallery._label_columns
            rows = [combo.model().item(r).text() for r in range(combo.count())]

            assert widget._labels[0] == "HelicalType: 1\nClassDistribution: 25.0%"

            # only the class number
            for col in ("Class", "rlnClassDistribution", "rlnHelicalType"):
                combo.toggle_item(rows.index(col))
            assert combo.checked_items() == ["Class"]
            assert widget._labels[:2] == ["Class 3", "Class 5"]
            assert combo.display_text() == "Class"

            # what the gallery draws for several lines
            combo.toggle_item(rows.index("rlnHelicalTypeProbability"))
            assert widget._labels[0] == "Class 3\nHelicalTypeProbability: 0.95"
            window.resize(400, 400)
            assert not widget.grab().isNull()
        finally:
            window.close()

    def test_hover_shows_the_class_and_follows_the_sort(self, qapp, tmp_path):
        from PySide6.QtCore import QEvent, QPoint
        from PySide6.QtGui import QHelpEvent

        gallery, window, panel, widget = self._open(tmp_path, helical=True)
        try:
            window.resize(600, 400)
            widget.grab()  # lay the tiles out
            first = widget._coords[0].center()
            assert widget.tooltip_at(first).startswith("Class 3\n")
            # nothing between the tiles or past the last one
            assert widget.tooltip_at(QPoint(-5, -5)) == ""

            panel._sort_column_combo.setCurrentText("None")
            widget.grab()
            assert widget.tooltip_at(widget._coords[0].center()).startswith("Class 1\n")

            event = QHelpEvent(QEvent.ToolTip, first, widget.mapToGlobal(first))
            assert widget.event(event)
        finally:
            window.close()

    def test_sort_menu(self, qapp, tmp_path):
        gallery, window, panel, widget = self._open(tmp_path, helical=True)
        try:
            sort_combo = panel._sort_column_combo
            assert [sort_combo.itemText(i) for i in range(sort_combo.count())] == [
                "Helical Type",
                "Abundance",
                "Name",
                "None",
            ]
            assert sort_combo.currentText() == "Helical Type"
            assert panel._sort_reverse_chk.isChecked() is False

            sort_combo.setCurrentText("None")
            assert _class_order(gallery) == [1, 2, 3, 4, 5]
            assert widget._labels[0] == "HelicalType: 2\nClassDistribution: 30.0%"

            # abundance comes with its own direction: large first
            sort_combo.setCurrentText("Abundance")
            assert panel._sort_reverse_chk.isChecked() is True
            assert gallery._dists[gallery._order[0]] == pytest.approx(0.30)

            panel._sort_reverse_chk.setChecked(False)
            assert gallery._dists[gallery._order[0]] == pytest.approx(0.05)
        finally:
            window.close()

    def test_plain_file_has_no_helical_type(self, qapp, tmp_path):
        _, window, panel, _ = self._open(tmp_path, helical=False)
        try:
            sort_combo = panel._sort_column_combo
            items = [sort_combo.itemText(i) for i in range(sort_combo.count())]
            assert items == ["Abundance", "Name", "None"]
            assert panel._label_columns_combo.checked_items() == [
                "Class",
                "rlnClassDistribution",
            ]
        finally:
            window.close()


class TestMultiSelectComboBox:
    def test_toggle_emits_the_selection(self, qapp):
        from helicon.lib.gui.gallery_widget import _MultiSelectComboBox

        combo = _MultiSelectComboBox()
        combo.set_items(["a", "b", "c"], checked=["c"])
        seen = []
        combo.selection_changed.connect(seen.append)
        combo.toggle_item(0)
        combo.toggle_item(2)
        assert seen == [["a", "c"], ["a"]]
        assert combo.display_text() == "a"
        combo.toggle_item(0)
        assert combo.display_text() == "(none)"

    def test_clicks_toggle_and_keep_the_list_open(self, qapp):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        from helicon.lib.gui.gallery_widget import _ControlPanel

        panel = _ControlPanel()
        panel.show_label_columns_ui(True, ["Class", "rlnA", "rlnB"], ["Class"])
        panel.show()
        combo = panel._label_columns_combo
        seen = []
        panel.label_columns_changed.connect(seen.append)
        try:
            combo.showPopup()
            qapp.processEvents()
            view = combo.view()
            for row in (1, 2):
                rect = view.visualRect(combo.model().index(row, 0))
                QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=rect.center())
                qapp.processEvents()
            assert seen == [["Class", "rlnA"], ["Class", "rlnA", "rlnB"]]
            assert view.isVisible()
        finally:
            combo.hidePopup()
            panel.close()
