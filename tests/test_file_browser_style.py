"""The look of the file browser: icons by kind, muted columns, folders first, dates."""

from datetime import datetime, timedelta

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from helicon.lib.gui import file_browser as fb


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_settings(tmp_path, monkeypatch):
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(
        fb, "QSettings", lambda *a: QSettings(ini, QSettings.Format.IniFormat)
    )


@pytest.fixture
def folder(tmp_path):
    for name in ("b_dir", "a_dir"):
        (tmp_path / name).mkdir()
    for name in ("z.star", "a.mrc", "stack.mrcs", "notes.txt", "x.pdf", "y.bin"):
        (tmp_path / name).write_bytes(b"x")
    return tmp_path


def _names(model):
    return [
        model.index(r, 0).data(Qt.ItemDataRole.DisplayRole)
        for r in range(model.rowCount())
    ]


class TestKinds:
    @pytest.mark.parametrize(
        "label, suffix, is_dir, kind",
        [
            ("Folder", "", True, "folder"),
            ("MRC", ".mrc", False, "image"),
            ("MRCS", ".mrcs", False, "stack"),
            ("TIF", ".tif", False, "image"),
            ("STAR", ".star", False, "table"),
            ("Metadata", ".star", False, "table"),
            ("CryoSPARC", ".cs", False, "table"),
            ("Text", ".txt", False, "text"),
            ("PDF", ".pdf", False, "pdf"),
            ("Browser", ".html", False, "web"),
            ("BIN", ".bin", False, "file"),
            ("File", "", False, "file"),
        ],
    )
    def test_the_kind_of_a_row(self, label, suffix, is_dir, kind):
        assert fb._icon_kind(label, suffix, is_dir) == kind

    def test_every_kind_has_an_icon_name(self):
        for kind in (
            "folder",
            "image",
            "stack",
            "table",
            "text",
            "pdf",
            "web",
            "plot",
            "file",
        ):
            assert kind in fb._KIND_ICONS


class TestModel:
    def test_the_rows_have_icons_and_kinds(self, qapp, folder):
        model = fb.FileBrowserModel(str(folder))
        kinds = {
            model.index(r, 0)
            .data(Qt.ItemDataRole.DisplayRole): model.index(r, 0)
            .data(fb.ROLE_KIND)
            for r in range(model.rowCount())
        }
        assert kinds["a_dir/"] == "folder" and kinds["a.mrc"] == "image"
        assert kinds["stack.mrcs"] == "stack" and kinds["z.star"] == "table"
        for r in range(model.rowCount()):
            assert not model.index(r, 0).data(Qt.ItemDataRole.DecorationRole).isNull()

    def test_the_numbers_are_right_aligned_and_the_columns_muted(self, qapp, folder):
        model = fb.FileBrowserModel(str(folder))
        size = model.index(0, fb.COL_SIZE)
        align = size.data(Qt.ItemDataRole.TextAlignmentRole)
        assert align & Qt.AlignmentFlag.AlignRight
        assert (
            model.index(0, fb.COL_TYPE).data(Qt.ItemDataRole.TextAlignmentRole) is None
        )
        muted = model.index(0, fb.COL_TYPE).data(Qt.ItemDataRole.ForegroundRole)
        assert (
            muted.color().name()
            == fb._THEME_COLORS[fb._resolved_theme(fb._saved_theme())]["muted"]
        )
        # the name keeps the colour of the text
        assert model.index(0, fb.COL_NAME).data(Qt.ItemDataRole.ForegroundRole) is None

    def test_a_theme_change_recolours_the_rows(self, qapp, folder):
        model = fb.FileBrowserModel(str(folder))
        seen = []
        model.dataChanged.connect(lambda *a: seen.append(a))
        model.set_theme_colors(fb._THEME_COLORS["Light"])
        assert seen
        assert (
            model.index(0, fb.COL_TYPE)
            .data(Qt.ItemDataRole.ForegroundRole)
            .color()
            .name()
            == fb._THEME_COLORS["Light"]["muted"]
        )

    @pytest.mark.parametrize(
        "order", [Qt.SortOrder.AscendingOrder, Qt.SortOrder.DescendingOrder]
    )
    def test_folders_stay_first_by_name_in_either_order(self, qapp, folder, order):
        model = fb.FileBrowserModel(str(folder))
        model.sort(fb.COL_NAME, order)
        names = _names(model)
        assert [n.endswith("/") for n in names] == [True, True] + [False] * 6
        folders, files = names[:2], names[2:]
        assert folders == sorted(folders, reverse=order == Qt.SortOrder.DescendingOrder)
        assert files == sorted(files, reverse=order == Qt.SortOrder.DescendingOrder)


class TestDates:
    def test_today_yesterday_and_older(self):
        now = datetime(2026, 10, 13, 21, 30)
        assert fb._format_modified(datetime(2026, 10, 13, 9, 5), now) == "Today 09:05"
        assert fb._format_modified(now - timedelta(days=1), now) == "Yesterday 21:30"
        assert fb._format_modified(now - timedelta(days=9), now) == "2026-10-04 21:30"


class TestThemes:
    @pytest.mark.parametrize("theme", ["Dark", "Light"])
    def test_the_colours_the_style_needs_are_in_each_theme(self, theme):
        colors = fb._THEME_COLORS[theme]
        for key in ("muted", "alt", "folder", "file_icon", "hover", "accent"):
            assert key in colors
        sheet = fb._browser_stylesheet(colors)
        assert colors["alt"] in sheet and colors["muted"] in sheet
