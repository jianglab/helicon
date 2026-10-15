"""The file browser keeps answering: slow work is done in threads, a little at a time."""

import threading
import time

import mrcfile
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from helicon.commands import display
from helicon.lib.gui import browser_parts as bp
from helicon.lib.gui import file_browser as fb


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _pump(app, seconds=5.0, until=lambda: False):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not until():
        app.processEvents()
        time.sleep(0.005)


class TestLongFolders:
    @pytest.fixture
    def many(self, tmp_path):
        for i in range(fb.SYNC_ROWS + 400):
            (tmp_path / f"f{i:04d}.txt").write_text("x")
        (tmp_path / "sub").mkdir()
        return tmp_path

    def test_a_long_folder_is_listed_in_slices_and_the_window_answers_between(
        self, qapp, many, monkeypatch
    ):
        # a slice is as many rows as fit in INSERT_SLICE_S: on a fast machine all of the 600
        # rows fit in the first, and nothing would be left to see between slices. One slice
        # that ends after the first twenty rows makes the test the same on every machine.
        from helicon.lib.gui import browser_model

        monkeypatch.setattr(browser_model, "INSERT_SLICE_S", 0.0)
        model = fb.FileBrowserModel(str(many))
        assert model._loading and model.rowCount() < fb.SYNC_ROWS + 401
        seen = []
        model.load_finished.connect(lambda: seen.append(True))
        slices = 0
        while not seen:
            slices += 1
            qapp.processEvents()
            assert slices < 5000
        assert slices >= 10  # the rows came in many slices, not in one
        assert model.rowCount() == fb.SYNC_ROWS + 401
        # folders first, then the files in order
        assert model.item(0, 0).text() == "sub/"
        assert model.item(1, 0).text() == "f0000.txt"

    def test_the_sizes_and_dates_arrive_from_threads(self, qapp, many):
        model = fb.FileBrowserModel(str(many))
        _pump(qapp, 20, lambda: not model._loading and model._stats_running == 0)
        assert model._stats_running == 0
        for row in (1, 100, model.rowCount() - 1):
            assert model.item(row, fb.COL_SIZE).text() == "1 B"
            assert model.item(row, fb.COL_MODIFIED).text().startswith("Today")
        assert model.item(0, fb.COL_MODIFIED).text() != ""  # the folder has a date too

    def test_a_small_folder_is_complete_at_once(self, qapp, tmp_path):
        for i in range(5):
            (tmp_path / f"s{i}.txt").write_text("xx")
        model = fb.FileBrowserModel(str(tmp_path))
        assert model.rowCount() == 5 and not model._loading
        assert model.item(0, fb.COL_SIZE).text() == "2 B"

    def test_going_elsewhere_drops_the_listing_that_was_under_way(
        self, qapp, many, tmp_path_factory
    ):
        other = tmp_path_factory.mktemp("other")
        (other / "only.txt").write_text("x")
        model = fb.FileBrowserModel(str(many))
        model.set_root_path(str(other))
        _pump(qapp, 5, lambda: not model._loading)
        names = [model.item(r, 0).text() for r in range(model.rowCount())]
        assert names == ["only.txt"]

    def test_sorting_waits_for_the_whole_folder(self, qapp, many):
        model = fb.FileBrowserModel(str(many))
        model.sort(fb.COL_NAME, fb.Qt.SortOrder.DescendingOrder)
        assert model._sort_state == (fb.COL_NAME, fb.Qt.SortOrder.DescendingOrder)
        _pump(qapp, 10, lambda: not model._loading)
        # a folder first, then the files, last to first
        assert model.item(0, 0).text() == "sub/"
        assert model.item(1, 0).text() == f"f{fb.SYNC_ROWS + 399:04d}.txt"

    def test_a_row_is_found_by_its_path_whatever_the_order(self, qapp, many):
        model = fb.FileBrowserModel(str(many))
        _pump(qapp, 10, lambda: not model._loading)
        path = str(many / "f0007.txt")
        row = model._row_for_filepath(path)
        assert model.item(row, 0).data(fb.Qt.ItemDataRole.UserRole) == path
        model.sort(fb.COL_NAME, fb.Qt.SortOrder.DescendingOrder)
        row2 = model._row_for_filepath(path)
        assert (
            row2 != row
            and model.item(row2, 0).data(fb.Qt.ItemDataRole.UserRole) == path
        )
        assert model._row_for_filepath(str(many / "none.txt")) == -1


class TestSelection:
    @pytest.fixture
    def browser(self, qapp, tmp_path):
        np_data = np.zeros((4, 8, 8), dtype=np.float32)
        with mrcfile.new(tmp_path / "s.mrcs", overwrite=True) as m:
            m.set_data(np_data)
        (tmp_path / "t.txt").write_text("text")
        w = fb.FolderBrowserWidget(start_dir=str(tmp_path))
        w.show()
        yield w
        w.wait_for_selection()
        w._preview.wait()
        w.close()

    def _select(self, w, name):
        for r in range(w._model.rowCount()):
            if w._model.index(r, 0).data() == name:
                w._tree.setCurrentIndex(w._model.index(r, 0))
                return
        raise KeyError(name)

    def test_the_buttons_of_a_file_are_worked_out_in_a_thread(
        self, browser, monkeypatch
    ):
        where = {}
        original = fb.FolderBrowserWidget._display_modes_for

        def spy(self, path):
            where["thread"] = threading.current_thread()
            return original(self, path)

        monkeypatch.setattr(fb.FolderBrowserWidget, "_display_modes_for", spy)
        self._select(browser, "s.mrcs")
        # nothing is shown until the thread has answered
        assert browser._action_bar.isHidden()
        browser.wait_for_selection()
        assert where["thread"] is not threading.main_thread()
        assert not browser._action_bar.isHidden()
        assert not browser._btn_gallery.isHidden()

    def test_the_answer_for_a_file_no_longer_selected_is_dropped(self, browser):
        self._select(browser, "s.mrcs")
        stale = browser._modes_serial
        self._select(browser, "t.txt")
        browser._on_modes_ready(
            stale, dict(path="x", modes=["slice"], slice_text="x", chimerax=True)
        )
        browser.wait_for_selection()
        assert browser._current_path.endswith("t.txt")

    def test_selecting_a_folder_hides_the_buttons_at_once(self, browser, tmp_path):
        self._select(browser, "t.txt")
        browser.wait_for_selection()
        assert not browser._action_bar.isHidden()
        (tmp_path / "d").mkdir()
        browser._refresh()
        _pump(QApplication.instance(), 2, lambda: not browser._model._loading)
        self._select(browser, "d/")
        assert browser._action_bar.isHidden() and browser._current_path is None
        browser.wait_for_selection()
        assert browser._action_bar.isHidden()


class TestPreviewOfLargeStacks:
    def test_a_stack_of_big_frames_reads_only_what_the_budget_allows(
        self, tmp_path, monkeypatch
    ):
        with mrcfile.new(tmp_path / "movie.mrcs", overwrite=True) as m:
            m.set_data(
                np.random.default_rng(0).normal(size=(30, 64, 64)).astype(np.float32)
            )
        monkeypatch.setattr(bp, "_STACK_PREVIEW_BYTES", 3 * 64 * 64 * 4)  # three frames
        info = bp.build_preview(tmp_path / "movie.mrcs")
        assert dict(info["facts"])["Images"] == "30 of 64 × 64"
        # three tiles, in a row of 3
        assert info["image"].shape == (96, 3 * 98 - 2)

    def test_one_frame_is_read_at_least(self, tmp_path, monkeypatch):
        with mrcfile.new(tmp_path / "m.mrcs", overwrite=True) as m:
            m.set_data(np.zeros((5, 16, 16), dtype=np.float32))
        monkeypatch.setattr(bp, "_STACK_PREVIEW_BYTES", 1)
        assert bp.build_preview(tmp_path / "m.mrcs")["image"] is not None

    def test_a_folder_is_listed_without_a_stat_of_each_entry(
        self, tmp_path, monkeypatch
    ):
        for i in range(50):
            (tmp_path / f"f{i}.txt").write_text("x")
        (tmp_path / "d").mkdir()
        from pathlib import Path

        def refuse(self, *a, **k):
            raise AssertionError(
                "a Path.is_dir() on every entry (a stat) is what froze the window"
            )

        monkeypatch.setattr(Path, "is_dir", refuse)
        info = bp._folder_preview(tmp_path)
        assert ("Contains", "50 files, 1 folders") in info["facts"]
        assert info["text"].splitlines()[0] == "d/"


class TestBackgroundOpening:
    def test_the_work_is_in_a_thread_and_the_result_on_the_main_thread(self, qapp):
        got = {}

        def work():
            got["work"] = threading.current_thread()
            return 42

        def then(result):
            got["then"] = (threading.current_thread(), result)

        display._run_in_thread(work, then)
        _pump(qapp, 5, lambda: "then" in got)
        assert got["work"] is not threading.main_thread()
        assert got["then"] == (threading.main_thread(), 42)
        assert QApplication.overrideCursor() is None  # the busy cursor is gone

    def test_an_error_stops_it_and_restores_the_cursor(self, qapp, capsys):
        done = []

        def work():
            raise OSError("unreadable")

        display._run_in_thread(work, lambda r: done.append(r))
        _pump(
            qapp,
            3,
            lambda: QApplication.overrideCursor() is None
            and not display._BACKGROUND_JOBS,
        )
        assert done == [] and QApplication.overrideCursor() is None
        assert "unreadable" in capsys.readouterr().out

    def test_a_star_file_is_read_in_the_background_when_asked(
        self, qapp, tmp_path, monkeypatch
    ):
        with mrcfile.new(tmp_path / "p.mrcs", overwrite=True) as m:
            m.set_data(np.zeros((3, 8, 8), dtype=np.float32))
            m.voxel_size = 1.0
        star = tmp_path / "particles.star"
        star.write_text(
            "data_optics\nloop_\n_rlnOpticsGroup\n_rlnImagePixelSize\n_rlnImageSize\n1 1.0 8\n\n"
            "data_particles\nloop_\n_rlnImageName\n"
            + "".join(f"{i + 1:06d}@{tmp_path / 'p.mrcs'}\n" for i in range(3))
        )
        shown, parsed_in = [], []
        original = display._parse_star_image_refs

        def spy(path):
            parsed_in.append(threading.current_thread())
            return original(path)

        monkeypatch.setattr(display, "_parse_star_image_refs", spy)
        monkeypatch.setattr(
            display, "_open_image_ref_stack", lambda *a, **k: shown.append(a)
        )
        display._open_file(None, str(star), mode="gallery", background=True)
        assert shown == []  # not yet: the file is being read
        _pump(qapp, 10, lambda: bool(shown))
        assert len(shown) == 1
        assert parsed_in[0] is not threading.main_thread()
        # and in turn, as before, when not asked
        shown.clear()
        display._open_file(None, str(star), mode="gallery")
        assert len(shown) == 1 and parsed_in[-1] is threading.main_thread()


class TestWarmUp:
    def test_napari_is_imported_in_a_thread(self, monkeypatch):
        seen = {}
        real_import = __import__

        def spy(name, *a, **k):
            if name == "napari":
                seen["thread"] = threading.current_thread()
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", spy)
        thread = display._warm_up_napari()
        thread.join(30)
        assert seen.get("thread") is thread
        assert thread.daemon


class TestGettextLookups:
    """napari asks gettext about the same language files hundreds of times."""

    def test_the_file_system_is_asked_once_per_question(self, monkeypatch):
        import gettext

        from helicon.lib.gui import viewer

        asked = []

        def find(domain, localedir=None, languages=None, all=False):
            asked.append((domain, localedir, languages, all))
            return ["a.mo", "b.mo"] if all else None

        monkeypatch.setattr(gettext, "find", find)
        monkeypatch.delattr(gettext, "_helicon_cached", raising=False)
        viewer._cache_gettext_lookups()
        for _ in range(3):
            assert gettext.find("napari", "/loc", ["de"]) is None
        assert len(asked) == 1
        assert gettext.find("napari", "/loc", ["de"], True) == ["a.mo", "b.mo"]
        assert len(asked) == 2  # all=True is another question
        assert gettext.find("napari", "/loc", ["fr"]) is None
        assert len(asked) == 3
        monkeypatch.delattr(gettext, "_helicon_cached")

    def test_the_language_of_the_environment_is_part_of_the_question(self, monkeypatch):
        import gettext

        from helicon.lib.gui import viewer

        asked = []
        monkeypatch.setattr(gettext, "find", lambda *a, **k: asked.append(a) or "x.mo")
        monkeypatch.delattr(gettext, "_helicon_cached", raising=False)
        viewer._cache_gettext_lookups()
        monkeypatch.setenv("LANGUAGE", "de")
        gettext.find("napari", "/loc")
        monkeypatch.setenv("LANGUAGE", "fr")
        gettext.find("napari", "/loc")
        assert len(asked) == 2
        monkeypatch.delattr(gettext, "_helicon_cached")

    def test_it_is_installed_once(self, monkeypatch):
        import gettext

        from helicon.lib.gui import viewer

        monkeypatch.setattr(gettext, "find", gettext.find)
        monkeypatch.delattr(gettext, "_helicon_cached", raising=False)
        viewer._cache_gettext_lookups()
        installed = gettext.find
        viewer._cache_gettext_lookups()
        assert gettext.find is installed
        monkeypatch.delattr(gettext, "_helicon_cached")
