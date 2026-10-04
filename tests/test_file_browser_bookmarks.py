from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from helicon.lib.gui import file_browser as fb


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_settings(tmp_path, monkeypatch):
    """Keep the tests away from the real saved settings of the user."""
    ini = str(tmp_path / "settings.ini")

    def settings(*args):
        return QSettings(ini, QSettings.Format.IniFormat)

    monkeypatch.setattr(fb, "QSettings", settings)


def _dirs(tmp_path, *names):
    out = []
    for n in names:
        d = tmp_path / n
        d.mkdir()
        out.append(str(d.resolve()))
    return out


class TestStoredBookmarks:
    def test_none_at_first(self):
        assert fb._get_bookmarks() == []

    def test_added_in_order_and_once(self, tmp_path):
        a, b = _dirs(tmp_path, "a", "b")
        fb._add_bookmark(a)
        fb._add_bookmark(b)
        fb._add_bookmark(a)
        assert fb._get_bookmarks() == [a, b]

    def test_a_single_bookmark_is_still_a_list(self, tmp_path):
        (a,) = _dirs(tmp_path, "a")
        fb._add_bookmark(a)
        assert fb._get_bookmarks() == [a]

    def test_removed(self, tmp_path):
        a, b = _dirs(tmp_path, "a", "b")
        fb._add_bookmark(a)
        fb._add_bookmark(b)
        fb._remove_bookmark(a)
        assert fb._get_bookmarks() == [b]
        assert not fb._is_bookmarked(a)

    def test_removing_what_is_not_there_is_harmless(self, tmp_path):
        (a,) = _dirs(tmp_path, "a")
        fb._remove_bookmark(a)
        assert fb._get_bookmarks() == []

    def test_the_last_one_can_be_removed(self, tmp_path):
        (a,) = _dirs(tmp_path, "a")
        fb._add_bookmark(a)
        fb._remove_bookmark(a)
        assert fb._get_bookmarks() == []

    def test_a_path_with_a_dot_or_a_link_is_the_same_folder(self, tmp_path):
        (a,) = _dirs(tmp_path, "a")
        link = tmp_path / "link"
        link.symlink_to(a)
        fb._add_bookmark(str(link))
        assert fb._is_bookmarked(a)
        fb._add_bookmark(a + "/.")
        assert fb._get_bookmarks() == [a]

    def test_a_folder_that_is_gone_is_kept(self, tmp_path):
        (a,) = _dirs(tmp_path, "a")
        fb._add_bookmark(a)
        Path(a).rmdir()
        assert fb._get_bookmarks() == [a]


def _labels(widget):
    widget._refresh_bookmarks_menu()
    return [a.text() for a in widget._bookmarks_menu.actions() if a.text()]


class TestBookmarksMenu:
    def _widget(self, path):
        return fb.FolderBrowserWidget(start_dir=str(path))

    def test_it_offers_to_bookmark_the_folder_shown(self, tmp_path, qapp):
        w = self._widget(tmp_path)
        labels = _labels(w)
        assert labels[0] == "Bookmark This Folder"
        assert "No bookmarks yet" in labels

    def test_the_shortcut_works_with_the_menu_closed(self, tmp_path, qapp):
        w = self._widget(tmp_path)
        assert w._bookmark_toggle_action.shortcut().toString() == "Ctrl+D"
        assert w._bookmark_toggle_action in w.actions()

    def test_toggling_bookmarks_and_unbookmarks(self, tmp_path, qapp):
        w = self._widget(tmp_path)
        w._bookmark_toggle_action.trigger()
        assert fb._is_bookmarked(str(tmp_path))
        labels = _labels(w)
        assert labels[0] == "Remove Bookmark of This Folder"
        assert str(tmp_path.resolve()) in labels
        w._bookmark_toggle_action.trigger()
        assert not fb._is_bookmarked(str(tmp_path))
        assert _labels(w)[0] == "Bookmark This Folder"

    def test_choosing_a_bookmark_goes_there(self, tmp_path, qapp):
        a, b = _dirs(tmp_path, "a", "b")
        fb._add_bookmark(b)
        w = self._widget(a)
        w._refresh_bookmarks_menu()
        action = next(x for x in w._bookmarks_menu.actions() if x.text() == b)
        action.trigger()
        assert w._model._root_path == b

    def test_an_unavailable_folder_is_shown_but_cannot_be_chosen(self, tmp_path, qapp):
        a, gone = _dirs(tmp_path, "a", "gone")
        fb._add_bookmark(gone)
        Path(gone).rmdir()
        w = self._widget(a)
        w._refresh_bookmarks_menu()
        action = next(
            x for x in w._bookmarks_menu.actions() if x.text().startswith(gone)
        )
        assert "not available" in action.text()
        assert not action.isEnabled()

    def test_a_bookmark_can_be_removed_from_the_menu(self, tmp_path, qapp):
        a, b = _dirs(tmp_path, "a", "b")
        fb._add_bookmark(a)
        fb._add_bookmark(b)
        w = self._widget(tmp_path)
        w._on_bookmark_removed(a)
        assert fb._get_bookmarks() == [b]

    def test_bookmarks_are_kept_for_the_next_window(self, tmp_path, qapp):
        a, b = _dirs(tmp_path, "a", "b")
        fb._add_bookmark(b)
        w = self._widget(a)
        assert b in _labels(w)

    def test_the_menu_sits_between_file_and_apps(self, tmp_path, qapp):
        w = self._widget(tmp_path)
        titles = [a.text() for a in w.menuBar().actions()]
        assert titles.index("File") + 1 == titles.index("Bookmarks")
        assert titles.index("Bookmarks") + 1 == titles.index("Apps")


class TestOrderAndClearing:
    def test_a_bookmark_moves_earlier_and_later(self, tmp_path):
        a, b, c = _dirs(tmp_path, "a", "b", "c")
        for p in (a, b, c):
            fb._add_bookmark(p)
        fb._move_bookmark(c, -1)
        assert fb._get_bookmarks() == [a, c, b]
        fb._move_bookmark(a, 1)
        assert fb._get_bookmarks() == [c, a, b]

    def test_moving_past_either_end_stops_at_the_end(self, tmp_path):
        a, b = _dirs(tmp_path, "a", "b")
        fb._add_bookmark(a)
        fb._add_bookmark(b)
        fb._move_bookmark(a, -5)
        fb._move_bookmark(b, 5)
        assert fb._get_bookmarks() == [a, b]

    def test_moving_an_unknown_folder_changes_nothing(self, tmp_path):
        a, b = _dirs(tmp_path, "a", "b")
        fb._add_bookmark(a)
        fb._move_bookmark(b, -1)
        assert fb._get_bookmarks() == [a]

    def test_clearing_removes_all(self, tmp_path):
        a, b = _dirs(tmp_path, "a", "b")
        fb._add_bookmark(a)
        fb._add_bookmark(b)
        fb._clear_bookmarks()
        assert fb._get_bookmarks() == []


class TestClearAllFromTheMenu:
    def _widget(self, path):
        return fb.FolderBrowserWidget(start_dir=str(path))

    def _clear_action(self, w):
        w._refresh_bookmarks_menu()
        return next(
            x for x in w._bookmarks_menu.actions() if x.text() == "Clear All Bookmarks"
        )

    def test_it_asks_first_and_clears_on_yes(self, tmp_path, qapp):
        a, b = _dirs(tmp_path, "a", "b")
        fb._add_bookmark(a)
        fb._add_bookmark(b)
        w = self._widget(tmp_path)
        yes = fb.QMessageBox.StandardButton.Yes
        with patch.object(fb.QMessageBox, "question", return_value=yes) as ask:
            self._clear_action(w).trigger()
        ask.assert_called_once()
        assert fb._get_bookmarks() == []
        assert "No bookmarks yet" in _labels(w)

    def test_it_keeps_everything_on_no(self, tmp_path, qapp):
        (a,) = _dirs(tmp_path, "a")
        fb._add_bookmark(a)
        w = self._widget(tmp_path)
        no = fb.QMessageBox.StandardButton.No
        with patch.object(fb.QMessageBox, "question", return_value=no):
            self._clear_action(w).trigger()
        assert fb._get_bookmarks() == [a]

    def test_the_entries_are_only_there_when_there_are_bookmarks(self, tmp_path, qapp):
        w = self._widget(tmp_path)
        assert "Clear All Bookmarks" not in _labels(w)
        assert "Manage Bookmarks…" not in _labels(w)
        fb._add_bookmark(str(tmp_path))
        assert "Clear All Bookmarks" in _labels(w)
        assert "Manage Bookmarks…" in _labels(w)


class TestManageDialog:
    def _three(self, tmp_path):
        a, b, c = _dirs(tmp_path, "a", "b", "c")
        for p in (a, b, c):
            fb._add_bookmark(p)
        return a, b, c

    def test_it_lists_the_bookmarks_in_order(self, tmp_path, qapp):
        a, b, c = self._three(tmp_path)
        assert fb._BookmarksDialog().paths() == [a, b, c]

    def test_move_buttons_reorder_and_save(self, tmp_path, qapp):
        a, b, c = self._three(tmp_path)
        d = fb._BookmarksDialog()
        d._list.setCurrentRow(2)
        d._up.click()
        assert d.paths() == [a, c, b]
        assert fb._get_bookmarks() == [a, c, b]
        assert d._selected() == c  # the entry that was moved stays selected

    def test_the_buttons_match_the_selection(self, tmp_path, qapp):
        self._three(tmp_path)
        d = fb._BookmarksDialog()
        d._list.setCurrentRow(0)
        assert not d._up.isEnabled() and d._down.isEnabled()
        d._list.setCurrentRow(2)
        assert d._up.isEnabled() and not d._down.isEnabled()

    def test_a_drag_and_drop_is_saved(self, tmp_path, qapp):
        a, b, c = self._three(tmp_path)
        d = fb._BookmarksDialog()
        item = d._list.takeItem(0)  # what a drop does: the entry lands elsewhere
        d._list.insertItem(2, item)
        d._save_order()
        assert fb._get_bookmarks() == [b, c, a]

    def test_remove_takes_one_and_selects_a_neighbour(self, tmp_path, qapp):
        a, b, c = self._three(tmp_path)
        d = fb._BookmarksDialog()
        d._list.setCurrentRow(1)
        d._remove.click()
        assert d.paths() == [a, c]
        assert d._selected() == c

    def test_clear_all_asks_first(self, tmp_path, qapp):
        a, b, c = self._three(tmp_path)
        d = fb._BookmarksDialog()
        no = fb.QMessageBox.StandardButton.No
        with patch.object(fb.QMessageBox, "question", return_value=no):
            d._clear.click()
        assert fb._get_bookmarks() == [a, b, c]
        yes = fb.QMessageBox.StandardButton.Yes
        with patch.object(fb.QMessageBox, "question", return_value=yes):
            d._clear.click()
        assert fb._get_bookmarks() == [] and d.paths() == []
        assert not d._clear.isEnabled()

    def test_the_menu_opens_it(self, tmp_path, qapp):
        self._three(tmp_path)
        w = fb.FolderBrowserWidget(start_dir=str(tmp_path))
        with patch.object(fb._BookmarksDialog, "exec", return_value=0) as run:
            w._manage_bookmarks()
        run.assert_called_once()
