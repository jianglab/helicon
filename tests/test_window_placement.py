"""The browser window is put in its place before it is shown, not moved after."""

import pytest
from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtWidgets import QApplication, QWidget

from helicon.lib.gui import viewer as v


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(v, "_get_qsettings", lambda: s)
    return s


class TestPlacedBeforeItIsShown:
    def test_a_saved_geometry_is_applied_before_the_window_is_visible(
        self, qapp, settings
    ):
        first = QWidget()
        first.setGeometry(120, 90, 700, 400)
        settings.setValue("dock_ba", first.saveGeometry())
        window = QWidget()
        assert not window.isVisible()
        assert v._restore_dock_geometry_now(window)
        assert (window.width(), window.height()) == (700, 400)
        assert window.property("geometry_restored") is True

    def test_without_a_saved_geometry_it_is_not_applied(self, qapp, settings):
        assert not v._restore_dock_geometry_now(QWidget())

    def test_a_damaged_geometry_is_not_applied_and_does_not_fail(self, qapp, settings):
        settings.setValue("dock_ba", QByteArray(b"not a geometry"))
        assert not v._restore_dock_geometry_now(QWidget())

    def test_the_default_is_a_size_and_a_place_on_the_screen(self, qapp):
        window = QWidget()
        v._place_dock_by_default(window, 1000, 700)
        area = qapp.primaryScreen().availableGeometry()
        assert window.width() <= area.width() and window.height() <= area.height()
        assert area.contains(window.pos())
        assert window.property("geometry_restored") is True

    def test_a_window_that_was_placed_is_not_moved_again_after_it_is_shown(
        self, qapp, settings
    ):
        saved = QWidget()
        saved.setGeometry(50, 50, 600, 400)
        settings.setValue("dock_ba", saved.saveGeometry())
        window = QWidget()
        v._restore_dock_geometry_now(window)
        window.resize(500, 300)  # whatever happens meanwhile
        v._restore_geometry(window, None)  # returns at once: no timer, nothing restored
        qapp.processEvents()
        assert (window.width(), window.height()) == (500, 300)

    def test_a_new_viewer_gets_the_size_of_the_one_closed_last(self, qapp, settings):
        saved = QWidget()
        saved.setGeometry(60, 70, 640, 480)
        settings.setValue("viewer_ba", saved.saveGeometry())
        window = QWidget()

        class _Viewer:
            class window:
                pass

        _Viewer.window._qt_window = window
        assert v._restore_viewer_geometry(_Viewer)
        assert (window.width(), window.height()) == (640, 480)
        assert not v._restore_viewer_geometry(object())  # not a viewer: nothing done

    def test_a_window_that_was_not_placed_is_still_restored_later(self, qapp, settings):
        saved = QWidget()
        saved.setGeometry(50, 50, 600, 400)
        settings.setValue("dock_ba", saved.saveGeometry())
        window = QWidget()
        v._restore_geometry(window, None)
        for _ in range(5):
            qapp.processEvents()
        assert (window.width(), window.height()) == (600, 400)
        window.close()
