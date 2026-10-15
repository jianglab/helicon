"""One look for the file browser and the windows it opens."""

import pytest
from PySide6.QtWidgets import QApplication

from _settings import use_theme
from helicon.lib.gui import file_browser as fb
from helicon.lib.gui import gallery_widget as gw
from helicon.lib.gui import theme


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(params=["Dark", "Light"])
def saved_theme(request, monkeypatch):
    use_theme(monkeypatch, request.param)
    return request.param


class TestSharedLook:
    def test_the_stylesheet_of_the_windows_has_the_common_widgets(self, saved_theme):
        sheet = theme._display_theme_stylesheet()
        colors = fb._THEME_COLORS[saved_theme]
        assert "QSlider::sub-page:horizontal" in sheet
        assert "QScrollBar:vertical" in sheet and "QToolTip" in sheet
        assert colors["accent"] in sheet
        # the common widgets come after the general rules, so they win
        assert sheet.index("QSlider::sub-page:horizontal") > sheet.index("QWidget {")

    def test_the_browser_has_them_too(self, saved_theme):
        sheet = fb._browser_stylesheet(fb._THEME_COLORS[saved_theme])
        assert "QSlider::sub-page:horizontal" in sheet and "QTreeView" in sheet

    def test_the_gallery_uses_the_colours_of_the_browser(self, saved_theme):
        colors = fb._THEME_COLORS[saved_theme]
        g = gw._gallery_theme_colors()
        assert g["accent"] == colors["accent"] == "#007bc2"
        assert g["panel"] == colors["window"] and g["border"] == colors["border"]
        assert colors["accent"] in gw._gallery_qss(g)

    def test_a_label_has_no_background_of_its_own(self, saved_theme):
        assert "QLabel { background: transparent; }" in fb._widgets_qss(
            fb._THEME_COLORS[saved_theme]
        )


class TestControls:
    def test_the_orthogonal_controls_are_styled_by_the_sheet_not_widget_by_widget(
        self, qapp, saved_theme
    ):
        bar = gw._ControlBar(8, 8, 8)
        for slider in bar._sliders:
            assert slider.styleSheet() == ""
        assert bar._movie_btn.styleSheet() == "" and bar._reset_btn.styleSheet() == ""
        assert not bar._movie_btn.icon().isNull()
        assert not bar._reset_btn.icon().isNull()

    def test_the_movie_button_shows_a_pause_while_it_plays(self, qapp, saved_theme):
        bar = gw._ControlBar(8, 8, 8)
        stopped = bar._movie_btn.icon().cacheKey()
        bar._movie_btn.setChecked(True)
        assert bar._movie_btn.icon().cacheKey() != stopped

    def test_the_gallery_panel_is_styled_by_the_sheet_and_has_an_icon(
        self, qapp, saved_theme
    ):
        panel = gw._ControlPanel()
        assert panel._brightness_slider.styleSheet() == ""
        assert panel._auto_btn.styleSheet() == ""
        assert not panel._auto_btn.icon().isNull()
        panel._apply_display_theme()
        assert not panel._auto_btn.icon().isNull()


class TestNapariTheme:
    def test_the_viewers_take_a_theme_made_of_the_browser_colours(self, saved_theme):
        pytest.importorskip("napari")
        from napari.utils.theme import get_theme

        theme_id = theme._napari_viewer_theme()
        assert theme_id == f"helicon-{saved_theme.lower()}"
        t = get_theme(theme_id)
        colors = fb._THEME_COLORS[saved_theme]
        assert str(t.current).lower() == colors["accent"]
        assert str(t.background).lower() == colors["window"]

    def test_it_falls_back_to_the_napari_theme_where_registering_fails(
        self, monkeypatch, tmp_path
    ):
        pytest.importorskip("napari")
        import napari.resources._icons as icons
        import napari.utils.theme as napari_theme

        # no icons written by an earlier run
        monkeypatch.setattr(icons, "_theme_path", lambda name: tmp_path / name)

        def boom(*a, **k):
            raise RuntimeError("no")

        monkeypatch.setattr(napari_theme, "register_theme", boom)
        monkeypatch.setattr(napari_theme, "available_themes", lambda: [])
        use_theme(monkeypatch, "Dark")
        assert theme._napari_viewer_theme() == "dark"


class TestNapariThemeIcons:
    """The icon files of a napari theme are written once, not on every start."""

    @pytest.fixture
    def folder(self, monkeypatch, tmp_path):
        pytest.importorskip("napari")
        import napari.resources._icons as icons
        import napari.utils.theme as napari_theme

        monkeypatch.setattr(icons, "_theme_path", lambda name: tmp_path / name)
        yield tmp_path
        napari_theme._themes.pop("helicon-test", None)

    def _colours(self):
        from napari.utils.theme import get_theme

        base = get_theme("dark").model_dump()
        base.update(id="helicon-test", label="Helicon test")
        return base

    def test_the_second_start_does_not_write_the_icons_again(self, folder, monkeypatch):
        import napari.utils.theme as napari_theme

        base = self._colours()
        theme._register_napari_theme("helicon-test", base)
        assert any((folder / "helicon-test").glob("*.svg"))
        napari_theme._themes.pop("helicon-test")

        def not_again(*a, **k):
            raise AssertionError("the icons were written again")

        monkeypatch.setattr(napari_theme, "register_theme", not_again)
        theme._register_napari_theme("helicon-test", base)
        assert "helicon-test" in napari_theme.available_themes()

    def test_other_colours_write_the_icons_again(self, folder, monkeypatch):
        import napari.utils.theme as napari_theme

        base = self._colours()
        theme._register_napari_theme("helicon-test", base)
        written = []
        real = napari_theme.register_theme
        monkeypatch.setattr(
            napari_theme, "register_theme", lambda *a: written.append(a) or real(*a)
        )
        theme._register_napari_theme("helicon-test", dict(base, icon="#123456"))
        assert len(written) == 1
        assert str(napari_theme.get_theme("helicon-test").icon).lower() == "#123456"

    def test_missing_icon_files_are_written_again(self, folder, monkeypatch):

        base = self._colours()
        theme._register_napari_theme("helicon-test", base)
        for svg in (folder / "helicon-test").glob("*.svg"):
            svg.unlink()
        theme._register_napari_theme("helicon-test", base)
        assert any((folder / "helicon-test").glob("*.svg"))
