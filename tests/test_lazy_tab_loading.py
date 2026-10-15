"""The tabs that load on first use say so while they load."""

from helicon.webApps import app
from _source import tab_source


class TestLoadingNote:
    def test_the_placeholder_carries_a_note_with_the_tabs_name(self):
        html = str(app._lazy_panel("hill", "HILL"))
        assert "Loading HILL" in html
        assert "helicon-lazy-loading" in html
        # the place the page appears is still there, under the id the server fills
        assert 'id="lazy_tab_hill"' in html

    def test_the_note_hides_once_the_page_is_in_the_placeholder(self):
        css = app._LAZY_PANEL_CSS
        assert ":has(> .shiny-html-output:not(:empty))" in css
        assert "display: none" in css

    @staticmethod
    def _page_html():
        import htmltools

        return htmltools.TagList(app._page("Light", "light", "Home")).render()["html"]

    def test_every_lazy_tab_has_its_note(self):
        html = self._page_html()
        for name, spec in app._LAZY_TABS.items():
            assert f'id="lazy_tab_{spec["prefix"]}"' in html, name
            assert f"Loading {name}" in html, name

    def test_the_style_is_in_the_page(self):
        html = self._page_html()
        assert ":has(> .shiny-html-output:not(:empty))" in html
        assert "helicon-lazy-loading" in html

    def test_a_tab_that_cannot_be_imported_shows_the_error_not_the_note_for_ever(self):
        source = tab_source(app)
        assert "could not be loaded" in source
        assert "failed to load tab" in source
