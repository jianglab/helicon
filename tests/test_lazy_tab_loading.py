"""The tabs that load on first use say so while they load."""

import re

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

    def test_every_lazy_tab_has_its_note(self):
        source = tab_source(app)
        labelled = dict(re.findall(r'_lazy_panel\(\s*"(\w+)",\s*"(\w+)"\s*\)', source))
        assert {spec["prefix"] for spec in app._LAZY_TABS.values()} <= set(labelled)
        for name, spec in app._LAZY_TABS.items():
            assert labelled[spec["prefix"]] == name

    def test_the_style_is_in_the_page(self):
        assert "_LAZY_PANEL_CSS" in tab_source(app)
        assert "ui.tags.style(_LAZY_PANEL_CSS)" in tab_source(app)

    def test_a_tab_that_cannot_be_imported_shows_the_error_not_the_note_for_ever(self):
        source = tab_source(app)
        assert "could not be loaded" in source
        assert "failed to load tab" in source
