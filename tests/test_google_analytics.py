"""The Google tag of the web apps, reporting pages without their queries."""

import helicon


def _page(monkeypatch, analytics=None):
    from shiny import ui

    from helicon.webApps import app

    if analytics is None:
        monkeypatch.delenv("HELICON_ANALYTICS", raising=False)
    else:
        monkeypatch.setenv("HELICON_ANALYTICS", analytics)
    return _render(app._page("System", "dark", "Home"))


def _render(tag):
    """The whole page, head content (where the tag goes) included."""
    import htmltools

    return htmltools.HTMLDocument(tag).render()["html"]


class TestTheTag:
    def test_the_app_carries_the_helicon_tag(self, monkeypatch):
        html = _page(monkeypatch)
        assert "googletagmanager.com/gtag/js?id=GT-579RJLLW" in html
        assert "gtag('config', ID" in html and '"GT-579RJLLW"' in html

    def test_it_can_be_turned_off(self, monkeypatch):
        assert "googletagmanager" not in _page(monkeypatch, "0")

    def test_only_the_page_and_tab_are_reported(self):
        tag = helicon.shiny.google_analytics("G-TEST")
        script = str(tag.head)  # head_content: an HTML dependency with a head
        # page_location is rebuilt from origin + pathname + ?tab=<name>; the
        # address as it is (carrying data URLs and server paths) never goes
        assert "location.origin + location.pathname" in script
        assert "location.href" not in script
        assert ".get('tab')" in script
        assert script.count("page_location:") == 4

    def test_no_id_no_tag(self):
        assert helicon.shiny.google_analytics("") is None
