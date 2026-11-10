"""In a browser-side (Shinylive) copy, the "server" input mode is a local folder.

There are no server files there, so the "server" input source is offered as the folder
of the visitor's computer they give the page: the same path field, the same file dialog
(confined to the folder), the same loaders. A choose button above the path field is all a
tab shows beyond what it shows on a server, and a tab changes nothing but to ask whether
it can read files by a path (``deployment.has_server_files``).
"""

import pytest

from _ui import page

import helicon
from helicon.webApps import deployment, local_folder


@pytest.fixture
def in_the_browser(monkeypatch):
    monkeypatch.delenv(deployment._OVERRIDE, raising=False)
    monkeypatch.setattr(deployment, "is_browser", lambda: True)


@pytest.fixture
def on_a_server(monkeypatch):
    monkeypatch.setenv(deployment._OVERRIDE, "local")


@pytest.fixture
def on_a_host(monkeypatch):
    monkeypatch.setenv(deployment._OVERRIDE, "cloud")


class TestHasServerFiles:
    def test_a_local_run_has_them(self, on_a_server):
        assert deployment.has_server_files()

    def test_a_browser_side_copy_has_what_the_visitor_gave_it(self, in_the_browser):
        assert deployment.is_cloud()
        assert deployment.has_server_files()

    def test_a_hosting_service_has_none(self, on_a_host):
        assert not deployment.has_server_files()

    def test_the_loaders_are_not_refused_in_a_browser_but_are_on_a_host(
        self, monkeypatch, in_the_browser
    ):
        shown = []
        monkeypatch.setattr("shiny.ui.modal_show", lambda *a, **k: shown.append(a))
        assert not deployment.refuse_server_mode()
        assert not shown
        monkeypatch.setattr(deployment, "is_browser", lambda: False)
        monkeypatch.setenv(deployment._OVERRIDE, "cloud")
        assert deployment.refuse_server_mode()
        assert shown


class TestSourceModes:
    def test_server_is_the_local_folder_in_a_browser(self, in_the_browser):
        modes = helicon.shiny.source_modes(deployment.is_cloud())
        assert list(modes) == ["server", "upload", "url"]
        assert local_folder.LABEL in str(modes["server"])

    def test_server_is_server_on_a_server(self, on_a_server):
        modes = helicon.shiny.source_modes(deployment.is_cloud())
        assert list(modes) == ["server", "upload", "url"]
        assert "server" in str(modes["server"])
        assert local_folder.LABEL not in str(modes["server"])

    def test_no_server_on_a_host(self, on_a_host):
        assert list(helicon.shiny.source_modes(deployment.is_cloud())) == [
            "upload",
            "url",
        ]


def _choose_buttons(tab):
    return tab.root.find_all(
        lambda n: n.tag == "button" and "helicon-local-folder" in n.classes()
    )


def _server_labels(tab):
    """What each "server" choice of the page's radio groups is called."""
    return [
        n.parent.text()
        for n in tab.root.walk()
        if n.tag == "input" and n.attrs.get("type") == "radio"
        if n.attrs.get("value") == "server"
    ]


def _tab_ui(module, name):
    import importlib

    return getattr(importlib.import_module(f"helicon.webApps.tabs.{module}"), name)


# tab -> (module, UI function, fields that take a path and are on the page at the start)
TABS = {
    "WhereIsMyClass": ("where_is_my_class_tab", "where_is_my_class_tab_ui", 1),
    "HelicalPitch": ("helical_pitch_tab", "helical_pitch_tab_ui", 2),
    "AbInitio3D": ("abinitio3d_ui", "abinitio3d_tab_ui", 2),
    "HI3D": ("hi3d_tab", "hi3d_tab_ui", 1),
    # drawn when their input mode is chosen: only the radio is on the page at the start
    "HelicalProjection": ("helical_projection_tab", "helical_projection_tab_ui", 0),
    "Denovo3D": ("denovo3d_ui", "denovo3d_tab_ui", 0),
    "HILL": ("hill_ui", "hill_tab_ui", 0),
}


@pytest.mark.parametrize("name", TABS)
class TestEveryTabWithFiles:
    def test_the_server_choice_is_called_the_local_folder_in_a_browser(
        self, name, in_the_browser
    ):
        module, function, _ = TABS[name]
        labels = _server_labels(page(_tab_ui(module, function)))
        assert labels and all(label == local_folder.LABEL for label in labels)

    def test_each_path_field_has_the_choose_button_above_it(self, name, in_the_browser):
        module, function, fields = TABS[name]
        tab = page(_tab_ui(module, function))
        buttons = _choose_buttons(tab)
        assert len(buttons) == fields
        for button in buttons:
            row = button.parent.find_all(lambda n: "hfp-field" in n.classes())
            assert len(row) == 1 and button.order < row[0].order

    def test_a_server_shows_none_of_it(self, name, on_a_server):
        module, function, _ = TABS[name]
        tab = page(_tab_ui(module, function))
        assert not _choose_buttons(tab)
        assert all(label == "server" for label in _server_labels(tab))

    def test_a_host_offers_no_server_choice(self, name, on_a_host):
        module, function, _ = TABS[name]
        tab = page(_tab_ui(module, function))
        assert not _choose_buttons(tab)
        assert not _server_labels(tab)


class TestThePathFieldOfADynamicInput:
    """HelicalProjection, Denovo3D and HILL draw their path field when the input mode
    is chosen: it must have the button too."""

    def _field(self):
        from shiny import ui

        return helicon.shiny.file_picker_field(
            ui.input_text("path", "A file"), "browse"
        )

    def test_in_a_browser(self, in_the_browser):
        import htmltools

        from _ui import Page

        html = htmltools.TagList(self._field()).render()["html"]
        tab = Page(html)
        assert len(_choose_buttons(tab)) == 1

    def test_on_a_server(self, on_a_server):
        import htmltools

        from _ui import Page

        html = htmltools.TagList(self._field()).render()["html"]
        assert not _choose_buttons(Page(html))
