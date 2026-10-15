"""Opening a web app's map in napari or ChimeraX on this computer."""

from types import SimpleNamespace

import mrcfile
import numpy as np
import pytest

from helicon.webApps import map_viewers as mv


@pytest.fixture
def cache(tmp_path, monkeypatch):
    import helicon

    monkeypatch.setattr(helicon, "cache_dir", tmp_path)
    return tmp_path


class TestWriteMap:
    def test_the_map_and_its_voxel_size_are_written(self, cache):
        vol = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        path = mv.write_map(vol, 1.75, "demo")
        assert path.parent == cache / "viewed_maps" and path.name.startswith("demo_")
        with mrcfile.open(path) as m:
            assert np.array_equal(m.data, vol) and float(m.voxel_size.x) == 1.75

    def test_a_float64_map_is_written_as_float32(self, cache):
        path = mv.write_map(np.zeros((2, 2, 2)), 1.0)
        with mrcfile.open(path) as m:
            assert m.data.dtype == np.float32

    def test_two_maps_do_not_overwrite_each_other(self, cache):
        a = mv.write_map(np.zeros((2, 2, 2)), 1.0, "x")
        b = mv.write_map(np.ones((2, 2, 2)), 1.0, "x")
        assert a != b and a.exists() and b.exists()

    def test_maps_older_than_a_day_are_removed(self, cache):
        import os
        import time

        folder = cache / "viewed_maps"
        folder.mkdir()
        old, new = folder / "old.mrc", folder / "new.mrc"
        old.touch()
        new.touch()
        long_ago = time.time() - 3 * 24 * 3600
        os.utime(old, (long_ago, long_ago))
        mv.write_map(np.zeros((2, 2, 2)), 1.0)
        assert not old.exists() and new.exists()


class TestCommands:
    def test_napari_shows_the_map_in_the_volume_view_in_its_own_process(self):
        command = mv.napari_command("/x/m.mrc")
        assert command[1] == "-c" and "_run_standalone" in command[2]
        assert command[-2:] == ["/x/m.mrc", "volume"]

    def test_chimerax_gets_the_file(self, monkeypatch):
        monkeypatch.setattr(mv, "chimerax_path", lambda: "/opt/ChimeraX")
        assert mv.chimerax_command("/x/m.mrc") == ["/opt/ChimeraX", "/x/m.mrc"]

    def test_without_chimerax_there_is_no_command(self, monkeypatch):
        monkeypatch.setattr(mv, "chimerax_path", lambda: None)
        assert mv.chimerax_command("/x/m.mrc") is None


class TestCanLaunch:
    def _session(self, host="localhost"):
        return SimpleNamespace(clientdata=SimpleNamespace(url_hostname=lambda: host))

    def _patch(self, monkeypatch, cloud=False, local=True):
        from helicon.webApps import deployment
        from helicon.webApps.tabs import home_tab

        monkeypatch.setattr(deployment, "is_cloud", lambda: cloud)
        monkeypatch.setattr(home_tab, "client_is_local", lambda s: local)

    def test_on_this_computer_it_may(self, monkeypatch):
        self._patch(monkeypatch)
        assert mv.can_launch(self._session()) is True

    def test_never_on_a_hosting_service(self, monkeypatch):
        self._patch(monkeypatch, cloud=True)
        assert mv.can_launch(self._session()) is False

    def test_never_for_a_visitor_from_elsewhere(self, monkeypatch):
        self._patch(monkeypatch, local=False)
        assert mv.can_launch(self._session()) is False

    def test_not_when_the_page_was_reached_by_another_host_name(self, monkeypatch):
        self._patch(monkeypatch)
        assert mv.can_launch(self._session("lab-server.example.org")) is False


class TestLaunch:
    def test_napari_is_started_detached_with_the_map(self, monkeypatch):
        started = []
        monkeypatch.setattr(mv, "napari_available", lambda: True)
        monkeypatch.setattr(
            mv,
            "_spawn",
            lambda command, log: started.append((command, log)) or object(),
        )
        assert mv.launch("napari", "/x/m.mrc") is None
        ((command, log),) = started
        assert command[-2:] == ["/x/m.mrc", "volume"] and "napari" in log

    def test_chimerax_is_started_with_the_map(self, monkeypatch):
        started = []
        monkeypatch.setattr(mv, "chimerax_path", lambda: "/opt/ChimeraX")
        monkeypatch.setattr(
            mv, "_spawn", lambda command, log: started.append(command) or object()
        )
        assert mv.launch("chimerax", "/x/m.mrc") is None
        assert started == [["/opt/ChimeraX", "/x/m.mrc"]]

    def test_a_missing_program_is_explained_and_nothing_is_started(self, monkeypatch):
        started = []
        monkeypatch.setattr(mv, "_spawn", lambda *a: started.append(a))
        monkeypatch.setattr(mv, "napari_available", lambda: False)
        monkeypatch.setattr(mv, "chimerax_path", lambda: None)
        assert "not installed" in mv.launch("napari", "/x/m.mrc")
        assert "not found" in mv.launch("chimerax", "/x/m.mrc")
        assert not started

    def test_a_start_that_fails_is_reported(self, monkeypatch):
        monkeypatch.setattr(mv, "napari_available", lambda: True)
        monkeypatch.setattr(mv, "_spawn", lambda *a: None)
        assert "could not be started" in mv.launch("napari", "/x/m.mrc")

    def test_an_unknown_viewer_is_refused(self):
        assert "unknown viewer" in mv.launch("paraview", "/x/m.mrc")


class TestButtons:
    def test_none_where_a_viewer_may_not_start(self, monkeypatch):
        monkeypatch.setattr(mv, "can_launch", lambda s: False)
        assert mv.viewer_buttons(object(), "map") == []

    def test_both_buttons_where_both_programs_are_found(self, monkeypatch):
        monkeypatch.setattr(mv, "can_launch", lambda s: True)
        monkeypatch.setattr(mv, "napari_available", lambda: True)
        monkeypatch.setattr(mv, "chimerax_path", lambda: "/opt/ChimeraX")
        html = "".join(str(b) for b in mv.viewer_buttons(object(), "map"))
        assert 'id="map_napari"' in html and 'id="map_chimerax"' in html
        assert "disabled" not in html

    def test_the_napari_button_says_view_in_3d_and_the_other_names_its_program(
        self, monkeypatch
    ):
        monkeypatch.setattr(mv, "can_launch", lambda s: True)
        monkeypatch.setattr(mv, "napari_available", lambda: True)
        monkeypatch.setattr(mv, "chimerax_path", lambda: "/opt/ChimeraX")
        html = "".join(str(b) for b in mv.viewer_buttons(object(), "map"))
        assert "View in 3D" in html and "View in ChimeraX" in html
        assert "View in napari" not in html

    def test_a_missing_program_has_a_disabled_button_that_says_why(self, monkeypatch):
        monkeypatch.setattr(mv, "can_launch", lambda s: True)
        monkeypatch.setattr(mv, "napari_available", lambda: True)
        monkeypatch.setattr(mv, "chimerax_path", lambda: None)
        html = "".join(str(b) for b in mv.viewer_buttons(object(), "map"))
        assert "disabled" in html and "ChimeraX was not found" in html


class TestButtonRow:
    def test_the_download_and_the_viewers_are_in_one_row(self, monkeypatch):
        monkeypatch.setattr(mv, "can_launch", lambda s: True)
        monkeypatch.setattr(mv, "napari_available", lambda: True)
        monkeypatch.setattr(mv, "chimerax_path", lambda: "/opt/ChimeraX")
        row = str(mv.button_row('<a id="dl">Download</a>', object(), "map"))
        assert row.startswith('<div style="display: flex; flex-wrap: wrap')
        order = [
            row.index(x) for x in ('id="dl"', "map_web", "map_napari", "map_chimerax")
        ]
        assert order == sorted(order)

    def test_where_no_viewer_may_start_the_row_holds_only_the_download(
        self, monkeypatch
    ):
        monkeypatch.setattr(mv, "can_launch", lambda s: False)
        row = str(mv.button_row('<a id="dl">Download</a>', object(), "map"))
        assert 'id="dl"' in row and "map_napari" not in row

    def test_the_row_can_be_centred(self, monkeypatch):
        monkeypatch.setattr(mv, "can_launch", lambda s: False)
        row = str(mv.button_row("x", object(), "map", justify="center"))
        assert "justify-content: center" in row


class TestTabs:
    def _src(self, name):
        import inspect

        from helicon.webApps.tabs import abinitio3d_tab, denovo3d_tab

        return inspect.getsource({"a": abinitio3d_tab, "d": denovo3d_tab}[name])

    def test_abinitio3d_offers_both_of_its_maps(self):
        src = self._src("a")
        for prefix in ("map_view", "relion_view"):
            assert f'"{prefix}",' in src[src.index("map_viewers.register(") :]
            assert f'session,\n                "{prefix}"' in src or (
                f'session,\n            "{prefix}"' in src
            )
        assert src.count("map_viewers.button_row(") == 2

    def test_denovo3d_offers_the_map_it_downloads(self):
        src = self._src("d")
        assert '"dn_map_view",' in src and "map_viewers.button_row(" in src
        # the same map as the download, not another computation
        assert src.count("_dn_map_for_download()") >= 2


class TestViewInBrowser:
    """Mol* in a new tab, loaded only when asked for."""

    def test_the_button_is_in_every_row_and_says_what_it_does(self, monkeypatch):
        monkeypatch.setattr(mv, "can_launch", lambda s: False)
        row = str(mv.button_row("x", object(), "map"))
        assert 'id="map_web"' in row and "View in browser" in row
        assert "new browser tab" in row

    def test_nothing_of_molstar_is_loaded_with_the_button_or_the_row(self, monkeypatch):
        monkeypatch.setattr(mv, "can_launch", lambda s: True)
        monkeypatch.setattr(mv, "napari_available", lambda: True)
        monkeypatch.setattr(mv, "chimerax_path", lambda: "/opt/ChimeraX")
        row = str(mv.button_row("x", object(), "map"))
        assert (
            "jsdelivr" not in row
            and "molstar.js" not in row
            and "molstar.css" not in row
        )

    def test_the_click_opens_the_tab_itself_so_it_is_not_blocked(self):
        assert "window.open('about:blank', '_blank')" in mv._OPEN_TAB_JS
        # the server then sends the tab to the page
        assert "helicon_open_map" in mv._TAB_HANDLER_JS
        assert "tab.location.href" in mv._TAB_HANDLER_JS

    def test_a_failed_preparation_closes_the_blank_tab(self):
        assert "msg.error" in mv._TAB_HANDLER_JS and "tab.close()" in mv._TAB_HANDLER_JS

    def test_the_page_loads_a_pinned_molstar_and_the_map_given_to_it(self):
        page = mv.molstar_page()
        assert f"molstar@{mv.MOLSTAR_VERSION}/build/viewer/molstar.js" in page
        assert "molstar.css" in page
        assert "loadVolumeFromUrl" in page and "format: 'ccp4'" in page
        assert "q.get('map')" in page  # the map's address comes with the request
        # no PDB download form on the left: the map is what the tab is for
        assert "layoutShowLeftPanel: false" in page

    def test_the_map_is_served_to_the_session_once_it_is_prepared(self):
        state = {"map": None}
        serve_map, _ = mv.web_routes(state)
        assert serve_map(None).status_code == 404
        state["map"] = mv.mrc_bytes(np.ones((2, 3, 4)), 2.0)
        r = serve_map(None)
        assert r.status_code == 200 and r.body == state["map"]
        assert r.headers["cache-control"] == "no-store"

    def test_the_page_is_html_and_not_cached(self):
        _, serve_page = mv.web_routes({"map": None})
        r = serve_page(None)
        assert r.status_code == 200 and r.media_type == "text/html"
        assert r.headers["cache-control"] == "no-store"

    def test_the_map_bytes_are_an_mrc_file_with_the_voxel_size(self, tmp_path):
        data = mv.mrc_bytes(np.arange(24, dtype=np.float64).reshape(2, 3, 4), 1.5)
        path = tmp_path / "m.mrc"
        path.write_bytes(data)
        with mrcfile.open(path) as m:
            assert m.data.shape == (2, 3, 4) and m.data.dtype == np.float32
            assert float(m.voxel_size.x) == 1.5


class TestButtonStyle:
    def _row(self, monkeypatch, **kw):
        monkeypatch.setattr(mv, "can_launch", lambda s: True)
        monkeypatch.setattr(mv, "napari_available", lambda: True)
        monkeypatch.setattr(mv, "chimerax_path", lambda: "/opt/ChimeraX")
        return str(mv.button_row("x", object(), "map", **kw))

    def test_the_buttons_look_like_the_download_button_beside_them(self, monkeypatch):
        blue = self._row(monkeypatch, button_class="btn-primary")
        assert blue.count("btn-primary") == 3 and "btn-outline" not in blue
        plain = self._row(monkeypatch)
        assert "btn-primary" not in plain and "btn-outline" not in plain

    def test_the_denovo3d_row_is_blue_like_its_download_button(self):
        import inspect

        from helicon.webApps.tabs import denovo3d_tab

        src = inspect.getsource(denovo3d_tab)
        row = src[src.index("map_viewers.button_row(") :][:700]
        assert 'class_="btn btn-primary shiny-download-link"' in row
        assert 'button_class="btn-primary"' in row


class TestDownloadLink:
    def test_it_is_a_blue_download_link_to_the_handler(self):
        session = SimpleNamespace(ns=lambda name: f"mod-{name}")
        html = str(mv.download_link(session, "download_map", "Download map (.mrc)"))
        assert 'id="mod-download_map"' in html
        assert "btn btn-primary shiny-download-link" in html
        assert "Download map (.mrc)" in html

    def test_another_style_can_be_asked_for(self):
        session = SimpleNamespace(ns=lambda name: name)
        html = str(mv.download_link(session, "x", "X", button_class="btn-default"))
        assert (
            "btn btn-default shiny-download-link" in html and "btn-primary" not in html
        )


class TestAbinitio3dButtonsAreBlue:
    def _src(self):
        import inspect

        from helicon.webApps.tabs import abinitio3d_tab

        return inspect.getsource(abinitio3d_tab)

    def test_the_two_map_rows_are_blue_downloads_included(self):
        src = self._src()
        assert src.count('button_class="btn-primary"') == 2
        assert '"download_map", "Download map (.mrc)"' in src
        assert '"download_relion_map", "Download map (.mrc)"' in src.replace(
            "\n", " "
        ).replace("  ", "")

    def test_the_download_filaments_button_is_blue_too(self):
        src = self._src()
        band = src[
            src.index("def pitch_band_download") : src.index("def _imposed_csym")
        ]
        assert "map_viewers.download_link(" in band and '"download_pitch_band"' in band
        # its handler is still the one registered by decorating it
        assert "def download_pitch_band():" in band


class TestManyMapsAtOnce:
    def test_maps_written_in_the_same_instant_get_different_names(
        self, cache, monkeypatch
    ):
        # a frozen clock: the second and the millisecond are the same for all
        monkeypatch.setattr(mv.time, "time", lambda: 1_700_000_000.123)
        paths = [mv.write_map(np.full((2, 2, 2), i), 1.0, "x") for i in range(20)]
        assert len({p.name for p in paths}) == 20
        assert all(p.exists() for p in paths)
