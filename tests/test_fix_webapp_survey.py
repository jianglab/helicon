"""Fixes from the web-app survey: sessions kept apart, hosted-copy guards,
map reading, ROI masking and gallery defaults."""

import asyncio
import gzip

import numpy as np
import pytest

from helicon.webApps import deployment


class _Root:
    """Stands in for a Shiny AppSession."""

    def __init__(self, client=("127.0.0.1", 5000), headers=None):
        self.closed = []
        self.http_conn = type(
            "Conn", (), {"client": client, "headers": dict(headers or {})}
        )()

    async def _unhandled_error(self, e):
        self.closed.append(e)


class _Proxy:
    """Stands in for a module's SessionProxy."""

    def __init__(self, root):
        self._root_session = root


class TestErrorModal:
    def test_each_session_gets_its_own_handler_and_modal(self, monkeypatch):
        from helicon.webApps import app

        shown = []
        monkeypatch.setattr(
            app.ui, "modal_show", lambda m, session=None: shown.append(session)
        )
        a, b = _Root(), _Root()
        app.install_error_modal(a)
        app.install_error_modal(_Proxy(b))
        assert "_unhandled_error" in vars(a) and "_unhandled_error" in vars(b)
        assert "_unhandled_error" not in vars(_Root())  # the class is untouched

        err = RuntimeError("boom")
        asyncio.run(b._unhandled_error(err))
        assert shown == [b]
        assert b.closed == [err] and a.closed == []


class TestPublicHosts:
    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1",
            "localhost",
            "::1",
            "[::1]",
            "169.254.169.254",
            "10.1.2.3",
            "192.168.0.1",
            "172.16.0.1",
            "100.64.0.1",
            "0.0.0.0",
            "224.0.0.1",
            "::ffff:127.0.0.1",
            "",
            None,
        ],
    )
    def test_internal_hosts_are_refused(self, host):
        assert not deployment.host_is_public(host)

    def test_public_literal_addresses_pass(self):
        assert deployment.host_is_public("8.8.8.8")
        assert deployment.host_is_public("2001:4860:4860::8888")

    def test_a_name_that_does_not_resolve_is_refused(self, monkeypatch):
        def fail(host, port):
            raise OSError("no such host")

        monkeypatch.setattr(deployment.socket, "getaddrinfo", fail)
        assert not deployment.host_is_public("nowhere.invalid")

    def test_a_name_resolving_to_a_private_address_is_refused(self, monkeypatch):
        monkeypatch.setattr(
            deployment.socket,
            "getaddrinfo",
            lambda host, port: [(2, 1, 6, "", ("10.0.0.5", 0))],
        )
        assert not deployment.host_is_public("intranet.example.org")

    def test_url_allowed_on_a_host(self):
        cloud = {"HELICON_DEPLOYMENT": "cloud"}
        for url in (
            "http://127.0.0.1:8000/x.map",
            "http://localhost/x.map",
            "http://169.254.169.254/latest/meta-data/",
            "https://[::1]/x.star",
            "ftp://10.0.0.1/x.mrcs",
        ):
            assert not deployment.url_allowed(url, cloud), url
        assert deployment.url_allowed("https://8.8.8.8/x.map", cloud)
        local = {"HELICON_DEPLOYMENT": "local"}
        assert deployment.url_allowed("http://127.0.0.1:8000/x.map", local)


class TestRefuseServerMode:
    def test_refused_with_a_popup_only_on_a_host(self, monkeypatch):
        from shiny import ui

        shown = []
        monkeypatch.setattr(ui, "modal_show", lambda m: shown.append(m))
        monkeypatch.setenv("HELICON_DEPLOYMENT", "cloud")
        assert deployment.refuse_server_mode() and len(shown) == 1
        monkeypatch.setenv("HELICON_DEPLOYMENT", "local")
        assert not deployment.refuse_server_mode() and len(shown) == 1

    def test_every_server_mode_loader_checks(self):
        import inspect

        from helicon.webApps.tabs import (
            abinitio3d_tab,
            denovo3d_tab,
            helical_pitch_tab,
            helical_projection_tab,
            hi3d_tab,
            hill_tab,
            where_is_my_class_tab,
        )

        counts = {
            abinitio3d_tab: 3,  # classes, params, relion_reconstruct
            helical_pitch_tab: 2,
            denovo3d_tab: 1,
            helical_projection_tab: 2,
            hill_tab: 1,
            hi3d_tab: 1,
            where_is_my_class_tab: 1,
        }
        for module, n in counts.items():
            src = inspect.getsource(module)
            assert src.count("deployment.refuse_server_mode()") == n, module

    def test_companion_lookups_are_off_on_a_host(self):
        import inspect

        from helicon.webApps.tabs import abinitio3d_tab, helical_pitch_tab

        for module in (abinitio3d_tab, helical_pitch_tab):
            src = inspect.getsource(module)
            body = src[src.index("def _fill_companion") :][:300]
            assert "deployment.is_cloud()" in body


class TestClientIsLocal:
    def test_by_the_connection_address(self):
        from helicon.webApps.tabs.home_tab import client_is_local

        assert client_is_local(_Root(("127.0.0.1", 1)))
        assert client_is_local(_Proxy(_Root(("::1", 1))))
        assert not client_is_local(_Root(("192.168.1.20", 1)))
        assert not client_is_local(_Root(None))
        assert not client_is_local(_Root(("testclient", 1)))
        # a proxy on this machine passing on a remote visitor
        assert not client_is_local(
            _Root(("127.0.0.1", 1), {"x-forwarded-for": "203.0.113.9"})
        )
        assert not client_is_local(object())


class TestHi3dMaps:
    def _write(self, path, data, apix=1.5):
        import mrcfile

        with mrcfile.new(str(path), overwrite=True) as m:
            m.set_data(data)
            m.voxel_size = apix

    def test_a_gz_map_is_read_without_touching_its_folder(self, tmp_path):
        from helicon.webApps.tabs.hi3d_tab import _read_map

        data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        plain = tmp_path / "plain.map"
        self._write(plain, data)
        gz = tmp_path / "emd_1.map.gz"
        gz.write_bytes(gzip.compress(plain.read_bytes()))
        plain.unlink()
        d, apix, crs = _read_map(gz)
        assert np.array_equal(d, data) and apix == pytest.approx(1.5)
        assert crs == [1, 2, 3]
        assert sorted(p.name for p in tmp_path.iterdir()) == ["emd_1.map.gz"]

    def test_an_unreadable_map_raises(self, tmp_path):
        from helicon.webApps.tabs.hi3d_tab import _read_map

        bad = tmp_path / "bad.map"
        bad.write_bytes(b"not a map")
        with pytest.raises(Exception):
            _read_map(bad)
        assert bad.exists()


class TestHi3dAngleRoi:
    def _cp(self, na=360):
        return np.ones((4, na), dtype=np.float32)

    def test_a_plain_range_zeroes_the_columns_outside(self):
        from helicon.webApps.tabs.hi3d_tab import _mask_angle_range

        cp = _mask_angle_range(self._cp(), -90, 90, 1.0)
        assert cp[:, 90:270].all() and not cp[:, :90].any()
        assert not cp[:, 270:].any()

    def test_a_wrapped_range_zeroes_the_columns_between(self):
        from helicon.webApps.tabs.hi3d_tab import _mask_angle_range

        # keep 90..180 and -180..-90; zero -90..90
        cp = _mask_angle_range(self._cp(), 90, -90, 1.0)
        assert cp.shape == (4, 360)
        assert cp[:, :90].all() and cp[:, 270:].all()
        assert not cp[:, 90:270].any()
        assert cp.all(axis=1).sum() == 0 and cp.any(axis=0).sum() == 180

    def test_wrapping_at_the_edge(self):
        from helicon.webApps.tabs.hi3d_tab import _mask_angle_range

        cp = _mask_angle_range(self._cp(), 180, 0, 1.0)  # keep -180..0
        assert cp[:, :180].all() and not cp[:, 180:].any()


class TestClampNumber:
    def test_bounds_and_bad_values(self):
        from helicon.lib.shiny import clamp_number

        assert clamp_number(10**9, 20, 2, 200) == 200
        assert clamp_number(-5, 20, 2, 200) == 2
        assert clamp_number(None, 20, 2, 200) == 20
        assert clamp_number("x", 20, 2, 200) == 20
        assert clamp_number(float("inf"), 20, 2, 200) == 20
        assert clamp_number(float("nan"), 1.5, 0, 5, float) == 1.5
        assert clamp_number(7.9, 1, 0, 5, float) == 5.0

    def test_the_tabs_clamp_their_costly_inputs(self):
        import inspect

        from helicon.webApps.tabs import abinitio3d_tab, helical_pitch_tab

        src = inspect.getsource(abinitio3d_tab)
        assert "clamp_number(input.phase_n_boot(), 20, 2, 200)" in src
        assert "max(2, int(n_boot))" not in src
        src = inspect.getsource(helical_pitch_tab)
        assert "bins=input.bins()" not in src


class TestGalleryDefaults:
    def test_no_reactive_value_defaults(self):
        import inspect

        from shiny import reactive

        from helicon.lib import shiny as hs

        src = inspect.getsource(hs)
        assert "=reactive.value(" not in src
        for p in inspect.signature(hs.image_gallery).parameters.values():
            assert not isinstance(p.default, reactive.Value), p.name

    def test_plain_values_and_defaults(self):
        from helicon.lib.shiny import image_gallery

        assert image_gallery("g") is None
        html = str(image_gallery("g", label="Classes", images=[np.zeros((32, 32))]))
        assert "Classes" in html


class TestBookmarks:
    def test_an_old_show_emdb_bookmark_is_ignored(self):
        from helicon.webApps import app, bookmark
        from helicon.webApps.tabs import denovo3d_tab

        assert "show_emdb" not in denovo3d_tab.BOOKMARK_DEFAULTS
        tab, inputs = bookmark.parse(
            "tab=Denovo3D&show_emdb=1&is_3d=1", app._TAB_MODULE_MAP
        )
        assert tab == "Denovo3D"
        assert "denovo3d-dn_show_emdb_input_mode" not in inputs
        assert inputs["denovo3d-dn_is_3d"] is True
