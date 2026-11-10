"""HI3D completes helical indexing in standalone browser (Shinylive/Pyodide) builds.

Covers the three failure modes that previously stopped the exported HI3D bundle
from showing indexing results:
1. ``trackpy`` has no Pyodide wheel (and no usable PyPI wheel on Python 3.12+),
   so ``find_peaks`` fell back to returning ``(None, None)`` and left the title
   on "Run indexing to see results".
2. ``hi3d_tab_ui`` relied on ``helicon webApps``'s shared page head for BokehJS,
   so a standalone export threw ``ReferenceError: Bokeh is not defined`` on
   every ``bokeh.embed.components`` figure.
3. ``hi3d_indexing_title`` read ``input.hi3d_twist_manual()`` without catching
   ``SilentException`` before ``hi3d_twist_rise_controls`` had round-tripped
   from the browser, blanking the title on the initial fit flush.
"""

import sys

import numpy as np
import pytest


def _synthetic_helical_tube(
    twist_deg: float = -30.0,
    rise_pix: float = 8.0,
    csym: int = 2,
    nz: int = 128,
    nxy: int = 64,
    radius_pix: float = 18.0,
    sigma_pix: float = 2.0,
) -> np.ndarray:
    """3D helical filament of Gaussian subunits with known ``(twist, rise, csym)``."""
    vol = np.zeros((nz, nxy, nxy), dtype=np.float32)
    cx = cy = nxy / 2.0
    hw = int(np.ceil(3.0 * sigma_pix))
    inv_2s2 = 0.5 / (sigma_pix * sigma_pix)
    n_turns = int((nz - 2 * hw) // rise_pix)
    for k in range(n_turns):
        zc = hw + k * rise_pix
        iz0 = max(0, int(np.floor(zc)) - hw)
        iz1 = min(nz, int(np.ceil(zc)) + hw + 1)
        zz = np.arange(iz0, iz1, dtype=np.float32)[:, None, None]
        gz = np.exp(-((zz - zc) ** 2) * inv_2s2)
        for m in range(csym):
            theta = np.deg2rad(k * twist_deg + 360.0 * m / csym)
            xc = cx + radius_pix * np.cos(theta)
            yc = cy + radius_pix * np.sin(theta)
            ix0 = max(0, int(np.floor(xc)) - hw)
            ix1 = min(nxy, int(np.ceil(xc)) + hw + 1)
            iy0 = max(0, int(np.floor(yc)) - hw)
            iy1 = min(nxy, int(np.ceil(yc)) + hw + 1)
            yy = np.arange(iy0, iy1, dtype=np.float32)[None, :, None]
            xx = np.arange(ix0, ix1, dtype=np.float32)[None, None, :]
            vol[iz0:iz1, iy0:iy1, ix0:ix1] += gz * np.exp(
                -((yy - yc) ** 2 + (xx - xc) ** 2) * inv_2s2
            )
    return vol


def _stub_emd_download(monkeypatch, tube: np.ndarray) -> None:
    """Intercept HI3D's default EMDB fetch and metadata lookup cleanly offline."""
    import helicon.lib.dataset as dataset
    import helicon.lib.path_utils as path_utils
    import helicon.webApps.tabs.hi3d_tab as hi3d_mod

    class _DummyResponse:
        headers = {"content-length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1024):
            yield b"stub"

    class _StubEMDB:
        def helical_structure_ids(self):
            return []

        def get_info(self, emd_id):
            return None

    monkeypatch.setattr(path_utils, "_get", lambda *a, **kw: _DummyResponse())
    monkeypatch.setattr(hi3d_mod, "_read_map", lambda path: (tube, 1.0, [1, 2, 3]))
    monkeypatch.setattr(dataset, "EMDB", _StubEMDB)


class TestFindPeaksWithoutTrackpy:
    def test_matches_trackpy_to_machine_precision(self):
        trackpy = pytest.importorskip("trackpy")
        from helicon.webApps.lib.hi3d_core import (
            _locate_2d,
            _refine_com_2d,
            auto_correlation,
            make_square_shape,
        )

        rng = np.random.default_rng(7)
        yy, xx = np.mgrid[:200, :360]
        cp = (
            np.cos(2 * np.pi * (yy / 18.0 + xx / 60.0))
            + 0.6 * np.cos(2 * np.pi * (yy / 18.0 - xx / 120.0))
            + 0.05 * rng.standard_normal((200, 360))
        )
        acf = auto_correlation(make_square_shape(cp), high_pass_fraction=1.0 / 360)
        acf2 = np.hstack((acf[:, -27:], acf, acf[:, :27]))

        f_tp = trackpy.locate(
            acf2,
            diameter=(9, 9),
            minmass=1.0,
            separation=(18, 18),
            characterize=False,
        )
        f_np = _locate_2d(acf2, diameter=(9, 9), minmass=1.0, separation=(18, 18))
        assert len(f_np) == len(f_tp) > 10
        np.testing.assert_allclose(
            f_np[["y", "x", "mass"]].values,
            f_tp[["y", "x", "mass"]].values,
            atol=1e-12,
            rtol=1e-12,
        )

        fr_tp = trackpy.refine_com(
            raw_image=acf2,
            image=acf2,
            radius=(4, 4),
            coords=f_tp,
            characterize=False,
        )
        fr_np = _refine_com_2d(acf2, acf2, radius=(4, 4), coords=f_np)
        np.testing.assert_allclose(
            fr_np[["y", "x", "mass"]].values,
            fr_tp[["y", "x", "mass"]].values,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_indexing_completes_when_trackpy_is_blocked(self, monkeypatch):
        from helicon.webApps.lib.hi3d_core import (
            compute_radial_profile,
            estimate_radial_range,
        )
        from helicon.webApps.tabs.hi3d_tab import _indexing_compute

        monkeypatch.setitem(sys.modules, "trackpy", None)
        tube = _synthetic_helical_tube(twist_deg=-30.0, rise_pix=8.0, csym=2)
        rp = compute_radial_profile(tube)
        r_lo, r_hi = estimate_radial_range(rp, thresh_ratio=0.1)

        res = _indexing_compute(
            data=tube,
            da=1.0,
            dz=1.0,
            apix=1.0,
            rmin=r_lo,
            rmax=r_hi,
            zrange=None,
            sqrt_transform=False,
            acf_2x=False,
            peak_width=9.0,
            peak_height=9.0,
        )

        assert res["fitted"] is True
        assert res["peaks"] is not None and len(res["peaks"]) >= 3
        assert isinstance(res["twist"], float)
        assert isinstance(res["rise"], float)
        assert res["csym"] == 2
        assert res["rise"] == pytest.approx(8.0, abs=0.25)
        assert abs(res["twist"]) == pytest.approx(30.0, abs=1.0)

    def test_blank_acf_returns_none_cleanly_without_trackpy(self, monkeypatch):
        from helicon.webApps.lib.hi3d_core import find_peaks

        monkeypatch.setitem(sys.modules, "trackpy", None)
        peaks, masses = find_peaks(np.zeros((128, 128)), da=1.0, dz=1.0)
        assert peaks is None
        assert masses is None


class TestStandalonePageLoadsBokehAndRendersTitle:
    def test_standalone_hi3d_page_includes_bokeh_scripts_once(self):
        pytest.importorskip("shiny")
        bokeh = pytest.importorskip("bokeh")
        from htmltools import HTMLDocument
        from shiny import ui

        from helicon.webApps import app
        from helicon.webApps.tabs.hi3d_tab import hi3d_tab_ui

        standalone = HTMLDocument(ui.page_fluid(hi3d_tab_ui("hi3d"))).render(
            lib_prefix=None
        )
        bokeh_deps = [
            d for d in standalone["dependencies"] if getattr(d, "name", "") == "bokeh"
        ]
        assert len(bokeh_deps) == 1
        head, body = standalone["html"].split("</head>", 1)
        assert bokeh.__version__ in head
        assert "cdn.bokeh.org" not in standalone["html"]
        assert bokeh.__version__ not in body

        combined = HTMLDocument(
            ui.page_fluid(app._page("Light", "light", "HI3D"))
        ).render(lib_prefix=None)
        combined_bokeh_deps = [
            d for d in combined["dependencies"] if getattr(d, "name", "") == "bokeh"
        ]
        assert len(combined_bokeh_deps) == 1

    def test_title_renders_fitted_values_before_manual_inputs_mount(self, monkeypatch):
        pytest.importorskip("shiny")
        pytest.importorskip("bokeh")
        import json
        import threading
        import time

        import uvicorn
        from shiny import App, ui
        from websockets.sync.client import connect

        import helicon.webApps.tabs.hi3d_tab as hi3d_mod

        monkeypatch.setitem(sys.modules, "trackpy", None)
        tube = _synthetic_helical_tube(twist_deg=-30.0, rise_pix=8.0, csym=2)
        _stub_emd_download(monkeypatch, tube)

        app_ui = ui.page_fluid(hi3d_mod.hi3d_tab_ui("hi3d_tab"))

        def server(input, output, session):
            hi3d_mod.hi3d_tab_server("hi3d_tab")

        srv = uvicorn.Server(
            uvicorn.Config(
                App(app_ui, server), host="127.0.0.1", port=0, log_level="error"
            )
        )
        thread = threading.Thread(target=srv.run, daemon=True)
        thread.start()
        while not srv.started:
            time.sleep(0.02)
        port = srv.servers[0].sockets[0].getsockname()[1]

        # Send static sidebar inputs, but deliberately omit hi3d_twist_manual /
        # hi3d_rise_manual / hi3d_csym_manual (from dynamic hi3d_twist_rise_controls)
        # so input.hi3d_twist_manual() raises SilentException on the initial fit
        # flush, just as in Pyodide before the client mounts those dynamic controls.
        title_html = ""
        try:
            with connect(f"ws://127.0.0.1:{port}/websocket/") as ws:
                ws.send(
                    json.dumps(
                        {
                            "method": "init",
                            "data": {
                                "hi3d_tab-hi3d_input_mode": "emd-xxxxx",
                                "hi3d_tab-hi3d_emd_id": "emd-10499",
                                "hi3d_tab-hi3d_da": 1.0,
                                "hi3d_tab-hi3d_dz": 1.0,
                                "hi3d_tab-hi3d_peak_width": 9.0,
                                "hi3d_tab-hi3d_peak_height": 9.0,
                                "hi3d_tab-hi3d_show_scf": False,
                                "hi3d_tab-hi3d_acf_2x": False,
                                ".clientdata_output_hi3d_tab-hi3d_indexing_title_hidden": False,
                            },
                        }
                    )
                )
                for _ in range(40):
                    msg = json.loads(ws.recv(timeout=10))
                    val = (msg.get("values") or {}).get("hi3d_tab-hi3d_indexing_title")
                    if isinstance(val, dict):
                        html = val.get("html", "")
                        if "twist=" in html:
                            title_html = html
                            break
        finally:
            srv.should_exit = True
            thread.join(timeout=5)

        assert "twist=" in title_html
        assert "rise=" in title_html and "csym=c2" in title_html
        assert "Run indexing to see results" not in title_html

    def test_standalone_hi3d_completes_indexing_in_browser(self, page, monkeypatch):
        pytest.importorskip("shiny")
        pytest.importorskip("bokeh")
        import threading
        import time

        import uvicorn
        from shiny import App, ui

        import helicon.webApps.tabs.hi3d_tab as hi3d_mod

        monkeypatch.setitem(sys.modules, "trackpy", None)
        tube = _synthetic_helical_tube(twist_deg=-30.0, rise_pix=8.0, csym=2)
        _stub_emd_download(monkeypatch, tube)

        app_ui = ui.page_fluid(hi3d_mod.hi3d_tab_ui("hi3d_tab"))

        def server(input, output, session):
            hi3d_mod.hi3d_tab_server("hi3d_tab")

        srv = uvicorn.Server(
            uvicorn.Config(
                App(app_ui, server), host="127.0.0.1", port=0, log_level="error"
            )
        )
        thread = threading.Thread(target=srv.run, daemon=True)
        thread.start()
        while not srv.started:
            time.sleep(0.02)
        port = srv.servers[0].sockets[0].getsockname()[1]
        try:
            page_errors = []
            page.on("pageerror", lambda err: page_errors.append(str(err)))
            page.goto(f"http://127.0.0.1:{port}/")
            title_loc = page.locator("#hi3d_tab-hi3d_indexing_title")
            title_loc.get_by_text("twist=").wait_for(timeout=15_000)
            page.get_by_role("button", name="Refine twist/rise").wait_for(
                timeout=15_000
            )
            page.locator("#hi3d_tab-hi3d_indexing_plot canvas").first.wait_for(
                timeout=15_000
            )
            title_text = title_loc.inner_text()
            assert "Run indexing to see results" not in title_text
            assert "rise=" in title_text and "csym=c2" in title_text
            assert page_errors == []
        finally:
            srv.should_exit = True
            thread.join(timeout=5)
