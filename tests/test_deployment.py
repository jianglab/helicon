"""Local or hosted: features that need the user's files are off on a host."""

import pytest

from helicon.webApps import deployment


class TestHostingService:
    @pytest.mark.parametrize(
        "env, host",
        [
            ({"R_CONFIG_ACTIVE": "connect_cloud"}, "Posit Connect Cloud"),
            ({"QUARTO_PROFILE": "connect_cloud"}, "Posit Connect Cloud"),
            ({"R_CONFIG_ACTIVE": "shinyapps"}, "shinyapps.io"),
            ({"POSIT_PRODUCT": "CONNECT"}, "Posit Connect"),
            ({"RSTUDIO_PRODUCT": "CONNECT"}, "Posit Connect"),
            ({"SHINY_PORT": "3838"}, "Shiny Server"),
            ({"SPACE_ID": "jianglab/helicon"}, "Hugging Face Spaces"),
            ({"K_SERVICE": "helicon"}, "Google Cloud Run"),
            ({"GAE_SERVICE": "default"}, "Google App Engine"),
            ({"AWS_LAMBDA_FUNCTION_NAME": "f"}, "AWS Lambda"),
            ({"ECS_CONTAINER_METADATA_URI_V4": "http://x"}, "AWS ECS"),
            ({"WEBSITE_SITE_NAME": "helicon"}, "Azure App Service"),
            ({"CONTAINER_APP_NAME": "helicon"}, "Azure Container Apps"),
            ({"DYNO": "web.1"}, "Heroku"),
            ({"RENDER_SERVICE_ID": "srv-1"}, "Render"),
            ({"FLY_APP_NAME": "helicon"}, "Fly.io"),
            ({"RAILWAY_PROJECT_ID": "p"}, "Railway"),
        ],
    )
    def test_each_service_is_recognised(self, env, host):
        assert deployment.hosting_service(env) == host
        assert deployment.is_cloud(env)

    def test_a_plain_local_run_is_local(self):
        # what `shiny run` (helicon webApps) itself sets
        env = {"SHINY_BROWSER_HOST": "127.0.0.1", "SHINY_BROWSER_PORT": "8000"}
        assert deployment.hosting_service(env) is None
        assert not deployment.is_cloud(env)

    def test_other_values_of_a_shared_variable_do_not_count(self):
        assert deployment.hosting_service({"R_CONFIG_ACTIVE": "default"}) is None
        assert deployment.hosting_service({"RSTUDIO_PRODUCT": "WORKBENCH"}) is None
        assert deployment.hosting_service({"DYNO": ""}) is None


class TestOverride:
    def test_cloud_can_be_declared(self):
        assert deployment.is_cloud({"HELICON_DEPLOYMENT": "cloud"})

    def test_local_wins_over_a_detected_host(self):
        env = {"HELICON_DEPLOYMENT": "local", "SHINY_PORT": "3838"}
        assert not deployment.is_cloud(env)


class TestBrowser:
    def test_not_browser_on_a_real_interpreter(self):
        assert not deployment.is_browser()
        assert not deployment.is_cloud({})

    def test_pyodide_counts_as_cloud(self, monkeypatch):
        monkeypatch.setattr(deployment.sys, "platform", "emscripten")
        assert deployment.is_browser()
        assert deployment.is_cloud({})

    def test_an_explicit_override_still_wins_in_the_browser(self, monkeypatch):
        monkeypatch.setattr(deployment.sys, "platform", "emscripten")
        assert not deployment.is_cloud({"HELICON_DEPLOYMENT": "local"})


from helicon.webApps.tabs import home_tab  # noqa: E402


class TestTheAppFollowsIt:
    def _page(self, monkeypatch, cloud):
        from shiny import ui

        from helicon.webApps import app

        monkeypatch.setenv("HELICON_DEPLOYMENT", "cloud" if cloud else "local")
        return str(ui.page_fluid(app._page("System", "dark", "Home")))

    def test_on_a_host_whereismyclass_is_there_and_reads_by_url(self, monkeypatch):
        # its micrographs are downloaded from the address of the star file, so it needs
        # nothing on the server; the launcher of the file explorer is the only thing missing
        html = self._page(monkeypatch, cloud=True)
        assert "runs only where your data are" not in html
        assert 'id="lazy_tab_where_is_my_class"' in html
        assert home_tab._DISPLAY_LABEL not in html

    def test_the_url_of_a_micrograph_is_checked_like_any_other_url(self):
        import inspect

        from helicon.webApps.tabs import where_is_my_class_tab

        source = inspect.getsource(where_is_my_class_tab)
        assert "deployment.url_allowed(address)" in source
        assert source.count("deployment.refuse_local_path(") == 1

    def test_locally_everything_is_there(self, monkeypatch):
        html = self._page(monkeypatch, cloud=False)
        assert "runs only where your data are" not in html
        # its page arrives when the tab is first opened: the page holds the place
        assert 'id="lazy_tab_where_is_my_class"' in html
        # the launcher of the file explorer, offered where the files are
        assert home_tab._DISPLAY_LABEL in html

    def test_home_title_bolds_helicon(self):
        from _ui import Page

        doc = Page(str(home_tab.home_tab_ui()))
        (title,) = doc.root.find_all(lambda n: "hh-title" in n.classes())
        assert " ".join(title.text().split()) == "Helicon data processing workflow"
        assert [(c.tag, c.text()) for c in title.children] == [("strong", "Helicon")]
        assert "font-size: inherit" in home_tab._CSS


class TestHomeTooltipsDoNotClip:
    """Bottom-row chips (ProCart, CtfSimulation) sit 36 px above the bottom of
    the SVG, so opening their tooltip below the chip clips it against
    .helicon-home's overflow:auto box (always in standalone page_fluid, and on
    shorter viewports in page_fillable). Every chip's tooltip must flip inside
    both .helicon-home's visible client area and the viewport."""

    def test_every_app_tooltip_stays_unclipped_in_fillable_and_fluid_layouts(
        self, page
    ):
        body = str(home_tab.home_tab_ui())
        for height_css, vp_height in (("height: 100vh;", 800), ("", 900)):
            page.set_viewport_size({"width": 1280, "height": vp_height})
            page.set_content(
                "<!doctype html><html><head><style>"
                "* { font-size: 10pt; padding: 0; border: 0; margin: 0; }"
                f"html, body {{ margin: 0; {height_css} }}"
                f"</style></head><body>{body}</body></html>"
            )
            # * { font-size: 10pt } from app.py must not shrink <strong>Helicon</strong>
            sizes = page.evaluate(
                """() => {
                    const t = document.querySelector('.hh-title');
                    const s = t.querySelector('strong');
                    return [getComputedStyle(t).fontSize, getComputedStyle(s).fontSize];
                }"""
            )
            assert sizes[0] == sizes[1]

            names = [a.name for a in home_tab.HOME_APPS]
            for name in names:
                chip = page.locator(f'.hh-app[data-name="{name}"]')
                chip.scroll_into_view_if_needed()
                chip.hover()
                metrics = page.evaluate(
                    """() => {
                        const home = document.querySelector('.helicon-home');
                        const tip = home.querySelector('.hh-tooltip');
                        const hb = home.getBoundingClientRect();
                        const tb = tip.getBoundingClientRect();
                        const visTop = Math.max(hb.top, 0);
                        const visBottom = Math.min(hb.top + home.clientHeight, window.innerHeight);
                        return {
                            hidden: tip.hidden,
                            tipTop: tb.top,
                            tipBottom: tb.bottom,
                            visTop,
                            visBottom,
                        };
                    }"""
                )
                assert not metrics["hidden"], name
                assert metrics["tipTop"] >= metrics["visTop"] - 0.5, (name, metrics)
                assert metrics["tipBottom"] <= metrics["visBottom"] + 0.5, (
                    name,
                    metrics,
                )


class TestHomeLinksToSiblingBundlesInTheBrowser:
    """Under a Shinylive export, Home has no shared navbar to switch tabs
    within -- each integrated app is its own standalone bundle, so its chip
    must link to that sibling bundle's page instead of a navbar tab."""

    def _tab_app(self):
        return next(a for a in home_tab.HOME_APPS if a.name == "HelicalProjection")

    def test_normally_an_integrated_app_switches_the_navbar_tab(self, monkeypatch):
        monkeypatch.setattr(deployment, "is_browser", lambda: False)
        svg = home_tab._app_svg(self._tab_app())
        assert 'data-tab="HelicalProjection"' in svg
        assert "data-url=" not in svg

    def test_in_the_browser_it_links_to_its_own_sibling_bundle_instead(
        self, monkeypatch
    ):
        monkeypatch.setattr(deployment, "is_browser", lambda: True)
        svg = home_tab._app_svg(self._tab_app())
        assert "data-tab=" not in svg
        assert 'data-url="helical_projection/"' in svg

    def test_every_integrated_app_names_a_real_shinylive_bundle(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "shinylive"))
        import build as shinylive_build

        for app in home_tab.HOME_APPS:
            if app.is_tab:
                assert app.bundle in shinylive_build.BUNDLES, app.name


class TestHomeNeverTriesToLaunchACommandInTheBrowser:
    """ProCart and CtfSimulation have both a command and a hosted url: outside
    the browser, clicking them tries the command first when the browser and
    the server are on the same machine (_HH_LOCAL_HOSTS, read from
    location.hostname). A Shinylive export has no server at all, however --
    including when, as tools/shinylive/README.md has a visitor do, it is
    served from plain localhost to test it -- so its own _js() must keep
    `local` false regardless of hostname, or that click looks like it does
    nothing (home_tab_server's own is_cloud() check declines the launch
    silently)."""

    def test_normally_local_depends_only_on_the_hostname(self, monkeypatch):
        monkeypatch.setattr(deployment, "is_browser", lambda: False)
        js = home_tab._js()
        assert "var _HH_IS_BROWSER = false;" in js
        assert "!_HH_IS_BROWSER && _HH_LOCAL_HOSTS" in js

    def test_in_the_browser_local_is_always_false(self, monkeypatch):
        monkeypatch.setattr(deployment, "is_browser", lambda: True)
        js = home_tab._js()
        assert "var _HH_IS_BROWSER = true;" in js


class TestHostedCopiesReadOnlyByUrl:
    def test_urls_only_on_a_host(self, monkeypatch):
        # no network: every name resolves to a public address
        monkeypatch.setattr(
            deployment.socket,
            "getaddrinfo",
            lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))],
        )
        host = {"HELICON_DEPLOYMENT": "cloud"}
        assert deployment.url_allowed("https://ftp.ebi.ac.uk/x.star", host)
        assert deployment.url_allowed("FTP://example.org/x.mrcs", host)
        assert not deployment.url_allowed("/etc/passwd", host)
        assert not deployment.url_allowed("file:///etc/passwd", host)
        assert not deployment.url_allowed("~/data/run_it025_data.star", host)

    def test_anything_locally(self):
        local = {"HELICON_DEPLOYMENT": "local"}
        assert deployment.url_allowed("/data/run_it025_data.star", local)

    def test_in_the_browser_a_public_url_is_allowed_without_resolving_it(
        self, monkeypatch
    ):
        # host_is_public can't run under Pyodide at all: socket.getaddrinfo
        # has no raw sockets to use and fails shut (OSError), rejecting every
        # URL -- but it also doesn't need to: the fetch runs in the
        # visitor's own browser, not on a shared server, so there is no
        # internal network for host_is_public's SSRF guard to protect
        monkeypatch.setattr(deployment.sys, "platform", "emscripten")

        def _no_sockets(host, port):
            raise OSError("no raw sockets under Pyodide")

        monkeypatch.setattr(deployment.socket, "getaddrinfo", _no_sockets)
        assert deployment.url_allowed(
            "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/"
            "EMPIAR/Class2D/job010/run_it020_classes.mrcs",
            {},
        )
        # the scheme check still applies: not a loophole for a local path
        assert not deployment.url_allowed("/etc/passwd", {})
        assert not deployment.url_allowed("file:///etc/passwd", {})

    def test_every_url_loader_checks(self):
        import inspect

        from helicon.webApps.tabs import (
            abinitio3d_tab,
            denovo3d_tab,
            helical_pitch_tab,
            helical_projection_tab,
            hi3d_tab,
            hill_tab,
        )

        counts = {
            abinitio3d_tab: 2,
            helical_pitch_tab: 2,
            denovo3d_tab: 1,
            helical_projection_tab: 2,
            hill_tab: 1,
            hi3d_tab: 1,
        }
        for module, n in counts.items():
            assert inspect.getsource(module).count("deployment.refuse_local_path(") == n


class TestBookmarksOnAHost:
    def test_a_server_mode_bookmark_falls_back_to_the_default(self, monkeypatch):
        from helicon.webApps import app, bookmark

        query = "tab=AbInitio3D&mode_params=server&server_params=/data/a.star&rise=4.8"
        monkeypatch.setenv("HELICON_DEPLOYMENT", "cloud")
        _, inputs = bookmark.parse(query, app._TAB_MODULE_MAP)
        assert "abinitio3d-input_mode_params" not in inputs
        assert inputs["abinitio3d-rise"] == 4.8
        monkeypatch.setenv("HELICON_DEPLOYMENT", "local")
        _, inputs = bookmark.parse(query, app._TAB_MODULE_MAP)
        assert inputs["abinitio3d-input_mode_params"] == "server"
