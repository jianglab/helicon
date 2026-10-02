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


class TestTheAppFollowsIt:
    def _page(self, monkeypatch, cloud):
        from shiny import ui

        from helicon.webApps import app

        monkeypatch.setenv("HELICON_DEPLOYMENT", "cloud" if cloud else "local")
        return str(ui.page_fluid(app._page("System", "dark", "Home")))

    def test_on_a_host_whereismyclass_shows_a_note(self, monkeypatch):
        html = self._page(monkeypatch, cloud=True)
        assert "WhereIsMyClass runs only where your data are" in html
        assert 'id="where_is_my_class-wimc_input_mode' not in html
        assert "Open file browser" not in html

    def test_locally_everything_is_there(self, monkeypatch):
        html = self._page(monkeypatch, cloud=False)
        assert "runs only where your data are" not in html
        assert "where_is_my_class-wimc_input_mode" in html
        assert "Open file browser" in html


class TestHostedCopiesReadOnlyByUrl:
    def test_urls_only_on_a_host(self):
        host = {"HELICON_DEPLOYMENT": "cloud"}
        assert deployment.url_allowed("https://ftp.ebi.ac.uk/x.star", host)
        assert deployment.url_allowed("FTP://example.org/x.mrcs", host)
        assert not deployment.url_allowed("/etc/passwd", host)
        assert not deployment.url_allowed("file:///etc/passwd", host)
        assert not deployment.url_allowed("~/data/run_it025_data.star", host)

    def test_anything_locally(self):
        local = {"HELICON_DEPLOYMENT": "local"}
        assert deployment.url_allowed("/data/run_it025_data.star", local)

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
