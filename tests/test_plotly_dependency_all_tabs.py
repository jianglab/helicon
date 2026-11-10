"""All Plotly-rendering web app tabs load plotly.js once in <head> and omit inline bundles."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

import helicon


REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "helicon"


class TestPlotlyDependencyAcrossAllTabs:
    def test_no_pio_to_html_inlines_plotlyjs_anywhere_in_src(self):
        offenders = []
        for py_file in sorted(REPO_SRC.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else None
                if name != "to_html":
                    continue
                kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
                inc = kw.get("include_plotlyjs")
                if not (isinstance(inc, ast.Constant) and inc.value is False):
                    offenders.append(f"{py_file.relative_to(REPO_SRC)}:{node.lineno}")
        assert offenders == []

    @pytest.mark.parametrize(
        "module_name",
        [
            "helicon.webApps.tabs.abinitio3d_ui",
            "helicon.webApps.tabs.abinitio3d_tab",
            "helicon.webApps.tabs.denovo3d_ui",
            "helicon.webApps.tabs.denovo3d_tab",
            "helicon.webApps.tabs.helical_pitch_tab",
            "helicon.webApps.tabs.helical_projection_tab",
            "helicon.webApps.tabs.helical_lattice_tab",
        ],
    )
    def test_importing_tab_module_delays_plotly_import(self, module_name):
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys, {module_name}; print('plotly' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        assert out.stdout.strip().splitlines()[-1] == "False"

    def test_helical_lattice_vendor_module_has_no_helicon_imports(self):
        lattice_path = REPO_SRC / "webApps" / "tabs" / "helical_lattice_tab.py"
        tree = ast.parse(lattice_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("helicon")
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0
                assert not (node.module or "").startswith("helicon")

    @pytest.mark.parametrize(
        "module_name, ui_attr",
        [
            ("helicon.webApps.tabs.denovo3d_ui", "denovo3d_tab_ui"),
            ("helicon.webApps.tabs.abinitio3d_ui", "abinitio3d_tab_ui"),
            ("helicon.webApps.tabs.helical_pitch_tab", "helical_pitch_tab_ui"),
            (
                "helicon.webApps.tabs.helical_projection_tab",
                "helical_projection_tab_ui",
            ),
            ("helicon.webApps.tabs.helical_lattice_tab", "helical_lattice_tab_ui"),
        ],
    )
    def test_each_standalone_tab_ui_injects_plotly_once_in_head(
        self, module_name, ui_attr
    ):
        import importlib
        from htmltools import HTMLDocument
        from plotly.offline import get_plotlyjs_version
        from shiny import ui

        mod = importlib.import_module(module_name)
        tab_ui = getattr(mod, ui_attr)
        rendered = HTMLDocument(ui.page_fluid(tab_ui("t"))).render(lib_prefix=None)
        plotly_deps = [
            d for d in rendered["dependencies"] if getattr(d, "name", "") == "plotly"
        ]
        assert len(plotly_deps) == 1
        head, body = rendered["html"].split("</head>", 1)
        assert get_plotlyjs_version() in head
        assert get_plotlyjs_version() not in body

    def test_unified_webapps_page_deduplicates_plotly_across_all_tabs(self):
        from htmltools import HTMLDocument
        from plotly.offline import get_plotlyjs_version
        from helicon.webApps import app
        from helicon.webApps.tabs import denovo3d_ui

        rendered = HTMLDocument(
            app._page("Light", "light", "Home"),
        ).render(lib_prefix=None)
        plotly_deps = [
            d for d in rendered["dependencies"] if getattr(d, "name", "") == "plotly"
        ]
        assert len(plotly_deps) == 1
        head, body = rendered["html"].split("</head>", 1)
        assert get_plotlyjs_version() in head
        assert get_plotlyjs_version() not in body

        # Even when Denovo3D's lazy UI is added alongside app._page, HTMLDependency
        # deduplicates across both helicon.shiny.plotly_dependency and
        # helical_lattice_tab._plotly_dependency.
        rendered_both = HTMLDocument(
            [
                app._page("Light", "light", "Home"),
                denovo3d_ui.denovo3d_tab_ui("denovo3d"),
            ]
        ).render(lib_prefix=None)
        plotly_deps_both = [
            d
            for d in rendered_both["dependencies"]
            if getattr(d, "name", "") == "plotly"
        ]
        assert len(plotly_deps_both) == 1

    @pytest.mark.parametrize(
        "module_name",
        [
            "helicon.webApps.tabs.denovo3d_ui",
            "helicon.webApps.tabs.abinitio3d_ui",
            "helicon.webApps.tabs.helical_pitch_tab",
            "helicon.webApps.tabs.helical_lattice_tab",
        ],
    )
    def test_every_fig_to_html_helper_omits_inline_plotlyjs(self, module_name):
        import importlib
        import plotly.graph_objects as go

        mod = importlib.import_module(module_name)
        fig = go.Figure(go.Scatter(x=[0, 1], y=[1, 2]))
        html = str(mod._fig_to_html(fig))
        assert "Plotly.newPlot" in html
        assert len(html) < 50_000

    def test_shared_plotly_dependency_returns_cached_singleton(self):
        dep = helicon.shiny.plotly_dependency()
        assert dep is helicon.shiny.plotly_dependency()


class TestBokehDependencyAcrossAllTabs:
    def test_no_bokeh_file_html_or_cdn_in_src(self):
        for py_file in sorted(REPO_SRC.rglob("*.py")):
            src = py_file.read_text(encoding="utf-8")
            assert "file_html" not in src, py_file
            assert 'mode="cdn"' not in src, py_file
            assert "mode='cdn'" not in src, py_file

    @pytest.mark.parametrize(
        "module_name",
        [
            "helicon.lib.shiny",
            "helicon.webApps.tabs.hill_bookmark",
            "helicon.webApps.tabs.hill_ui",
            "helicon.webApps.tabs.hi3d_tab",
        ],
    )
    def test_importing_bokeh_tab_modules_delays_bokeh_import(self, module_name):
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys, {module_name}; print('bokeh' in sys.modules)",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        assert out.stdout.strip().splitlines()[-1] == "False"

    @pytest.mark.parametrize(
        "module_name, ui_attr",
        [
            ("helicon.webApps.tabs.hi3d_tab", "hi3d_tab_ui"),
            ("helicon.webApps.tabs.hill_ui", "hill_tab_ui"),
        ],
    )
    def test_each_bokeh_tab_ui_injects_offline_bokeh_once_in_head(
        self, module_name, ui_attr
    ):
        import importlib
        import bokeh
        from htmltools import HTMLDocument
        from shiny import ui

        mod = importlib.import_module(module_name)
        tab_ui = getattr(mod, ui_attr)
        rendered = HTMLDocument(ui.page_fluid(tab_ui("t"))).render(lib_prefix=None)
        bokeh_deps = [
            d for d in rendered["dependencies"] if getattr(d, "name", "") == "bokeh"
        ]
        assert len(bokeh_deps) == 1
        head, body = rendered["html"].split("</head>", 1)
        assert bokeh.__version__ in head
        assert "cdn.bokeh.org" not in rendered["html"]
        assert bokeh.__version__ not in body

    def test_unified_webapps_page_deduplicates_bokeh_across_all_tabs(self):
        import bokeh
        from htmltools import HTMLDocument
        from helicon.webApps import app
        from helicon.webApps.tabs import hill_bookmark, hill_ui

        dep = helicon.shiny.bokeh_dependency()
        assert dep is hill_bookmark.bokeh_dependency()

        rendered = HTMLDocument(
            [
                app._page("Light", "light", "Home"),
                hill_ui.hill_tab_ui("hill"),
            ]
        ).render(lib_prefix=None)
        bokeh_deps = [
            d for d in rendered["dependencies"] if getattr(d, "name", "") == "bokeh"
        ]
        assert len(bokeh_deps) == 1
        head, body = rendered["html"].split("</head>", 1)
        assert bokeh.__version__ in head
        assert "cdn.bokeh.org" not in rendered["html"]
        assert bokeh.__version__ not in body

    def test_hi3d_bokeh_to_html_helper_is_compact_without_inline_bokehjs(self):
        from bokeh.plotting import figure
        from helicon.webApps.tabs import hi3d_tab

        fig = figure(width=200, height=150)
        fig.line([0, 1], [1, 2])
        html = str(hi3d_tab._bokeh_to_html(fig))
        assert "Bokeh.embed.embed_items" in html
        assert len(html) < 50_000
