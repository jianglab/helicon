"""Every web app sidebar scrolls, in `helicon webApps` and in the Shinylive builds.

A bslib sidebar layout scrolls its sidebar only when the layout is bounded by the
window: the page must be fillable, and so must every wrapper between the page and
the layout. A fluid page (or a plain ``div`` around the layout) lets the layout grow
to its content, and then no sidebar scrolls. The test renders the page each bundle
is built from and asks whether it is fillable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "shinylive"))
import build as shinylive_build  # noqa: E402

# Home is not a sidebar tab: it has its own template.
SIDEBAR_BUNDLES = [n for n, b in shinylive_build.BUNDLES.items() if b.is_module]


def _app_ui(bundle):
    """The ``app_ui`` of the app.py that ``build`` writes for *bundle*."""
    module = (
        bundle.module
        if bundle.mode == "package"
        else f"helicon.webApps.tabs.{bundle.module}"
    )
    source = shinylive_build._APP_PY_TEMPLATE.format(
        module=module,
        attr=bundle.module.rsplit(".", 1)[-1],
        title=bundle.title,
        head_import="",
        head_call="",
    )
    namespace = {}
    exec(compile(source, f"<{bundle.name} app.py>", "exec"), namespace)
    return namespace["app_ui"]


class TestShinyliveSidebarsScroll:
    @pytest.mark.parametrize("name", SIDEBAR_BUNDLES)
    def test_the_page_fills_the_window(self, name):
        html = str(_app_ui(shinylive_build.BUNDLES[name]).tagify())
        assert "bslib-page-fill" in html

    def test_every_sidebar_tab_is_covered(self):
        assert len(SIDEBAR_BUNDLES) == 8


class TestHelicalProjectionSidebarScrolls:
    def test_the_layout_sits_in_a_fillable_page(self):
        from helicon.webApps.tabs.helical_projection_tab import (
            helical_projection_tab_ui,
        )

        html = str(helical_projection_tab_ui("hp").tagify())
        assert "bslib-page-fill" in html
        assert html.index("bslib-page-fill") < html.index("bslib-sidebar-layout")
