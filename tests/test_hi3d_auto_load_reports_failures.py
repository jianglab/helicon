"""HI3D's auto-loaded default map must report a download failure, not hide it.

``_auto_load_default_map`` used to call ``asyncio.to_thread(_download_emd,
...)`` directly and swallow every exception (``except Exception: pass``, "silent
failure on startup"). A slow or flaky connection to EMDB then left the results
pane sat on "Run indexing to see results" forever, with no sign of why --
reported directly by a user who hit exactly that on a real browser build.
Every *other* load path in this tab (the "Change EMDB ID" button, a URL or
server map) already shows a progress bar and a "Download failed" modal on
error, through ``_do_load_emd``/``_run_download``; the automatic first load
should not be the one exception.
"""

import ast
from pathlib import Path

TAB = Path(__file__).resolve().parents[1] / "src/helicon/webApps/tabs/hi3d_tab.py"


def _function(name):
    tree = ast.parse(TAB.read_text())
    (node,) = (
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    return node


def _calls(node):
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return names


class TestAutoLoadDefaultMap:
    def test_it_delegates_to_the_feedback_giving_loader(self):
        calls = _calls(_function("_auto_load_default_map"))
        assert "_do_load_emd" in calls

    def test_it_no_longer_swallows_every_exception_on_its_own(self):
        node = _function("_auto_load_default_map")
        for n in ast.walk(node):
            if isinstance(n, ast.ExceptHandler):
                raise AssertionError(
                    "a bare except here hides a download failure with no "
                    "feedback -- let _do_load_emd's own modal report it instead"
                )

    def test_do_load_emd_shows_a_modal_on_failure(self):
        # the behaviour _auto_load_default_map now relies on, pinned down so a
        # future refactor of _run_download cannot silently drop it either
        calls = _calls(_function("_run_download"))
        assert "modal_show" in calls
