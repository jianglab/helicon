"""Every web app handler that fetches a user's URL first refuses what a hosting service must not read."""

import ast
import sys
from pathlib import Path

import pytest

import helicon

TABS = Path(helicon.__file__).parent / "webApps" / "tabs"
GUARDS = {"refuse_local_path", "url_allowed", "refuse_server_mode"}


def _called(node):
    """Names of the functions called directly in a function (not in the functions inside it)."""
    names, stack = set(), list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(n, ast.Call):
            f = n.func
            names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
        stack.extend(ast.iter_child_nodes(n))
    return names


def _unguarded(tree):
    """Functions that fetch a URL although neither they nor all of their callers check it."""
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    called = {id(f): _called(f) for f in functions}

    def guarded(f, seen=()):
        if called[id(f)] & GUARDS:
            return True
        callers = [
            g for g in functions if f.name in called[id(g)] and g.name not in seen
        ]
        return bool(callers) and all(guarded(g, seen + (f.name,)) for g in callers)

    return [
        f.name
        for f in functions
        if any(
            n.endswith("_from_url") or n == "download_micrograph" for n in called[id(f)]
        )
        and not guarded(f)
    ]


class TestUrlLoadersAreGuarded:
    def test_every_loader_of_a_url_checks_it_first(self):
        found = {
            path.name: _unguarded(ast.parse(path.read_text()))
            for path in sorted(TABS.glob("*_tab.py"))
        }
        assert {name: bad for name, bad in found.items() if bad} == {}

    def test_the_check_notices_a_loader_without_it(self):
        tree = ast.parse(
            "def load(url):\n    return compute.get_images_from_url(url)\n"
        )
        assert _unguarded(tree) == ["load"]


class TestRedirectsOnAHostedCopy:
    def test_a_redirect_to_a_private_address_is_refused(self, monkeypatch):
        from types import SimpleNamespace

        from helicon.lib import path_utils
        from helicon.webApps import deployment

        monkeypatch.setattr(deployment, "is_cloud", lambda e=None: True)
        monkeypatch.setattr(
            deployment, "host_is_public", lambda h: h == "public.example"
        )
        import requests

        seen = []

        def fake_get(url, **kw):
            seen.append(url)
            assert kw["allow_redirects"] is False
            return SimpleNamespace(
                is_redirect=True,
                headers={"location": "http://169.254.169.254/latest"},
                close=lambda: None,
            )

        monkeypatch.setattr(requests, "get", fake_get)
        with pytest.raises(IOError):
            path_utils._get("https://public.example/x")
        assert seen == ["https://public.example/x"]


class TestRequestsPatchedForBrowser:
    """requests makes a real socket connection, which Pyodide has none of;
    _get must route it through pyodide-http's browser-fetch patch instead,
    but only in the browser -- pyodide-http is not one of helicon's own
    dependencies, so outside the browser it must not even be imported."""

    def test_outside_the_browser_nothing_is_imported(self, monkeypatch):
        from helicon.lib import path_utils
        from helicon.webApps import deployment

        monkeypatch.setattr(deployment, "is_browser", lambda: False)
        # import pyodide_http would raise (None is the standard sentinel
        # for "block this import"): reaching the end proves it was never
        # attempted
        monkeypatch.setitem(sys.modules, "pyodide_http", None)
        path_utils._patch_requests_for_browser()

    def test_in_the_browser_pyodide_http_is_patched(self, monkeypatch):
        from types import ModuleType

        from helicon.lib import path_utils
        from helicon.webApps import deployment

        monkeypatch.setattr(deployment, "is_browser", lambda: True)
        calls = []
        fake = ModuleType("pyodide_http")
        fake.patch_all = lambda: calls.append("patched")
        monkeypatch.setitem(sys.modules, "pyodide_http", fake)
        path_utils._patch_requests_for_browser()
        assert calls == ["patched"]
