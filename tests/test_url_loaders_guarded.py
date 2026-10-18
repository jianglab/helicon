"""Every web app handler that fetches a user's URL first refuses what a hosting service must not read."""

import ast
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
