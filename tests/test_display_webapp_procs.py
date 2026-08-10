"""Unit tests for file-browser web-app instance reuse.

Web-app buttons follow the same convention as napari/gallery:
- ``New`` unchecked → navigate the most recently used alive shiny instance
- ``New`` checked → spawn a coexisting instance (side-by-side comparison)
- ``_terminate_web_apps`` reaps every instance when display exits

Also covers the Opera→xdg-open double-tab browser bug.
"""

import json
import subprocess
import sys
import urllib.request

import pytest

from helicon.commands.display import (
    _WebAppState,
    _WEB_APP_INSTANCES,
    _launch_or_reuse_web_app,
    _terminate_web_apps,
    _web_app_active,
    _web_app_alive,
)


def _dummy_proc():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


@pytest.fixture(autouse=True)
def _clean_registry():
    _WEB_APP_INSTANCES.clear()
    yield
    _terminate_web_apps()


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def test_terminate_web_apps_cleans_up_all():
    for _ in range(3):
        s = _WebAppState(_dummy_proc(), "tok")
        s.base_url = "http://localhost:1/"
        _WEB_APP_INSTANCES.append(s)
    procs = [s.proc for s in list(_WEB_APP_INSTANCES)]
    _terminate_web_apps()
    for proc in procs:
        assert proc.wait(timeout=10) is not None
    assert _WEB_APP_INSTANCES == []


def test_reuse_navigates_most_recent(monkeypatch):
    older = _WebAppState(_dummy_proc(), "tok_old")
    older.base_url = "http://localhost:1111/"
    older.last_used = 1.0
    newer = _WebAppState(_dummy_proc(), "tok_new")
    newer.base_url = "http://localhost:2222/"
    newer.last_used = 2.0
    _WEB_APP_INSTANCES.extend([older, newer])

    calls = []

    def fake_urlopen(req, timeout=10):
        calls.append(req.full_url)
        return _FakeResponse({"ok": True, "alive": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        "helicon.lib.shiny.launch_shiny_app",
        lambda *a, **kw: pytest.fail("must not spawn"),
    )

    _launch_or_reuse_web_app("WhereIsMyClass", {"helicon_tab": '"X"'})

    assert calls == ["http://localhost:2222/helicon/navigate?token=tok_new"]
    assert older.proc.poll() is None
    assert newer.proc.poll() is None


def test_reuse_passes_saved_theme(monkeypatch):
    existing = _WebAppState(_dummy_proc(), "tok_theme")
    existing.base_url = "http://localhost:3333/"
    _WEB_APP_INSTANCES.append(existing)

    calls = []

    def fake_urlopen(req, timeout=10):
        calls.append(json.loads(req.data.decode())["query_params"])
        return _FakeResponse({"ok": True, "alive": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("helicon.commands.display._get_display_theme", lambda: "Light")

    _launch_or_reuse_web_app("WhereIsMyClass", {"helicon_tab": '"X"'})

    assert calls == [{"helicon_tab": '"X"', "helicon_theme": "Light"}]


def test_reuse_falls_back_to_next_instance(monkeypatch):
    dead_url = _WebAppState(_dummy_proc(), "tok_dead")
    dead_url.base_url = "http://localhost:1111/"
    dead_url.last_used = 2.0
    live = _WebAppState(_dummy_proc(), "tok_live")
    live.base_url = "http://localhost:2222/"
    live.last_used = 1.0
    _WEB_APP_INSTANCES.extend([dead_url, live])

    def fake_urlopen(req, timeout=10):
        if "1111" in req.full_url:
            raise OSError("connection refused")
        return _FakeResponse({"ok": True, "alive": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        "helicon.lib.shiny.launch_shiny_app",
        lambda *a, **kw: pytest.fail("must not spawn"),
    )

    _launch_or_reuse_web_app("WhereIsMyClass", {"helicon_tab": '"X"'})

    assert live.last_used >= dead_url.last_used


def test_reuse_spawns_when_all_dead(monkeypatch):
    dead = _dummy_proc()
    dead.terminate()
    dead.wait(timeout=10)
    s = _WebAppState(dead, "tok")
    s.base_url = "http://localhost:1/"
    _WEB_APP_INSTANCES.append(s)

    launched = []

    def fake_launch(
        app_file,
        env=None,
        block=True,
        query_params=None,
        reload=False,
        url_callback=None,
    ):
        launched.append(query_params)
        new_proc = _dummy_proc()
        if url_callback is not None:
            url_callback("http://localhost:9999/?x=1")
        return new_proc

    monkeypatch.setattr("helicon.lib.shiny.launch_shiny_app", fake_launch)
    monkeypatch.setattr("helicon.lib.shiny._open_browser", lambda url: None)

    _launch_or_reuse_web_app("WhereIsMyClass", {"helicon_tab": '"X"'})

    assert len(launched) == 1
    assert len(_web_app_alive()) == 1
    assert _web_app_active().base_url == "http://localhost:9999/"


def test_new_window_coexists_without_killing(monkeypatch):
    existing = _WebAppState(_dummy_proc(), "tok1")
    existing.base_url = "http://localhost:1234/"
    existing.last_used = 1.0
    _WEB_APP_INSTANCES.append(existing)

    def fake_urlopen(req, timeout=10):
        pytest.fail("navigate must not run when new_window=True")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    launched = []

    def fake_launch(
        app_file,
        env=None,
        block=True,
        query_params=None,
        reload=False,
        url_callback=None,
    ):
        launched.append(query_params)
        new_proc = _dummy_proc()
        if url_callback is not None:
            url_callback("http://localhost:9999/?x=1")
        return new_proc

    monkeypatch.setattr("helicon.lib.shiny.launch_shiny_app", fake_launch)
    monkeypatch.setattr("helicon.lib.shiny._open_browser", lambda url: None)

    _launch_or_reuse_web_app("WhereIsMyClass", {"helicon_tab": '"X"'}, new_window=True)

    assert len(launched) == 1
    assert existing.proc.poll() is None  # still alive
    assert len(_web_app_alive()) == 2
    # New instance is most recent
    assert _web_app_active().base_url == "http://localhost:9999/"


def test_after_new_window_reuse_targets_newest(monkeypatch):
    older = _WebAppState(_dummy_proc(), "tok_old")
    older.base_url = "http://localhost:1111/"
    older.last_used = 1.0
    newer = _WebAppState(_dummy_proc(), "tok_new")
    newer.base_url = "http://localhost:2222/"
    newer.last_used = 2.0
    _WEB_APP_INSTANCES.extend([older, newer])

    calls = []

    def fake_urlopen(req, timeout=10):
        calls.append(req.full_url)
        return _FakeResponse({"ok": True, "alive": True})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    _launch_or_reuse_web_app("Denovo3D", {"helicon_tab": '"Denovo3D"'})

    assert calls == ["http://localhost:2222/helicon/navigate?token=tok_new"]
    assert older.proc.poll() is None


def test_fresh_launch_opens_browser(monkeypatch):
    def fake_launch(
        app_file,
        env=None,
        block=True,
        query_params=None,
        reload=False,
        url_callback=None,
    ):
        proc = _dummy_proc()
        from helicon.lib.shiny import encode_query_params

        url = f"http://localhost:9999/?{encode_query_params(query_params)}"
        if url_callback is not None:
            url_callback(url)
        return proc

    monkeypatch.setattr("helicon.lib.shiny.launch_shiny_app", fake_launch)
    opened = []
    monkeypatch.setattr(
        "helicon.lib.shiny._open_browser", lambda url: opened.append(url)
    )

    _launch_or_reuse_web_app(
        "WhereIsMyClass",
        {"_inputs_": "", "helicon_tab": '"WhereIsMyClass"'},
    )

    assert len(opened) == 1
    state = _web_app_active()
    assert state.base_url == "http://localhost:9999/"
    assert f"helicon_token={state.token}" in opened[0]


def test_rapid_two_clicks_do_not_open_two_tabs(monkeypatch):
    """Two clicks before URL resolves: second navigates after first is ready.

    If the first instance has no base_url yet, navigate fails and a second
    spawn happens; once the first's callback fires we still only open one
    browser from the winning path.  Here we simulate both launches and
    only fire the second callback (first never got a URL).
    """
    proc1 = _dummy_proc()
    proc2 = _dummy_proc()
    procs = iter([proc1, proc2])

    def fake_launch(
        app_file,
        env=None,
        block=True,
        query_params=None,
        reload=False,
        url_callback=None,
    ):
        proc = next(procs)
        proc._url_callback = url_callback
        return proc

    monkeypatch.setattr("helicon.lib.shiny.launch_shiny_app", fake_launch)
    opened = []
    monkeypatch.setattr(
        "helicon.lib.shiny._open_browser", lambda url: opened.append(url)
    )

    qp = {"_inputs_": "", "helicon_tab": '"WhereIsMyClass"'}
    _launch_or_reuse_web_app("WhereIsMyClass", qp)
    # First instance has no base_url yet → second click spawns another
    _launch_or_reuse_web_app("WhereIsMyClass", qp)

    assert len(_WEB_APP_INSTANCES) == 2
    # Only fire second callback (as if first was superseded / never ready)
    _WEB_APP_INSTANCES[1].proc._url_callback(
        f"http://localhost:9999/?helicon_token={_WEB_APP_INSTANCES[1].token}"
    )
    assert len(opened) == 1


def test_open_browser_uses_single_launcher(monkeypatch):
    from helicon.lib.shiny import _open_browser

    pops = []

    class FakePopen:
        def __init__(self, *a, **k):
            pops.append((a, k))

    monkeypatch.setattr("subprocess.Popen", FakePopen)
    monkeypatch.setattr(
        "shutil.which",
        lambda name: f"/usr/bin/{name}" if name == "xdg-open" else None,
    )
    monkeypatch.setattr("helicon.lib.shiny._is_wsl", lambda: False)

    _open_browser("http://localhost:48603/?t=1")

    assert len(pops) == 1
    assert pops[0][0][0] == ["/usr/bin/xdg-open", "http://localhost:48603/?t=1"]


def test_terminate_escalates_to_kill(monkeypatch):
    """Hung children that ignore SIGTERM must be kill()'d."""
    from helicon.commands import display as d

    class HungProc:
        def __init__(self):
            self._alive = True
            self.terminate_calls = 0
            self.kill_calls = 0

        def poll(self):
            return None if self._alive else 0

        def terminate(self):
            self.terminate_calls += 1
            # ignore SIGTERM

        def kill(self):
            self.kill_calls += 1
            self._alive = False

        def wait(self, timeout=None):
            if self._alive:
                raise subprocess.TimeoutExpired(cmd="hung", timeout=timeout)
            return 0

    hung = HungProc()
    s = _WebAppState(hung, "tok")
    s.base_url = "http://localhost:1/"
    _WEB_APP_INSTANCES.append(s)

    # Don't wait 2s real time
    monkeypatch.setattr(d.time, "monotonic", lambda: 0.0)

    wait_calls = []

    def fake_wait(timeout=None):
        wait_calls.append(timeout)
        if hung._alive:
            raise subprocess.TimeoutExpired(cmd="hung", timeout=timeout)
        return 0

    hung.wait = fake_wait
    _terminate_web_apps()

    assert hung.terminate_calls == 1
    assert hung.kill_calls == 1
    assert _WEB_APP_INSTANCES == []


def test_make_pdeathsig_preexec_linux():
    from helicon.lib.shiny import _make_pdeathsig_preexec
    import sys

    fn = _make_pdeathsig_preexec()
    if sys.platform == "linux":
        assert callable(fn)
        # Should not raise when invoked in this process
        fn()
    else:
        assert fn is None


def test_launch_shiny_app_passes_preexec(monkeypatch):
    """launch_shiny_app must wire parent-death preexec on Linux."""
    from helicon.lib import shiny as shiny_mod

    captured = {}

    class FakePopen:
        def __init__(self, *a, **k):
            captured.update(k)
            self.stdout = iter([])
            self.returncode = None

        def poll(self):
            return None

        def wait(self):
            return 0

    monkeypatch.setattr(
        (
            shiny_mod.subprocess
            if hasattr(shiny_mod, "subprocess")
            else __import__("subprocess")
        ),
        "Popen",
        FakePopen,
    )
    # Patch at the point of use inside the function
    import subprocess as sp

    monkeypatch.setattr(sp, "Popen", FakePopen)

    # Force a preexec so we can assert it was passed
    sentinel = object()
    monkeypatch.setattr(shiny_mod, "_make_pdeathsig_preexec", lambda: sentinel)

    shiny_mod.launch_shiny_app(
        "helicon.webApps.app:app", block=False, url_callback=lambda u: None
    )

    assert captured.get("preexec_fn") is sentinel
