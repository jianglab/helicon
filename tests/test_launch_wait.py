"""The browser is opened when the web server answers, not when it says it runs."""

import http.server
import socket
import threading
import time

import pytest

from helicon.lib import shiny


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Quiet(http.server.BaseHTTPRequestHandler):
    code = 200

    def do_GET(self):
        self.send_response(self.code)
        self.end_headers()

    def log_message(self, *args):
        pass


def _serve_later(port, delay, code=200):
    """A server that starts listening ``delay`` seconds from now."""
    handler = type("H", (_Quiet,), {"code": code})
    box = {}

    def run():
        time.sleep(delay)
        box["server"] = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
        box["server"].serve_forever()

    threading.Thread(target=run, daemon=True).start()
    return box


class TestWaitUntilServing:
    def test_it_waits_for_a_server_that_starts_late(self):
        port = _free_port()
        box = _serve_later(port, 0.8)
        t0 = time.monotonic()
        assert shiny._wait_until_serving(f"http://localhost:{port}/", interval=0.05)
        assert time.monotonic() - t0 >= 0.7
        box["server"].shutdown()

    def test_an_error_page_is_an_answer(self):
        port = _free_port()
        box = _serve_later(port, 0.0, code=500)
        assert shiny._wait_until_serving(f"http://localhost:{port}/", interval=0.05)
        box["server"].shutdown()

    def test_it_gives_up_when_nothing_comes(self):
        port = _free_port()
        assert not shiny._wait_until_serving(
            f"http://localhost:{port}/", timeout=0.5, interval=0.05
        )

    def test_it_stops_when_the_server_process_has_exited(self):
        class Gone:
            def poll(self):
                return 1

        t0 = time.monotonic()
        assert not shiny._wait_until_serving(
            f"http://localhost:{_free_port()}/", proc=Gone(), timeout=30
        )
        assert time.monotonic() - t0 < 5

    def test_the_query_string_is_not_sent(self):
        port = _free_port()
        box = _serve_later(port, 0.0)
        assert shiny._wait_until_serving(
            f"http://localhost:{port}/?tab=HILL&x=1", interval=0.05
        )
        box["server"].shutdown()


class TestLaunchOpensWhenServing:
    """launch_shiny_app, with a stand-in for the server process that prints its
    'running' line at once and starts answering a second later."""

    @pytest.fixture
    def fake(self, monkeypatch):
        port = _free_port()
        box = _serve_later(port, 1.0)

        class Proc:
            stdout = iter([f"INFO:     Uvicorn running on http://0.0.0.0:{port}\n"])

            def poll(self):
                return None

            def wait(self):
                return 0

        import subprocess

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: Proc())
        yield port, box
        if "server" in box:
            box["server"].shutdown()

    def test_the_browser_is_opened_after_the_server_answers(self, fake, monkeypatch):
        port, box = fake
        opened = []
        monkeypatch.setattr(
            shiny, "_open_browser", lambda url: opened.append(("server" in box, url))
        )
        shiny.launch_shiny_app("x.y:app", block=True, query_params={"tab": "HILL"})
        assert opened and opened[0][0] is True  # the server was already up
        assert opened[0][1].startswith(f"http://localhost:{port}/?")

    def test_the_url_callback_too(self, fake, monkeypatch):
        port, box = fake
        seen = []
        monkeypatch.setattr(shiny, "_open_browser", lambda url: seen.append("opened"))
        shiny.launch_shiny_app(
            "x.y:app", block=True, url_callback=lambda u: seen.append("server" in box)
        )
        assert seen == [True]
