"""helicon.shiny.background_task: long work that does not block other sessions."""

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

APP = """
import time

from shiny import App, reactive, render, ui

import helicon

app_ui = ui.page_fluid(
    ui.input_numeric("seconds", "Seconds", 3),
    ui.input_checkbox("fail", "Fail", False),
    ui.input_task_button("run", "Run"),
    ui.output_text("result"),
    ui.input_action_button("ping", "Ping"),
    ui.output_text("pong"),
)


def server(input, output, session):
    answer = reactive.value("")

    def work(job, progress):
        for k in range(job["n"]):
            progress.set(k, message=f"step {k}")
            time.sleep(job["seconds"] / job["n"])
        if job["fail"]:
            raise RuntimeError("it broke")
        return job["seconds"] * 2

    def apply(job, result):
        answer.set(f"done {result}")

    def on_error(job, e):
        answer.set(f"error {e} for {job['seconds']}")

    task = helicon.shiny.background_task(
        "run", work, apply, on_error, progress_max=4, session=session
    )

    @reactive.effect
    @reactive.event(input.run)
    def _start():
        task.invoke(dict(seconds=input.seconds(), n=4, fail=input.fail()))

    @render.text
    def result():
        return answer()

    @render.text
    def pong():
        return f"pong {input.ping()}"


app = App(app_ui, server)
"""


@pytest.fixture(scope="module")
def task_app(tmp_path_factory):
    folder = tmp_path_factory.mktemp("background_task")
    (folder / "app.py").write_text(APP)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "shiny",
            "run",
            "--no-dev-mode",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            str(folder / "app.py"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(folder),
    )
    port, output = [], []

    def _reader():
        for line in proc.stdout:
            output.append(line)
            match = re.search(r"Uvicorn running on http://[\d.]+:(\d+)", line)
            if match:
                port.append(int(match.group(1)))

    threading.Thread(target=_reader, daemon=True).start()
    deadline = time.time() + 60
    while not port and time.time() < deadline:
        time.sleep(0.1)
    if not port:
        proc.terminate()
        pytest.fail("server did not start\n" + "".join(output))
    yield f"http://127.0.0.1:{port[0]}/"
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


class TestBackgroundTask:
    def test_other_sessions_stay_responsive_while_it_runs(
        self, page, browser, task_app
    ):
        page.goto(task_app)
        page.wait_for_selector("#pong:has-text('pong 0')")
        page.locator("#run").click()
        # the button shows it is busy, and the progress bar comes up
        page.wait_for_selector("#run[disabled]", timeout=5000)
        page.wait_for_selector(".shiny-progress-notification", timeout=5000)

        # a second visitor is served at once, not after the work
        other = browser.new_page()
        try:
            started = time.time()
            other.goto(task_app)
            other.locator("#ping").click()
            other.wait_for_selector("#pong:has-text('pong 1')", timeout=2500)
            assert time.time() - started < 2.5
        finally:
            other.close()
        # the same session too
        page.locator("#ping").click()
        page.wait_for_selector("#pong:has-text('pong 1')", timeout=2500)
        assert page.locator("#result").inner_text() == ""

        page.wait_for_selector("#result:has-text('done 6')", timeout=15000)
        page.wait_for_selector("#run:not([disabled])", timeout=5000)

    def test_an_error_reaches_the_session_with_its_job(self, page, task_app):
        page.goto(task_app)
        page.locator("#seconds").fill("0.4")
        page.locator("#fail").check()
        page.locator("#run").click()
        page.wait_for_selector(
            "#result:has-text('error it broke for 0.4')", timeout=15000
        )
        page.wait_for_selector("#run:not([disabled])", timeout=5000)
