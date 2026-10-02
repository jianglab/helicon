"""helicon.shiny's file picker: a dialog that browses the server's files."""

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import helicon
from helicon.lib import shiny_file_picker as fp

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _tree(root):
    """A small RELION-like project."""
    job = root / "project" / "Class2D" / "job010"
    job.mkdir(parents=True)
    for i in ("001", "002", "010", "025"):
        (job / f"run_it{i}_data.star").write_text("x")
        (job / f"run_it{i}_classes.mrcs").write_bytes(b"0" * 2048)
    (job / "note.txt").write_text("x")
    (root / "project" / "Class2D" / "job012").mkdir()
    (root / "project" / ".hidden").mkdir()
    return job


class TestListFolder:
    def test_folders_first_in_natural_order_then_matching_files(self, tmp_path):
        job = _tree(tmp_path)
        entries, cut = fp.list_folder(job, ("*.star",))
        assert [e["name"] for e in entries] == [
            "run_it001_data.star",
            "run_it002_data.star",
            "run_it010_data.star",
            "run_it025_data.star",
        ]
        assert cut == 0
        entries, _ = fp.list_folder(job.parent, ("*.star",))
        assert [e["name"] for e in entries] == ["job010", "job012"]

    def test_patterns_ignore_case_and_none_means_all(self, tmp_path):
        (tmp_path / "A.STAR").write_text("x")
        (tmp_path / "b.cs").write_text("x")
        (tmp_path / "c.txt").write_text("x")
        names = lambda pats: [e["name"] for e in fp.list_folder(tmp_path, pats)[0]]
        assert names(("*.star", "*.cs")) == ["A.STAR", "b.cs"]
        assert names(()) == ["A.STAR", "b.cs", "c.txt"]

    def test_hidden_names_on_request(self, tmp_path):
        _tree(tmp_path)
        project = tmp_path / "project"
        assert [e["name"] for e in fp.list_folder(project)[0]] == ["Class2D"]
        shown = [e["name"] for e in fp.list_folder(project, show_hidden=True)[0]]
        assert shown == [".hidden", "Class2D"]

    def test_sizes_and_times_of_files(self, tmp_path):
        job = _tree(tmp_path)
        entries, _ = fp.list_folder(job, ("*.mrcs",))
        assert entries[0]["size"] == 2048 and entries[0]["mtime"] > 0
        assert fp._size_text(2048) == "2.0 KB" and fp._size_text(5) == "5 B"

    def test_a_huge_folder_is_capped(self, tmp_path):
        for i in range(30):
            (tmp_path / f"m{i}.mrcs").write_text("x")
        entries, cut = fp.list_folder(tmp_path, max_entries=10)
        assert len(entries) == 10 and cut == 20

    def test_an_unreadable_folder_raises(self, tmp_path):
        with pytest.raises(OSError):
            fp.list_folder(tmp_path / "missing")


class TestStartFolder:
    def test_a_file_in_the_field_opens_its_folder_selected(self, tmp_path):
        job = _tree(tmp_path)
        folder, name = fp._start_folder(lambda: str(job / "run_it025_data.star"))
        assert folder == job and name == "run_it025_data.star"

    def test_a_folder_or_a_missing_file_in_one(self, tmp_path):
        job = _tree(tmp_path)
        assert fp._start_folder(str(job)) == (job, None)
        assert fp._start_folder(str(job / "gone.star")) == (job, None)

    def test_a_url_falls_back_to_the_last_folder_used(self, tmp_path, monkeypatch):
        job = _tree(tmp_path)
        monkeypatch.setattr(fp, "_RECENT", [])
        fp._remember(job)
        assert fp._start_folder("https://ftp.ebi.ac.uk/x/run_it020_data.star") == (
            job,
            None,
        )

    def test_recent_folders_are_newest_first_without_repeats(self, monkeypatch):
        monkeypatch.setattr(fp, "_RECENT", [])
        for f in ("/a", "/b", "/a", "/c"):
            fp._remember(f)
        assert fp._RECENT == ["/c", "/a", "/b"]


APP = """
from shiny import App, reactive, ui
import helicon

app_ui = ui.page_fluid(
    ui.div(
        ui.input_text("path", "File", value=__START__),
        helicon.shiny.file_picker_button("browse"),
        class_="hfp-field",
    )
)

def server(input, output, session):
    chosen = helicon.shiny.file_picker_server(
        "browse", patterns=("*.star",), title="Pick", start=lambda: input.path()
    )

    @reactive.effect
    @reactive.event(chosen)
    def _():
        ui.update_text("path", value=chosen())

app = App(app_ui, server)
"""


@pytest.fixture(scope="module")
def picker_app(tmp_path_factory):
    root = tmp_path_factory.mktemp("file_picker")
    job = _tree(root)
    folder = root / "app"
    folder.mkdir()
    start = repr(str(job / "run_it025_data.star"))
    (folder / "app.py").write_text(APP.replace("__START__", start))
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
    yield f"http://127.0.0.1:{port[0]}/", job
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _open(page, url):
    page.goto(url)
    page.wait_for_selector("#browse-open")
    page.locator("#browse-open").click()
    page.wait_for_selector(".hfp-row.hfp-on", timeout=15000)


def _cwd(page):
    return page.locator(".hfp").get_attribute("data-cwd")


class TestTheDialog:
    def test_it_opens_at_the_fields_file(self, page, picker_app):
        url, job = picker_app
        _open(page, url)
        assert _cwd(page) == str(job)
        assert page.locator(".hfp-row.hfp-on").get_attribute("data-name") == (
            "run_it025_data.star"
        )
        names = page.locator(".hfp-row").evaluate_all(
            "rs => rs.map(r => r.dataset.name)"
        )
        assert "note.txt" not in names and "run_it001_classes.mrcs" not in names

    def test_typing_filters_and_the_keyboard_chooses(self, page, picker_app):
        url, job = picker_app
        _open(page, url)
        page.locator(".hfp-filter").fill("it00")
        visible = page.locator(".hfp-row:visible").evaluate_all(
            "rs => rs.map(r => r.dataset.name)"
        )
        assert visible == ["..", "run_it001_data.star", "run_it002_data.star"]
        # the first match is selected as you type, and Enter takes it
        assert page.locator(".hfp-row.hfp-on").get_attribute("data-name") == (
            "run_it001_data.star"
        )
        page.locator(".hfp-filter").press("ArrowDown")
        page.locator(".hfp-filter").press("ArrowUp")
        page.locator(".hfp-filter").press("Enter")
        page.wait_for_selector(".modal", state="detached")
        assert page.locator("#path").input_value() == str(job / "run_it001_data.star")

    def test_folders_open_by_double_click_and_backspace_goes_up(self, page, picker_app):
        url, job = picker_app
        _open(page, url)
        page.locator(".hfp-list").focus()
        page.keyboard.press("Backspace")
        page.wait_for_function(
            f"document.querySelector('.hfp').dataset.cwd === {str(job.parent)!r}"
        )
        assert page.locator(".hfp-row.hfp-on").get_attribute("data-name") == "job010"
        page.locator(".hfp-row[data-name=job012]").dblclick()
        page.wait_for_function(
            f"document.querySelector('.hfp').dataset.cwd === "
            f"{str(job.parent / 'job012')!r}"
        )

    def test_a_typed_path_and_the_select_button(self, page, picker_app):
        url, job = picker_app
        _open(page, url)
        page.locator(".hfp-path").fill(str(job / "run_it010_data.star"))
        page.locator(".hfp-path").press("Enter")
        page.wait_for_function(
            "document.querySelector('.hfp-row.hfp-on') && "
            "document.querySelector('.hfp-row.hfp-on').dataset.name === "
            "'run_it010_data.star'"
        )
        assert page.locator(".hfp-choose").is_enabled()
        page.locator(".hfp-choose").click()
        page.wait_for_selector(".modal", state="detached")
        assert page.locator("#path").input_value() == str(job / "run_it010_data.star")


class TestSourceModes:
    def test_server_comes_first_when_local(self):
        modes = helicon.shiny.source_modes(False, ("upload", "url", "emd-xxxxx"))
        assert list(modes) == ["server", "upload", "url", "emd-xxxxx"]

    def test_no_server_on_a_host(self):
        assert list(helicon.shiny.source_modes(True)) == ["upload", "url"]

    def test_labelled_modes(self):
        modes = helicon.shiny.source_modes(False, (("1", "upload"), ("2", "url")))
        assert list(modes) == ["server", "1", "2"] and modes["2"] == "url"
