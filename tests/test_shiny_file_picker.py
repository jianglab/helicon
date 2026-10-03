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

    def test_a_url_falls_back_to_the_last_folder_used(self, tmp_path):
        job = _tree(tmp_path)
        recent = []
        fp._remember(job, recent)
        assert fp._start_folder(
            "https://ftp.ebi.ac.uk/x/run_it020_data.star", recent
        ) == (job, None)

    def test_recent_folders_are_newest_first_without_repeats(self):
        recent = []
        for f in ("/a", "/b", "/a", "/c"):
            fp._remember(f, recent)
        assert recent == ["/c", "/a", "/b"]

    def test_recent_folders_are_kept_per_session_and_tab(self):
        class Root:
            pass

        class Proxy:
            def __init__(self, root):
                self._root_session = root

        a, b = Root(), Root()
        fp._remember("/data/a", fp.recent_folders(Proxy(a), "hill"))
        assert fp.recent_folders(a, "hill") == ["/data/a"]
        assert fp.recent_folders(Proxy(Proxy(a)), "hill") is fp.recent_folders(
            a, "hill"
        )
        # another session, or another tab of the same one, has its own
        assert fp.recent_folders(b, "hill") == []
        assert fp.recent_folders(a, "abinitio3d") == []

    def test_a_pickers_tab_is_the_first_part_of_its_id(self):
        class Session:
            def __init__(self, prefix):
                self.prefix = prefix

            def ns(self, name):
                return f"{self.prefix}-{name}"

        assert fp.picker_scope(Session("abinitio3d-params_browse")) == "abinitio3d"
        assert fp.picker_scope(Session("browse")) == "browse"

    def test_folders_of_earlier_visits_come_from_the_page(self):
        class Input(dict):
            def __getitem__(self, key):
                value = dict.__getitem__(self, key)
                return lambda: value

        class Root:
            def __init__(self, value=None):
                self.input = Input()
                if value is not None:
                    self.input[fp._BROWSER_RECENT_INPUT] = value

            def root_scope(self):
                return self

        assert fp.browser_folders(Root(), "hill") == []  # not sent yet
        sent = {"hill": ["/d/a", 3, "", "/d/b"], "hi3d": ["/d/c"]}
        assert fp.browser_folders(Root(sent), "hill") == ["/d/a", "/d/b"]
        # each tab its own
        assert fp.browser_folders(Root(sent), "hi3d") == ["/d/c"]
        assert fp.browser_folders(Root(sent), "abinitio3d") == []
        assert fp.browser_folders(Root("not a dict"), "hill") == []
        many = {"hill": [f"/d/{i}" for i in range(20)]}
        assert fp.browser_folders(Root(many), "hill") == many["hill"][: fp._MAX_RECENT]

    def test_this_sessions_folders_come_before_earlier_visits(self):
        assert fp._merged(["/s"], ["/b", "/s", "/c"]) == ["/s", "/b", "/c"]

    def test_the_picker_refuses_on_a_host(self, monkeypatch):
        shown = []
        monkeypatch.setattr(fp.ui, "notification_show", lambda *a, **k: shown.append(a))
        monkeypatch.setenv("HELICON_DEPLOYMENT", "cloud")
        assert fp._server_files_refused()
        assert shown
        monkeypatch.setenv("HELICON_DEPLOYMENT", "local")
        assert not fp._server_files_refused()


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
    # a second place to pick from, for the browser's memory of folders
    (root / "project" / "other").mkdir()
    (root / "project" / "other" / "a_data.star").write_text("x")
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


def _names(page):
    return page.locator(".hfp-row").evaluate_all("rs => rs.map(r => r.dataset.name)")


class TestSortingAndMemory:
    def test_columns_sort_both_ways(self, page, picker_app):
        url, job = picker_app
        _open(page, url)
        page.locator("#browse-all_files").check()  # the .mrcs and .txt files too
        page.wait_for_function(
            "document.querySelectorAll('.hfp-row[data-name=\"note.txt\"]').length === 1"
        )
        size = page.locator("th.hfp-sort[data-sort=size]")
        size.click()  # largest first
        names = _names(page)
        assert names[0] == ".."
        # equal sizes fall back to the names, in the same direction
        mrcs = sorted((n for n in names if n.endswith(".mrcs")), reverse=True)
        assert names[1:5] == mrcs
        assert size.get_attribute("aria-sort") == "descending"
        size.click()  # and the other way
        names = _names(page)
        assert names[0] == ".." and names[-1].endswith(".mrcs")
        assert size.get_attribute("aria-sort") == "ascending"
        name = page.locator("th.hfp-sort[data-sort=name]")
        name.click()
        assert _names(page)[1:3] == ["note.txt", "run_it001_classes.mrcs"]
        name.click()
        assert _names(page)[1] == "run_it025_data.star"
        assert name.get_attribute("aria-sort") == "descending"

    def test_folders_stay_first(self, page, picker_app):
        url, job = picker_app
        _open(page, url)
        page.locator(".hfp-list").focus()
        page.keyboard.press("Backspace")  # to Class2D: two folders
        page.wait_for_function(
            f"document.querySelector('.hfp').dataset.cwd === {str(job.parent)!r}"
        )
        page.locator("th.hfp-sort[data-sort=name]").click()
        assert _names(page) == ["..", "job012", "job010"]

    def test_a_new_visit_opens_where_the_last_one_picked(self, page, picker_app):
        url, job = picker_app
        other = job.parents[1] / "other"
        _open(page, url)
        page.locator(".hfp-path").fill(str(other))
        page.locator(".hfp-path").press("Enter")
        page.wait_for_selector(".hfp-row[data-name='a_data.star']")
        page.locator(".hfp-row[data-name='a_data.star']").dblclick()
        page.wait_for_selector(".modal", state="detached")
        # a new session, with nothing in the field to start from
        page.reload()
        page.wait_for_selector("#browse-open")
        page.locator("#path").fill("")
        page.locator("#path").press("Tab")
        page.locator("#browse-open").click()
        page.wait_for_selector(".hfp-row")
        page.wait_for_function(
            f"document.querySelector('.hfp').dataset.cwd === {str(other)!r}"
        )
        chips = page.locator(".hfp-chip").evaluate_all(
            "cs => cs.map(c => c.getAttribute('title'))"
        )
        assert str(other) in chips


class TestSourceModes:
    def test_server_comes_first_when_local(self):
        modes = helicon.shiny.source_modes(False, ("upload", "url", "emd-xxxxx"))
        assert list(modes) == ["server", "upload", "url", "emd-xxxxx"]

    def test_no_server_on_a_host(self):
        assert list(helicon.shiny.source_modes(True)) == ["upload", "url"]

    def test_labelled_modes(self):
        modes = helicon.shiny.source_modes(False, (("1", "upload"), ("2", "url")))
        assert list(modes) == ["server", "1", "2"] and modes["2"] == "url"
