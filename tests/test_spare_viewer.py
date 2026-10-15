"""A napari window process started ahead of the request for a new window."""

import io
import os
import subprocess
import sys
import time

import pytest

from helicon.lib.gui import spare_viewer

# a stand-in for the window process: it records how it was started
FAKE = (
    "import os, sys\n"
    "out = os.environ['SPARE_TEST_OUT']\n"
    "if sys.argv[1] == '--wait':\n"
    "    open(out, 'w').write('spare:' + sys.stdin.readline())\n"
    "else:\n"
    "    open(out, 'w').write('fresh:' + sys.argv[1] + ':' + sys.argv[2])\n"
)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    out = tmp_path / "out.txt"
    monkeypatch.setenv("SPARE_TEST_OUT", str(out))
    monkeypatch.delenv("HELICON_NO_SPARE_VIEWER", raising=False)
    monkeypatch.setattr(spare_viewer, "_COMMAND_FOR", FAKE)
    monkeypatch.setattr(spare_viewer, "_spare", None)
    yield out
    if spare_viewer._spare is not None:
        spare_viewer._spare.kill()


def _wait_for(path, seconds=30):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if path.exists() and path.read_text():
            return path.read_text()
        time.sleep(0.05)
    raise AssertionError("the process wrote nothing")


class TestOpening:
    def test_a_waiting_spare_gets_the_file(self, fake):
        assert spare_viewer.start_spare()
        spare_viewer.open_in_new_process("/data/a.mrc", "volume")
        said = _wait_for(fake)
        assert said.startswith("spare:")
        assert '"/data/a.mrc"' in said and '"volume"' in said

    def test_the_spare_is_used_once(self, fake):
        spare_viewer.start_spare()
        spare_viewer.open_in_new_process("/data/a.mrc")
        assert spare_viewer._spare is None

    def test_without_a_spare_the_window_process_starts_with_the_file(self, fake):
        spare_viewer.open_in_new_process("/data/a.mrc", "volume")
        assert _wait_for(fake) == "fresh:/data/a.mrc:volume"

    def test_a_spare_that_has_died_is_not_used(self, fake):
        spare_viewer.start_spare()
        spare_viewer._spare.kill()
        spare_viewer._spare.wait()
        spare_viewer.open_in_new_process("/data/b.mrc", "slice")
        assert _wait_for(fake) == "fresh:/data/b.mrc:slice"

    def test_a_second_start_keeps_the_waiting_spare(self, fake):
        spare_viewer.start_spare()
        first = spare_viewer._spare
        assert spare_viewer.start_spare()
        assert spare_viewer._spare is first

    def test_a_dead_spare_is_replaced(self, fake):
        spare_viewer.start_spare()
        first = spare_viewer._spare
        first.kill()
        first.wait()
        assert spare_viewer.start_spare()
        assert spare_viewer._spare is not first

    def test_the_switch_turns_the_spares_off(self, fake, monkeypatch):
        monkeypatch.setenv("HELICON_NO_SPARE_VIEWER", "1")
        assert not spare_viewer.enabled()
        assert not spare_viewer.start_spare()
        assert spare_viewer._spare is None

    def test_a_zero_does_not_switch_them_off(self, monkeypatch):
        monkeypatch.setenv("HELICON_NO_SPARE_VIEWER", "0")
        assert spare_viewer.enabled()


class TestReadingTheJob:
    def test_the_path_and_the_mode_are_read(self):
        stream = io.StringIO('{"path": "/x/y.mrcs", "mode": "gallery"}\n')
        assert spare_viewer.read_job(stream) == ("/x/y.mrcs", "gallery")

    def test_an_empty_mode_is_the_default_one(self):
        stream = io.StringIO('{"path": "/x/y.mrcs", "mode": ""}\n')
        assert spare_viewer.read_job(stream) == ("/x/y.mrcs", None)

    def test_a_stream_that_ends_gives_no_job(self):
        assert spare_viewer.read_job(io.StringIO("")) is None

    def test_what_the_browser_writes_is_what_the_spare_reads(self, fake):
        import json

        spare_viewer.start_spare()
        spare_viewer.open_in_new_process("/we ird/é.mrc", "volume")
        line = _wait_for(fake).removeprefix("spare:")
        assert spare_viewer.read_job(io.StringIO(line)) == ("/we ird/é.mrc", "volume")
        assert json.loads(line)["path"] == "/we ird/é.mrc"


class TestTheRealProcess:
    def test_a_spare_whose_browser_goes_away_ends_without_a_window(self, tmp_path):
        pytest.importorskip("napari")
        pytest.importorskip("PySide6")
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                spare_viewer._COMMAND_FOR,
                spare_viewer.WAIT_FLAG,
            ],
            stdin=subprocess.DEVNULL,
            env=env,
            timeout=180,
            capture_output=True,
        )
        assert done.returncode == 0, done.stderr.decode()[-500:]
