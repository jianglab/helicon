"""The check that keeps the names of private data out of the history."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_private_names as cpn  # noqa: E402


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "ok.txt").write_text("nothing here\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "start")
    names = tmp_path.parent / f"{tmp_path.name}-names.txt"
    names.write_text("# the list\nSecretProject\nre:job0\\d\\d\\b  # a job number\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HELICON_PRIVATE_NAMES", str(names))
    return tmp_path


class TestPatterns:
    def test_names_match_without_regard_to_case_and_comments_are_skipped(
        self, tmp_path
    ):
        names = tmp_path / "n.txt"
        names.write_text("# comment\n\nSecretProject  # why\nre:run_it0\\d+\n")
        patterns = cpn.load_patterns(names)
        assert len(patterns) == 2
        hits = cpn.find("see secretproject here\nplain\nrun_it025", patterns)
        assert [h.split("   <- ")[0] for h in hits] == [
            "1: see secretproject here",
            "3: run_it025",
        ]

    def test_a_dot_in_a_name_is_a_dot(self, tmp_path):
        names = tmp_path / "n.txt"
        names.write_text("a.b\n")
        assert cpn.find("aXb", cpn.load_patterns(names)) == []
        assert cpn.find("a.b", cpn.load_patterns(names)) != []

    def test_no_list_of_names_is_no_problem(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HELICON_PRIVATE_NAMES", str(tmp_path / "missing.txt"))
        assert cpn.load_patterns() == []


class TestStaged:
    def test_a_clean_change_passes(self, repo, capsys):
        (repo / "a.py").write_text("x = 1\n")
        _git(repo, "add", "a.py")
        assert cpn.main(["--staged"]) == 0

    def test_an_added_line_that_names_private_data_fails_with_its_file(
        self, repo, capsys
    ):
        (repo / "a.py").write_text("x = 1\n# from SecretProject job012\n")
        _git(repo, "add", "a.py")
        assert cpn.main(["--staged"]) == 1
        err = capsys.readouterr().err
        assert "a.py" in err and "SecretProject" in err

    def test_a_path_that_names_it_fails(self, repo):
        (repo / "secretproject_out").mkdir()
        (repo / "secretproject_out" / "r.txt").write_text("clean\n")
        _git(repo, "add", "secretproject_out")
        assert cpn.main(["--staged"]) == 1

    def test_a_line_that_is_not_added_is_not_checked(self, repo):
        (repo / "old.txt").write_text("SecretProject\n")
        _git(repo, "add", "old.txt")
        _git(repo, "commit", "-q", "-m", "neutral")
        (repo / "old.txt").write_text("SecretProject\nmore\n")
        _git(repo, "add", "old.txt")
        assert (
            cpn.main(["--staged"]) == 0
        )  # (the name is already there; only "more" is new)


class TestMessage:
    def test_the_message_is_checked(self, repo, tmp_path):
        good = tmp_path / "good.txt"
        good.write_text("Add the thing\n")
        bad = tmp_path / "bad.txt"
        bad.write_text("Add the thing for job012\n")
        assert cpn.main(["--message", str(good)]) == 0
        assert cpn.main(["--message", str(bad)]) == 1


class TestRange:
    def test_the_commits_of_a_range_are_checked_message_and_changes(self, repo, capsys):
        (repo / "b.txt").write_text("fine\n")
        _git(repo, "add", "b.txt")
        _git(repo, "commit", "-q", "-m", "fine")
        (repo / "c.txt").write_text("the SecretProject data\n")
        _git(repo, "add", "c.txt")
        _git(repo, "commit", "-q", "-m", "job012 results")
        assert cpn.main(["--range", "HEAD~2..HEAD~1"]) == 0
        assert cpn.main(["--range", "HEAD~1..HEAD"]) == 1
        err = capsys.readouterr().err
        assert "message" in err and "c.txt" in err


def _hits(text):
    return cpn.find(text, cpn.shape_rules())


class TestShapeRules:
    """What identifies private data by its shape, with no list of names."""

    @pytest.mark.parametrize(
        "line",
        [
            "path = '/home/alice/data/set1/job.star'",  # private-ok
            "cd /Users/bob/projects/x",  # private-ok
            "/net/scratch/lab/CS-sample/J298/x.mrc",  # private-ok
            "/scratch/alice/run1",  # private-ok
            "Extract/job670/particles.star",  # private-ok
            "the files of P123/J45",  # private-ok
            "P123 J45",  # private-ok
        ],
    )
    def test_what_identifies_private_data_is_found(self, line):  # private-ok
        assert _hits(line), line

    @pytest.mark.parametrize(
        "line",
        [
            "path = '/home/user/data/x.star'",
            "/home/runner/work/helicon",
            "/home/u/proj/Class2D",
            "/home/$USER/x and /home/<you>/y",
            "Class2D/job010/run_it020_data.star",
            "Extract/job001/mics.star",
            "Refine3D/job123/run_data.star",
            "find_job P1 J7",
            "/mnt/data/page_01.jpg",
            "an ordinary line of code, x = np.zeros(3)",
            "https://ftp.ebi.ac.uk/empiar/10940/Class2D/job210/x.star  EMPIAR public",  # private-ok
        ],
    )
    def test_placeholders_and_examples_pass(self, line):
        assert _hits(line) == [], line

    def test_a_line_that_is_about_it_can_say_so(self):
        assert _hits("a path like /home/alice/x is refused") != []  # private-ok
        assert _hits("a path like /home/alice/x is refused  # private-ok") == []

    def test_the_hit_says_what_it_is(self):
        (hit,) = _hits("/home/alice/x")  # private-ok
        assert hit.startswith("1: /home/alice/x") and hit.endswith(  # private-ok
            "<- a home folder"
        )  # private-ok

    def test_the_identity_of_the_person_running_it_is_found(self, monkeypatch):
        monkeypatch.setattr("getpass.getuser", lambda: "wanda99")
        monkeypatch.setattr("socket.gethostname", lambda: "bigbox7.lab.example")
        monkeypatch.setattr(
            cpn.Path, "home", classmethod(lambda cls: Path("/srv/wanda99"))
        )
        labels = {
            h.split("<- ")[1] for h in _hits("by wanda99 on bigbox7, in /srv/wanda99/x")
        }
        assert labels == {"your user name"}  # the first rule that fits names the line
        assert _hits("ran on bigbox7") and _hits("in /srv/wanda99/x")
        assert _hits("a line about nothing in particular") == []

    def test_short_or_generic_identities_are_not_used(self, monkeypatch):
        monkeypatch.setattr("getpass.getuser", lambda: "root")
        monkeypatch.setattr("socket.gethostname", lambda: "runner")
        monkeypatch.setattr(cpn.Path, "home", classmethod(lambda cls: Path("/root")))
        assert cpn.own_identity() == []


class TestWithoutAnyList:
    """The hooks and CI work for everybody, with no file of names."""

    def test_a_staged_home_folder_fails_and_the_text_says_what(
        self, tmp_path, monkeypatch, capsys
    ):
        _git(tmp_path, "init", "-q", "-b", "main")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HELICON_PRIVATE_NAMES", str(tmp_path / "none.txt"))
        (tmp_path / "a.py").write_text(
            "DATA = '/home/alice/set/x.star'\n"  # private-ok
        )  # private-ok
        _git(tmp_path, "add", "a.py")
        assert cpn.main(["--staged"]) == 1
        err = capsys.readouterr().err
        assert "a.py" in err and "a home folder" in err and "private-ok" in err

    def test_an_audit_of_the_tracked_files_finds_old_lines_too(
        self, tmp_path, monkeypatch, capsys
    ):
        _git(tmp_path, "init", "-q", "-b", "main")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HELICON_PRIVATE_NAMES", str(tmp_path / "none.txt"))
        (tmp_path / "a.py").write_text(
            "x = 1\nDATA = '/scratch/lab/x/y.star'\n"  # private-ok
        )  # private-ok
        _git(tmp_path, "add", "a.py")
        _git(tmp_path, "commit", "-q", "-m", "old")
        assert cpn.main(["--all"]) == 1
        assert "a.py:2:" in capsys.readouterr().err
