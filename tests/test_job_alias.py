"""The aliases of RELION jobs, and the bookmarks that have names: in the files, the sidebar, the list."""

import os

import pytest
from PySide6.QtWidgets import QApplication

from helicon.lib.gui import browser_parts as bp
from helicon.lib.gui import job_alias as ja
from _settings import own_settings as _own_settings

PIPELINE = """
# version 30001

data_pipeline_general

_rlnPipeLineJobCounter                       12

# version 30001

data_pipeline_processes

loop_
_rlnPipeLineProcessName #1
_rlnPipeLineProcessAlias #2
_rlnPipeLineProcessTypeLabel #3
_rlnPipeLineProcessStatusLabel #4
Class2D/job010/       None relion.class2d  Succeeded
Class2D/job011/       None relion.class2d  Succeeded
Extract/job005/       None relion.extract  Succeeded

"""


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_settings(tmp_path, monkeypatch, qapp):
    _own_settings(monkeypatch, tmp_path / "settings.ini")


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    for job in ("Class2D/job010", "Class2D/job011", "Extract/job005"):
        (root / job).mkdir(parents=True)
        (root / job / "RELION_JOB_EXIT_SUCCESS").write_text("")
    (root / "default_pipeline.star").write_text(PIPELINE)
    return root


def _alias_in_pipeline(root, process):
    for line in (root / "default_pipeline.star").read_text().splitlines():
        if line.startswith(process):
            return line.split()[1]


class TestSettingAliases:
    def test_an_alias_is_a_link_to_the_job_and_a_line_of_the_pipeline(self, project):
        ja.set_alias(project / "Class2D/job010", "thin_filaments")
        link = project / "Class2D" / "thin_filaments"
        assert link.is_symlink() and os.readlink(link) == "../Class2D/job010/"
        assert link.resolve() == (project / "Class2D/job010").resolve()
        assert (
            _alias_in_pipeline(project, "Class2D/job010/") == "Class2D/thin_filaments/"
        )
        assert (
            _alias_in_pipeline(project, "Class2D/job011/") == "None"
        )  # the others stay

    def test_it_is_read_from_the_links(self, project):
        ja.set_alias(project / "Class2D/job010", "b_alias")
        ja.set_alias(project / "Class2D/job011", "a_alias")
        assert ja.aliases(project / "Class2D") == {
            "job010": ["b_alias"],
            "job011": ["a_alias"],
        }
        assert ja.aliases(project / "Extract") == {}
        assert ja.aliases(project / "missing") == {}

    def test_changing_the_alias_takes_the_old_link_away(self, project):
        job = project / "Class2D/job010"
        ja.set_alias(job, "first")
        ja.set_alias(job, "second")
        assert not (project / "Class2D" / "first").exists()
        assert (project / "Class2D" / "second").is_symlink()
        assert _alias_in_pipeline(project, "Class2D/job010/") == "Class2D/second/"

    def test_removing_it_takes_the_link_and_the_line_away(self, project):
        job = project / "Class2D/job010"
        ja.set_alias(job, "gone_soon")
        ja.set_alias(job, None)
        assert ja.aliases(project / "Class2D") == {}
        assert _alias_in_pipeline(project, "Class2D/job010/") == "None"
        ja.set_alias(job, None)  # no alias: nothing to do, and no failure

    def test_the_same_alias_can_be_set_again(self, project):
        job = project / "Class2D/job010"
        ja.set_alias(job, "same")
        ja.set_alias(job, "same")
        assert (project / "Class2D" / "same").is_symlink()

    @pytest.mark.parametrize(
        "bad", ["", "has space", "a/b", ".hidden", "job042", "tab\tin"]
    )
    def test_a_name_that_cannot_be_an_alias_is_refused(self, project, bad):
        with pytest.raises(ValueError):
            ja.set_alias(project / "Class2D/job010", bad)
        assert ja.aliases(project / "Class2D") == {}

    def test_an_alias_that_names_another_job_or_folder_is_refused(self, project):
        (project / "Class2D" / "notes").mkdir()
        with pytest.raises(ValueError, match="in use"):
            ja.set_alias(project / "Class2D/job010", "notes")
        ja.set_alias(project / "Class2D/job011", "taken")
        with pytest.raises(ValueError, match="in use"):
            ja.set_alias(project / "Class2D/job010", "taken")

    def test_a_folder_that_is_not_a_job_is_refused(self, project):
        with pytest.raises(ValueError, match="not a job"):
            ja.set_alias(project / "Class2D", "x")

    def test_a_project_that_RELION_is_writing_is_left_alone(self, project):
        (project / ".relion_lock").mkdir()
        with pytest.raises(OSError, match="RELION"):
            ja.set_alias(project / "Class2D/job010", "later")
        assert ja.aliases(project / "Class2D") == {}
        assert (project / ".relion_lock").is_dir()  # it was not ours to take away

    def test_the_lock_is_given_back(self, project):
        ja.set_alias(project / "Class2D/job010", "ok")
        assert not (project / ".relion_lock").exists()

    def test_without_a_pipeline_only_the_link_is_made(self, project):
        (project / "default_pipeline.star").unlink()
        ja.set_alias(project / "Class2D/job010", "no_pipeline")
        assert (project / "Class2D" / "no_pipeline").is_symlink()

    def test_the_pipeline_keeps_its_other_lines_and_mode(self, project):
        path = project / "default_pipeline.star"
        os.chmod(path, 0o640)
        before = path.read_text()
        ja.set_alias(project / "Extract/job005", "mine")
        after = path.read_text().splitlines()
        changed = [(a, b) for a, b in zip(before.splitlines(), after) if a != b]
        assert len(changed) == 1 and "Extract/mine/" in changed[0][1]
        assert (path.stat().st_mode & 0o777) == 0o640

    def test_links_that_do_not_point_to_a_job_are_not_aliases(self, project):
        os.symlink("elsewhere", project / "Class2D" / "to_elsewhere")
        os.symlink("../Extract/job005/", project / "Class2D" / "to_another_type")
        os.symlink("job010", project / "Class2D" / "short_form")
        assert ja.aliases(project / "Class2D") == {"job010": ["short_form"]}


class TestAliasesInTheSidebar:
    def test_a_job_with_an_alias_is_shown_as_the_job_and_its_alias(self, qapp, project):
        ja.set_alias(project / "Class2D/job010", "thin")
        side = bp.PlacesSidebar()
        side.refresh(str(project / "Class2D/job010"))
        side.wait()
        node = side.topLevelItem(1)
        types = {
            node.child(i).text(0): node.child(i) for i in range(1, node.childCount())
        }
        labels = [
            types["Class2D"].child(i).text(0)
            for i in range(types["Class2D"].childCount())
        ]
        assert labels == ["job011", "job010(thin)"]  # one row for the job and its alias
        assert "alias thin" in types["Class2D"].child(1).toolTip(0)

    def test_the_menu_of_a_job_sets_edits_and_removes_its_alias(
        self, qapp, project, monkeypatch
    ):
        side = bp.PlacesSidebar()
        side.refresh(str(project / "Class2D/job010"))
        side.wait()
        got = []
        side.alias_requested.connect(lambda path, alias: got.append((path, alias)))
        from PySide6.QtWidgets import QInputDialog

        node = side.topLevelItem(1)
        job = next(
            node.child(i).child(j)
            for i in range(1, node.childCount())
            for j in range(node.child(i).childCount())
            if node.child(i).child(j).text(0) == "job010"
        )
        monkeypatch.setattr(side, "itemAt", lambda pos: job)
        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, **k: ("pretty", True))
        )
        picked = {}

        def choose(menu, wanted):
            picked["texts"] = [a.text() for a in menu.actions()]
            return next(a for a in menu.actions() if a.text() == wanted)

        monkeypatch.setattr(
            side, "_run_menu", lambda menu, pos: choose(menu, "Add alias…")
        )
        side._menu(side.rect().center())
        assert got == [(str(project / "Class2D/job010"), "pretty")]
        assert picked["texts"] == ["Add alias…"]
        job.setData(0, bp.PlacesSidebar._ROLE_ALIAS, "pretty")
        monkeypatch.setattr(
            side, "_run_menu", lambda menu, pos: choose(menu, "Remove alias")
        )
        side._menu(side.rect().center())
        assert got[-1] == (str(project / "Class2D/job010"), "")
        assert picked["texts"] == ["Edit alias…", "Remove alias"]


class TestAliasesInTheList:
    def test_the_job_and_its_alias_are_one_row(self, qapp, project):
        from helicon.lib.gui.browser_model import FileBrowserModel

        ja.set_alias(project / "Class2D/job010", "thin")
        model = FileBrowserModel(str(project / "Class2D"))
        for _ in range(50):
            qapp.processEvents()
        names = [model.item(r, 0).text() for r in range(model.rowCount())]
        assert names == ["job010(thin)/", "job011/"]
        path = model.item(0, 0).data(bp.Qt.ItemDataRole.UserRole)
        assert path == str(project / "Class2D" / "job010")

    def test_a_link_to_something_else_is_still_a_row_of_its_own(self, qapp, project):
        from helicon.lib.gui.browser_model import FileBrowserModel

        os.symlink("../Extract/job005/", project / "Class2D" / "other_type")
        model = FileBrowserModel(str(project / "Class2D"))
        names = [model.item(r, 0).text() for r in range(model.rowCount())]
        assert "other_type/" in names and len(names) == 3


class TestNamesOfBookmarks:
    def test_the_default_name_is_the_folder_and_a_name_can_be_given(self, tmp_path):
        from helicon.lib.gui import browser_settings as bs

        folder = tmp_path / "data_set_7"
        folder.mkdir()
        bs._add_bookmark(str(folder))
        key = bs._bookmark_key(str(folder))
        assert bs._bookmark_name(key) == "data_set_7"
        bs._rename_bookmark(key, "the good one")
        assert bs._bookmark_name(key) == "the good one"
        assert bs._get_bookmark_names() == {key: "the good one"}
        assert bs._bookmark_text(key) == f"the good one  ({key})"
        bs._rename_bookmark(key, "")
        assert bs._get_bookmark_names() == {} and bs._bookmark_text(key) == key

    def test_the_name_of_the_folder_is_not_kept_as_a_name(self, tmp_path):
        from helicon.lib.gui import browser_settings as bs

        folder = tmp_path / "plain"
        folder.mkdir()
        bs._add_bookmark(str(folder))
        key = bs._bookmark_key(str(folder))
        bs._rename_bookmark(key, "plain")
        assert bs._get_bookmark_names() == {}

    def test_only_a_bookmark_can_have_a_name_and_it_goes_with_it(self, tmp_path):
        from helicon.lib.gui import browser_settings as bs

        folder = tmp_path / "a"
        folder.mkdir()
        bs._rename_bookmark(str(folder), "x")  # not bookmarked
        assert bs._get_bookmark_names() == {}
        bs._add_bookmark(str(folder))
        key = bs._bookmark_key(str(folder))
        bs._rename_bookmark(key, "mine")
        bs._remove_bookmark(key)
        assert bs._get_bookmark_names() == {}
        bs._add_bookmark(str(folder))
        assert bs._bookmark_name(key) == "a"  # not the old name
        bs._rename_bookmark(key, "again")
        bs._clear_bookmarks()
        assert bs._get_bookmark_names() == {}

    def test_a_damaged_setting_gives_no_names(self):
        from helicon.lib.gui import browser_settings as bs
        from PySide6.QtCore import QSettings

        QSettings("helicon", "display").setValue("bookmark_names", "{not json")
        assert bs._get_bookmark_names() == {}
        QSettings("helicon", "display").setValue("bookmark_names", "[1, 2]")
        assert bs._get_bookmark_names() == {}

    def test_the_sidebar_shows_the_name_and_the_folder_in_the_tooltip(
        self, qapp, tmp_path
    ):
        folder = tmp_path / "far_away_folder"
        folder.mkdir()
        side = bp.PlacesSidebar()
        side.refresh(
            str(tmp_path),
            bookmarks=[str(folder)],
            bookmark_names={str(folder): "the good one"},
        )
        side.wait()
        section = next(
            side.topLevelItem(i)
            for i in range(side.topLevelItemCount())
            if side.topLevelItem(i).text(0) == "BOOKMARKS"
        )
        assert section.child(0).text(0) == "the good one"
        assert str(folder) in section.child(0).toolTip(0)
        side.refresh(str(tmp_path), bookmarks=[str(folder)])  # no name: the folder's
        side.wait()
        section = next(
            side.topLevelItem(i)
            for i in range(side.topLevelItemCount())
            if side.topLevelItem(i).text(0) == "BOOKMARKS"
        )
        assert section.child(0).text(0) == "far_away_folder"

    def test_the_menu_of_a_bookmark_renames_it(self, qapp, tmp_path, monkeypatch):
        folder = tmp_path / "f"
        folder.mkdir()
        side = bp.PlacesSidebar()
        side.refresh(str(tmp_path), bookmarks=[str(folder)])
        side.wait()
        from PySide6.QtWidgets import QInputDialog

        section = next(
            side.topLevelItem(i)
            for i in range(side.topLevelItemCount())
            if side.topLevelItem(i).text(0) == "BOOKMARKS"
        )
        item = section.child(0)
        got = []
        side.bookmark_renamed.connect(lambda p, n: got.append((p, n)))
        monkeypatch.setattr(side, "itemAt", lambda pos: item)
        monkeypatch.setattr(
            QInputDialog, "getText", staticmethod(lambda *a, **k: ("  new name ", True))
        )
        monkeypatch.setattr(
            side,
            "_run_menu",
            lambda menu, pos: next(
                a for a in menu.actions() if a.text().startswith("Rename")
            ),
        )
        side._menu(side.rect().center())
        assert got == [(str(folder), "new name")]

    def test_the_dialog_renames_the_selected_bookmark(self, qapp, tmp_path):
        from helicon.lib.gui import browser_settings as bs

        folder = tmp_path / "dialog_folder"
        folder.mkdir()
        bs._add_bookmark(str(folder))
        dialog = bs._BookmarksDialog()
        dialog._list.setCurrentRow(0)
        dialog.rename_selected("by hand")
        assert (
            dialog._list.item(0).text() == f"by hand  ({bs._bookmark_key(str(folder))})"
        )
        assert dialog._list.item(0).data(
            bp.Qt.ItemDataRole.UserRole
        ) == bs._bookmark_key(str(folder))
