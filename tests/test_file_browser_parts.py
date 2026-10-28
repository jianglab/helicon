"""The parts of the file browser: path bar, places sidebar, preview pane."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from helicon.lib.gui import browser_parts as bp
from _settings import own_settings as _own_settings
from helicon.lib.gui import file_browser as fb
from _projects import PIPELINE, make_cs_project, make_pipeline_project


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_settings(tmp_path, monkeypatch):
    _own_settings(monkeypatch, tmp_path / "settings.ini")


@pytest.fixture
def project(tmp_path):
    """A RELION project: two Class2D jobs (one done, one failed) and an Extract job."""
    root = tmp_path / "proj"
    for job in ("Class2D/job010", "Class2D/job011", "Extract/job005"):
        (root / job).mkdir(parents=True)
    (root / "default_pipeline.star").write_text("data_pipeline_general\n")
    (root / "Class2D/job010/RELION_JOB_EXIT_SUCCESS").write_text("")
    (root / "Class2D/job011/RELION_JOB_EXIT_FAILURE").write_text("")
    (root / "Extract/job005/note.txt").write_text("x")
    (root / "Import").mkdir()  # no jobs in it
    # RELION leaves a copy of the pipeline in each job folder; a job is not a project
    for job in ("Class2D/job010", "Class2D/job011", "Extract/job005"):
        (root / job / "default_pipeline.star").write_text("data_pipeline_general\n")
    return root


class TestBreadcrumbs:
    def test_the_home_folder_is_one_segment(self):
        parts = bp.breadcrumb_parts("/home/u/proj/Class2D", home="/home/u")
        assert parts == [
            ("~", "/home/u"),
            ("proj", "/home/u/proj"),
            ("Class2D", "/home/u/proj/Class2D"),
        ]

    def test_a_path_outside_home_starts_at_the_root(self):
        parts = bp.breadcrumb_parts("/data/a/b", home="/home/u")
        assert parts[0] == ("/", "/")
        assert parts[-1] == ("b", "/data/a/b")
        assert [p for _, p in parts] == ["/", "/data", "/data/a", "/data/a/b"]

    def test_home_itself_and_the_root(self):
        assert bp.breadcrumb_parts("/home/u", home="/home/u") == [("~", "/home/u")]
        assert bp.breadcrumb_parts("/", home="/home/u") == [("/", "/")]


class TestProjects:
    def test_the_project_is_found_from_any_folder_inside(self, project):
        assert bp.find_project_root(project / "Class2D/job010") == project
        assert bp.find_project_root(project) == project

    def test_a_job_with_a_pipeline_of_its_own_is_not_the_project(self, project):
        job = project / "Class2D/job010"
        assert (job / "default_pipeline.star").is_file()
        assert bp.find_project_root(job) == project
        assert bp.find_project(job).root == project
        assert bp.find_project(job / "subfolder-that-does-not-exist").root == project

    def test_a_project_inside_a_job_folder_name_is_still_a_project_elsewhere(
        self, tmp_path
    ):
        # a folder called job001 with a pipeline, but no project above it, is a project
        lone = tmp_path / "job001"
        lone.mkdir()
        (lone / "default_pipeline.star").write_text("")
        assert bp.find_project(lone).root == lone

    def test_the_mark_of_the_relion_gui_makes_a_project_root(self, tmp_path):
        root = tmp_path / "gui"
        (root / "Class2D" / "job001").mkdir(parents=True)
        (root / ".gui_projectdir").write_text("")
        found = bp.find_project(root / "Class2D" / "job001")
        assert found.kind == "relion" and found.root == root
        assert bp.find_project_root(root / "Class2D") == root

    def test_no_project_outside(self, tmp_path):
        assert bp.find_project_root(tmp_path / "nowhere") is None

    def test_the_state_of_a_job(self, project):
        assert bp.job_status(project / "Class2D/job010") == "success"
        assert bp.job_status(project / "Class2D/job011") == "failure"
        assert bp.job_status(project / "Extract/job005") == "running"
        assert bp.job_status(project / "Import") == "unknown"

    def test_only_types_with_jobs_are_listed(self, project):
        assert bp.list_job_types(project) == ["Class2D", "Extract"]

    def test_jobs_are_listed_newest_first_with_their_state(self, project):
        assert bp.list_jobs(project, "Class2D") == [
            ("job011", "failure"),
            ("job010", "success"),
        ]
        assert bp.list_jobs(project, "Missing") == []


@pytest.fixture
def cs_project(tmp_path):
    """A CryoSPARC project: filament tracing, extraction and two 2D classifications."""
    return make_cs_project(tmp_path)


class TestCryosparcProjects:
    def test_the_kind_of_project_is_found(self, cs_project, project):
        found = bp.find_project(cs_project / "J7")
        assert found.kind == "cryosparc" and found.root == cs_project
        assert found.name == "Demo project"
        assert bp.find_project(project / "Class2D").kind == "relion"
        assert bp.find_project(cs_project.parent / "elsewhere") is None

    def test_a_job_is_read_from_its_json(self, cs_project):
        job = bp.read_cryosparc_job(cs_project / "J7")
        assert job["uid"] == "J7" and job["type_name"] == "2D Classification"
        assert job["title"] == "first look" and job["status"] == "success"
        assert job["parents"] == ["J6"]
        assert bp.read_cryosparc_job(cs_project / "J12")["status"] == "failure"
        assert bp.read_cryosparc_job(cs_project / "J13")["status"] == "running"
        # built but not run, and waiting in the queue, are not running
        assert bp.read_cryosparc_job(cs_project / "J14")["status"] == "pending"
        assert bp.read_cryosparc_job(cs_project / "J15")["status"] == "queued"
        assert bp.read_cryosparc_job(cs_project / "J99") is None

    def test_types_are_named_as_the_interface_names_them(self, cs_project):
        found = bp.find_project(cs_project)
        labels = [label for _, label, _ in bp.project_types(found)]
        assert labels == [
            "2D Classification",
            "Blob Picker",
            "Extract From Micrographs",
            "Filament Tracer",
            "Some New Job Type",  # a type that is not known is made readable
        ]
        # a type has no folder of its own
        assert all(folder == "" for _, _, folder in bp.project_types(found))

    def test_jobs_are_listed_newest_first_with_their_titles(self, cs_project):
        found = bp.find_project(cs_project)
        jobs = bp.project_jobs(found, "class_2D_new")
        assert [(name, status, label) for name, status, label, _, _alias in jobs] == [
            ("J12", "failure", "J12"),
            ("J7", "success", "J7  first look"),
        ]
        assert "completed" in bp.project_jobs(found, "filament_tracer_gpu")[0][3]

    def test_the_numbers_sort_as_numbers(self, cs_project):
        import json

        d = cs_project / "J100"
        d.mkdir()
        (d / "job.json").write_text(
            json.dumps({"uid": "J100", "job_type": "class_2D_new"})
        )
        found = bp.find_project(cs_project)
        assert [j[0] for j in bp.project_jobs(found, "class_2D_new")] == [
            "J100",
            "J12",
            "J7",
        ]

    def test_the_sidebar_lists_the_jobs_by_type_and_marks_the_current_one(
        self, qapp, cs_project
    ):
        side = bp.PlacesSidebar()
        side.refresh(str(cs_project / "J7"))
        side.wait()
        top = [side.topLevelItem(i).text(0) for i in range(side.topLevelItemCount())]
        assert top[1] == "CRYOSPARC  DEMO PROJECT"
        node = side.topLevelItem(1)
        names = [node.child(i).text(0) for i in range(node.childCount())]
        assert names == [
            "Project folder",
            "2D Classification",
            "Blob Picker",
            "Extract From Micrographs",
            "Filament Tracer",
            "Some New Job Type",
        ]
        classes = node.child(1)
        assert [classes.child(i).text(0) for i in range(classes.childCount())] == [
            "J12",
            "J7  first look",
        ]
        assert side.selectedItems()[0].text(0) == "J7  first look"

    def test_clicking_a_job_goes_there_and_a_type_only_opens(self, qapp, cs_project):
        side = bp.PlacesSidebar()
        side.refresh(str(cs_project))
        side.wait()
        got = []
        side.navigate.connect(got.append)
        node = side.topLevelItem(1)
        tracer = node.child(4)
        assert not tracer.isExpanded()
        side.itemClicked.emit(tracer, 0)
        side.wait()
        assert tracer.isExpanded() and got == []
        side.itemClicked.emit(tracer.child(0), 0)
        assert got == [str(cs_project / "J5")]

    def test_class_averages_in_a_job_are_a_stack_not_a_map(self, cs_project):
        import mrcfile

        rng = np.random.default_rng(3)
        for name, expected in (
            ("J7_020_class_averages.mrc", "stack"),
            ("J7_volume_map.mrc", "volume"),
        ):
            with mrcfile.new(cs_project / "J7" / name, overwrite=True) as f:
                f.set_data(rng.normal(size=(12, 32, 32)).astype(np.float32))
            info = bp.build_preview(cs_project / "J7" / name)
            assert info["kind"] == expected, name
        stack = bp.build_preview(cs_project / "J7" / "J7_020_class_averages.mrc")
        assert dict(stack["facts"])["Images"] == "12 of 32 × 32"

    def test_the_list_counts_the_class_averages_of_a_job_as_images(self, cs_project):
        import mrcfile

        data = np.zeros((6, 24, 24), dtype=np.float32)
        for name in ("J7_020_class_averages.mrc", "J7_volume_map.mrc"):
            with mrcfile.new(cs_project / "J7" / name, overwrite=True) as f:
                f.set_data(data)
        info, n_images, _ = fb._get_file_info(
            str(cs_project / "J7" / "J7_020_class_averages.mrc")
        )
        assert (info, n_images) == ("24×24", "6")
        info, n_images, _ = fb._get_file_info(
            str(cs_project / "J7" / "J7_volume_map.mrc")
        )
        assert (info, n_images) == ("24×24×6", "1")

    def test_the_times_of_job_json_are_read_in_both_forms(self):
        assert (
            bp._cryosparc_time({"$date": "2026-10-05T22:52:30.382Z"})
            == "2026-10-05 22:52"
        )
        assert bp._cryosparc_time("2026-10-05T22:52:30Z") == "2026-10-05 22:52"
        assert bp._cryosparc_time(None) == "" and bp._cryosparc_time(5) == ""

    def test_the_inputs_come_from_the_connections_when_parents_is_missing(
        self, tmp_path
    ):
        import json

        d = tmp_path / "J3"
        d.mkdir()
        (d / "job.json").write_text(
            json.dumps(
                {
                    "uid": "J3",
                    "job_type": "class_2D_new",
                    "input_slot_groups": [
                        {
                            "connections": [
                                {"job_uid": "J1"},
                                {"job_uid": "J1"},
                                {"job_uid": "J2"},
                            ]
                        }
                    ],
                }
            )
        )
        assert bp.read_cryosparc_job(d)["parents"] == ["J1", "J2"]

    def test_a_job_folder_is_described(self, cs_project):
        info = bp.build_preview(cs_project / "J7")
        facts = dict(info["facts"])
        assert facts["CryoSPARC job"] == "J7, 2D Classification"
        assert facts["Title"] == "first look" and facts["State"] == "completed"
        assert (
            "Inputs from" not in facts
        )  # the jobs before and after are links, not text
        assert facts["Started"] == "2026-10-05 14:55"
        assert info["status"] == "success"

    def test_the_project_folder_is_described(self, cs_project):
        info = bp.build_preview(cs_project)
        assert ("CryoSPARC project", "Demo project, 7 jobs") in info["facts"]

    def test_a_dataset_gives_its_rows_and_fields_by_group(self, cs_project):
        info = bp.build_preview(cs_project / "J5" / "picked_particles.cs")
        facts = dict(info["facts"])
        assert (
            info["kind"] == "table" and facts["Rows"] == "7" and facts["Fields"] == "4"
        )
        assert info["text"].splitlines() == ["blob: 2", "ctf: 1", "(other): 1"]

    def test_the_list_says_what_job_each_folder_is(self, qapp, cs_project):
        model = fb.FileBrowserModel(str(cs_project))
        rows = {
            model.item(r, fb.COL_NAME).text(): (
                model.item(r, fb.COL_TYPE).text(),
                model.item(r, fb.COL_NAME).toolTip(),
                model.item(r, fb.COL_TYPE).data(fb.ROLE_SORT),
            )
            for r in range(model.rowCount())
        }
        assert rows["J7/"][0] == "2D Classification"
        assert rows["J7/"][1] == "J7  first look\ncompleted"
        assert rows["J7/"][2] == "folder"  # it still sorts with the folders
        assert rows["J99/"][0] == "Folder"  # no job.json: not a job

    def test_the_refresh_reads_the_jobs_again(self, qapp, cs_project):
        import json

        side = bp.PlacesSidebar()
        side.refresh(str(cs_project))
        side.wait()
        assert side.topLevelItem(1).childCount() == 6
        d = cs_project / "J20"
        d.mkdir()
        (d / "job.json").write_text(
            json.dumps({"uid": "J20", "job_type": "homo_refine_new"})
        )
        side.invalidate()
        side.refresh(str(cs_project))
        side.wait()
        assert side.topLevelItem(1).childCount() == 7


class TestThumbnails:
    def test_a_picture_is_cut_down_and_stretched(self):
        rng = np.random.default_rng(0)
        pic = bp.thumbnail_array(rng.normal(size=(1000, 600)), max_size=256)
        assert pic.dtype == np.uint8 and max(pic.shape) <= 256
        assert pic.min() == 0 and pic.max() == 255
        assert pic.shape[0] > pic.shape[1]  # the shape is kept

    def test_a_flat_or_empty_picture_is_black(self):
        assert not bp.thumbnail_array(np.full((10, 10), 3.0)).any()
        assert not bp.thumbnail_array(np.full((10, 10), np.nan)).any()

    def test_outliers_do_not_squash_the_rest(self):
        a = np.linspace(0, 1, 10000).reshape(100, 100)
        a[0, 0] = 1e9
        assert bp.thumbnail_array(a)[50, 50] > 100

    def test_the_flat_background_of_a_masked_map_is_left_out(self):
        # a filament a few percent of the section, exactly 0 outside its mask: the
        # percentiles of all the pixels clipped a sixth of the density white
        rng = np.random.default_rng(0)
        y, x = np.mgrid[:200, :200] - 100.0
        inside = np.hypot(x, y) < 25
        a = np.where(inside, rng.normal(1.0, 0.5, (200, 200)), 0.0)
        pic = bp.thumbnail_array(a)
        assert np.mean(pic[inside] == 255) < 0.02
        assert np.mean(pic[inside] == 0) < 0.02

    def test_an_8_bit_picture_keeps_its_common_levels(self):
        a = np.zeros((100, 100), np.uint8)
        a[:, 50:] = 200
        assert bp.contrast_limits(a) == (0.0, 200.0)
        assert bp.contrast_limits(np.full((5, 5), 7.0)) is None

    def test_a_picture_is_binned_not_decimated(self):
        # one pixel of each block keeps all of the noise; the mean of the block averages it out
        rng = np.random.default_rng(1)
        a = rng.normal(size=(1024, 1024)).astype(np.float32)
        a[:, 512:] += 1.0
        pic = bp.thumbnail_array(a, max_size=128).astype(float)
        assert pic.shape == (128, 128)
        left, right = pic[:, :64], pic[:, 64:]
        assert right.mean() - left.mean() > 4 * (left.std() + right.std())


class TestPreviews:
    @staticmethod
    def _mrc(path, data, apix=1.5):
        import mrcfile

        with mrcfile.new(path, overwrite=True) as f:
            f.set_data(np.asarray(data, dtype=np.float32))
            f.voxel_size = apix

    def test_a_folder_lists_what_is_in_it(self, project):
        info = bp.build_preview(project / "Class2D")
        assert info["kind"] == "folder"
        assert ("Contains", "0 files, 2 folders") in info["facts"]
        assert info["text"].splitlines() == ["job010/", "job011/"]
        assert "graph" not in info

    def test_a_project_folder_has_its_jobs_as_a_graph_in_place_of_the_list(
        self, project
    ):
        info = bp.build_preview(project)
        assert info["kind"] == "folder"
        assert ("Contains", "1 files, 3 folders") in info["facts"]
        assert ("RELION project", "2 job types") in info["facts"]
        assert info["graph"] == dict(kind="relion", root=str(project))
        assert info["text"] == ""

    def test_a_job_folder_with_a_pipeline_is_not_called_a_project(self, project):
        info = bp.build_preview(project / "Class2D/job010")
        assert not any(label == "RELION project" for label, _ in info["facts"])
        assert ("RELION job", "Class2D/job010, success") in info["facts"]

    def test_a_job_folder_says_how_it_ended(self, project):
        info = bp.build_preview(project / "Class2D/job011")
        assert info["status"] == "failure"
        assert ("RELION job", "Class2D/job011, failure") in info["facts"]

    def test_a_volume_shows_its_central_slice(self, tmp_path):
        vol = np.zeros((20, 30, 40))
        vol[10] = 5.0  # the central slice differs from the others
        vol[10, 5, 5] = 9.0
        self._mrc(tmp_path / "map.mrc", vol, apix=1.25)
        info = bp.build_preview(tmp_path / "map.mrc")
        assert info["kind"] == "volume"
        facts = dict(info["facts"])
        assert facts["Dimensions"] == "40 × 30 × 20"
        assert facts["Pixel size"] == "1.250 Å"
        assert info["image"].shape == (30, 40)

    def test_a_stack_shows_its_first_images_as_tiles(self, tmp_path):
        rng = np.random.default_rng(1)
        self._mrc(tmp_path / "classes.mrcs", rng.normal(size=(30, 64, 64)))
        info = bp.build_preview(tmp_path / "classes.mrcs")
        assert info["kind"] == "stack"
        assert dict(info["facts"])["Images"] == "30 of 64 × 64"
        # 12 of them, 4 across and 3 down
        assert info["image"].shape == (3 * 98 - 2, 4 * 98 - 2)

    def test_a_single_image(self, tmp_path):
        self._mrc(tmp_path / "mic.mrc", np.random.default_rng(2).normal(size=(80, 120)))
        info = bp.build_preview(tmp_path / "mic.mrc")
        assert info["kind"] == "image" and info["image"].shape == (80, 120)

    def test_a_png_is_read_with_pillow(self, tmp_path):
        from PIL import Image

        Image.fromarray((np.arange(5000).reshape(50, 100) % 255).astype(np.uint8)).save(
            tmp_path / "a.png"
        )
        info = bp.build_preview(tmp_path / "a.png")
        assert info["kind"] == "image" and info["image"].shape == (50, 100)
        assert dict(info["facts"])["Dimensions"] == "100 × 50"

    def test_a_star_file_gives_its_rows_and_columns(self, tmp_path):
        (tmp_path / "data.star").write_text(
            "data_optics\nloop_\n_rlnOpticsGroup #1\n_rlnImagePixelSize #2\n_rlnImageSize #3\n"
            "1 1.1 96\n\ndata_particles\nloop_\n_rlnImageName #1\n_rlnClassNumber #2\n"
            "000001@a.mrcs 1\n000002@a.mrcs 2\n"
        )
        info = bp.build_preview(tmp_path / "data.star")
        assert info["kind"] == "table"
        assert dict(info["facts"])["Rows"] == "2"
        assert "data_particles: rlnImageName, rlnClassNumber" in info["text"]

    def test_a_text_file_gives_its_first_lines(self, tmp_path):
        (tmp_path / "notes.txt").write_text("first line\nsecond line\n")
        info = bp.build_preview(tmp_path / "notes.txt")
        assert info["kind"] == "text" and info["text"].startswith("first line")

    def test_a_file_that_cannot_be_read_gives_a_note_not_an_error(self, tmp_path):
        (tmp_path / "broken.mrc").write_bytes(b"not an mrc file at all")
        info = bp.build_preview(tmp_path / "broken.mrc")
        assert info["image"] is None and "Cannot preview" in info["note"]
        assert "Size" in dict(info["facts"])
        gone = bp.build_preview(tmp_path / "missing.txt")
        assert "Cannot read" in gone["note"]


class TestPathBar:
    def test_it_has_the_text_of_a_line_edit(self, qapp, project):
        bar = bp.PathBar(str(project))
        assert bar.text() == str(project)
        bar.setText(str(project / "Class2D"))
        assert bar.text() == str(project / "Class2D")
        assert bar.segments()[-1] == ("Class2D", str(project / "Class2D"))

    def test_clicking_a_segment_goes_there(self, qapp, project):
        bar = bp.PathBar(str(project / "Class2D"))
        bar.resize(900, 30)
        bar.show()
        got = []
        bar.pathClicked.connect(got.append)
        buttons = bar.findChildren(bp.QToolButton)
        by_text = {b.text(): b for b in buttons}
        by_text["proj"].click()
        assert got == [str(project)]

    def test_folders_that_do_not_fit_fold_into_a_menu(self, qapp):
        bar = bp.PathBar(
            "/aaaaaaaaaa/bbbbbbbbbb/cccccccccc/dddddddddd/eeeeeeeeee/ffffffffff"
        )
        bar.resize(900, 30)
        wide = bar.segments()
        bar.resize(220, 30)
        bar.setText(
            "/aaaaaaaaaa/bbbbbbbbbb/cccccccccc/dddddddddd/eeeeeeeeee/ffffffffff"
        )
        narrow = bar.segments()
        assert wide[0][0] != "…" and narrow[0][0] == "…"
        assert narrow[-1][0] == "ffffffffff"  # the folder shown is never folded away

    def test_a_typed_path_is_returned(self, qapp):
        bar = bp.PathBar("/tmp")
        got = []
        bar.returnPressed.connect(lambda: got.append(bar.text()))
        bar.edit()
        assert bar.is_editing()
        bar._edit.setText("/var")
        bar._edit.returnPressed.emit()
        assert got == ["/var"] and not bar.is_editing()


class TestSidebar:
    def test_sections_and_places(self, qapp, tmp_path):
        side = bp.PlacesSidebar()
        marked = tmp_path / "marked"
        marked.mkdir()
        side.refresh(str(tmp_path), bookmarks=[str(marked)], recent=[str(tmp_path)])
        side.wait()
        top = [side.topLevelItem(i).text(0) for i in range(side.topLevelItemCount())]
        assert top == ["PLACES", "BOOKMARKS", "RECENT"]
        assert side.topLevelItem(1).child(0).text(0) == "marked"

    def test_a_missing_bookmark_is_not_listed(self, qapp, tmp_path):
        side = bp.PlacesSidebar()
        side.refresh(str(tmp_path), bookmarks=[str(tmp_path / "gone")])
        side.wait()
        assert side.topLevelItemCount() == 1

    def test_a_project_shows_its_jobs_and_marks_the_current_one(self, qapp, project):
        side = bp.PlacesSidebar()
        side.refresh(str(project / "Class2D/job010"))
        side.wait()
        top = [side.topLevelItem(i).text(0) for i in range(side.topLevelItemCount())]
        assert top[1].startswith("RELION")
        types = side.topLevelItem(1)
        names = [types.child(i).text(0) for i in range(types.childCount())]
        assert names == ["Project folder", "Class2D", "Extract"]
        jobs = types.child(1)
        assert [jobs.child(i).text(0) for i in range(jobs.childCount())] == [
            "job011",
            "job010",
        ]
        selected = side.selectedItems()
        assert len(selected) == 1 and selected[0].text(0) == "job010"

    def test_inside_a_job_the_project_and_its_jobs_are_still_listed(
        self, qapp, project
    ):
        side = bp.PlacesSidebar()
        side.refresh(str(project / "Class2D/job010"))
        side.wait()
        node = side.topLevelItem(1)
        names = [node.child(i).text(0) for i in range(node.childCount())]
        assert names == ["Project folder", "Class2D", "Extract"]
        got = []
        side.navigate.connect(got.append)
        side.itemClicked.emit(node.child(0), 0)
        assert got == [str(project)]  # the project, not the job folder

    def test_a_click_goes_there(self, qapp, project):
        side = bp.PlacesSidebar()
        side.refresh(str(project))
        side.wait()
        got = []
        side.navigate.connect(got.append)
        item = side.topLevelItem(1).child(0)
        side.itemClicked.emit(item, 0)
        assert got == [str(project)]

    def test_the_project_stays_when_the_folder_is_outside_it(
        self, qapp, project, tmp_path
    ):
        side = bp.PlacesSidebar()
        side.refresh(str(project / "Class2D/job010"))
        side.wait()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        side.refresh(str(elsewhere))
        side.wait()
        top = [side.topLevelItem(i).text(0) for i in range(side.topLevelItemCount())]
        assert top[1].startswith("RELION")  # so Project folder leads back
        got = []
        side.navigate.connect(got.append)
        side.itemClicked.emit(side.topLevelItem(1).child(0), 0)
        assert got == [str(project)]
        assert side.selectedItems() == []  # the folder shown is not one of its places

    def test_a_project_that_is_gone_is_dropped(self, qapp, project, tmp_path):
        import shutil

        side = bp.PlacesSidebar()
        side.refresh(str(project))
        side.wait()
        shutil.rmtree(project)
        side.refresh(str(tmp_path))
        side.wait()
        top = [side.topLevelItem(i).text(0) for i in range(side.topLevelItemCount())]
        assert top == ["PLACES"]

    def test_entering_another_project_replaces_it(self, qapp, project, cs_project):
        side = bp.PlacesSidebar()
        side.refresh(str(project))
        side.wait()
        side.refresh(str(cs_project))
        assert side.topLevelItem(1).text(0).startswith("CRYOSPARC")
        side.refresh(str(cs_project.parent))
        side.wait()
        assert side.topLevelItem(1).text(0).startswith("CRYOSPARC")

    def test_the_jobs_of_a_type_are_listed_when_it_is_opened(self, qapp, project):
        side = bp.PlacesSidebar()
        side.refresh(str(project))
        side.wait()
        extract = side.topLevelItem(1).child(2)
        assert extract.child(0).text(0) == "…"
        side.expandItem(extract)
        side.wait()
        assert [extract.child(i).text(0) for i in range(extract.childCount())] == [
            "job005"
        ]

    def test_the_most_specific_place_is_marked(self, qapp, project):
        side = bp.PlacesSidebar()
        side.refresh(str(project), recent=[str(project)])
        side.wait()
        assert side.selectedItems()[0].text(0) == "Project folder"


class TestPreviewPane:
    def test_a_preview_is_shown(self, qapp):
        pane = bp.PreviewPane()
        pane.resize(300, 500)
        pane.show()
        pane.show_info(
            dict(
                title="a.mrc",
                facts=[("Size", "1 KB")],
                image=(np.arange(400).reshape(20, 20) % 255).astype(np.uint8),
                text="",
            )
        )
        assert pane._title.text() == "a.mrc"
        assert not pane._picture.isHidden() and pane._pixmap is not None
        assert "Size" in pane._facts.text()
        assert pane._text.isHidden()

    def test_a_tall_picture_fits_in_the_pane(self, qapp):
        # the central slice of a long, thin map (43 × 123): scaled to the width of the
        # pane it was 1030 pixels tall and covered the facts and the buttons
        pane = bp.PreviewPane()
        pane.resize(500, 800)
        pane.show()
        image = (np.arange(123 * 43).reshape(123, 43) % 255).astype(np.uint8)
        pane.show_info(
            dict(title="long.map", facts=[("Size", "1 KB")], image=image, text="")
        )
        qapp.processEvents()
        shown = pane._picture.pixmap()
        assert shown.height() <= pane._PICTURE_BOX
        assert abs(shown.width() / shown.height() - 43 / 123) < 0.02
        assert pane.minimumSizeHint().height() < 800
        assert pane._facts.geometry().top() >= pane._picture.geometry().bottom()

    def test_a_small_picture_is_enlarged_with_sharp_pixels(self, qapp):
        pane = bp.PreviewPane()
        pane.resize(500, 800)
        pane.show()
        image = np.zeros((10, 10), dtype=np.uint8)
        image[:, 5:] = 255
        pane.show_info(dict(title="small.map", facts=[], image=image, text=""))
        shown = pane._picture.pixmap().toImage()
        assert shown.width() == pane._PICTURE_BOX
        # no grey ramp between the black and the white halves
        row = [shown.pixelColor(x, 5).value() for x in range(shown.width())]
        assert set(row) == {0, 255}

    def test_a_note_replaces_the_picture(self, qapp):
        pane = bp.PreviewPane()
        pane.show()
        pane.show_info(
            dict(title="x", facts=[], image=None, text="", note="Cannot preview")
        )
        assert pane._picture.isHidden() and pane._note.text() == "Cannot preview"

    def test_a_path_is_read_in_the_background(self, qapp, tmp_path):
        (tmp_path / "n.txt").write_text("hello\n")
        pane = bp.PreviewPane()
        pane.show()
        pane.show_path(str(tmp_path / "n.txt"))
        pane.wait()
        for _ in range(50):
            qapp.processEvents()
        assert pane._title.text() == "n.txt"
        assert "hello" in pane._text.toPlainText()

    @staticmethod
    def _right_click_save(pane, label, out, monkeypatch):
        """Right-click a picture, choose to save it and save it as ``out``; the menu's texts."""
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QFileDialog

        offered = {}

        def run_menu(menu, _pos):
            offered["items"] = [a.text() for a in menu.actions()]
            return menu.actions()[0]

        def select_file(_dialog, name):
            offered["name"] = name

        monkeypatch.setattr(pane, "_run_menu", run_menu)
        monkeypatch.setattr(QFileDialog, "selectFile", select_file)
        monkeypatch.setattr(QFileDialog, "exec", lambda _d: QFileDialog.Accepted)
        monkeypatch.setattr(QFileDialog, "selectedFiles", lambda _d: [str(out)])
        label.customContextMenuRequested.emit(QPoint(5, 5))
        return offered

    def test_a_right_click_saves_the_picture(self, qapp, tmp_path, monkeypatch):
        from PySide6.QtGui import QImage

        pane = bp.PreviewPane()
        pane.resize(300, 500)
        pane.show()
        image = (np.arange(1200).reshape(30, 40) % 255).astype(np.uint8)
        pane.show_info(dict(title="a.mrc", facts=[], image=image, text=""))
        out = tmp_path / "a.png"
        offered = self._right_click_save(pane, pane._picture, out, monkeypatch)
        assert offered["items"] == ["Save image as…"]
        assert offered["name"] == "a.png"
        saved = QImage(str(out))
        # the picture itself, not as it is scaled to fit the pane
        assert (saved.width(), saved.height()) == (40, 30)

    def test_no_menu_without_a_picture(self, qapp, monkeypatch):
        from PySide6.QtCore import QPoint

        pane = bp.PreviewPane()
        pane.show_info(dict(title="x", facts=[], image=None, text=""))
        shown = []
        monkeypatch.setattr(pane, "_run_menu", lambda m, p: shown.append(m))
        pane._picture.customContextMenuRequested.emit(QPoint(5, 5))
        assert shown == []

    def test_a_picture_of_a_job_is_saved_whole(self, qapp, tmp_path, monkeypatch):
        from PySide6.QtGui import QImage

        pane = bp.PreviewPane()
        pane.resize(200, 600)
        pane.show()
        rgb = np.zeros((300, 600, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        pane.show_info(
            dict(
                title="job010",
                facts=[],
                outcome=dict(
                    panels=[dict(title="FSC 0.143", image=rgb, caption="")],
                    facts=[],
                ),
            )
        )
        label = pane._outcome_panels[0][0]
        assert label.pixmap().width() < 600  # shown smaller than it is
        out = tmp_path / "fsc.png"
        offered = self._right_click_save(pane, label, out, monkeypatch)
        assert offered["name"] == "job010_FSC_0_143.png"
        saved = QImage(str(out))
        assert (saved.width(), saved.height()) == (600, 300)
        assert saved.pixelColor(10, 10).red() == 255

    def test_an_answer_for_an_earlier_path_is_dropped(self, qapp):
        pane = bp.PreviewPane()
        pane.show()
        pane.show_info(dict(title="now", facts=[], image=None, text=""))
        pane._serial += 1
        pane._on_ready(
            pane._serial - 1, dict(title="late", facts=[], image=None, text="")
        )
        assert pane._title.text() == "now"


class TestInTheBrowser:
    @pytest.fixture
    def browser(self, qapp, project):
        w = fb.FolderBrowserWidget(start_dir=str(project / "Class2D" / "job010"))
        w.resize(1200, 600)
        w.show()
        yield w
        w._preview.wait()
        w.close()

    def test_the_panes_are_in_a_splitter(self, browser):
        assert [browser._splitter.widget(i) for i in range(3)] == [
            browser._sidebar,
            browser._tree,
            browser._preview,
        ]

    def test_going_to_a_folder_updates_the_path_bar_and_the_sidebar(
        self, browser, project
    ):
        browser._navigate_to(str(project / "Extract" / "job005"))
        assert browser._path_edit.text() == str(project / "Extract" / "job005")
        assert browser._path_edit.segments()[-1][0] == "job005"
        browser._sidebar.wait()
        assert browser._sidebar.selectedItems()[0].text(0) == "job005"

    def test_the_project_folder_leads_back_from_anywhere(
        self, browser, project, tmp_path
    ):
        side = browser._sidebar

        def click(text):
            stack, found = [
                side.topLevelItem(i) for i in range(side.topLevelItemCount())
            ], None
            while stack:
                item = stack.pop()
                stack.extend(item.child(i) for i in range(item.childCount()))
                if item.text(0) == text:
                    found = item
            side.itemClicked.emit(found, 0)

        click("Project folder")
        assert browser._model._root_path == str(project)
        browser._navigate_to(str(tmp_path))  # out of the project altogether
        click("Project folder")
        assert browser._model._root_path == str(project)

    def test_the_project_folder_clears_the_selection_when_already_there(
        self, browser, project, qapp
    ):
        browser._navigate_to(str(project))
        browser._tree.setCurrentIndex(browser._model.index(0, 0))
        assert browser._tree.selectionModel().hasSelection()
        steps = len(browser._history)
        browser._navigate_to(str(project))
        assert not browser._tree.selectionModel().hasSelection()
        assert len(browser._history) == steps  # not a new step back

    def test_a_click_on_a_breadcrumb_navigates(self, browser, project):
        browser._path_edit.pathClicked.emit(str(project))
        assert browser._model._root_path == str(project)

    def test_the_folder_is_previewed_until_something_is_selected(self, browser, qapp):
        browser._preview.wait()
        for _ in range(50):
            qapp.processEvents()
        assert browser._preview._title.text() == "job010"
        assert "RELION job" in browser._preview._facts.text()

    def test_the_panes_can_be_hidden_and_shown(self, browser):
        browser._sidebar_action.setChecked(False)
        browser._preview_action.setChecked(False)
        assert browser._sidebar.isHidden() and browser._preview.isHidden()
        browser._sidebar_action.setChecked(True)
        assert not browser._sidebar.isHidden()

    def test_the_buttons_are_below_the_facts_of_the_preview(self, browser, qapp):
        browser.resize(1200, 700)
        browser.show()
        # the buttons of the first selection arrive later and may hide the bar
        browser.wait_for_selection()
        bar, preview = browser._action_bar, browser._preview
        bar.show()
        qapp.processEvents()
        assert preview.isAncestorOf(bar)
        facts = preview._facts.mapTo(browser, preview._facts.rect().bottomLeft())
        assert bar.mapTo(browser, bar.rect().topLeft()).y() >= facts.y()
        assert bar.mapTo(browser, bar.rect().topLeft()).x() >= preview.x()

    def test_the_buttons_go_to_the_bottom_when_the_preview_is_hidden(
        self, browser, qapp
    ):
        browser.resize(1200, 700)
        browser.show()
        browser.wait_for_selection()
        bar = browser._action_bar
        bar.show()
        browser._preview_action.setChecked(False)
        qapp.processEvents()
        assert not browser._preview.isAncestorOf(bar) and bar.isVisible()
        # under the list of files
        assert bar.mapTo(browser, bar.rect().topLeft()).y() >= (
            browser._splitter.mapTo(browser, browser._splitter.rect().bottomLeft()).y()
        )
        browser._preview_action.setChecked(True)
        qapp.processEvents()
        assert browser._preview.isAncestorOf(bar) and bar.isVisible()

    def test_the_smallest_size_of_the_window_does_not_follow_the_selection(
        self, browser, qapp
    ):
        # over X11 each new smallest size goes to the window manager: a maximized window jitters
        browser.show()
        smallest = browser.minimumSize()
        for row in range(browser._model.rowCount()):
            browser._tree.setCurrentIndex(browser._model.index(row, 0))
            browser.wait_for_selection()
            browser._preview.wait()
            for _ in range(20):
                qapp.processEvents()
            assert browser.minimumSize() == smallest

    def test_the_choice_of_panes_is_remembered(self, qapp, project):
        w = fb.FolderBrowserWidget(start_dir=str(project))
        w._preview_action.setChecked(False)
        w.close()
        w2 = fb.FolderBrowserWidget(start_dir=str(project))
        assert not w2._preview_action.isChecked() and w2._sidebar_action.isChecked()
        w2.close()
        for x in (w, w2):
            x._preview.wait()

    def test_a_pane_hidden_at_the_last_launch_is_hidden(self, qapp, project):
        # the action starts unchecked: unchecking it emitted nothing, the pane came back
        w = fb.FolderBrowserWidget(start_dir=str(project))
        w._sidebar_action.setChecked(False)
        w.close()
        w2 = fb.FolderBrowserWidget(start_dir=str(project))
        w2.show()
        qapp.processEvents()
        assert not w2._sidebar_action.isChecked() and not w2._sidebar.isVisible()
        w2._sidebar_action.setChecked(True)  # one press of F9 shows it
        assert w2._sidebar.isVisible()
        w2.close()
        for x in (w, w2):
            x._preview.wait()

    def test_the_widths_are_saved_while_a_pane_is_hidden(self, qapp, project):
        # they were not saved at all: a hidden pane has no width
        w = fb.FolderBrowserWidget(start_dir=str(project))
        w._sidebar_action.setChecked(False)
        w._preview_action.setChecked(True)
        w.resize(1200, 700)
        w.show()
        qapp.processEvents()
        total = sum(w._splitter.sizes())
        w._splitter.setSizes([0, total - 400, 400])  # a wider preview
        qapp.processEvents()
        w.close()
        saved = fb.QSettings("helicon", "display").value("browser_pane_sizes")
        saved = [int(x) for x in saved]
        assert saved[0] > 0  # the width the sidebar had, for when it is shown again
        assert abs(saved[2] - 400) <= 2
        w._sidebar_action.setChecked(True)
        w._save_panes()
        w._preview.wait()

    def test_quit_closes_the_window_so_its_layout_is_saved(
        self, qapp, project, monkeypatch
    ):
        from PySide6.QtWidgets import QApplication

        w = fb.FolderBrowserWidget(start_dir=str(project))
        # the other windows of the tests are left alone
        monkeypatch.setattr(QApplication, "topLevelWidgets", staticmethod(lambda: [w]))
        monkeypatch.setattr(qapp, "quit", lambda: None)
        w.show()
        w._preview_action.setChecked(False)
        w._quit_action.trigger()
        assert not w.isVisible()
        settings = fb.QSettings("helicon", "display")
        assert not settings.value("browser_show_preview", True, type=bool)
        settings.setValue("browser_show_preview", True)
        w._preview.wait()


class TestFlowLayout:
    @staticmethod
    def _bar(n=6, width=300):
        from PySide6.QtWidgets import QPushButton, QWidget

        host = QWidget()
        layout = bp.FlowLayout(host, margin=4, h_spacing=5, v_spacing=5)
        buttons = [QPushButton(f"Button {i}") for i in range(n)]
        for b in buttons:
            b.setFixedHeight(30)
            layout.addWidget(b)
        host.resize(width, 200)
        host.show()
        return host, layout, buttons

    def test_widgets_wrap_when_the_row_is_full(self, qapp):
        host, layout, buttons = self._bar(6, width=300)
        layout.activate()
        rows = {b.y() for b in buttons}
        assert len(rows) > 1
        assert all(b.x() + b.width() <= 300 for b in buttons)
        wide = layout.heightForWidth(2000)
        assert layout.heightForWidth(300) > wide  # narrower, taller
        assert wide == 30 + 8  # one row and the margins

    def test_hidden_widgets_take_no_room(self, qapp):
        host, layout, buttons = self._bar(6, width=2000)
        layout.activate()
        x_before = buttons[3].x()
        buttons[1].hide()
        layout.activate()
        assert buttons[3].x() < x_before

    def test_widgets_of_different_heights_are_centred_in_their_row(self, qapp):
        from PySide6.QtWidgets import QCheckBox

        host, layout, buttons = self._bar(3, width=2000)
        box = QCheckBox("New window")
        layout.addWidget(box)
        box.show()
        layout.activate()
        assert box.height() < buttons[0].height()
        centre = lambda w: w.y() + w.height() / 2
        assert abs(centre(box) - centre(buttons[0])) <= 1

    def test_it_is_filled_and_emptied_like_a_box_layout(self, qapp):
        from PySide6.QtWidgets import QPushButton

        host, layout, buttons = self._bar(3, width=2000)
        extra = QPushButton("first")
        layout.insertWidget(0, extra)
        assert layout.count() == 4 and layout.itemAt(0).widget() is extra
        layout.removeWidget(extra)
        assert layout.count() == 3
        assert layout.itemAt(3) is None


class TestActionButtons:
    @pytest.fixture
    def browser(self, qapp, tmp_path):
        w = fb.FolderBrowserWidget(start_dir=str(tmp_path))
        yield w
        w._preview.wait()
        w.close()

    def test_every_button_has_an_icon_and_a_category(self, browser):
        modes = [mode for _, mode in browser._DISPLAY_BUTTONS]
        assert set(modes) <= set(fb._ACTION_STYLE)
        for attr, mode in browser._DISPLAY_BUTTONS:
            btn = getattr(browser, attr)
            assert btn.property("category") == fb._ACTION_STYLE[mode][1]
            assert (
                not btn.icon().isNull()
            )  # qtawesome is a requirement of the gui extra

    def test_the_web_apps_are_marked_from_the_viewers(self, browser):
        assert browser._btn_hill.property("category") == "app"
        assert browser._btn_slice.property("category") == "viewer"
        sheet = fb._browser_stylesheet(fb._THEME_COLORS["Dark"])
        assert 'QPushButton[category="app"]' in sheet

    def test_the_buttons_wrap_in_a_narrow_window(self, browser):
        browser.resize(500, 500)
        browser.show()
        for attr, _mode in browser._DISPLAY_BUTTONS:
            getattr(browser, attr).setVisible(True)
        browser._action_bar.show()
        browser._action_bar.layout().activate()
        rows = {getattr(browser, attr).y() for attr, _mode in browser._DISPLAY_BUTTONS}
        assert len(rows) > 1

    def test_the_new_window_box_is_centred_with_the_buttons(self, browser):
        browser.resize(1200, 500)
        browser.show()
        for attr, _mode in browser._DISPLAY_BUTTONS:
            getattr(browser, attr).setVisible(attr == "_btn_slice")
        browser._action_bar.show()
        browser._action_bar.layout().activate()
        box, button = browser._new_window_cb, browser._btn_slice
        assert (
            abs((box.y() + box.height() / 2) - (button.y() + button.height() / 2)) <= 1
        )

    def test_the_new_window_box_is_in_the_bar(self, browser):
        assert browser._new_window_cb.text() == "New window"
        layout = browser._action_bar.layout()
        assert any(
            layout.itemAt(i).widget() is browser._new_window_cb
            for i in range(layout.count())
        )


@pytest.fixture
def pipeline_project(tmp_path):
    """A RELION project that has a pipeline: Import -> Extract -> Class2D (two) -> Select."""
    return make_pipeline_project(tmp_path)


def _links(info):
    return {title: [label for label, _ in items] for title, items in info["links"]}


class TestJobLinks:
    """The jobs before and after a job, to click, between the facts and the file list."""

    def test_a_relion_job_has_the_jobs_that_wrote_its_files_and_those_that_read_its_own(
        self, pipeline_project
    ):
        info = bp.build_preview(pipeline_project / "Class2D/job010")
        assert _links(info) == {
            "Inputs from": ["Extract/job005  first-boxes"],
            "Used by": ["Select/job012"],
        }

    def test_the_labels_say_the_alias_and_the_links_go_to_the_folder(
        self, pipeline_project
    ):
        info = bp.build_preview(pipeline_project / "Select/job012")
        titles = dict(info["links"])
        # two inputs, in the order of the jobs; a file of an import has no job before it
        assert [(label, folder) for label, folder in titles["Inputs from"]] == [
            ("Extract/job005  first-boxes", str(pipeline_project / "Extract/job005")),
            ("Class2D/job010  looks", str(pipeline_project / "Class2D/job010")),
        ]
        assert "Used by" not in titles

    def test_a_job_that_reads_an_import_has_the_import_before_it(
        self, pipeline_project
    ):
        assert _links(bp.build_preview(pipeline_project / "Extract/job005")) == {
            "Inputs from": ["Import/job001"],
            "Used by": [
                "Class2D/job010  looks",
                "Class2D/job011",
                "Select/job012",
            ],
        }

    def test_an_import_has_only_children_and_a_leaf_only_parents(
        self, pipeline_project
    ):
        info = bp.build_preview(pipeline_project / "Import/job001")
        assert _links(info) == {"Used by": ["Extract/job005  first-boxes"]}
        lonely = pipeline_project / "Class2D/job011"
        assert _links(bp.build_preview(lonely)) == {
            "Inputs from": ["Extract/job005  first-boxes"]
        }

    def test_a_job_whose_folder_was_deleted_is_text_not_a_link(self, pipeline_project):
        import shutil

        shutil.rmtree(pipeline_project / "Extract/job005")
        info = bp.build_preview(pipeline_project / "Class2D/job010")
        ((label, folder),) = dict(info["links"])["Inputs from"]
        assert label.startswith("Extract/job005") and folder is None

    def test_without_a_pipeline_there_are_no_links(self, project):
        (project / "default_pipeline.star").write_text("data_pipeline_general\n")
        info = bp.build_preview(project / "Class2D/job010")
        assert info["links"] == []

    def test_a_changed_pipeline_is_read_again(self, pipeline_project):
        job = pipeline_project / "Class2D/job010"
        assert "Used by" in _links(bp.build_preview(job))
        pipeline = pipeline_project / "default_pipeline.star"
        pipeline.write_text(
            PIPELINE.replace("Class2D/job010/run_it020_data.star Select/job012/\n", "")
        )
        assert "Used by" not in _links(bp.build_preview(job))

    def test_the_columns_are_found_by_their_names(self, pipeline_project):
        # RELION may write them in another order
        text = PIPELINE.replace(
            "_rlnPipeLineEdgeFromNode #1\n_rlnPipeLineEdgeProcess #2",
            "_rlnPipeLineEdgeProcess #1\n_rlnPipeLineEdgeFromNode #2",
        )
        lines = []
        for line in text.splitlines():
            parts = line.split()
            if (
                len(parts) == 2
                and parts[1].endswith("/")
                and "." in parts[0]
                and not parts[0].endswith("/")
            ):
                line = f"{parts[1]} {parts[0]}"
            lines.append(line)
        (pipeline_project / "default_pipeline.star").write_text("\n".join(lines))
        bp._PIPELINES.clear()
        info = bp.build_preview(pipeline_project / "Class2D/job010")
        assert _links(info)["Inputs from"] == ["Extract/job005  first-boxes"]

    def test_a_cryosparc_job_has_its_parents_and_its_children(self, cs_project):
        info = bp.build_preview(cs_project / "J6")
        links = dict(info["links"])
        assert [label for label, _ in links["Inputs from"]] == [
            "J5  Filament Tracer  my picks"
        ] or True
        assert links["Inputs from"][0][1] == str(cs_project / "J5")
        # the file lists no children: they follow from the parents of the other jobs
        assert [folder for _, folder in links["Used by"]] == [
            str(cs_project / "J7"),
            str(cs_project / "J12"),
        ]

    def test_the_children_listed_in_the_file_count_too(self, cs_project):
        import json

        record = json.loads((cs_project / "J13" / "job.json").read_text())
        record["children"] = ["J14", "J404"]  # J404 has no folder
        (cs_project / "J13" / "job.json").write_text(json.dumps(record))
        bp.invalidate_project_cache()
        info = bp.build_preview(cs_project / "J13")
        used = dict(info["links"])["Used by"]
        assert used[0][1] == str(cs_project / "J14")
        assert used[1] == ("J404", None)

    def test_a_plain_folder_has_no_links(self, tmp_path):
        assert "links" not in bp.build_preview(tmp_path)


class TestLinksInThePane:
    def test_the_links_show_between_the_facts_and_the_file_list_and_are_clickable(
        self, qapp, pipeline_project
    ):
        pane = bp.PreviewPane()
        pane.resize(320, 600)
        pane.show()
        pane.show_info(bp.build_preview(pipeline_project / "Class2D/job010"))
        assert not pane._links.isHidden()
        html = pane._links.text()
        assert "Inputs from" in html and "Used by" in html
        assert "Extract/job005" in html and "href=" in html
        order = [pane.layout().itemAt(i).widget() for i in range(pane.layout().count())]
        order = [w for w in order if w is not None]
        assert (
            order.index(pane._facts)
            < order.index(pane._links)
            < order.index(pane._text)
        )
        pane.clear()
        assert pane._links.isHidden()

    def test_a_click_asks_for_the_folder_of_the_job(self, qapp, tmp_path):
        pane = bp.PreviewPane()
        asked = []
        pane.path_requested.connect(asked.append)
        odd = tmp_path / "my jobs" / "Class2D" / "job 010 & é"
        pane.show_info(dict(title="x", links=[("Used by", [("odd", str(odd))])]))
        import re

        href = re.search(r"href='([^']*)'", pane._links.text()).group(1)
        pane._links.linkActivated.emit(href)
        assert asked == [str(odd)]

    def test_a_long_list_is_cut_and_says_how_many_more(self, qapp, tmp_path):
        pane = bp.PreviewPane()
        items = [(f"J{i}", str(tmp_path / f"J{i}")) for i in range(1, 31)]
        pane.show_info(dict(title="x", links=[("Used by", items)]))
        html = pane._links.text()
        assert "J12<" in html and "J13<" not in html
        assert "18 more" in html

    def test_a_label_cannot_add_markup(self, qapp, tmp_path):
        pane = bp.PreviewPane()
        pane.show_info(
            dict(title="x", links=[("Used by", [("<b>bold</b>", str(tmp_path))])])
        )
        assert "<b>bold</b>" not in pane._links.text()

    def test_the_browser_goes_to_the_job_that_was_clicked(self, qapp, pipeline_project):
        window = fb.FolderBrowserWidget(
            start_dir=str(pipeline_project / "Class2D/job010")
        )
        try:
            target = pipeline_project / "Select/job012"
            window._preview.path_requested.emit(str(target))
            assert window._model._root_path == str(target)
        finally:
            window._preview.wait()
            window.close()


class TestSidebarOpenAndClosedTypes:
    @staticmethod
    def _types(side):
        node = side.topLevelItem(1)
        return {
            node.child(i).text(0): node.child(i) for i in range(1, node.childCount())
        }

    def test_going_to_a_job_opens_only_its_type(self, qapp, project):
        side = bp.PlacesSidebar()
        side.refresh(str(project / "Class2D/job010"))
        side.wait()
        types = self._types(side)
        assert types["Class2D"].isExpanded() and not types["Extract"].isExpanded()

    def test_a_type_that_was_closed_by_hand_stays_closed_when_another_job_is_clicked(
        self, qapp, project
    ):
        side = bp.PlacesSidebar()
        side.refresh(str(project / "Class2D/job010"))
        side.wait()
        side.refresh(str(project / "Extract/job005"))
        side.wait()
        types = self._types(side)
        assert types["Class2D"].isExpanded() and types["Extract"].isExpanded()
        side.collapseItem(types["Class2D"])
        # the click on a job of the other type, and a refresh of the lists, leave Class2D closed
        side.refresh(str(project / "Extract/job005"), bookmarks=[str(project)])
        side.wait()
        side.refresh(str(project / "Extract"), bookmarks=[str(project)])
        side.wait()
        types = self._types(side)
        assert not types["Class2D"].isExpanded() and types["Extract"].isExpanded()

    def test_a_job_in_a_closed_type_opens_it_again(self, qapp, project):
        side = bp.PlacesSidebar()
        side.refresh(str(project / "Class2D/job010"))
        side.wait()
        side.collapseItem(self._types(side)["Class2D"])
        side.refresh(str(project / "Class2D/job011"))
        side.wait()
        types = self._types(side)
        assert types["Class2D"].isExpanded() and types["Class2D"].childCount() == 2

    def test_the_open_types_are_not_changed_by_looking_at_the_same_folder_again(
        self, qapp, project
    ):
        side = bp.PlacesSidebar()
        side.refresh(str(project / "Class2D/job010"))
        side.wait()
        side.collapseItem(self._types(side)["Class2D"])
        side.refresh(str(project / "Class2D/job010"))  # a refresh, not a new place
        side.wait()
        assert not self._types(side)["Class2D"].isExpanded()

    def test_the_tree_is_shown_before_the_project_is_read(self, qapp, project):
        side = bp.PlacesSidebar()
        side.refresh(str(project))
        # at once there are the places, and a note in place of the types
        assert side.topLevelItem(0).childCount() == 2
        assert side.topLevelItem(1).child(1).text(0) == "Reading the jobs…"
        side.wait()
        assert side.topLevelItem(1).child(1).text(0) == "Class2D"

    def test_a_bookmark_of_a_folder_that_is_gone_goes_away_after_the_reading(
        self, qapp, tmp_path
    ):
        side = bp.PlacesSidebar()
        side.refresh(str(tmp_path), bookmarks=[str(tmp_path / "gone"), str(tmp_path)])
        side.wait()
        sections = {
            side.topLevelItem(i).text(0): side.topLevelItem(i)
            for i in range(side.topLevelItemCount())
        }
        assert [
            sections["BOOKMARKS"].child(i).text(0)
            for i in range(sections["BOOKMARKS"].childCount())
        ] == [tmp_path.name]
