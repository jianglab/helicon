"""The parts of the file browser: path bar, places sidebar, preview pane."""

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from helicon.lib.gui import browser_parts as bp
from helicon.lib.gui import file_browser as fb


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_settings(tmp_path, monkeypatch):
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(
        fb, "QSettings", lambda *a: QSettings(ini, QSettings.Format.IniFormat)
    )


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
    import json

    root = tmp_path / "CS-demo"
    root.mkdir()
    (root / "project.json").write_text(
        json.dumps({"uid": "P1", "title": "Demo project"})
    )

    def job(uid, job_type, status, title="", parents=(), **extra):
        d = root / uid
        d.mkdir()
        record = dict(uid=uid, job_type=job_type, status=status, title=title, **extra)
        # the real files list the inputs as "parents" (and in the connections)
        record["parents"] = list(parents)
        record["input_slot_groups"] = [
            dict(connections=[dict(job_uid=p) for p in parents])
        ]
        (d / "job.json").write_text(json.dumps(record))
        return d

    j5 = job("J5", "filament_tracer_gpu", "completed", "my picks")
    job("J6", "extract_micrographs_multi", "completed", parents=["J5"])
    job(
        "J7", "class_2D_new", "completed", "first look", parents=["J6"],
        created_at={"$date": "2026-10-05T14:50:00.000Z"},
        started_at={"$date": "2026-10-05T14:55:49.123Z"},
        completed_at={"$date": "2026-10-05T16:01:02.000Z"},
    )  # fmt: skip
    job("J12", "class_2D_new", "failed", parents=["J6"])
    job("J13", "some_new_job_type", "running")
    job("J14", "blob_picker_gpu", "building")
    job("J15", "blob_picker_gpu", "queued")
    dtype = [
        ("blob/path", "S20"),
        ("blob/idx", "<u4"),
        ("ctf/df1_A", "<f4"),
        ("uid", "<u8"),
    ]
    with open(j5 / "picked_particles.cs", "wb") as f:
        np.save(f, np.zeros(7, dtype=dtype))
    (root / "J99").mkdir()  # no job.json: not a job
    return root


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
        assert [(name, status, label) for name, status, label, _ in jobs] == [
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
        got = []
        side.navigate.connect(got.append)
        node = side.topLevelItem(1)
        tracer = node.child(4)
        assert not tracer.isExpanded()
        side.itemClicked.emit(tracer, 0)
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
        assert facts["Inputs from"] == "J6"
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
        assert side.topLevelItem(1).childCount() == 6
        d = cs_project / "J20"
        d.mkdir()
        (d / "job.json").write_text(
            json.dumps({"uid": "J20", "job_type": "homo_refine_new"})
        )
        side.invalidate()
        side.refresh(str(cs_project))
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


class TestPreviews:
    @staticmethod
    def _mrc(path, data, apix=1.5):
        import mrcfile

        with mrcfile.new(path, overwrite=True) as f:
            f.set_data(np.asarray(data, dtype=np.float32))
            f.voxel_size = apix

    def test_a_folder_lists_what_is_in_it(self, project):
        info = bp.build_preview(project)
        assert info["kind"] == "folder"
        assert ("Contains", "1 files, 3 folders") in info["facts"]
        assert "Class2D/" in info["text"].splitlines()
        assert ("RELION project", "2 job types") in info["facts"]

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
        top = [side.topLevelItem(i).text(0) for i in range(side.topLevelItemCount())]
        assert top == ["PLACES", "BOOKMARKS", "RECENT"]
        assert side.topLevelItem(1).child(0).text(0) == "marked"

    def test_a_missing_bookmark_is_not_listed(self, qapp, tmp_path):
        side = bp.PlacesSidebar()
        side.refresh(str(tmp_path), bookmarks=[str(tmp_path / "gone")])
        assert side.topLevelItemCount() == 1

    def test_a_project_shows_its_jobs_and_marks_the_current_one(self, qapp, project):
        side = bp.PlacesSidebar()
        side.refresh(str(project / "Class2D/job010"))
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
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        side.refresh(str(elsewhere))
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
        shutil.rmtree(project)
        side.refresh(str(tmp_path))
        top = [side.topLevelItem(i).text(0) for i in range(side.topLevelItemCount())]
        assert top == ["PLACES"]

    def test_entering_another_project_replaces_it(self, qapp, project, cs_project):
        side = bp.PlacesSidebar()
        side.refresh(str(project))
        side.refresh(str(cs_project))
        assert side.topLevelItem(1).text(0).startswith("CRYOSPARC")
        side.refresh(str(cs_project.parent))
        assert side.topLevelItem(1).text(0).startswith("CRYOSPARC")

    def test_the_jobs_of_a_type_are_listed_when_it_is_opened(self, qapp, project):
        side = bp.PlacesSidebar()
        side.refresh(str(project))
        extract = side.topLevelItem(1).child(2)
        assert extract.child(0).text(0) == "…"
        side.expandItem(extract)
        assert [extract.child(i).text(0) for i in range(extract.childCount())] == [
            "job005"
        ]

    def test_the_most_specific_place_is_marked(self, qapp, project):
        side = bp.PlacesSidebar()
        side.refresh(str(project), recent=[str(project)])
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

    def test_the_choice_of_panes_is_remembered(self, qapp, project):
        w = fb.FolderBrowserWidget(start_dir=str(project))
        w._preview_action.setChecked(False)
        w.close()
        w2 = fb.FolderBrowserWidget(start_dir=str(project))
        assert not w2._preview_action.isChecked() and w2._sidebar_action.isChecked()
        w2.close()
        for x in (w, w2):
            x._preview.wait()


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
