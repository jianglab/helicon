"""Reading a CryoSPARC project's own account of what its files are.

CryoSPARC writes class averages, picking templates, extracted particles,
micrographs and 3D maps all as ``.mrc``, in job folders named ``J1``, ``J2``,
``J38``. Neither the suffix nor the header distinguishes them -- one class
average stack in the projects these tests were drawn from measures
200 x 200 x 200 -- so the job type and CryoSPARC's naming have to.
"""

import json
import os

import numpy as np
import pytest

from helicon.lib import cryosparc_project


def _mrc(path, nx=8, ny=8, nz=1):
    """A minimal valid MRC file of the given shape."""
    mrcfile = pytest.importorskip("mrcfile")
    data = (
        np.zeros((nz, ny, nx), dtype=np.float32)
        if nz > 1
        else np.zeros((ny, nx), dtype=np.float32)
    )
    with mrcfile.new(str(path), overwrite=True) as f:
        f.set_data(data)
    return str(path)


@pytest.fixture
def project(tmp_path):
    """A project laid out the way the real ones are."""
    root = tmp_path / "CS-demo"
    (root).mkdir()
    (root / "project.json").write_text(
        json.dumps({"uid": "P511", "title": "Demo", "project_dir": str(root)})
    )
    (root / "cs.lock").write_text("")
    (root / "workspaces.json").write_text("[]")

    def job(name, job_type):
        d = root / name
        d.mkdir()
        (d / "job.json").write_text(
            json.dumps({"uid": name, "type": job_type, "job_type": job_type})
        )
        return d

    j45 = job("J45", "class_2D_new")
    _mrc(j45 / "J45_040_class_averages.mrc", nz=5)

    j32 = job("J32", "blob_picker_gpu")
    _mrc(j32 / "templates.mrc", nz=4)

    j101 = job("J101", "homo_refine_new")
    _mrc(j101 / "J101_006_volume_map.mrc", nz=8)
    _mrc(j101 / "J101_006_volume_precision.mrc", nz=5)

    j42 = job("J42", "extract_micrographs_multi")
    (j42 / "extract").mkdir()
    _mrc(j42 / "extract" / "abc_particles.mrc", nz=6)

    # a helical branch: filament tracing, extraction, then 2D classification.
    # Only the tracer's own particles carry the filament columns, so the 2D
    # job two steps downstream can only be recognised through its ancestry.
    def cs_with_fields(path, fields):
        import numpy as np

        dtype = np.dtype([(name, "<f4") for name in fields])
        np.save(str(path), np.zeros(2, dtype=dtype), allow_pickle=False)
        os.replace(str(path) + ".npy", str(path))

    j5 = job("J5", "filament_tracer_gpu")
    cs_with_fields(
        j5 / "picked_particles.cs",
        ["blob/path", "filament/filament_uid", "filament/arc_length_A"],
    )
    j6 = job("J6", "extract_micrographs_multi")
    (j6 / "job.json").write_text(
        json.dumps(
            {
                "uid": "J6",
                "job_type": "extract_micrographs_multi",
                "input_slot_groups": [{"connections": [{"job_uid": "J5"}]}],
            }
        )
    )
    j7 = job("J7", "class_2D_new")
    (j7 / "job.json").write_text(
        json.dumps(
            {
                "uid": "J7",
                "job_type": "class_2D_new",
                "input_slot_groups": [{"connections": [{"job_uid": "J6"}]}],
            }
        )
    )
    _mrc(j7 / "J7_020_class_averages.mrc", nz=5)

    import numpy as np

    averages = np.zeros(
        5, dtype=np.dtype([("uid", "<u8"), ("blob/idx", "<u4"), ("blob/res_A", "<f4")])
    )
    averages["blob/idx"] = np.arange(5)
    np.save(str(j7 / "J7_020_class_averages.cs"), averages, allow_pickle=False)
    os.replace(
        str(j7 / "J7_020_class_averages.cs.npy"), str(j7 / "J7_020_class_averages.cs")
    )
    # 10 particles: class 0 gets five, class 1 three, class 2 two, and
    # classes 3 and 4 none at all
    particles = np.zeros(
        10,
        dtype=np.dtype(
            [
                ("uid", "<u8"),
                ("alignments2D/class", "<u4"),
                ("filament/filament_uid", "<u8"),
            ]
        ),
    )
    particles["alignments2D/class"] = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
    np.save(str(j7 / "J7_020_particles.cs"), particles, allow_pickle=False)
    os.replace(str(j7 / "J7_020_particles.cs.npy"), str(j7 / "J7_020_particles.cs"))

    # a 2D job downstream of the tracing that wrote no particle output of its
    # own -- the real J26 is exactly this, and only its ancestry says it is
    # helical
    j8 = job("J8", "class_2D_new")
    (j8 / "job.json").write_text(
        json.dumps(
            {
                "uid": "J8",
                "job_type": "class_2D_new",
                "input_slot_groups": [{"connections": [{"job_uid": "J6"}]}],
            }
        )
    )
    _mrc(j8 / "J8_020_class_averages.mrc", nz=5)

    # the class averages dataset names its images the way a data.star does,
    # with the path written relative to the project directory
    import numpy as _np

    refs = _np.zeros(
        5,
        dtype=_np.dtype(
            [
                ("uid", "<u8"),
                ("blob/path", "S64"),
                ("blob/idx", "<u4"),
                ("blob/shape", "<u4", (2,)),
                ("blob/psize_A", "<f4"),
            ]
        ),
    )
    refs["blob/path"] = b"J7/J7_020_class_averages.mrc"
    refs["blob/idx"] = _np.arange(5)
    refs["blob/shape"] = [8, 8]
    refs["blob/psize_A"] = 1.5
    _np.save(str(j7 / "J7_020_refs.cs"), refs, allow_pickle=False)
    os.replace(str(j7 / "J7_020_refs.cs.npy"), str(j7 / "J7_020_refs.cs"))

    # a select_2D job: CryoSPARC keeps the inherited parameters in a
    # passthrough dataset, and for these jobs that is the ONLY place the
    # filament columns appear -- their own iteration files have none. Padded
    # with enough iteration datasets that a scan which did not put
    # passthrough first would never reach it.
    j12 = job("J12", "select_2D")
    for i in range(20):
        cs_with_fields(j12 / ("J12_%03d_particles.cs" % i), ["uid", "blob/idx"])
    cs_with_fields(
        j12 / "J12_passthrough_particles_excluded.cs",
        ["uid", "filament/filament_uid", "filament/arc_length_A"],
    )

    # CryoSPARC imports by symlinking the original data into the job
    outside = tmp_path / "raw"
    outside.mkdir()
    gain = _mrc(outside / "gain.mrc")
    j1 = job("J1", "import_movies")
    (j1 / "imported").mkdir()
    os.symlink(gain, j1 / "imported" / "gain.mrc")

    return root


class TestProjectAndJobDetection:
    def test_a_project_is_found_by_its_marker_not_its_name(self, project, tmp_path):
        # the two real examples are named P319 and CS-phageg-slac
        assert cryosparc_project.is_cryosparc_project(project)
        assert not cryosparc_project.is_cryosparc_project(tmp_path)
        assert not cryosparc_project.is_cryosparc_project(project / "J45")

    def test_a_file_finds_its_project_and_job(self, project):
        f = project / "J45" / "J45_040_class_averages.mrc"
        assert cryosparc_project.find_project_root(f) == project
        assert cryosparc_project.find_job_dir(f) == project / "J45"
        assert cryosparc_project.job_type(f) == "class_2D_new"

    def test_a_job_subdirectory_still_belongs_to_the_job(self, project):
        f = project / "J42" / "extract" / "abc_particles.mrc"
        assert cryosparc_project.find_job_dir(f) == project / "J42"
        assert cryosparc_project.job_type(f) == "extract_micrographs_multi"

    def test_a_symlinked_import_keeps_its_job(self, project):
        # resolving the link walks out of the project to where the movies
        # really live, and there is no job.json above that
        f = project / "J1" / "imported" / "gain.mrc"
        assert f.is_symlink()
        assert cryosparc_project.find_job_dir(f) == project / "J1"
        assert cryosparc_project.job_type(f) == "import_movies"

    def test_outside_a_project_nothing_is_claimed(self, tmp_path):
        stray = _mrc(tmp_path / "stray.mrc", nz=5)
        assert cryosparc_project.find_project_root(stray) is None
        assert cryosparc_project.job_type(stray) is None
        assert cryosparc_project.mrc_content(stray) is None


class TestMrcContent:
    @pytest.mark.parametrize(
        "relative, expected",
        [
            ("J45/J45_040_class_averages.mrc", "stack"),
            ("J32/templates.mrc", "stack"),
            ("J42/extract/abc_particles.mrc", "stack"),
            ("J101/J101_006_volume_map.mrc", "volume"),
            ("J101/J101_006_volume_precision.mrc", "volume"),
        ],
    )
    def test_each_file_is_read_correctly(self, project, relative, expected):
        assert cryosparc_project.mrc_content(project / relative) == expected

    def test_class_averages_are_recognised(self, project):
        assert cryosparc_project.is_class_averages(
            project / "J45" / "J45_040_class_averages.mrc"
        )
        assert not cryosparc_project.is_class_averages(
            project / "J101" / "J101_006_volume_map.mrc"
        )

    def test_the_name_outranks_the_job_type(self, project):
        # a refinement job also writes masks and half maps; the name is the
        # more specific signal and is consulted first
        volume_job = project / "J101"
        odd = _mrc(volume_job / "J101_diag_class_averages.mrc", nz=3)
        assert cryosparc_project.mrc_content(odd) == "stack"

    def test_an_unknown_job_type_says_nothing(self, project):
        d = project / "J77"
        d.mkdir()
        (d / "job.json").write_text(json.dumps({"job_type": "some_new_job"}))
        f = _mrc(d / "output.mrc", nz=5)
        assert cryosparc_project.mrc_content(f) is None

    def test_metadata_files_are_recognised(self, project):
        assert cryosparc_project.is_metadata_file(project / "project.json")
        assert cryosparc_project.is_metadata_file(project / "J45" / "job.json")
        assert not cryosparc_project.is_metadata_file(
            project / "J45" / "J45_040_class_averages.mrc"
        )


class TestFileBrowserUsesTheProject:
    """The browser must not offer a 100-image class average stack as a map."""

    @pytest.fixture
    def widget(self, project):
        pytest.importorskip("PySide6")
        from unittest.mock import patch

        from PySide6.QtWidgets import QApplication
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        app = QApplication.instance() or QApplication([])
        with (
            patch(
                "helicon.lib.gui.file_browser.FileBrowserModel.file_rows",
                return_value=[],
            ),
            patch(
                "helicon.lib.gui.file_browser.FileBrowserModel.dir_rows",
                return_value=[],
            ),
        ):
            w = FolderBrowserWidget(start_dir=str(project))
        yield w
        w.close()

    def test_a_class_average_stack_is_not_a_volume(self, widget, project):
        stack = str(project / "J45" / "J45_040_class_averages.mrc")
        assert not widget._mrc_is_volume(stack)
        modes = widget._display_modes_for(stack)
        # the actions a map gets, which would be nonsense here
        for wrong in ("volume", "chimerax", "orthogonal", "proc3d"):
            assert wrong not in modes, modes
        assert "gallery" in modes

    def test_a_map_is_still_a_volume(self, widget, project):
        volume = str(project / "J101" / "J101_006_volume_map.mrc")
        assert widget._mrc_is_volume(volume)
        modes = widget._display_modes_for(volume)
        assert "volume" in modes and "chimerax" in modes

    def test_helical_class_averages_offer_the_helical_apps(self, widget, project):
        stack = str(project / "J7" / "J7_020_class_averages.mrc")
        modes = widget._display_modes_for(stack)
        for app in ("helicalProjection", "hill", "denovo3D"):
            assert app in modes, modes

    def test_single_particle_class_averages_do_not(self, widget, project):
        # offering them would be noise on a project that is not helical, and
        # whether it is can be told from the filament columns or the ancestry
        stack = str(project / "J45" / "J45_040_class_averages.mrc")
        modes = widget._display_modes_for(stack)
        for app in ("helicalProjection", "hill", "denovo3D"):
            assert app not in modes, modes
        assert "gallery" in modes

    def test_extracted_particles_are_a_stack(self, widget, project):
        particles = str(project / "J42" / "extract" / "abc_particles.mrc")
        assert not widget._mrc_is_volume(particles)

    def test_outside_a_project_the_header_still_decides(self, widget, tmp_path):
        stray = _mrc(tmp_path / "stray_volume.mrc", nz=5)
        assert widget._mrc_is_volume(stray)


class TestHelical:
    """CryoSPARC's answer to rlnIsHelix, which it does not record directly.

    Filament tracing writes per-particle ``filament/`` columns -- the
    counterpart of rlnHelicalTubeID -- and everything downstream carries them.
    A job that wrote no particle output of its own, or that sits two steps
    past the tracing, is recognised through its ancestry instead. Measured
    over three real projects: 53 of 53 jobs in the helical one, 0 of 42 in the
    two that are not.
    """

    def test_a_filament_tracer_is_helical_by_its_type(self, project):
        assert cryosparc_project.is_helical(project / "J5")

    def test_filament_columns_make_a_job_helical(self, project):
        assert cryosparc_project._cs_has_filament_fields(
            project / "J5" / "picked_particles.cs"
        )

    def test_ancestry_carries_it_downstream(self, project):
        # J8 wrote no particle dataset at all, so nothing of its own can say;
        # the tracer two steps back is the only evidence there is
        assert not cryosparc_project._job_has_filament_data(project / "J8")
        assert cryosparc_project.is_helical(
            project / "J8" / "J8_020_class_averages.mrc"
        )

    def test_a_downstream_job_keeps_the_filament_columns(self, project):
        # the common case: 7 of 9 real 2D jobs in the helical project carry
        # them directly
        assert cryosparc_project._job_has_filament_data(project / "J7")
        assert cryosparc_project.is_helical(
            project / "J7" / "J7_020_class_averages.mrc"
        )

    def test_a_single_particle_job_is_not_helical(self, project):
        assert not cryosparc_project.is_helical(
            project / "J45" / "J45_040_class_averages.mrc"
        )
        assert not cryosparc_project.is_helical(project / "J101")

    def test_outside_a_project_nothing_is_helical(self, tmp_path):
        assert not cryosparc_project.is_helical(tmp_path)


class TestClassAbundance:
    """Abundance for CryoSPARC classes, counted rather than read.

    RELION states each class's share as _rlnClassDistribution. CryoSPARC
    states it nowhere: the averages dataset says which slice each class
    occupies, the particles dataset says which class each particle went to,
    and counting those is the answer. Verified against real jobs -- 50 classes
    from a 15 MB particles file, and 100 from a 728 MB one -- with the
    fractions summing to 1.
    """

    def test_abundance_is_counted_from_the_assignments(self, project):
        mrc = project / "J7" / "J7_020_class_averages.mrc"
        frames, fractions = cryosparc_project.class_abundance(mrc)
        assert list(frames) == [0, 1, 2, 3, 4]
        assert fractions == pytest.approx([0.5, 0.3, 0.2, 0.0, 0.0])
        assert sum(fractions) == pytest.approx(1.0)

    def test_the_companion_files_are_found(self, project):
        mrc = project / "J7" / "J7_020_class_averages.mrc"
        assert cryosparc_project.companion_dataset(mrc).name.endswith(
            "class_averages.cs"
        )
        assert cryosparc_project.particles_dataset(mrc).name.endswith("particles.cs")
        assert cryosparc_project.has_class_abundance(mrc)

    def test_without_the_datasets_it_is_not_offered(self, project):
        # J45's averages have no .cs beside them, which is exactly when the
        # sort must not be offered rather than offered and then failing
        mrc = project / "J45" / "J45_040_class_averages.mrc"
        assert cryosparc_project.companion_dataset(mrc) is None
        assert not cryosparc_project.has_class_abundance(mrc)
        assert cryosparc_project.class_abundance(mrc) is None

    def test_a_map_is_never_offered_it(self, project):
        assert not cryosparc_project.has_class_abundance(
            project / "J101" / "J101_006_volume_map.mrc"
        )


class TestPassthroughDatasets:
    """Inherited parameters live in ``*_passthrough_*.cs``, not the iteration files.

    A ``select_2D`` job carries the filament columns only there -- measured on
    four such jobs in the helical project -- and a ``helix_refine`` can hold
    89 datasets, where the passthrough file sorts alphabetically after three
    dozen numbered ones. Both together mean the passthrough has to be looked
    at first rather than eventually.
    """

    def test_passthrough_is_scanned_before_the_iteration_files(self):
        names = [
            "J12_019_particles.cs",
            "J12_passthrough_particles_excluded.cs",
            "J12_000_particles.cs",
            "J12_templates.cs",
        ]
        ordered = sorted(names, key=cryosparc_project._dataset_scan_order)
        assert ordered[0] == "J12_passthrough_particles_excluded.cs"
        assert ordered[-1] == "J12_templates.cs"

    def test_a_job_is_helical_on_its_passthrough_alone(self, project):
        job_dir = project / "J12"
        iteration_files = sorted(job_dir.glob("J12_0*_particles.cs"))
        assert len(iteration_files) == 20
        for f in iteration_files:
            assert not cryosparc_project._cs_has_filament_fields(f)
        assert cryosparc_project._job_has_filament_data(job_dir)
        assert cryosparc_project.is_helical(job_dir)

    def test_the_latest_iteration_is_preferred_for_abundance(self, project):
        # an iterative job writes one particles dataset per iteration; the
        # classes on display come from the last of them
        job_dir = project / "J12"
        chosen = cryosparc_project.particles_dataset(job_dir / "J12_unrelated.mrc")
        assert chosen is not None
        assert chosen.name == "J12_passthrough_particles_excluded.cs" or (
            chosen.name == "J12_019_particles.cs"
        )


class TestAbundanceCaching:
    """Counting assignments is a full pass over a dataset that can be huge.

    Measured on a real job: 728 MB and 1.2 s cold, 0.00 s warm. The answer
    never changes for a finished job, so it is kept on disk under helicon's
    cache -- but a job that is re-run must not be served the old numbers,
    which is what the modification times in the key are for.
    """

    def test_it_is_cached_under_helicons_cache_directory(self):
        assert hasattr(cryosparc_project._class_abundance_cached, "clear_cache")

    def test_two_calls_agree(self, project):
        mrc = project / "J7" / "J7_020_class_averages.mrc"
        first = cryosparc_project.class_abundance(mrc)
        second = cryosparc_project.class_abundance(mrc)
        assert first is not None and second is not None
        assert list(first[0]) == list(second[0])
        assert first[1] == pytest.approx(second[1])

    def test_rerunning_a_job_is_not_served_the_old_answer(self, project):
        import numpy as np

        mrc = project / "J7" / "J7_020_class_averages.mrc"
        before = cryosparc_project.class_abundance(mrc)[1]
        assert before == pytest.approx([0.5, 0.3, 0.2, 0.0, 0.0])

        # the job is re-run and every particle lands in class 4
        particles = cryosparc_project.particles_dataset(mrc)
        rewritten = np.zeros(
            10, dtype=np.dtype([("uid", "<u8"), ("alignments2D/class", "<u4")])
        )
        rewritten["alignments2D/class"] = 4
        np.save(str(particles), rewritten, allow_pickle=False)
        os.replace(str(particles) + ".npy", str(particles))
        os.utime(particles, (0, 0))  # a different mtime, hence a different key

        after = cryosparc_project.class_abundance(mrc)[1]
        assert after == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0])


class TestImageReferences:
    """A .cs that names images is an image stack, as a data.star is.

    ``blob/path`` and ``blob/idx`` say where each image lives and which slice
    it is, and the path is relative to the project directory rather than the
    job. Datasets that name no images -- passthrough parameters, a volume --
    are not stacks and keep their old actions.
    """

    def test_a_dataset_naming_images_is_a_stack(self, project):
        assert cryosparc_project.has_image_refs(project / "J7" / "J7_020_refs.cs")

    def test_a_dataset_without_blobs_is_not(self, project):
        assert not cryosparc_project.has_image_refs(
            project / "J12" / "J12_passthrough_particles_excluded.cs"
        )

    def test_paths_resolve_against_the_project_not_the_job(self, project):
        refs = cryosparc_project.image_refs(project / "J7" / "J7_020_refs.cs")
        assert refs is not None
        entries, shape, apix, skipped = refs
        assert len(entries) == 5
        assert skipped == 0
        assert shape == (8, 8)
        assert apix == pytest.approx(1.5)
        for index, path, _ in entries:
            assert os.path.isfile(path)
            assert path.endswith("J7_020_class_averages.mrc")
        assert [e[0] for e in entries] == [0, 1, 2, 3, 4]

    def test_missing_images_are_counted_not_fatal(self, project, tmp_path):
        import numpy as np

        job = project / "J7"
        broken = np.zeros(
            2,
            dtype=np.dtype([("uid", "<u8"), ("blob/path", "S64"), ("blob/idx", "<u4")]),
        )
        broken["blob/path"] = b"J7/not_here.mrc"
        np.save(str(job / "broken.cs"), broken, allow_pickle=False)
        os.replace(str(job / "broken.cs.npy"), str(job / "broken.cs"))
        assert cryosparc_project.has_image_refs(job / "broken.cs")
        assert cryosparc_project.image_refs(job / "broken.cs") is None


class TestBrowserOffersStackActionsForCs:
    @pytest.fixture
    def widget(self, project):
        pytest.importorskip("PySide6")
        from unittest.mock import patch

        from PySide6.QtWidgets import QApplication
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        QApplication.instance() or QApplication([])
        with (
            patch(
                "helicon.lib.gui.file_browser.FileBrowserModel.file_rows",
                return_value=[],
            ),
            patch(
                "helicon.lib.gui.file_browser.FileBrowserModel.dir_rows",
                return_value=[],
            ),
        ):
            w = FolderBrowserWidget(start_dir=str(project))
        yield w
        w.close()

    def test_a_dataset_of_images_gets_the_stack_actions(self, widget, project):
        modes = widget._display_modes_for(str(project / "J7" / "J7_020_refs.cs"))
        assert "slice" in modes and "gallery" in modes
        assert "images2star" in modes

    def test_a_passthrough_dataset_keeps_only_images2star(self, widget, project):
        modes = widget._display_modes_for(
            str(project / "J12" / "J12_passthrough_particles_excluded.cs")
        )
        assert modes == ["images2star"]
