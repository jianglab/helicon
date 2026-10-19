"""What a job made, as pictures, for the preview of a job folder."""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from helicon.lib.gui import browser_parts as bp
from helicon.lib.gui import job_outcome as jo
from _settings import own_settings as _own_settings


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_cache(tmp_path, monkeypatch):
    """The counts that took long are kept in a folder of the test, not in the cache of the user."""
    monkeypatch.setenv("HELICON_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(autouse=True)
def own_settings(tmp_path, monkeypatch, qapp):
    _own_settings(monkeypatch, tmp_path / "settings.ini")


def _mrc(path, data, apix=1.5):
    import mrcfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as f:
        f.set_data(np.asarray(data, dtype=np.float32))
        f.voxel_size = apix


def _volume(n=24, seed=0):
    z, y, x = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((x - n / 2) ** 2 + (y - n / 2) ** 2 + (z - n / 2) ** 2)
    rng = np.random.default_rng(seed)
    return (np.exp(-(r**2) / 30.0) + 0.05 * rng.normal(size=(n, n, n))).astype(
        np.float32
    )


def _loop(block, columns, rows):
    head = f"\n# version 30001\n\ndata_{block}\n\nloop_\n"
    head += "".join(f"_{c} #{i + 1}\n" for i, c in enumerate(columns))
    return head + "\n".join(" ".join(str(v) for v in row) for row in rows) + "\n\n"


def _fsc_rows(n=60, cut=0.2):
    rows = []
    for i in range(n):
        x = 0.4 * i / (n - 1)
        rows.append(
            (
                i,
                f"{x:.6f}",
                f"{1 / x if x else 999:.4f}",
                f"{max(0.0, 1 - (x / cut) ** 2):.4f}",
            )
        )
    return rows


class TestFscResolution:
    def test_the_curve_is_followed_to_where_it_crosses(self):
        x = np.array([0.0, 0.1, 0.2, 0.3])
        y = np.array([1.0, 0.8, 0.0, 0.0])
        # 0.143 is a fifth of the way from 0.8 to 0 ... between 0.1 and 0.2 at 0.1 + 0.1 * 0.657/0.8
        assert jo.fsc_resolution(x, y, 0.143) == pytest.approx(
            1 / (0.1 + 0.1 * (0.8 - 0.143) / 0.8)
        )

    def test_another_level(self):
        x = np.array([0.0, 0.2, 0.4])
        y = np.array([1.0, 0.6, 0.2])
        assert jo.fsc_resolution(x, y, 0.5) == pytest.approx(1 / 0.25)

    def test_a_curve_that_stays_above_has_no_crossing(self):
        assert jo.fsc_resolution([0.0, 0.1, 0.2], [1.0, 0.9, 0.8]) is None

    def test_a_curve_that_starts_below_has_none_either(self):
        assert jo.fsc_resolution([0.0, 0.1], [0.05, 0.01]) is None


class TestReadingStar:
    def test_a_loop_is_found_by_its_block_and_ends_with_the_block(self, tmp_path):
        star = tmp_path / "m.star"
        star.write_text(
            _loop(
                "model_classes",
                ["rlnReferenceImage", "rlnClassDistribution"],
                [("1@a.mrcs", 0.7), ("2@a.mrcs", 0.3)],
            )
            + _loop(
                "model_class_1",
                ["rlnResolution", "rlnGoldStandardFsc"],
                [(0.1, 0.9), (0.2, 0.1)],
            )
        )
        classes = jo._star_loop(star, "model_classes")
        assert classes["rlnClassDistribution"] == ["0.7", "0.3"]
        fsc = jo._star_loop(star, "model_class_1")
        assert fsc["rlnGoldStandardFsc"] == ["0.9", "0.1"]
        assert jo._star_loop(star, "nothing") == {}
        assert jo._star_loop(tmp_path / "missing.star", "x") == {}

    def test_a_value_of_a_block_without_a_loop(self, tmp_path):
        star = tmp_path / "p.star"
        star.write_text(
            "\ndata_general\n\n_rlnFinalResolution   2.68\n_rlnBfactorUsedForSharpening  -68.2\n\ndata_fsc\n\n_rlnFinalResolution 9\n"
        )
        assert jo._star_value(star, "general", "rlnFinalResolution") == "2.68"
        assert jo._star_value(star, "general", "rlnBfactorUsedForSharpening") == "-68.2"
        assert jo._star_value(star, "general", "rlnMissing") is None

    def test_the_highest_numbered_file(self, tmp_path):
        for n in (1, 9, 20, 3):
            (tmp_path / f"run_it{n:03d}_classes.mrcs").write_text("")
        (tmp_path / "run_it099_other.mrcs").write_text("")
        assert (
            jo._last_numbered(tmp_path, r"run_it(\d+)_classes\.mrcs").name
            == "run_it020_classes.mrcs"
        )
        assert jo._last_numbered(tmp_path, r"nothing(\d+)") is None
        assert jo._last_numbered(tmp_path / "missing", r"x(\d+)") is None

    def test_the_angles_of_the_first_particles(self, tmp_path):
        star = tmp_path / "run_data.star"
        star.write_text(
            "data_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n\n"
            + _loop(
                "particles",
                ["rlnCoordinateX", "rlnAngleRot", "rlnAngleTilt"],
                [(i, i * 2.0, 90.0 + i) for i in range(10)],
            )
        )
        rot, tilt = jo._particle_angles(star, cap=4)
        assert list(rot) == [0.0, 2.0, 4.0, 6.0] and list(tilt) == [
            90.0,
            91.0,
            92.0,
            93.0,
        ]

    def test_a_table_without_angles_gives_none(self, tmp_path):
        star = tmp_path / "d.star"
        star.write_text(_loop("particles", ["rlnCoordinateX"], [(1,), (2,)]))
        assert jo._particle_angles(star) is None


class TestPlots:
    def test_the_views_are_a_picture_with_the_particles_where_the_angles_are(
        self, qapp
    ):
        rng = np.random.default_rng(0)
        image = jo.view_distribution_image(
            rng.uniform(-180, 180, 4000), rng.normal(90, 4, 4000)
        )
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        assert len({tuple(p) for p in image.reshape(-1, 3)}) > 20

    def test_the_fsc_is_a_picture(self, qapp):
        x = np.linspace(0, 0.4, 50)
        image = jo._plot_fsc([("a", x, np.clip(1 - 4 * x, 0, 1))], (0.143, 0.5))
        assert image.shape[2] == 3 and image.std() > 10

    def test_without_a_qt_application_a_plot_is_left_out_not_a_crash(self, monkeypatch):
        from PySide6.QtGui import QGuiApplication

        monkeypatch.setattr(QGuiApplication, "instance", staticmethod(lambda: None))
        assert jo._plot_panel("FSC", lambda: jo._plot_fsc([], ()), "x") is None


@pytest.fixture
def relion(tmp_path):
    return tmp_path / "proj"


def _make_refine(job, data=True, model=True, mapfile=True):
    if mapfile:
        _mrc(job / "run_class001.mrc", _volume())
    if model:
        (job / "run_model.star").write_text(
            _loop(
                "model_class_1",
                [
                    "rlnSpectralIndex",
                    "rlnResolution",
                    "rlnAngstromResolution",
                    "rlnGoldStandardFsc",
                ],
                _fsc_rows(),
            )
        )
    if data:
        rng = np.random.default_rng(1)
        rows = [
            (f"{rng.uniform(-180, 180):.2f}", f"{rng.normal(90, 5):.2f}")
            for _ in range(500)
        ]
        (job / "run_data.star").write_text(
            "data_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n\n"
            + _loop("particles", ["rlnAngleRot", "rlnAngleTilt"], rows)
        )


class TestRelionJobs:
    def test_a_refinement_gives_the_map_the_fsc_and_the_views(self, relion):
        job = relion / "Refine3D" / "job014"
        job.mkdir(parents=True)
        _make_refine(job)
        outcome = jo.build_outcome(job, "relion")
        assert [p["title"] for p in outcome["panels"]] == [
            "Map, central section",
            "FSC",
            "Views",
        ]
        assert all(p["image"].dtype == np.uint8 for p in outcome["panels"])
        resolution = dict(outcome["facts"])["Resolution"]
        assert resolution.endswith("Å (FSC 0.143)")
        # the curve falls below 0.143 at about 1 - (x/0.2)^2 = 0.143: x = 0.1854, 5.39 Å
        assert float(resolution.split()[0]) == pytest.approx(5.4, abs=0.2)

    def test_what_is_missing_is_left_out_and_nothing_gives_none(self, relion):
        job = relion / "Refine3D" / "job001"
        job.mkdir(parents=True)
        _make_refine(job, data=False)
        outcome = jo.build_outcome(job, "relion")
        assert [p["title"] for p in outcome["panels"]] == [
            "Map, central section",
            "FSC",
        ]
        empty = relion / "Refine3D" / "job002"
        empty.mkdir()
        assert jo.build_outcome(empty, "relion") is None

    def test_the_map_of_the_last_iteration_when_the_final_one_is_not_there_yet(
        self, relion
    ):
        job = relion / "Refine3D" / "job003"
        job.mkdir(parents=True)
        _mrc(job / "run_it002_class001.mrc", _volume())
        _mrc(job / "run_it007_class001.mrc", _volume(seed=3))
        outcome = jo.build_outcome(job, "relion")
        assert "run_it007_class001.mrc" in outcome["panels"][0]["caption"]

    def test_post_processing_gives_the_resolution_the_map_and_the_fsc(self, relion):
        job = relion / "PostProcess" / "job018"
        job.mkdir(parents=True)
        _mrc(job / "postprocess.mrc", _volume())
        (job / "postprocess.star").write_text(
            "\ndata_general\n\n_rlnFinalResolution   2.681491\n_rlnBfactorUsedForSharpening   -68.23474\n"
            + _loop(
                "fsc",
                [
                    "rlnSpectralIndex",
                    "rlnResolution",
                    "rlnAngstromResolution",
                    "rlnFourierShellCorrelationCorrected",
                ],
                _fsc_rows(),
            )
        )
        outcome = jo.build_outcome(job, "relion")
        assert [p["title"] for p in outcome["panels"]] == [
            "Sharpened map, central section",
            "FSC",
        ]
        facts = dict(outcome["facts"])
        assert facts["Resolution"] == "2.68 Å (FSC 0.143)"
        assert facts["Sharpening B-factor"] == "-68.2 Å²"
        assert "2.68 Å" in outcome["panels"][1]["caption"]

    def test_a_2d_classification_shows_the_last_iteration_the_most_populated_first(
        self, relion
    ):
        job = relion / "Class2D" / "job010"
        job.mkdir(parents=True)
        for it in (1, 2):
            _mrc(
                job / f"run_it{it:03d}_classes.mrcs",
                np.random.default_rng(it).normal(size=(10, 16, 16)),
            )
        # the shares of the last iteration: class 8 has most, then class 3
        shares = [0.02] * 10
        shares[7], shares[2] = 0.4, 0.3
        (job / "run_it002_model.star").write_text(
            _loop(
                "model_classes",
                ["rlnReferenceImage", "rlnClassDistribution"],
                [
                    (f"{i + 1:06d}@Class2D/job010/run_it002_classes.mrcs", s)
                    for i, s in enumerate(shares)
                ],
            )
        )
        outcome = jo.build_outcome(job, "relion")
        assert outcome["title"] == "Class averages, iteration 2"
        caption = outcome["panels"][0]["caption"]
        assert caption.startswith("10 classes at iteration 2")
        assert "class 8 (40.0%), 3 (30.0%)" in caption

    def test_at_most_a_screenful_of_classes_is_shown(self, relion):
        job = relion / "Class2D" / "job011"
        job.mkdir(parents=True)
        _mrc(
            job / "run_it001_classes.mrcs",
            np.random.default_rng(0).normal(size=(100, 16, 16)),
        )
        outcome = jo.build_outcome(job, "relion")
        h, w = outcome["panels"][0]["image"].shape[:2]
        assert w >= jo.TILE * 5 and h <= jo.TILE * (jo.MAX_TILES // 5 + 2)
        assert outcome["panels"][0]["caption"].startswith("100 classes")

    def test_3d_classes_give_the_central_section_of_each(self, relion):
        job = relion / "Class3D" / "job020"
        job.mkdir(parents=True)
        for it in (1, 3):
            for k in (1, 2, 3):
                _mrc(
                    job / f"run_it{it:03d}_class{k:03d}.mrc", _volume(seed=10 * it + k)
                )
        (job / "run_it003_model.star").write_text(
            _loop(
                "model_classes",
                ["rlnReferenceImage", "rlnClassDistribution"],
                [
                    (f"Class3D/job020/run_it003_class{k:03d}.mrc", s)
                    for k, s in zip((1, 2, 3), (0.5, 0.3, 0.2))
                ],
            )
        )
        outcome = jo.build_outcome(job, "relion")
        assert outcome["title"] == "3D classes, iteration 3"
        assert "1 (50.0%), 2 (30.0%), 3 (20.0%)" in outcome["panels"][0]["caption"]

    def test_an_initial_model_and_a_mask(self, relion):
        job = relion / "InitialModel" / "job005"
        job.mkdir(parents=True)
        _mrc(job / "initial_model.mrc", _volume())
        assert jo.build_outcome(job, "relion")["title"] == "Initial model"
        mask = relion / "MaskCreate" / "job015"
        mask.mkdir(parents=True)
        _mrc(mask / "mask.mrc", (_volume() > 0.3).astype(np.float32))
        assert (
            jo.build_outcome(mask, "relion")["panels"][0]["title"]
            == "Mask, central section"
        )

    def test_a_job_of_a_kind_that_is_not_known_gives_none(self, relion):
        job = relion / "Import" / "job001"
        job.mkdir(parents=True)
        (job / "movies.star").write_text("x")
        assert jo.build_outcome(job, "relion") is None

    def test_a_file_that_cannot_be_read_gives_none_not_an_error(self, relion):
        job = relion / "Refine3D" / "job009"
        job.mkdir(parents=True)
        (job / "run_class001.mrc").write_bytes(b"not a map")
        (job / "run_model.star").write_text("garbage")
        assert jo.build_outcome(job, "relion") is None


class TestCryosparcJobs:
    @staticmethod
    def _cs(path, fields):
        n = len(next(iter(fields.values())))
        dtype = [(k, v.dtype, v.shape[1:]) for k, v in fields.items()]
        table = np.zeros(n, dtype=dtype)
        for k, v in fields.items():
            table[k] = v
        with open(path, "wb") as f:
            np.save(f, table)

    def test_class_averages_of_the_last_iteration_the_most_populated_first(
        self, tmp_path
    ):
        job = tmp_path / "J14"
        job.mkdir()
        _mrc(
            job / "J14_005_class_averages.mrc",
            np.random.default_rng(0).normal(size=(6, 16, 16)),
        )
        _mrc(
            job / "J14_020_class_averages.mrc",
            np.random.default_rng(1).normal(size=(6, 16, 16)),
        )
        classes = np.array([4] * 50 + [1] * 30 + [0] * 5, dtype=np.uint32)
        self._cs(
            job / "J14_020_particles.cs",
            {"uid": np.arange(85, dtype=np.uint64), "alignments2D/class": classes},
        )
        outcome = jo.build_outcome(job, "cryosparc", "class_2D_new")
        assert outcome["title"] == "Class averages, iteration 20"
        caption = outcome["panels"][0]["caption"]
        assert caption.startswith("6 classes at iteration 20")
        assert "class 5 (58.8%), 2 (35.3%), 1 (5.9%)" in caption

    def test_a_reconstruction_gives_the_map_and_the_views(self, tmp_path):
        job = tmp_path / "J20"
        job.mkdir()
        _mrc(job / "J20_001_volume_map.mrc", _volume(seed=1))
        _mrc(job / "J20_003_volume_map.mrc", _volume(seed=2))
        rng = np.random.default_rng(0)
        self._cs(
            job / "J20_003_particles.cs",
            {
                "uid": np.arange(300, dtype=np.uint64),
                "alignments3D/pose": rng.normal(scale=1.0, size=(300, 3)).astype(
                    np.float32
                ),
            },
        )
        outcome = jo.build_outcome(job, "cryosparc", "nonuniform_refine_new")
        assert [p["title"] for p in outcome["panels"]] == [
            "Map, central section",
            "Views",
        ]
        assert "J20_003_volume_map.mrc" in outcome["panels"][0]["caption"]

    def test_other_jobs_give_none(self, tmp_path):
        job = tmp_path / "J2"
        job.mkdir()
        assert (
            jo.build_outcome(job, "cryosparc", "patch_motion_correction_multi") is None
        )
        assert jo.build_outcome(job, "cryosparc", None) is None
        assert (
            jo.build_outcome(job, "cryosparc", "class_2D_new") is None
        )  # nothing made yet


def _cs_file(path, fields):
    """A ``.cs`` file: a structured numpy array; strings are bytes, as CryoSPARC writes them."""
    n = len(next(iter(fields.values())))
    table = np.zeros(n, dtype=[(k, v.dtype, v.shape[1:]) for k, v in fields.items()])
    for k, v in fields.items():
        table[k] = v
    with open(path, "wb") as f:
        np.save(f, table)


def _star_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestTheCtfOfRelion:
    @staticmethod
    def _micrographs(job, n=40):
        rng = np.random.default_rng(0)
        rows = [
            (
                f"mic{i}.mrc",
                f"{rng.uniform(10000, 25000):.1f}",
                f"{rng.uniform(10000, 25000):.1f}",
                f"{rng.uniform(2.5, 8):.2f}",
            )
            for i in range(n)
        ]
        _star_file(
            job / "micrographs_ctf.star",
            "data_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n"
            + _loop(
                "micrographs",
                [
                    "rlnMicrographName",
                    "rlnDefocusU",
                    "rlnDefocusV",
                    "rlnCtfMaxResolution",
                ],
                rows,
            ),
        )

    def test_ctffind_gives_the_histograms_of_the_defocus_and_of_the_fit_resolution(
        self, relion
    ):
        job = relion / "CtfFind" / "job004"
        self._micrographs(job)
        outcome = jo.build_outcome(job, "relion")
        assert [p["title"] for p in outcome["panels"]] == [
            "Defocus",
            "Maximum resolution of the CTF fit",
        ]
        facts = dict(outcome["facts"])
        assert facts["Micrographs"] == "40"
        assert facts["Defocus"].startswith("median 1.") or facts["Defocus"].startswith(
            "median 2."
        )

    def test_the_defocus_is_in_micrometres_from_the_mean_of_the_two_axes(self, relion):
        job = relion / "CtfFind" / "job001"
        _star_file(
            job / "micrographs_ctf.star",
            _loop(
                "micrographs",
                ["rlnDefocusU", "rlnDefocusV", "rlnCtfMaxResolution"],
                [(10000, 20000, 4.0), (20000, 30000, 6.0)],
            ),
        )
        facts = dict(jo.build_outcome(job, "relion")["facts"])
        assert facts["Defocus"].startswith("median 2.00 µm (1.50 to 2.50)")
        assert facts["Max resolution of the fit"] == "median 5.00 Å"

    def test_a_table_without_the_resolution_gives_the_defocus_only(self, relion):
        job = relion / "CtfFind" / "job002"
        _star_file(
            job / "micrographs_ctf.star",
            _loop(
                "micrographs",
                ["rlnDefocusU", "rlnDefocusV"],
                [(10000, 10000), (12000, 12000)],
            ),
        )
        assert [p["title"] for p in jo.build_outcome(job, "relion")["panels"]] == [
            "Defocus"
        ]

    def test_no_table_gives_none(self, relion):
        job = relion / "CtfFind" / "job003"
        job.mkdir(parents=True)
        assert jo.build_outcome(job, "relion") is None


class TestTheCtfOfCryosparc:
    def test_the_defocus_and_the_fit_resolution_come_from_the_exposures(self, tmp_path):
        job = tmp_path / "J7"
        job.mkdir()
        rng = np.random.default_rng(0)
        _cs_file(
            job / "exposures_ctf_estimated.cs",
            {
                "uid": np.arange(30, dtype=np.uint64),
                "ctf/df1_A": rng.uniform(10000, 20000, 30),
                "ctf/df2_A": rng.uniform(10000, 20000, 30),
                "ctf/ctf_fit_to_A": rng.uniform(3, 7, 30),
            },
        )
        outcome = jo.build_outcome(job, "cryosparc", "patch_ctf_estimation_multi")
        assert len(outcome["panels"]) == 2
        assert dict(outcome["facts"])["Micrographs"] == "30"

    def test_a_job_that_wrote_no_exposures_gives_none(self, tmp_path):
        job = tmp_path / "J8"
        job.mkdir()
        assert jo.build_outcome(job, "cryosparc", "patch_ctf_estimation_multi") is None


class TestTheCtfRefinement:
    @staticmethod
    def _particles(path, defocus):
        rows = [
            (f"{i + 1:06d}@Extract/job001/s.mrcs", d, d + 100.0)
            for i, d in enumerate(defocus)
        ]
        _star_file(
            path,
            "data_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n"
            + _loop("particles", ["rlnImageName", "rlnDefocusU", "rlnDefocusV"], rows),
        )

    def test_the_change_of_the_defocus_is_compared_with_the_input_of_the_job(
        self, relion
    ):
        before = relion / "CtfRefine" / "job001"
        job = relion / "CtfRefine" / "job002"
        before.mkdir(parents=True)
        job.mkdir(parents=True)
        base = np.linspace(10000, 20000, 200)
        self._particles(before / "particles_ctf_refine.star", base)
        self._particles(job / "particles_ctf_refine.star", base + 50.0)
        (job / "job.star").write_text(
            "data_joboptions_values\n\nloop_\n_rlnJobOptionVariable #1\n_rlnJobOptionValue #2\n"
            "   fn_data CtfRefine/job001/particles_ctf_refine.star \n"
        )
        outcome = jo.build_outcome(job, "relion")
        assert [p["title"] for p in outcome["panels"]] == ["Change of the defocus"]
        assert dict(outcome["facts"])["Defocus change"].startswith("median +50.0 Å")
        assert (
            "200 particles compared with job001/particles_ctf_refine.star"
            in outcome["panels"][0]["caption"]
        )

    def test_particles_are_matched_by_name_when_some_were_dropped(self, relion):
        before = relion / "CtfRefine" / "job001"
        job = relion / "CtfRefine" / "job002"
        before.mkdir(parents=True)
        job.mkdir(parents=True)
        base = np.linspace(10000, 20000, 100)
        self._particles(before / "particles_ctf_refine.star", base)
        keep = np.arange(100) >= 10
        rows = [
            (f"{i + 1:06d}@Extract/job001/s.mrcs", d + 30.0, d + 130.0)
            for i, d in enumerate(base)
            if keep[i]
        ]
        _star_file(
            job / "particles_ctf_refine.star",
            _loop("particles", ["rlnImageName", "rlnDefocusU", "rlnDefocusV"], rows),
        )
        (job / "job.star").write_text(
            "   fn_data CtfRefine/job001/particles_ctf_refine.star \n"
        )
        delta, _source = jo._defocus_change(job)
        assert len(delta) == 90 and np.allclose(delta, 30.0)

    def test_a_refinement_that_changed_no_defocus_shows_no_histogram(self, relion):
        before = relion / "CtfRefine" / "job001"
        job = relion / "CtfRefine" / "job002"
        before.mkdir(parents=True)
        job.mkdir(parents=True)
        self._particles(before / "particles_ctf_refine.star", [10000.0, 11000.0])
        self._particles(job / "particles_ctf_refine.star", [10000.0, 11000.0])
        (job / "job.star").write_text(
            "   fn_data CtfRefine/job001/particles_ctf_refine.star \n"
        )
        assert jo.build_outcome(job, "relion") is None

    def test_the_fitted_aberrations_are_shown_as_maps(self, relion):
        job = relion / "CtfRefine" / "job003"
        rng = np.random.default_rng(0)
        for name in (
            "beamtilt_delta-phase_iter-fit_optics-group_1_N-3.mrc",
            "aberr_delta-phase_iter-fit_optics-group_1_N-4.mrc",
            "mag_disp_x_fit_optics-group_1.mrc",
            "mag_disp_y_fit_optics-group_1.mrc",
        ):
            _mrc(job / name, rng.normal(size=(32, 32)))
        outcome = jo.build_outcome(job, "relion")
        assert [p["title"] for p in outcome["panels"]] == [
            "Beam tilt, fitted",
            "Higher-order aberrations, fitted",
            "Magnification, x, fitted",
            "Magnification, y, fitted",
        ]


class TestTheMotionCorrection:
    @staticmethod
    def _job(relion, n=12):
        job = relion / "MotionCorr" / "job003"
        rng = np.random.default_rng(0)
        total = np.linspace(3.0, 30.0, n)
        rows = []
        for i in range(n):
            meta = f"MotionCorr/job003/mic{i}.star"
            rows.append((f"mic{i}_PS.mrc", f"mic{i}.mrc", meta, 1, f"{total[i]:.3f}"))
            steps = rng.normal(size=(10, 2))
            steps *= total[i] / np.linalg.norm(steps, axis=1).sum()
            xy = np.vstack([[0, 0], np.cumsum(steps[:-1], axis=0)])
            _star_file(
                relion / meta,
                "data_general\n\n_rlnImageSizeZ 10\n\n"
                + _loop(
                    "global_shift",
                    [
                        "rlnMicrographFrameNumber",
                        "rlnMicrographShiftX",
                        "rlnMicrographShiftY",
                    ],
                    [(k + 1, x, y) for k, (x, y) in enumerate(xy)],
                ),
            )
        _star_file(
            job / "corrected_micrographs.star",
            _loop(
                "micrographs",
                [
                    "rlnCtfPowerSpectrum",
                    "rlnMicrographName",
                    "rlnMicrographMetadata",
                    "rlnOpticsGroup",
                    "rlnAccumMotionTotal",
                ],
                rows,
            ),
        )
        return job

    def test_the_histogram_of_the_total_motion_and_the_motion_of_a_few_micrographs(
        self, relion
    ):
        outcome = jo.build_outcome(self._job(relion), "relion")
        assert [p["title"] for p in outcome["panels"]] == [
            "Total motion",
            "Examples of motion",
        ]
        facts = dict(outcome["facts"])
        assert facts["Micrographs"] == "12"
        assert facts["Total motion"] == "median 16.5 Å (3.0 to 30.0)"
        assert outcome["panels"][1]["caption"].startswith(f"{jo.MOTIONS} micrographs")

    def test_without_the_files_of_the_frames_the_histogram_is_still_there(self, relion):
        job = self._job(relion)
        for star in (relion / "MotionCorr" / "job003").glob("mic*.star"):
            star.unlink()
        for star in (relion / "MotionCorr").glob("mic*.star"):
            star.unlink()
        # the per-micrograph files are in the project folder
        for star in relion.glob("MotionCorr/job003/mic*.star"):
            star.unlink()
        outcome = jo.build_outcome(job, "relion")
        assert [p["title"] for p in outcome["panels"]] == ["Total motion"]

    def test_the_frame_shifts_are_read_from_the_global_shift_table(self, relion):
        star = relion / "m.star"
        _star_file(
            star,
            _loop(
                "global_shift",
                [
                    "rlnMicrographFrameNumber",
                    "rlnMicrographShiftX",
                    "rlnMicrographShiftY",
                ],
                [(1, 0.0, 0.0), (2, 1.0, 2.0), (3, 1.5, 2.5)],
            ),
        )
        x, y = jo._frame_shifts(star)
        assert list(x) == [0.0, 1.0, 1.5] and list(y) == [0.0, 2.0, 2.5]
        assert jo._frame_shifts(relion / "missing.star") is None

    def test_the_trajectories_are_a_picture(self, qapp):
        x = np.cumsum(np.random.default_rng(0).normal(size=20))
        image = jo._plot_trajectories([("5 Å", x, x[::-1]), ("9 Å", x * 2, x)])
        assert image.shape[2] == 3 and image.std() > 10


class TestThePolishing:
    @staticmethod
    def _job(relion, frames=20, pixel_size=0.8):
        job = relion / "Polish" / "job022"
        rows = [(f, -5.0 - 4.0 * f, -0.01 * f) for f in range(frames)]
        _star_file(
            job / "bfactors.star",
            _loop(
                "perframe_bfactors",
                [
                    "rlnMovieFrameNumber",
                    "rlnBfactorUsedForSharpening",
                    "rlnFittedInterceptGuinierPlot",
                ],
                rows,
            ),
        )
        _star_file(
            job / "shiny.star",
            _loop("optics", ["rlnOpticsGroup", "rlnImagePixelSize"], [(1, pixel_size)]),
        )
        return job

    def test_the_contribution_of_the_frames_is_drawn_for_a_few_resolutions(
        self, relion
    ):
        outcome = jo.build_outcome(self._job(relion), "relion")
        assert [p["title"] for p in outcome["panels"]] == ["Contribution of the frames"]
        assert dict(outcome["facts"])["Frames"] == "20"

    def test_the_later_frames_count_less_and_more_so_at_a_high_resolution(self):
        frames = np.arange(10)
        curves = jo.frame_contributions(
            frames, -4.0 * frames, np.zeros(10), [10.0, 3.0]
        )
        coarse, fine = curves[0][2], curves[1][2]
        assert coarse.sum() == pytest.approx(1.0) and fine.sum() == pytest.approx(1.0)
        assert np.all(np.diff(coarse) < 0) and np.all(np.diff(fine) < 0)
        assert fine[0] > coarse[0] and fine[-1] < coarse[-1]

    def test_the_scale_of_a_frame_counts(self):
        shares = jo.frame_contributions([0, 1], [0.0, 0.0], [0.0, np.log(3.0)], [5.0])[
            0
        ][2]
        assert shares == pytest.approx([0.25, 0.75])

    def test_resolutions_finer_than_the_nyquist_limit_are_left_out(self, relion):
        outcome = jo.build_outcome(self._job(relion, pixel_size=1.4), "relion")
        assert outcome is not None  # 10, 5, 3 Å: 2.5 and 2 are beyond 2.8 Å

    def test_a_job_without_the_table_gives_none(self, relion):
        job = relion / "Polish" / "job001"
        job.mkdir(parents=True)
        assert jo.build_outcome(job, "relion") is None


class TestTheTablesOfParticles:
    def test_the_rows_and_the_distinct_values_are_counted_in_one_pass(self, tmp_path):
        star = tmp_path / "particles.star"
        rows = [(f"{i + 1:06d}@s.mrcs", f"m{i // 5}.mrc") for i in range(100)]
        _star_file(
            star,
            "data_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n"
            + _loop("particles", ["rlnImageName", "rlnMicrographName"], rows),
        )
        scan = jo._scan_table(
            star, ("particles",), ["rlnMicrographName", "rlnImageName"], keep=8
        )
        assert scan["rows"] == 100 and len(scan["distinct"]) == 20
        assert 4 <= len(scan["samples"]) < 16
        # spread over the table, not the first rows
        assert int(scan["samples"][-1][1].split("@")[0]) > 60

    def test_a_column_that_is_not_there_gives_none(self, tmp_path):
        star = tmp_path / "p.star"
        _star_file(star, _loop("particles", ["rlnImageName"], [("1@a.mrcs",)]))
        assert jo._scan_table(star, ("particles",), ["rlnNothing"]) is None
        assert jo._scan_table(tmp_path / "missing.star", ("particles",), ["x"]) is None


class TestTheExtraction:
    @staticmethod
    def _job(relion, particles=60, micrographs=6):
        job = relion / "Extract" / "job009"
        rng = np.random.default_rng(0)
        rows = []
        for m in range(micrographs):
            stack = f"Extract/job009/Movies/m{m}.mrcs"
            count = particles // micrographs
            _mrc(relion / stack, rng.normal(size=(count, 16, 16)))
            rows += [(f"{k + 1:06d}@{stack}", f"mic{m}.mrc") for k in range(count)]
        _star_file(
            job / "particles.star",
            "data_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n"
            + _loop("particles", ["rlnImageName", "rlnMicrographName"], rows),
        )
        return job

    def test_the_numbers_and_a_few_of_the_particles(self, relion):
        outcome = jo.build_outcome(self._job(relion), "relion")
        facts = dict(outcome["facts"])
        assert facts["Particles"] == "60"
        assert facts["Micrographs"] == "6"
        assert facts["Particles per micrograph"] == "10.0 on average"
        panel = outcome["panels"][0]
        assert panel["title"] == "Examples"
        assert panel["caption"].startswith(f"{jo.EXAMPLES} of the 60 particles")
        assert panel["image"].shape[1] >= jo.TILE * 4

    def test_the_numbers_are_there_when_the_stacks_are_not(self, relion):
        job = self._job(relion)
        for stack in (relion / "Extract" / "job009" / "Movies").glob("*.mrcs"):
            stack.unlink()
        outcome = jo.build_outcome(job, "relion")
        assert outcome["panels"] == [] and dict(outcome["facts"])["Particles"] == "60"

    def test_fewer_particles_than_examples_shows_them_all(self, relion):
        outcome = jo.build_outcome(
            self._job(relion, particles=6, micrographs=2), "relion"
        )
        assert outcome["panels"][0]["caption"].startswith("6 of the 6 particles")

    def test_cryosparc_counts_the_particles_and_the_micrographs(self, tmp_path):
        job = tmp_path / "J30"
        job.mkdir()
        n, per = 40, 10
        _mrc(
            tmp_path / "J30" / "extract" / "a.mrc",
            np.random.default_rng(0).normal(size=(per, 16, 16)),
        )
        _cs_file(
            job / "extracted_particles.cs",
            {
                "uid": np.arange(n, dtype=np.uint64),
                "blob/path": np.array([b">J30/extract/a.mrc"] * n),
                "blob/idx": np.arange(n, dtype=np.uint32) % per,
                "location/micrograph_uid": np.arange(n, dtype=np.uint64) // per,
            },
        )
        outcome = jo.build_outcome(job, "cryosparc", "extract_micrographs_multi")
        facts = dict(outcome["facts"])
        assert facts["Particles"] == "40" and facts["Micrographs"] == "4"
        assert facts["Particles per micrograph"] == "10.0 on average"
        assert outcome["panels"][0]["caption"].startswith(f"{jo.EXAMPLES} of the 40")


class TestTheImport:
    @pytest.mark.parametrize(
        ("name", "block", "column", "label"),
        [
            ("movies.star", "movies", "rlnMicrographMovieName", "Movies imported"),
            (
                "micrographs.star",
                "micrographs",
                "rlnMicrographName",
                "Micrographs imported",
            ),
            ("particles.star", "particles", "rlnImageName", "Particles imported"),
        ],
    )
    def test_relion_counts_what_was_imported(self, relion, name, block, column, label):
        job = relion / "Import" / "job001"
        _star_file(job / name, _loop(block, [column], [(f"f{i}",) for i in range(7)]))
        outcome = jo.build_outcome(job, "relion")
        assert outcome["facts"] == [(label, "7")] and outcome["panels"] == []

    def test_an_import_of_another_kind_gives_none(self, relion):
        job = relion / "Import" / "job002"
        job.mkdir(parents=True)
        (job / "ref.mrc").write_text("x")
        assert jo.build_outcome(job, "relion") is None

    @pytest.mark.parametrize(
        ("field", "job_type", "label"),
        [
            ("movie_blob/path", "import_movies", "Movies imported"),
            ("micrograph_blob/path", "import_micrographs", "Micrographs imported"),
            ("blob/path", "import_particles", "Particles imported"),
        ],
    )
    def test_cryosparc_counts_what_was_imported(self, tmp_path, field, job_type, label):
        job = tmp_path / "J1"
        job.mkdir()
        _cs_file(
            job / "imported_movies.cs",
            {"uid": np.arange(5, dtype=np.uint64), field: np.array([b"x"] * 5)},
        )
        assert jo.build_outcome(job, "cryosparc", job_type)["facts"] == [(label, "5")]


class TestThePicking:
    @staticmethod
    def _job(
        relion,
        kind="ManualPick",
        picks=(3, 5, 0, 4),
        helical=False,
        option="do_startend",
    ):
        job = relion / kind / "job005"
        listing = []
        for i, count in enumerate(picks):
            name = f"{kind}/job005/mic{i}_coords.star"
            listing.append((f"mic{i}.mrc", name))
            if count:
                _star_file(
                    relion / name,
                    _loop(
                        "",
                        ["rlnCoordinateX", "rlnCoordinateY"],
                        [(k, k) for k in range(count)],
                    ),
                )
        _star_file(
            job / f"{kind.lower()}.star",
            _loop(
                "coordinate_files",
                ["rlnMicrographName", "rlnMicrographCoordinates"],
                listing,
            ),
        )
        if helical:
            (job / "job.star").write_text(f"   {option} Yes \n")
        return job

    def test_particles_picked_by_hand(self, relion):
        facts = dict(jo.build_outcome(self._job(relion), "relion")["facts"])
        assert facts["Particles"] == "12"
        assert facts["Micrographs"] == "4"
        assert facts["Particles per micrograph"] == "3.0 on average"

    def test_a_filament_is_picked_by_its_start_and_its_end(self, relion):
        facts = dict(
            jo.build_outcome(
                self._job(relion, picks=(4, 2, 2, 4), helical=True), "relion"
            )["facts"]
        )
        assert facts["Filaments"] == "6"
        assert facts["Filaments per micrograph"] == "1.5 on average"

    def test_helical_segments_picked_automatically_are_segments(self, relion):
        job = self._job(
            relion,
            "AutoPick",
            picks=(10, 20),
            helical=True,
            option="do_pick_helical_segments",
        )
        facts = dict(jo.build_outcome(job, "relion")["facts"])
        assert facts["Helical segments"] == "30"

    def test_a_large_project_is_counted_from_some_of_the_micrographs(
        self, relion, monkeypatch
    ):
        monkeypatch.setattr(jo, "MAX_COORDINATE_FILES", 4)
        job = self._job(relion, picks=(2,) * 12)
        facts = dict(jo.build_outcome(job, "relion")["facts"])
        assert facts["Particles"] == "about 24" and facts["Micrographs"] == "12"

    def test_without_the_coordinate_files_nothing_is_counted(self, relion):
        job = self._job(relion)
        for star in relion.glob("ManualPick/job005/mic*_coords.star"):
            star.unlink()
        assert jo.build_outcome(job, "relion") is None

    def test_cryosparc_particles_and_filaments(self, tmp_path):
        job = tmp_path / "J9"
        job.mkdir()
        n = 30
        _cs_file(
            job / "picked_particles.cs",
            {
                "uid": np.arange(n, dtype=np.uint64),
                "location/micrograph_uid": np.arange(n, dtype=np.uint64) // 15,
            },
        )
        facts = dict(jo.build_outcome(job, "cryosparc", "blob_picker_gpu")["facts"])
        assert facts["Particles"] == "30" and facts["Micrographs"] == "2"
        _cs_file(
            job / "picked_particles.cs",
            {
                "uid": np.arange(n, dtype=np.uint64),
                "location/micrograph_uid": np.arange(n, dtype=np.uint64) // 15,
                "filament/filament_uid": np.arange(n, dtype=np.uint64) // 5,
            },
        )
        facts = dict(jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")["facts"])
        assert facts["Filaments"] == "6" and facts["Segments"] == "30"


class TestTheSelection:
    @staticmethod
    def _classification(relion, classes=10, per_class=20, kind="Class2D"):
        job = relion / kind / "job010"
        rows = []
        for k in range(classes):
            rows += [(f"{i + 1:06d}@Extract/s.mrcs", k + 1) for i in range(per_class)]
        _star_file(
            job / "run_it020_data.star",
            "data_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n"
            + _loop("particles", ["rlnImageName", "rlnClassNumber"], rows),
        )
        _star_file(
            job / "run_it020_model.star",
            _loop(
                "model_classes",
                ["rlnReferenceImage", "rlnClassDistribution"],
                [(f"{k + 1:06d}@x.mrcs", 1.0 / classes) for k in range(classes)],
            ),
        )
        (job / "run_it020_optimiser.star").write_text("data_optimiser_general\n")
        return job

    @staticmethod
    def _select(
        relion, chosen, per_class=20, model="Class2D/job010/run_it020_optimiser.star"
    ):
        job = relion / "Select" / "job011"
        rows = [
            (f"{i + 1:06d}@Extract/s.mrcs", k) for k in chosen for i in range(per_class)
        ]
        _star_file(
            job / "particles.star",
            "data_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n"
            + _loop("particles", ["rlnImageName", "rlnClassNumber"], rows),
        )
        (job / "job.star").write_text(f'   fn_model {model} \n    fn_data "" \n')
        return job

    def test_2d_classes_and_particles_selected_of_all_of_them(self, relion):
        self._classification(relion)
        job = self._select(relion, [2, 5, 7])
        _star_file(
            job / "class_averages.star",
            _loop("", ["rlnReferenceImage"], [(f"{k:06d}@c.mrcs",) for k in (1, 2, 3)]),
        )
        outcome = jo.build_outcome(job, "relion")
        facts = dict(outcome["facts"])
        assert facts["2D classes selected"] == "3 of 10"
        assert facts["Particles selected"] == "60 of 200 (30.0%)"

    def test_the_selected_class_averages_are_shown(self, relion):
        self._classification(relion)
        job = self._select(relion, [2, 5])
        _mrc(
            job / "class_averages.mrcs",
            np.random.default_rng(0).normal(size=(2, 16, 16)),
        )
        outcome = jo.build_outcome(job, "relion")
        assert [p["title"] for p in outcome["panels"]] == ["Selected class averages"]
        assert outcome["panels"][0]["caption"] == "2 selected classes"

    def test_3d_classes_are_counted_from_the_class_numbers_of_the_particles(
        self, relion
    ):
        self._classification(relion, classes=4, kind="Class3D")
        job = self._select(
            relion, [1, 3], model="Class3D/job010/run_it020_optimiser.star"
        )
        facts = dict(jo.build_outcome(job, "relion")["facts"])
        assert facts["3D classes selected"] == "2 of 4"
        assert facts["Particles selected"] == "40 of 80 (50.0%)"

    def test_a_selection_of_particles_gives_the_number_selected_of_the_input(
        self, relion
    ):
        job = relion / "Select" / "job020"
        _star_file(
            relion / "Extract" / "job001" / "particles.star",
            _loop("particles", ["rlnImageName"], [(f"{i}@s.mrcs",) for i in range(50)]),
        )
        _star_file(
            job / "particles.star",
            _loop("particles", ["rlnImageName"], [(f"{i}@s.mrcs",) for i in range(20)]),
        )
        (job / "job.star").write_text(
            '   fn_data Extract/job001/particles.star \n   fn_model "" \n'
        )
        facts = dict(jo.build_outcome(job, "relion")["facts"])
        assert facts == {"Particles selected": "20 of 50 (40.0%)"}

    def test_when_the_classification_is_gone_only_what_was_kept_is_counted(
        self, relion
    ):
        job = self._select(relion, [1, 2])
        facts = dict(jo.build_outcome(job, "relion")["facts"])
        assert facts["Particles selected"] == "40"
        assert facts["Classes selected"] == "2"

    def test_no_selected_particles_gives_none(self, relion):
        job = relion / "Select" / "job030"
        job.mkdir(parents=True)
        assert jo.build_outcome(job, "relion") is None

    def test_cryosparc_counts_the_selected_and_the_excluded(self, tmp_path):
        job = tmp_path / "J12"
        job.mkdir()
        _cs_file(
            job / "particles_selected.cs",
            {
                "uid": np.arange(30, dtype=np.uint64),
                "alignments2D/class": np.arange(30, dtype=np.uint32) % 3,
            },
        )
        _cs_file(
            job / "particles_excluded.cs",
            {
                "uid": np.arange(70, dtype=np.uint64),
                "alignments2D/class": 3 + np.arange(70, dtype=np.uint32) % 7,
            },
        )
        facts = dict(jo.build_outcome(job, "cryosparc", "select_2D")["facts"])
        assert facts["2D classes selected"] == "3 of 10"
        assert facts["Particles selected"] == "30 of 100 (30.0%)"


class TestPlotsOfOddValues:
    def test_a_histogram_of_values_that_are_all_the_same_is_drawn(self, qapp):
        image = jo._plot_histogram(
            np.full(50, 3.0) + 1e-13 * np.arange(50), "x", "t", 3.0
        )
        assert image.shape[2] == 3 and image.std() > 5

    def test_a_histogram_with_a_few_outliers_keeps_the_range_of_the_rest(self, qapp):
        values = np.concatenate(
            [np.random.default_rng(0).normal(5, 0.2, 500), [1000.0]]
        )
        image = jo._plot_histogram(values, "x", "t")
        assert image.std() > 10


class TestCryosparcAsItWritesItsFiles:
    """The tables of a CryoSPARC job as the jobs write them: only the new fields in the table of the
    job, the others in a ``passthrough`` table with the same rows."""

    def test_the_filaments_come_from_the_passthrough_table(self, tmp_path):
        job = tmp_path / "J8"
        job.mkdir()
        n = 60
        _cs_file(
            job / "extracted_particles.cs",
            {
                "uid": np.arange(n, dtype=np.uint64),
                "location/micrograph_uid": np.arange(n, dtype=np.uint64) // 30,
            },
        )
        _cs_file(
            job / "J8_passthrough_particles.cs",
            {
                "uid": np.arange(n, dtype=np.uint64),
                "filament/filament_uid": np.arange(n, dtype=np.uint64) // 10,
            },
        )
        _cs_file(job / "micrographs.cs", {"uid": np.arange(2, dtype=np.uint64)})
        facts = dict(jo.build_outcome(job, "cryosparc", "inspect_picks_v2")["facts"])
        assert facts["Filaments"] == "6" and facts["Segments"] == "60"
        assert facts["Micrographs"] == "2"

    def test_a_table_of_millions_of_rows_is_counted_from_blocks_spread_over_it(
        self, monkeypatch
    ):
        monkeypatch.setattr(jo, "EXACT_BYTES", 8000)
        monkeypatch.setattr(jo, "BLOCK_ROWS", 400)
        n = 20_000
        filament = np.arange(n, dtype=np.uint64) // 20  # 1000 filaments of 20 segments
        micrograph = filament // 50
        count, exact = jo._cs_distinct([micrograph, filament], n, 8)
        assert not exact and abs(count - 1000) <= 40
        small, exact = jo._cs_distinct([micrograph, filament], 900, 8)
        assert exact and small == 45

    def test_the_selected_templates_are_the_classes_selected(self, tmp_path):
        job = tmp_path / "J33"
        job.mkdir()
        classes = np.arange(100, dtype=np.uint32) % 4
        _cs_file(
            job / "particles_selected.cs",
            {"uid": np.arange(30, dtype=np.uint64), "alignments2D/class": classes[:30]},
        )
        _cs_file(
            job / "particles_excluded.cs",
            {"uid": np.arange(70, dtype=np.uint64), "alignments2D/class": classes[:70]},
        )
        _cs_file(job / "templates_selected.cs", {"uid": np.arange(5, dtype=np.uint64)})
        _cs_file(job / "templates_excluded.cs", {"uid": np.arange(15, dtype=np.uint64)})
        _mrc(
            job / "templates_selected.mrc",
            np.random.default_rng(0).normal(size=(5, 16, 16)),
        )
        outcome = jo.build_outcome(job, "cryosparc", "select_2D")
        facts = dict(outcome["facts"])
        assert facts["2D classes selected"] == "5 of 20"
        assert facts["Particles selected"] == "30 of 100 (30.0%)"
        assert outcome["panels"][0]["caption"] == "5 selected classes"

    def test_the_rigid_trajectories_are_read_from_the_files_the_table_names(
        self, tmp_path
    ):
        job = tmp_path / "J4"
        (tmp_path / "J4" / "motioncorrected").mkdir(parents=True)
        n, frames = 8, 12
        names = []
        for i in range(n):
            xy = np.cumsum(np.random.default_rng(i).normal(size=(frames, 2)), axis=0)[
                None
            ]
            name = f"J4/motioncorrected/m{i}_rigid_traj.npy"
            np.save(tmp_path / name, xy)
            names.append(name.encode())
        _cs_file(
            job / "micrographs_rigid_aligned.cs",
            {
                "uid": np.arange(n, dtype=np.uint64),
                "rigid_motion/path": np.array(names),
                "rigid_motion/idx": np.zeros(n, dtype=np.uint32),
                "rigid_motion/psize_A": np.full(n, 0.5, dtype=np.float32),
            },
        )
        outcome = jo.build_outcome(job, "cryosparc", "patch_motion_correction_multi")
        assert [p["title"] for p in outcome["panels"]] == [
            "Total motion",
            "Examples of motion",
        ]
        assert dict(outcome["facts"])["Micrographs"] == "8"
        # the length of the path in Å: the pixel size times the sum of the steps
        totals = []
        for i in range(n):
            xy = (
                np.cumsum(np.random.default_rng(i).normal(size=(frames, 2)), axis=0)
                * 0.5
            )
            totals.append(np.hypot(*np.diff(xy, axis=0).T).sum())
        assert dict(outcome["facts"])["Total motion"].startswith(
            f"median {np.median(totals):.1f} Å ({min(totals):.1f} to {max(totals):.1f})"
        )

    def test_the_maps_of_the_classes_of_a_3d_classification(self, tmp_path):
        job = tmp_path / "J40"
        job.mkdir()
        for k, size in enumerate((30, 10, 60)):
            _mrc(job / f"J40_class_{k:02d}_00012_volume.mrc", _volume(seed=k))
            _cs_file(
                job / f"J40_class_{k:02d}_00012_particles.cs",
                {"uid": np.arange(size, dtype=np.uint64)},
            )
        _mrc(job / "J40_class_00_00005_volume.mrc", _volume(seed=9))
        outcome = jo.build_outcome(job, "cryosparc", "class_3D")
        assert outcome["title"] == "3D classes, iteration 12"
        assert "1 (30.0%), 2 (10.0%), 3 (60.0%)" in outcome["panels"][0]["caption"]
        assert dict(outcome["facts"])["Particles"] == "100"

    def test_a_reconstruction_without_an_iteration_number_in_its_names(self, tmp_path):
        job = tmp_path / "J26"
        job.mkdir()
        _mrc(job / "J26_volume_map.mrc", _volume())
        _mrc(job / "J26_volume_map_half_A.mrc", _volume(seed=1))
        rng = np.random.default_rng(0)
        _cs_file(
            job / "J26_particles.cs",
            {
                "uid": np.arange(200, dtype=np.uint64),
                "alignments3D/pose": rng.normal(size=(200, 3)).astype(np.float32),
            },
        )
        outcome = jo.build_outcome(job, "cryosparc", "homo_reconstruct")
        assert [p["title"] for p in outcome["panels"]] == [
            "Map, central section",
            "Views",
        ]
        assert outcome["panels"][0]["caption"].startswith("J26_volume_map.mrc")

    def test_the_failed_movies_are_not_the_imported_ones(self, tmp_path):
        job = tmp_path / "J1"
        job.mkdir()
        _cs_file(
            job / "failed_movies.cs",
            {
                "uid": np.arange(0, dtype=np.uint64),
                "movie_blob/path": np.array([], dtype="S3"),
            },
        )
        _cs_file(
            job / "imported_movies.cs",
            {
                "uid": np.arange(9, dtype=np.uint64),
                "movie_blob/path": np.array([b"x"] * 9),
            },
        )
        assert jo.build_outcome(job, "cryosparc", "import_movies")["facts"] == [
            ("Movies imported", "9")
        ]


class TestCountingSlowlyAndKeepingTheAnswer:
    @staticmethod
    def _picked(job, segments=6000, per_filament=20):
        job.mkdir(parents=True, exist_ok=True)
        filament = np.arange(segments, dtype=np.uint64) // per_filament
        _cs_file(
            job / "picked_particles.cs",
            {
                "uid": np.arange(segments, dtype=np.uint64),
                "location/micrograph_uid": filament // 10,
                "filament/filament_uid": filament,
                "padding": np.zeros((segments, 40), dtype=np.float64),
            },
        )

    @pytest.fixture(autouse=True)
    def small_limit(self, monkeypatch):
        monkeypatch.setattr(jo, "EXACT_BYTES", 100_000)
        monkeypatch.setattr(jo, "BLOCK_ROWS", 300)

    def test_a_large_table_gives_an_estimate_and_the_way_to_the_exact_number(
        self, tmp_path
    ):
        job = tmp_path / "J5"
        self._picked(job)
        outcome = jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")
        facts = dict(outcome["facts"])
        assert facts["Filaments"].startswith("about ")
        assert "estimated" in facts[""]
        exact = outcome["refine"]()
        assert dict(exact["facts"])["Filaments"] == "300" and "refine" not in exact
        assert "" not in dict(exact["facts"])

    def test_the_exact_number_is_there_at_once_the_next_time(self, tmp_path):
        job = tmp_path / "J5"
        self._picked(job)
        jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")["refine"]()
        again = jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")
        assert dict(again["facts"])["Filaments"] == "300" and "refine" not in again

    def test_a_table_that_was_written_again_is_counted_again(self, tmp_path):
        job = tmp_path / "J5"
        self._picked(job)
        jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")["refine"]()
        self._picked(job, segments=8000)
        again = jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")
        assert dict(again["facts"])["Filaments"].startswith("about ")

    def test_counting_in_chunks_gives_the_same_number_as_at_once(self, monkeypatch):
        n = 5000
        filament = np.arange(n, dtype=np.uint64) // 7
        micrograph = filament // 3
        monkeypatch.setattr(jo, "EXACT_BYTES", 8 * 200)  # chunks of 200 rows
        found, exact = jo._cs_distinct([micrograph, filament], n, 8, True)
        assert exact and found == len(set(filament.tolist()))

    def test_the_outcome_is_kept_in_the_cache_of_helicon(self, tmp_path, monkeypatch):
        job = tmp_path / "J5"
        self._picked(job)
        made = []
        real = jo._build
        monkeypatch.setattr(jo, "_build", lambda *a: made.append(a) or real(*a))
        first = jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")
        again = jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")
        assert len(made) == 1  # the second came from the cache
        assert first["facts"] == again["facts"]
        assert (tmp_path / "cache" / "job_outcome").is_dir()

    def test_a_job_that_changed_is_read_again(self, tmp_path, monkeypatch):
        job = tmp_path / "J5"
        self._picked(job, segments=4000)
        made = []
        real = jo._build
        monkeypatch.setattr(jo, "_build", lambda *a: made.append(a) or real(*a))
        jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")
        self._picked(job, segments=5000)
        jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")
        (job / "new_file.txt").write_text("a job that goes on writes files")
        jo.build_outcome(job, "cryosparc", "filament_tracer_gpu")
        assert len(made) == 3

    def test_the_outcome_of_a_relion_job_with_pictures_is_kept_too(
        self, relion, monkeypatch
    ):
        job = relion / "Refine3D" / "job014"
        job.mkdir(parents=True)
        _make_refine(job)
        made = []
        real = jo._build
        monkeypatch.setattr(jo, "_build", lambda *a: made.append(a) or real(*a))
        first = jo.build_outcome(job, "relion")
        again = jo.build_outcome(job, "relion")
        assert len(made) == 1
        assert [p["title"] for p in again["panels"]] == [
            p["title"] for p in first["panels"]
        ]
        assert (again["panels"][0]["image"] == first["panels"][0]["image"]).all()

    def test_it_can_be_made_without_the_cache(self, tmp_path):
        job = tmp_path / "J5"
        self._picked(job)
        jo.build_outcome(job, "cryosparc", "filament_tracer_gpu", use_cache=False)
        assert not (tmp_path / "cache" / "job_outcome").exists()

    def test_what_is_not_an_outcome_is_kept_too_and_does_not_fail(self, relion):
        job = relion / "Import" / "job001"
        job.mkdir(parents=True)
        assert jo.build_outcome(job, "relion") is None
        assert jo.build_outcome(job, "relion") is None

    def test_the_signature_is_the_names_sizes_and_times_of_the_folder(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        first = jo._signature(tmp_path)
        (tmp_path / "a.txt").write_text("ab")
        assert jo._signature(tmp_path) != first
        assert jo._signature(tmp_path / "missing") == ()

    def test_the_pane_shows_the_estimate_and_then_the_exact_number(self, qapp):
        pane = bp.PreviewPane()
        pane.show()
        outcome = dict(
            title="Picking",
            panels=[],
            facts=[("Filaments", "about 5"), ("", "estimated")],
            refine=lambda: dict(panels=[], facts=[("Filaments", "7")]),
        )
        info = dict(
            title="J1", facts=[("Type", "x")] + outcome["facts"], outcome=outcome
        )
        pane._serial += 1
        pane._on_ready(pane._serial, info)
        assert "about 5" in pane._facts.text()
        for _ in range(200):
            qapp.processEvents()
            if "about 5" not in pane._facts.text():
                break
            import time

            time.sleep(0.02)
        text = pane._facts.text()
        assert ">7<" in text and "estimated" not in text and ">x<" in text

    def test_an_answer_for_a_path_no_longer_shown_is_dropped(self, qapp):
        pane = bp.PreviewPane()
        pane._serial = 4
        pane._base_facts = [("Type", "x")]
        pane._on_refined(3, dict(facts=[("Filaments", "7")]))
        assert ">7<" not in pane._facts.text()


class TestMoreJobTypes:
    def test_a_reconstruction_gives_the_central_section_of_the_map_or_of_a_half(
        self, relion
    ):
        job = relion / "Reconstruct3D" / "job100"
        _mrc(job / "run_class001_half1.mrc", _volume())
        outcome = jo.build_outcome(job, "relion")
        assert outcome["panels"][0]["caption"].startswith("run_class001_half1.mrc")
        _mrc(job / "run_class001.mrc", _volume(seed=2))
        assert jo.build_outcome(job, "relion")["panels"][0]["caption"].startswith(
            "run_class001.mrc"
        )

    def test_the_local_resolution_is_a_section_and_a_histogram_of_the_values_inside_the_map(
        self, relion
    ):
        job = relion / "LocalRes" / "job031"
        volume = np.zeros((20, 20, 20), dtype=np.float32)
        volume[5:15, 5:15, 5:15] = np.random.default_rng(0).uniform(3, 6, (10, 10, 10))
        _mrc(job / "relion_locres.mrc", volume)
        outcome = jo.build_outcome(job, "relion")
        assert [p["title"] for p in outcome["panels"]] == [
            "Local resolution, central section",
            "Histogram of the local resolution",
        ]
        text = dict(outcome["facts"])["Local resolution"]
        assert (
            text.startswith("median 4.") and "(3." in text
        )  # the empty part is not counted

    def test_modelangelo_counts_chains_residues_and_atoms(self, relion):
        job = relion / "ModelAngelo" / "job032"
        header = "data_1\n#\nloop_\n" + "".join(
            f"_atom_site.{name}\n"
            for name in (
                "group_PDB",
                "id",
                "type_symbol",
                "label_atom_id",
                "label_alt_id",
                "label_comp_id",
                "label_asym_id",
                "label_entity_id",
                "label_seq_id",
            )
        )
        rows = "".join(
            f"ATOM {i} C CA . MET {chain} ? {residue}\n"
            for i, (chain, residue) in enumerate(
                [("A", 1), ("A", 1), ("A", 2), ("B", 1)]
            )
        )
        _star_file(job / "job032.cif", header + rows)
        _star_file(job / "job032_raw.cif", header + rows * 3)
        facts = dict(jo.build_outcome(job, "relion")["facts"])
        assert (
            facts["Chains"] == "2"
            and facts["Residues"] == "3"
            and facts["Atoms"] == "4"
        )
        assert facts["Model"] == "job032.cif"

    def test_cryosparc_sharpening_curation_subsets_and_downsampling(self, tmp_path):
        job = tmp_path / "J1"
        _mrc(job / "J1_map_sharp.mrc", _volume())
        assert (
            jo.build_outcome(job, "cryosparc", "sharpen")["panels"][0]["title"]
            == "Sharpened map, central section"
        )
        job = tmp_path / "J2"
        job.mkdir()
        _cs_file(
            job / "0_exposures_accepted.cs", {"uid": np.arange(90, dtype=np.uint64)}
        )
        _cs_file(
            job / "0_exposures_rejected.cs", {"uid": np.arange(10, dtype=np.uint64)}
        )
        assert dict(
            jo.build_outcome(job, "cryosparc", "curate_exposures_v2")["facts"]
        ) == {"Exposures accepted": "90 of 100 (90.0%)"}
        job = tmp_path / "J3"
        job.mkdir()
        for k in range(3):
            _cs_file(
                job / f"J3_particles_set_{k}.cs",
                {"uid": np.arange(40, dtype=np.uint64)},
            )
        facts = dict(jo.build_outcome(job, "cryosparc", "subset_particles")["facts"])
        assert facts == {"Subsets": "3", "Particles": "40 in each, 120 in all"}
        job = tmp_path / "J4"
        job.mkdir()
        _cs_file(
            job / "downsampled_particles.cs",
            {
                "uid": np.arange(5, dtype=np.uint64),
                "blob/shape": np.tile(np.array([[128, 128]], dtype=np.uint32), (5, 1)),
                "blob/psize_A": np.full(5, 1.25, dtype=np.float32),
            },
        )
        facts = dict(
            jo.build_outcome(job, "cryosparc", "downsample_particles")["facts"]
        )
        assert facts == {"Particles": "5", "Box": "128 pixels of 1.250 Å"}

    def test_cryosparc_ctf_refinement_gives_the_particles_not_a_reconstruction(
        self, tmp_path
    ):
        job = tmp_path / "J5"
        job.mkdir()
        _cs_file(
            job / "particles.cs",
            {
                "uid": np.arange(12, dtype=np.uint64),
                "alignments3D/pose": np.zeros((12, 3), dtype=np.float32),
            },
        )
        outcome = jo.build_outcome(job, "cryosparc", "ctf_refine_global")
        assert outcome["panels"] == [] and dict(outcome["facts"]) == {"Particles": "12"}

    @pytest.mark.parametrize(
        ("job_type", "stem"), [("hetero_refine", "00062"), ("homo_abinit", "final")]
    )
    def test_the_classes_of_a_heterogeneous_refinement_or_an_ab_initio_run(
        self, tmp_path, job_type, stem
    ):
        job = tmp_path / "J6"
        job.mkdir()
        for k, size in enumerate((30, 70)):
            _mrc(job / f"J6_class_{k:02d}_{stem}_volume.mrc", _volume(seed=k))
            _cs_file(
                job / f"J6_class_{k:02d}_{stem}_particles.cs",
                {"uid": np.arange(size, dtype=np.uint64)},
            )
        outcome = jo.build_outcome(job, "cryosparc", job_type)
        assert "1 (30.0%), 2 (70.0%)" in outcome["panels"][0]["caption"]
        expected = "3D classes" if stem == "final" else "3D classes, iteration 62"
        assert outcome["title"] == expected

    def test_the_frame_weights_of_a_reference_based_motion_correction(self, tmp_path):
        job = tmp_path / "J7"
        job.mkdir()
        frames, bins = 20, 100
        x = np.linspace(0, 1, bins)[None, :]
        decay = np.exp(-np.arange(frames)[:, None] * (0.05 + 2 * x))
        weights = decay / decay.sum(axis=0) * frames
        np.save(job / "refm_empirical_dw.npy", weights.astype(np.float32))
        _cs_file(
            job / "hyperparams.cs",
            {
                "uid": np.arange(1, dtype=np.uint64),
                "refmotion_doseweights/psize_A": np.full(1, 0.5, dtype=np.float32),
            },
        )
        _cs_file(job / "movies.cs", {"uid": np.arange(30, dtype=np.uint64)})
        _cs_file(job / "movies_failed.cs", {"uid": np.arange(2, dtype=np.uint64)})
        _cs_file(job / "particles_0.cs", {"uid": np.arange(500, dtype=np.uint64)})
        outcome = jo.build_outcome(job, "cryosparc", "reference_motion_correction")
        assert [p["title"] for p in outcome["panels"]] == ["Contribution of the frames"]
        assert dict(outcome["facts"]) == {
            "Frames": "20",
            "Movies": "30 (2 failed)",
            "Particles": "500",
        }

    def test_the_resolutions_of_the_frame_contribution_stop_at_the_nyquist_limit(self):
        assert jo._contribution_resolutions(0.0) == [10.0, 5.0, 3.0, 2.0, 1.5]
        assert jo._contribution_resolutions(2.4) == [10.0, 5.0, 3.0]
        assert jo._contribution_resolutions(9.0) == [10.0, 5.0]  # always two


class TestTheAxisOfAHistogram:
    @pytest.mark.parametrize(
        ("span", "step"),
        [
            (0.3, 0.05),
            (0.5, 0.1),
            (1.4, 0.5),
            (2.4, 0.5),
            (4, 1),
            (6, 2),
            (15, 5),
            (45, 10),
        ],
    )
    def test_the_ticks_are_whole_numbers_or_halves_for_the_usual_ranges(
        self, span, step
    ):
        assert jo._tick_step(span) == pytest.approx(step)

    def test_the_range_ends_at_ticks(self):
        values = np.random.default_rng(0).normal(3.1, 0.4, 400)
        low, high, step = jo._histogram_limits(values)
        assert step == 0.5 and low % 0.5 == 0 and high % 0.5 == 0
        assert low <= values.min() + 1 and high >= np.percentile(values, 99)

    def test_the_resolution_axis_ends_where_5_percent_are_beyond_but_not_after_10(self):
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.normal(3.1, 0.3, 380), rng.uniform(10, 40, 20)])
        _low, high, _step = jo._histogram_limits(values, jo.CTF_RESOLUTION_RANGE)
        assert high <= 4.5 and np.sum(values > high) >= 20
        values = rng.uniform(5, 40, 400)  # most are of low resolution: 10 Å is the end
        assert jo._histogram_limits(values, jo.CTF_RESOLUTION_RANGE)[1] == 10.0

    def test_the_caption_says_how_many_micrographs_are_not_shown(self, relion):
        job = relion / "CtfFind" / "job001"
        rng = np.random.default_rng(0)
        resolution = np.concatenate([rng.normal(3.2, 0.3, 90), np.full(10, 25.0)])
        rows = [(15000, 15000, f"{r:.2f}") for r in resolution]
        _star_file(
            job / "micrographs_ctf.star",
            _loop(
                "micrographs",
                ["rlnDefocusU", "rlnDefocusV", "rlnCtfMaxResolution"],
                rows,
            ),
        )
        panels = jo.build_outcome(job, "relion")["panels"]
        caption = panels[1]["caption"]
        assert "10 beyond" in caption and "not shown" in caption

    def test_a_histogram_of_whole_numbers_is_drawn(self, qapp):
        image = jo._plot_histogram(
            np.random.default_rng(0).normal(20, 5, 300), "x", "t", 20.0
        )
        assert image.shape[2] == 3 and image.std() > 5


class TestTheCaptionOfTheViews:
    def test_all_the_particles_need_no_explanation(self):
        assert jo._views_caption(258_368) == "258,368 particles"
        assert jo._views_caption(1000, 1000) == "1,000 particles"

    def test_the_first_ones_are_said_to_be_the_first_ones(self):
        assert (
            jo._views_caption(300_000, 459_187)
            == "the first 300,000 of 459,187 particles"
        )
        assert (
            jo._views_caption(jo.VIEW_PARTICLES)
            == f"the first {jo.VIEW_PARTICLES:,} particles"
        )

    def test_a_refinement_with_fewer_particles_than_the_limit_says_just_how_many(
        self, relion
    ):
        job = relion / "Refine3D" / "job014"
        job.mkdir(parents=True)
        _make_refine(job)
        outcome = jo.build_outcome(job, "relion")
        assert outcome["panels"][2]["caption"] == "500 particles"


class TestInThePreview:
    def test_a_job_that_made_something_shows_it_in_place_of_its_files(self, relion):
        job = relion / "Refine3D" / "job014"
        job.mkdir(parents=True)
        _make_refine(job)
        (relion / "default_pipeline.star").write_text("data_pipeline_general\n")
        info = bp.build_preview(job)
        assert info["outcome"]["title"] == "Refinement"
        assert info["text"] == ""
        assert ("Resolution", dict(info["outcome"]["facts"])["Resolution"]) in info[
            "facts"
        ]

    def test_a_job_that_made_nothing_shows_no_list_of_files_either(self, relion):
        job = relion / "Refine3D" / "job001"
        job.mkdir(parents=True)
        (job / "run.out").write_text("x")
        (relion / "default_pipeline.star").write_text("data_pipeline_general\n")
        info = bp.build_preview(job)
        assert "outcome" not in info and info["text"] == ""
        assert ("RELION job", "Refine3D/job001, success") in info["facts"] or any(
            label == "RELION job" for label, _ in info["facts"]
        )

    def test_a_plain_folder_has_no_outcome(self, tmp_path):
        assert "outcome" not in bp.build_preview(tmp_path)

    def test_the_pane_shows_each_picture_with_its_title_and_caption(self, qapp, relion):
        job = relion / "Refine3D" / "job014"
        job.mkdir(parents=True)
        _make_refine(job)
        (relion / "default_pipeline.star").write_text("data_pipeline_general\n")
        pane = bp.PreviewPane()
        pane.resize(340, 800)
        pane.show()
        pane.show_info(bp.build_preview(job))
        assert pane._outcome_scroll is not None and not pane._outcome_scroll.isHidden()
        assert pane._text.isHidden()  # no list of files
        assert len(pane._outcome_panels) == 3
        shown = [label.pixmap() for label, _ in pane._outcome_panels]
        assert all(p is not None and not p.isNull() for p in shown)
        assert all(p.width() <= 340 for p in shown)
        texts = [
            w.text()
            for w in pane._outcome_scroll.widget().findChildren(type(pane._note))
            if w.text()
        ]
        assert "Map, central section" in texts and "FSC" in texts and "Views" in texts
        # narrower: the pictures follow
        wide = shown[0].width()
        pane.resize(220, 800)
        assert pane._outcome_panels[0][0].pixmap().width() < wide

    def test_another_folder_hides_it_again(self, qapp, relion, tmp_path):
        job = relion / "Refine3D" / "job014"
        job.mkdir(parents=True)
        _make_refine(job)
        (relion / "default_pipeline.star").write_text("data_pipeline_general\n")
        pane = bp.PreviewPane()
        pane.show()
        pane.show_info(bp.build_preview(job))
        pane.show_info(bp.build_preview(tmp_path))
        assert pane._outcome_scroll.isHidden()
        pane.show_info(bp.build_preview(job))
        pane.clear()
        assert pane._outcome_scroll.isHidden()

    def test_the_module_is_not_loaded_before_a_job_is_looked_at(self, tmp_path):
        import subprocess
        import sys

        code = (
            "import sys; from helicon.lib.gui import browser_parts; "
            "assert 'helicon.lib.gui.job_outcome' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


def _helical_table(job, name="run_it025_data.star", track=True):
    """A Class2D table of segments: mic_a has three filaments (the second with two segments
    missing), mic_b has one; the segments are 30 Å apart, 10 pixels on the micrograph."""
    rows = []

    def filament(mic, tube, y, steps):
        for i in steps:
            row = [mic, tube, 50 + 10 * i, y, f"{i:06d}@Extract/job005/x.mrcs"]
            rows.append(row + ([f"{30.0 * i:.6f}"] if track else []))

    filament("MotionCorr/job002/mic_a.mrc", 1, 40, range(8))
    filament("MotionCorr/job002/mic_a.mrc", 2, 100, [0, 1, 2, 5, 6])
    filament("MotionCorr/job002/mic_a.mrc", 3, 160, range(4))
    filament("MotionCorr/job002/mic_b.mrc", 1, 40, range(12))
    columns = [
        "rlnMicrographName",
        "rlnHelicalTubeID",
        "rlnCoordinateX",
        "rlnCoordinateY",
        "rlnImageName",
    ] + (["rlnHelicalTrackLengthAngst"] if track else [])
    _star_file(
        job / name,
        "data_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n" + _loop("particles", columns, rows),
    )


class TestTheMicrographWithTheMostFilaments:
    def test_the_tables_of_classifications_and_refinements(self, tmp_path):
        assert jo.is_particle_table(tmp_path / "Class2D/job010/run_it025_data.star")
        assert jo.is_particle_table(tmp_path / "Class3D/job011/run_ct5_it025_data.star")
        assert jo.is_particle_table(tmp_path / "Refine3D/job012/run_data.star")
        assert not jo.is_particle_table(tmp_path / "Class2D/job010/run_it025_model.star")
        assert not jo.is_particle_table(tmp_path / "Extract/job005/run_data.star")

    def test_fragmentation_counts_the_filaments_with_gaps(self):
        fragmented, n, spacing, missing = jo.filament_fragmentation(
            [[0, 30, 60, 90], [0, 30, 120, 150], [60], [0, 60, 30]]
        )
        assert (fragmented, n, spacing, missing) == (1, 4, 30.0, 2)
        assert jo.filament_fragmentation([[5.0], [7.0]])[2] is None

    def test_the_micrograph_is_shown_with_its_segments(self, relion):
        job = relion / "Class2D" / "job010"
        _helical_table(job)
        (relion / "default_pipeline.star").write_text("data_pipeline_general\n")
        _mrc(relion / "MotionCorr/job002/mic_a.mrc", np.zeros((200, 300)))
        outcome = jo.busiest_micrograph(job / "run_it025_data.star")
        panel = outcome["panels"][0]
        assert panel["title"] == "Micrograph with the most filaments"
        assert panel["image"].shape == (200, 300, 3)
        # a coloured dot at a segment, the grey micrograph elsewhere
        assert len(set(panel["image"][40, 50])) > 1
        assert len(set(panel["image"][5, 5])) == 1
        first, second = panel["caption"].split("\n")
        assert first == "mic_a.mrc: 3 filaments, 17 segments"
        assert second.startswith("Fragmented: 1 of 3 filaments (33%) have gaps, 2")
        assert ("Filaments", "4 in 2 micrographs") in outcome["facts"]

    def test_without_the_micrograph_the_positions_alone(self, relion):
        job = relion / "Refine3D" / "job012"
        _helical_table(job, "run_data.star", track=False)
        outcome = jo.busiest_micrograph(job / "run_data.star")
        caption = outcome["panels"][0]["caption"]
        assert "micrograph not found" in caption
        assert caption.endswith("Fragmentation unknown: no rlnHelicalTrackLengthAngst")

    def test_a_table_without_filaments_has_none(self, relion):
        job = relion / "Refine3D" / "job014"
        job.mkdir(parents=True)
        _make_refine(job)
        assert jo.busiest_micrograph(job / "run_data.star") is None

    def test_the_preview_of_the_table_shows_it_after_the_second_reading(
        self, qapp, relion
    ):
        job = relion / "Class2D" / "job010"
        _helical_table(job)
        info = bp.build_preview(job / "run_it025_data.star")
        assert info["text"]  # the columns of the table, at once
        pane = bp.PreviewPane()
        pane.show()
        pane.show_info(info)
        assert pane._outcome_scroll is None or pane._outcome_scroll.isHidden()
        pane._on_refined(pane._serial, info["outcome"]["refine"]())
        assert not pane._outcome_scroll.isHidden()
        assert len(pane._outcome_panels) == 1
        assert "4 in 2 micrographs" in pane._facts.text()


class TestCountsOnly:
    def test_a_job_with_numbers_but_no_pictures_shows_the_facts_and_no_picture_area(
        self, qapp, relion
    ):
        job = relion / "Import" / "job001"
        _star_file(
            job / "movies.star",
            _loop("movies", ["rlnMicrographMovieName"], [("a",), ("b",)]),
        )
        (relion / "default_pipeline.star").write_text("data_pipeline_general\n")
        info = bp.build_preview(job)
        assert ("Movies imported", "2") in info["facts"]
        pane = bp.PreviewPane()
        pane.show()
        pane.show_info(info)
        assert pane._outcome_scroll is None or pane._outcome_scroll.isHidden()
