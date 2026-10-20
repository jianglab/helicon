"""relion_reconstruct from the AbInitio3D tab's segments."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helicon.webApps.lib import helical_pitch_relion as relion


class TestImageNames:
    def test_relative_stacks_start_from_the_project_directory(self):
        names = ["000001@Extract/job1/a.mrcs", "7@/abs/b.mrcs", "c.mrc"]
        out = relion._absolute_image_names(names, "/proj")
        assert out == [
            "000001@/proj/Extract/job1/a.mrcs",
            "7@/abs/b.mrcs",
            "/proj/c.mrc",
        ]


class TestReconstruct:
    def test_a_missing_stack_is_reported_before_relion_runs(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(relion, "find_relion_reconstruct", lambda: "/bin/true")
        seg = pd.DataFrame(
            dict(rlnImageName=["1@Extract/none.mrcs"], rlnAngleRot=[0.0])
        )
        with pytest.raises(FileNotFoundError, match="project directory"):
            relion.reconstruct(seg, str(tmp_path))

    @pytest.mark.skipif(
        relion.find_relion_reconstruct() is None,
        reason="relion_reconstruct not on PATH",
    )
    def test_side_views_reconstruct(self, tmp_path):
        import mrcfile

        box = 32
        stack = tmp_path / "Extract" / "segs.mrcs"
        stack.parent.mkdir()
        rng = np.random.default_rng(0)
        with mrcfile.new(str(stack)) as m:
            m.set_data(rng.standard_normal((6, box, box)).astype(np.float32))
            m.voxel_size = 2.0
        seg = pd.DataFrame(
            dict(
                rlnImageName=[f"{i + 1:06d}@Extract/segs.mrcs" for i in range(6)],
                rlnAngleRot=np.arange(6) * 30.0,
                rlnAngleTilt=90.0,
                rlnAnglePsi=0.0,
                rlnOriginXAngst=0.0,
                rlnOriginYAngst=0.0,
                rlnOpticsGroup=1,
            )
        )
        seg.attrs["optics"] = pd.DataFrame(
            dict(
                rlnOpticsGroupName=["o1"],
                rlnOpticsGroup=[1],
                rlnImagePixelSize=[2.0],
                rlnImageSize=[box],
                rlnImageDimensionality=[2],
                rlnVoltage=[300.0],
                rlnSphericalAberration=[2.7],
                rlnAmplitudeContrast=[0.1],
            )
        )
        out = relion.reconstruct(seg, str(tmp_path), work_dir=str(tmp_path / "work"))
        assert out["volume"].shape == (box, box, box)
        assert out["apix"] == pytest.approx(2.0) and out["n_segments"] == 6


def _segments(tmp_path, n=6, box=32):
    """``n`` segments in a real stack, with their optics."""
    import mrcfile

    stack = tmp_path / "Extract" / "segs.mrcs"
    stack.parent.mkdir(exist_ok=True)
    with mrcfile.new(str(stack), overwrite=True) as m:
        m.set_data(np.zeros((n, box, box), np.float32))
    seg = pd.DataFrame(
        dict(
            rlnImageName=[f"{i + 1:06d}@Extract/segs.mrcs" for i in range(n)],
            rlnAngleRot=0.0,
            rlnAngleTilt=90.0,
            rlnAnglePsi=0.0,
        )
    )
    seg.attrs["optics"] = pd.DataFrame(dict(rlnImageSize=[box]))
    return seg


class _FakeRun:
    """Stands in for subprocess.run: records commands, writes the map."""

    def __init__(self, fail_mpi=False):
        self.commands, self.cwds, self.fail_mpi = [], [], fail_mpi

    def __call__(self, command, **kw):
        import mrcfile
        from types import SimpleNamespace

        self.commands.append(command)
        self.cwds.append(kw.get("cwd"))
        if self.fail_mpi and "-n" in command:
            return SimpleNamespace(returncode=1, stdout="", stderr="mpi broke")
        out = command[command.index("--o") + 1]
        with mrcfile.new(out, overwrite=True) as m:
            m.set_data(np.zeros((4, 4, 4), np.float32))
            m.voxel_size = 1.0
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class TestMpi:
    def test_mpirun_comes_from_the_mpi_library_the_program_uses(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace

        (tmp_path / "lib").mkdir()
        (tmp_path / "bin").mkdir()
        (tmp_path / "lib" / "libmpi.so.40").touch()
        (tmp_path / "bin" / "mpirun").touch()
        ldd = f"\tlibmpi.so.40 => {tmp_path}/lib/libmpi.so.40 (0x0000)\n"
        monkeypatch.delenv("HELICON_MPIRUN", raising=False)
        monkeypatch.setattr(
            relion.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=ldd)
        )
        assert relion._mpirun_for("/x/relion_reconstruct_mpi") == str(
            tmp_path / "bin" / "mpirun"
        )

    def test_the_environment_can_name_mpirun(self, monkeypatch, tmp_path):
        # any executable file: /bin/true is not there on every system (macOS)
        mpirun = tmp_path / "my_mpirun"
        mpirun.write_text("#!/bin/sh\n")
        mpirun.chmod(0o755)
        monkeypatch.setenv("HELICON_MPIRUN", str(mpirun))
        assert relion._mpirun_for("/x/relion_reconstruct_mpi") == str(mpirun)

    def test_a_named_mpirun_that_does_not_exist_is_none(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HELICON_MPIRUN", str(tmp_path / "missing"))
        assert relion._mpirun_for("/x/relion_reconstruct_mpi") is None

    def test_processes_are_limited_by_cpus_memory_and_segments(self, monkeypatch):
        import helicon

        free_gb = 40.0
        monkeypatch.setattr(
            helicon,
            "available_cpu",
            lambda mem_gb_per_cpu=None: int(free_gb / mem_gb_per_cpu),
        )
        # 256-pixel box: ~4.7 GB a process, so 40 GB fit 8 of them
        assert relion._mpi_layout(32, 256, 100000) == (8, 4)
        assert relion._mpi_layout(4, 256, 100000) == (4, 1)
        assert relion._mpi_layout(32, 64, 1000) == (5, 6)  # 200 segments each
        assert relion._mpi_layout(1, 64, 100000) == (1, 1)

    def test_the_mpi_program_is_used_with_the_c_symmetry(self, tmp_path, monkeypatch):
        fake = _FakeRun()
        monkeypatch.setattr(relion.subprocess, "run", fake)
        monkeypatch.setattr(relion, "find_relion_reconstruct", lambda: "/r/rr")
        monkeypatch.setattr(
            relion, "find_relion_reconstruct_mpi", lambda: ("/r/rr_mpi", "/m/mpirun")
        )
        monkeypatch.setattr(relion, "_mpi_layout", lambda cpu, box, n: (4, 2))
        out = relion.reconstruct(
            _segments(tmp_path), str(tmp_path), cpu=8, csym=3, work_dir=str(tmp_path)
        )
        (command,) = fake.commands
        assert command[:7] == [
            "/m/mpirun",
            "--oversubscribe",
            "--bind-to",
            "none",
            "-n",
            "4",
            "/r/rr_mpi",
        ]
        assert command[command.index("--sym") + 1] == "c3"
        assert command[command.index("--j") + 1] == "2"
        assert out["mpi"] == 4 and out["csym"] == 3

    def test_a_failed_mpi_run_falls_back_to_threads(self, tmp_path, monkeypatch):
        fake = _FakeRun(fail_mpi=True)
        monkeypatch.setattr(relion.subprocess, "run", fake)
        monkeypatch.setattr(relion, "find_relion_reconstruct", lambda: "/r/rr")
        monkeypatch.setattr(
            relion, "find_relion_reconstruct_mpi", lambda: ("/r/rr_mpi", "/m/mpirun")
        )
        monkeypatch.setattr(relion, "_mpi_layout", lambda cpu, box, n: (4, 2))
        out = relion.reconstruct(
            _segments(tmp_path), str(tmp_path), cpu=8, work_dir=str(tmp_path)
        )
        assert len(fake.commands) == 2
        assert fake.commands[1][0] == "/r/rr"
        assert fake.commands[1][fake.commands[1].index("--j") + 1] == "8"
        assert out["mpi"] == 0

    def test_without_mpi_the_threaded_program_runs(self, tmp_path, monkeypatch):
        fake = _FakeRun()
        monkeypatch.setattr(relion.subprocess, "run", fake)
        monkeypatch.setattr(relion, "find_relion_reconstruct", lambda: "/r/rr")
        monkeypatch.setattr(relion, "find_relion_reconstruct_mpi", lambda: None)
        relion.reconstruct(_segments(tmp_path), str(tmp_path), cpu=8, csym=2)
        (command,) = fake.commands
        assert command[0] == "/r/rr" and command[command.index("--sym") + 1] == "c2"


class TestTemporaryFolder:
    def _fake(self, monkeypatch, tmp_path, fail=False):
        import tempfile

        made = []
        real = tempfile.mkdtemp

        def mkdtemp(*a, **k):
            k["dir"] = str(tmp_path)
            made.append(Path(real(*a, **k)))
            return str(made[-1])

        monkeypatch.setattr(relion.tempfile, "mkdtemp", mkdtemp)
        monkeypatch.setattr(relion, "find_relion_reconstruct", lambda: "/r/rr")
        monkeypatch.setattr(relion, "find_relion_reconstruct_mpi", lambda: None)
        if fail:
            from types import SimpleNamespace

            monkeypatch.setattr(
                relion.subprocess,
                "run",
                lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="x"),
            )
        else:
            monkeypatch.setattr(relion.subprocess, "run", _FakeRun())
        return made

    def test_the_temporary_folder_is_removed_after_the_run(self, tmp_path, monkeypatch):
        made = self._fake(monkeypatch, tmp_path)
        out = relion.reconstruct(_segments(tmp_path), str(tmp_path))
        assert out["volume"].shape == (4, 4, 4) and out["path"] is None
        assert len(made) == 1 and not made[0].exists()

    def test_and_after_a_failed_run(self, tmp_path, monkeypatch):
        made = self._fake(monkeypatch, tmp_path, fail=True)
        with pytest.raises(RuntimeError):
            relion.reconstruct(_segments(tmp_path), str(tmp_path))
        assert len(made) == 1 and not made[0].exists()

    def test_a_given_work_dir_is_kept(self, tmp_path, monkeypatch):
        self._fake(monkeypatch, tmp_path)
        work = tmp_path / "work"
        out = relion.reconstruct(_segments(tmp_path), str(tmp_path), work_dir=work)
        assert Path(out["path"]).exists() and out["path"].startswith(str(work))


class TestFusedAvailability:
    def _run(self, monkeypatch, text):
        from types import SimpleNamespace

        calls = []

        def fake(command, **kw):
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout=text, stderr="")

        monkeypatch.setattr(relion.subprocess, "run", fake)
        relion._FUSED_CHECKED.clear()
        return calls

    def test_a_program_with_the_option_has_it(self, monkeypatch):
        self._run(monkeypatch, "  --fused_extract (false) : Particles are ...")
        assert relion.fused_extract_available("/r/rr") is True

    def test_a_program_without_the_option_does_not(self, monkeypatch):
        self._run(monkeypatch, "  --i () : Input STAR file")
        assert relion.fused_extract_available("/r/rr") is False

    def test_it_is_asked_once_for_each_program(self, monkeypatch):
        calls = self._run(monkeypatch, "--fused_extract")
        relion.fused_extract_available("/r/rr")
        relion.fused_extract_available("/r/rr")
        assert len(calls) == 1

    def test_without_the_program_there_is_nothing(self, monkeypatch):
        monkeypatch.setattr(relion, "find_relion_reconstruct", lambda: None)
        assert relion.fused_extract_available() is False


def _job_star(path, **options):
    import starfile

    table = pd.DataFrame(
        dict(
            rlnJobOptionVariable=list(options),
            rlnJobOptionValue=[str(v) for v in options.values()],
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write(dict(joboptions_values=table), str(path), overwrite=True)


class TestExtractionSettings:
    def test_they_are_read_from_the_extract_job(self, tmp_path):
        _job_star(
            tmp_path / "Extract/job009/job.star",
            do_invert="Yes",
            do_norm="No",
            bg_diameter=-1,
            do_extract_helical_tubes="Yes",
            helical_tube_outer_diameter=200,
        )
        s = relion.extraction_settings(tmp_path, "000001@Extract/job009/a/b.mrcs")
        assert s == dict(
            invert=True, norm=False, bg_diameter=-1.0, helical_diameter=200.0
        )

    def test_a_job_of_other_particles_gives_relions_defaults(self, tmp_path):
        s = relion.extraction_settings(tmp_path, "1@Polish/job025/x_shiny.mrcs")
        assert s["invert"] and s["norm"] and s["helical_diameter"] is None

    def test_a_missing_job_gives_the_defaults_too(self, tmp_path):
        s = relion.extraction_settings(tmp_path, "1@Extract/job099/x.mrcs")
        assert s["invert"] and s["norm"]

    def test_no_tube_without_helical_extraction(self, tmp_path):
        _job_star(
            tmp_path / "Extract/job1/job.star",
            do_extract_helical_tubes="No",
            helical_tube_outer_diameter=200,
        )
        s = relion.extraction_settings(tmp_path, "1@Extract/job1/x.mrcs")
        assert s["helical_diameter"] is None


class TestFusedArgs:
    def _args(self, **kw):
        base = dict(invert=True, norm=True, bg_diameter=-1.0, helical_diameter=200.0)
        base.update(kw)
        return relion.fused_args(2.0, 301, 100, None, base)

    def _value(self, args, name):
        return args[args.index(name) + 1]

    def test_the_box_is_rectangular_and_even(self):
        a = self._args()
        assert self._value(a, "--box_x") == "302" and self._value(a, "--box_y") == "100"
        assert self._value(a, "--box_z") == "302"  # the map is as long as the box
        assert self._value(a, "--angpix") == "2"
        assert "--fused_extract" in a and "--fused_rotate_to_horizontal" in a

    def test_the_depth_can_differ_from_the_length(self):
        a = relion.fused_args(2.0, 300, 100, 200, {})
        assert self._value(a, "--box_z") == "200"

    def test_the_segments_are_cut_as_the_extract_job_did(self):
        a = self._args()
        assert "--fused_invert_contrast" in a and "--fused_norm" in a
        assert self._value(a, "--fused_helical_diameter") == "200"
        # the background is outside the tube: its radius in output pixels
        assert float(self._value(a, "--fused_bg_radius")) == pytest.approx(50.0)

    def test_without_normalisation_or_inversion_neither_is_asked_for(self):
        a = self._args(invert=False, norm=False)
        assert "--fused_norm" not in a and "--fused_invert_contrast" not in a
        assert "--fused_bg_radius" not in a

    def test_the_background_diameter_of_the_job_wins_over_the_tube(self):
        a = self._args(bg_diameter=300.0)
        assert float(self._value(a, "--fused_bg_radius")) == pytest.approx(75.0)


class TestTurnToHorizontal:
    def _segments(self, prior=-30.0, psi=-25.0):
        return pd.DataFrame(
            dict(
                rlnAnglePsiPrior=[prior],
                rlnAnglePsi=[psi],
                rlnOriginXAngst=[10.0],
                rlnOriginYAngst=[0.0],
                rlnDefocusAngle=[10.0],
            )
        )

    def test_psi_is_taken_relative_to_the_prior(self):
        out = relion.turn_to_horizontal(self._segments())
        assert out["rlnAnglePsi"].iloc[0] == pytest.approx(5.0)

    def test_psi_stays_in_range(self):
        out = relion.turn_to_horizontal(self._segments(prior=170.0, psi=-175.0))
        assert -180 <= out["rlnAnglePsi"].iloc[0] < 180
        assert out["rlnAnglePsi"].iloc[0] == pytest.approx(15.0)

    def test_the_origin_is_turned_by_the_prior(self):
        out = relion.turn_to_horizontal(self._segments(prior=90.0, psi=90.0))
        assert out["rlnOriginXAngst"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert out["rlnOriginYAngst"].iloc[0] == pytest.approx(10.0)

    def test_the_length_of_the_shift_is_kept(self):
        out = relion.turn_to_horizontal(self._segments())
        x, y = out["rlnOriginXAngst"].iloc[0], out["rlnOriginYAngst"].iloc[0]
        assert np.hypot(x, y) == pytest.approx(10.0)

    def test_the_astigmatism_turns_with_the_image(self):
        out = relion.turn_to_horizontal(self._segments(prior=-30.0))
        assert out["rlnDefocusAngle"].iloc[0] == pytest.approx(-20.0)

    def test_the_input_is_not_changed(self):
        seg = self._segments()
        relion.turn_to_horizontal(seg)
        assert seg["rlnAnglePsi"].iloc[0] == -25.0

    def test_segments_from_a_turned_extraction_are_left_as_they_are(self):
        seg = self._segments()
        seg["rlnParticleExtractionAngle"] = 30.0
        out = relion.turn_to_horizontal(seg)
        assert out["rlnAnglePsi"].iloc[0] == -25.0


class TestFusedReconstruct:
    def _setup(self, tmp_path, monkeypatch, available=True):
        fake = _FakeRun()
        monkeypatch.setattr(relion.subprocess, "run", fake)
        monkeypatch.setattr(relion, "find_relion_reconstruct", lambda: "/r/rr")
        monkeypatch.setattr(relion, "find_relion_reconstruct_mpi", lambda: None)
        monkeypatch.setattr(
            relion, "fused_extract_available", lambda exe=None: available
        )
        mic = tmp_path / "MotionCorr" / "m1.mrc"
        mic.parent.mkdir()
        mic.touch()
        seg = pd.DataFrame(
            dict(
                rlnImageName=["1@Extract/job1/a.mrcs"],
                rlnMicrographName=["MotionCorr/m1.mrc"],
                rlnCoordinateX=[100.0],
                rlnCoordinateY=[200.0],
                rlnAnglePsiPrior=[-30.0],
                rlnAnglePsi=[-25.0],
                rlnAngleRot=[0.0],
            )
        )
        return fake, seg

    def test_the_options_the_turned_star_and_the_micrographs_are_passed(
        self, tmp_path, monkeypatch
    ):
        import starfile

        fake, seg = self._setup(tmp_path, monkeypatch)
        out = relion.reconstruct(
            seg,
            str(tmp_path),
            work_dir=tmp_path / "w",
            fused=dict(angpix=2.5, length=200, diameter=100),
        )
        (command,) = fake.commands
        assert (
            "--fused_extract" in command and "--fused_rotate_to_horizontal" in command
        )
        assert command[command.index("--box_x") + 1] == "200"
        assert command[command.index("--box_y") + 1] == "100"
        assert out["fused"] is True
        written = starfile.read(str(tmp_path / "w" / "segments.star"))
        assert written["rlnMicrographName"].iloc[0] == str(
            tmp_path / "MotionCorr/m1.mrc"
        )
        assert written["rlnAnglePsi"].iloc[0] == pytest.approx(5.0)

    def test_the_stored_images_need_not_exist_for_a_fused_run(
        self, tmp_path, monkeypatch
    ):
        fake, seg = self._setup(tmp_path, monkeypatch)
        relion.reconstruct(
            seg, str(tmp_path), fused=dict(angpix=2, length=64, diameter=64)
        )
        assert len(fake.commands) == 1

    def test_a_missing_micrograph_is_reported_before_relion_runs(
        self, tmp_path, monkeypatch
    ):
        fake, seg = self._setup(tmp_path, monkeypatch)
        seg["rlnMicrographName"] = "MotionCorr/none.mrc"
        with pytest.raises(FileNotFoundError, match="micrograph paths"):
            relion.reconstruct(
                seg, str(tmp_path), fused=dict(angpix=2, length=64, diameter=64)
            )
        assert not fake.commands

    def test_the_coordinates_are_needed(self, tmp_path, monkeypatch):
        fake, seg = self._setup(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="rlnCoordinateX"):
            relion.reconstruct(
                seg.drop(columns="rlnCoordinateX"),
                str(tmp_path),
                fused=dict(angpix=2, length=64, diameter=64),
            )

    def test_a_program_without_the_option_refuses(self, tmp_path, monkeypatch):
        fake, seg = self._setup(tmp_path, monkeypatch, available=False)
        with pytest.raises(RuntimeError, match="no --fused_extract"):
            relion.reconstruct(
                seg, str(tmp_path), fused=dict(angpix=2, length=64, diameter=64)
            )

    def test_without_fused_nothing_changes(self, tmp_path, monkeypatch):
        fake, seg = self._setup(tmp_path, monkeypatch)
        stack = tmp_path / "Extract" / "job1" / "a.mrcs"
        stack.parent.mkdir(parents=True)
        stack.touch()
        out = relion.reconstruct(seg, str(tmp_path))
        (command,) = fake.commands
        assert "--fused_extract" not in command and out["fused"] is False


class TestVirtualParticles:
    """A modified RELION cuts the images from the micrographs, by their own paths."""

    def _run(self, tmp_path, monkeypatch, work_dir):
        import starfile

        fake = _FakeRun()
        monkeypatch.setattr(relion.subprocess, "run", fake)
        monkeypatch.setattr(relion, "find_relion_reconstruct", lambda: "/r/rr")
        monkeypatch.setattr(relion, "find_relion_reconstruct_mpi", lambda: None)
        seg = _segments(tmp_path)
        seg["rlnMicrographName"] = "MotionCorr/m1.mrc"
        relion.reconstruct(seg, str(tmp_path), work_dir=work_dir)
        written = starfile.read(str(Path(work_dir) / "segments.star"))["particles"]
        return fake, written

    def test_the_micrograph_paths_start_from_the_project_directory(
        self, tmp_path, monkeypatch
    ):
        fake, written = self._run(tmp_path, monkeypatch, tmp_path / "w")
        assert written["rlnMicrographName"].iloc[0] == str(
            tmp_path / "MotionCorr/m1.mrc"
        )

    def test_relion_runs_in_the_project_directory(self, tmp_path, monkeypatch):
        fake, _ = self._run(tmp_path, monkeypatch, tmp_path / "w")
        assert fake.cwds == [str(tmp_path)]

    def test_a_relative_work_dir_is_still_found_from_there(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake, _ = self._run(tmp_path, monkeypatch, "w")
        (command,) = fake.commands
        assert Path(command[command.index("--i") + 1]).is_absolute()
        assert Path(command[command.index("--o") + 1]).is_absolute()
