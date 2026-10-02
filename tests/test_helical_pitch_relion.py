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
        self.commands, self.fail_mpi = [], fail_mpi

    def __call__(self, command, **kw):
        import mrcfile
        from types import SimpleNamespace

        self.commands.append(command)
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

    def test_the_environment_can_name_mpirun(self, monkeypatch):
        monkeypatch.setenv("HELICON_MPIRUN", "/bin/true")
        assert relion._mpirun_for("/x/relion_reconstruct_mpi") == "/bin/true"

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
