"""relion_reconstruct from the AbInitio3D tab's segments."""

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
