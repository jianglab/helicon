"""The 3D alignment of a cryoSPARC job on particles imported from RELION reaches the star file.

For such a file Helicon takes the particles from the star file that was imported, and puts what
cryoSPARC changed on top of it. The 2D class, alignment and CTF were taken over; the 3D class, pose,
shift, half set and score of a refinement or a 3D classification were not, so the star file kept the
angles of the original. The values must be those of the conversion of the same file that does not go
through the imported star file.
"""

import numpy as np
import pytest

from helicon.lib import io
from test_helical_pitch_cs_loader import make_import_project


def _with_3d_alignment(cs_file, seed=0):
    """The .cs file of the fixture, with the fields of a 3D refinement added; returns the rows."""
    cs = np.load(cs_file, allow_pickle=True)
    n = len(cs)
    rng = np.random.default_rng(seed)
    extra = {
        "alignments3D/class": (np.arange(n) % 2).astype(np.uint32),
        "alignments3D/split": (np.arange(n) % 2).astype(np.uint32),
        "alignments3D/cross_cor": rng.uniform(0.1, 1.0, n).astype(np.float32),
        "alignments3D/pose": rng.normal(scale=0.8, size=(n, 3)).astype(np.float32),
        "alignments3D/shift": rng.normal(scale=2.0, size=(n, 2)).astype(np.float32),
    }
    dtype = [(name, cs.dtype[name]) for name in cs.dtype.names]
    dtype += [(name, v.dtype, v.shape[1:]) for name, v in extra.items()]
    out = np.zeros(n, dtype=dtype)
    for name in cs.dtype.names:
        out[name] = cs[name]
    for name, v in extra.items():
        out[name] = v
    with open(cs_file, "wb") as f:
        np.save(f, out)
    return out


@pytest.fixture
def converted(tmp_path):
    """The same refinement, converted through the imported star file and by itself."""
    cs_file = make_import_project(tmp_path, {"J1": [1, 1, 2, 2], "J2": [3, 3]})
    _with_3d_alignment(cs_file)
    through_import = io.cs2dataframe(cs_file)
    assert through_import.attrs["convention"] == "relion"  # the imported star was used

    alone = tmp_path / "elsewhere"
    alone.mkdir()
    copy = alone / "J9_particles.cs"
    copy.write_bytes(open(cs_file, "rb").read())
    plain = io.dataframe_cryosparc_to_relion(
        io.cs2dataframe(str(copy), ignore_bad_particle_path=2)
    )
    key = "rlnLogLikeliContribution"  # one value for each particle: lines the two up
    return (
        through_import.sort_values(key).reset_index(drop=True),
        plain.sort_values(key).reset_index(drop=True),
    )


class TestThe3DAlignmentIsTaken:
    @pytest.mark.parametrize(
        "column",
        [
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnOriginXAngst",
            "rlnOriginYAngst",
        ],
    )
    def test_the_pose_and_the_shift_are_those_of_the_plain_conversion(
        self, converted, column
    ):
        imported, plain = converted
        assert np.allclose(imported[column], plain[column], atol=1e-4)

    def test_the_class_the_half_set_and_the_score(self, converted):
        imported, plain = converted
        for column in ("rlnClassNumber", "rlnRandomSubset", "rlnLogLikeliContribution"):
            assert np.allclose(imported[column], plain[column]), column
        assert set(imported["rlnRandomSubset"]) == {1, 2}

    def test_what_only_the_star_file_has_is_kept(self, converted):
        imported, _ = converted
        # the helical columns of RELION, which the .cs file of a job on imported particles lacks
        assert "rlnHelicalTubeID" in imported and "rlnAnglePsiPrior" in imported
        assert np.allclose(imported["rlnAnglePsiPrior"], -30.0)

    def test_the_angles_are_not_those_of_the_original_star(self, converted):
        imported, _ = converted
        # the original star file has no rlnAngleRot at all: they come from cryoSPARC's pose
        assert imported["rlnAngleRot"].abs().max() > 1.0
