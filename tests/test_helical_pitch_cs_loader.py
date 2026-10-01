"""Tests for reading cryoSPARC particle files into RELION-style parameters.

The files are small structured arrays written to ``tmp_path``: the loader's
job is to read a file, so a file it must be, but each holds only the fields
under test.
"""

import json

import numpy as np
import pandas as pd
import pytest
import starfile

from helicon.webApps.lib import helical_pitch_compute as hp


def _cs(fields, n):
    dtype = [(name, value.dtype, value.shape[1:]) for name, value in fields.items()]
    arr = np.zeros(n, dtype=dtype)
    for name, value in fields.items():
        arr[name] = value
    return arr


def _alignment_fields(n):
    return {
        "alignments2D/class": np.arange(n, dtype=np.uint32) % 3,
        "alignments2D/pose": np.linspace(-1.0, 1.0, n).astype(np.float32),
        "alignments2D/class_posterior": np.full(n, 0.5, np.float32),
    }


def _save(path, arr):
    with open(path, "wb") as f:
        np.save(f, arr)
    return str(path)


class TestNativeFilaments:
    def test_coordinates_prior_and_track_come_from_cryosparc_fields(self, tmp_path):
        n = 4
        fields = {
            "blob/path": np.array([b"J3/extract/a.mrc"] * n),
            "blob/idx": np.arange(n, dtype=np.uint32),
            "location/micrograph_path": np.array([b"S1/mic1.mrc"] * n),
            "location/micrograph_shape": np.tile(
                np.array([4000, 6000], np.uint32), (n, 1)
            ),
            "location/center_x_frac": np.array([0.1, 0.2, 0.3, 0.4], np.float32),
            "location/center_y_frac": np.array([0.5, 0.5, 0.5, 0.5], np.float32),
            "location/micrograph_psize_A": np.full(n, 0.83, np.float32),
            "filament/filament_uid": np.array([5, 5, 7, 7], np.uint64),
            "filament/position_A": np.array([0.0, 30.0, 0.0, 30.0], np.float32),
            "filament/filament_pose": np.full(n, np.deg2rad(20.0), np.float32),
            **_alignment_fields(n),
        }
        df = hp.get_class2d_params_from_file(_save(tmp_path / "p.cs", _cs(fields, n)))
        assert np.allclose(df["rlnCoordinateX"], [600, 1200, 1800, 2400], atol=0.1)
        assert np.allclose(df["rlnCoordinateY"], 2000, atol=0.1)
        assert np.allclose(df["rlnAnglePsiPrior"], -20.0)
        assert list(df["rlnHelicalTrackLengthAngst"]) == [0.0, 30.0, 0.0, 30.0]
        assert list(df["rlnHelicalTubeID"]) == [5, 5, 7, 7]
        assert np.allclose(df["rlnMicrographPixelSize"], 0.83)
        assert list(df["rlnClassNumber"]) == [1, 2, 3, 1]


class TestRelionImport:
    """The library's RELION-import path, seen through the tab's loader."""

    def test_geometry_comes_from_the_import_star(self, tmp_path):
        cs_file = make_import_project(tmp_path, {"J1": [1, 1, 2, 2]})
        df = hp.get_class2d_params_from_file(cs_file)
        assert sorted(df["rlnHelicalTrackLengthAngst"]) == [0.0, 0.0, 14.25, 14.25]
        assert np.allclose(df["rlnAnglePsiPrior"], -30.0)
        assert "optics" in df.attrs

    def test_missing_star_is_a_clear_error(self, tmp_path):
        cs_file = make_import_project(tmp_path, {"J1": [1, 1]})
        (tmp_path / "CS-proj" / "J1" / "particles.star").unlink()
        with pytest.raises(ValueError, match="import job"):
            hp.get_class2d_params_from_file(cs_file)


def make_import_project(tmp_path, jobs, chunked=True):
    """A cryoSPARC project whose import jobs kept their source star.

    ``jobs`` maps an import job to the tube ID of each of its particles, all
    on one micrograph. Returns the path of a Class2D-like .cs file selecting
    every imported particle, in reverse order.
    """
    project = tmp_path / "CS-proj"
    (project / "J9").mkdir(parents=True, exist_ok=True)
    rows = []
    uid = 1000
    for job, tubes in jobs.items():
        (project / job).mkdir()
        n = len(tubes)
        track = np.zeros(n)
        for t in set(tubes):
            k = np.flatnonzero(np.asarray(tubes) == t)
            track[k] = np.arange(len(k)) * 14.25
        particles = pd.DataFrame(
            dict(
                rlnCoordinateX=np.arange(n) * 10.0 + 100,
                rlnCoordinateY=np.full(n, 50.0),
                rlnHelicalTubeID=tubes,
                rlnAnglePsiPrior=np.full(n, -30.0),
                rlnHelicalTrackLengthAngst=track,
                rlnImageName=[f"{i + 1:08d}@Extract/{job}.mrcs" for i in range(n)],
                rlnMicrographName=["mic1.mrc"] * n,
                rlnOpticsGroup=[1] * n,
            )
        )
        optics = pd.DataFrame(
            dict(
                rlnOpticsGroupName=["opticsGroup1"],
                rlnOpticsGroup=[1],
                rlnMicrographOriginalPixelSize=[1.15],
                rlnImagePixelSize=[1.15],
            )
        )
        starfile.write(
            dict(optics=optics, particles=particles), project / job / "particles.star"
        )
        uids = np.arange(uid, uid + n, dtype=np.uint64)
        uid += n
        name = "imported_particles_0000.cs" if chunked else "imported_particles.cs"
        _save(project / job / name, _cs({"uid": uids}, n))
        rows += [
            (u, f"{job}/imported/0123456789012_{job}.mrcs", i)
            for i, u in enumerate(uids)
        ]
    rows = rows[::-1]
    n = len(rows)
    fields = {
        "uid": np.array([r[0] for r in rows], np.uint64),
        "blob/path": np.array([r[1].encode() for r in rows]),
        "blob/idx": np.array([r[2] for r in rows], np.uint32),
        "blob/psize_A": np.full(n, 1.15, np.float32),
        "alignments2D/shift": np.tile(np.array([2.0, -1.0], np.float32), (n, 1)),
        **_alignment_fields(n),
    }
    return _save(project / "J9" / "J9_particles.cs", _cs(fields, n))
