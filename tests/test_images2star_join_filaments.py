import argparse
import sys

import numpy as np
import pandas as pd
import pytest
import starfile

import helicon
from helicon.commands import images2star


def _filament(mgraph, tube, start, end, step=50.0, stack="a.mrcs", first=1):
    """Particles picked from start to end (RELION start-end convention)."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start)
    track = np.arange(0, length + 0.1, step)
    xy = start + track[:, None] * (end - start) / length
    return pd.DataFrame(
        {
            "rlnImageName": [f"{first + i:06d}@{stack}" for i in range(len(track))],
            "rlnMicrographName": mgraph,
            "rlnCoordinateX": xy[:, 0],
            "rlnCoordinateY": xy[:, 1],
            "rlnHelicalTubeID": tube,
            "rlnHelicalTrackLengthAngst": track,
            "rlnOpticsGroup": 1,
        }
    )


def _write_star(path, particles):
    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [1],
            "rlnOpticsGroupName": ["opticsGroup1"],
            "rlnImagePixelSize": [1.0],
            "rlnImageSize": [64],
            "rlnVoltage": [300.0],
            "rlnSphericalAberration": [2.7],
            "rlnAmplitudeContrast": [0.1],
        }
    )
    starfile.write({"optics": optics, "particles": particles}, path, overwrite=True)


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["helicon", "images2star", *argv])
    monkeypatch.setattr(images2star.helicon, "log_command_line", lambda: None)
    parser = images2star.add_args(argparse.ArgumentParser())
    args = images2star.check_args(parser.parse_args(argv), parser)
    images2star.main(args)


class TestFilamentPiecesOverlap:
    def test_collinear_overlapping(self):
        assert helicon.filament_pieces_overlap(
            [0, 0], [400, 0], [300, 0], [700, 0]
        )

    def test_contained(self):
        assert helicon.filament_pieces_overlap(
            [0, 0], [400, 0], [100, 0], [200, 0]
        )

    def test_collinear_with_gap(self):
        assert not helicon.filament_pieces_overlap(
            [0, 0], [400, 0], [500, 0], [700, 0]
        )

    def test_parallel_offset(self):
        assert not helicon.filament_pieces_overlap(
            [0, 0], [400, 0], [0, 50], [400, 50]
        )

    def test_crossing(self):
        assert not helicon.filament_pieces_overlap(
            [0, 0], [400, 0], [200, 0], [200, 300]
        )

    def test_single_particle_on_filament(self):
        assert helicon.filament_pieces_overlap(
            [0, 0], [400, 0], [150, 0], [150, 0]
        )


class TestJoinCollinearFilaments:
    def test_transitive_join_and_renumber(self):
        data = pd.concat(
            [
                _filament("m1", 1, (0, 0), (200, 0)).assign(src=0),
                _filament("m1", 1, (600, 0), (1000, 0)).assign(src=1),
                _filament("m1", 2, (150, 0), (650, 0)).assign(src=1),
                _filament("m1", 7, (0, 500), (400, 500)).assign(src=0),
            ],
            ignore_index=True,
        )
        out = helicon.join_collinear_filaments(data, piece_attrs=["src", "rlnHelicalTubeID"])
        assert sorted(out["rlnHelicalTubeID"].unique()) == [1, 2]
        joined = out[out["rlnCoordinateY"] == 0]
        assert joined["rlnHelicalTubeID"].nunique() == 1
        assert out[out["rlnCoordinateY"] == 500]["rlnHelicalTubeID"].nunique() == 1
        # overlapping particles are removed, leaving one particle every 50 pixels
        np.testing.assert_allclose(np.sort(joined["rlnCoordinateX"]), np.arange(0, 1001, 50))
        np.testing.assert_allclose(
            joined["rlnHelicalTrackLengthAngst"], joined["rlnCoordinateX"], atol=1e-3
        )

    def test_remove_overlapping_particles_with_phase_offset(self):
        data = pd.concat(
            [
                _filament("m1", 1, (0, 0), (500, 0)).assign(src=0),
                _filament("m1", 1, (430, 0), (830, 0)).assign(src=1),
            ],
            ignore_index=True,
        )
        out = helicon.join_collinear_filaments(data, piece_attrs=["src", "rlnHelicalTubeID"])
        assert out["rlnHelicalTubeID"].nunique() == 1
        # 430 and 480 are 20 pixels (< 50/2) from particles of the larger piece;
        # 530 is 30 pixels from 500 and kept
        np.testing.assert_allclose(
            np.sort(out["rlnCoordinateX"]),
            np.concatenate([np.arange(0, 501, 50), np.arange(530, 831, 50)]),
        )

    def test_inter_box_distance_zero_keeps_all(self):
        data = pd.concat(
            [
                _filament("m1", 1, (0, 0), (500, 0)).assign(src=0),
                _filament("m1", 1, (425, 0), (825, 0)).assign(src=1),
            ],
            ignore_index=True,
        )
        out = helicon.join_collinear_filaments(
            data, piece_attrs=["src", "rlnHelicalTubeID"], inter_box_distance=0
        )
        assert len(out) == len(data)

    def test_estimate_inter_box_distance(self):
        data = pd.concat(
            [_filament("m1", 1, (0, 0), (300, 400), step=20), _filament("m1", 2, (0, 500), (0, 900), step=20)],
            ignore_index=True,
        )
        d = helicon.estimate_inter_box_distance(data, ["rlnMicrographName", "rlnHelicalTubeID"])
        assert d == pytest.approx(20)


class TestImages2starJoinFilaments:
    def _inputs(self, tmp_path):
        a = pd.concat(
            [
                _filament("m1.mrc", 1, (100, 100), (550, 100), stack="a.mrcs"),
                _filament("m1.mrc", 2, (1000, 100), (1000, 500), stack="a.mrcs", first=101),
            ],
            ignore_index=True,
        )
        b = pd.concat(
            [
                # overlaps file a tube 1, picked in the opposite direction
                _filament("m1.mrc", 1, (800, 100), (400, 100), stack="b.mrcs"),
                # parallel to file a tube 1 but offset
                _filament("m1.mrc", 2, (100, 900), (500, 900), stack="b.mrcs", first=101),
                # collinear with file a tube 2 but separated by a gap
                _filament("m1.mrc", 3, (1000, 700), (1000, 900), stack="b.mrcs", first=201),
                _filament("m2.mrc", 1, (100, 100), (500, 100), stack="b.mrcs", first=301),
                a.iloc[[0]],  # duplicate particle
            ],
            ignore_index=True,
        )
        _write_star(tmp_path / "a.star", a)
        _write_star(tmp_path / "b.star", b)
        return a, b

    def test_join(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a, b = self._inputs(tmp_path)
        _run(monkeypatch, ["a.star", "b.star", "out.star", "--ignoreBadParticlePath", "2", "--verbose", "0"])
        out = starfile.read(tmp_path / "out.star")["particles"]
        # 1 duplicate rlnImageName, and file b particles at x=400-550 overlap file a tube 1
        assert len(out) == len(a) + len(b) - 1 - 4
        m1 = out[out["rlnMicrographName"] == "m1.mrc"]
        m2 = out[out["rlnMicrographName"] == "m2.mrc"]
        assert m1["rlnHelicalTubeID"].nunique() == 4
        assert sorted(m1["rlnHelicalTubeID"].unique()) == [1, 2, 3, 4]
        assert m2["rlnHelicalTubeID"].unique().tolist() == [1]

        joined = m1[m1["rlnCoordinateY"] == 100]
        joined = joined[joined["rlnCoordinateX"] < 900]
        assert joined["rlnHelicalTubeID"].nunique() == 1
        np.testing.assert_allclose(np.sort(joined["rlnCoordinateX"]), np.arange(100, 801, 50))
        # track length runs along the start->end direction of the larger piece (file a)
        np.testing.assert_allclose(
            joined["rlnHelicalTrackLengthAngst"], joined["rlnCoordinateX"] - 100, atol=1e-3
        )
        # unmerged filaments keep their track lengths
        gap = m1[(m1["rlnCoordinateX"] == 1000) & (m1["rlnCoordinateY"] >= 700)]
        np.testing.assert_allclose(
            gap["rlnHelicalTrackLengthAngst"], gap["rlnCoordinateY"] - 700, atol=1e-3
        )

    def test_join_disabled(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a, b = self._inputs(tmp_path)
        _run(
            monkeypatch,
            ["a.star", "b.star", "out.star", "--ignoreBadParticlePath", "2", "--verbose", "0", "--joinFilaments", "0"],
        )
        out = starfile.read(tmp_path / "out.star")["particles"]
        assert len(out) == len(a) + len(b)
        m1 = out[out["rlnMicrographName"] == "m1.mrc"]
        assert sorted(m1["rlnHelicalTubeID"].unique()) == [1, 2, 3]
