import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import starfile

import helicon
from helicon.commands import images2star
from helicon.plugins.images2star import savestartend
from helicon.lib.exceptions import (
    HeliconError,
    HeliconFileExistsError,
    HeliconValueError,
)


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


def _write_star(path, particles, voltage=300.0):
    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [1],
            "rlnOpticsGroupName": ["opticsGroup1"],
            "rlnImagePixelSize": [1.0],
            "rlnImageSize": [64],
            "rlnVoltage": [voltage],
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
        assert helicon.filament_pieces_overlap([0, 0], [400, 0], [300, 0], [700, 0])

    def test_contained(self):
        assert helicon.filament_pieces_overlap([0, 0], [400, 0], [100, 0], [200, 0])

    def test_collinear_with_gap(self):
        assert not helicon.filament_pieces_overlap([0, 0], [400, 0], [500, 0], [700, 0])

    def test_parallel_offset(self):
        assert not helicon.filament_pieces_overlap([0, 0], [400, 0], [0, 50], [400, 50])

    def test_crossing(self):
        assert not helicon.filament_pieces_overlap(
            [0, 0], [400, 0], [200, 0], [200, 300]
        )

    def test_single_particle_on_filament(self):
        assert helicon.filament_pieces_overlap([0, 0], [400, 0], [150, 0], [150, 0])


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
        out = helicon.join_collinear_filaments(
            data, piece_attrs=["src", "rlnHelicalTubeID"]
        )
        assert sorted(out["rlnHelicalTubeID"].unique()) == [1, 2]
        joined = out[out["rlnCoordinateY"] == 0]
        assert joined["rlnHelicalTubeID"].nunique() == 1
        assert out[out["rlnCoordinateY"] == 500]["rlnHelicalTubeID"].nunique() == 1
        # overlapping particles are removed, leaving one particle every 50 pixels
        np.testing.assert_allclose(
            np.sort(joined["rlnCoordinateX"]), np.arange(0, 1001, 50)
        )
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
        out = helicon.join_collinear_filaments(
            data, piece_attrs=["src", "rlnHelicalTubeID"]
        )
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
            [
                _filament("m1", 1, (0, 0), (300, 400), step=20),
                _filament("m1", 2, (0, 500), (0, 900), step=20),
            ],
            ignore_index=True,
        )
        d = helicon.estimate_inter_box_distance(
            data, ["rlnMicrographName", "rlnHelicalTubeID"]
        )
        assert d == pytest.approx(20)


class TestImages2starJoinFilaments:
    def _inputs(self, tmp_path):
        a = pd.concat(
            [
                _filament("m1.mrc", 1, (100, 100), (550, 100), stack="a.mrcs"),
                _filament(
                    "m1.mrc", 2, (1000, 100), (1000, 500), stack="a.mrcs", first=101
                ),
            ],
            ignore_index=True,
        )
        b = pd.concat(
            [
                # overlaps file a tube 1, picked in the opposite direction
                _filament("m1.mrc", 1, (800, 100), (400, 100), stack="b.mrcs"),
                # parallel to file a tube 1 but offset
                _filament(
                    "m1.mrc", 2, (100, 900), (500, 900), stack="b.mrcs", first=101
                ),
                # collinear with file a tube 2 but separated by a gap
                _filament(
                    "m1.mrc", 3, (1000, 700), (1000, 900), stack="b.mrcs", first=201
                ),
                _filament(
                    "m2.mrc", 1, (100, 100), (500, 100), stack="b.mrcs", first=301
                ),
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
        _run(
            monkeypatch,
            [
                "a.star",
                "b.star",
                "out.star",
                "--ignoreBadParticlePath",
                "2",
                "--verbose",
                "0",
            ],
        )
        star = starfile.read(tmp_path / "out.star")
        # both inputs have the same optics group: it is written once
        assert len(star["optics"]) == 1
        out = star["particles"]
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
        np.testing.assert_allclose(
            np.sort(joined["rlnCoordinateX"]), np.arange(100, 801, 50)
        )
        # track length runs along the start->end direction of the larger piece (file a)
        np.testing.assert_allclose(
            joined["rlnHelicalTrackLengthAngst"],
            joined["rlnCoordinateX"] - 100,
            atol=1e-3,
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
            [
                "a.star",
                "b.star",
                "out.star",
                "--ignoreBadParticlePath",
                "2",
                "--verbose",
                "0",
                "--joinFilaments",
                "0",
            ],
        )
        out = starfile.read(tmp_path / "out.star")["particles"]
        assert len(out) == len(a) + len(b)
        m1 = out[out["rlnMicrographName"] == "m1.mrc"]
        assert sorted(m1["rlnHelicalTubeID"].unique()) == [1, 2, 3]

    def test_conflicting_optics_groups_are_an_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a, b = self._inputs(tmp_path)
        _write_star(tmp_path / "b.star", b, voltage=200.0)
        with pytest.raises(HeliconValueError, match="optics group 1"):
            _run(
                monkeypatch,
                [
                    "a.star",
                    "b.star",
                    "out.star",
                    "--ignoreBadParticlePath",
                    "2",
                    "--verbose",
                    "0",
                ],
            )
        assert not (tmp_path / "out.star").exists()


def _segments(coords):
    """Start-end pairs of a RELION start-end coordinate table, as sorted tuples."""
    xy = coords[["rlnCoordinateX", "rlnCoordinateY"]].values.reshape(-1, 4)
    return sorted(tuple(np.round(r, 2)) for r in xy)


class TestFilamentStartEnd:
    def test_direction_follows_track_length(self):
        data = pd.concat(
            [
                _filament("m1", 1, (100, 100), (500, 400)),
                # picked from the right to the left, listed in reverse order
                _filament("m1", 2, (800, 0), (400, 0)).iloc[::-1],
                _filament("m1", 3, (0, 900), (0, 1000)).iloc[[0]],
            ],
            ignore_index=True,
        )
        lines = helicon.filament_start_end(data)
        assert lines["rlnHelicalTubeID"].tolist() == [1, 2, 3]
        np.testing.assert_allclose(
            lines[["startX", "startY", "endX", "endY"]].values[:2],
            [[100, 100, 500, 400], [800, 0, 400, 0]],
            atol=1e-6,
        )
        np.testing.assert_allclose(lines["length"][:2], [500, 400])
        assert lines["n"].tolist() == [11, 9, 1]
        np.testing.assert_allclose(lines["maxOffset"], 0, atol=1e-6)

    def test_direction_without_track_length(self):
        data = _filament("m1", 1, (800, 0), (400, 0)).drop(
            columns="rlnHelicalTrackLengthAngst"
        )
        lines = helicon.filament_start_end(data)
        np.testing.assert_allclose(
            lines[["startX", "startY", "endX", "endY"]].values, [[800, 0, 400, 0]]
        )

    def test_best_fit_line(self):
        data = _filament("m1", 1, (0, 0), (400, 0))
        data.loc[data.index % 2 == 1, "rlnCoordinateY"] = 4.0
        lines = helicon.filament_start_end(data)
        assert lines["startY"].iloc[0] == pytest.approx(lines["endY"].iloc[0])
        # 5 particles at y=0 and 4 at y=4: the line is at y=16/9
        assert lines["startY"].iloc[0] == pytest.approx(16 / 9)
        assert lines["maxOffset"].iloc[0] == pytest.approx(4 - 16 / 9)


class TestSaveStartEnd:
    @pytest.mark.parametrize(
        "micrograph, subpath",
        [
            ("MotionCorr/job002/Movies/m1.mrc", "Movies/m1_manualpick.star"),
            ("/data/proj/MotionCorr/job002/a/b/m1.mrc", "a/b/m1_manualpick.star"),
            ("micrographs/m1.mrc", "m1_manualpick.star"),
        ],
    )
    def test_coordinate_file_subpath(self, micrograph, subpath):
        assert str(savestartend._start_end_file_subpath(micrograph)) == subpath

    def test_join_and_save(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        TestImages2starJoinFilaments()._inputs(tmp_path)
        argv = [
            "a.star",
            "b.star",
            "out.star",
            "--ignoreBadParticlePath",
            "2",
            "--verbose",
            "0",
            "--saveStartEnd",
            "startend",
        ]
        _run(monkeypatch, argv)
        star = starfile.read(tmp_path / "startend" / "manualpick.star")
        # the RELION 5.1+ pick type block comes first
        assert list(star) == ["general", "coordinate_files"]
        assert star["general"] == {"rlnMicrographPickType": "startend"}
        coord_list = star["coordinate_files"]
        assert coord_list["rlnMicrographName"].tolist() == ["m1.mrc", "m2.mrc"]
        files = coord_list["rlnMicrographCoordinates"].tolist()
        assert [Path(f).name for f in files] == [
            "m1_manualpick.star",
            "m2_manualpick.star",
        ]
        m1 = starfile.read(files[0])
        assert m1.columns.tolist() == [
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnClassNumber",
            "rlnAnglePsi",
            "rlnAutopickFigureOfMerit",
        ]
        # initialized as RELION AutoPick does
        assert (m1[["rlnClassNumber", "rlnAutopickFigureOfMerit"]] == 0).all(axis=None)
        assert (m1["rlnAnglePsi"] == -999).all()
        # the joined filament spans file a tube 1 and file b tube 1, in the
        # start->end direction of the larger piece (file a)
        assert _segments(m1) == [
            (100, 100, 800, 100),
            (100, 900, 500, 900),
            (1000, 100, 1000, 500),
            (1000, 700, 1000, 900),
        ]
        assert _segments(starfile.read(files[1])) == [(100, 100, 500, 100)]

        # an existing coordinate list is not overwritten without --force
        with pytest.raises(HeliconFileExistsError, match="manualpick.star"):
            _run(monkeypatch, argv[:2] + ["out2.star"] + argv[3:])
        _run(monkeypatch, argv[:2] + ["out2.star"] + argv[3:] + ["--force", "1"])

    def test_relion_project_paths(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "default_pipeline.star").write_text("")
        (tmp_path / "Select/job010").mkdir(parents=True)
        mgraph = "MotionCorr/job002/Movies/m1.mrc"
        particles = pd.concat(
            [
                _filament(mgraph, 1, (100, 100), (500, 100)),
                _filament(mgraph, 2, (0, 0), (0, 100), first=50).iloc[[0]],
            ],
            ignore_index=True,
        )
        _write_star(tmp_path / "a.star", particles)
        _run(
            monkeypatch,
            [
                "a.star",
                "Select/job010/particles.star",
                "--ignoreBadParticlePath",
                "2",
                "--verbose",
                "0",
                "--saveStartEnd",
                "External/job011",
            ],
        )
        coord_list = starfile.read(tmp_path / "External/job011/manualpick.star")[
            "coordinate_files"
        ]
        assert coord_list["rlnMicrographName"].tolist() == [mgraph]
        assert coord_list["rlnMicrographCoordinates"].tolist() == [
            "External/job011/Movies/m1_manualpick.star"
        ]
        # the single-particle filament has no start-end line
        coords = starfile.read(tmp_path / "External/job011/Movies/m1_manualpick.star")
        assert _segments(coords) == [(100, 100, 500, 100)]

    def test_requires_helical_tube_id(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        particles = _filament("m1.mrc", 1, (0, 0), (200, 0))
        _write_star(tmp_path / "a.star", particles.drop(columns="rlnHelicalTubeID"))
        with pytest.raises(HeliconError, match="rlnHelicalTubeID"):
            _run(
                monkeypatch,
                [
                    "a.star",
                    "out.star",
                    "--ignoreBadParticlePath",
                    "2",
                    "--verbose",
                    "0",
                    "--saveStartEnd",
                    "startend",
                ],
            )
