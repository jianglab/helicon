import argparse
import sys

import numpy as np
import pandas as pd
import pytest
import starfile

from helicon.commands import images2star
from helicon.lib.exceptions import HeliconError


def _filament(mgraph, tube, start, end, step=50.0, stack="full.mrcs", first=1):
    """Particles extracted from start to end (RELION start-end convention)."""
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


def _recover(monkeypatch, param, inp="refined.star"):
    _run(
        monkeypatch,
        [
            inp,
            "out.star",
            "--ignoreBadParticlePath",
            "2",
            "--verbose",
            "0",
            "--recoverFullFilaments",
            param,
        ],
    )
    return starfile.read("out.star")["particles"]


def _full():
    """The joined filaments, re-extracted with one inter-box distance."""
    return pd.concat(
        [
            _filament("m1.mrc", 1, (100, 100), (900, 100), stack="m1.mrcs"),  # 17
            _filament("m1.mrc", 2, (100, 500), (500, 500), stack="m1.mrcs", first=18),
            _filament("m2.mrc", 1, (100, 100), (500, 100), stack="m2.mrcs"),  # 9
        ],
        ignore_index=True,
    )


def _refined():
    """Refined particles from two picking jobs, A and B, each numbering its own
    tubes: a few pixels off the joined lines and out of phase with the boxes."""
    a1 = _filament("m1.mrc", 1, (110, 101.5), (410, 101.5), stack="A.mrcs")
    b1 = _filament("m1.mrc", 1, (110, 498), (160, 498), stack="B.mrcs")
    b2 = _filament("m1.mrc", 2, (360, 99), (560, 99), stack="B.mrcs", first=10)
    a2 = _filament("m2.mrc", 1, (103, 101), (303, 101), stack="A.mrcs", first=20)
    far = _filament(
        "m1.mrc", 9, (2000, 2000), (2000, 2040), stack="A.mrcs", first=30
    )  # one particle
    return pd.concat([a1, b1, b2, a2, far], ignore_index=True)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_star(tmp_path / "full.star", _full())
    _write_star(tmp_path / "refined.star", _refined())
    return tmp_path


def _filaments(particles):
    return sorted(
        set(zip(particles["rlnMicrographName"], particles["rlnHelicalTubeID"]))
    )


class TestMatchByCoordinates:
    def test_pieces_from_several_picking_jobs(self, inputs, monkeypatch, caplog):
        out = _recover(
            monkeypatch, "minFraction=0.5:matchBy=coords:fullStarFile=full.star"
        )
        # m1 tube 1 is covered 7/17 by A and 5/17 by B, but 10/17 by both;
        # m1 tube 2 only 2/9 (B tube 1, sharing the tube id with A tube 1)
        assert _filaments(out) == [("m1.mrc", 1), ("m2.mrc", 1)]
        assert len(out) == 17 + 9
        assert not out["rlnImageName"].duplicated().any()
        stacks = set(out["rlnImageName"].str.split("@").str[-1])
        assert stacks == {"m1.mrcs", "m2.mrcs"}
        assert "1/20 particles have no particle of full.star" in caplog.text

    def test_auto_min_fraction(self, inputs, monkeypatch):
        # helices (10, 17), (2, 9), (5, 9): minFraction = (17 - 15) / (35 - 30) = 0.4
        out = _recover(monkeypatch, "matchBy=coords:fullStarFile=full.star")
        assert _filaments(out) == [("m1.mrc", 1), ("m2.mrc", 1)]

    def test_max_distance(self, inputs, monkeypatch):
        # the particles of A tube 1 and B tube 2 are 10 pixels from the boxes,
        # those of A tube 1 on m2 only 3 along and 1 across
        out = _recover(
            monkeypatch,
            "minFraction=0.5:matchBy=coords:maxDistance=5:fullStarFile=full.star",
        )
        assert _filaments(out) == [("m2.mrc", 1)]

    def test_input_without_tube_id(self, inputs, monkeypatch):
        _write_star(
            inputs / "refined.star", _refined().drop(columns="rlnHelicalTubeID")
        )
        out = _recover(
            monkeypatch, "minFraction=0.5:matchBy=coords:fullStarFile=full.star"
        )
        assert _filaments(out) == [("m1.mrc", 1), ("m2.mrc", 1)]

    def test_no_match(self, inputs, monkeypatch):
        _write_star(inputs / "refined.star", _refined().iloc[[-1]])
        with pytest.raises(HeliconError, match="no filament"):
            _recover(monkeypatch, "matchBy=coords:fullStarFile=full.star")

    def test_unknown_mode(self, inputs, monkeypatch):
        with pytest.raises(HeliconError, match="matchBy"):
            _recover(monkeypatch, "matchBy=xyz:fullStarFile=full.star")


class TestMatchByImage:
    def test_subset(self, inputs, monkeypatch):
        full = _full()
        _write_star(inputs / "refined.star", full.iloc[[0, 1, 2, 17, 26, 27, 28, 29]])
        out = _recover(monkeypatch, "minFraction=0.3:fullStarFile=full.star")
        assert _filaments(out) == [("m2.mrc", 1)]
        assert len(out) == 9

    def test_not_a_subset(self, inputs, monkeypatch):
        # a particle of an existing filament that is not in the full star file
        particles = _full().iloc[[0, 1]].copy()
        particles.loc[1, "rlnImageName"] = "000099@m1.mrcs"
        _write_star(inputs / "refined.star", particles)
        with pytest.raises(HeliconError, match="1/2 particles are not in"):
            _recover(monkeypatch, "minFraction=0.5:fullStarFile=full.star")
