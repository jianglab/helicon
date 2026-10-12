"""Filament paths from the curated picks of an EMPIAR dataset, per micrograph.

The truth for the micrograph denoising benchmarks: the segment centres of the
particle sets that were picked and curated by the authors, joined into one
polyline per filament. Coordinates are in pixels of the micrograph, as RELION
writes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import starfile

# Where the data sets are: the folder that holds the downloaded EMPIAR entries
# (HELICON_BENCH_EMPIAR), and the RELION project of EMPIAR-10940 (in vitro tau,
# "easy"; HELICON_BENCH_EMPIAR_10940), whose refined particle set is the curated
# pick.  Nothing is guessed: a missing variable is an error that says so.


def _folder(variable: str) -> Path:
    import os

    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"set {variable} to the folder of the data set")
    return Path(value)


def _tau_10230() -> Path:
    return (
        _folder("HELICON_BENCH_EMPIAR") / "empiar-10230-AD-tau/data/AD_patient2_EMPIAR"
    )


def _tau_10940() -> Path:
    return _folder("HELICON_BENCH_EMPIAR_10940")


REFINED_10940 = "Refine3D/job014/run_data.star"

# micrograph binning that brings each dataset to about 5 Angstrom per pixel
BIN = {"10230": 4, "10940": 6}


@dataclass
class MicrographTruth:
    """One micrograph and the filaments picked on it.

    Attributes
    ----------
    name : str
        The micrograph's file stem.
    path : pathlib.Path
        The micrograph (.mrc).
    paths : list of numpy.ndarray
        One ``(n, 2)`` array of ``(x, y)`` pixel coordinates per filament, in
        order along the filament.
    kinds : list of str
        The type each filament was picked as (``"PHF"``, ``"SF"``).
    """

    name: str
    path: Path
    paths: list
    kinds: list


def _stem(image_name: str) -> str:
    return Path(image_name.split("@")[1]).stem


def _subset(mics: list, limit: int | None, seed: int) -> list:
    mics.sort(key=lambda m: m.name)
    if limit is not None and limit < len(mics):
        rng = np.random.default_rng(seed)
        mics = [mics[i] for i in sorted(rng.choice(len(mics), limit, replace=False))]
    return mics


def tau_10230(limit: int | None = None, seed: int = 0) -> list[MicrographTruth]:
    """The EMPIAR-10230 micrographs with their PHF and SF paths.

    Parameters
    ----------
    limit : int, optional
        Use this many micrographs, chosen at random (``seed``); default all.

    Returns
    -------
    list of MicrographTruth
        Micrographs that have at least one filament.
    """
    tau = _tau_10230()
    coords = tau / "Coordinates"
    out: dict[str, MicrographTruth] = {}
    for kind in ("PHF", "SF"):
        df = starfile.read(coords / f"{kind}_particles.star")
        df = df.assign(_mic=df["rlnImageName"].map(_stem))
        for (mic, tube), grp in df.groupby(["_mic", "rlnHelicalTubeID"], sort=False):
            grp = grp.sort_values("rlnHelicalTrackLength")
            xy = grp[["rlnCoordinateX", "rlnCoordinateY"]].to_numpy(float)
            if len(xy) < 2:
                continue
            m = out.setdefault(
                mic,
                MicrographTruth(mic, tau / "DW_Micrographs" / f"{mic}.mrc", [], []),
            )
            m.paths.append(xy)
            m.kinds.append(kind)
    return _subset([m for m in out.values() if m.path.exists()], limit, seed)


def tau_10940(limit: int | None = None, seed: int = 0) -> list[MicrographTruth]:
    """The EMPIAR-10940 micrographs with the filaments of the refined particle set.

    Parameters
    ----------
    limit : int, optional
        Use this many micrographs, chosen at random (``seed``); default all.

    Returns
    -------
    list of MicrographTruth
        Micrographs that have at least one filament.
    """
    tau = _tau_10940()
    df = starfile.read(tau / REFINED_10940)["particles"]
    out = []
    for mic, grp in df.groupby("rlnMicrographName", sort=False):
        path = tau / mic
        if not path.exists():
            continue
        m = MicrographTruth(Path(mic).stem, path, [], [])
        for _, tube in grp.groupby("rlnHelicalTubeID"):
            tube = tube.sort_values("rlnHelicalTrackLengthAngst")
            xy = tube[["rlnCoordinateX", "rlnCoordinateY"]].to_numpy(float)
            if len(xy) >= 2:
                m.paths.append(xy)
                m.kinds.append("tau")
        if m.paths:
            out.append(m)
    return _subset(out, limit, seed)


def load(dataset: str, limit: int | None = None, seed: int = 0) -> list:
    """The micrographs of ``dataset`` (``"10230"`` or ``"10940"``) with their paths."""
    return {"10230": tau_10230, "10940": tau_10940}[dataset](limit, seed)
