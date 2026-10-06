"""Trace the segments of a Class2D job into the 3D sets that were refined from it.

A Class2D job's segments were selected, re-extracted and refined in many rounds
into a few 3D reconstructions, each of one helical type (by extensive manual
work). Every refined segment keeps its micrograph, its helical tube and its
position along the tube (``rlnHelicalTrackLengthAngst``), so it can be found
again among the Class2D segments. This writes, for every Class2D segment, which
of the sets it ended in. The sets are a noisy reference (a set may hold some
segments of other types), used only to score methods, never to guide them.

    python build_reference.py --class2d run_it025_data.star \
        --set typeA=Refine3D/job1/run_data.star --set typeB=... --out reference.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

KEYS = ["rlnMicrographName", "rlnHelicalTubeID", "rlnHelicalTrackLengthAngst"]


def load(path) -> pd.DataFrame:
    df = starfile.read(path)
    if isinstance(df, dict):
        df = df.get("particles", next(v for v in df.values() if "rlnImageName" in v))
    df = df[[c for c in KEYS + ["rlnClassNumber"] if c in df.columns]].copy()
    df["rlnMicrographName"] = (
        df["rlnMicrographName"].astype(str).map(lambda s: Path(s).stem)
    )
    return df


def match(parent: pd.DataFrame, child: pd.DataFrame, tol: float) -> np.ndarray:
    """For each parent segment: is there a child segment on the same tube within ``tol`` A?"""
    child = child.sort_values(KEYS)
    grp = child.groupby(KEYS[:2], sort=False)["rlnHelicalTrackLengthAngst"]
    pos = {k: np.sort(v.to_numpy()) for k, v in grp}
    hit = np.zeros(len(parent), bool)
    for key, idx in parent.groupby(KEYS[:2], sort=False).indices.items():
        arr = pos.get(key)
        if arr is None:
            continue
        x = parent["rlnHelicalTrackLengthAngst"].to_numpy()[idx]
        j = np.clip(np.searchsorted(arr, x), 1, len(arr) - 1) if len(arr) > 1 else None
        if j is None:
            hit[idx] = np.abs(arr[0] - x) <= tol
        else:
            near = np.minimum(np.abs(arr[j] - x), np.abs(arr[j - 1] - x))
            hit[idx] = near <= tol
    return hit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--class2d", required=True, help="run_itNNN_data.star of the Class2D job"
    )
    p.add_argument("--set", action="append", default=[], metavar="NAME=STAR")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--tol", type=float, default=None, help="A; default 0.6 of the spacing"
    )
    a = p.parse_args()

    parent = load(a.class2d)
    step = float(
        parent.sort_values(KEYS)
        .groupby(KEYS[:2])["rlnHelicalTrackLengthAngst"]
        .diff()
        .dropna()
        .replace(0, np.nan)
        .median()
    )
    tol = a.tol or 0.6 * step
    print(
        f"{len(parent):,} Class2D segments, spacing {step:.2f} A, tolerance {tol:.2f} A"
    )
    for spec in a.set:
        name, star = spec.split("=", 1)
        child = load(star)
        parent[name] = match(parent, child, tol)
        print(
            f"{name}: {len(child):,} segments, {int(parent[name].sum()):,} found in the Class2D job"
        )
    parent.to_parquet(a.out)


if __name__ == "__main__":
    main()
