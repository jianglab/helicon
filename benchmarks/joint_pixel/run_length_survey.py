"""Run lengths (consecutive segments of a filament in one class) of a Class2D job.

    python run_length_survey.py --job-dir <...> --iteration 20 --name 10940
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=20)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    p = params.copy()
    p["cls"] = p["rlnClassNumber"].astype(int)
    p["track"] = p["rlnHelicalTrackLengthAngst"].astype(float)
    p["fil"] = p.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False).ngroup()
    p = p.sort_values(["fil", "track"], kind="stable").reset_index(drop=True)
    counts = p.cls.value_counts()
    nclass = int((counts >= max(100, 0.002 * len(p))).sum())
    d = p.groupby("fil").track.diff().dropna()
    spacing = float(d[d > 0].median())
    box = images[0].shape[1]
    print(
        f"[{a.name}] {len(p):,} segments, {p.fil.nunique():,} filaments; {len(images)} classes ({nclass} with enough segments); box {box} px x {apix} A = {box * apix:.0f} A; segment spacing {spacing:.1f} A (overlap of neighbours {100 * (1 - spacing / (box * apix)):.0f}%)"
    )
    c, f = p.cls.values, p.fil.values
    start = np.r_[True, (c[1:] != c[:-1]) | (f[1:] != f[:-1])]
    rid = np.cumsum(start) - 1
    length = np.bincount(rid)
    seg_run = length[rid]
    n_runs = len(length)
    print(
        f"  {n_runs:,} runs; mean run {length.mean():.2f} segments; per run, share of lengths 1-8: "
        + " ".join(f"{np.mean(length == k):.2f}" for k in range(1, 9))
        + f" 9+: {np.mean(length >= 9):.2f}"
    )
    print(
        f"  per segment, share in runs of 1/2/3/4-5/6-10/11+: "
        + " / ".join(
            f"{np.mean(m):.2f}"
            for m in [
                seg_run == 1,
                seg_run == 2,
                seg_run == 3,
                (seg_run >= 4) & (seg_run <= 5),
                (seg_run >= 6) & (seg_run <= 10),
                seg_run > 10,
            ]
        )
        + f"; median run of a segment {np.median(seg_run):.0f}"
    )
    # the same, in A along the filament
    tr = p.track.values
    span = np.zeros(n_runs)
    first = np.flatnonzero(start)
    last = np.r_[first[1:] - 1, len(p) - 1]
    span = tr[last] - tr[first] + spacing
    print(
        f"  run extent along the filament (A): median {np.median(span):.0f}, 25/75%: {np.percentile(span, [25, 75]).round(0).tolist()}"
    )
    # how often the class changes between neighbours: the class jumps
    same_fil = f[1:] == f[:-1]
    jump = np.abs(c[1:] - c[:-1])[same_fil]
    print(
        f"  share of consecutive pairs in the same class {np.mean(c[1:][same_fil] == c[:-1][same_fil]):.2f}"
    )
    # order of segments: are the positions at regular steps?
    print(
        f"  spacing quantiles 5/25/50/75/95%: {np.percentile(d[d > 0], [5, 25, 50, 75, 95]).round(1).tolist()}; share of zero spacings {np.mean(d <= 0):.3f}"
    )


if __name__ == "__main__":
    main()
