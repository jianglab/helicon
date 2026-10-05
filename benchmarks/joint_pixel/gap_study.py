"""How gaps left by a 2D class selection distort run lengths, and a gap-aware count.

Select2D removes whole classes, so a filament keeps its segments only where they were
in a kept class: a run of one class is cut wherever a removed class interrupted it
and the segments that were next to each other are no longer. Here the gaps are made
on EMPIAR-10940 (which has none: segments every 14.2 A) by removing classes at
random until a given share of the segments is gone.

    python gap_study.py --job-dir <...> --iteration 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402


def run_table(p, spacing, gap_aware, max_run=12):
    """Distribution of the run lengths (shares of runs of 1..max_run, then more).

    ``p`` has ``fil``, ``track`` and ``cls`` sorted by filament and position. Naively
    a run is consecutive rows of one class. Gap-aware: the filament is cut into chains
    where the spacing exceeds 1.5 times the nominal one (a gap), a run is counted only
    if it is complete -- neither end of it at a gap or the end of the filament -- and
    its length is in segments.
    """
    c, f, t = p.cls.values, p.fil.values, p.track.values
    gap = (
        np.r_[True, (f[1:] != f[:-1]) | ((t[1:] - t[:-1]) > 1.5 * spacing)]
        if gap_aware
        else np.r_[True, f[1:] != f[:-1]]
    )
    chain = np.cumsum(gap) - 1
    start = np.r_[True, (c[1:] != c[:-1]) | (chain[1:] != chain[:-1])]
    rid = np.cumsum(start) - 1
    length = np.bincount(rid)
    first = np.flatnonzero(start)
    last = np.r_[first[1:] - 1, len(c) - 1]
    if gap_aware:
        # complete: not the first or last run of its chain
        ch_first = gap[first]
        ch_last = np.r_[gap[first[1:]], True]
        keep = ~(ch_first | ch_last)
    else:
        keep = np.ones(len(length), bool)
    L = length[keep]
    h = np.array(
        [np.mean(L == k) for k in range(1, max_run + 1)] + [np.mean(L > max_run)]
    )
    return h, int(keep.sum()), float(L.mean())


def distance(a, b):
    """Total variation distance between two distributions."""
    return 0.5 * float(np.abs(a - b).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=20)
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    p = params.copy()
    p["cls"] = p["rlnClassNumber"].astype(int)
    p["track"] = p["rlnHelicalTrackLengthAngst"].astype(float)
    p["fil"] = p.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False).ngroup()
    p = p.sort_values(["fil", "track"], kind="stable").reset_index(drop=True)
    d = p.groupby("fil").track.diff().dropna()
    spacing = float(d[d > 0].median())
    print(f"{len(p):,} segments, spacing {spacing:.1f} A")
    ref, n_ref, mean_ref = run_table(p, spacing, gap_aware=True)
    naive_full, _, mean_full = run_table(p, spacing, gap_aware=False)
    print(
        f"no gaps: complete runs {n_ref:,}, mean {mean_ref:.2f}; shares of lengths 1-6: {ref[:6].round(3).tolist()}"
    )
    counts = p.cls.value_counts()
    rng = np.random.default_rng(0)
    print(
        "\nremoved   naive: mean run, distance to the full data   gap-aware: complete runs, mean, distance   share of consecutive pairs still neighbours"
    )
    for share in (0.1, 0.2, 0.3, 0.5):
        rows = []
        for seed in range(5):
            r = np.random.default_rng(seed)
            order = r.permutation(counts.index.values)
            cum = np.cumsum(counts[order].values) / len(p)
            gone = set(order[: int(np.searchsorted(cum, share)) + 1])
            q = p[~p.cls.isin(gone)].reset_index(drop=True)
            hn, _, mn = run_table(q, spacing, gap_aware=False)
            hg, ng, mg = run_table(q, spacing, gap_aware=True)
            near = np.mean(
                np.diff(q.track.values)[q.fil.values[1:] == q.fil.values[:-1]]
                <= 1.5 * spacing
            )
            rows.append(
                (
                    mn,
                    distance(hn, naive_full),
                    ng,
                    mg,
                    distance(hg, ref),
                    near,
                    1 - len(q) / len(p),
                )
            )
        m = np.mean(rows, axis=0)
        print(
            f"{100 * m[6]:5.0f}%   naive {m[0]:.2f}, {m[1]:.3f}        gap-aware {int(m[2]):6,d}, {m[3]:.2f}, {m[4]:.3f}        {m[5]:.2f}"
        )


if __name__ == "__main__":
    main()
