"""Two rings on synthetic mixtures: when are the two types found, and not invented?

Filaments of two types that share their classes (the second type's repeat is
``ratio`` times the first's; ``ratio`` 1 is a single type, the null), at several
filament lengths and shares of wrong labels. Every condition is run for several
seeds, with several random starts of the fit.

    python mixture_synthetic.py --out mixture.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic as syn  # noqa: E402

from helicon.webApps.lib import helical_pitch_mixture as mx  # noqa: E402
from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402


def one(period, ratio, length, mislabel, share, n_fil, seed, restarts=6, n_classes=12):
    periods = (period, period * ratio)
    params, types = syn.class2d_table_mixture(
        periods,
        (1 - share, share),
        n_classes,
        n_fil,
        length,
        mislabel=mislabel,
        seed=seed,
    )
    pairs = ph.prepare_pairs(params, use_refined_path=False, max_sep=2000.0)
    est = ph.estimate_period(pairs)
    est, _ = ph.sync_filament_directions(pairs, est)
    one_obj = mx.one_ring_objective(pairs, est["period"], est["phases"])
    truth = np.array(
        [types[int(k[0][3:6]) * 10 + k[1] - 1] for k in pairs.filament_keys]
    )
    rng = np.random.default_rng(seed + 7)
    best = None
    for _ in range(restarts):
        lab = rng.integers(0, 2, pairs.n_filaments)
        two = mx.fit_two_rings(pairs, lab, (est["period"] * 0.93, est["period"] * 1.07))
        if two is not None and (best is None or two["objective"] > best["objective"]):
            best = two
    if best is None:
        return None
    lo, hi = np.sort(best["periods"])
    acc = max(np.mean(best["labels"] == truth), np.mean(best["labels"] != truth))
    return dict(
        period=period,
        ratio=ratio,
        length=length,
        mislabel=mislabel,
        share=share,
        n_fil=n_fil,
        seed=seed,
        pooled=est["period"],
        one=one_obj,
        two=best["objective"],
        gain=best["objective"] - one_obj,
        found_ratio=hi / lo,
        found_lo=lo,
        found_hi=hi,
        accuracy=acc,
        smaller=min(best["n_filaments"]) / pairs.n_filaments,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    grid = dict(
        ratio=[1.0, 1.03, 1.06, 1.10, 1.20],
        length=[200.0, 400.0, 800.0, 1600.0],
        mislabel=[0.0, 0.3, 0.6],
        share=[0.5, 0.2],
    )
    rows = []
    conds = list(itertools.product(*grid.values()))
    for i, vals in enumerate(conds):
        cond = dict(zip(grid, vals))
        for seed in range(a.seeds):
            r = one(400.0, n_fil=200, seed=seed, **cond)
            if r:
                rows.append(r)
        print(f"{i + 1}/{len(conds)} {cond}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
