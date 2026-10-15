"""Metadata ring, pixel ring and the two together on synthetic data.

For every condition (filament length, share of wrong class labels, number of
filaments, position noise) and every random seed, the repeat and the class
phases are estimated three ways with the same segment pairs and the same class
averages: the metadata alone (``alpha = 0``, the existing AbInitio3D ring), the
pixel offsets alone (``alpha = 1``) and the two together. The truth is known.

    python run.py --out results.csv [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic as syn  # noqa: E402

from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402
from helicon.webApps.lib import helical_pitch_pixel as px  # noqa: E402

METHODS = {
    "meta": 0.0,
    "pixel": 1.0,
    "joint": 0.5,
    "joint25": 0.25,
    "joint75": 0.75,
    "joint_gated": "joint_gated",
    "adaptive": "adaptive",
    "meta_cal": "meta_cal",
    "adaptive_cal": "adaptive_cal",
    "adaptive_a75": "adaptive_a75",
    "adaptive_a90": "adaptive_a90",
    "adapt2": "adapt2",
    "adapt2_a75": "adapt2_a75",
    "adapt2_a90": "adapt2_a90",
    "gated_meta": "gated_meta",
    "gated_meta25": "gated_meta25",
}
_CACHE = {}


def averages_and_offsets(period, n_classes, mislabel, seed, apix, structure_seed=0):
    """Class averages and their registration, once per (contamination, seed)."""
    key = (period, n_classes, mislabel, seed, structure_seed)
    if key not in _CACHE:
        # a window a little shorter than the repeat, as real class averages are
        nx = 2 * int(round(0.96 * period / apix / 2))
        images, phase_true = syn.class_averages(
            period,
            n_classes,
            nx=nx,
            apix=apix,
            contamination=mislabel,
            seed=seed,
            structure_seed=structure_seed,
        )
        offsets = px.class_pixel_offsets(images, list(range(n_classes)), apix)
        similarity = px.self_similarity(images, list(range(n_classes)), apix)
        _CACHE[key] = (phase_true, offsets, similarity)
    return _CACHE[key]


def one(
    period,
    n_classes,
    length,
    mislabel,
    n_fil,
    position_sd,
    seed,
    apix=4.0,
    structure_seed=0,
    methods=None,
):
    phase_true, offsets, similarity = averages_and_offsets(
        period, n_classes, mislabel, seed, apix, structure_seed
    )
    params = syn.class2d_table(
        period,
        n_classes,
        n_fil=n_fil,
        length=length,
        mislabel=mislabel,
        position_sd=position_sd,
        seed=seed + 1000,
    )
    pairs = ph.prepare_pairs(params, use_refined_path=False)
    rows = []
    for name, alpha in METHODS.items():
        if methods and name not in methods:
            continue
        t = time.time()
        if alpha == "joint_gated":
            est = px.estimate_period_joint(pairs, offsets, 0.5, similarity=similarity)
        elif alpha == "adaptive":
            est = px.estimate_period_adaptive(pairs, offsets, similarity)
        elif alpha in ("adaptive_a75", "adaptive_a90"):
            est = px.estimate_period_adaptive(
                pairs,
                offsets,
                similarity,
                alpha_short=0.75 if alpha.endswith("75") else 0.9,
            )
        elif alpha in ("gated_meta", "gated_meta25"):
            est = px.estimate_period_gated(
                pairs,
                offsets,
                similarity,
                alpha_basin=0.25 if alpha.endswith("25") else 0.0,
            )
            est["period"] = ph.calibrate_period(pairs, est["period"])["period"]
        elif isinstance(alpha, str) and alpha.startswith("adapt2"):
            short = {"adapt2": None, "adapt2_a75": 0.75, "adapt2_a90": 0.9}[alpha]
            est = px.estimate_period_adaptive(
                pairs, offsets, similarity, alpha_short=short
            )
            if est["stage"] == "metadata":
                est["period"] = ph.calibrate_period(pairs, est["period"])["period"]
        elif alpha == "meta_cal":
            # the AbInitio3D estimate as shipped: the metadata ring, then
            # calibrated against synthetic labels on the data's own geometry
            est = px.estimate_period_joint(pairs, offsets, alpha=0.0)
            est["period"] = ph.calibrate_period(pairs, est["period"])["period"]
        elif alpha == "adaptive_cal":
            est = px.estimate_period_adaptive(pairs, offsets, similarity)
            if est["stage"] == "metadata":
                est["period"] = ph.calibrate_period(pairs, est["period"])["period"]
        else:
            est = px.estimate_period_joint(pairs, offsets, alpha=alpha)
        rows.append(
            dict(
                method=name,
                period=period,
                n_classes=n_classes,
                length=length,
                mislabel=mislabel,
                n_fil=n_fil,
                position_sd=position_sd,
                seed=seed,
                estimate=est["period"],
                error_pct=100.0 * (est["period"] / period - 1.0),
                phase_rms=syn.phase_error(est["phases"], phase_true, period),
                score=est["score"],
                support=est["support"],
                double_ratio=est["double_ratio"],
                seconds=time.time() - t,
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--period", type=float, default=400.0)
    ap.add_argument("--classes", type=int, default=12)
    ap.add_argument("--structure-seed", type=int, default=0)
    ap.add_argument("--methods", default="")
    ap.add_argument(
        "--lengths-in-periods",
        action="store_true",
        help="read the filament lengths and the position noise as repeats",
    )
    a = ap.parse_args()
    if a.quick:
        grid = dict(length=[400.0], mislabel=[0.0, 0.4], n_fil=[60], position_sd=[0.0])
        seeds = 2
    else:
        grid = dict(
            length=[100.0, 200.0, 400.0, 800.0, 1600.0],
            mislabel=[0.0, 0.2, 0.4, 0.6],
            n_fil=[30, 200],
            position_sd=[0.0, 20.0],
        )
        seeds = a.seeds
    if a.lengths_in_periods:
        grid["length"] = [x * a.period / 400.0 for x in grid["length"]]
        grid["position_sd"] = [x * a.period / 400.0 for x in grid["position_sd"]]
    methods = [m for m in a.methods.split(",") if m]
    rows = []
    keys = list(grid)
    conditions = list(itertools.product(*grid.values()))
    for i, vals in enumerate(conditions):
        cond = dict(zip(keys, vals))
        for seed in range(seeds):
            rows += one(
                a.period,
                a.classes,
                seed=seed,
                structure_seed=a.structure_seed,
                methods=methods,
                **cond,
            )
        print(f"{i + 1}/{len(conditions)} {cond}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
