"""The joint ring on a real data set: EMPIAR-10940 (tau filaments), RELION Class2D.

Runs the AbInitio3D pipeline as the tab does (counterpart merge, ring, filament
directions, calibration) and the pixel-assisted estimates on the same segment
pairs, on the full data and under stress:

* ``subsets``: random subsets of the filaments (the amount of data)
* ``degrade``: a share of the segments put in a random class (poor assignments;
  the class averages stay as they are, which flatters the pixels)
* ``trim``: filaments cut to their first ``L`` A (shorter filaments)

The truth is the deposited structure's twist (1.21 deg per 4.75 A rise, C2 or a
2_1 axis, so a projection repeat of 706.6 A); it is used only to score.

    python real_10940.py --job-dir <.../Class2D/job010> --out real.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic as syn  # noqa: E402

# where the studies save and read their intermediate results (a similarity matrix,
# a table); set JOINT_PIXEL_OUT to move it
OUT = Path(os.environ.get("JOINT_PIXEL_OUT", "joint_pixel_out"))
OUT.mkdir(parents=True, exist_ok=True)

from helicon.webApps.lib import helical_pitch_compute as compute  # noqa: E402
from helicon.webApps.lib import helical_pitch_map as maps  # noqa: E402
from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402
from helicon.webApps.lib import helical_pitch_pixel as px  # noqa: E402

RISE = 4.75
TWIST = 1.21
FACTOR = 2  # the projection repeats at half the pitch (C2 or a 2_1 axis)
P_REF = 360.0 * RISE / (FACTOR * TWIST)
MAX_SEP = 2000.0


def load(job_dir, iteration):
    job = Path(job_dir)
    raw = compute.get_class2d_helix_params_from_file(
        str(job / f"run_it{iteration:03d}_data.star")
    )
    params, _, _ = compute.split_distinct_filaments(raw, 50)
    data, apix = compute.get_class2d_from_file(
        str(job / f"run_it{iteration:03d}_classes.mrcs")
    )
    return params, list(data), float(apix)


def selections(params, images):
    """The two selection rules, fixed in advance (class numbers, 1-based)."""
    n = len(images)
    counts = compute.get_class_abundance(params, n)
    nonblank = [i for i in range(n) if images[i].max() > images[i].min()]
    by_size = sorted(nonblank, key=lambda i: -counts[i])
    return {
        "top20": [i + 1 for i in by_size[:20]],
        "all": [i + 1 for i in nonblank],
    }


def merged_classes(params, images, apix, class_numbers):
    """The production analysis (as the tab runs it) on the full data."""
    at = [c - 1 for c in class_numbers]
    found = ph.pair_counterparts(images, at)
    counterparts = [
        dict(a=d["a"] + 1, b=d["b"] + 1, corr=d["corr"], dx=d["dx"]) for d in found
    ]
    result = ph.analyze(
        params,
        class_ids=class_numbers,
        n_boot=2,
        counterparts=counterparts,
        image_apix=apix,
        max_sep=5700.0,
    )
    return result


def oriented_averages(result, images, apix):
    cls = [int(c) for c in result.class_ids]
    chosen = [images[c - 1] for c in cls]
    return maps.straighten_classes(chosen, result.class_zdir, apix, apix)


def estimate_all(used, cls, offsets, sim, sign, ref_phases=None, ref_period=None):
    """Meta (as shipped), gated and joint estimates on one set of segments."""
    t0 = time.time()
    pairs = ph.prepare_pairs(used, class_ids=cls, max_sep=MAX_SEP)
    if len(pairs.D) == 0:
        return []
    est = ph.estimate_period(pairs)
    est, flipped = ph.sync_filament_directions(pairs, est)
    if flipped.any():
        est = ph.estimate_period(pairs)
    rows = []

    def record(name, e, period, secs):
        row = dict(
            method=name,
            estimate=period,
            err_vs_ref_pct=100.0 * (period / P_REF - 1.0),
            score=e["score"],
            support=e["support"],
            double_ratio=e["double_ratio"],
            n_pairs=len(pairs.D),
            n_filaments=pairs.n_filaments,
            seconds=secs,
        )
        if ref_period:
            row["err_vs_full_pct"] = 100.0 * (period / ref_period - 1.0)
        if ref_phases is not None:
            ref_A = -np.asarray(ref_phases) * ref_period / (2 * np.pi)
            row["phase_rms_vs_full"] = syn.phase_error(e["phases"], ref_A, ref_period)
        rows.append(row)

    t = time.time()
    cal = ph.calibrate_period(pairs, est["period"])
    record("meta_cal", est, cal["period"], time.time() - t0)
    t = time.time()
    g = px.estimate_period_gated(pairs, offsets, sim, sign=sign)
    record(
        "gated_meta",
        g,
        ph.calibrate_period(pairs, g["period"])["period"],
        time.time() - t,
    )
    t = time.time()
    j = px.estimate_period_joint(pairs, offsets, 0.5, sign=sign, similarity=sim)
    record("joint_gated", j, j["period"], time.time() - t)
    t = time.time()
    a = px.estimate_period_adaptive(pairs, offsets, sim, sign=sign)
    period = a["period"]
    if a["stage"] == "metadata":
        period = ph.calibrate_period(pairs, period)["period"]
    record("adaptive", a, period, time.time() - t)
    return rows


def filament_keys(df):
    return df.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False).ngroup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rules", default="top20,all")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()

    params, images, apix = load(a.job_dir, a.iteration)
    print(
        f"{len(params):,} segments, {len(images)} classes, {apix} A/pixel", flush=True
    )
    sels = selections(params, images)
    allrows = []

    def save():
        pd.DataFrame(allrows).to_csv(a.out, index=False)

    for rule in a.rules.split(","):
        numbers = sels[rule]
        t = time.time()
        result = merged_classes(params, images, apix, numbers)
        cls = [int(c) for c in result.class_ids]
        used = result.params_used
        print(
            f"[{rule}] {len(numbers)} classes -> {len(cls)} after merging "
            f"({len(result.merged_classes)} merged); production period "
            f"{result.period:.1f} A ({100 * (result.period / P_REF - 1):+.2f}% vs "
            f"{P_REF:.1f}); {time.time() - t:.0f} s",
            flush=True,
        )
        oriented = oriented_averages(result, images, apix)
        offsets = px.class_pixel_offsets(oriented, list(range(len(cls))), apix)
        sim = px.self_similarity(oriented, list(range(len(cls))), apix)
        sign, medians = px.pixel_sign(offsets, result.phases, result.period)
        print(
            f"[{rule}] {len(offsets)} registered class pairs of "
            f"{len(offsets.dx)}; pixel sign {sign:+.0f} (median class residual "
            f"{medians[0]:.1f} A at +1, {medians[1]:.1f} A at -1)",
            flush=True,
        )
        _, per_class = px.pixel_residuals(offsets, result.phases, result.period, sign)
        diag = pd.DataFrame(
            dict(
                rule=rule,
                cls=cls,
                fit=result.class_fit,
                pixel_residual_A=per_class,
                segments=result.class_count,
            )
        )
        diag.to_csv(a.out.replace(".csv", f"_{rule}_classes.csv"), index=False)

        # reference phases: the full-data joint ring
        full = estimate_all(used, cls, offsets, sim, sign)
        ref_period = [r for r in full if r["method"] == "meta_cal"][0]["estimate"]
        pairs0 = ph.prepare_pairs(used, class_ids=cls, max_sep=MAX_SEP)
        e0 = ph.estimate_period(pairs0)
        e0, fl = ph.sync_filament_directions(pairs0, e0)
        ref = px.estimate_period_gated(pairs0, offsets, sim, sign=sign)
        ref_phases = ref["phases"]
        for r in full:
            allrows.append(dict(rule=rule, stage="full", param=0, seed=0, **r))
        save()
        print(
            pd.DataFrame(full)[["method", "estimate", "err_vs_ref_pct", "score"]]
            .round(3)
            .to_string(index=False),
            flush=True,
        )

        key = filament_keys(used)
        n_fil = int(key.max()) + 1
        # --- subsets of filaments
        for n in (20, 50, 150, 500):
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                pick = rng.choice(n_fil, size=min(n, n_fil), replace=False)
                sub = used[np.isin(key.values, pick)]
                try:
                    rows = estimate_all(
                        sub, cls, offsets, sim, sign, ref_phases, ref_period
                    )
                except ValueError:
                    continue
                for r in rows:
                    allrows.append(
                        dict(rule=rule, stage="subsets", param=n, seed=seed, **r)
                    )
            save()
            print(f"[{rule}] subsets {n} done", flush=True)
        # --- poor assignments
        for eps in (0.2, 0.4, 0.6):
            for seed in range(a.seeds):
                rng = np.random.default_rng(100 + seed)
                bad = used.copy()
                flip = rng.random(len(bad)) < eps
                labels = bad["rlnClassNumber"].values.copy()
                labels[flip] = rng.choice(cls, size=int(flip.sum()))
                bad["rlnClassNumber"] = labels
                rows = estimate_all(
                    bad, cls, offsets, sim, sign, ref_phases, ref_period
                )
                for r in rows:
                    allrows.append(
                        dict(rule=rule, stage="degrade", param=eps, seed=seed, **r)
                    )
            save()
            print(f"[{rule}] degrade {eps} done", flush=True)
        # --- shorter filaments: the first L A of every filament
        track = used["rlnHelicalTrackLengthAngst"].astype(float)
        start = track.groupby(key.values).transform("min")
        for length in (350.0, 700.0, 1400.0):
            sub = used[(track - start).values <= length]
            rows = estimate_all(sub, cls, offsets, sim, sign, ref_phases, ref_period)
            for r in rows:
                allrows.append(dict(rule=rule, stage="trim", param=length, seed=0, **r))
            save()
            print(f"[{rule}] trim {length} done", flush=True)


if __name__ == "__main__":
    main()
