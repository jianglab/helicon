"""The two rings of EMPIAR-10230 against the manually labelled PHF and SF filaments.

The segments of the RELION job come from two extractions, Extract/job670 (the
labelled PHF coordinates, 86% of its segments found by coordinate in
PHF_particles.star) and Extract/job671 (SF_particles.star, 86%): the source of a
segment is its type. The labels are used here to score the fit and to fit each
type on its own (the oracle), never to guide the fit. They are manual and
"not quite perfect".

    python real_10230_truth.py --job-dir <.../job684> --iteration 25 --out t.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402
import synthetic as syn  # noqa: E402

from helicon.webApps.lib import helical_pitch_mixture as mx  # noqa: E402
from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402

SF_JOB = "Extract/job671"


def type_of_filaments(used, keys):
    """Share of each filament's segments from the SF extraction, in ``keys`` order."""
    sf = used["rlnImageName"].str.contains(SF_JOB).astype(float)
    share = sf.groupby(
        [used["rlnMicrographName"].values, used["rlnHelicalTubeID"].values]
    ).mean()
    return np.array([share.get((k[0], k[1]), np.nan) for k in keys])


def oracle(pairs, member, lo=500.0, hi=1100.0):
    w = member[pairs.pair_filament].astype(float)
    return ph.estimate_period(pairs, pair_weights=w, periods=np.arange(lo, hi, 2.0))


def phase_diff(a, b, period):
    """RMS difference of two sets of class phases, as a share of the repeat."""
    best = np.inf
    for s in (1.0, -1.0):
        d = s * np.asarray(a) - np.asarray(b)
        c = np.angle(np.mean(np.exp(1j * d)))
        r = np.angle(np.exp(1j * (d - c)))
        best = min(best, float(np.sqrt(np.mean(r**2)) / (2 * np.pi)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=25)
    ap.add_argument("--rule", default="all")
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    params, images, apix = R.load(a.job_dir, a.iteration)
    numbers = R.selections(params, images)[a.rule]
    result = R.merged_classes(params, images, apix, numbers)
    cls = [int(c) for c in result.class_ids]
    used = result.params_used
    pairs = ph.prepare_pairs(used, class_ids=cls, max_sep=R.MAX_SEP)
    est = ph.estimate_period(pairs)
    est, _ = ph.sync_filament_directions(pairs, est)
    one = mx.one_ring_objective(pairs, est["period"], est["phases"])

    share = type_of_filaments(used, pairs.filament_keys)
    pure_phf = share <= 0.1
    pure_sf = share >= 0.9
    print(
        f"{pairs.n_filaments} filaments: {int(pure_phf.sum())} PHF-pure, "
        f"{int(pure_sf.sum())} SF-pure, {int((~pure_phf & ~pure_sf).sum())} mixed "
        f"(SF share between 0.1 and 0.9); segments {len(used):,}",
        flush=True,
    )

    # the oracle: each type fitted on its own filaments
    o_phf, o_sf = oracle(pairs, pure_phf), oracle(pairs, pure_sf)
    o_all = oracle(pairs, pure_phf | pure_sf)
    print(
        f"ORACLE (labels used): PHF filaments alone {o_phf['period']:.1f} A "
        f"(score {o_phf['score']:.3f}), SF alone {o_sf['period']:.1f} A "
        f"(score {o_sf['score']:.3f}); both pooled {o_all['period']:.1f} A; "
        f"data's one ring {est['period']:.1f} A",
        flush=True,
    )

    # the two-ring fit, without labels
    fil = result.filaments
    lookup = {
        (m, int(t)): p
        for m, t, p, ok in zip(
            fil.rlnMicrographName, fil.rlnHelicalTubeID, fil.pitch, fil.fitted
        )
        if ok
    }
    pitch = np.array(
        [lookup.get((k[0], int(k[1])), np.nan) for k in pairs.filament_keys]
    )
    rng = np.random.default_rng(0)
    starts = [mx.starting_labels(pairs, pitch, rng)] + [
        rng.integers(0, 2, pairs.n_filaments) for _ in range(a.restarts)
    ]
    best = None
    for lab in starts:
        two = mx.fit_two_rings(pairs, lab, (est["period"] * 0.93, est["period"] * 1.07))
        if two is not None and (best is None or two["objective"] > best["objective"]):
            best = two
    order = np.argsort(best["periods"])
    lo_t, hi_t = best["periods"][order]
    print(
        f"TWO RINGS (no labels): {lo_t:.1f} A ({best['n_filaments'][order[0]]} "
        f"filaments) and {hi_t:.1f} A ({best['n_filaments'][order[1]]}); objective "
        f"{best['objective']:.4f} (one ring {one:.4f})",
        flush=True,
    )

    # scored against the labels, on the filaments that are pure
    keep = pure_phf | pure_sf
    truth = pure_sf[keep].astype(int)  # 1 = SF
    ring = (best["labels"][keep] == order[1]).astype(int)  # 1 = the longer repeat
    seg_w = np.bincount(pairs.seg_filament, minlength=pairs.n_filaments)[keep]
    tab = pd.crosstab(
        pd.Series(np.where(truth == 1, "SF", "PHF"), name="true type"),
        pd.Series(
            np.where(ring == 1, f"ring {hi_t:.0f} A", f"ring {lo_t:.0f} A"), name="ring"
        ),
    )
    print("\nconfusion (pure filaments):\n" + tab.to_string(), flush=True)
    for name, r_long in (
        ("longer repeat = SF", ring == 1),
        ("longer repeat = PHF", ring == 0),
    ):
        predicted_sf = r_long if name.endswith("SF") else ~r_long
        acc = float(np.mean(predicted_sf.astype(int) == truth))
        accw = float(np.sum((predicted_sf.astype(int) == truth) * seg_w) / seg_w.sum())
        print(f"  {name}: filaments {acc:.3f}, segments {accw:.3f}")
    base = max(truth.mean(), 1 - truth.mean())
    print(f"  (always the larger type: {base:.3f})")

    span = pairs.filament_span[keep]
    print("\naccuracy by filament span (the better of the two labelings):")
    for lo, hi in ((0, 400), (400, 800), (800, 1600), (1600, 1e9)):
        m = (span >= lo) & (span < hi)
        if m.sum() < 10:
            continue
        acc = max(np.mean(ring[m] == truth[m]), np.mean(ring[m] != truth[m]))
        print(f"  span {lo:5.0f}-{hi:6.0f} A: n={int(m.sum()):4d}, accuracy {acc:.3f}")

    # the class phases against the oracle's
    o_by = {0: o_phf, 1: o_sf}
    print("\nclass phases of each ring against the oracle's (RMS, share of repeat):")
    for t in (0, 1):
        for name, o in (("PHF", o_phf), ("SF", o_sf)):
            print(
                f"  ring {best['periods'][t]:.0f} A vs oracle {name} "
                f"{o['period']:.0f} A: {phase_diff(best['phases'][t], o['phases'], o['period']):.3f}"
            )
    # an upper bound: the rings fitted on the true types, and the objective they reach
    scores = []
    for o in (o_phf, o_sf):
        keep_s, swap_s, w = mx.filament_scores(pairs, o["period"], o["phases"])
        scores.append(np.maximum(keep_s, swap_s))
    tot = max(float(w.sum()), 1e-12)
    truth_all = np.where(pure_sf, 1, 0)
    obj_true = float(
        (np.where(truth_all == 1, scores[1], scores[0])[keep] * w[keep]).sum()
        / w[keep].sum()
    )
    obj_found = float(
        (np.maximum.reduce([scores[0], scores[1]])[keep] * w[keep]).sum()
        / w[keep].sum()
    )
    print(
        f"\nmean agreement on the pure filaments: with the true types' rings and "
        f"true assignment {obj_true:.4f}; best of the two oracle rings {obj_found:.4f}; "
        f"one ring {mx.one_ring_objective(pairs, est['period'], est['phases']):.4f}"
    )
    pd.DataFrame(
        dict(
            span=pairs.filament_span,
            sf_share=share,
            ring=(best["labels"] == order[1]).astype(int),
        )
    ).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
