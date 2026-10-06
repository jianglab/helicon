"""Split a group of classes that holds two helical types by the filaments' repeats.

The classes are grouped by their averages (the Types card, ``--cut``), and the
group holding most of two reference sets is taken. Two rings are fitted to it
(``helical_pitch_mixture.fit_two_rings``), each filament going to the ring it
agrees with better, from several starts; the start with the best objective is
kept (no labels used). The filaments' rings are then compared with the reference
sets (filaments mostly in one or the other).

    python pair_mixture.py --class2d run_it025_data.star --mrcs run_it025_classes.mrcs \
        --reference reference.parquet --set-a typeA --set-b typeB
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import starfile

from helicon.webApps.lib import helical_pitch_mixture as mx
from helicon.webApps.lib import helical_pitch_phase as ph
from helicon.webApps.lib import helical_pitch_types as ty

OUT = Path(os.environ.get("TYPES_OUT", "types_out"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--class2d", required=True)
    p.add_argument("--mrcs", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--set-a", required=True)
    p.add_argument("--set-b", required=True)
    p.add_argument("--cut", type=float, default=0.15)
    p.add_argument("--restarts", type=int, default=8)
    a = p.parse_args()

    ref = pd.read_parquet(a.reference)
    cls = ref["rlnClassNumber"].to_numpy().astype(int) - 1
    n = int(cls.max()) + 1
    with mrcfile.open(a.mrcs, permissive=True) as f:
        has_signal = [x.max() > x.min() for x in f.data]
    total = np.bincount(cls, minlength=n).astype(float)
    floor = max(100, int(0.002 * total.sum()))
    classes = [c for c in range(n) if total[c] >= floor and has_signal[c]]
    S = np.load(OUT / "similarity.npy")
    groups = ty.pixel_groups(S, a.cut)
    per = {
        s: np.bincount(cls, weights=ref[s].to_numpy(float), minlength=n)[classes]
        for s in (a.set_a, a.set_b)
    }
    k = int(
        np.argmax(
            [
                (per[a.set_a] + per[a.set_b])[groups == g].sum()
                for g in range(groups.max() + 1)
            ]
        )
    )
    chosen = [classes[i] + 1 for i in np.flatnonzero(groups == k)]
    print(
        f"group {k} at cut {a.cut}: {len(chosen)} classes; reference segments "
        f"{a.set_a} {per[a.set_a][groups == k].sum():,.0f}, {a.set_b} {per[a.set_b][groups == k].sum():,.0f}"
    )

    d = starfile.read(a.class2d)
    d = d["particles"] if isinstance(d, dict) else d
    d = d[d["rlnClassNumber"].astype(int).isin(chosen)].copy()
    result = ph.analyze(d, class_ids=chosen, n_boot=2)
    cl = [int(c) for c in result.class_ids]
    pairs = ph.prepare_pairs(result.params_used, class_ids=cl, max_sep=4000.0)
    est = ph.estimate_period(pairs)
    est, _ = ph.sync_filament_directions(pairs, est)
    one = mx.one_ring_objective(pairs, est["period"], est["phases"])
    print(
        f"{len(cl)} classes, {pairs.n_filaments} filaments; one ring: repeat {est['period']:.1f} A, objective {one:.4f}"
    )

    fil = result.filaments
    look = {
        (m, int(t)): q
        for m, t, q, ok in zip(
            fil.rlnMicrographName, fil.rlnHelicalTubeID, fil.pitch, fil.fitted
        )
        if ok
    }
    pitch = np.array([look.get((x[0], int(x[1])), np.nan) for x in pairs.filament_keys])
    rng = np.random.default_rng(0)
    starts = [("by repeat", mx.starting_labels(pairs, pitch, rng))] + [
        (f"random {i}", rng.integers(0, 2, pairs.n_filaments))
        for i in range(a.restarts)
    ]
    # reference: the filaments mostly in one set or the other
    ref["fil"] = (
        ref["rlnMicrographName"].astype(str) + "|" + ref["rlnHelicalTubeID"].astype(str)
    )
    sh = ref.groupby("fil")[[a.set_a, a.set_b]].sum()
    fa = (sh[a.set_a] / (sh[a.set_a] + sh[a.set_b]).clip(lower=1)).to_dict()
    nn = (sh[a.set_a] + sh[a.set_b]).to_dict()
    keys = [Path(str(x[0])).stem + "|" + str(x[1]) for x in pairs.filament_keys]
    truth = np.array(
        [
            (
                (0 if fa.get(q, 0.5) >= 0.8 else 1 if fa.get(q, 0.5) <= 0.2 else -1)
                if nn.get(q, 0) >= 5
                else -1
            )
            for q in keys
        ]
    )
    print(
        f"filaments with a reference call: {int((truth >= 0).sum())} ({int((truth == 0).sum())} {a.set_a}, {int((truth == 1).sum())} {a.set_b})"
    )

    best = None
    for name, labels in starts:
        two = mx.fit_two_rings(
            pairs, labels, (est["period"] * 0.93, est["period"] * 1.07)
        )
        if two is None:
            continue
        lab = two["labels"]
        ok = truth >= 0
        agree = max(np.mean(lab[ok] == truth[ok]), np.mean(lab[ok] != truth[ok]))
        lo, hi = np.sort(two["periods"])
        print(
            f"  start {name:10s}: repeats {lo:.1f} / {hi:.1f} A, filaments {two['n_filaments'].tolist()}, "
            f"objective gain {two['objective'] - one:+.4f}, agreement with the reference {agree:.2f}"
        )
        if best is None or two["objective"] > best[0]["objective"]:
            best = (two, name, agree)
    if best:
        print(f"best by objective: start {best[1]}, agreement {best[2]:.2f}")
        lab, ok = best[0]["labels"], truth >= 0
        flip = np.mean(lab[ok] == truth[ok]) < 0.5
        right = (lab == truth) ^ flip
        span = np.asarray(pairs.filament_span, float)
        edges = np.percentile(span[ok], [0, 25, 50, 75, 100])
        print("agreement by filament length (reference-called filaments):")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok & (span >= lo) & (span <= hi)
            print(
                f"  {lo:5.0f}-{hi:5.0f} A: {int(m.sum())} filaments, {right[m].mean():.2f}"
            )
        rep = best[0]["periods"]
        for t, name in ((0, a.set_a), (1, a.set_b)):
            ring = int(t != flip)
            print(f"  {name}: ring of {rep[ring]:.1f} A")


if __name__ == "__main__":
    main()
