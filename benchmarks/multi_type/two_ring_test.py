"""When does a group of classes hold two helical types? The two-ring gain.

For each group of classes (the Types card at ``--cut``), the AbInitio3D analysis
gives one ring; two rings are then fitted from a few starts and the best by
objective kept. The gain of the objective over one ring, the two repeats, and how
the filaments divide are printed, to see how a group of one type differs from a
group of two. A fit with more freedom always gains something, so what matters is
how much the gain is for groups known (or believed) to be of one type.

    python two_ring_test.py --name 10940 --star run_it020_data.star --mrcs run_it020_classes.mrcs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mrcfile
import numpy as np
import starfile

from helicon.webApps.lib import helical_pitch_mixture as mx
from helicon.webApps.lib import helical_pitch_phase as ph
from helicon.webApps.lib import helical_pitch_types as ty

OUT = Path(os.environ.get("TYPES_OUT", "types_out"))


def two_ring_gain(params, class_numbers, restarts=3, seed=0):
    """One ring, two rings, and what the second adds, for the given classes.

    Returns
    -------
    dict or None
        ``classes``, ``filaments``, ``repeat`` (one ring), ``repeats`` (two),
        ``n`` (filaments in each ring), ``span`` (median length of each ring's
        filaments), ``gain`` and ``held`` (the gain on filaments held out,
        :func:`helical_pitch_mixture.heldout_gain`); None if the analysis or the fit failed.
    """
    result = ph.analyze(params, class_ids=class_numbers, n_boot=2)
    cl = [int(c) for c in result.class_ids]
    pairs = ph.prepare_pairs(result.params_used, class_ids=cl, max_sep=4000.0)
    est = ph.estimate_period(pairs)
    est, _ = ph.sync_filament_directions(pairs, est)
    one = mx.one_ring_objective(pairs, est["period"], est["phases"])
    fil = result.filaments
    look = {
        (m, int(t)): q
        for m, t, q, ok in zip(
            fil.rlnMicrographName, fil.rlnHelicalTubeID, fil.pitch, fil.fitted
        )
        if ok
    }
    pitch = np.array([look.get((k[0], int(k[1])), np.nan) for k in pairs.filament_keys])
    rng = np.random.default_rng(seed)
    starts = [mx.starting_labels(pairs, pitch, rng)] + [
        rng.integers(0, 2, pairs.n_filaments) for _ in range(restarts - 1)
    ]
    best = None
    for labels in starts:
        two = mx.fit_two_rings(
            pairs, labels, (est["period"] * 0.93, est["period"] * 1.07)
        )
        if two is not None and (best is None or two["objective"] > best["objective"]):
            best = two
    if best is None:
        return None
    held = mx.heldout_gain(
        pairs, starts[0], (est["period"] * 0.93, est["period"] * 1.07), est["period"]
    )
    span = np.asarray(pairs.filament_span, float)
    return dict(
        classes=len(cl),
        filaments=pairs.n_filaments,
        repeat=float(est["period"]),
        repeats=[float(x) for x in best["periods"]],
        n=[int(x) for x in best["n_filaments"]],
        span=[float(np.median(span[best["labels"] == t])) for t in (0, 1)],
        gain=float(best["objective"] - one),
        held=float(held),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--star", required=True)
    p.add_argument("--mrcs", required=True)
    p.add_argument("--cut", type=float, default=0.15)
    a = p.parse_args()

    with mrcfile.open(a.mrcs, permissive=True) as f:
        imgs = [np.asarray(x, float) for x in f.data]
    d = starfile.read(a.star)
    d = d["particles"] if isinstance(d, dict) else d
    cls = d["rlnClassNumber"].to_numpy().astype(int) - 1
    total = np.bincount(cls, minlength=len(imgs)).astype(float)
    floor = max(100, int(0.002 * total.sum()))
    classes = [
        c
        for c in range(len(imgs))
        if total[c] >= floor and imgs[c].max() > imgs[c].min()
    ]
    S = np.load(OUT / f"similarity_{a.name}.npy")
    groups = ty.pixel_groups(S, a.cut)
    print(
        f"{a.name}: {len(classes)} classes, {int(groups.max()) + 1} groups at cut {a.cut}"
    )
    for g in range(int(groups.max()) + 1):
        numbers = [classes[i] + 1 for i in np.flatnonzero(groups == g)]
        sub = d[d["rlnClassNumber"].astype(int).isin(numbers)]
        try:
            r = two_ring_gain(sub, numbers)
        except Exception as e:
            print(f"  group {g}: {len(numbers)} classes failed: {e}")
            continue
        if r is None:
            print(f"  group {g}: {len(numbers)} classes, no two-ring fit")
            continue
        lo, hi = sorted(r["repeats"])
        print(
            f"  group {g}: {r['classes']} classes, {r['filaments']} filaments, one ring {r['repeat']:.0f} A; "
            f"two rings {lo:.0f}/{hi:.0f} A (ratio {hi / lo:.2f}), filaments {r['n']}, "
            f"median length {r['span'][0]:.0f}/{r['span'][1]:.0f} A, gain {r['gain']:+.3f}, held-out {r['held']:+.3f}"
        )


if __name__ == "__main__":
    main()
