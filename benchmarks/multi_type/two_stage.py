"""Two stages: filaments grouped by the alignment of their class sequences, then split
by pitch inside each group with the ring fit.

Stage 1 clusters the sampled filaments (average linkage on the alignment score per
letter, ``filament_alignment.py``), which separates types that differ in their
classes. Stage 2 takes each group on its own: its filaments' segments go through the
AbInitio3D analysis (a ring and a repeat for each filament), and two rings are fitted
(``helical_pitch_mixture``); the group is split when the second ring gains enough
(the gain is not a calibrated test: see ``two_ring_test.py``). Every filament ends in
a (group, ring), compared with the reference sets by the adjusted Rand index and a
table of groups against sets.

    python two_stage.py --reference reference.parquet --mrcs run_it025_classes.mrcs \
        --similarity similarity.npy --star run_it025_data.star
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from filament_alignment import add_arguments, prepare


def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    t = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(t, (ai, bi), 1)
    c2 = lambda x: x * (x - 1) / 2
    sij, sa, sb = c2(t).sum(), c2(t.sum(1)).sum(), c2(t.sum(0)).sum()
    exp = sa * sb / c2(len(a))
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def stage_one(A, length, k, method="average"):
    """Groups of the filaments from their alignment scores (per letter)."""
    n = len(A)
    per = A / np.minimum.outer(length, length)
    sim = per / np.percentile(per[np.triu_indices(n, 1)], 99.5)
    D = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(D, 0.0)
    return (
        fcluster(linkage(squareform(D, checks=False), method=method), k, "maxclust") - 1
    )


def table(labels, truth, sets, title):
    t = pd.crosstab(
        pd.Series(labels, name="group"), pd.Series([sets[i] for i in truth], name="set")
    )
    print(f"\n{title}\n{t.to_string()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument("--star", help="Class2D data star, for stage 2")
    p.add_argument(
        "--groups", type=int, default=0, help="stage 1 groups; 0 tries several"
    )
    p.add_argument(
        "--min-group", type=int, default=100, help="filaments a group needs for stage 2"
    )
    p.add_argument("--split-gain", type=float, default=0.08)
    a = p.parse_args()

    d = prepare(a)
    sets, truth, length, A, keys = d["sets"], d["truth"], d["length"], d["A"], d["keys"]
    print("\nStage 1: adjusted Rand index of the groups with the reference sets")
    for k in (3, 4, 5, 6, 8, 10):
        print(f"  {k:2d} groups: ARI {ari(stage_one(A, length, k), truth):.3f}")
    k = a.groups or 5
    g1 = stage_one(A, length, k)
    table(g1, truth, sets, f"stage 1 with {k} groups (filaments per reference set)")
    if not a.star:
        return

    import starfile

    from helicon.webApps.lib import helical_pitch_mixture as mx
    from helicon.webApps.lib import helical_pitch_phase as ph

    star = starfile.read(a.star)
    star = star["particles"] if isinstance(star, dict) else star
    stem = star["rlnMicrographName"].astype(str).map(lambda s: Path(s).stem)
    star["_key"] = stem + "|" + star["rlnHelicalTubeID"].astype(str)
    star = star[star["_key"].isin(set(keys))]
    print(f"\n{len(star):,} segments of the {len(keys)} sampled filaments")
    group_of = dict(zip(keys, g1))
    star["_g"] = star["_key"].map(group_of)

    final = np.zeros(len(keys), dtype=int)
    next_id = 0
    key_index = {k_: i for i, k_ in enumerate(keys)}
    for g in range(k):
        rows = star[star["_g"] == g]
        members = [key_index[x] for x in rows["_key"].unique()]
        print(f"\nstage 2, group {g}: {len(members)} filaments, {len(rows):,} segments")
        ids = None
        counts = rows["rlnClassNumber"].value_counts()
        ids = [int(c) for c in counts[counts >= max(20, 0.002 * len(rows))].index]
        split = None
        if len(members) >= a.min_group and len(ids) >= 3:
            try:
                res = ph.analyze(
                    rows.drop(columns=["_key", "_g"]), class_ids=ids, n_boot=2
                )
                cl = [int(c) for c in res.class_ids]
                pairs = ph.prepare_pairs(res.params_used, class_ids=cl, max_sep=4000.0)
                est = ph.estimate_period(pairs)
                est, _ = ph.sync_filament_directions(pairs, est)
                one = mx.one_ring_objective(pairs, est["period"], est["phases"])
                fil = res.filaments
                look = {
                    (m, int(t)): q
                    for m, t, q, ok in zip(
                        fil.rlnMicrographName,
                        fil.rlnHelicalTubeID,
                        fil.pitch,
                        fil.fitted,
                    )
                    if ok
                }
                pitch = np.array(
                    [look.get((x[0], int(x[1])), np.nan) for x in pairs.filament_keys]
                )
                rng = np.random.default_rng(0)
                best = None
                for lab in [mx.starting_labels(pairs, pitch, rng)] + [
                    rng.integers(0, 2, pairs.n_filaments) for _ in range(3)
                ]:
                    two = mx.fit_two_rings(
                        pairs, lab, (est["period"] * 0.93, est["period"] * 1.07)
                    )
                    if two is not None and (
                        best is None or two["objective"] > best["objective"]
                    ):
                        best = two
                if best is not None:
                    gain = best["objective"] - one
                    lo, hi = sorted(best["periods"])
                    print(
                        f"  one ring {est['period']:.0f} A; two rings {lo:.0f}/{hi:.0f} A, gain {gain:+.3f}, filaments {best['n_filaments'].tolist()}"
                    )
                    if gain >= a.split_gain:
                        keys_g = [
                            Path(str(x[0])).stem + "|" + str(x[1])
                            for x in pairs.filament_keys
                        ]
                        split = dict(zip(keys_g, best["labels"]))
            except Exception as e:
                print("  stage 2 failed:", e)
        for i in members:
            final[i] = next_id + (split.get(keys[i], 0) if split else 0)
        next_id += 2 if split else 1
        print("  split into two rings" if split else "  kept as one group")

    print(
        f"\nafter stage 2: ARI {ari(final, truth):.3f} ({len(set(final))} groups); stage 1 alone: {ari(g1, truth):.3f}"
    )
    table(final, truth, sets, "final groups (filaments per reference set)")


if __name__ == "__main__":
    main()
