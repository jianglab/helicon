"""Two stages: filaments grouped by their class usage, then split by pitch inside each group.

Stage 1 describes each filament by the share of its segments in each class, blurred
by the similarity of the class averages (a segment counts partly towards the classes
that look like its own), and groups the filaments by k-means on that. It separates
types that differ in their classes, without first putting each class in one group.

Stage 2 takes each group on its own: its segments go through the AbInitio3D analysis
(a ring of classes and a repeat), one ring and two rings are fitted
(``helical_pitch_mixture``), and the group is split when a second ring helps on
filaments it was not fitted to (the held-out gain) and the two repeats are clearly
apart. Every filament of the data set is grouped; the filaments of exactly one
reference set score the result (adjusted Rand index, the share of filaments in a
group led by their own set, and a table of groups against sets).

    python usage_two_stage.py --star run_it025_data.star --reference reference.parquet \
        --mrcs run_it025_classes.mrcs --similarity similarity.npy --groups 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import starfile
from scipy.cluster.vq import kmeans2

from two_stage import ari

SKIP = (
    "rlnMicrographName",
    "rlnHelicalTubeID",
    "rlnHelicalTrackLengthAngst",
    "rlnClassNumber",
)


def usage_groups(fil, letter, S, k, starts=5):
    """k-means groups of the filaments by their class usage blurred by ``S``.

    ``fil`` is each segment's filament and ``letter`` its class among those in
    ``S`` (-1 if none); the start whose groups are tightest is kept.
    """
    n_fil = int(fil.max()) + 1
    U = np.zeros((n_fil, len(S)))
    ok = letter >= 0
    np.add.at(U, (fil[ok], letter[ok]), 1)
    U /= np.maximum(U.sum(1, keepdims=True), 1)
    F = U @ S
    F -= F.mean(0)
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    best = None
    for seed in range(starts):
        centres, lab = kmeans2(F, k, seed=seed, minit="++")
        cost = float(((F - centres[lab]) ** 2).sum())
        if best is None or cost < best[0]:
            best = (cost, lab)
    return best[1]


def stem_key(name):
    """A filament key ``micrograph|tube`` with the micrograph reduced to its stem."""
    mic, tube = name.rsplit("|", 1)
    return f"{Path(mic).stem}|{tube}"


def purity(groups, truth):
    """Share of filaments in a group whose largest set is their own."""
    right = sum(np.bincount(truth[groups == g]).max() for g in np.unique(groups))
    return right / len(truth)


def split_by_rings(rows, min_filaments, min_gain, min_apart):
    """Fit one and two rings to the segments ``rows`` of one group.

    Returns
    -------
    tuple
        A dict (filament key -> ring 0 or 1) if the group splits, else None; and a
        line describing the fit.
    """
    from helicon.webApps.lib import helical_pitch_mixture as mx
    from helicon.webApps.lib import helical_pitch_phase as ph

    counts = rows["rlnClassNumber"].value_counts()
    ids = [int(c) for c in counts[counts >= max(20, 0.002 * len(rows))].index]
    if len(ids) < 3:
        return None, "too few classes"
    res = ph.analyze(rows, class_ids=ids, n_boot=2)
    cl = [int(c) for c in res.class_ids]
    pairs = ph.prepare_pairs(res.params_used, class_ids=cl, max_sep=4000.0)
    est = ph.estimate_period(pairs)
    est, _ = ph.sync_filament_directions(pairs, est)
    one = mx.one_ring_objective(pairs, est["period"], est["phases"])
    fil = res.filaments
    look = {
        (m, int(t)): q
        for m, t, q, ok in zip(
            fil.rlnMicrographName, fil.rlnHelicalTubeID, fil.pitch, fil.fitted
        )
        if ok
    }
    pitch = np.array([look.get((x[0], int(x[1])), np.nan) for x in pairs.filament_keys])
    rng = np.random.default_rng(0)
    start = mx.starting_labels(pairs, pitch, rng)
    periods = (est["period"] * 0.93, est["period"] * 1.07)
    best = None
    for lab in [start] + [rng.integers(0, 2, pairs.n_filaments) for _ in range(3)]:
        two = mx.fit_two_rings(pairs, lab, periods)
        if two is not None and (best is None or two["objective"] > best["objective"]):
            best = two
    if best is None:
        return None, f"one ring {est['period']:.0f} A; two rings did not fit"
    held = mx.heldout_gain(pairs, start, periods, est["period"])
    lo, hi = sorted(best["periods"])
    n = np.bincount(best["labels"], minlength=2)
    line = (
        f"one ring {est['period']:.0f} A; two rings {lo:.0f} and {hi:.0f} A "
        f"({n[0]} and {n[1]} filaments); gain {best['objective'] - one:+.3f}, "
        f"held out {held:+.3f}"
    )
    if held < min_gain or hi / lo - 1 < min_apart or n.min() < min_filaments:
        return None, line
    keys = [f"{Path(str(x[0])).stem}|{int(x[1])}" for x in pairs.filament_keys]
    return dict(zip(keys, best["labels"])), line


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--star", required=True, help="the Class2D _data.star")
    p.add_argument("--reference", required=True)
    p.add_argument("--mrcs", required=True)
    p.add_argument("--similarity", required=True)
    p.add_argument("--groups", type=int, default=8, help="stage 1 groups")
    p.add_argument(
        "--min-filaments", type=int, default=50, help="each ring of a split needs this"
    )
    p.add_argument(
        "--min-gain", type=float, default=0.01, help="held-out gain needed to split"
    )
    p.add_argument(
        "--min-apart",
        type=float,
        default=0.08,
        help="the two repeats must differ by this fraction to split",
    )
    a = p.parse_args()

    star = starfile.read(a.star)["particles"]
    ref = pd.read_parquet(a.reference)
    assert (
        len(ref) == len(star)
        and (
            ref["rlnClassNumber"].to_numpy() == star["rlnClassNumber"].to_numpy()
        ).all()
    ), "the reference is not in the order of the star file"
    sets = [c for c in ref.columns if c not in SKIP]
    cls = star["rlnClassNumber"].to_numpy().astype(int) - 1
    n_classes = int(cls.max()) + 1
    with mrcfile.open(a.mrcs, permissive=True) as f:
        has_signal = [x.max() > x.min() for x in f.data]
    total = np.bincount(cls, minlength=n_classes).astype(float)
    floor = max(100, int(0.002 * total.sum()))
    classes = [c for c in range(n_classes) if total[c] >= floor and has_signal[c]]
    S = np.load(a.similarity)
    assert S.shape == (len(classes),) * 2, "the similarity does not match the classes"
    letter = np.full(n_classes, -1)
    letter[classes] = np.arange(len(classes))

    key = (
        star["rlnMicrographName"].astype(str)
        + "|"
        + star["rlnHelicalTubeID"].astype(str)
    ).astype("category")
    fil = key.cat.codes.to_numpy()
    names = np.asarray(key.cat.categories, dtype=str)
    n_fil = len(names)
    per_fil = np.zeros((n_fil, len(sets)), dtype=np.int64)
    np.add.at(per_fil, fil, ref[sets].to_numpy().astype(np.int64))
    scored = np.flatnonzero((per_fil > 0).sum(1) == 1)
    truth = per_fil[scored].argmax(1)
    print(
        f"{n_fil:,} filaments, {len(star):,} segments; {len(scored):,} filaments of "
        f"exactly one set score the result"
    )

    g1 = usage_groups(fil, letter[cls], S, a.groups)

    def report(groups, title):
        g = groups[scored]
        print(
            f"\n{title}: ARI {ari(g, truth):.3f}, {purity(g, truth):.1%} of filaments "
            f"in a group led by their own set"
        )
        t = pd.crosstab(
            pd.Series(g, name="group"), pd.Series([sets[i] for i in truth], name="set")
        )
        print(t.to_string())

    report(g1, f"stage 1, {a.groups} groups")

    final = np.zeros(n_fil, dtype=int)
    next_id = 0
    drop = [c for c in star.columns if c.startswith("_")]
    for g in range(a.groups):
        members = np.flatnonzero(g1 == g)
        rows = star.iloc[np.flatnonzero(g1[fil] == g)].drop(columns=drop)
        t0 = time.time()
        try:
            split, line = split_by_rings(rows, a.min_filaments, a.min_gain, a.min_apart)
        except Exception as e:  # a group the analysis cannot handle stays whole
            split, line = None, f"analysis failed: {e}"
        print(
            f"\ngroup {g}: {len(members):,} filaments, {len(rows):,} segments "
            f"({time.time() - t0:.0f} s)\n  {line}\n  "
            + ("split in two" if split else "kept as one")
        )
        if split:
            matched = sum(stem_key(names[i]) in split for i in members)
            print(f"  {matched:,} of the group's filaments have a ring")
        final[members] = next_id
        if split:
            final[members] += np.array(
                [split.get(stem_key(names[i]), 0) for i in members]
            )
        next_id += 2 if split else 1
    report(final, f"after stage 2, {next_id} groups")


if __name__ == "__main__":
    main()
