"""The type of each segment from the classes along its filament, against the Types card.

The Types card groups the classes by how alike their averages are and gives each
filament the group most of its segments are in. ``helical_pitch_context`` reads
the sequence of classes along each filament instead: a hidden Markov chain over
the positions of a filament, with a type that stays and a class that follows the
type's own transitions (a port of RELION's ``helix_filament_context``). Both are
run on every filament of the reference that belongs to exactly one set, and scored
against the sets: the adjusted Rand index with the sets, the purity of the groups
(the share of filaments in the majority set of their group: a mixed group is the
costly error), and for each pair of sets the balanced accuracy of telling them
apart by the groups.

    python filament_context.py --reference reference.parquet \
        --mrcs run_it025_classes.mrcs --similarity similarity.npy --types 3,4,5,6,8
"""

from __future__ import annotations

import argparse
import time

import mrcfile
import numpy as np
import pandas as pd

from helicon.webApps.lib import helical_pitch_context as hc
from helicon.webApps.lib import helical_pitch_types as ty

KEYS = ("rlnMicrographName", "rlnHelicalTubeID", "rlnHelicalTrackLengthAngst")


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def purity(groups, truth):
    """Share of the filaments that are in the majority set of their group."""
    ok = groups >= 0
    hit = 0
    for g in np.unique(groups[ok]):
        hit += np.bincount(truth[ok & (groups == g)]).max()
    return hit / max(int(ok.sum()), 1)


def balanced(groups, truth, i, j):
    """Best balanced accuracy of telling set i from set j by their groups."""
    ok = groups >= 0
    a, b = groups[ok & (truth == i)], groups[ok & (truth == j)]
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    k = int(groups.max()) + 1
    pa = np.bincount(a, minlength=k) / len(a)
    pb = np.bincount(b, minlength=k) / len(b)
    return float(
        0.5 * (np.where(pa >= pb, pa, 0).sum() + np.where(pb > pa, pb, 0).sum())
    )


def reference_filaments(ref, sets, chains, min_segments):
    """The set of each filament of ``chains`` that is in exactly one set (else -1)."""
    flags = ref[sets].to_numpy()
    in_set = np.zeros((chains.n_filaments, len(sets)), dtype=np.int64)
    np.add.at(in_set, chains.filament, flags.astype(np.int64))
    n = np.bincount(chains.filament, minlength=chains.n_filaments)
    one = ((in_set > 0).sum(1) == 1) & (n >= min_segments)
    return np.where(one, in_set.argmax(1), -1)


def vote(chains, class_group):
    """Each filament's group: the one most of its segments' classes are in."""
    n_groups = int(class_group.max()) + 1
    g = class_group[chains.cls]
    ok = (chains.cls >= 0) & (g >= 0)
    counts = np.zeros((chains.n_filaments, n_groups))
    np.add.at(counts, (chains.filament[ok], g[ok]), 1.0)
    out = counts.argmax(1)
    out[counts.sum(1) == 0] = -1
    return out


def report(name, groups, truth, sets, seconds=None):
    use = truth >= 0
    g = groups[use]
    g = np.where(g < 0, g.max() + 1, g)
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            b = balanced(np.where(use, groups, -1), truth, i, j)
            if np.isfinite(b):
                pairs.append(b)
    t = f"  {seconds:6.1f} s" if seconds is not None else ""
    print(
        f"  {name:34s} ARI {ari(g, truth[use]):.3f}  purity {purity(g, truth[use]):.3f}  "
        f"mean pair accuracy {np.mean(pairs):.3f} (min {np.min(pairs):.3f}){t}",
        flush=True,
    )
    return pairs


def alike(T, ft, r, chains, classes, S, truth, sets, pmax=None):
    """Which types look like the same structure by their class averages, and the truth."""
    usage = hc.class_usage(chains, r["tpost"])[:, classes]
    L = hc.type_likeness(usage, S)
    use = truth >= 0
    # the set most of the filaments of each type are in
    major = np.full(T, -1)
    for t in range(T):
        sel = use & (ft == t)
        if sel.sum():
            major[t] = np.bincount(truth[sel], minlength=len(sets)).argmax()
    same, diff = [], []
    for i in range(T):
        for j in range(i + 1, T):
            (same if major[i] == major[j] else diff).append(L[i, j])
    fmt = lambda v: "none" if not v else f"{np.min(v):.2f} to {np.max(v):.2f}"
    print(
        f"      between-type likeness of types of one set: {fmt(same)}; "
        f"of different sets: {fmt(diff)}",
        flush=True,
    )
    if pmax is not None:
        # does a type differ in how confidently its segments are classified?
        fp = np.bincount(chains.filament, weights=pmax, minlength=chains.n_filaments)
        fp = fp / np.maximum(
            np.bincount(chains.filament, minlength=chains.n_filaments), 1
        )
        print(
            "      by type (set most of its filaments are in, filaments, mean class probability): "
            + "; ".join(
                f"{sets[major[t]] if major[t] >= 0 else '-'} {int((use & (ft == t)).sum())} "
                f"{fp[use & (ft == t)].mean():.2f}"
                for t in range(T)
                if (use & (ft == t)).any()
            ),
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--mrcs", required=True)
    p.add_argument("--similarity", required=True)
    p.add_argument("--types", default="3,4,5,6,8")
    p.add_argument("--min-segments", type=int, default=12)
    p.add_argument("--iter", type=int, default=3)
    p.add_argument("--temper", type=float, default=1.0)
    p.add_argument("--prob", type=float, default=0.9, help="probability of every class")
    p.add_argument(
        "--pmax",
        default="",
        help="parquet with a column pmax (rlnMaxValueProbDistribution) in the order of the reference",
    )
    p.add_argument(
        "--cv-max-segments", type=int, default=0, help="0: no cross-validation"
    )
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--app", action="store_true", help="also run it as AbInitio3D does")
    p.add_argument("--app-fit-segments", type=int, default=400_000)
    p.add_argument(
        "--no-full", action="store_true", help="skip the fits of all segments"
    )
    p.add_argument(
        "--counterparts",
        action="store_true",
        help="with --app: merge the classes that are 180-degree counterparts and read "
        "the filaments picked backwards the other way round",
    )
    a = p.parse_args()

    ref = pd.read_parquet(a.reference)
    sets = [c for c in ref.columns if c not in KEYS + ("rlnClassNumber",)]
    pmax_all = pd.read_parquet(a.pmax)["pmax"].to_numpy() if a.pmax else None
    t0 = time.time()
    chains = hc.build_chains(
        ref["rlnMicrographName"],
        ref["rlnHelicalTubeID"],
        ref["rlnHelicalTrackLengthAngst"],
        ref["rlnClassNumber"],
        prob=(
            pd.read_parquet(a.pmax)["pmax"].to_numpy()
            if a.pmax
            else np.full(len(ref), a.prob)
        ),
        temper=a.temper,
    )
    print(
        f"{chains.n_segments:,} segments, {chains.n_filaments:,} filaments, "
        f"{chains.n_chains:,} chains (longest {chains.chain_len[0]} steps of "
        f"{chains.step:.1f} A), {chains.n_classes} classes; laid out in "
        f"{time.time() - t0:.1f} s",
        flush=True,
    )
    truth = reference_filaments(ref, sets, chains, a.min_segments)
    print(
        "filaments of exactly one set, with at least",
        a.min_segments,
        "segments:",
        {s: int((truth == i).sum()) for i, s in enumerate(sets)},
        flush=True,
    )

    # the Types card: the classes grouped by their averages, then a vote
    with mrcfile.open(a.mrcs, permissive=True) as f:
        has_signal = [x.max() > x.min() for x in f.data]
    total = np.bincount(chains.cls[chains.cls >= 0], minlength=chains.n_classes)
    floor = max(100, int(0.002 * total.sum()))
    classes = [
        c for c in range(chains.n_classes) if total[c] >= floor and has_signal[c]
    ]
    S = np.load(a.similarity)
    assert S.shape == (len(classes), len(classes)), "similarity does not match classes"
    print("\nTypes card first stage (class groups by their averages, then a vote)")
    pairs_card = {}
    for k in (3, 4, 5, 6, 8):
        cg, _ = ty.pixel_groups_n(S, k)
        letter = np.full(chains.n_classes, -1)
        letter[classes] = cg
        pairs_card[k] = report(
            f"card, {k} class groups", vote(chains, letter), truth, sets
        )

    # the Types card as the app now does it: classes that look like no other are
    # left out, then the filaments are split by sets of classes one side never uses
    grouped = np.flatnonzero(ty.pixel_groups(S) >= 0)
    if len(grouped) < 3:
        grouped = np.arange(len(classes))
    used = [classes[i] for i in grouped]
    Sg = S[np.ix_(grouped, grouped)]
    col = np.full(chains.n_classes, -1)
    col[used] = np.arange(len(used))
    ok = col[chains.cls] >= 0
    usage = np.zeros((chains.n_filaments, len(used)))
    np.add.at(usage, (chains.filament[ok], col[chains.cls[ok]]), 1.0)
    t0 = time.time()
    splits = ty.marker_groups(usage, Sg)
    print(
        f"\nTypes card as the app does it (marker splits; {len(splits)} splits, {time.time() - t0:.0f} s)"
    )
    for n in range(2, min(len(splits), 7) + 2):
        report(f"card markers, {n} groups", splits[n - 2]["labels"], truth, sets)

    print("\nFilament context (hidden Markov chain over the filaments)", flush=True)
    results = {}
    for T in [] if a.no_full else [int(x) for x in a.types.split(",")]:
        t0 = time.time()
        r = hc.fit(chains, T, n_iter=a.iter, cpu=a.cpu or None)
        dt = time.time() - t0
        ft, share = hc.filament_types(chains, r["tpost"])
        results[T] = (ft, r)
        report(f"context, {T} types (rho {r['model'].rho:.2g})", ft, truth, sets, dt)
        alike(T, ft, r, chains, used, Sg, truth, sets, pmax_all)
        print(
            f"      log-likelihood per segment {r['ll_train'] / chains.n_segments:.4f}",
            flush=True,
        )

    if a.app:
        # what AbInitio3D does: the classes of the Types card only, a model learnt from a
        # part of the filaments, then every filament decoded with it
        t0 = time.time()
        keys = [tuple(k.rsplit("|", 1)) for k in chains.filament_key]
        pairs = None
        if a.counterparts:
            from helicon.webApps.lib import helical_pitch_phase as phase

            with mrcfile.open(a.mrcs, permissive=True) as f:
                images = np.asarray(f.data)
            found = phase.pair_counterparts(list(images), classes)
            pairs = [
                dict(a=d["a"] + 1, b=d["b"] + 1, corr=d["corr"], dx=d["dx"])
                for d in found
            ]
            print(f"{len(pairs)} pairs of classes are counterparts", flush=True)
        labels, info = hc.context_groups(
            ref["rlnMicrographName"],
            ref["rlnHelicalTubeID"],
            ref["rlnHelicalTrackLengthAngst"],
            ref["rlnClassNumber"],
            pd.read_parquet(a.pmax)["pmax"].to_numpy() if a.pmax else None,
            [c + 1 for c in classes],
            keys,
            [int(x) for x in a.types.split(",")],
            max_fit_segments=a.app_fit_segments,
            n_iter=a.iter,
            cpu=a.cpu or None,
            counterparts=pairs,
            select=False,
        )
        print(
            f"\nFilament context as the app does it ({len(classes)} classes, fit on {a.app_fit_segments:,} segments; {time.time() - t0:.0f} s in all)"
        )
        for n, g in labels.items():
            report(f"app context, {n} types", g, truth, sets)

    if a.cv_max_segments:
        t0 = time.time()
        best, scores = hc.select_n_types(
            chains,
            3,
            max(int(x) for x in a.types.split(",")),
            max_segments=a.cv_max_segments,
            cpu=a.cpu or None,
        )
        print(
            f"\nheld-out log-likelihood per segment: "
            + " ".join(f"{k}: {v:.4f}" for k, v in scores.items())
            + f" -> {best} types ({time.time() - t0:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
