"""Are the two rings of a real data set real?

1. Split half: the two-ring fit on two random halves of the filaments, fitted
   independently. If the rings are real, both halves find the same repeats, and
   the rings of one half give the filaments of the other the same rings as that
   half's own fit.
2. A null from the data's own geometry: class labels of ONE perfect ring laid on the
   data set's own segment positions and filament lengths (noise tuned to the real
   one-ring objective), the two-ring fit run on them, and the gain compared with
   the real gain.

    python real_two_rings_validation.py --job-dir <...> --iteration 25 --out v.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402

from helicon.webApps.lib import helical_pitch_mixture as mx  # noqa: E402
from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402

_PAIRS = []


def best_two(pairs, est_period, rng, restarts):
    best = None
    for _ in range(restarts):
        lab = rng.integers(0, 2, pairs.n_filaments)
        two = mx.fit_two_rings(pairs, lab, (est_period * 0.93, est_period * 1.07))
        if two is not None and (best is None or two["objective"] > best["objective"]):
            best = two
    return best


def one_ring(pairs):
    est = ph.estimate_period(pairs)
    est, _ = ph.sync_filament_directions(pairs, est)
    return est, mx.one_ring_objective(pairs, est["period"], est["phases"])


def null_replicate(args):
    seed, noise, period = args
    pairs = _PAIRS[0]
    lab = ph.synthetic_labels(pairs, period, pairs.n_classes, noise, seed)
    fake = dataclasses.replace(pairs, seg_class=lab)
    est, one = one_ring(fake)
    best = best_two(fake, est["period"], np.random.default_rng(seed), 3)
    lo, hi = np.sort(best["periods"])
    return dict(
        seed=seed,
        noise=noise,
        one=one,
        two=best["objective"],
        gain=best["objective"] - one,
        sep=hi / lo - 1,
        smaller=min(best["n_filaments"]) / fake.n_filaments,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=20)
    ap.add_argument("--rule", default="all")
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--nulls", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    params, images, apix = R.load(a.job_dir, a.iteration)
    numbers = R.selections(params, images)[a.rule]
    result = R.merged_classes(params, images, apix, numbers)
    cls = [int(c) for c in result.class_ids]
    pairs = ph.prepare_pairs(result.params_used, class_ids=cls, max_sep=R.MAX_SEP)
    est, one = one_ring(pairs)
    rng = np.random.default_rng(0)
    full = best_two(pairs, est["period"], rng, 4)
    lo, hi = np.sort(full["periods"])
    gain = full["objective"] - one
    order = np.argsort(full["periods"])
    print(
        f"one ring {est['period']:.1f} A (objective {one:.4f}); two rings "
        f"{lo:.1f} / {hi:.1f} A, filaments {full['n_filaments'][order].tolist()}, "
        f"gain {gain:+.4f}",
        flush=True,
    )

    # 1. split half
    print("\nsplit half (filaments)", flush=True)
    n = pairs.n_filaments
    pf = pairs.pair_filament
    for s in range(a.splits):
        r = np.random.default_rng(100 + s)
        half = r.random(n) < 0.5
        fits = []
        for part in (half, ~half):
            sub = dataclasses.replace(pairs, W=pairs.W * part[pf])
            e, o = one_ring(sub)
            fits.append((sub, best_two(sub, e["period"], r, 3)))
        (subA, fa), (subB, fb) = fits
        pa, pb = np.sort(fa["periods"]), np.sort(fb["periods"])
        # the rings of half A, applied to every filament
        scoreA = []
        for t in (0, 1):
            keep, swap, w = mx.filament_scores(pairs, fa["periods"][t], fa["phases"][t])
            scoreA.append(np.maximum(keep, swap))
        labelA = np.argmax(np.vstack(scoreA), axis=0)
        ia, ib = np.argsort(fa["periods"]), np.argsort(fb["periods"])
        remap = {int(ia[0]): 0, int(ia[1]): 1}
        la = np.array([remap[int(x)] for x in labelA])
        mapb = {int(ib[0]): 0, int(ib[1]): 1}
        lb = np.array([mapb[int(x)] for x in fb["labels"]])
        # agree on half B's filaments, between half A's rings and half B's own
        agree = float(np.mean(la[~half] == lb[~half]))
        print(
            f"  split {s}: half A {pa[0]:.1f} / {pa[1]:.1f} A, half B "
            f"{pb[0]:.1f} / {pb[1]:.1f} A; held-out filaments given the same ring "
            f"by A's rings and B's own: {agree:.2f}",
            flush=True,
        )

    # 2. the null on the data's own geometry
    print("\nnull: one perfect ring on this data set's own geometry", flush=True)
    _PAIRS.clear()
    _PAIRS.append(pairs)
    chosen = None
    for noise in (0.2, 0.35, 0.5, 0.65, 0.8):
        lab = ph.synthetic_labels(pairs, est["period"], pairs.n_classes, noise, 1)
        fake = dataclasses.replace(pairs, seg_class=lab)
        e, o = one_ring(fake)
        print(
            f"  noise {noise:.2f}: one-ring objective {o:.4f} (real {one:.4f})",
            flush=True,
        )
        if chosen is None or abs(o - one) < abs(chosen[1] - one):
            chosen = (noise, o)
    noise = chosen[0]
    with mp.get_context("fork").Pool(6) as pool:
        rows = pool.map(
            null_replicate, [(i, noise, est["period"]) for i in range(a.nulls)]
        )
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    p = (1 + int((d.gain >= gain).sum())) / (1 + len(d))
    print(
        f"  noise {noise:.2f}; {len(d)} null replicates: gain median {d.gain.median():.4f}, "
        f"max {d.gain.max():.4f}; separation median {d['sep'].median():.3f}, max "
        f"{d['sep'].max():.3f}\n  real gain {gain:+.4f} -> p = {p:.3f} "
        f"(share of nulls at least as large)",
        flush=True,
    )


if __name__ == "__main__":
    main()
