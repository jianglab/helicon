"""What the filament neighbours of a class pair add to the correlation of its averages.

For two classes a and b the segments of a and of b that are near each other on the
same filament give (1) how often they are neighbours (against what the sizes of the
classes lead to expect) and (2) the distances between them, d = position(b) -
position(a), which for two views of one helix must equal, modulo the repeat, the
axial offset that registers the average of b on that of a. EMPIAR-10230, with the
manually labelled PHF and SF classes (labels only score).

    python neighbour_study.py --job-dir <.../job684> --iteration 25
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402

from helicon.webApps.lib import denovo3d_register as DR  # noqa: E402
from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402


def auc(same, diff):
    same, diff = np.asarray(same, float), np.asarray(diff, float)
    ok_s, ok_d = np.isfinite(same), np.isfinite(diff)
    same, diff = same[ok_s], diff[ok_d]
    if len(same) == 0 or len(diff) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([same, diff])).rank().values
    return float(
        (r[: len(same)].sum() - len(same) * (len(same) + 1) / 2)
        / (len(same) * len(diff))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=25)
    ap.add_argument("--min-segments", type=int, default=100)
    ap.add_argument("--max-sep", type=float, default=1500.0)
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    sf = params["rlnImageName"].str.contains("Extract/job671")
    counts = params["rlnClassNumber"].astype(int).value_counts()
    classes = sorted(
        int(c)
        for c in counts.index
        if counts[c] >= a.min_segments and images[c - 1].max() > images[c - 1].min()
    )
    share = {c: sf[params["rlnClassNumber"].astype(int) == c].mean() for c in classes}
    use = [c for c in classes if share[c] < 0.1 or share[c] > 0.5]
    typ = {c: int(share[c] > 0.5) for c in use}
    print(f"{len(use)} classes of a clear type", flush=True)

    pairs = ph.prepare_pairs(
        params, class_ids=use, max_sep=a.max_sep, use_refined_path=False
    )
    cls_of = {int(c): i for i, c in enumerate(pairs.class_ids)}
    K = len(pairs.class_ids)
    ca, cb = pairs.seg_class[pairs.I], pairs.seg_class[pairs.J]
    w = pairs.W
    # neighbour counts within a window, and the pooled repeat for the offset
    est = ph.estimate_period(pairs)
    P = est["period"]
    print(f"{len(pairs.D):,} same-polarity pairs; pooled repeat {P:.0f} A", flush=True)
    n_seg = np.bincount(pairs.seg_class, minlength=K).astype(float)

    rows = []
    # registration (plain) for the same classes
    t0 = time.time()
    prepared, factor = ph._register_prepared(images, [c - 1 for c in use], 128)
    todo = [(i - 1, j - 1) for i, j in combinations(use, 2)]
    res = ph._register_all(prepared, todo, DR.FLIPS, None, refine=False)
    print(f"registered {len(todo)} pairs in {time.time() - t0:.0f} s", flush=True)
    reg = {(i + 1, j + 1): r for (i, j), r in zip(todo, res)}
    for window in (300.0, 800.0, a.max_sep):
        near = (np.abs(pairs.D) <= window) * w
        N = np.zeros((K, K))
        np.add.at(N, (ca, cb), near)
        N = N + N.T
        tot = N.sum()
        exp = np.outer(N.sum(1), N.sum(1)) / max(tot, 1e-12)
        for i, j in combinations(use, 2):
            ii, jj = cls_of[i], cls_of[j]
            rows.append(
                dict(
                    window=window,
                    a=i,
                    b=j,
                    same=int(typ[i] == typ[j]),
                    lift=float(N[ii, jj] / max(exp[ii, jj], 1e-12)),
                    n=float(N[ii, jj]),
                )
            )
    d = pd.DataFrame(rows)
    pix = pd.DataFrame(
        [
            dict(
                a=i,
                b=j,
                corr=float(r["corr"]),
                dx_pix=float(r["dx"]) * factor[i - 1] * apix,
                flip=(r["flip_x"], r["flip_y"]),
            )
            for (i, j), r in reg.items()
        ]
    )
    print(
        "\n=== neighbour frequency (lift = observed / expected from the class sizes) ==="
    )
    for window, g in d.groupby("window"):
        s, x = g[g.same == 1], g[g.same == 0]
        print(
            f"window {window:5.0f} A: same-type lift median {s.lift.median():.2f} (zero in {np.mean(s.n == 0):.2f}); different-type lift median {x.lift.median():.2f} (zero in {np.mean(x.n == 0):.2f}); AUC of lift {auc(s.lift, x.lift):.3f}"
        )
    full = d[d.window == a.max_sep].merge(pix, on=["a", "b"])
    print(
        f"pixel correlation AUC {auc(full[full.same == 1]['corr'], full[full.same == 0]['corr']):.3f}"
    )
    # combination: rank average of the pixel correlation and the log lift
    lift = np.log(full.lift.clip(lower=1e-3))
    z = lambda v: (v - v.mean()) / v.std()
    comb = z(full["corr"]) + z(lift)
    print(
        f"combined (z-scored correlation + log lift) AUC {auc(comb[full.same == 1], comb[full.same == 0]):.3f}"
    )

    # distances against the registered offset
    print(
        "\n=== the distances between the two classes' segments against the registered offset ==="
    )
    out = []
    for i, j in combinations(use, 2):
        ii, jj = cls_of[i], cls_of[j]
        m1 = (ca == ii) & (cb == jj)
        m2 = (ca == jj) & (cb == ii)
        dd = np.concatenate([pairs.D[m1], -pairs.D[m2]])  # position(b) - position(a)
        ww = np.concatenate([w[m1], w[m2]])
        if len(dd) < 20:
            out.append(
                dict(
                    a=i,
                    b=j,
                    same=int(typ[i] == typ[j]),
                    n=len(dd),
                    R=np.nan,
                    pred=np.nan,
                )
            )
            continue
        z_ = (ww * np.exp(2j * np.pi * dd / P)).sum() / ww.sum()
        out.append(
            dict(
                a=i,
                b=j,
                same=int(typ[i] == typ[j]),
                n=len(dd),
                R=float(abs(z_)),
                pred=float(np.angle(z_) * P / (2 * np.pi)),
            )
        )
    o = pd.DataFrame(out).merge(pix, on=["a", "b"])
    o["wrapped"] = ((o.dx_pix - o.pred) + P / 2) % P - P / 2
    o["wrapped_neg"] = ((-o.dx_pix - o.pred) + P / 2) % P - P / 2
    s = o[(o.same == 1) & o.R.notna()]
    x = o[(o.same == 0) & o.R.notna()]
    print(
        f"pairs with at least 20 neighbouring segment pairs: same-type {len(s)} of {int((o.same == 1).sum())}, different-type {len(x)} of {int((o.same == 0).sum())}"
    )
    print(
        f"concentration of the distances mod the repeat (R): same-type median {s.R.median():.2f}, different-type median {x.R.median():.2f}, AUC {auc(s.R, x.R):.3f}"
    )
    for lab, col in (
        ("registered offset", "wrapped"),
        ("minus the registered offset", "wrapped_neg"),
    ):
        v = s[col].abs()
        print(
            f"same-type pairs, {lab} against the distances: median |difference| {v.median():.0f} A; within 20 A: {np.mean(v < 20):.2f}; within 40 A: {np.mean(v < 40):.2f}"
        )
    strong = s[s.R > 0.5]
    for lab, col in (("registered offset", "wrapped"), ("minus", "wrapped_neg")):
        v = strong[col].abs()
        print(
            f"  of those with R > 0.5 (n={len(strong)}): {lab}: median {v.median():.0f} A; within 20 A: {np.mean(v < 20):.2f}"
        )
    lowov = s[(s["corr"] < 0.9)]
    print(
        f"same-type pairs with a registered correlation under 0.9 (n={len(lowov)}): median |difference| {lowov.wrapped.abs().median():.0f} A"
    )
    o.to_csv(R.OUT / "neighbour_study.csv", index=False)
    d.to_csv(R.OUT / "neighbour_lift.csv", index=False)


if __name__ == "__main__":
    main()
