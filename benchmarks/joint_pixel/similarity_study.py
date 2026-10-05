"""How the class-average similarity depends on the overlap, and what to correlate.

Two class averages cover different stretches of the repeat; registering one onto the
other along the axis leaves them overlapping over only part of their width, and the
correlation over a short overlap can be high by chance. EMPIAR-10230, with the
manually labelled PHF and SF classes (class SF share below 0.1 or above 0.5).

    python similarity_study.py --job-dir <.../job684> --iteration 25
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402

from helicon.webApps.lib import denovo3d_register as DR  # noqa: E402
from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402


def auc(same, diff):
    same, diff = np.asarray(same), np.asarray(diff)
    if len(same) == 0 or len(diff) == 0:
        return float("nan")
    # Mann-Whitney U as a probability
    ranks = pd.Series(np.concatenate([same, diff])).rank().values
    u = ranks[: len(same)].sum() - len(same) * (len(same) + 1) / 2
    return float(u / (len(same) * len(diff)))


def register(prepared, classes, variant):
    imgs = {}
    for c in classes:
        a = np.asarray(prepared[c], dtype=np.float32)
        if variant == "modulation":
            a = a - a.mean(axis=1, keepdims=True)  # the part that varies along the axis
        imgs[c] = a
    todo = [(i, j) for i, j in combinations(classes, 2)]
    res = ph._register_all(imgs, todo, DR.FLIPS, None, refine=False)
    return todo, res


def profile_similarity(prepared, classes):
    """Correlation of the profiles across the filament (the axial mean), with a
    shift of a few pixels: no registration along the axis, so no overlap."""
    prof = {}
    for c in classes:
        p = np.asarray(prepared[c], dtype=float).mean(axis=1)
        p = p - np.median(p)
        prof[c] = p / max(np.linalg.norm(p), 1e-12)
    out = {}
    for i, j in combinations(classes, 2):
        best = -1.0
        for s in range(-4, 5):
            a, b = prof[i], np.roll(prof[j], s)
            best = max(best, float(a @ b))
        out[(i, j)] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=25)
    ap.add_argument("--min-segments", type=int, default=100)
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
    print(f"{len(use)} classes of a clear type ({sum(typ.values())} SF)", flush=True)
    idx = [c - 1 for c in use]
    prepared, factor = ph._register_prepared(images, idx, 128)
    W = next(iter(prepared.values())).shape[1]
    print(f"prepared images {next(iter(prepared.values())).shape}", flush=True)

    rows = []
    for variant in ("plain", "modulation"):
        todo, res = register(prepared, idx, variant)
        for (i, j), r in zip(todo, res):
            ci, cj = i + 1, j + 1
            overlap = max(0.0, (W - abs(r["dx"])) / W)
            rows.append(
                dict(
                    variant=variant,
                    a=ci,
                    b=cj,
                    same=int(typ[ci] == typ[cj]),
                    both_sf=int(typ[ci] == typ[cj] == 1),
                    corr=float(r["corr"]),
                    overlap=overlap,
                )
            )
    prof = profile_similarity(prepared, idx)
    for (i, j), v in prof.items():
        ci, cj = i + 1, j + 1
        rows.append(
            dict(
                variant="profile",
                a=ci,
                b=cj,
                same=int(typ[ci] == typ[cj]),
                both_sf=int(typ[ci] == typ[cj] == 1),
                corr=v,
                overlap=1.0,
            )
        )
    d = pd.DataFrame(rows)
    d.to_csv(R.OUT / "similarity_study.csv", index=False)
    print("\nAUC of the similarity for same-type against different-type pairs:")
    for v, g in d.groupby("variant"):
        print(
            f"  {v:10s}: {auc(g[g.same == 1].corr, g[g.same == 0].corr):.3f}   same-type mean {g[g.same == 1].corr.mean():.3f}, different {g[g.same == 0].corr.mean():.3f}"
        )
    print("\nplain similarity by the overlap of the registration (share of the width):")
    g = d[d.variant == "plain"].copy()
    g["bin"] = pd.cut(g.overlap, [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
    for b, h in g.groupby("bin", observed=True):
        s, df_ = h[h.same == 1].corr, h[h.same == 0].corr
        print(
            f"  overlap {str(b):12s}: n same {len(s):3d} mean {s.mean() if len(s) else float('nan'):.3f}; n different {len(df_):3d} mean {df_.mean() if len(df_) else float('nan'):.3f}; AUC {auc(s, df_):.3f}"
        )
    print("\nmodulation similarity by overlap:")
    g = d[d.variant == "modulation"].copy()
    g["bin"] = pd.cut(g.overlap, [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
    for b, h in g.groupby("bin", observed=True):
        s, df_ = h[h.same == 1].corr, h[h.same == 0].corr
        print(
            f"  overlap {str(b):12s}: n same {len(s):3d} mean {s.mean() if len(s) else float('nan'):.3f}; n different {len(df_):3d} mean {df_.mean() if len(df_) else float('nan'):.3f}; AUC {auc(s, df_):.3f}"
        )


if __name__ == "__main__":
    main()
