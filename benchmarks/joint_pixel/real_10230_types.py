"""Types from the class averages and the filaments' class usage, against the labels.

EMPIAR-10230 (AD tau; PHF and SF filaments labelled manually). The labels score the
result; nothing else uses them.

    python real_10230_types.py --job-dir <.../job684> --iteration 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402

from helicon.webApps.lib import helical_pitch_types as ty  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=25)
    ap.add_argument("--min-segments", type=int, default=100)
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    sf = params["rlnImageName"].str.contains("Extract/job671")
    counts_all = params["rlnClassNumber"].astype(int).value_counts()
    classes = [
        int(c)
        for c in counts_all.index
        if counts_all[c] >= a.min_segments and images[c - 1].max() > images[c - 1].min()
    ]
    classes.sort()
    print(f"{len(classes)} classes with >= {a.min_segments} segments", flush=True)
    S = ty.class_similarity(
        [images[c - 1] for c in range(1, len(images) + 1)],
        [c - 1 for c in classes],
        apix,
    )
    U, keys = ty.usage_matrix(params, classes)
    Uc = ty.usage_similarity(U)
    # truth: SF share per class and per filament
    cshare = np.array(
        [sf[params["rlnClassNumber"].astype(int) == c].mean() for c in classes]
    )
    cweight = np.array([counts_all[c] for c in classes], dtype=float)
    key_share = sf.groupby(
        [params["rlnMicrographName"].values, params["rlnHelicalTubeID"].values]
    ).mean()
    fshare = np.array([key_share.get(k, np.nan) for k in keys])
    pure = (fshare <= 0.1) | (fshare >= 0.9)
    ftruth = (fshare >= 0.9).astype(int)
    print(
        f"{len(keys)} filaments, {int(pure.sum())} pure ({int((fshare >= 0.9).sum())} SF)",
        flush=True,
    )

    def score(name, A, n_groups=2):
        g = ty.spectral_groups(A, n_groups)
        # class level: segment-weighted accuracy calling each group by its majority type
        pred_sf = np.zeros(len(classes), bool)
        for k in range(n_groups):
            m = g == k
            if m.any():
                pred_sf[m] = np.average(cshare[m], weights=cweight[m]) > 0.5
        class_acc = float(
            np.sum(np.where(pred_sf, cshare, 1 - cshare) * cweight) / cweight.sum()
        )
        # filaments
        lab, share = ty.assign_filaments(U, g, n_groups, 0.6)
        group_sf = np.array(
            [
                (
                    pred_sf[g == k].any()
                    and (np.average(cshare[g == k], weights=cweight[g == k]) > 0.5)
                    if (g == k).any()
                    else False
                )
                for k in range(n_groups)
            ]
        )
        called = lab >= 0
        pred = np.where(called, group_sf[np.maximum(lab, 0)].astype(int), -1)
        ok = pure & called
        acc = float(np.mean(pred[ok] == ftruth[ok])) if ok.any() else float("nan")
        # balanced accuracy over the two true types
        tpr = [
            (
                np.mean(pred[pure & called & (ftruth == t)] == t)
                if (pure & called & (ftruth == t)).any()
                else np.nan
            )
            for t in (0, 1)
        ]
        print(
            f"{name:28s} groups {np.bincount(g, minlength=n_groups).tolist()}; class calls "
            f"{class_acc:.3f}; filaments called {called[pure].mean():.2f}, accuracy {acc:.3f} "
            f"(PHF {tpr[0]:.2f}, SF {tpr[1]:.2f}); mean group SF share "
            + ", ".join(
                f"{np.average(cshare[g==k], weights=cweight[g==k]):.2f}"
                for k in range(n_groups)
                if (g == k).any()
            ),
            flush=True,
        )

    base = (
        max(cweight[cshare <= 0.5].sum(), cweight[cshare > 0.5].sum()) / cweight.sum()
    )
    print(
        f"baseline (everything the larger type): class {base:.3f}, filaments {max(ftruth[pure].mean(), 1 - ftruth[pure].mean()):.3f}"
    )
    for w in (1.0, 0.75, 0.5, 0.25, 0.0):
        score(
            {1.0: "pixels only", 0.0: "usage only"}.get(w, f"pixels {w:.2f} + usage"),
            ty.affinity(S, Uc, w),
        )
    np.save(R.OUT / "types10230_S.npy", S)
    pd.DataFrame(dict(cls=classes, sf_share=cshare, n=cweight)).to_csv(
        R.OUT / "types10230_classes.csv", index=False
    )


if __name__ == "__main__":
    main()
