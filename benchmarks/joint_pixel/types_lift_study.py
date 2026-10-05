"""Does adding how often two classes are neighbours to the correlation of their
averages improve the grouping into types?  EMPIAR-10230 against the labels.

    python types_lift_study.py --job-dir <.../job684> --iteration 25
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
    ap.add_argument("--window", type=float, default=800.0)
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    sf = params["rlnImageName"].str.contains("Extract/job671")
    counts = params["rlnClassNumber"].astype(int).value_counts()
    floor = max(100, int(0.002 * len(params)))
    classes = sorted(
        int(c)
        for c in counts.index
        if counts[c] >= floor and images[c - 1].max() > images[c - 1].min()
    )
    S = np.load(R.OUT / "types10230_S.npy")
    saved = pd.read_csv(R.OUT / "types10230_classes.csv").cls.astype(int).tolist()
    assert S.shape[0] == len(saved)
    classes = saved  # the classes the saved similarity was computed for
    lift = ty.neighbour_lift(params, classes, a.window)
    cshare = np.array(
        [sf[params["rlnClassNumber"].astype(int) == c].mean() for c in classes]
    )
    cweight = np.array([counts[c] for c in classes], dtype=float)
    U, keys = ty.usage_matrix(params, classes)
    fsf = sf.groupby(
        [params["rlnMicrographName"].values, params["rlnHelicalTubeID"].values]
    ).mean()
    fshare = np.array([fsf.get(k, np.nan) for k in keys])
    pure = (fshare <= 0.1) | (fshare >= 0.9)
    truth = (fshare >= 0.9).astype(int)
    print(
        f"{len(classes)} classes; window {a.window:.0f} A; lift quantiles 5/50/95%: {np.percentile(lift[np.triu_indices(len(lift), 1)], [5, 50, 95]).round(2).tolist()}"
    )
    print(
        f"best possible class calls (majority type per class): {np.sum(np.maximum(cshare, 1 - cshare) * cweight) / cweight.sum():.3f}\n"
    )
    print(
        "beta  cut   groups (classes)        class calls  filaments called  accuracy (PHF / SF)"
    )
    for beta in (0.0, 0.15, 0.3, 0.5):
        A = ty.neighbour_affinity(S, lift, beta)
        for cut in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            g = ty.pixel_groups(A, cut)
            ng = int(g.max()) + 1
            if ng < 1:
                print(f"{beta:4.2f}  {cut:.2f}   none")
                continue
            gsf = np.array(
                [
                    np.average(cshare[g == k], weights=cweight[g == k]) > 0.5
                    for k in range(ng)
                ]
            )
            pred_sf = np.where(g >= 0, gsf[np.maximum(g, 0)], False)
            class_acc = float(
                np.sum(np.where(pred_sf, cshare, 1 - cshare) * cweight) / cweight.sum()
            )
            lab, share = ty.assign_filaments(U, g, ng, 0.6)
            called = lab >= 0
            pred = np.where(called, gsf[np.maximum(lab, 0)].astype(int), -1)
            ok = pure & called
            tpr = [
                (
                    np.mean(pred[pure & called & (truth == t)] == t)
                    if (pure & called & (truth == t)).any()
                    else np.nan
                )
                for t in (0, 1)
            ]
            sizes = ",".join(str(int((g == k).sum())) for k in range(ng))
            print(
                f"{beta:4.2f}  {cut:.2f}   {sizes:22s}  {class_acc:.3f}        {called[pure].mean():.2f}              {np.mean(pred[ok] == truth[ok]):.3f} ({tpr[0]:.2f} / {tpr[1]:.2f})"
            )


if __name__ == "__main__":
    main()
