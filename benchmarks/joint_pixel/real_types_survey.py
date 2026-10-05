"""Types from the class averages (no labels): do the groups' own rings come out clean?

The classes of a Class2D job are grouped by the registered correlation of their
averages (average linkage, cut at ``--cut``); every group is then run through the
AbInitio3D analysis (counterpart merge, ring, class fits, class-average check) on
its own and compared with the pooled run. If the groups are types, each ring should be
coherent where the pooled one is not: a label-free check.

    python real_types_survey.py --job-dir <...> --iteration 20 --name 10943_mixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402

from helicon.webApps.lib import helical_pitch_pixel as px  # noqa: E402
from helicon.webApps.lib import helical_pitch_types as ty  # noqa: E402


def run_group(params, images, apix, numbers):
    result = R.merged_classes(params, images, apix, numbers)
    cls = [int(c) for c in result.class_ids]
    row = dict(
        classes_in=len(numbers),
        classes=len(cls),
        filaments=result.n_filaments,
        repeat=result.period,
        sd=result.period_sd,
        score=float(result.scan_scores.max()),
        support=result.support,
        double_ratio=result.double_ratio,
        median_fit=float(np.nanmedian(result.class_fit)),
        well_fitting=int(np.sum(result.class_fit >= 0.5)),
    )
    try:
        check = px.class_pixel_check(
            [images[c - 1] for c in cls],
            result.class_zdir,
            result.phases,
            result.period,
            apix,
        )
    except Exception:
        check = None
    if check is not None:
        row.update(
            pixel_disagree=int(check.disagree().sum()),
            pixel_median=float(np.nanmedian(check.residual)),
        )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=20)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cut", type=float, default=0.15)
    ap.add_argument(
        "--beta", type=float, default=0.0, help="weight of the neighbour frequency"
    )
    ap.add_argument(
        "--saved",
        action="store_true",
        help="use the similarity saved by an earlier run",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    n = len(params)
    counts = params["rlnClassNumber"].astype(int).value_counts()
    floor = max(100, int(0.002 * n))
    classes = sorted(
        int(c)
        for c in counts.index
        if counts[c] >= floor and images[c - 1].max() > images[c - 1].min()
    )
    print(
        f"[{a.name}] {n:,} segments, {len(classes)} classes with >= {floor} segments",
        flush=True,
    )
    if a.saved:
        S = np.load(R.OUT / f"types_{a.name}_S.npy")
        assert S.shape[0] == len(classes)
    else:
        S = ty.class_similarity(images, [c - 1 for c in classes], apix)
        np.save(R.OUT / f"types_{a.name}_S.npy", S)
    if a.beta > 0:
        lift = ty.neighbour_lift(params, classes, 800.0)
        S = ty.neighbour_affinity(S, lift, a.beta)
        print(f"  neighbour frequency weight {a.beta}", flush=True)
    off = S[np.triu_indices(len(S), 1)]
    print(
        f"  registered correlation of class pairs: quantiles 5/25/50/75/95% = {np.percentile(off, [5, 25, 50, 75, 95]).round(2).tolist()}",
        flush=True,
    )
    print(
        f"  suggested cut (largest gap in the tree): {ty.suggest_cut(S):.3f}",
        flush=True,
    )
    for cut in (0.15, 0.20, 0.25, 0.30, 0.35):
        g = ty.pixel_groups(S, cut)
        sizes = [
            (
                int((g == k).sum()),
                int(sum(counts[classes[i]] for i in np.flatnonzero(g == k))),
            )
            for k in range(g.max() + 1)
        ]
        print(
            f"  cut {cut:.2f}: groups (classes, segments): {sizes}; unassigned classes {int((g < 0).sum())}",
            flush=True,
        )
    g = ty.pixel_groups(S, a.cut)
    U, keys = ty.usage_matrix(params, classes)
    ng = g.max() + 1
    lab, share = ty.assign_filaments(U, g, ng, 0.6)
    print(
        f"  cut {a.cut}: {ng} groups; filaments called {np.mean(lab >= 0):.2f}: "
        + ", ".join(f"group {k}: {int((lab == k).sum())}" for k in range(ng)),
        flush=True,
    )
    rows = []
    for k in range(ng):
        numbers = [classes[i] for i in np.flatnonzero(g == k)]
        try:
            row = run_group(params, images, apix, numbers)
        except Exception as e:
            print(f"  group {k}: analysis failed ({type(e).__name__}: {e})", flush=True)
            continue
        row.update(name=a.name, group=k, segments=int(sum(counts[c] for c in numbers)))
        rows.append(row)
        print(
            f"  group {k} ({len(numbers)} classes, {row['segments']:,} segments, {row['filaments']} filaments): "
            f"repeat {row['repeat']:.1f} +- {row['sd']:.1f} A, score {row['score']:.3f}, support {row['support']:.2f}, "
            f"median fit {row['median_fit']:.2f}, {row['well_fitting']} of {row['classes']} classes >= 0.5"
            + (
                f"; pixels: {row['pixel_disagree']} disagree, median {row['pixel_median']:.0f} A"
                if "pixel_disagree" in row
                else ""
            ),
            flush=True,
        )
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
