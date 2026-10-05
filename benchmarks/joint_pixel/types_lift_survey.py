"""Group sizes of the classes of a job with and without the neighbour frequency.

Uses the similarity saved by real_types_survey.py (plain correlation).

    python types_lift_survey.py --job-dir <...> --iteration 20 --name 10943_mixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402

from helicon.webApps.lib import helical_pitch_types as ty  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=20)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    counts = params["rlnClassNumber"].astype(int).value_counts()
    floor = max(100, int(0.002 * len(params)))
    classes = sorted(
        int(c)
        for c in counts.index
        if counts[c] >= floor and images[c - 1].max() > images[c - 1].min()
    )
    S = np.load(R.OUT / f"types_{a.name}_S.npy")
    assert S.shape[0] == len(classes), (S.shape, len(classes))
    lift = ty.neighbour_lift(params, classes, 800.0)
    w = np.array([counts[c] for c in classes], dtype=float)
    U, keys = ty.usage_matrix(params, classes)
    print(
        f"[{a.name}] {len(classes)} classes; lift quantiles 5/50/95%: {np.percentile(lift[np.triu_indices(len(lift), 1)], [5, 50, 95]).round(2).tolist()}"
    )
    for beta in (0.0, 0.3):
        for cut in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            g = ty.pixel_groups(ty.neighbour_affinity(S, lift, beta), cut)
            ng = int(g.max()) + 1
            sizes = [(int((g == k).sum()), int(w[g == k].sum())) for k in range(ng)]
            lab, _ = (
                ty.assign_filaments(U, g, max(ng, 1), 0.6)
                if ng
                else (np.array([]), None)
            )
            called = float(np.mean(lab >= 0)) if ng else 0.0
            print(
                f"  beta {beta:.1f} cut {cut:.2f}: {ng} groups (classes, segments) {sizes[:6]}{' ...' if ng > 6 else ''}; unassigned {int((g < 0).sum())}; filaments called {called:.2f}"
            )


if __name__ == "__main__":
    main()
