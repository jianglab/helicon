"""The two-ring gain on synthetic groups of one type (the null) and of two.

One type: filaments of one repeat, with a spread of repeats from filament to
filament (``period_sd``) and a share of poorly assigned segments. Two types:
repeats 400 and 400 x ratio sharing their classes. For filaments of several
lengths. Shows how large the gain is when nothing is there to find.

    python two_ring_null.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_pixel import synthetic as syn  # noqa: E402

from two_ring_test import two_ring_gain  # noqa: E402

K, N_FIL = 12, 300


def main():
    print(
        "length  period_sd  mislabel | one type: gain, held-out gain | two types (ratio 1.10, 1.20): gain/held-out"
    )
    for length in (400.0, 800.0, 1600.0):
        for sd in (0.0, 0.03):
            for mis in (0.0, 0.3):
                one, mix = [], []
                for seed in range(3):
                    t = syn.class2d_table(
                        400.0,
                        K,
                        n_fil=N_FIL,
                        length=length,
                        period_sd=sd,
                        mislabel=mis,
                        seed=seed,
                    )
                    r = two_ring_gain(t, list(range(1, K + 1)), restarts=3)
                    if r:
                        lo, hi = sorted(r["repeats"])
                        one.append((r["gain"], hi / lo, r["held"]))
                two = []
                for ratio in (1.10, 1.20):
                    t, _ = syn.class2d_table_mixture(
                        (400.0, 400.0 * ratio),
                        n_classes=K,
                        n_fil=N_FIL,
                        length=length,
                        mislabel=mis,
                        seed=1,
                    )
                    r = two_ring_gain(t, list(range(1, K + 1)), restarts=3)
                    two.append(None if r is None else (r["gain"], r["held"]))
                g = np.array([x[0] for x in one])
                rr = np.array([x[1] for x in one])
                hh = np.array([x[2] for x in one])
                print(
                    f"{length:6.0f}  {sd:9.2f}  {mis:8.1f} | {g.mean():+.3f}, held-out {hh.mean():+.3f} (max {hh.max():+.3f}) | "
                    + ", ".join(
                        "-" if x is None else f"{x[0]:+.3f}/{x[1]:+.3f}" for x in two
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
