"""Do the true PHF and SF rings agree with the class averages better than the
partition the unlabeled fit found?  (EMPIAR-10230, labels used only to score.)

    python real_10230_pixel_rings.py --job-dir <...> --iteration 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10230_truth as T  # noqa: E402
import real_10940 as R  # noqa: E402

from helicon.webApps.lib import helical_pitch_mixture as mx  # noqa: E402
from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402
from helicon.webApps.lib import helical_pitch_pixel as px  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=25)
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    numbers = R.selections(params, images)["all"]
    result = R.merged_classes(params, images, apix, numbers)
    cls = [int(c) for c in result.class_ids]
    used = result.params_used
    pairs = ph.prepare_pairs(used, class_ids=cls, max_sep=R.MAX_SEP)
    est = ph.estimate_period(pairs)
    est, _ = ph.sync_filament_directions(pairs, est)
    share = T.type_of_filaments(used, pairs.filament_keys)
    pure_phf, pure_sf = share <= 0.1, share >= 0.9
    oracle_phf, oracle_sf = T.oracle(pairs, pure_phf), T.oracle(pairs, pure_sf)

    rng = np.random.default_rng(0)
    best = None
    for _ in range(6):
        two = mx.fit_two_rings(
            pairs,
            rng.integers(0, 2, pairs.n_filaments),
            (est["period"] * 0.93, est["period"] * 1.07),
        )
        if two is not None and (best is None or two["objective"] > best["objective"]):
            best = two
    rings = {
        "pooled (one ring)": (est["period"], est["phases"]),
        "true PHF ring": (oracle_phf["period"], oracle_phf["phases"]),
        "true SF ring": (oracle_sf["period"], oracle_sf["phases"]),
        f"unlabeled ring {best['periods'][0]:.0f} A": (
            best["periods"][0],
            best["phases"][0],
        ),
        f"unlabeled ring {best['periods'][1]:.0f} A": (
            best["periods"][1],
            best["phases"][1],
        ),
    }
    for name, (period, phases) in rings.items():
        check = px.class_pixel_check(
            [images[c - 1] for c in cls], result.class_zdir, phases, period, apix
        )
        r = check.residual
        agree = check.agree()
        good = result.class_fit >= 0.5
        print(
            f"{name:28s} repeat {period:6.1f} A: class-average residual median "
            f"{np.nanmedian(r):5.0f} A, {int(check.disagree().sum()):2d} of {len(r)} "
            f"classes disagree; mean residual over the {int(good.sum())} well-fitting "
            f"classes {np.nanmean(r[good]):5.0f} A",
            flush=True,
        )


if __name__ == "__main__":
    main()
