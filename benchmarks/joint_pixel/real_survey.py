"""One Class2D job through the whole analysis: the tab's pitch estimate, the class-average
check, and the two-ring fit. No truth is used; for a first look at a data set.

    python real_survey.py --job-dir <...> --iteration 20 --name 10943_A1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402

from helicon.webApps.lib import helical_pitch_mixture as mx  # noqa: E402
from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402
from helicon.webApps.lib import helical_pitch_pixel as px  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=20)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rules", default="top20,all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    params, images, apix = R.load(a.job_dir, a.iteration)
    fil = params.groupby(["rlnMicrographName", "rlnHelicalTubeID"]).ngroups
    print(
        f"[{a.name}] {len(params):,} segments, {fil:,} filaments, {len(images)} "
        f"classes, {images[0].shape[1]} px, {apix} A/pixel",
        flush=True,
    )
    sels = R.selections(params, images)
    rows = []
    for rule in a.rules.split(","):
        numbers = sels[rule]
        try:
            result = R.merged_classes(params, images, apix, numbers)
        except Exception as e:
            print(f"  [{rule}] analysis failed: {type(e).__name__}: {e}", flush=True)
            continue
        cls = [int(c) for c in result.class_ids]
        row = dict(
            name=a.name,
            rule=rule,
            segments=len(params),
            filaments=result.n_filaments,
            classes_in=len(numbers),
            classes=len(cls),
            repeat=result.period,
            sd=result.period_sd,
            score=float(result.scan_scores.max()),
            support=result.support,
            double_ratio=result.double_ratio,
            groups=len(result.groups),
            median_fit=float(np.nanmedian(result.class_fit)),
            well_fitting=int(np.sum(result.class_fit >= 0.5)),
            span_median=float(np.median(result.filaments["span"])),
        )
        try:
            check = px.class_pixel_check(
                [images[c - 1] for c in cls],
                result.class_zdir,
                result.phases,
                result.period,
                apix,
            )
        except Exception as e:
            check = None
            print(f"  pixel check failed: {e}", flush=True)
        if check is not None:
            good = result.class_fit >= 0.5
            r = check.residual
            row.update(
                pixel_pairs=check.n_pairs,
                pixel_sign=check.sign,
                pixel_median=float(np.nanmedian(r)),
                pixel_disagree=int(check.disagree().sum()),
                pixel_good_median=(
                    float(np.nanmedian(r[good])) if good.any() else np.nan
                ),
                pixel_poor_median=(
                    float(np.nanmedian(r[~good])) if (~good).any() else np.nan
                ),
            )
        # two rings
        pairs = ph.prepare_pairs(result.params_used, class_ids=cls, max_sep=R.MAX_SEP)
        est = ph.estimate_period(pairs)
        est, _ = ph.sync_filament_directions(pairs, est)
        one = mx.one_ring_objective(pairs, est["period"], est["phases"])
        rng = np.random.default_rng(0)
        best = None
        for _ in range(6):
            two = mx.fit_two_rings(
                pairs,
                rng.integers(0, 2, pairs.n_filaments),
                (est["period"] * 0.93, est["period"] * 1.07),
            )
            if two is not None and (
                best is None or two["objective"] > best["objective"]
            ):
                best = two
        if best is not None:
            lo, hi = np.sort(best["periods"])
            row.update(
                two_lo=lo,
                two_hi=hi,
                two_gain=best["objective"] - one,
                two_small=int(best["n_filaments"].min()),
            )
        rows.append(row)
        print(
            f"  [{rule}] {len(numbers)} -> {len(cls)} classes: repeat {row['repeat']:.1f} "
            f"+- {row['sd']:.1f} A (score {row['score']:.3f}, support {row['support']:.2f}, "
            f"double {row['double_ratio']:.2f}, groups {row['groups']}); median fit "
            f"{row['median_fit']:.2f}, {row['well_fitting']} classes >= 0.5; pixel: "
            + (
                f"{row['pixel_disagree']} of {len(cls)} classes disagree (good "
                f"{row['pixel_good_median']:.0f} A, poor {row['pixel_poor_median']:.0f} A); "
                if check is not None
                else "n/a; "
            )
            + (
                f"two rings {row['two_lo']:.0f}/{row['two_hi']:.0f} A gain {row['two_gain']:+.3f}"
                if best is not None
                else ""
            ),
            flush=True,
        )
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
