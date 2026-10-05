"""Does the pixel check find the classes of another helical type? (EMPIAR-10230)

A RELION Class2D of EMPIAR-10230, tau filaments of two polymorphs (PHF and SF) in
one data set. The pooled ring is a blend; a class of the other type has segments
that do not sit at one azimuth of the ring, and, independently, an average that
does not lie where the ring puts it. This runs the AbInitio3D analysis as the tab
does, reads the class-average check, and then re-estimates the repeat from the
classes the pixels agree with and from those they do not.

    python real_mixed.py --job-dir <.../Class2D/jobNNN> --iteration 25 --out r.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402

from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402
from helicon.webApps.lib import helical_pitch_pixel as px  # noqa: E402


def original_classes(numbers, result, keep):
    """The selected class numbers whose merged representative is in ``keep``."""
    keep = {int(c) for c in keep}
    merged = {int(g): int(k) for g, k in (result.merged_into or {}).items()}
    return [c for c in numbers if c in keep or merged.get(c) in keep]


def analyze(params, images, apix, numbers):
    return R.merged_classes(params, images, apix, numbers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rules", default="top20,all")
    a = ap.parse_args()

    params, images, apix = R.load(a.job_dir, a.iteration)
    print(
        f"{len(params):,} segments, {len(images)} classes, {apix} A/pixel", flush=True
    )
    sels = R.selections(params, images)
    rows = []
    for rule in a.rules.split(","):
        numbers = sels[rule]
        result = analyze(params, images, apix, numbers)
        cls = [int(c) for c in result.class_ids]
        check = px.class_pixel_check(
            [images[c - 1] for c in cls],
            result.class_zdir,
            result.phases,
            result.period,
            apix,
        )
        print(
            f"\n[{rule}] {len(numbers)} classes -> {len(cls)}; pooled repeat "
            f"{result.period:.1f} +- {result.period_sd:.1f} A; score "
            f"{result.scan_scores.max():.3f}; support {result.support:.2f}; "
            f"double ratio {result.double_ratio:.2f}; metadata groups "
            f"{[len(g) for g in result.groups]} -> {[round(p, 1) for p in result.group_periods]}",
            flush=True,
        )
        if check is None:
            print("  no pixel check (too few registered pairs)")
            continue
        table = pd.DataFrame(
            dict(
                cls=cls,
                segments=result.class_count,
                fit=np.round(result.class_fit, 2),
                pixel_residual_A=np.round(check.residual, 0),
                disagree=check.disagree(),
            )
        ).sort_values("fit")
        print(
            f"  pixel sign {check.sign:+.0f}, {check.n_pairs} registered pairs, "
            f"cutoff {check.cutoff:.0f} A"
        )
        print(table.to_string(index=False), flush=True)
        table.insert(0, "rule", rule)
        table.to_csv(a.out.replace(".csv", f"_{rule}_classes.csv"), index=False)

        agree = [c for c, g in zip(cls, check.agree()) if g]
        off = [c for c, g in zip(cls, check.disagree()) if g]
        for name, members in (("agree", agree), ("disagree", off)):
            chosen = original_classes(numbers, result, members)
            if len(members) < 3:
                print(f"  {name}: {len(members)} class(es), too few for a ring")
                continue
            sub = analyze(params, images, apix, chosen)
            print(
                f"  {name}: {len(members)} classes ({len(chosen)} before merging) -> "
                f"repeat {sub.period:.1f} +- {sub.period_sd:.1f} A; score "
                f"{sub.scan_scores.max():.3f}; support {sub.support:.2f}; "
                f"median fit {np.nanmedian(sub.class_fit):.2f}",
                flush=True,
            )
            rows.append(
                dict(
                    rule=rule,
                    subset=name,
                    classes=len(members),
                    repeat=sub.period,
                    sd=sub.period_sd,
                    score=float(sub.scan_scores.max()),
                    median_fit=float(np.nanmedian(sub.class_fit)),
                )
            )
        rows.append(
            dict(
                rule=rule,
                subset="pooled",
                classes=len(cls),
                repeat=result.period,
                sd=result.period_sd,
                score=float(result.scan_scores.max()),
                median_fit=float(np.nanmedian(result.class_fit)),
            )
        )
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
