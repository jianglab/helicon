"""Does a segment's shift along the filament carry a finer phase than its class?

A class is a sector of the repeat; a segment's origin shift along the filament is how
far it had to be moved to match the class average, so (class phase + shift) should be
the segment's own phase, and along one filament it should advance linearly with the
position. This compares, for the segments of the classes of one type, the scatter
around each filament's fitted phase line of (a) the class phase alone and (b) the class
phase plus the shift, and relates the residual to the alignment confidence and to the
run length.

    python shift_phase_study.py --job-dir <...> --iteration 25 --classes "1,2,3"
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
from helicon.webApps.lib import helical_pitch_types as ty  # noqa: E402


def wrap(a, period=360.0):
    return (a + period / 2.0) % period - period / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=25)
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--cut", type=float, default=0.15)
    ap.add_argument("--saved", default="types10230")
    ap.add_argument(
        "--rule",
        default="",
        help="use the classes of this rule (all, top20) instead of a saved group",
    )
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    if a.rule:
        numbers = R.selections(params, images)[a.rule]
        print(f"rule {a.rule}: {len(numbers)} classes", flush=True)
    else:
        cdf = pd.read_csv(R.OUT / f"{a.saved}_classes.csv")
        S = np.load(R.OUT / f"{a.saved}_S.npy")
        g = ty.pixel_groups(S, a.cut)
        numbers = [int(c) for c, k in zip(cdf.cls, g) if k == a.group]
        print(f"group {a.group}: {len(numbers)} classes", flush=True)
    result = R.merged_classes(params, images, apix, numbers)
    used = result.params_used
    P = result.period
    print(
        f"repeat {P:.1f} A, {result.n_filaments} filaments, median class fit {np.nanmedian(result.class_fit):.2f}",
        flush=True,
    )
    seg = ph.segment_angles(used, result, fold=1)
    s = result.segments  # row, filament, direction, azimuth (class phase, rad), ...
    row = s["row"].values
    df = pd.DataFrame(
        dict(
            row=row,
            fil=s["filament"].values,
            azimuth=np.degrees(s["azimuth"].values),
            rot=seg["rlnAngleRot"].values[row],
            ox=used["rlnOriginXAngst"].astype(float).values[row],
            maxp=used["rlnMaxValueProbDistribution"].astype(float).values[row],
            logl=used["rlnLogLikeliContribution"].astype(float).values[row],
            cls=used["rlnClassNumber"].astype(int).values[row],
            track=used["rlnHelicalTrackLengthAngst"].astype(float).values[row],
        )
    ).dropna()
    # the sign conventions between the ring's class angle, the filament's fitted angle
    # and the shift are tried both ways
    best = None
    for s1 in (1.0, -1.0):
        d = wrap(df.rot.values - s1 * df.azimuth.values)
        c = np.angle(np.mean(np.exp(1j * np.radians(d))))
        r = wrap(d - np.degrees(c))
        if best is None or np.std(r) < best[0]:
            best = (np.std(r), s1, np.degrees(c))
    _, s1, c0 = best
    df["res"] = wrap(
        df.rot.values - s1 * df.azimuth.values - c0
    )  # degrees; class phase alone
    print(
        f"\nresidual of the class phase from the filament's fitted angle: sd {df.res.std():.1f} deg ({df.res.std() * P / 360:.0f} A); the sector of a class is about {360 / max(len(numbers), 1):.0f} deg",
        flush=True,
    )
    perA = 360.0 / P  # degrees of phase per A of shift
    for s2 in (1.0, -1.0):
        r2 = wrap(df.res.values - s2 * df.ox.values * perA)
        print(
            f"  with the shift added (sign {s2:+.0f}): sd {np.std(r2):.1f} deg ({np.std(r2) * P / 360:.0f} A)"
        )
    # a regression of the residual on the shift
    k = np.polyfit(df.ox.values, df.res.values, 1)
    print(
        f"  regression of the residual on the shift: slope {k[0]:+.3f} deg/A (a shift that is the phase: {perA:+.3f} or {-perA:+.3f}); correlation {np.corrcoef(df.ox, df.res)[0, 1]:+.3f}"
    )
    # the same, for the typical segments (robust)
    inner = df[np.abs(df.res) < 60]
    kk = np.polyfit(inner.ox.values, inner.res.values, 1)
    print(
        f"  the same on the segments within 60 deg of their line ({len(inner) / len(df):.2f} of them): slope {kk[0]:+.3f} deg/A, correlation {np.corrcoef(inner.ox, inner.res)[0, 1]:+.3f}"
    )
    # what the residual says: the confidence and the run length
    df = df.sort_values(["fil", "track"]).reset_index(drop=True)
    start = np.r_[
        True,
        (df.cls.values[1:] != df.cls.values[:-1])
        | (df.fil.values[1:] != df.fil.values[:-1]),
    ]
    rid = np.cumsum(start) - 1
    df["run"] = np.bincount(rid)[rid]
    df["bad"] = df.res.abs() > 90
    print(
        f"\nsegments more than 90 deg off their filament's line: {100 * df.bad.mean():.1f}%"
    )
    from segment_info_study import auc  # noqa: E402

    ok = df.bad.values
    for name, v in (
        ("low max probability", -df.maxp.values),
        ("low log-likelihood", -df.logl.values),
        ("|shift|", np.abs(df.ox.values)),
        ("short run", 1.0 / df.run.values),
    ):
        print(f"  AUC of {name:22s} for those segments: {auc(v[ok], v[~ok]):.3f}")
    print(
        "  share off the line by run length: "
        + ", ".join(
            f"{lab}: {100 * df.bad[m].mean():.1f}%"
            for lab, m in (
                ("1", df.run == 1),
                ("2-3", df.run.between(2, 3)),
                ("4-9", df.run.between(4, 9)),
                ("10+", df.run >= 10),
            )
        )
    )
    print(
        "  share off the line by confidence quartile: "
        + ", ".join(
            f"Q{q + 1}: {100 * df.bad[pd.qcut(df.maxp, 4, labels=False) == q].mean():.1f}%"
            for q in range(4)
        )
    )
    # per filament: mean residual magnitude
    fil = df.groupby("fil").agg(
        n=("res", "size"),
        mean_abs=("res", lambda v: np.mean(np.abs(v))),
        run=("run", "mean"),
        maxp=("maxp", "mean"),
    )
    f2 = fil[fil.n >= 10]
    print(
        f"\nper filament (>= 10 segments, n={len(f2)}): mean |residual| quantiles 10/50/90%: {np.percentile(f2.mean_abs, [10, 50, 90]).round(1).tolist()} deg; correlation with mean run length {np.corrcoef(f2.mean_abs, f2.run)[0, 1]:+.3f}, with mean confidence {np.corrcoef(f2.mean_abs, f2.maxp)[0, 1]:+.3f}"
    )


if __name__ == "__main__":
    main()
