"""Two rings on a real data set: one type (EMPIAR-10940) or two (EMPIAR-10230).

The AbInitio3D analysis as the tab runs it, then the two-ring fit from several
starts (the filaments split by their own repeat, and at random), the one-ring
objective for comparison, and the class-average check of each ring.

    python real_two_rings.py --job-dir <...> --iteration 25 --out r.csv
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
    ap.add_argument("--rule", default="all")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    params, images, apix = R.load(a.job_dir, a.iteration)
    numbers = R.selections(params, images)[a.rule]
    result = R.merged_classes(params, images, apix, numbers)
    cls = [int(c) for c in result.class_ids]
    pairs = ph.prepare_pairs(result.params_used, class_ids=cls, max_sep=R.MAX_SEP)
    est = ph.estimate_period(pairs)
    est, _ = ph.sync_filament_directions(pairs, est)
    one = mx.one_ring_objective(pairs, est["period"], est["phases"])
    print(
        f"{len(cls)} classes, {pairs.n_filaments} filaments; one ring: repeat "
        f"{est['period']:.1f} A, objective {one:.4f}",
        flush=True,
    )

    fil = result.filaments
    lookup = {
        (m, int(t)): p
        for m, t, p, ok in zip(
            fil.rlnMicrographName, fil.rlnHelicalTubeID, fil.pitch, fil.fitted
        )
        if ok
    }
    pitch = np.array(
        [lookup.get((k[0], int(k[1])), np.nan) for k in pairs.filament_keys]
    )
    rng = np.random.default_rng(0)
    starts = [("by repeat", mx.starting_labels(pairs, pitch, rng))] + [
        (f"random {i}", rng.integers(0, 2, pairs.n_filaments))
        for i in range(a.restarts)
    ]
    rows, best = [], None
    for name, labels in starts:
        two = mx.fit_two_rings(
            pairs, labels, (est["period"] * 0.93, est["period"] * 1.07)
        )
        if two is None:
            continue
        lo, hi = np.sort(two["periods"])
        print(
            f"  start {name:10s}: repeats {lo:.1f} / {hi:.1f} A, filaments "
            f"{two['n_filaments'].tolist()}, objective {two['objective']:.4f} "
            f"(gain {two['objective'] - one:+.4f}), {two['rounds']} rounds",
            flush=True,
        )
        rows.append(
            dict(
                start=name,
                lo=lo,
                hi=hi,
                n0=int(two["n_filaments"][0]),
                n1=int(two["n_filaments"][1]),
                objective=two["objective"],
                gain=two["objective"] - one,
                one_ring_repeat=est["period"],
            )
        )
        if best is None or two["objective"] > best["objective"]:
            best = two
    pd.DataFrame(rows).to_csv(a.out, index=False)
    if best is None:
        return
    print("\nbest:", np.round(best["periods"], 1), best["n_filaments"].tolist())
    # per ring: the class-average check, and how much the class phases differ
    sub = []
    for t in (0, 1):
        check = px.class_pixel_check(
            [images[c - 1] for c in cls],
            result.class_zdir,
            best["phases"][t],
            best["periods"][t],
            apix,
        )
        r = None if check is None else check.residual
        print(
            f"ring {t}: repeat {best['periods'][t]:.1f} A, score {best['scores'][t]:.3f},"
            + (
                ""
                if r is None
                else f" class-average residual median {np.nanmedian(r):.0f} A,"
                f" {int(check.disagree().sum())} of {len(r)} classes disagree"
            )
        )
        sub.append(r)
    d = best["phases"][0] - best["phases"][1]
    print("filaments by ring and span (A):")
    span = pairs.filament_span
    for t in (0, 1):
        s = span[best["labels"] == t]
        print(f"  ring {t}: n={len(s)}, median span {np.median(s):.0f}")
    pd.DataFrame(
        dict(
            cls=cls,
            phase0=best["phases"][0],
            phase1=best["phases"][1],
            resid0=sub[0] if sub[0] is not None else np.nan,
            resid1=sub[1] if sub[1] is not None else np.nan,
            fit=result.class_fit,
        )
    ).to_csv(a.out.replace(".csv", "_classes.csv"), index=False)


if __name__ == "__main__":
    main()
