"""How gaps left by a 2D class selection affect the ring fit.

Select2D removes whole classes, so the segments that stay have gaps between them, and
the views of the removed classes are missing from the ring. Four ways of removing a
share of the segments, from a data set that has none (EMPIAR-10940):

  random   : classes chosen at random
  worst    : the classes with the lowest fit first (what removing junk does)
  arc      : the classes in one contiguous arc of the ring (a missing view)
  thinning : segments chosen at random in every class (gaps, no class missing)

The pitch estimate (ring, filament directions, calibration, as the tab does it) is run on
what remains and compared with the full data's.

    python gap_ring_study.py --job-dir <...> --iteration 20 --out g.csv
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


def ring(used, cls, oriented=None, apix=1.0):
    pairs = ph.prepare_pairs(used, class_ids=cls, max_sep=R.MAX_SEP)
    if len(pairs.D) == 0:
        return None
    est = ph.estimate_period(pairs)
    est, fl = ph.sync_filament_directions(pairs, est)
    if fl.any():
        est = ph.estimate_period(pairs)
    cal = ph.calibrate_period(pairs, est["period"])
    fit = ph.class_fit(pairs, est["phases"], est["period"])
    hist, _ = ph.class_pair_histogram(pairs)
    weight = hist.sum((1, 2)) + hist.sum((0, 2))
    gated = None
    if oriented is not None:
        # the self-similarity of the averages of the classes that are left vetoes a
        # sub-multiple of the repeat (no pixel offsets: the metadata ring, gated)
        ids = [int(c) for c in pairs.class_ids]
        sim = px.self_similarity(
            [oriented[c] for c in ids], list(range(len(ids))), apix
        )
        g = px.estimate_period_gated(
            pairs, None, sim, alpha_basin=0.0, alpha_phases=0.0
        )
        gated = ph.calibrate_period(pairs, g["period"])["period"]
    return dict(
        gated=gated,
        raw=est["period"],
        period=cal["period"],
        bias=cal["bias"],
        phases=est["phases"],
        ids=[int(c) for c in pairs.class_ids],
        fit=fit,
        support=est["support"],
        double_ratio=est["double_ratio"],
        gap=float(ph.largest_phase_gap(est["phases"], weight)),
        pairs=len(pairs.D),
        score=float(est["score"]),
    )


def phase_rms(a_ids, a_ph, b_ids, b_ph):
    """RMS difference (share of the circle) of the phases of the classes in both."""
    common = [c for c in a_ids if c in set(b_ids)]
    if len(common) < 3:
        return float("nan")
    ia = [a_ids.index(c) for c in common]
    ib = [b_ids.index(c) for c in common]
    best = np.inf
    for s in (1.0, -1.0):
        d = s * np.asarray(a_ph)[ia] - np.asarray(b_ph)[ib]
        c0 = np.angle(np.mean(np.exp(1j * d)))
        r = np.angle(np.exp(1j * (d - c0)))
        best = min(best, float(np.sqrt(np.mean(r**2)) / (2 * np.pi)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=20)
    ap.add_argument("--rule", default="all")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    numbers = R.selections(params, images)[a.rule]
    result = R.merged_classes(params, images, apix, numbers)
    used = result.params_used
    cls = [int(c) for c in result.class_ids]
    oriented_all = R.oriented_averages(result, images, apix)
    oriented = {c: im for c, im in zip(cls, oriented_all)}
    base = ring(used, cls, oriented, apix)
    print(
        f"full data: {len(cls)} classes, {len(used):,} segments: repeat {base['period']:.1f} A (raw {base['raw']:.1f}, calibration bias {100 * base['bias']:+.2f}%), median fit {np.nanmedian(base['fit']):.2f}, support {base['support']:.2f}, double ratio {base['double_ratio']:.2f}, largest empty arc {base['gap']:.0f} deg",
        flush=True,
    )
    counts = used["rlnClassNumber"].astype(int).value_counts()
    seg_total = len(used)
    fit_by = dict(zip(base["ids"], base["fit"]))
    ph_by = dict(zip(base["ids"], base["phases"]))
    rows = []
    for scenario in ("random", "worst", "arc", "thinning"):
        for share in (0.1, 0.2, 0.3, 0.5):
            for draw in range(a.draws):
                rng = np.random.default_rng(1000 * draw + int(100 * share))
                if scenario == "thinning":
                    sub = used[rng.random(seg_total) >= share]
                    keep = [
                        c for c in cls if c in set(sub["rlnClassNumber"].astype(int))
                    ]
                else:
                    if scenario == "random":
                        order = list(rng.permutation(cls))
                    elif scenario == "worst":
                        order = sorted(
                            cls, key=lambda c: (np.nan_to_num(fit_by[c], nan=-9))
                        )
                    else:
                        centre = rng.uniform(-np.pi, np.pi)
                        order = sorted(
                            cls,
                            key=lambda c: abs(
                                np.angle(np.exp(1j * (ph_by[c] - centre)))
                            ),
                        )
                    gone, acc = set(), 0
                    for c in order:
                        if acc >= share * seg_total:
                            break
                        gone.add(int(c))
                        acc += int(counts.get(int(c), 0))
                    sub = used[~used["rlnClassNumber"].astype(int).isin(gone)]
                    keep = [c for c in cls if c not in gone]
                if len(keep) < 3:
                    continue
                r = ring(sub, keep, oriented, apix)
                if r is None:
                    continue
                rows.append(
                    dict(
                        scenario=scenario,
                        removed=1 - len(sub) / seg_total,
                        draw=draw,
                        classes=len(keep),
                        period=r["period"],
                        raw=r["raw"],
                        err_pct=100 * (r["period"] / base["period"] - 1),
                        gated_err_pct=100 * (r["gated"] / base["gated"] - 1),
                        raw_err_pct=100 * (r["raw"] / base["raw"] - 1),
                        bias_pct=100 * r["bias"],
                        phase_rms=phase_rms(
                            r["ids"], r["phases"], base["ids"], base["phases"]
                        ),
                        median_fit=float(np.nanmedian(r["fit"])),
                        support=r["support"],
                        double_ratio=r["double_ratio"],
                        gap=r["gap"],
                        pairs=r["pairs"],
                        score=r["score"],
                    )
                )
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"  {scenario} {share:.1f} done", flush=True)


if __name__ == "__main__":
    main()
