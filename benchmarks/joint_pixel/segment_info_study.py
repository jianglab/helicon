"""What each segment's alignment (shift and score) and the run of consecutive
segments in one class say about the quality of its assignment.

EMPIAR-10230 (labelled) and others. A segment is "of the wrong type" when its true
type (the extraction it comes from) differs from the majority type of its class
(classes of at least 90% one type).

    python segment_info_study.py --job-dir <.../job684> --iteration 25 [--labelled]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return float(
        (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    )


def runs(cls, fil):
    """Length of the run of equal consecutive classes each segment is in."""
    n = len(cls)
    start = np.r_[True, (cls[1:] != cls[:-1]) | (fil[1:] != fil[:-1])]
    rid = np.cumsum(start) - 1
    length = np.bincount(rid)
    return length[rid], rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    ap.add_argument("--iteration", type=int, default=25)
    ap.add_argument("--labelled", action="store_true")
    a = ap.parse_args()
    params, images, apix = R.load(a.job_dir, a.iteration)
    p = params.copy()
    p["cls"] = p["rlnClassNumber"].astype(int)
    p["track"] = p["rlnHelicalTrackLengthAngst"].astype(float)
    p["fil"] = p.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False).ngroup()
    p = p.sort_values(["fil", "track"]).reset_index(drop=True)
    print(
        f"{len(p):,} segments, {p.fil.nunique():,} filaments, {apix} A/pixel, box {images[0].shape[1]} px"
    )
    cols = [
        c
        for c in (
            "rlnOriginXAngst",
            "rlnOriginYAngst",
            "rlnMaxValueProbDistribution",
            "rlnLogLikeliContribution",
            "rlnNormCorrection",
            "rlnNrOfSignificantSamples",
            "rlnAnglePsi",
            "rlnAnglePsiPrior",
        )
        if c in p
    ]
    print("\ncolumns and their quantiles (1/25/50/75/99%):")
    for c in cols:
        v = p[c].astype(float)
        print(
            f"  {c:30s} {np.percentile(v, [1, 25, 50, 75, 99]).round(3).tolist()}  mean {v.mean():.3f} sd {v.std():.3f} skew {pd.Series(v).skew():.2f}"
        )
    step = np.median(np.diff(p.track[p.fil == p.fil.iloc[0]]))
    print(
        f"\ntypical segment spacing {np.median(p.groupby('fil').track.diff().dropna()):.1f} A"
    )
    p["ox"] = p["rlnOriginXAngst"].astype(float) if "rlnOriginXAngst" in p else np.nan
    p["maxp"] = p["rlnMaxValueProbDistribution"].astype(float)
    p["logl"] = p["rlnLogLikeliContribution"].astype(float)
    # score against the shift along the filament
    print(
        "\nalignment confidence (max probability) against the shift along the filament (A):"
    )
    bins = np.percentile(p.ox, np.linspace(0, 100, 9))
    g = p.groupby(
        pd.cut(p.ox, np.unique(bins), include_lowest=True), observed=True
    ).maxp.agg(["count", "mean", "median"])
    print(g.round(4).to_string())
    print(
        f"  correlation of max probability with |shift|: {np.corrcoef(p.maxp, p.ox.abs())[0, 1]:+.3f}; log-likelihood with |shift|: {np.corrcoef(p.logl, p.ox.abs())[0, 1]:+.3f}"
    )
    print(
        f"  shift along the filament: sd {p.ox.std():.2f} A, kurtosis {pd.Series(p.ox).kurt():.2f} (0 for a normal), skew {pd.Series(p.ox).skew():.2f}"
    )
    # score against the track length (position along the filament)
    p["rel"] = p.track - p.groupby("fil").track.transform("min")
    p["end"] = np.minimum(p.rel, p.groupby("fil").rel.transform("max") - p.rel)
    print("\nconfidence against the distance to the nearest end of the filament:")
    eb = [0, 50, 100, 200, 400, 800, 1e9]
    print(
        p.groupby(pd.cut(p.end, eb), observed=True)
        .maxp.agg(["count", "mean"])
        .round(4)
        .to_string()
    )
    # runs
    length, rid = runs(p.cls.values, p.fil.values)
    p["run"] = length
    print("\nrun length (segments in the same class in a row), all segments:")
    print(
        "  share of segments by run length 1/2/3/4-5/6-10/11+: "
        + " / ".join(
            f"{np.mean(m):.2f}"
            for m in [
                p.run == 1,
                p.run == 2,
                p.run == 3,
                (p.run >= 4) & (p.run <= 5),
                (p.run >= 6) & (p.run <= 10),
                p.run > 10,
            ]
        )
    )
    print(f"  median run {p.run.median():.0f} segments; mean {p.run.mean():.1f}")
    print(
        "  confidence by run length: "
        + ", ".join(
            f"{lab}: {p.maxp[m].mean():.3f}"
            for lab, m in (
                ("1", p.run == 1),
                ("2-3", (p.run >= 2) & (p.run <= 3)),
                ("4-9", (p.run >= 4) & (p.run <= 9)),
                ("10+", p.run >= 10),
            )
        )
    )
    if not a.labelled:
        return
    sf = p["rlnImageName"].str.contains("Extract/job671").astype(float)
    cshare = sf.groupby(p.cls).mean()
    ccount = p.groupby("cls").size()
    clear = cshare[((cshare <= 0.1) | (cshare >= 0.9)) & (ccount >= 100)]
    majority = (clear > 0.5).astype(int)
    seg_major = p.cls.map(majority)
    use = seg_major.notna()
    wrong = (sf.values.astype(int) != seg_major.values) & use.values
    print(
        f"\nsegments in clear-type classes: {int(use.sum()):,}; of the wrong type for their class: {int(wrong.sum()):,} ({100 * wrong[use.values].mean():.1f}%)"
    )
    # neighbours' agreement: share of the +-4 neighbours in the same class group (type)
    seg_type = seg_major.fillna(-1).values
    n = len(p)
    same = np.zeros(n)
    cnt = np.zeros(n)
    for s in range(1, 5):
        ok = p.fil.values[s:] == p.fil.values[:-s]
        for a_, b_ in (
            (slice(s, None), slice(None, -s)),
            (slice(None, -s), slice(s, None)),
        ):
            ev = ok & (seg_type[a_] >= 0) & (seg_type[b_] >= 0)
            same[a_] += np.where(ev, seg_type[a_] == seg_type[b_], 0)
            cnt[a_] += ev
    agree = np.where(cnt > 0, same / np.maximum(cnt, 1), np.nan)
    feats = {
        "max probability (low = bad)": -p.maxp.values,
        "log-likelihood contribution (low = bad)": -p.logl.values,
        "|shift along the filament|": p.ox.abs().values,
        "short run (1 / run length)": 1.0 / p.run.values,
        "neighbours of another type (+-4 segments)": 1.0 - agree,
    }
    print("\nAUC for finding the segments of the wrong type (0.5 = no information):")
    for name, v in feats.items():
        print(f"  {name:45s} {auc(v[wrong], v[use.values & ~wrong]):.3f}")
    # runs: wrong-type segments by run length
    print("\nwrong-type share by the run length of the segment:")
    for lab, m in (
        ("1", p.run == 1),
        ("2-3", (p.run >= 2) & (p.run <= 3)),
        ("4-9", (p.run >= 4) & (p.run <= 9)),
        ("10+", p.run >= 10),
    ):
        mm = m.values & use.values
        print(
            f"  run {lab:5s}: {mm.sum():7,} segments, {100 * wrong[mm].mean():5.1f}% wrong type"
        )
    # per filament
    fil = p.groupby("fil").agg(
        n=("run", "size"),
        run=("run", "mean"),
        maxp=("maxp", "mean"),
        sf=("rlnImageName", lambda s: s.str.contains("Extract/job671").mean()),
    )
    fw = pd.Series(wrong, index=p.index).groupby(p.fil).mean()
    fil["wrong"] = fw
    pure = (fil.sf <= 0.1) | (fil.sf >= 0.9)
    print(
        f"\nper filament: correlation of the share of wrong-type segments with the filament's mean run length {np.corrcoef(fil.wrong[pure], fil.run[pure])[0, 1]:+.3f}, with its mean confidence {np.corrcoef(fil.wrong[pure], fil.maxp[pure])[0, 1]:+.3f}"
    )
    p.to_pickle(R.OUT / "segment_info_10230.pkl")


if __name__ == "__main__":
    main()
