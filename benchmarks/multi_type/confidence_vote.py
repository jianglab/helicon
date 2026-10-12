"""Filament votes that count the segments RELION was sure of.

The Types card gives each filament the group most of its segments' classes are in.
About a third of the segments sit in a wrong class, and each of them votes as much as
a right one. RELION writes for every segment how sure it was of its class: the
largest value of its probability distribution over classes and orientations
(``rlnMaxValueProbDistribution``) and how many samples were significant
(``rlnNrOfSignificantSamples``). Here the vote is weighted by them, or limited to the
sure segments, and scored against the reference sets.

First a direct check: do the segments whose class agrees with their filament's set
have a higher probability than those whose class does not?

    python confidence_vote.py --star run_it025_data.star --reference reference.parquet \
        --mrcs run_it025_classes.mrcs --similarity similarity.npy
"""

from __future__ import annotations

import argparse

import mrcfile
import numpy as np
import pandas as pd
import starfile

from filament_alignment import auc
from two_stage import ari
from compare_stage1 import balanced

SKIP = (
    "rlnMicrographName",
    "rlnHelicalTubeID",
    "rlnHelicalTrackLengthAngst",
    "rlnClassNumber",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--star", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--mrcs", required=True)
    p.add_argument("--similarity", required=True)
    p.add_argument("--min-segments", type=int, default=5)
    a = p.parse_args()

    star = starfile.read(a.star)
    star = star["particles"] if isinstance(star, dict) else star
    ref = pd.read_parquet(a.reference)
    # the reference was built from this star file, row by row
    assert len(ref) == len(star)
    assert (ref["rlnClassNumber"].to_numpy() == star["rlnClassNumber"].to_numpy()).all()
    prob = star["rlnMaxValueProbDistribution"].to_numpy(float)
    nsig = star["rlnNrOfSignificantSamples"].to_numpy(float)
    sets = [c for c in ref.columns if c not in SKIP]
    cls = ref["rlnClassNumber"].to_numpy().astype(int) - 1
    n_classes = int(cls.max()) + 1
    print(
        f"{len(star):,} segments; max probability: median {np.median(prob):.3f}, "
        f"middle half {np.percentile(prob, 25):.3f} to {np.percentile(prob, 75):.3f}; "
        f"significant samples: median {np.median(nsig):.0f}"
    )

    # filaments, and the one reference set each belongs to
    key = (
        ref["rlnMicrographName"].astype(str) + "|" + ref["rlnHelicalTubeID"].astype(str)
    ).astype("category")
    fil = key.cat.codes.to_numpy()
    n_fil = len(key.cat.categories)
    flags = ref[sets].to_numpy().astype(np.int32)
    per_fil = np.zeros((n_fil, len(sets)), dtype=np.int64)
    np.add.at(per_fil, fil, flags)
    n_seg = np.bincount(fil, minlength=n_fil)
    single = ((per_fil > 0).sum(1) == 1) & (n_seg >= a.min_segments)
    truth_fil = np.where(single, per_fil.argmax(1), -1)
    print(
        f"{int(single.sum()):,} filaments of exactly one set with {a.min_segments}+ segments"
    )

    # 1. direct check: a segment's class agrees with its filament's set when the class's
    # reference segments are mostly of that set
    per_cls = np.zeros((n_classes, len(sets)))
    np.add.at(per_cls, cls, flags)
    cls_set = np.where(per_cls.sum(1) > 0, per_cls.argmax(1), -1)
    seg_truth = truth_fil[fil]
    known = (seg_truth >= 0) & (cls_set[cls] >= 0)
    agree = cls_set[cls] == seg_truth
    print(
        f"\nsegments of one-set filaments whose class is mostly of that set: {agree[known].mean():.2f}"
    )
    for name, v in (
        ("max probability", prob),
        ("significant samples (fewer = surer)", -nsig),
    ):
        print(
            f"  AUC of {name} for agreeing against disagreeing segments: {auc(v[known & agree], v[known & ~agree]):.3f}"
        )
    for q in (0.25, 0.5, 0.75):
        cut = np.quantile(prob, q)
        m = known & (prob >= cut)
        print(
            f"  segments with probability >= {cut:.3f} (top {100 * (1 - q):.0f}%): {agree[m].mean():.2f} agree"
        )

    # 2. the card's vote, weighted
    from helicon.webApps.lib import helical_pitch_types as ty

    with mrcfile.open(a.mrcs, permissive=True) as f:
        has_signal = [x.max() > x.min() for x in f.data]
    total = np.bincount(cls, minlength=n_classes).astype(float)
    floor = max(100, int(0.002 * total.sum()))
    classes = [c for c in range(n_classes) if total[c] >= floor and has_signal[c]]
    S = np.load(a.similarity)
    weights = {
        "every segment the same": np.ones(len(prob)),
        "weight = probability": prob,
        "weight = probability squared": prob**2,
        "only the top half by probability": (prob >= np.median(prob)).astype(float),
        "only the top quarter": (prob >= np.quantile(prob, 0.75)).astype(float),
        "only 1 significant sample": (nsig <= 1).astype(float),
    }
    ok = truth_fil >= 0
    for k in (5, 6, 8):
        cg, _ = ty.pixel_groups_n(S, k)
        group_of = np.full(n_classes, -1)
        group_of[classes] = cg
        g = group_of[cls]
        print(
            f"\nthe card's class groups ({k}), the filament's group by a weighted vote"
        )
        print(
            f"{'':36s}{'ARI':>7s}{'filaments placed':>18s}   balanced accuracy for each pair of sets"
        )
        for name, w in weights.items():
            m = g >= 0
            votes = np.zeros((n_fil, int(cg.max()) + 1))
            np.add.at(votes, (fil[m], g[m]), w[m])
            vote = np.where(votes.sum(1) > 0, votes.argmax(1), -1)
            placed = ok & (vote >= 0)
            lab = np.where(vote[ok] < 0, vote.max() + 1, vote[ok])
            pairs = [
                balanced(vote[ok], truth_fil[ok], i, j)
                for i in range(len(sets))
                for j in range(i + 1, len(sets))
            ]
            print(
                f"  {name:34s}{ari(lab, truth_fil[ok]):7.3f}{placed.sum() / ok.sum():17.0%}   "
                + " ".join(f"{x:.2f}" for x in pairs)
            )


if __name__ == "__main__":
    main()
