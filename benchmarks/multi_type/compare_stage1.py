"""The first stage as the Types card does it, against the alignment of class sequences.

The Types card groups the classes by how alike their averages are and gives each
filament the group most of its segments are in. The alignment (``filament_alignment.py``)
clusters filaments directly from their class sequences. Both are run on the same
sampled filaments and scored against the reference sets: the adjusted Rand index of
the groups with the sets, and for each pair of sets how well the groups tell them
apart (the balanced accuracy of the best two-way assignment of groups).

    python compare_stage1.py --reference reference.parquet \
        --mrcs run_it025_classes.mrcs --similarity similarity.npy
"""

from __future__ import annotations

import argparse

import numpy as np

from filament_alignment import add_arguments, prepare
from two_stage import ari, stage_one


def vote(seqs, class_group, min_share=0.0):
    """Each filament's group: the one most of its segments' classes are in."""
    n_groups = int(class_group.max()) + 1
    out = np.full(len(seqs), -1)
    for i, s in enumerate(seqs):
        g = class_group[s[s >= 0]]
        g = g[g >= 0]
        if len(g):
            counts = np.bincount(g, minlength=n_groups)
            if counts.max() >= min_share * len(g):
                out[i] = int(counts.argmax())
    return out


def balanced(groups, truth, i, j):
    """Best balanced accuracy of telling set i from set j by their groups."""
    ok = groups >= 0
    a, b = groups[ok & (truth == i)], groups[ok & (truth == j)]
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    k = int(groups.max()) + 1
    pa = np.bincount(a, minlength=k) / len(a)
    pb = np.bincount(b, minlength=k) / len(b)
    return float(
        0.5 * (np.where(pa >= pb, pa, 0).sum() + np.where(pb > pa, pb, 0).sum())
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    a = p.parse_args()
    d = prepare(a)
    sets, seqs, truth, length, A, S = (
        d["sets"],
        d["seqs"],
        d["truth"],
        d["length"],
        d["A"],
        d["similarity"],
    )
    from helicon.webApps.lib import helical_pitch_types as ty

    results = {}
    for k in (3, 4, 5, 6, 8):
        cg, _ = ty.pixel_groups_n(S, k)
        results[f"card: class groups ({k}) + vote"] = vote(seqs, cg)
    for k in (3, 4, 5, 6, 8):
        results[f"alignment clustering ({k} groups)"] = stage_one(A, length, k)

    print(
        "\nadjusted Rand index with the reference sets (filaments the card cannot place count as their own group)"
    )
    for name, g in results.items():
        print(f"  {name:36s} {ari(np.where(g < 0, g.max() + 1, g), truth):.3f}")

    pairs = [(i, j) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    show = {
        "card, 5 class groups": results["card: class groups (5) + vote"],
        "alignment, 5 groups": results["alignment clustering (5 groups)"],
        "card, 8 class groups": results["card: class groups (8) + vote"],
        "alignment, 8 groups": results["alignment clustering (8 groups)"],
    }
    print("\nbalanced accuracy of telling two sets apart by their groups")
    print(f"{'':50s}" + "".join(f"{k:>22s}" for k in show))
    for i, j in pairs:
        print(
            f"{sets[i][:22] + ' vs ' + sets[j][:22]:50s}"
            + "".join(f"{balanced(g, truth, i, j):22.3f}" for g in show.values())
        )


if __name__ == "__main__":
    main()
