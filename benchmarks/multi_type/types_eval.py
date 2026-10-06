"""Score the grouping of classes into types against the refined 3D sets.

Runs the Types card's steps (classes above the abundance floor, registered
correlation of every pair, average-linkage groups at a cut) on a Class2D job and
scores the groups against the reference of ``build_reference.py``:

* class level: each class is labelled by the set most of its *reference*
  segments are in (classes with few reference segments are not scored); the
  groups are compared with those labels (adjusted Rand index, purity);
* a table of the groups by the sets, in reference segments.

The reference is noisy, so the numbers say which grouping agrees better, not how
close to the truth any of them is.

    python types_eval.py --mrcs run_it025_classes.mrcs --reference reference.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd

from helicon.webApps.lib import helical_pitch_types as ty

OUT = Path(os.environ.get("TYPES_OUT", "types_out"))
OUT.mkdir(parents=True, exist_ok=True)


def ari(a, b) -> float:
    """Adjusted Rand index of two labelings."""
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    t = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(t, (ai, bi), 1)
    c2 = lambda x: x * (x - 1) / 2
    sij, sa, sb = c2(t).sum(), c2(t.sum(1)).sum(), c2(t.sum(0)).sum()
    exp = sa * sb / c2(len(a))
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def reference(ref: pd.DataFrame, n_classes: int):
    """Per class: total segments, reference segments per set, set names."""
    sets = [
        c
        for c in ref.columns
        if c
        not in (
            "rlnMicrographName",
            "rlnHelicalTubeID",
            "rlnHelicalTrackLengthAngst",
            "rlnClassNumber",
        )
    ]
    cls = ref["rlnClassNumber"].to_numpy().astype(int) - 1
    total = np.bincount(cls, minlength=n_classes).astype(float)
    per_set = np.stack(
        [
            np.bincount(cls, weights=ref[s].to_numpy(float), minlength=n_classes)
            for s in sets
        ],
        axis=1,
    )
    return total, per_set, sets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mrcs", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--cuts", default="0.05,0.08,0.1,0.12,0.15,0.2,0.25,0.3")
    p.add_argument(
        "--min-ref",
        type=int,
        default=300,
        help="reference segments for a class to be scored",
    )
    a = p.parse_args()

    with mrcfile.open(a.mrcs, permissive=True) as f:
        images = [np.asarray(x, float) for x in f.data]
        apix = float(f.voxel_size.x)
    ref = pd.read_parquet(a.reference)
    total, per_set, sets = reference(ref, len(images))

    floor = max(100, int(0.002 * total.sum()))
    classes = [
        c
        for c in range(len(images))
        if total[c] >= floor and images[c].max() > images[c].min()
    ]
    print(
        f"{len(classes)} of {len(images)} classes above the floor of {floor}; apix {apix}"
    )
    cache = OUT / "similarity.npy"
    if cache.exists():
        S = np.load(cache)
    else:
        S = ty.class_similarity(
            [images[c] for c in classes], list(range(len(classes))), apix
        )
        np.save(cache, S)

    lab_n = per_set[classes].sum(1)
    major = per_set[classes].argmax(1)
    purity = per_set[classes].max(1) / np.maximum(lab_n, 1)
    scored = lab_n >= a.min_ref
    print(
        f"{int(scored.sum())} classes with at least {a.min_ref} reference segments; "
        f"median purity {np.median(purity[scored]):.2f}"
    )
    print(
        "classes by majority set:",
        {s: int(((major == i) & scored).sum()) for i, s in enumerate(sets)},
    )

    for cut in [float(x) for x in a.cuts.split(",")]:
        g = ty.pixel_groups(S, cut)
        n_groups = int(g.max()) + 1
        ok = scored & (g >= 0)
        score = ari(g[ok], major[ok]) if ok.sum() > 2 else float("nan")
        # reference segments of the grouped classes that sit in their group's majority set
        tab = np.zeros((n_groups, len(sets)))
        for k in range(n_groups):
            tab[k] = per_set[np.asarray(classes)[g == k]].sum(0)
        pur = tab.max(1).sum() / max(tab.sum(), 1)
        print(
            f"cut {cut:.2f}: {n_groups} groups, {int((g < 0).sum())} classes not grouped, "
            f"ARI {score:.2f}, segment purity {pur:.2f}"
        )
        if abs(cut - 0.15) < 1e-9:
            print(
                pd.DataFrame(tab.astype(int), columns=sets)
                .assign(classes=np.bincount(g[g >= 0]))
                .to_string()
            )
    print("suggested cut:", ty.suggest_cut(S))


if __name__ == "__main__":
    main()
