"""How filament consistency changes with the cut of the class grouping (no labels).

Classes are grouped by the registered correlation of their averages
(``helical_pitch_types.pixel_groups``). A filament is of one type, so its
segments should fall in one group: the share of each filament's segments that
are in its majority group ("filament consistency") stays high while the cut
separates types, and drops when it splits one type into views. The table shows
that curve for a range of cuts, with (optionally) the agreement with a reference.

    python cut_study.py --star run_it020_data.star --mrcs run_it020_classes.mrcs --name 10940
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import scipy.sparse as sp
import starfile

from helicon.webApps.lib import helical_pitch_types as ty

OUT = Path(os.environ.get("TYPES_OUT", "types_out"))
OUT.mkdir(parents=True, exist_ok=True)


def filament_consistency(labels, classes, cls, fil_code, n_fil, min_segments=5):
    """Share of the segments of each filament in its majority group.

    Parameters
    ----------
    labels : numpy.ndarray
        Group of each class in ``classes`` (-1: not grouped).
    classes : sequence of int
        Class numbers (0-based) of the rows of ``labels``.
    cls, fil_code : numpy.ndarray
        Per segment, its class (0-based) and filament code.
    n_fil : int
        Number of filaments.

    Returns
    -------
    float, int
        The consistency over the segments of grouped classes (1 for one group), and
        the number of groups.
    """
    n_groups = int(labels.max()) + 1
    if n_groups < 2:
        return 1.0, max(n_groups, 0)
    group_of = np.full(int(cls.max()) + 1, -1)
    group_of[np.asarray(classes)] = labels
    g = group_of[cls]
    m = g >= 0
    T = (
        sp.coo_matrix((np.ones(m.sum()), (fil_code[m], g[m])), shape=(n_fil, n_groups))
        .tocsr()
        .toarray()
    )
    T = T[T.sum(1) >= min_segments]
    return float(T.max(1).sum() / T.sum()), n_groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--star", required=True)
    p.add_argument("--mrcs", required=True)
    p.add_argument("--name", required=True)
    p.add_argument(
        "--cuts", default="0.05,0.07,0.09,0.10,0.12,0.14,0.15,0.17,0.20,0.25"
    )
    a = p.parse_args()

    with mrcfile.open(a.mrcs, permissive=True) as f:
        images = [np.asarray(x, float) for x in f.data]
        apix = float(f.voxel_size.x)
    d = starfile.read(a.star)
    d = d["particles"] if isinstance(d, dict) else d
    cls = d["rlnClassNumber"].to_numpy().astype(int) - 1
    fil = (
        d["rlnMicrographName"].astype(str) + "_" + d["rlnHelicalTubeID"].astype(str)
    ).astype("category")
    fcode, n_fil = fil.cat.codes.to_numpy(), len(fil.cat.categories)
    total = np.bincount(cls, minlength=len(images)).astype(float)
    floor = max(100, int(0.002 * total.sum()))
    classes = [
        c
        for c in range(len(images))
        if total[c] >= floor and images[c].max() > images[c].min()
    ]
    print(
        f"{a.name}: {len(classes)} of {len(images)} classes above {floor} segments; apix {apix:.2f}"
    )
    cache = OUT / f"similarity_{a.name}.npy"
    if cache.exists():
        S = np.load(cache)
    else:
        S = ty.class_similarity(
            [images[c] for c in classes], list(range(len(classes))), apix
        )
        np.save(cache, S)
    for cut in [float(x) for x in a.cuts.split(",")]:
        labels = ty.pixel_groups(S, cut)
        cons, k = filament_consistency(labels, classes, cls, fcode, n_fil)
        sizes = np.bincount(labels[labels >= 0], minlength=k) if k else []
        print(
            f"cut {cut:.2f}: {k} groups {list(sizes)[:8]}, {int((labels < 0).sum())} not grouped, filament consistency {cons:.3f}"
        )
    print("suggested cut (largest gap):", round(ty.suggest_cut(S), 3))


if __name__ == "__main__":
    main()
