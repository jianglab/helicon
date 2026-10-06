"""Do two reference sets that share their classes differ in their filaments' repeats?

Two helical types can fall into the same 2D classes (look alike in the class
averages), so no grouping of classes can tell them apart. What differs is the
repeat of the filaments. This runs the AbInitio3D analysis on the classes the two
sets share, and compares the per-filament repeats of the filaments that belong
mostly to one set or the other (reference of ``build_reference.py``).

    python pair_rings.py --class2d run_it025_data.star --reference reference.parquet \
        --set-a typeA --set-b typeB
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

from helicon.webApps.lib import helical_pitch_phase as ph


def auc(pos, neg):
    v = np.concatenate([pos, neg])
    r = v.argsort().argsort() + 1.0
    return float(
        (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--class2d", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--set-a", required=True)
    p.add_argument("--set-b", required=True)
    p.add_argument("--min-ref", type=int, default=300)
    a = p.parse_args()

    ref = pd.read_parquet(a.reference)
    cls = ref["rlnClassNumber"].to_numpy().astype(int)
    n = cls.max() + 1
    ca = np.bincount(cls, weights=ref[a.set_a].to_numpy(float), minlength=n)
    cb = np.bincount(cls, weights=ref[a.set_b].to_numpy(float), minlength=n)
    chosen = [c for c in range(1, n) if ca[c] + cb[c] >= a.min_ref]
    print(
        f"{len(chosen)} classes with {a.min_ref}+ reference segments of the two sets:",
        chosen,
    )

    d = starfile.read(a.class2d)
    d = d["particles"] if isinstance(d, dict) else d
    d = d[d["rlnClassNumber"].astype(int).isin(chosen)].copy()
    print(f"{len(d):,} segments in those classes")
    r = ph.analyze(d, class_ids=chosen, n_boot=2)
    print(
        f"pooled repeat {r.period:.1f} A, double_ratio {r.double_ratio:.2f}, "
        f"{r.n_filaments} filaments, {len(r.class_ids)} classes after merging"
    )

    ref["fil"] = (
        ref["rlnMicrographName"].astype(str) + "|" + ref["rlnHelicalTubeID"].astype(str)
    )
    share = ref.groupby("fil")[[a.set_a, a.set_b]].sum()
    share["a"] = share[a.set_a] / (share[a.set_a] + share[a.set_b]).clip(lower=1)
    share["n"] = share[a.set_a] + share[a.set_b]
    fil = r.filaments.copy()
    fil["fil"] = [Path(str(m)).stem + "|" + str(t) for m, t in r.filament_keys]
    fil = fil.merge(share, left_on="fil", right_index=True, how="inner")
    fil = fil[fil["fitted"] & (fil["n"] >= 5)]
    A = fil[fil["a"] >= 0.8]["pitch"].to_numpy()
    B = fil[fil["a"] <= 0.2]["pitch"].to_numpy()
    print(
        f"filaments mostly {a.set_a}: {len(A)}, median repeat {np.median(A):.0f} A (IQR {np.percentile(A, 25):.0f}-{np.percentile(A, 75):.0f})"
    )
    print(
        f"filaments mostly {a.set_b}: {len(B)}, median repeat {np.median(B):.0f} A (IQR {np.percentile(B, 25):.0f}-{np.percentile(B, 75):.0f})"
    )
    if len(A) and len(B):
        print(f"AUC of the per-filament repeat for telling them apart: {auc(A, B):.3f}")


if __name__ == "__main__":
    main()
