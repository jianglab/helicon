"""Isotropic controls and a joint sweep of oriented.enhance (scratch)."""

import sys, time, os, itertools
from pathlib import Path
import mrcfile, numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M, truth as T, oriented as O

n = int(sys.argv[1])
lo, hi = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, n)
data = {}
for ds in ("10230", "10940"):
    binned = Path(os.environ["DENOISE_BENCH_OUT"]) / f"binned_{ds}"
    for m in T.load(ds, 40)[lo:hi]:
        with mrcfile.open(binned / f"{m.name}.mrc", permissive=True) as f:
            img = np.asarray(f.data, np.float32)
            apix = float(f.voxel_size.x)
        data.setdefault(ds, []).append(
            (img, apix, M.distance_map(img.shape, m.paths, T.BIN[ds], apix))
        )


def score(fn):
    return {
        ds: np.mean([M.ridge_auc(fn(i, a), d, a) for i, a, d in items])
        for ds, items in data.items()
    }


def show(name, r):
    print(
        f"{name:34s} 10230 {r['10230']:.4f}  10940 {r['10940']:.4f}  mean {(r['10230']+r['10940'])/2:.4f}",
        flush=True,
    )


mode = sys.argv[4] if len(sys.argv) > 4 else "all"
if mode in ("all", "controls"):
    for sigma in (15.0, 21.0, 28.0, 35.0, 45.0, 60.0):
        show(
            f"isotropic gaussian sigma {sigma:.0f} A",
            score(lambda im, a, s=sigma: ndi.gaussian_filter(im, s / a)),
        )
if mode in ("all", "joint"):
    for across, along, sharp in itertools.product(
        (24.0, 28.0, 34.0, 40.0), (80.0, 120.0), (1.0, 2.0)
    ):
        show(
            f"across {across:.0f} along {along:.0f} sharp {sharp:.0f}",
            score(
                lambda im, a, kw=dict(
                    across=across, along=along, sharp=sharp
                ): O.enhance(im, a, **kw)
            ),
        )
