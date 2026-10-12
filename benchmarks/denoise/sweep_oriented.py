"""One-parameter-at-a-time sweep of oriented.enhance (scratch)."""

import sys, time, os
from pathlib import Path
import mrcfile, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M, truth as T, oriented as O

n = int(sys.argv[1])
base = dict(across=20.0, along=150.0, n_angles=24, sharp=4.0, mix=1.0)
grid = {
    "along": [80.0, 110.0, 150.0, 200.0],
    "across": [12.0, 16.0, 20.0, 28.0],
    "n_angles": [12, 18, 24, 36],
    "sharp": [1.0, 2.0, 4.0, 8.0, 16.0],
    "mix": [0.5, 0.75, 1.0],
}
data = {}
for ds in ("10230", "10940"):
    binned = Path(os.environ["DENOISE_BENCH_OUT"]) / f"binned_{ds}"
    for m in T.load(ds, 40)[:n]:
        with mrcfile.open(binned / f"{m.name}.mrc", permissive=True) as f:
            img = np.asarray(f.data, np.float32)
            apix = float(f.voxel_size.x)
        data.setdefault(ds, []).append(
            (img, apix, M.distance_map(img.shape, m.paths, T.BIN[ds], apix))
        )


def score(kw):
    out = {}
    for ds, items in data.items():
        out[ds] = np.mean(
            [M.ridge_auc(O.enhance(i, a, **kw), d, a) for i, a, d in items]
        )
    return out


t0 = time.time()
ref = score(base)
print(
    f"{'base':22s} 10230 {ref['10230']:.4f}  10940 {ref['10940']:.4f}  ({time.time()-t0:.0f} s)",
    flush=True,
)
for name, values in grid.items():
    for v in values:
        kw = dict(base, **{name: v})
        r = score(kw)
        print(
            f"{name}={v!s:<10s}          10230 {r['10230']:.4f}  10940 {r['10940']:.4f}  mean {(r['10230']+r['10940'])/2:.4f}",
            flush=True,
        )
