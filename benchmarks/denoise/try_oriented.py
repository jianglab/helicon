"""Score the along-the-filament smoothing against the baselines (scratch)."""

import sys, time, os
from pathlib import Path
import mrcfile, numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M, truth as T, oriented as O
from scipy import ndimage as ndi

ds, n = sys.argv[1], int(sys.argv[2])
binned = Path(os.environ["DENOISE_BENCH_OUT"]) / f"binned_{ds}"
mics = T.load(ds, 40)[:n]


def intensity_auc(img, dist):
    return M.roc_auc(-img[dist < M.CORE], -img[(dist > M.FAR_MIN) & (dist < M.FAR_MAX)])


rows = []
for k, m in enumerate(mics):
    with mrcfile.open(binned / f"{m.name}.mrc", permissive=True) as f:
        img = np.asarray(f.data, float)
        apix = float(f.voxel_size.x)
    dist = M.distance_map(img.shape, m.paths, T.BIN[ds], apix)
    methods = {
        "raw": lambda: img,
        "gauss50": lambda: ndi.gaussian_filter(img, 50 / apix / 2.355),
    }
    for along in (150.0, 300.0, 600.0):
        methods[f"along{int(along)}"] = lambda a=along: O.enhance(
            img, apix, across=20.0, along=a
        )
    methods["ridge300"] = lambda: O.ridge_map(img, apix, across=25.0, along=300.0)[0]
    for name, fn in methods.items():
        t0 = time.time()
        out = fn()
        dt = time.time() - t0
        rows.append(
            dict(
                method=name,
                contrast=M.contrast(out, dist),
                ridge_auc=M.ridge_auc(out, dist, apix),
                intensity_auc=intensity_auc(out, dist),
                seconds=dt,
            )
        )
    print(k + 1, len(mics), flush=True)
df = pd.DataFrame(rows)
print(ds, n, "micrographs")
print(
    df.groupby("method")[["contrast", "ridge_auc", "intensity_auc", "seconds"]]
    .mean()
    .round(3)
    .to_string()
)
