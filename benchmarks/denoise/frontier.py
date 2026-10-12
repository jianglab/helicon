"""Score methods on the far and near AUCs (scratch). python frontier.py N_FIRST N_LAST"""

import sys, os, time
from pathlib import Path
import mrcfile, numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M, truth as T, oriented as O

lo, hi = int(sys.argv[1]), int(sys.argv[2])
out = Path(os.environ["DENOISE_BENCH_OUT"])
data = {}
for ds in ("10230", "10940"):
    for m in T.load(ds, 40)[lo:hi]:
        with mrcfile.open(out / f"binned_{ds}" / f"{m.name}.mrc", permissive=True) as f:
            img = np.asarray(f.data, np.float32)
            apix = float(f.voxel_size.x)
        topaz = out / f"topaz_{ds}" / f"{m.name}.mrc"
        t = None
        if topaz.exists():
            with mrcfile.open(topaz, permissive=True) as g:
                t = np.asarray(g.data, np.float32)
        data.setdefault(ds, []).append(
            (img, apix, M.distance_map(img.shape, m.paths, T.BIN[ds], apix), t)
        )


def run(name, fn):
    res = {}
    for ds, items in data.items():
        v = []
        for i, a, d, t in items:
            o = fn(i, a, t)
            if o is not None:
                v.append(M.ridge_aucs(o, d, a))
        res[ds] = np.mean(v, axis=0)
    print(
        f"{name:36s} 10230 far {res['10230'][0]:.3f} near {res['10230'][1]:.3f} | 10940 far {res['10940'][0]:.3f} near {res['10940'][1]:.3f}",
        flush=True,
    )


run("raw", lambda i, a, t: i)
for s in (15.0, 21.0, 28.0, 35.0, 45.0, 60.0):
    run(f"gaussian sigma {s:.0f} A", lambda i, a, t, s=s: ndi.gaussian_filter(i, s / a))
run("topaz U-Net", lambda i, a, t: t)
for across, along, sharp in (
    (20, 150, 4),
    (20, 80, 1),
    (28, 80, 1),
    (28, 120, 2),
    (34, 80, 1),
    (40, 120, 2),
):
    run(
        f"oriented across {across} along {along} sharp {sharp}",
        lambda i, a, t, kw=dict(
            across=float(across), along=float(along), sharp=float(sharp)
        ): O.enhance(i, a, **kw),
    )
