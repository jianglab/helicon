"""Centreline picking scores of each method (scratch). python picking.py FIRST LAST"""

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
        data.setdefault(ds, []).append((img, apix, t, m.paths))


def run(name, fn):
    parts = []
    for ds, items in data.items():
        rows = []
        for i, a, t, paths in items:
            o = fn(i, a, t)
            if o is None:
                continue
            r = M.pick_scores(o, paths, T.BIN[ds], a)
            rows.append((r["f1"], r["ap"], r["recall_at_p80"]))
        v = np.nanmean(np.array(rows), axis=0)
        parts.append(f"{ds}: F1 {v[0]:.3f} AP {v[1]:.3f} recall@P80 {v[2]:.3f}")
    print(f"{name:30s} " + " | ".join(parts), flush=True)


run("raw", lambda i, a, t: i)
for s in (15.0, 21.0, 28.0, 35.0, 45.0, 60.0, 80.0):
    run(f"gaussian sigma {s:.0f} A", lambda i, a, t, s=s: ndi.gaussian_filter(i, s / a))
run("topaz U-Net", lambda i, a, t: t)
for across, along, sharp in (
    (20, 150, 4),
    (20, 80, 1),
    (20, 250, 4),
    (28, 150, 4),
    (28, 250, 8),
):
    run(
        f"oriented {across}/{along}/{sharp}",
        lambda i, a, t, kw=dict(
            across=float(across), along=float(along), sharp=float(sharp)
        ): O.enhance(i, a, **kw),
    )
