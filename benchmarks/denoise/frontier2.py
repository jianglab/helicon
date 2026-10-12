"""Detection at equal sharpness: each method against the Gaussian curve at its own width.
python frontier2.py FIRST LAST"""

import sys, os
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
            (img, apix, M.distance_map(img.shape, m.paths, T.BIN[ds], apix), t, m.paths)
        )


def measure(fn):
    res = {}
    for ds, items in data.items():
        rows = []
        for i, a, d, t, paths in items:
            o = fn(i, a, t)
            if o is None:
                continue
            far, near = M.ridge_aucs(o, d, a)
            rows.append((M.profile_fwhm(o, paths, T.BIN[ds], a), far))
        rows = np.array(rows)
        res[ds] = np.nanmean(rows, axis=0)
    return res


curve = {ds: [] for ds in data}
for s in (10.0, 15.0, 21.0, 28.0, 35.0, 45.0, 60.0, 80.0):
    r = measure(lambda i, a, t, s=s: ndi.gaussian_filter(i, s / a))
    for ds in data:
        curve[ds].append(r[ds])
    print(
        f"gaussian sigma {s:3.0f} A   "
        + "  ".join(f"{ds}: width {r[ds][0]:5.0f} A far {r[ds][1]:.3f}" for ds in data),
        flush=True,
    )
curve = {ds: np.array(sorted(c, key=lambda r: r[0])) for ds, c in curve.items()}


def gain(name, fn):
    r = measure(fn)
    parts = []
    for ds in data:
        w, far = r[ds]
        ref = np.interp(w, curve[ds][:, 0], curve[ds][:, 1])
        parts.append(
            f"{ds}: width {w:5.0f} A far {far:.3f} gaussian at that width {ref:.3f} gain {far - ref:+.3f}"
        )
    print(f"{name:34s} " + "  ".join(parts), flush=True)


gain("raw", lambda i, a, t: i)
gain("topaz U-Net", lambda i, a, t: t)
for across, along, sharp in (
    (16, 80, 1),
    (20, 80, 1),
    (20, 150, 4),
    (28, 80, 1),
    (28, 120, 2),
    (34, 80, 1),
):
    gain(
        f"oriented {across}/{along}/{sharp}",
        lambda i, a, t, kw=dict(
            across=float(across), along=float(along), sharp=float(sharp)
        ): O.enhance(i, a, **kw),
    )
