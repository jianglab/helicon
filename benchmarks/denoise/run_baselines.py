"""Score the baseline micrograph filters on the curated filament paths.

Each micrograph is binned, filtered by each method and scored by ``metrics``
(contrast of the filaments over the background noise, and the AUC of a plain
ridge detector). Results go to a CSV in ``OUT``.

    python run_baselines.py --dataset 10940 --n 12
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
import truth as T  # noqa: E402

OUT = Path(os.environ.get("DENOISE_BENCH_OUT", "denoise_bench_out"))
OUT.mkdir(parents=True, exist_ok=True)


def bin_image(img: np.ndarray, f: int) -> np.ndarray:
    ny, nx = (s // f * f for s in img.shape)
    return img[:ny, :nx].reshape(ny // f, f, nx // f, f).mean((1, 3))


def methods(apix: float) -> dict:
    """name -> function(image) for the filters compared."""
    from helicon.lib import curvelet as C

    def gauss(width):
        return lambda im: ndi.gaussian_filter(im, width / apix / 2.355)

    return {
        "raw": lambda im: im,
        "gauss20": gauss(20.0),
        "gauss50": gauss(50.0),
        "udct": lambda im: C.curvelet_denoise_udct(im, pixel_size=apix),
        "mct": lambda im: C.curvelet_denoise_mct(im, pixel_size=apix),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=12, help="micrographs, chosen at random")
    p.add_argument("--dataset", default="10230", choices=sorted(T.BIN))
    p.add_argument("--bin", type=int, default=None, help="default: per dataset")
    p.add_argument("--only", default="", help="comma-separated method names")
    p.add_argument(
        "--external",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="also score images denoised elsewhere: DIR/<micrograph>.mrc (binned)",
    )
    a = p.parse_args()

    a.bin = a.bin or T.BIN[a.dataset]
    mics = T.load(a.dataset, a.n)
    rows = []
    for k, m in enumerate(mics):
        with mrcfile.open(m.path, permissive=True) as f:
            img = np.asarray(f.data, float)
            apix0 = float(f.voxel_size.x)
        img = bin_image(img, a.bin)
        apix = apix0 * a.bin
        dist = M.distance_map(img.shape, m.paths, a.bin, apix)
        meth = methods(apix)
        if a.only:
            meth = {n: meth[n] for n in a.only.split(",")}
        for spec in a.external:
            name, folder = spec.split("=", 1)
            done = Path(folder) / f"{m.name}.mrc"
            if done.exists():
                with mrcfile.open(done, permissive=True) as g:
                    meth[name] = lambda im, d=np.asarray(g.data, float): d
        for name, fn in meth.items():
            t0 = time.time()
            out = fn(img)
            rows.append(
                dict(
                    micrograph=m.name,
                    method=name,
                    contrast=M.contrast(out, dist),
                    ridge_auc=M.ridge_auc(out, dist, apix),
                    seconds=time.time() - t0,
                )
            )
        print(k + 1, len(mics), m.name, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(
        OUT / f"baselines_{a.dataset}.csv", index=False
    )  # one row per image and method
    print(
        df.groupby("method")[["contrast", "ridge_auc", "seconds"]]
        .mean()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
