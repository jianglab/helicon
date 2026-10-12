"""Write the binned micrographs of the benchmark selection, for tools that run
outside Python (Topaz-Denoise): ``OUT/binned_<dataset>/<name>.mrc``.

    python bin_micrographs.py --dataset 10940 --n 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mrcfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_baselines as R  # noqa: E402
import truth as T  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--dataset", default="10230", choices=sorted(T.BIN))
    p.add_argument("--bin", type=int, default=None, help="default: per dataset")
    a = p.parse_args()
    a.bin = a.bin or T.BIN[a.dataset]
    out = R.OUT / f"binned_{a.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    for m in T.load(a.dataset, a.n):
        with mrcfile.open(m.path, permissive=True) as f:
            img = np.asarray(f.data, float)
            apix = float(f.voxel_size.x) * a.bin
        with mrcfile.new(out / f"{m.name}.mrc", overwrite=True) as g:
            g.set_data(R.bin_image(img, a.bin).astype(np.float32))
            g.voxel_size = apix
    print(len(list(out.glob("*.mrc"))), "micrographs in", out)


if __name__ == "__main__":
    main()
