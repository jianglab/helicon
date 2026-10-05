"""Save the per-filament repeats of a real data set (for looking for two types)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_10940 as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--job-dir", required=True)
ap.add_argument("--iteration", type=int, default=25)
ap.add_argument("--rule", default="all")
ap.add_argument("--out", required=True)
a = ap.parse_args()
params, images, apix = R.load(a.job_dir, a.iteration)
sels = R.selections(params, images)
result = R.merged_classes(params, images, apix, sels[a.rule])
fil = result.filaments.copy()
fil.to_csv(a.out, index=False)
print(
    "pooled repeat",
    result.period,
    "n filaments",
    len(fil),
    "fitted",
    int(fil.fitted.sum()),
)
print("heterogeneity", result.heterogeneity)
