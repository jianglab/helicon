"""Joint multi-image helical parameter search.

A single 2D class average determines the helical twist unreliably: measured on
ten good class averages of EMPIAR-10940 only 5/10 peaked at the correct value,
with score margins of 0.001-0.002, i.e. comparable to noise. Since every class
average of a given filament shares the same twist/rise, combining their score
curves resolves this -- the joint peak margin was ~50x the single-image margin
and correct for every subset size tested.

Two refinements matter and are implemented here:

* the curves are z-scored before averaging, because classes differ in both score
  scale and score spread, so a plain average is dominated by whichever image has
  the largest dynamic range (measured 65x smaller margin);
* the z-score uses a shrinkage floor, so an uninformative (flat) curve is
  downweighted instead of having its noise amplified to unit variance.

The floor is a fraction of the *median* curve spread, and that choice has a
known limitation: it protects against a loud outlier but only while flat curves
are a minority. Measured with three informative curves, flat curves get weight
0.01-0.03 and the answer stays right up to three of them, but at four the median
spread becomes the flat value, the floor collapses, flat curves get weight 0.78,
and the joint peak goes wrong. Scaling the floor to the *largest* spread instead
fixes that case and breaks the more important one -- a single class average with
wild scores then outvotes every good image. Since the user selects which classes
to include, protecting against one bad selection is worth more than tolerating a
majority of them, so this is left as is rather than made adaptive.

Refining each image's psi and dy against a reference built from the others was
tried and deliberately left out. Measured with an estimator independent of the
one that produced the transform (slope of the intensity-weighted row centroid,
so the comparison is not circular), it helps raw class averages -- mean |tilt|
5.48 -> 2.98 deg -- but the tab's own auto-transform already reaches 0.07 deg,
and refining those images pushes them back out to 0.72 deg (worst 0.16 -> 2.17),
replicated at 8 and 16 images. The cause is reference bias: the images end up
mutually consistent (match correlation 0.89) while collectively rotated away
from horizontal. So the match correlation must never be used as a convergence
criterion -- it improves while the answer gets worse.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

SHRINK = 0.25


# ──────────────────────────────────────────────────────────────────────────
# Joint scoring
# ──────────────────────────────────────────────────────────────────────────


def combine_score_curves(curves, shrink=SHRINK):
    """Combine per-image score curves into one joint curve.

    Parameters
    ----------
    curves : sequence of 1-D arrays
        One score curve per image, all sampled on the same parameter grid.
    shrink : float
        Shrinkage floor, as a fraction of the median curve spread. Curves much
        flatter than the median contribute proportionally less.

    Returns
    -------
    (combined, weights)
        ``combined`` is the joint curve; ``weights`` has one value per input
        curve (near 1 for informative, near 0 for flat) and is worth reporting
        so the user can see which selections carried the answer.
    """
    C = np.asarray(curves, dtype=float)
    if C.ndim == 1:
        C = C[None, :]
    sd = np.nanstd(C, axis=1, keepdims=True)
    med = float(np.nanmedian(sd))
    eps = shrink * med if np.isfinite(med) and med > 0 else 1e-12
    Z = (C - np.nanmean(C, axis=1, keepdims=True)) / (sd + eps)
    weights = (sd.ravel() / (sd.ravel() + eps)).astype(float)
    return np.nanmean(Z, axis=0), weights


def joint_best(curves, params, shrink=SHRINK):
    """Pick the parameter combination favoured by the combined curve.

    ``params`` lists the parameter values (or tuples) matching the curve
    samples. Returns (best_param, margin_over_runner_up, combined, weights).
    """
    combined, weights = combine_score_curves(curves, shrink)
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return None, 0.0, combined, weights
    i = int(np.nanargmax(combined))
    ordered = np.sort(finite)
    margin = float(ordered[-1] - ordered[-2]) if ordered.size > 1 else 0.0
    return params[i], margin, combined, weights


def combine_results(results, shrink=SHRINK):
    """Re-rank per-image solver results into one joint ranking.

    ``results`` is the flat list produced by running every (image, twist, rise)
    task, each entry ``(score, return_data, params)`` in the shape
    ``denovo3d_pipeline.process_one_task`` returns, where ``params[2]`` is the
    image label and ``params[5:7]`` are twist and rise.

    Only (twist, rise) pairs scored by *every* image are ranked, so a task that
    was skipped or raised cannot make its pair look artificially good. Returns a
    list in the same ``(score, return_data, params)`` shape -- score replaced by
    the combined z-score, and the payload taken from the image that scored the
    pair highest -- plus the per-image weights keyed by image label.
    """
    by_pair = {}
    for r in results:
        params = r[2]
        key = (round(float(params[5]), 6), round(float(params[6]), 6))
        by_pair.setdefault(key, {})[params[2]] = r

    images = sorted({k for v in by_pair.values() for k in v})
    pairs = sorted(k for k, v in by_pair.items() if len(v) == len(images))
    if not pairs or len(images) < 2:
        return sorted(results, key=lambda x: x[0], reverse=True), {}

    curves = [[by_pair[p][im][0] for p in pairs] for im in images]
    combined, weights = combine_score_curves(curves, shrink)

    joint = []
    for p, c in zip(pairs, combined):
        best = max(by_pair[p].values(), key=lambda r: r[0])
        joint.append((float(c), best[1], best[2]))
    joint.sort(key=lambda x: x[0], reverse=True)
    return joint, dict(zip(images, weights))
