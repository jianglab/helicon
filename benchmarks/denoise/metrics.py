"""Scores of a micrograph against the picked filament paths.

Two measures that do not depend on the denoiser's own transform:

* ``contrast``: how far the filaments stand from the background in units of the
  background's own noise (a contrast-to-noise ratio);
* ``ridge_auc`` (``far``) and ``near_auc``: how well a plain ridge detector, run on the image, separates
  pixels on the filaments from pixels away from every filament (the area under
  the ROC curve, 0.5 for chance). It is the part of picking that the image,
  not the picker, decides. The two differ in what the filament is compared with:
  ``far``, pixels 150-400 A from every filament (is a filament there?), and
  ``near``, pixels 80-140 A from it, just outside (where is its axis?). A heavy
  blur raises ``far`` and lowers ``near``: it turns a filament into a broad dark
  band, easy to see and hard to place. A denoiser is better only if it moves
  both up, so the two are always reported together.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as draw_line

# Å from a path: on the filament, clearly off every filament, and just outside it
CORE = 20.0
FAR_MIN, FAR_MAX = 150.0, 400.0
NEAR_MIN, NEAR_MAX = 80.0, 140.0


def distance_map(shape, paths, scale: float, apix: float) -> np.ndarray:
    """Distance (Å) of every pixel from the nearest filament path.

    Parameters
    ----------
    shape : tuple of int
        ``(ny, nx)`` of the image.
    paths : list of numpy.ndarray
        Filament paths in pixels of the *original* micrograph.
    scale : float
        Original pixels per image pixel (the binning).
    apix : float
        Pixel size of the image in Å.
    """
    ny, nx = shape
    mask = np.zeros(shape, bool)
    for p in paths:
        q = np.asarray(p, float) / scale
        for a, b in zip(q[:-1], q[1:]):
            x0, y0, x1, y1 = (int(round(v)) for v in (*a, *b))
            rr, cc = draw_line(y0, x0, y1, x1)
            ok = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
            mask[rr[ok], cc[ok]] = True
    if not mask.any():
        return np.full(shape, np.inf)
    return ndi.distance_transform_edt(~mask) * apix


def _robust_std(x: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def contrast(image, dist) -> float:
    """CNR of the filaments: |mean on - mean off| / robust std off."""
    on = dist < CORE
    off = (dist > FAR_MIN) & (dist < FAR_MAX)
    if on.sum() < 50 or off.sum() < 50:
        return float("nan")
    sd = _robust_std(image[off])
    return float(abs(image[on].mean() - image[off].mean()) / max(sd, 1e-12))


def ridge_response(image, sigma_px: float) -> np.ndarray:
    """Positive curvature across a dark ridge: the larger Hessian eigenvalue.

    Filaments are dark in these micrographs; a line is a ridge of the
    Gaussian-smoothed image with a positive second derivative across it.
    """
    g = ndi.gaussian_filter(np.asarray(image, float), sigma_px)
    gy, gx = np.gradient(g)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    hxy = 0.5 * (gxy + gyx)
    tr, det = gxx + gyy, gxx * gyy - hxy**2
    lam = 0.5 * tr + np.sqrt(np.maximum(0.25 * tr**2 - det, 0.0))
    return np.maximum(lam, 0.0) * sigma_px**2


def roc_auc(score_pos: np.ndarray, score_neg: np.ndarray) -> float:
    """Area under the ROC curve by the rank-sum statistic (ties count half)."""
    if len(score_pos) == 0 or len(score_neg) == 0:
        return float("nan")
    allv = np.concatenate([score_pos, score_neg])
    ranks = _rank(allv)
    n1, n0 = len(score_pos), len(score_neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    # average the ranks of ties
    edges = np.flatnonzero(np.diff(xs) != 0) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(x)]):
        if hi - lo > 1:
            ranks[order[lo:hi]] = 0.5 * (lo + 1 + hi)
    return ranks


def ridge_auc(image, dist, apix: float, width: float = 100.0, n=200_000, seed=0):
    """AUC of the ridge detector for on-filament against far-from-filament pixels."""
    resp = ridge_response(image, 0.25 * width / apix)
    rng = np.random.default_rng(seed)
    pos = resp[dist < CORE]
    neg = resp[(dist > FAR_MIN) & (dist < FAR_MAX)]
    if len(pos) > n:
        pos = rng.choice(pos, n, replace=False)
    if len(neg) > n:
        neg = rng.choice(neg, n, replace=False)
    return roc_auc(pos, neg)


def ridge_aucs(image, dist, apix: float, width: float = 100.0, n=200_000, seed=0):
    """``(far, near)``: the AUC of the ridge detector for on-filament pixels
    against pixels far from, and just outside, every filament."""
    resp = ridge_response(image, 0.25 * width / apix)
    rng = np.random.default_rng(seed)

    def pick(x):
        return rng.choice(x, n, replace=False) if len(x) > n else x

    pos = pick(resp[dist < CORE])
    far = pick(resp[(dist > FAR_MIN) & (dist < FAR_MAX)])
    near = pick(resp[(dist > NEAR_MIN) & (dist < NEAR_MAX)])
    return roc_auc(pos, far), roc_auc(pos, near)


def profile_fwhm(image, paths, scale: float, apix: float, reach=200.0, step=None):
    """Width (A, full width at half depth) of the mean dark profile across filaments.

    The image is sampled along the normals of every picked path, the profiles are
    averaged (the two sides together) and the width of the dip below the level
    ``reach`` away is measured. It says how sharp a denoiser leaves the
    filaments: a blur widens it, and what a method gains in detection by
    blurring more is not an improvement. Methods are compared at one width.

    Returns
    -------
    float
        The width, NaN if no path lies inside the image or there is no dip.
    """
    step = step or apix / 2.0
    ny, nx = image.shape
    offsets = np.arange(-reach, reach + 1e-9, step)
    total = np.zeros(len(offsets))
    count = 0
    for path in paths:
        q = np.asarray(path, float) / scale
        if len(q) < 2:
            continue
        # points about one pixel apart, along the polyline
        seg = np.hypot(*np.diff(q, axis=0).T)
        s = np.r_[0.0, np.cumsum(seg)]
        t = np.arange(0.0, s[-1], 1.0)
        if len(t) < 6:
            continue
        x, y = np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])
        tx = np.gradient(ndi.uniform_filter1d(x, 7))
        ty = np.gradient(ndi.uniform_filter1d(y, 7))
        norm = np.hypot(tx, ty)
        ok = norm > 0
        nxn, nyn = -ty[ok] / norm[ok], tx[ok] / norm[ok]
        px = x[ok][:, None] + nxn[:, None] * offsets[None, :] / apix
        py = y[ok][:, None] + nyn[:, None] * offsets[None, :] / apix
        inside = (px >= 0) & (px <= nx - 1) & (py >= 0) & (py <= ny - 1)
        good = inside.all(axis=1)
        if not good.any():
            continue
        vals = ndi.map_coordinates(
            np.asarray(image, float), [py[good], px[good]], order=1
        )
        total += vals.sum(axis=0)
        count += int(good.sum())
    if count < 50:
        return float("nan")
    prof = total / count
    prof = 0.5 * (prof + prof[::-1])
    mid = len(offsets) // 2
    base = np.mean(prof[np.abs(offsets) > 0.6 * reach])
    depth = base - prof[mid - 2 : mid + 3].min()
    if depth <= 0:
        return float("nan")
    below = prof < base - depth / 2.0
    right = mid
    while right + 1 < len(prof) and below[right + 1]:
        right += 1
    if right + 1 >= len(prof):
        return float("nan")
    # linear interpolation of the half-depth crossing
    level = base - depth / 2.0
    f = (level - prof[right]) / (prof[right + 1] - prof[right] + 1e-30)
    half = (right + f - mid) * step
    return float(2.0 * half)


def _path_mask(shape, paths, scale):
    """Pixels on the picked paths (a boolean image)."""
    ny, nx = shape
    mask = np.zeros(shape, bool)
    for p in paths:
        q = np.asarray(p, float) / scale
        for a, b in zip(q[:-1], q[1:]):
            x0, y0, x1, y1 = (int(round(v)) for v in (*a, *b))
            rr, cc = draw_line(y0, x0, y1, x1)
            ok = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
            mask[rr[ok], cc[ok]] = True
    return mask


def pick_scores(
    image, paths, scale, apix, width=100.0, tol=30.0, min_length=150.0, levels=24
):
    """Centrelines found by one fixed procedure, against the picked paths.

    The ridge response (the same detector as :func:`ridge_auc`) is thresholded
    with hysteresis (a component is kept if it reaches the high level, and
    extends to half of it), thinned to a skeleton, and components shorter than
    ``min_length`` A dropped. Precision is the share of the found centreline
    within ``tol`` A of a picked path; recall is the share of the picked paths
    within ``tol`` A of a found centreline. This is closer to picking than the
    AUCs: a method must make filaments continuous (so that they survive the
    length filter) without bridging neighbouring ones or inventing lines.

    The high level is swept over ``levels`` quantiles of the response; the
    best F1 and the average precision (area under the recall-precision curve)
    are returned.

    Returns
    -------
    dict
        ``f1`` (best), ``ap`` and ``recall_at_p80`` (the recall where the
        precision first falls below 0.8, going from strict to lenient).
    """
    from skimage.morphology import skeletonize

    resp = ridge_response(image, 0.25 * width / apix)
    truth = _path_mask(image.shape, paths, scale)
    if not truth.any():
        return dict(f1=float("nan"), ap=float("nan"), recall_at_p80=float("nan"))
    d_truth = ndi.distance_transform_edt(~truth) * apix
    tol_truth = d_truth <= tol
    n_truth = int(truth.sum())
    min_px = max(3, int(round(min_length / apix)))
    qs = np.quantile(resp[resp > 0], np.linspace(0.995, 0.80, levels))
    pts = []
    for high in qs:
        low = 0.5 * high
        lab, n = ndi.label(resp > low, structure=np.ones((3, 3)))
        if n == 0:
            continue
        top = ndi.maximum(resp, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = top >= high
        sk = skeletonize(keep[lab])
        lab2, n2 = ndi.label(sk, structure=np.ones((3, 3)))
        if n2 == 0:
            continue
        size = np.bincount(lab2.ravel())
        ok = size >= min_px
        ok[0] = False
        sk = ok[lab2]
        if not sk.any():
            continue
        precision = float(tol_truth[sk].mean())
        d_found = ndi.distance_transform_edt(~sk) * apix
        recall = float((d_found[truth] <= tol).mean())
        pts.append((recall, precision))
    if not pts:
        return dict(f1=0.0, ap=0.0, recall_at_p80=0.0)
    pts = np.array(pts)
    f1 = float(np.max(2 * pts[:, 0] * pts[:, 1] / np.maximum(pts.sum(1), 1e-12)))
    order = np.argsort(pts[:, 0])
    r, p = pts[order, 0], pts[order, 1]
    ap = float(np.trapezoid(np.maximum.accumulate(p[::-1])[::-1], r))
    hi = pts[pts[:, 1] >= 0.8]
    return dict(f1=f1, ap=ap, recall_at_p80=float(hi[:, 0].max()) if len(hi) else 0.0)
