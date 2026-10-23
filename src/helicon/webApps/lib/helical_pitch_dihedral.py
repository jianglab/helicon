"""Test for a two-fold axis perpendicular to the helical axis (dihedral symmetry).

A perpendicular dyad at azimuth ``alpha`` makes the density satisfy
``rho(R, psi, z) = rho(R, 2*alpha - psi, 2*z0 - z)``: the view of a filament at
azimuth ``theta`` is the mirror image, along the filament, of the view at
``2*alpha - theta``. The class averages need not be centred on the dyad: only
their ring azimuths and straightened averages (filament horizontal) are used.
"""

import numpy as np


def _mirror_cc(a, b, max_shift):
    """Best correlation of ``a`` flipped along the filament with ``b``."""
    flipped = a[:, ::-1]
    y = b - b.mean()
    ny = np.sqrt((y * y).sum())
    best = -1.0
    for s in range(-max_shift, max_shift + 1):
        x = np.roll(flipped, s, axis=1)
        x = x - x.mean()
        d = np.sqrt((x * x).sum()) * ny
        if d > 0:
            best = max(best, float((x * y).sum() / d))
    return best


def dihedral_scan(images, azimuths, period=360.0, step=2.5, max_shift=None):
    """Score every azimuth of a perpendicular two-fold axis.

    Parameters
    ----------
    images : sequence of 2D arrays
        Class averages with the filament horizontal (all the same size).
    azimuths : sequence of float
        Azimuth of each average about the helical axis, in degrees.
    period : float
        Azimuth period in degrees (360 / C for C-fold symmetry about the axis).
    step : float
        Step of the scanned dyad azimuth, in degrees.
    max_shift : int, optional
        Largest axial shift, in pixels, tried between a view and its partner
        (the dyad is not at the middle of the average). Default: a fifth of the
        width.

    Returns
    -------
    dict
        ``alpha`` (scanned azimuths, within ``period / 2``), ``score`` (mean
        mirror correlation over the paired classes), ``best_alpha``,
        ``best_score``, ``contrast`` (best minus the median score: a dyad
        gives a sharp peak, a plain helix a flat curve) and ``pairs`` (for the
        best azimuth: ``(i, j, cc)`` of each class and its partner).
    """
    az = np.asarray(azimuths, float) % period
    imgs = [np.asarray(im, np.float32) for im in images]
    n = len(imgs)
    shape = np.min([im.shape for im in imgs], axis=0)
    imgs = [im[: shape[0], : shape[1]] for im in imgs]
    if max_shift is None:
        max_shift = max(1, int(shape[1]) // 5)
    gap = (
        period / n
    )  # a partner farther than this from the wanted azimuth is no partner
    cache = {}

    def cc(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = _mirror_cc(imgs[i], imgs[j], max_shift)
        return cache[(i, j)]

    def pairs_at(alpha):
        out = []
        for i in range(n):
            off = ((az - (2 * alpha - az[i])) + period / 2) % period - period / 2
            j = int(np.argmin(np.abs(off)))
            if abs(off[j]) <= gap:
                out.append((i, j, cc(i, j)))
        return out

    alphas = np.arange(0.0, period / 2 + 1e-9, step)
    score = np.full(len(alphas), np.nan)
    for k, alpha in enumerate(alphas):
        p = pairs_at(alpha)
        if len(p) >= n / 2:
            score[k] = np.mean([c for _, _, c in p])
    if np.isnan(score).all():
        return dict(
            alpha=alphas,
            score=score,
            best_alpha=None,
            best_score=None,
            contrast=0.0,
            pairs=[],
        )
    k = int(np.nanargmax(score))
    return dict(
        alpha=alphas,
        score=score,
        best_alpha=float(alphas[k]),
        best_score=float(score[k]),
        contrast=float(score[k] - np.nanmedian(score)),
        pairs=pairs_at(alphas[k]),
    )
