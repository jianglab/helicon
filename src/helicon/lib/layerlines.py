"""Layer lines of a helix, and a filter that keeps only them.

The 2D projection of a helix has its Fourier transform on *layer lines*: lines
perpendicular to the helical axis at the heights

    Y = m / rise + n / pitch,        pitch = 360 * rise / |twist|

with ``m`` any integer and ``n`` a multiple of the rotational symmetry ``csym``
(the order of the Bessel function ``J_n`` that gives the line its intensity). On
the layer line of order ``n`` the transform is ``J_n(2 pi R X)`` for a thin shell
of radius ``R``, which is negligible closer to the axis than about ``n / (2 pi R)``
and oscillates beyond. Noise has no such structure: it fills the plane evenly. A
mask that keeps the neighbourhood of the layer lines, and of each only where it
can carry signal, takes out the noise between them without touching the signal.

How much is taken out depends on the helix. A thin helix with few subunits per
turn has widely spaced lines and loses most of the noise; one of large radius,
small twist and high resolution (an amyloid at 1 A/pixel) has so many lines in
so small a space that the mask covers a good part of the plane and gains little.
:func:`layer_line_filter` reports the share of white-noise variance that remains.

The axis is along the rows of the image (axis 0), as in HILL.
"""

from __future__ import annotations

import functools

import numpy as np

# below this share of its first maximum, the Bessel function J_n is taken as zero
BESSEL_FLOOR = 0.05


def pitch(twist: float, rise: float) -> float:
    """The pitch in A of a helix of ``twist`` degrees and ``rise`` A (the rise
    itself when there is no twist)."""
    return 360.0 * rise / abs(twist) if twist else rise


@functools.lru_cache(maxsize=8)
def bessel_onsets(n_max: int, floor: float = BESSEL_FLOOR) -> np.ndarray:
    """The argument ``z`` above which ``|J_n(z)|`` reaches ``floor`` of its first maximum.

    Parameters
    ----------
    n_max : int
        Orders 0 to ``n_max``.
    floor : float, optional
        Defaults to :data:`BESSEL_FLOOR`.

    Returns
    -------
    numpy.ndarray
        ``(n_max + 1,)`` increasing with the order; 0 for order 0.
    """
    from scipy.special import jn_zeros, jv

    onsets = np.zeros(n_max + 1)
    for n in range(1, n_max + 1):
        # J_n rises from 0 to a single maximum before its first zero
        z = np.linspace(0.0, float(jn_zeros(n, 1)[0]), 400)
        j = np.abs(jv(n, z))
        above = np.flatnonzero(j >= floor * j.max())
        onsets[n] = float(z[above[0]]) if len(above) else 0.0
    return onsets


def layer_line_heights(twist, rise, csym=1, y_max=None, radius=None, x_max=None):
    """Heights and Bessel orders of the layer lines of a helix.

    Parameters
    ----------
    twist : float
        Degrees.
    rise : float
        A.
    csym : int, optional
        Rotational symmetry about the axis; the orders are its multiples.
    y_max : float, optional
        Highest ``|Y|`` wanted, in 1/A.
    radius, x_max : float, optional
        With both, orders whose line cannot carry signal within ``|X| <= x_max``
        (1/A) for a helix of outer radius ``radius`` (A) are left out.

    Returns
    -------
    heights, orders : numpy.ndarray
        One entry per line (both signs of the order and of ``m``).
    """
    p = pitch(twist, rise)
    step = csym / p  # between consecutive orders, 1/A
    n_cap = None
    if radius is not None and x_max is not None:
        z_max = 2.0 * np.pi * radius * x_max
        n_probe = int(z_max) + 8
        on = bessel_onsets(n_probe)
        n_cap = int(np.searchsorted(on, z_max, side="right")) - 1
        n_cap = (max(n_cap, 0) // csym) * csym
    y_max = float(y_max) if y_max is not None else 0.5 / 1.0
    m_max = int(np.floor(y_max * rise)) + 1
    heights, orders = [], []
    for m in range(-m_max, m_max + 1):
        y0 = m / rise
        if n_cap is None:
            j_max = int(np.ceil((y_max + abs(y0)) / step)) + 1
        else:
            j_max = n_cap // csym
        for j in range(-j_max, j_max + 1):
            y = y0 + j * step
            if abs(y) <= y_max + 1.0 / max(p, 1.0):
                heights.append(y)
                orders.append(j * csym)
    return np.asarray(heights), np.asarray(orders)


def layer_line_mask(
    shape,
    apix,
    twist,
    rise,
    csym=1,
    radius=None,
    cutoff_res=None,
    width=4.0,
    floor=BESSEL_FLOOR,
):
    """The mask that keeps the layer lines, in the layout of ``numpy.fft.rfft2``.

    Parameters
    ----------
    shape : tuple of int
        ``(ny, nx)`` of the image; the helical axis is along ``ny``.
    apix : float
        Pixel size, A.
    twist, rise, csym : float
        The helix: degrees, A and the rotational symmetry.
    radius : float
        Outer radius of the helix, A. It sets how far from the axis each layer
        line can carry signal (the Bessel orders); it is needed because a helix
        of small twist has far more lines than the plane has room for.
    cutoff_res : float, optional
        Resolution limit in A; nothing is kept beyond it. Defaults to Nyquist.
    width : float, optional
        Full width at half maximum of each line, in pixels of the transform
        (1/(ny apix)). A filament of finite length, or one that bends, has wider
        lines; 2 is for a straight, ideal one, and class averages of real data
        did best at 3 to 5. Defaults to 4.
    floor : float, optional
        Each order is kept from the radius at which its Bessel function reaches
        this share of its first maximum (:func:`bessel_onsets`). Defaults to
        :data:`BESSEL_FLOOR`.

    Returns
    -------
    numpy.ndarray
        ``(ny, nx // 2 + 1)`` in [0, 1].
    """
    if radius is None:
        raise ValueError(
            "radius (A) is needed to tell where each layer line has signal"
        )
    ny, nx = shape
    fy = np.fft.fftfreq(ny, apix)
    fx = np.fft.rfftfreq(nx, apix)
    x_max = 0.5 / apix if cutoff_res is None else min(0.5 / apix, 1.0 / cutoff_res)
    y_max = x_max
    sigma = width * (1.0 / (ny * apix)) / 2.3548
    heights, orders = layer_line_heights(twist, rise, csym, y_max, radius, x_max)

    # the layer lines of each order, as one profile along Y; an order's signal
    # starts at the radius-dependent onset, and every smaller order's does earlier
    n_top = int(np.abs(orders).max()) if len(orders) else 0
    onsets = bessel_onsets(max(n_top, 1), floor) / (2.0 * np.pi * radius)
    by_order = {}
    for y, n in zip(heights, orders):
        by_order.setdefault(abs(int(n)), []).append(y)
    levels = sorted(by_order)
    cumulative = np.zeros(ny)
    columns = np.zeros((len(levels), ny))
    for k, n in enumerate(levels):
        ys = np.asarray(by_order[n])
        # max over the lines of this order, in chunks to bound the memory
        prof = np.zeros(ny)
        for lo in range(0, len(ys), 64):
            d = fy[:, None] - ys[None, lo : lo + 64]
            prof = np.maximum(prof, np.exp(-0.5 * (d / sigma) ** 2).max(axis=1))
        cumulative = np.maximum(cumulative, prof)
        columns[k] = cumulative
    start = np.array([onsets[n] if n < len(onsets) else np.inf for n in levels])
    # for each column of the transform, the highest order whose signal has started
    idx = np.searchsorted(start, fx, side="right") - 1
    idx = np.clip(idx, 0, len(levels) - 1)
    mask = columns[idx].T
    # nothing beyond the resolution limit, with a soft edge
    r = np.hypot(fy[:, None], fx[None, :])
    edge = 0.1 * x_max
    mask = mask * np.clip((x_max - r) / edge, 0.0, 1.0)
    return mask


def noise_power(power, mask, floor=0.02, min_count=12):
    """The noise power at every frequency, from the pixels between the layer lines.

    Nothing but noise lies there, so the median power of the pixels the mask
    leaves out, ring by ring, estimates it (for complex Gaussian noise the power
    is exponential and its median is ``ln 2`` of its mean).

    Parameters
    ----------
    power : numpy.ndarray
        ``|rfft2(image)|**2``.
    mask : numpy.ndarray
        :func:`layer_line_mask` of the same shape.
    floor : float, optional
        The mask below which a pixel counts as between the lines.
    min_count : int, optional
        A ring with fewer such pixels takes the estimate of the nearest ring
        that has enough.

    Returns
    -------
    numpy.ndarray
        The mean noise power of each pixel, the shape of ``power``.
    """
    from helicon.lib import rings

    ring = rings.rfft2_rings(power.shape)
    off = mask < floor
    n_ring = int(ring.max()) + 1
    medians = rings.grouped_median(
        power[off], rings.group_index(ring[off], n_ring), min_count=min_count
    )
    est = medians / np.log(2.0)
    ok = np.flatnonzero(np.isfinite(est))
    if len(ok) == 0:
        return np.zeros_like(power)
    est = np.interp(np.arange(n_ring), ok, est[ok])
    return est[ring]


def signal_power(power, noise, smooth=(3, 5)):
    """The signal power at every pixel of an ``rfft2``: its local mean power less the noise.

    A pixel alone has the spread of an exponential, and weighting it by itself
    leaves the noise that happens to stand out ("musical noise"), so the power is
    averaged over a neighbourhood first.

    Parameters
    ----------
    power : numpy.ndarray
        ``|rfft2(image)|**2``.
    noise : numpy.ndarray
        The noise power of each pixel (:func:`noise_power`).
    smooth : tuple of int, optional
        Size of the neighbourhood (rows, columns). Defaults to (3, 5).

    Returns
    -------
    numpy.ndarray
        The signal power, 0 where the mean is below the noise.
    """
    from scipy import ndimage as ndi

    # The rows of an rfft2 are periodic. Its columns run from 0 to Nyquist and
    # are not: wrapping them would average the highest frequencies into the
    # neighbourhood of the zero frequency, where the equator and the strongest
    # layer lines are. They are reflected instead (the transform is symmetric
    # about its first column).
    mean = ndi.uniform_filter(power, size=smooth, mode=("wrap", "reflect"))
    return np.maximum(mean - noise, 0.0)


def layer_line_filter(
    image,
    apix,
    twist,
    rise,
    csym=1,
    radius=None,
    cutoff_res=None,
    width=4.0,
    axis=0,
    wiener=True,
    strength=2.0,
    smooth=(3, 5),
):
    """Keep only the layer lines of a helix in a straightened image.

    Parameters
    ----------
    image : numpy.ndarray
        2D image of a straight filament.
    apix, twist, rise, csym, radius, cutoff_res, width :
        As for :func:`layer_line_mask`.
    axis : int, optional
        The axis of the image along the filament: 0 (rows, the default) or 1.
    wiener : bool, optional
        Also weight every pixel on the lines by the Wiener gain
        ``signal / (signal + noise)``: the noise power is estimated from the pixels
        between the lines (:func:`noise_power`) and the signal power is the local
        mean of the power less that. The resolution limit then needs no choosing:
        where the signal is weaker than the noise the weight is near 0. Defaults
        to True; with False only the mask is applied, and the resolution limit
        (``cutoff_res``) is what keeps the noise of the high frequencies out.
    strength : float, optional
        Scales the noise power; above 1 removes more. Defaults to 2 (1 left more
        noise on real class averages).
    smooth : tuple of int, optional
        Size (rows, columns of the transform) of the neighbourhood averaged to
        estimate the signal power. Defaults to (3, 5).

    Returns
    -------
    filtered : numpy.ndarray
        The filtered image, of the shape and dtype of ``image`` (float).
    kept : float
        The share of the variance of white noise that remains: the mean square
        of the mask over the whole transform.
    """
    img = np.asarray(image, dtype=float)
    if axis == 1:
        img = img.T
    elif axis != 0:
        raise ValueError("axis must be 0 or 1")
    mask = layer_line_mask(
        img.shape, apix, twist, rise, csym, radius, cutoff_res, width
    )
    F = np.fft.rfft2(img)
    if wiener:
        power = np.abs(F) ** 2
        noise = strength * noise_power(power, mask)
        signal = signal_power(power, noise, smooth)
        mask = mask * signal / np.maximum(signal + noise, 1e-30)
    out = np.fft.irfft2(F * mask, s=img.shape)
    # an rfft2 column other than 0 and Nyquist stands for two of the full transform
    w = np.full(mask.shape[1], 2.0)
    w[0] = 1.0
    if img.shape[1] % 2 == 0:
        w[-1] = 1.0
    kept = float((mask**2 * w[None, :]).sum() / (w[None, :] * np.ones_like(mask)).sum())
    return (out.T if axis == 1 else out), kept
