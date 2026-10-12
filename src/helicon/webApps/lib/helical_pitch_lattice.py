"""The subunit lattice of a helix, read from its class averages along the axis.

The ring fit of the AbInitio3D tab measures the repeat: the distance after which
the views of a filament come back. Turning that into a twist needs to know how
many turns of the symmetry, ``360 / csym`` each, the helix makes over one repeat.
For a slowly twisting filament whose subunits the class averages do not resolve
(an amyloid) that is one, and ``twist = 360 * rise / (csym * repeat)``. For a helix
whose subunits they do resolve it can be many: VipA/VipB (EMD-2699, C6, 29.4
degrees, 21.8 A) turns 25 times 60 degrees over its 1112 A repeat, because 51
subunits turn 1499.4 degrees, 0.6 short of 25 periods.

That number comes from the class averages themselves, in real space. An average
shifted along the axis by ``k`` subunits has its subunits back where subunits were,
turned by ``k * twist`` about the axis; it matches itself well when that turn is
close to a multiple of ``360 / csym``. So the shifts at which an average matches
itself are the multiples of the rise, and the strongest of them, ``k_s`` subunits,
says that ``csym * twist`` is close to ``360 * m / k_s`` for some ``m``. With the
repeat, ``n = repeat / rise`` subunits, the twist is then

    twist = 360 * j / (csym * n),    j = the whole number nearest n * m / k_s

where a tie is the hand, which a projection cannot tell (``j`` and ``n - j`` are
mirror images), and the lower one is taken. What none of this tells is ``csym``
itself: every ``csym`` with the same ``csym * twist`` puts its subunits at the
same places along the axis. It is the user's choice, as it is for amyloids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# a self-match peak must stand this far above its surroundings to count
MIN_PROMINENCE = 0.03
# how far a peak may sit from a multiple of the rise, as a share of the rise
TOLERANCE = 0.06


@dataclass
class Lattice:
    """What :func:`detect` found.

    Attributes
    ----------
    rise : float
        A: the spacing of the self-match peaks.
    k_s : int
        Subunits in the strongest near-repeat.
    m : int
        Turns of ``360 / csym`` in that near-repeat: ``csym * twist`` is close to
        ``360 * m / k_s``.
    peaks : list of tuple
        ``(shift A, correlation, multiple of the rise)`` of the peaks explained.
    shifts, correlation : numpy.ndarray
        The self-match curve: mean correlation of the averages with themselves
        shifted along the axis.
    """

    rise: float
    k_s: int
    m: int
    peaks: list = field(default_factory=list)
    shifts: np.ndarray = None
    correlation: np.ndarray = None


def self_match(images, apix, max_shift=None, step=0.1):
    """Mean correlation of straightened class averages with themselves shifted along the axis.

    Parameters
    ----------
    images : sequence of numpy.ndarray
        Class averages with the filament horizontal (``z`` along the columns).
    apix : float
        Their pixel size, A.
    max_shift : float, optional
        Largest shift, A. Defaults to 45 per cent of the length of the averages,
        so that the overlap stays more than half of them.
    step : float, optional
        Step of the shifts, in pixels. Defaults to 0.1.

    Returns
    -------
    shifts : numpy.ndarray
        A.
    correlation : numpy.ndarray
        The mean over the averages, at each shift.
    """
    ims = [np.asarray(im, dtype=float) for im in images]
    ny, nx = ims[0].shape
    rows = slice(int(ny * 0.15), int(ny * 0.85))  # the filament, not the background
    xs = [im[rows] - im[rows].mean() for im in ims]
    max_px = (0.45 * nx) if max_shift is None else min(max_shift / apix, 0.45 * nx)
    shifts_px = np.arange(2.0, max_px + 1e-9, step)
    out = np.empty(len(shifts_px))
    for k, d in enumerate(shifts_px):
        i0 = int(np.floor(d))
        f = d - i0
        values = []
        for x in xs:
            a = x[:, : nx - i0 - 1]
            b = (1.0 - f) * x[:, i0 : nx - 1] + f * x[:, i0 + 1 : nx]
            a, b = a - a.mean(), b - b.mean()
            values.append(
                (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30)
            )
        out[k] = float(np.mean(values))
    return shifts_px * apix, out


def _peaks(shifts, corr, min_prominence=MIN_PROMINENCE):
    """Local maxima of the curve that stand ``min_prominence`` above the lower of their two flanks."""
    found = []
    for i in range(1, len(corr) - 1):
        if not (corr[i] >= corr[i - 1] and corr[i] > corr[i + 1]):
            continue
        left = corr[: i + 1][::-1]
        right = corr[i:]
        # the lowest point on each side before the curve rises above this peak
        lo_l = left[
            : np.argmax(left > corr[i]) if (left > corr[i]).any() else len(left)
        ].min()
        lo_r = right[
            : np.argmax(right > corr[i]) if (right > corr[i]).any() else len(right)
        ].min()
        if corr[i] - max(lo_l, lo_r) >= min_prominence:
            found.append((float(shifts[i]), float(corr[i])))
    return found


def detect(images, apix, max_shift=None):
    """The subunit lattice of a set of class averages, or None if they do not show one.

    The rise is the spacing that explains the self-match peaks best: the summed
    heights of the peaks that sit on its multiples, less a penalty for each of its
    multiples where there is no peak (which a spacing of half the rise would have).
    At least two multiples must be explained. ``k_s`` is the multiple with the
    highest peak; ``m`` is the ``360 * m / k_s`` whose pattern of near and far turns
    the heights of the other multiples follow best.

    Parameters
    ----------
    images : sequence of numpy.ndarray
        Straightened class averages (filament horizontal).
    apix : float
        Their pixel size, A.
    max_shift : float, optional
        See :func:`self_match`.

    Returns
    -------
    Lattice or None
    """
    shifts, corr = self_match(images, apix, max_shift=max_shift)
    peaks = _peaks(shifts, corr)
    if len(peaks) < 2:
        return None
    d_max = float(shifts[-1])
    mean_h = float(np.mean([h for _, h in peaks]))
    best = None
    for d, _ in peaks:
        for k in range(1, 7):
            r = d / k
            if r < 3.0 * apix:  # finer than the averages can show
                continue
            tol = TOLERANCE * r
            explained, used = 0.0, []
            for pd, ph in peaks:
                mult = int(round(pd / r))
                if mult >= 1 and abs(pd - mult * r) <= tol:
                    explained += ph
                    used.append((pd, ph, mult))
            multiples = range(1, int(d_max / r) + 1)
            missing = sum(
                1 for mult in multiples if not any(u[2] == mult for u in used)
            )
            score = explained - 0.5 * mean_h * missing
            if len({u[2] for u in used}) >= 2 and (best is None or score > best[0]):
                best = (score, r, used)
    if best is None:
        return None
    _, rise, used = best
    # the rise refined on every peak explained: a least-squares fit through zero
    mults = np.array([u[2] for u in used], dtype=float)
    pos = np.array([u[0] for u in used])
    rise = float((mults * pos).sum() / (mults * mults).sum())
    # the strongest multiple, keeping the highest peak of each multiple
    by_mult = {}
    for pd, ph, mult in used:
        if mult not in by_mult or ph > by_mult[mult][1]:
            by_mult[mult] = (pd, ph)
    k_s = max(by_mult, key=lambda mult: by_mult[mult][1])
    m = _turns_in_near_repeat(by_mult, k_s)
    peaks_out = sorted((pd, ph, mult) for mult, (pd, ph) in by_mult.items())
    return Lattice(
        rise=rise,
        k_s=int(k_s),
        m=int(m),
        peaks=peaks_out,
        shifts=shifts,
        correlation=corr,
    )


def _turns_in_near_repeat(by_mult, k_s):
    """The ``m`` of ``csym * twist ~ 360 * m / k_s`` that the peak heights follow best.

    A shift by ``k`` subunits turns the helix by ``k * m / k_s`` of a period; its
    match is better the closer that is to a whole number. So the heights should
    rise and fall with ``cos(2 pi k m / k_s)``; the ``m`` (coprime to ``k_s``, at
    most half of it: ``m`` and ``k_s - m`` are mirror images) whose cosines
    correlate best with the heights is taken.
    """
    if k_s <= 2:
        return 1 if k_s == 2 else 0
    ks = np.array(sorted(by_mult), dtype=float)
    heights = np.array([by_mult[k][1] for k in sorted(by_mult)])
    best, best_m = None, 1
    for m in range(1, k_s // 2 + 1):
        if np.gcd(m, k_s) != 1:
            continue
        predicted = np.cos(2.0 * np.pi * ks * m / k_s)
        if heights.std() < 1e-12 or predicted.std() < 1e-12:
            c = 0.0
        else:
            c = float(np.corrcoef(predicted, heights)[0, 1])
        if best is None or c > best:
            best, best_m = c, m
    return best_m


def periods_per_repeat(repeat, rise, lattice=None):
    """How many turns of ``360 / csym`` the helix makes over one repeat, and in how many subunits.

    Parameters
    ----------
    repeat : float
        The repeat of the ring fit, A.
    rise : float
        A.
    lattice : Lattice, optional
        From :func:`detect`. Without one, or when the near-repeat is a single
        subunit (``m`` 0: the helix turns little per subunit), one turn per repeat.

    Returns
    -------
    j : int
        Turns of ``360 / csym`` per repeat.
    n : float
        Subunits per repeat: the nearest whole number when a lattice is given,
        ``repeat / rise`` otherwise.
    """
    n_exact = float(repeat) / float(rise)
    if lattice is None or lattice.m == 0:
        return 1, n_exact
    n = max(1, int(round(n_exact)))
    target = n * lattice.m / lattice.k_s
    j = int(
        np.floor(target + 0.5 - 1e-9)
    )  # nearest; a tie (the hand) goes to the lower
    return max(1, j), float(n)


def twist(repeat, rise, csym, lattice=None):
    """The twist, degrees (positive: the hand is the caller's), for one ``csym``.

    ``360 * j / (csym * n)`` with ``j`` and ``n`` from :func:`periods_per_repeat`;
    without a lattice that is ``360 * rise / (csym * repeat)``.
    """
    j, n = periods_per_repeat(repeat, rise, lattice)
    return 360.0 * j / (max(int(csym), 1) * n)
