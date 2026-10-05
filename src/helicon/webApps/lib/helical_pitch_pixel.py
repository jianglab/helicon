"""Pixel evidence for the class ring of AbInitio3D: registered class averages.

The ring of :mod:`helical_pitch_phase` gives every class ``k`` a phase
``theta_k`` on a circle of circumference ``P`` (the repeat) and reads both from
the *metadata*: which class each segment was put in, and where along its
filament it lies. A pair of segments of classes ``a`` and ``b``, ``d`` apart,
satisfies ``d = theta_b - theta_a (mod P)``.

Two class averages are images of the same filament at two places on the repeat,
so registering one onto the other along the axis measures the same difference
from the *pixels*: shifting ``b`` by ``dx`` brings it onto ``a`` when
``dx = theta_b - theta_a (mod P)``. It has the form of a segment pair with the
distance ``d`` replaced by ``dx``, and enters the synchronization the same way:

    H_pix(P) = sum over registered pairs of w * exp(2 pi i dx / P)

What each says that the other cannot:

* The pixel offsets do not depend on how well the segments were assigned to
  classes, only on the class averages, so poor assignments (which blur the
  averages and add false pairs to the metadata) do not enter them. They also
  span the width of an image, not the length of a filament, so they bear on the
  repeat of a short filament that no segment pair can reach.
* The metadata has far more pairs, from all the filaments, and is not misled by
  two averages that overlap little or look alike.

Registration is twist-independent: offsets are axial, as in
:mod:`denovo3d_register`, so nothing here assumes the twist or the rise.

This module is a prototype of the integration. The two terms are normalised
separately (each by the degrees of the classes in it, so that its largest
eigenvalue is at most 1) and combined as ``(1 - alpha) H_meta + alpha H_pix``,
which keeps the number of pairs of one kind from deciding the weight of the
other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import helical_pitch_phase as ph


@dataclass
class PixelOffsets:
    """Registered axial offsets between class averages.

    Attributes
    ----------
    a, b : np.ndarray
        Positions in the list of classes given (row of the ring), ``a < b``.
    dx : np.ndarray
        Shift, in A, that brings class ``b`` onto class ``a``: the phase
        difference ``theta_b - theta_a`` modulo the repeat.
    corr, prominence : np.ndarray
        Correlation at the best offset, and how far that peak stands above the
        rest of the profile (in standard deviations).
    weight : np.ndarray
        The weight each pair carries (0 for a pair that is not used).
    n_classes : int
    """

    a: np.ndarray
    b: np.ndarray
    dx: np.ndarray
    corr: np.ndarray
    prominence: np.ndarray
    weight: np.ndarray
    n_classes: int

    def __len__(self):
        return int(np.count_nonzero(self.weight))


def pair_weight(corr, prominence, min_corr=0.3, min_prominence=0.0):
    """Weight of a registered pair: its correlation squared, or 0 if unreliable.

    A pair that overlaps little still has a best offset and a correlation that
    can look respectable; what gives it away is a profile with no distinct peak
    (a low ``prominence``), see :func:`denovo3d_register._dx_profile`.
    """
    corr = np.asarray(corr, dtype=float)
    prominence = np.asarray(prominence, dtype=float)
    ok = (corr >= min_corr) & (prominence >= min_prominence)
    return np.where(ok, corr**2, 0.0)


def class_pixel_offsets(
    images,
    classes,
    apix,
    max_width=128,
    rot_range=3.0,
    dy_range=3.0,
    min_corr=0.3,
    min_prominence=0.0,
    progress=None,
    pairs=None,
):
    """Register every pair of the selected class averages along the axis.

    Parameters
    ----------
    images : sequence of 2D arrays
        Class averages, the filament roughly horizontal, the axis to the right.
    classes : sequence of int
        Positions in ``images`` of the selected classes, in the order of the
        rows of the ring (``PhasePitchResult.class_ids``).
    apix : float
        Pixel size of ``images``, in A.
    max_width : int, optional
        Images are block-averaged to at most this width first. Defaults to 128.
    rot_range, dy_range : float, optional
        Search ranges for the in-plane rotation (degrees) and the shift across
        the filament (pixels) around the estimate made for every image, so that
        the offset along the axis is not biased by how each was centred.
    min_corr, min_prominence : float, optional
        A pair below either gets no weight (:func:`pair_weight`). The default
        for the prominence is 0 (not used): for images that span a repeat or
        more the correlation profile has a peak at every repeat, which lowers
        the prominence of a perfectly good pair (2.6-2.8 on noiseless synthetic
        averages, where every offset was within 6 A of the truth).
    progress : callable, optional
    pairs : sequence of (int, int), optional
        Only these pairs of positions in ``images`` (first before second in
        ``classes``), instead of every pair.

    Returns
    -------
    PixelOffsets
    """
    from . import denovo3d_register as R

    classes = [int(c) for c in classes]
    K = len(classes)
    prepared, factor = ph._register_prepared(images, classes, max_width)
    same_polarity = [(False, False)]
    if pairs is None:
        todo = [(i, j) for x, i in enumerate(classes) for j in classes[x + 1 :]]
    else:
        todo = [(int(i), int(j)) for i, j in pairs]
    if not todo:
        z = np.zeros(0)
        return PixelOffsets(z.astype(int), z.astype(int), z, z, z, z, K)

    def register(a, b, **kw):
        return R.register_pair(a, b, rot_range=rot_range, dy_range=dy_range, **kw)

    # the shared pool of ph._register_all uses the default search ranges, so
    # the ranges asked for here are applied by registering directly
    out = []
    for k, (i, j) in enumerate(todo):
        if progress is not None:
            progress(k, len(todo))
        out.append(
            register(prepared[i], prepared[j], flips=same_polarity, refine=False)
        )
    row = {c: r for r, c in enumerate(classes)}
    a = np.array([row[i] for i, _ in todo])
    b = np.array([row[j] for _, j in todo])
    # the pixel size of the registered image is the original's times its binning
    dx = np.array(
        [res["dx"] * factor[i] * float(apix) for (i, _), res in zip(todo, out)]
    )
    corr = np.array([res["corr"] for res in out])
    prom = np.array([res["prominence"] for res in out])
    return PixelOffsets(
        a, b, dx, corr, prom, pair_weight(corr, prom, min_corr, min_prominence), K
    )


@dataclass
class SelfSimilarity:
    """How much an average repeats along the axis at a given lag.

    A class average that repeats with the period ``P`` matches itself shifted by
    ``P``. Pairs of segments, and pairs of registered averages, cannot tell the
    repeat from a half of it (phases consistent at ``P`` are consistent at
    ``P / 2``) when the filaments are shorter than the repeat. The averages can,
    wherever the lag is shorter than the image: at half the repeat the average
    does *not* match itself. So a candidate period whose lag can be observed is
    kept only if the averages repeat at it; one too long to observe is
    neither kept nor rejected.

    Attributes
    ----------
    lag : np.ndarray
        Lags, in A, up to the longest with enough overlap.
    corr : np.ndarray
        Mean over the classes of the correlation of an average with itself
        shifted by the lag, after the part that does not vary along the axis
        (the body of the filament, which matches at any lag) is removed.
    """

    lag: np.ndarray
    corr: np.ndarray

    def gate(self, periods):
        """1 for a period whose lag is too long to observe, else the correlation."""
        periods = np.asarray(periods, dtype=float)
        if not len(self.lag):
            return np.ones(len(periods))
        seen = periods <= self.lag[-1]
        g = np.ones(len(periods))
        g[seen] = np.clip(np.interp(periods[seen], self.lag, self.corr), 0.0, 1.0)
        return g


def self_similarity(images, classes, apix, max_width=128, min_overlap=0.35):
    """The self-similarity of the selected class averages along the axis.

    Parameters
    ----------
    images, classes, apix, max_width
        As for :func:`class_pixel_offsets`.
    min_overlap : float, optional
        Smallest share of the image width that must overlap itself at a lag;
        longer lags are not observed. Defaults to the registration's 0.35.

    Returns
    -------
    SelfSimilarity
    """
    classes = [int(c) for c in classes]
    prepared, factor = ph._register_prepared(images, classes, max_width)
    profiles, step = [], None
    for c in classes:
        a = np.asarray(prepared[c], dtype=np.float64)
        a = a - a.mean(axis=1, keepdims=True)
        nx = a.shape[1]
        n = 2 * nx
        num = np.fft.irfft(np.abs(np.fft.rfft(a, n=n, axis=1)) ** 2, n=n, axis=1).sum(0)
        e = np.concatenate([[0.0], np.cumsum((a**2).sum(axis=0))])
        cols = int(np.floor((1.0 - min_overlap) * nx))
        lags = np.arange(0, cols + 1)
        # the overlap of the image with itself shifted by k: columns k..nx and 0..nx-k
        den = np.sqrt((e[nx] - e[lags]) * e[nx - lags])
        with np.errstate(divide="ignore", invalid="ignore"):
            profiles.append(np.where(den > 0, num[lags] / den, 0.0))
        step = float(apix) * factor[c]
    if not profiles:
        return SelfSimilarity(np.zeros(0), np.zeros(0))
    m = min(len(p) for p in profiles)
    corr = np.mean([p[:m] for p in profiles], axis=0)
    return SelfSimilarity(np.arange(m) * step, corr)


def _normalised(H, n_ab):
    """``H`` divided by the geometric mean of the class degrees (Hermitian)."""
    deg = n_ab.sum(-1)
    norm = np.sqrt(deg[..., :, None] * deg[..., None, :]) + 1e-12
    return H / norm


def pixel_matrices(offsets, periods, sign=1.0):
    """The normalised pixel term ``H_pix`` at every candidate period.

    Returns an array ``(len(periods), K, K)``, Hermitian, with the part that does
    not depend on which classes the offsets are between subtracted -- at a long
    candidate period every offset is near zero and every pair agrees, as for the
    metadata term of :func:`helical_pitch_phase.phase_scan`.
    """
    periods = np.atleast_1d(np.asarray(periods, dtype=float))
    K = offsets.n_classes
    use = offsets.weight > 0
    a, b = offsets.a[use], offsets.b[use]
    w = offsets.weight[use]
    dx = sign * offsets.dx[use]
    H = np.zeros((len(periods), K, K), dtype=complex)
    if not len(w):
        return H
    E = np.exp(2j * np.pi * dx[None, :] / periods[:, None])  # (n, pairs)
    H[:, a, b] = w[None, :] * E  # each pair of classes is registered once
    H = H + np.conj(np.transpose(H, (0, 2, 1)))
    n_ab = np.zeros((K, K))
    np.add.at(n_ab, (a, b), w)
    n_ab = n_ab + n_ab.T
    ebar = (E @ w) / w.sum()
    H = H - n_ab[None] * ebar.real[:, None, None]
    return _normalised(H, n_ab[None])


def metadata_matrices(hist, centers, periods, length_scale=800.0):
    """The normalised metadata term ``H_meta``: what ``phase_scan`` diagonalises."""
    periods = np.atleast_1d(np.asarray(periods, dtype=float))
    w = np.exp(-0.5 * (centers / length_scale) ** 2)
    hw = hist * w
    n_ab = hw.sum(2)
    n_ab = n_ab + n_ab.T
    K = hist.shape[0]
    total = hw.sum()
    marg = hw.sum((0, 1))
    E = np.exp(2j * np.pi * centers[None, :] / periods[:, None])
    H = (hw.reshape(K * K, -1).astype(complex) @ E.T).T.reshape(len(periods), K, K)
    H = H + np.conj(np.transpose(H, (0, 2, 1)))
    ebar = (E @ marg) / max(total, 1e-12)
    H = H - n_ab[None] * ebar.real[:, None, None]
    return _normalised(H, n_ab[None])


def joint_scan(
    hist,
    centers,
    offsets,
    periods,
    alpha=0.5,
    length_scale=800.0,
    sign=1.0,
    chunk=32,
    gate=None,
):
    """Score every candidate period with the metadata and pixel terms together.

    ``(1 - alpha) H_meta + alpha H_pix``: ``alpha = 0`` is the metadata ring of
    :func:`helical_pitch_phase.phase_scan` and ``alpha = 1`` the pixels alone.
    ``gate``, one factor per period (:meth:`SelfSimilarity.gate`), multiplies
    the scores, so a period at which the averages do not repeat cannot win.

    Returns
    -------
    scores : np.ndarray
        Top eigenvalue at each period.
    phases : np.ndarray
        ``(n_periods, K)`` class phases in radians, from the top eigenvector.
    """
    periods = np.asarray(periods, dtype=float)
    K = hist.shape[0]
    scores = np.zeros(len(periods))
    phases = np.zeros((len(periods), K))
    for s in range(0, len(periods), chunk):
        p = periods[s : s + chunk]
        H = np.zeros((len(p), K, K), dtype=complex)
        if alpha < 1.0:
            H += (1.0 - alpha) * metadata_matrices(hist, centers, p, length_scale)
        if alpha > 0.0 and len(offsets):
            H += alpha * pixel_matrices(offsets, p, sign)
        vals, vecs = np.linalg.eigh(H)
        scores[s : s + chunk] = vals[:, -1]
        phases[s : s + chunk] = np.angle(vecs[:, :, -1])
    if gate is not None:
        scores = scores * np.asarray(gate, dtype=float)
    return scores, phases


def estimate_period_joint(
    pairs,
    offsets,
    alpha=0.5,
    length_scale=800.0,
    periods=None,
    refine_fraction=0.03,
    sign=1.0,
    similarity=None,
):
    """Coarse scan then a fine scan, as :func:`helical_pitch_phase.estimate_period`.

    ``similarity`` (:func:`self_similarity`) gates the coarse scan only, so that
    a sub-multiple of the repeat is not taken for it; the fine scan about the
    chosen period is not gated, so the gate cannot shift the peak.

    Returns a dict with ``period`` (A), ``score``, the coarse ``periods`` and
    ``scores``, ``phases`` (radians per class) at the best period, and the same
    ``support`` and ``double_ratio`` diagnostics.
    """
    hist, centers = ph.class_pair_histogram(pairs)
    if periods is None:
        periods = np.arange(100.0, pairs.max_sep + 0.1, 2.0)
    gate = None if similarity is None else similarity.gate(periods)
    scores, _ = joint_scan(
        hist, centers, offsets, periods, alpha, length_scale, sign, gate=gate
    )
    p0, _ = ph._peak(scores, periods)
    fine = np.arange(p0 * (1 - refine_fraction), p0 * (1 + refine_fraction), 0.25)
    fs, fph = joint_scan(hist, centers, offsets, fine, alpha, length_scale, sign)
    p, k = ph._peak(fs, fine)
    support = float(pairs.W[np.abs(pairs.D) > p].sum() / max(pairs.W.sum(), 1e-12))
    s2, _ = joint_scan(
        hist, centers, offsets, np.array([2.0 * p]), alpha, length_scale, sign
    )
    return dict(
        period=p,
        score=float(fs[k]),
        periods=np.asarray(periods),
        scores=scores,
        phases=fph[k],
        support=support,
        double_ratio=float(s2[0] / fs[k]) if fs[k] > 0 else 0.0,
    )


def estimate_period_adaptive(
    pairs,
    offsets,
    similarity=None,
    alpha=0.5,
    window=0.08,
    accept=0.9,
    length_scale=800.0,
    sign=1.0,
    alpha_short=None,
):
    """Joint estimate, sharpened by the metadata where the metadata can be trusted.

    The pixel offsets are few and each good to a pixel or two, so the repeat they
    give is uncertain by a few percent, while the metadata, from all the pairs, is
    precise to a fraction of a percent once filaments are about a repeat long --
    and badly biased low where they are shorter. So:

    1. the joint scan (``alpha``, gated by ``similarity``) finds the repeat;
    2. the metadata alone is scanned in a window of ``window`` of the repeat about
       it. If its peak is well inside the window (closer to the centre than
       ``accept`` of the half-width) it is taken: two independent measurements
       agree, and the sharper one wins. If the peak is at the edge, the
       metadata has run off, as it does when the filaments are shorter than the
       repeat, and the joint estimate stands.

    ``alpha_short``, if given, is the weight of the pixels in the final scan when
    the metadata is rejected (the first step's ``alpha`` is kept for finding the
    repeat). The class phases are those of the joint fit at the repeat returned.
    The returned dict has ``stage`` (``"metadata"`` or ``"joint"``) and
    ``joint_period`` as well.
    """
    first = estimate_period_joint(
        pairs, offsets, alpha, length_scale, sign=sign, similarity=similarity
    )
    p0 = first["period"]
    first["joint_period"] = p0
    first["stage"] = "joint"
    grid = np.arange(p0 * (1 - window), p0 * (1 + window), 0.5)
    second = estimate_period_joint(
        pairs,
        offsets,
        alpha=0.0,
        length_scale=length_scale,
        periods=grid,
        refine_fraction=0.01,
    )
    p = second["period"]
    if abs(p / p0 - 1.0) < accept * window:
        hist, centers = ph.class_pair_histogram(pairs)
        _, phases = joint_scan(
            hist, centers, offsets, np.array([p]), alpha, length_scale, sign
        )
        second.update(
            phases=phases[0],
            joint_period=p0,
            stage="metadata",
            support=first["support"],
        )
        return second
    if alpha_short is not None and alpha_short != alpha:
        fine = estimate_period_joint(
            pairs,
            offsets,
            alpha_short,
            length_scale,
            periods=np.arange(p0 * (1 - 0.03), p0 * (1 + 0.03), 0.5),
            refine_fraction=0.01,
            sign=sign,
        )
        fine.update(joint_period=p0, stage="joint", support=first["support"])
        return fine
    return first


def estimate_period_gated(
    pairs,
    offsets,
    similarity=None,
    alpha_basin=0.0,
    alpha_phases=0.5,
    window=0.03,
    length_scale=800.0,
    sign=1.0,
):
    """The metadata's repeat with the pixels' veto, and the pixels' class phases.

    What each term is good at differs. The repeat the pixel offsets give is
    uncertain by several percent, and by an amount that depends on the images
    (+2% for 12 classes of a 400 A repeat, -11% for 20 classes of 700 A in the
    synthetic tests): offsets reach only a fraction of a repeat, so the repeat
    is set by the few pairs near half of it. The class phases, given the repeat,
    are good to a fraction of a percent whatever the labels are. The metadata
    gives the sharpest repeat once filaments are about a repeat long. So:

    1. the repeat is searched with the metadata (``alpha_basin`` = 0) or a little
       of the pixels, gated by the self-similarity of the averages, which keeps a
       sub-multiple of the repeat from winning;
    2. the metadata alone is scanned in a window of ``window`` of the repeat
       about it, for the sharp value;
    3. the class phases are those of the joint fit (``alpha_phases``) at that
       repeat, from the pixels as much as from the labels.

    The returned dict is that of :func:`estimate_period_joint` at the repeat
    found, with ``coarse_period`` added.
    """
    first = estimate_period_joint(
        pairs, offsets, alpha_basin, length_scale, sign=sign, similarity=similarity
    )
    p0 = first["period"]
    second = estimate_period_joint(
        pairs,
        offsets,
        alpha=0.0,
        length_scale=length_scale,
        periods=np.arange(p0 * (1 - window), p0 * (1 + window), 0.5),
        refine_fraction=0.01,
    )
    p = second["period"]
    hist, centers = ph.class_pair_histogram(pairs)
    _, phases = joint_scan(
        hist, centers, offsets, np.array([p]), alpha_phases, length_scale, sign
    )
    second.update(phases=phases[0], coarse_period=p0, support=first["support"])
    return second


def pixel_residuals(offsets, phases, period, sign=1.0):
    """How far each registered offset is from what the ring says it should be.

    The ring puts class ``k`` at the phase ``phases[k]`` (radians, the
    eigenvector's angle, which is minus ``2 pi theta / P``), so a pair is
    expected ``dx = (phases[a] - phases[b]) P / (2 pi)`` apart.

    Returns
    -------
    residual : np.ndarray
        Per pair, in A, wrapped to half a period either way; NaN for a pair
        that has no weight.
    class_score : np.ndarray
        Per class, the weighted median absolute residual over its pairs
        (A); NaN for a class with no pair. A large value marks a class whose
        pixels and metadata disagree: blurred, two views mixed, or a wrong
        origin.
    """
    d = (phases[offsets.a] - phases[offsets.b]) * period / (2 * np.pi)
    res = (sign * offsets.dx - d + period / 2.0) % period - period / 2.0
    res = np.where(offsets.weight > 0, res, np.nan)
    score = np.full(offsets.n_classes, np.nan)
    for k in range(offsets.n_classes):
        mine = ((offsets.a == k) | (offsets.b == k)) & (offsets.weight > 0)
        if np.any(mine):
            score[k] = _weighted_median(np.abs(res[mine]), offsets.weight[mine])
    return res, score


def pixel_agreement(offsets, phases, period, sign=1.0):
    """How well the registered offsets of each class agree with the ring, 0 to 1.

    The mean, over the registered pairs the class is in (each weighted by its
    correlation squared), of ``cos(2 pi r / P)``, with ``r`` the difference between
    the offset and what the ring says (:func:`pixel_residuals`), so that 1 is
    "every offset exactly where the ring puts it", 0 is a residual of a quarter of
    the repeat on average, and a class whose averages are placed at random against
    the ring has about 0. Negative means are clipped to 0, to put the score on the
    same fixed 0 to 1 scale as the class fit.

    Returns
    -------
    np.ndarray
        Per class; NaN for a class with no registered pair.
    """
    d = (phases[offsets.a] - phases[offsets.b]) * period / (2 * np.pi)
    res = (sign * offsets.dx - d + period / 2.0) % period - period / 2.0
    c = np.cos(2 * np.pi * res / period)
    w = offsets.weight
    K = offsets.n_classes
    num = np.bincount(offsets.a, weights=w * c, minlength=K) + np.bincount(
        offsets.b, weights=w * c, minlength=K
    )
    den = np.bincount(offsets.a, weights=w, minlength=K) + np.bincount(
        offsets.b, weights=w, minlength=K
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den > 0, num / den, np.nan)
    return np.clip(mean, 0.0, 1.0)


def pixel_sign(offsets, phases, period):
    """Which direction of the axis the registered offsets are in, from the data.

    The class averages are registered in their own frame, and the ring's axial
    coordinate is the metadata's; if the two run opposite ways every offset is
    negated. The sign whose offsets agree better with the ring (smaller median
    class residual, :func:`pixel_residuals`) is returned, with both medians.

    Returns
    -------
    sign : float
        +1 or -1.
    medians : tuple of float
        The median residual (A) at +1 and at -1.
    """
    med = []
    for sign in (1.0, -1.0):
        _, per_class = pixel_residuals(offsets, phases, period, sign)
        med.append(
            float(np.nanmedian(per_class)) if np.any(np.isfinite(per_class)) else np.inf
        )
    return (1.0 if med[0] <= med[1] else -1.0), (med[0], med[1])


# A class whose registered averages lie further than this fraction of the repeat
# from where the ring puts it is said to disagree with the pixels. On EMPIAR-10940
# the classes whose segments fit the ring (fit 0.5 and above) were all within 0.10
# of a repeat; those that fit poorly had medians of 0.29 and 0.16 of a repeat for
# the two class selections tried.
DISAGREE_FRACTION = 0.12


@dataclass
class PixelCheck:
    """The registered class averages against the ring, class by class.

    Attributes
    ----------
    residual : np.ndarray
        Per class of the ring, how far (A) the registered offsets to the other
        classes lie from what the ring says (weighted median over the class's
        pairs); NaN for a class with no registered pair.
    sign : float
        +1 or -1: the direction the offsets run relative to the ring.
    n_pairs : int
        Registered pairs of classes that were used.
    cutoff : float
        The residual above which a class is said to disagree (A).
    agreement : np.ndarray or None
        Per class, the 0 to 1 score of :func:`pixel_agreement` (1: every registered
        offset where the ring puts it).
    pairs : np.ndarray or None
        Per class, the number of registered pairs it is in.
    """

    residual: np.ndarray
    sign: float
    n_pairs: int
    cutoff: float
    agreement: np.ndarray = None
    pairs: np.ndarray = None

    def disagree(self):
        """Per class: the pixels place it elsewhere than its segments do."""
        r = np.asarray(self.residual, dtype=float)
        return np.isfinite(r) & (r > self.cutoff)

    def agree(self):
        """Per class: the pixels place it where its segments do."""
        r = np.asarray(self.residual, dtype=float)
        return np.isfinite(r) & (r <= self.cutoff)


def class_pixel_check(images, class_zdir, phases, period, apix, max_width=128):
    """Compare the registered class averages with the ring, class by class.

    An independent check of the ring: it uses the images and not the segments
    (positions or labels), so a class whose segments are poorly assigned, a mix
    of views or of another type shows up as one whose averages do not lie where
    the ring puts it. It is not used to change the ring.

    Parameters
    ----------
    images : sequence of 2D arrays
        The class averages of the ring's classes, in the order of the ring's
        rows (``PhasePitchResult.class_ids``).
    class_zdir : sequence of float
        Per class, the direction of the ring's axial coordinate in the average
        (``PhasePitchResult.class_zdir``), to turn them all the same way.
    phases : np.ndarray
        The ring's class phases (``PhasePitchResult.phases``).
    period : float
        The repeat, in A.
    apix : float
        Pixel size of the averages, in A.

    Returns
    -------
    PixelCheck or None
        None with fewer than three classes, or without enough registered pairs.
    """
    from . import helical_pitch_map as maps

    K = len(images)
    if K < 3:
        return None
    oriented = maps.straighten_classes(images, class_zdir, apix, apix)
    offsets = class_pixel_offsets(oriented, list(range(K)), apix, max_width)
    if len(offsets) < 3:
        return None
    sign, _ = pixel_sign(offsets, phases, period)
    _, residual = pixel_residuals(offsets, phases, period, sign)
    return PixelCheck(
        residual,
        sign,
        len(offsets),
        DISAGREE_FRACTION * float(period),
        pixel_agreement(offsets, phases, period, sign),
        np.bincount(offsets.a, minlength=K) + np.bincount(offsets.b, minlength=K),
    )


def _weighted_median(x, w):
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def candidate_agreement(
    images, zdir, phases, period, sign, apix, cand_images, cand_zdir, cand_phases
):
    """The pixel agreement unselected classes would have if added to the ring.

    Each candidate's average is registered against the averages of the ring's
    classes, and scored as :func:`pixel_agreement` scores the classes of the ring,
    with the candidate at the azimuth its segments give it
    (:func:`helical_pitch_phase.candidate_fit`, ``placement=True``).

    Parameters
    ----------
    images, zdir, phases : sequence
        The ring's class averages, their axial directions and azimuths (radians).
    period : float
        The repeat, in A.
    sign : float
        ``PixelCheck.sign`` of the ring's own check.
    apix : float
        Pixel size of the averages, in A.
    cand_images, cand_zdir, cand_phases : sequence
        The same for the candidates.

    Returns
    -------
    np.ndarray
        Per candidate; NaN if it has no registered pair.
    """
    from . import helical_pitch_map as maps

    K, M = len(images), len(cand_images)
    if K < 2 or M == 0:
        return np.full(M, np.nan)
    oriented = maps.straighten_classes(
        list(images) + list(cand_images),
        list(zdir) + list(cand_zdir),
        apix,
        apix,
    )
    offsets = class_pixel_offsets(
        oriented,
        list(range(K + M)),
        apix,
        pairs=[(k, K + m) for m in range(M) for k in range(K)],
    )
    every = np.concatenate([np.asarray(phases, float), np.asarray(cand_phases, float)])
    return pixel_agreement(offsets, every, period, sign)[K:]
