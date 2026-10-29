"""The repeat from the segment metadata and the class averages together.

The class ring of :mod:`helical_pitch_phase` reads the repeat ``P`` from the
segments: which class each was put in, and where along its filament it lies. It
is precise when the filaments reach well past half a repeat, and biased, with
no warning from its own resampling, when they do not (filaments half a repeat
long put EMPIAR-10940's 707 A repeat at 520 A).

The class averages say something the segment pairs cannot: whether a helical
map at the twist ``P`` implies explains them. For a candidate repeat, the ring's
azimuths at ``P`` place the averages, a map is solved from two thirds of them
(the denovo3D joint fit), and the other third is aligned to it; the score is how
well the held-out averages fit (``corr``), wherever they align. Held out, an
average cannot flatter a map made from itself. The score uses both sources: with
random azimuths instead of the ring's it peaked 10% off on EMPIAR-10940; with
the ring's it peaked at the repeat even where the ring's own repeat was 26% off.

The two scores are combined over the candidate repeats, each counted in
standard errors (filaments resampled for the ring, classes for the maps), the
ring's only as far as its pairs reach half the candidate (:data:`SUPPORT_FULL`),
and the averages' self-similarity veto applied. The best repeat is then refined
by the ring where its pairs reach, otherwise by the maps.

The relation between the repeat and the pitch is chosen by the same fit:
the pitch is ``csym`` repeats, and of :data:`SYMMETRIES` the one whose maps fit
the held-out averages best is used (EMPIAR-10940: 2, EMPIAR-10943 type A: 1, in
12 of 12 trials). Only one turn of ``360 / csym`` per repeat is considered: not
helices whose averages show their subunits (:mod:`helical_pitch_lattice`).

Measured with the settings below frozen (18 synthetic helices, EMPIAR-10943
type A in 9 stress conditions, EMPIAR-10940): within 5% of the truth in 28 of
30 runs, against 24 of 30 for the ring alone, and never further off than it.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

SYMMETRIES = (1, 2)
FOLDS = 3
# candidate repeats: a scan in steps of 8%, refined in steps of 2% about the
# best of each score
COARSE_STEP = 0.08
FINE_STEP = 0.02
N_BOOT_FILAMENTS = 30
N_BOOT_CLASSES = 200
# the map score resolves the repeat to several percent: it is smoothed over
# that, so that its noise cannot choose within one basin
SMOOTH = 0.05
# the ring counts fully from this share of its pair weight beyond half the
# candidate repeat
SUPPORT_FULL = 0.1
# the best repeat is refined within this fraction of it
WINDOW = 0.10
MIN_CLASSES = 2 * FOLDS


def heldout_fit(images, ring_deg, period, rise, csym, apix, fold):
    """How well the averages of one fold fit the map of the others.

    Parameters
    ----------
    images : list of np.ndarray
        Straightened class averages (:func:`helical_pitch_map.straighten_classes`).
    ring_deg : sequence of float
        Their angles on the ring, 360 per repeat.
    period, rise, apix : float
        Candidate repeat, helical rise and pixel size, in A.
    csym : int
        Repeats per pitch.
    fold : int
        The held-out averages are those whose index is ``fold`` modulo
        :data:`FOLDS`.

    Returns
    -------
    np.ndarray
        Correlation of each held-out average with the map's side projection
        where it aligns best, in index order.
    """
    from .denovo3d_jointsolve import joint_reconstruct
    from .helical_pitch_map import _align_all, _joint_geometry

    # left-handed; the other hand gives the same projections at mirrored azimuths
    twist = -360.0 * rise / (csym * period)
    phis = [-float(a) / csym for a in ring_deg]
    test = np.arange(len(images)) % FOLDS == fold
    apix2d, apix3d, geometry = _joint_geometry(images, apix, rise, 3)
    volume, _ = joint_reconstruct(
        [im for im, t in zip(images, test) if not t],
        [p for p, t in zip(phis, test) if not t],
        twist_degree=twist,
        rise_pixel=rise / apix3d,
        csym=1,
        algorithm=dict(model="gauss"),
        target_apix2d=apix2d,
        cpu=1,
        **geometry,
    )
    held = [im for im, t in zip(images, test) if t]
    aligned = _align_all(held, volume, twist, rise, apix3d, apix2d, 1, 1)
    return np.array([a["corr"] for a in aligned])


def _fits(images, ring_deg, periods, rise, csym, apix, pool):
    """Held-out fit of every average at every period, ``(n_periods, n_classes)``."""
    tasks = [
        (i, f, pool.submit(heldout_fit, images, ring_deg[i], P, rise, csym, apix, f))
        for i, P in enumerate(periods)
        for f in range(FOLDS)
    ]
    out = np.zeros((len(periods), len(images)))
    index = np.arange(len(images))
    for i, f, future in tasks:
        out[i, index % FOLDS == f] = future.result()
    return out


def _t_curve(scores, boot):
    """How many standard errors below the best each period is (0 at the best)."""
    k = int(np.argmax(scores))
    d = boot - boot[:, [k]]
    sd = d.std(axis=0)
    sd[k] = np.inf
    return d.mean(axis=0) / np.maximum(sd, 1e-9)


def _smooth(periods, values):
    """``values`` (periods along the first axis) smoothed over :data:`SMOOTH` in log."""
    lp = np.log(periods)
    kern = np.exp(-0.5 * ((lp[:, None] - lp[None, :]) / SMOOTH) ** 2)
    return (kern / kern.sum(axis=1, keepdims=True)) @ values


def combine(periods, meta, meta_boot, fits, half_support, gate=None, rng=0):
    """The repeat the ring and the maps agree on.

    Parameters
    ----------
    periods : np.ndarray
        Candidate repeats, in A, increasing.
    meta, meta_boot : np.ndarray
        Ring score at each, and for resampled filaments ``(n_boot, n_periods)``.
    fits : np.ndarray
        Held-out fit of each class average, ``(n_periods, n_classes)``.
    half_support : np.ndarray
        Share of the ring's pair weight farther apart than half of each period.
    gate : np.ndarray, optional
        The averages' self-similarity factor at each; 0 rules a period out.

    Returns
    -------
    dict
        ``period`` (the map's, on the grid), ``basin`` (the best period),
        ``source`` (``"metadata"`` where the ring's pairs reach half the basin,
        which then refines it, else ``"map"``), ``period_sd`` (of the map's
        peak over resampled classes), ``map_score`` (smoothed) and
        ``weight`` (the ring's) at each period.
    """
    rng = np.random.default_rng(rng)
    n = fits.shape[1]
    smooth = _smooth(periods, fits)
    picks = [rng.integers(0, n, n) for _ in range(N_BOOT_CLASSES)]
    boot = np.array([smooth[:, p].mean(axis=1) for p in picks])
    map_score = smooth.mean(axis=1)
    weight = np.clip(np.asarray(half_support, float) / SUPPORT_FULL, 0.0, 1.0)
    total = weight * _t_curve(meta, meta_boot) + _t_curve(map_score, boot)
    if gate is not None:
        total = np.where(np.asarray(gate) > 0, total, -np.inf)
    k = int(np.argmax(total))
    near = np.abs(np.log(periods / periods[k])) <= WINDOW
    best = np.flatnonzero(near)[np.argmax(map_score[near])]
    peaks = periods[np.argmax(boot[:, near], axis=1)] if near.sum() > 1 else [0.0]
    return dict(
        period=float(periods[best]),
        basin=float(periods[k]),
        source="metadata" if weight[k] >= 0.5 else "map",
        period_sd=float(np.std(peaks)),
        map_score=map_score,
        weight=weight,
    )


def _half_support(pairs, periods):
    w, d = pairs.W, np.abs(pairs.D)
    order = np.argsort(d)
    beyond = w.sum() - np.concatenate([[0.0], np.cumsum(w[order])])
    k = np.searchsorted(d[order], np.asarray(periods) / 2.0, side="right")
    return beyond[k] / max(w.sum(), 1e-12)


def estimate(
    pairs,
    images,
    apix,
    rise,
    gate=None,
    cpu=None,
    length_scale=800.0,
    progress=None,
):
    """The repeat from the ring's segment pairs and its class averages together.

    Parameters
    ----------
    pairs : SegmentPairs
        The ring's pairs, filament directions synced.
    images : list of np.ndarray
        The class averages of ``pairs.class_ids``, straightened.
    apix, rise : float
        Pixel size of the averages and helical rise, in A.
    gate : callable, optional
        :meth:`helical_pitch_pixel.SelfSimilarity.gate`.
    cpu : int, optional
        Worker processes; at most :func:`helicon.available_cpu`, the default.
    progress : callable, optional
        Called with a short message before each stage.

    Returns
    -------
    dict
        That of :func:`combine`, with ``csym`` (repeats per pitch the averages
        fit best), ``fit_by_csym`` (their best mean held-out fit for each of
        :data:`SYMMETRIES`), ``periods``, ``meta_score`` and ``workers``.

    Raises
    ------
    ValueError
        With fewer than :data:`MIN_CLASSES` classes.
    """
    import helicon

    from . import helical_pitch_phase as ph
    from .pools import TaskPool

    def step(msg):
        if progress is not None:
            progress(msg)

    if len(images) < MIN_CLASSES:
        raise ValueError(
            f"{len(images)} classes: the maps need at least {MIN_CLASSES}, "
            f"{FOLDS} folds of two"
        )
    hist, centers = ph.class_pair_histogram(pairs)

    def ring(periods):
        scores, phases = ph.phase_scan(hist, centers, periods, length_scale)
        return scores, np.degrees(phases)

    lo, hi = np.log(100.0), np.log(max(float(pairs.max_sep), 200.0))
    periods = np.exp(np.arange(lo, hi, COARSE_STEP))
    if gate is not None:
        periods = periods[np.asarray(gate(periods)) > 0]
    if len(periods) == 0:
        raise ValueError("the class averages rule out every candidate repeat")
    meta, ring_deg = ring(periods)
    available = max(1, int(helicon.available_cpu()))
    workers = max(1, min(int(cpu or available), available))
    pool = TaskPool(workers)
    try:
        fits = {}
        for c in SYMMETRIES:
            step(f"maps from the class averages at {len(periods)} repeats, C{c}")
            fits[c] = _fits(images, ring_deg, periods, rise, c, apix, pool)
        fit_by_csym = {c: float(f.mean(axis=1).max()) for c, f in fits.items()}
        csym = max(fit_by_csym, key=fit_by_csym.get)
        fit = fits[csym]
        # refine about the best of each score
        step("refining about the best repeats")
        best = {periods[np.argmax(meta)], periods[np.argmax(fit.mean(axis=1))]}
        extra = np.array(
            sorted(
                p * np.exp(k * FINE_STEP)
                for p in best
                for k in (-2, -1, 1, 2)
                if np.min(np.abs(np.log(periods / (p * np.exp(k * FINE_STEP)))))
                > FINE_STEP / 2
            )
        )
        if gate is not None and len(extra):
            extra = extra[np.asarray(gate(extra)) > 0]
        if len(extra):
            m, d = ring(extra)
            f = _fits(images, d, extra, rise, csym, apix, pool)
            order = np.argsort(np.concatenate([periods, extra]))
            periods = np.concatenate([periods, extra])[order]
            meta = np.concatenate([meta, m])[order]
            fit = np.concatenate([fit, f])[order]
    finally:
        pool.shutdown()
    step("resampling filaments and classes")
    rng = np.random.default_rng(0)
    n_fil = int(pairs.seg_filament.max()) + 1
    meta_boot = []
    for _ in range(N_BOOT_FILAMENTS):
        count = np.bincount(rng.integers(0, n_fil, n_fil), minlength=n_fil)
        h, _ = ph.class_pair_histogram(
            pairs, pair_weights=count[pairs.seg_filament[pairs.I]]
        )
        meta_boot.append(ph.phase_scan(h, centers, periods, length_scale)[0])
    out = combine(
        periods,
        meta,
        np.array(meta_boot),
        fit,
        _half_support(pairs, periods),
        gate=None if gate is None else gate(periods),
    )
    out.update(
        csym=int(csym),
        fit_by_csym=fit_by_csym,
        periods=periods,
        meta_score=meta,
        workers=workers,
    )
    logger.info(
        "repeat from the ring and the class-average maps: %.1f A (%s), C%d; "
        "%d worker processes of %d CPUs available",
        out["period"] if out["source"] == "map" else out["basin"],
        out["source"],
        csym,
        workers,
        available,
    )
    return out
