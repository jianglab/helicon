"""Two rings: filaments of two helical types that share their classes.

A 2D classification of a mixture (tau PHF and SF filaments, say) puts segments of
both types into the same classes, since a side view of one looks like a side view
of the other at some azimuth. Classes cannot then be split into one set per type
(:func:`helical_pitch_phase.class_groups` finds one group); what differs between
the types is, per filament, how the class label advances along it: each type has
its own repeat and its own phase for every class.

This fits two rings together, each with a repeat and a phase per class, and
gives every filament to the ring along which its class labels advance best:

1. fit each ring to the filaments given to it (:func:`helical_pitch_phase.estimate_period`
   with the pair weights of those filaments);
2. score every filament against both rings, by the mean agreement
   ``sum W cos(2 pi D / P - (phi_a - phi_b)) / sum W`` of its segment pairs, taking
   either direction of the filament;
3. give each filament to the better ring; repeat.

It is a prototype and, like any mixture fit, always finds a "better" two-ring
solution than a one-ring one, so what it finds has to be compared with what it
finds in data of one type (:func:`objective_gain`).
"""

from __future__ import annotations

import numpy as np

from . import helical_pitch_phase as ph


def filament_scores(pairs, period, phases):
    """Per filament, the mean agreement of its pairs with a ring, both directions.

    Returns
    -------
    keep, swap : np.ndarray
        Mean ``cos(2 pi D / P - (phi_a - phi_b))`` of the filament's pairs as they
        are, and with the filament reversed (``D`` negated); 0 for a filament
        without a pair.
    weight : np.ndarray
        The pair weight of each filament.
    """
    n = pairs.n_filaments
    pf = pairs.pair_filament
    a, b = pairs.seg_class[pairs.I], pairs.seg_class[pairs.J]
    ang = 2 * np.pi * pairs.D / period
    dphi = phases[a] - phases[b]
    weight = np.bincount(pf, weights=pairs.W, minlength=n)
    safe = np.maximum(weight, 1e-12)
    keep = np.bincount(pf, weights=pairs.W * np.cos(ang - dphi), minlength=n) / safe
    swap = np.bincount(pf, weights=pairs.W * np.cos(-ang - dphi), minlength=n) / safe
    return keep, swap, weight


def _fit_ring(pairs, member, around, window, length_scale):
    """A ring from the filaments in ``member`` (bool per filament), near ``around``."""
    w = member[pairs.pair_filament].astype(float)
    periods = np.arange(around * (1 - window), around * (1 + window), 2.0)
    return ph.estimate_period(
        pairs, pair_weights=w, periods=periods, length_scale=length_scale
    )


def fit_two_rings(
    pairs,
    labels,
    periods,
    window=0.12,
    n_iter=20,
    length_scale=800.0,
    train=None,
):
    """Alternate the fit of two rings and the giving of filaments to them.

    Parameters
    ----------
    pairs : SegmentPairs
    labels : np.ndarray
        The starting ring of every filament, 0 or 1.
    periods : tuple of float
        The starting repeat of each ring; each is then searched within ``window``
        of its last value.
    window : float, optional
        Fraction of the repeat searched about the last repeat. Defaults to 0.12.
    n_iter : int, optional
        Most rounds. Defaults to 20.
    train : np.ndarray, optional
        Boolean per filament: only these take part in fitting the rings. The
        others are given to the better ring all the same, which is how a ring pair
        is judged on filaments it was not fitted to (:func:`heldout_gain`).

    Returns
    -------
    dict
        ``labels``, ``periods`` (2,), ``phases`` (2, K), ``scores`` (the rings'
        own coherence), ``objective`` (mean agreement of all pairs with the ring
        their filament is in), ``n_filaments`` per ring, ``rounds``, and per
        filament ``best`` (its agreement with its ring) and ``weight``.
    """
    labels = np.asarray(labels, dtype=int).copy()
    use = np.ones(len(labels), bool) if train is None else np.asarray(train, bool)
    p = [float(periods[0]), float(periods[1])]
    rings = [None, None]
    rounds = 0
    for rounds in range(1, n_iter + 1):
        for t in (0, 1):
            member = (labels == t) & use
            if member.sum() < 3:
                return None
            rings[t] = _fit_ring(pairs, member, p[t], window, length_scale)
            p[t] = rings[t]["period"]
        scores = []
        for t in (0, 1):
            keep, swap, weight = filament_scores(pairs, p[t], rings[t]["phases"])
            scores.append(np.maximum(keep, swap))
        new = np.argmax(np.vstack(scores), axis=0)
        new[weight == 0] = labels[weight == 0]
        if np.array_equal(new, labels):
            break
        labels = new
    total = max(float(weight.sum()), 1e-12)
    best = np.vstack(scores).max(axis=0)
    return dict(
        labels=labels,
        periods=np.array(p),
        phases=np.vstack([rings[0]["phases"], rings[1]["phases"]]),
        scores=np.array([rings[0]["score"], rings[1]["score"]]),
        objective=float((best * weight).sum() / total),
        n_filaments=np.array([(labels == 0).sum(), (labels == 1).sum()]),
        rounds=rounds,
        best=best,
        weight=weight,
    )


def one_ring_objective(pairs, period, phases):
    """Mean agreement of all pairs with a single ring (each filament either way)."""
    keep, swap, weight = filament_scores(pairs, period, phases)
    return float((np.maximum(keep, swap) * weight).sum() / max(weight.sum(), 1e-12))


def heldout_gain(pairs, labels, periods, one_period, folds=2, seed=0, restarts=4, **kw):
    """What a second ring adds on filaments it was not fitted to.

    A fit with more freedom always agrees better with the data it was fitted to,
    so the gain of :func:`objective_gain` is positive even for filaments of one
    type. Here the filaments are split into ``folds`` parts; for each, one ring and
    two rings are fitted to the other parts, and both are then judged on this
    part, every filament going to the ring it agrees with better. For one type
    the second ring has nothing to offer there (the gain is about 0), for two it
    does.

    Parameters
    ----------
    pairs : SegmentPairs
    labels, periods : see :func:`fit_two_rings`
        The starting rings.
    one_period : float
        The repeat the single ring starts from.
    folds : int, optional
        Defaults to 2.
    restarts : int, optional
        Starts for each fold: ``labels`` and ``restarts - 1`` random ones; the one
        that agrees best with the filaments it was fitted to is kept (a fit to half
        of the filaments may fall into one ring from a random start). Defaults to 4.

    Returns
    -------
    float
        The gain of the two-ring objective over the one-ring one, over the filaments
        held out, averaged over the folds (weighted by their pairs).
    """
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, pairs.n_filaments)
    num = den = 0.0
    for k in range(folds):
        train = fold != k
        test = ~train
        two = None
        for start in [np.asarray(labels)] + [
            rng.integers(0, 2, pairs.n_filaments) for _ in range(restarts - 1)
        ]:
            trial = fit_two_rings(pairs, start, periods, train=train, **kw)
            if trial is None:
                continue
            fitted = float(
                (trial["best"] * trial["weight"])[train].sum()
                / max(trial["weight"][train].sum(), 1e-12)
            )
            if two is None or fitted > two["fitted"]:
                two = dict(trial, fitted=fitted)
        if two is None:
            continue
        one = _fit_ring(pairs, train, one_period, 0.12, kw.get("length_scale", 800.0))
        keep, swap, weight = filament_scores(pairs, one["period"], one["phases"])
        single = np.maximum(keep, swap)
        w = weight[test]
        num += float(((two["best"] - single)[test] * w).sum())
        den += float(w.sum())
    return num / max(den, 1e-12)


def objective_gain(pairs, two, period, phases):
    """How much better two rings fit than one: the difference of the objectives."""
    return two["objective"] - one_ring_objective(pairs, period, phases)


def starting_labels(pairs, filament_pitch, rng):
    """Starting rings for the filaments: by their own repeat, or at random.

    ``filament_pitch`` is each filament's repeat from its own fit (NaN if none);
    the filaments below the median repeat start in ring 0 and the others in ring 1;
    those without a repeat start at random.
    """
    pitch = np.asarray(filament_pitch, dtype=float)
    labels = rng.integers(0, 2, pairs.n_filaments)
    ok = np.isfinite(pitch)
    if ok.sum() >= 10:
        labels[ok] = (pitch[ok] > np.median(pitch[ok])).astype(int)
    return labels


def segment_scores(pairs, rings, window=800.0, direction=None):
    """Per segment, how well its neighbours' labels follow each ring.

    For every ring ``(period, phases)`` the mean agreement
    ``cos(2 pi D / P - (phi_a - phi_b))`` of the pairs the segment is in whose
    members are at most ``window`` A apart: a local version of
    :func:`filament_scores`.

    Parameters
    ----------
    pairs : SegmentPairs
    rings : sequence of (float, np.ndarray)
        Repeat and class phases of each ring.
    window : float, optional
        Largest separation of the pairs used, in A. Defaults to 800.
    direction : np.ndarray, optional
        +1 or -1 per filament; by default each filament goes the way that agrees
        best with any of the rings.

    Returns
    -------
    scores : np.ndarray
        ``(len(rings), n_segments)``; 0 for a segment without a pair in the window.
    direction : np.ndarray
        The direction used for every filament.
    """
    n_seg = len(pairs.seg_class)
    pf = pairs.pair_filament
    a, b = pairs.seg_class[pairs.I], pairs.seg_class[pairs.J]
    if direction is None:
        best = np.full(pairs.n_filaments, -np.inf)
        direction = np.ones(pairs.n_filaments)
        for period, phases in rings:
            keep, swap, _ = filament_scores(pairs, period, phases)
            direction = np.where(
                np.maximum(keep, swap) > best,
                np.where(keep >= swap, 1.0, -1.0),
                direction,
            )
            best = np.maximum(best, np.maximum(keep, swap))
    near = (np.abs(pairs.D) <= window) * pairs.W
    sign = direction[pf]
    scores = np.zeros((len(rings), n_seg))
    mass = np.bincount(pairs.I, weights=near, minlength=n_seg) + np.bincount(
        pairs.J, weights=near, minlength=n_seg
    )
    safe = np.maximum(mass, 1e-12)
    for t, (period, phases) in enumerate(rings):
        c = near * np.cos(sign * 2 * np.pi * pairs.D / period - (phases[a] - phases[b]))
        scores[t] = (
            np.bincount(pairs.I, weights=c, minlength=n_seg)
            + np.bincount(pairs.J, weights=c, minlength=n_seg)
        ) / safe
    scores[:, mass == 0] = 0.0
    return scores, direction


def assign_segments(pairs, scores, switch_cost=4.0, junk=0.3):
    """The ring (or junk) of every segment, along its filament, with rare switches.

    A Viterbi path through each filament's segments, in the order along it,
    maximising the sum of the segments' scores for the state they are in minus
    ``switch_cost`` for every change of state. States are the rings and a junk
    state of constant score ``junk`` (a stretch that no ring explains better).
    A switch cost large against the number of segments of a filament gives one
    state to the whole filament; a small one lets the state follow the scores.

    Parameters
    ----------
    scores : np.ndarray
        ``(n_rings, n_segments)`` from :func:`segment_scores`.
    switch_cost : float, optional
        Cost of a change of state, in units of a segment's score. Defaults to 4.
    junk : float or None, optional
        Score of the junk state; None leaves it out. Defaults to 0.3.

    Returns
    -------
    np.ndarray
        State of every segment: 0 to ``n_rings - 1`` for a ring, ``n_rings`` for junk.
    """
    n_rings = scores.shape[0]
    full = (
        scores if junk is None else np.vstack([scores, np.full(scores.shape[1], junk)])
    )
    n_states = full.shape[0]
    state = np.zeros(scores.shape[1], dtype=int)
    order = np.lexsort((pairs.seg_pos, pairs.seg_filament))
    f = pairs.seg_filament[order]
    cuts = np.flatnonzero(np.diff(f)) + 1
    for seg in np.split(order, cuts):
        u = full[:, seg]
        n = len(seg)
        best = u[:, 0].copy()
        back = np.zeros((n_states, n), dtype=int)
        for i in range(1, n):
            stay = best
            top = int(np.argmax(best))
            move = best[top] - switch_cost
            choose_stay = stay >= move
            back[:, i] = np.where(choose_stay, np.arange(n_states), top)
            best = np.where(choose_stay, stay, move) + u[:, i]
        k = int(np.argmax(best))
        for i in range(n - 1, -1, -1):
            state[seg[i]] = k
            k = back[k, i]
    return state
