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


# Most of a two-ring fit is the scan of candidate repeats (an eigenvector for
# each). The grid is 2 A, and its peak is interpolated, so the fine scan about it
# needs only a few A, not the 3% the first estimate of a data set searches.
REFINE_FRACTION = 0.005
# After the first round a ring's repeat moves little: it is searched this close
# to its last value, and the whole window again only if it reaches the edge.
NEAR_WINDOW = 0.04


def _fit_ring(pairs, member, around, window, length_scale, near=None):
    """A ring from the filaments in ``member`` (bool per filament), near ``around``.

    With ``near``, the repeat is first searched within ``near`` of ``around``, and
    within ``window`` only if it ends at the edge of that.
    """
    w = member[pairs.pair_filament].astype(float)
    for width in ([near] if near else []) + [window]:
        periods = np.arange(around * (1 - width), around * (1 + width), 2.0)
        ring = ph.estimate_period(
            pairs,
            pair_weights=w,
            periods=periods,
            length_scale=length_scale,
            refine_fraction=REFINE_FRACTION,
        )
        if abs(ring["period"] / around - 1) < 0.9 * width:
            break
    return ring


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
            rings[t] = _fit_ring(
                pairs,
                member,
                p[t],
                window,
                length_scale,
                near=NEAR_WINDOW if rounds > 1 else None,
            )
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
    starts = [np.asarray(labels)] + [
        rng.integers(0, 2, pairs.n_filaments) for _ in range(restarts - 1)
    ]
    num = den = 0.0
    for k in range(folds):
        a, b = _heldout_fold(pairs, fold, k, starts, periods, one_period, **kw)
        num += a
        den += b
    return num / max(den, 1e-12)


def _heldout_fold(pairs, fold, k, starts, periods, one_period, **kw):
    """One fold of :func:`heldout_gain`: rings fitted to the filaments of the other
    folds, judged on those of fold ``k``. Returns the weighted sum of the gain over
    them, and the sum of their weights."""
    train = fold != k
    test = ~train
    two = None
    for start in starts:
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
        return 0.0, 0.0
    one = _fit_ring(pairs, train, one_period, 0.12, kw.get("length_scale", 800.0))
    keep, swap, weight = filament_scores(pairs, one["period"], one["phases"])
    single = np.maximum(keep, swap)
    w = weight[test]
    return float(((two["best"] - single)[test] * w).sum()), float(w.sum())


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


# The two repeats of a split must differ by this fraction: two rings a few percent
# apart split one type by the spread of its filaments' repeats
MIN_APART = 0.08


def two_pitch_prepare(params, class_ids, max_filaments=None, seed=0):
    """The first step of :func:`two_pitch_test`: the ring of one group.

    The segments (of a random sample of ``max_filaments`` filaments, if given) go
    through :func:`helical_pitch_phase.analyze`; the segment pairs, the single
    ring, the starting split for two rings and the ring fit are kept for the
    two-ring fits (:func:`two_pitch_start`, :func:`two_pitch_fold`), which can
    then run apart, in other processes.

    Returns
    -------
    dict
        ``pairs``, ``period`` (one ring), ``phases``, ``start`` (labels),
        ``periods`` (the two starting repeats), ``fit`` (:func:`ring_fit`) and
        ``seed``.
    """
    keys = params.groupby(
        ["rlnMicrographName", "rlnHelicalTubeID"], sort=False
    ).ngroup()
    if max_filaments is not None and keys.max() + 1 > max_filaments:
        rng = np.random.default_rng(seed)
        keep = rng.choice(int(keys.max()) + 1, max_filaments, replace=False)
        params = params[keys.isin(keep).values]
    res = ph.analyze(params, class_ids=[int(c) for c in class_ids], n_boot=2)
    fit = ring_fit(res, params)
    cl = [int(c) for c in res.class_ids]
    pairs = ph.prepare_pairs(res.params_used, class_ids=cl, max_sep=4000.0)
    est = ph.estimate_period(pairs)
    est, _ = ph.sync_filament_directions(pairs, est)
    fil = res.filaments
    own = {
        (m, int(t)): q
        for m, t, q, ok in zip(
            fil.rlnMicrographName, fil.rlnHelicalTubeID, fil.pitch, fil.fitted
        )
        if ok
    }
    pitch = np.array([own.get((x[0], int(x[1])), np.nan) for x in pairs.filament_keys])
    start = starting_labels(pairs, pitch, np.random.default_rng(seed))
    return dict(
        pairs=pairs,
        period=float(est["period"]),
        phases=est["phases"],
        start=start,
        periods=(est["period"] * 0.93, est["period"] * 1.07),
        fit=fit,
        seed=seed,
    )


def _start_labels(prep, k):
    """Start ``k`` of the two-ring fits: the prepared split, then random ones."""
    if k == 0:
        return np.asarray(prep["start"])
    rng = np.random.default_rng([prep["seed"], k])
    return rng.integers(0, 2, prep["pairs"].n_filaments)


def two_pitch_start(prep, k):
    """The two-ring fit to all the filaments from start ``k``."""
    return fit_two_rings(prep["pairs"], _start_labels(prep, k), prep["periods"])


def two_pitch_fold(prep, k, folds=2, restarts=4):
    """Held-out fold ``k`` of :func:`heldout_gain` for a prepared group.

    Returns
    -------
    tuple
        The fold's sum of (two-ring minus one-ring agreement) times weight over
        its held-out filaments, and the sum of their weights.
    """
    pairs = prep["pairs"]
    fold = np.random.default_rng(prep["seed"]).integers(0, folds, pairs.n_filaments)
    return _heldout_fold(
        pairs,
        fold,
        k,
        [_start_labels(prep, i) for i in range(restarts)],
        prep["periods"],
        prep["period"],
    )


def two_pitch_finish(prep, fits, folds):
    """The result of :func:`two_pitch_test` from its pieces: the two-ring fits
    from every start and the held-out folds."""
    pairs = prep["pairs"]
    fits = [f for f in fits if f is not None]
    if not fits:
        return dict(period=prep["period"], fit=prep["fit"], heldout=np.nan)
    best = max(fits, key=lambda f: f["objective"])
    num = sum(x[0] for x in folds)
    den = sum(x[1] for x in folds)
    order = np.argsort(best["periods"])
    ring = np.argsort(order)[np.asarray(best["labels"], dtype=int)]
    return dict(
        period=prep["period"],
        fit=prep["fit"],
        periods=tuple(float(best["periods"][i]) for i in order),
        heldout=float(num / max(den, 1e-12)),
        rings={
            (str(x[0]), int(x[1])): int(r) for x, r in zip(pairs.filament_keys, ring)
        },
        counts=tuple(int(np.sum(ring == i)) for i in (0, 1)),
    )


def two_pitch_test(params, class_ids, max_filaments=None, seed=0, restarts=4, folds=2):
    """One ring or two, for the filaments of one group.

    The segments go through :func:`helical_pitch_phase.analyze` (a ring of the
    classes and a repeat); one ring and two rings are fitted, and the gain of the
    second ring is measured on filaments it was not fitted to
    (:func:`heldout_gain`). :func:`two_pitch_tests` does the same for several
    groups at once, in parallel.

    Parameters
    ----------
    params : pandas.DataFrame
        Class2D parameters of the group's filaments.
    class_ids : sequence of int
        The classes to place on the ring (class numbers).
    max_filaments : int, optional
        Fit a random sample of this many filaments (for the gain only: then
        ``rings`` covers the sample).
    restarts : int, optional
        Starts of each two-ring fit: the filaments split at their median repeat,
        and ``restarts - 1`` at random. Defaults to 4.
    folds : int, optional
        Held-out folds. Defaults to 2.

    Returns
    -------
    dict
        ``period`` (one ring), ``fit`` (:func:`ring_fit`), ``periods`` (the two,
        ascending), ``heldout`` (gain), ``rings`` (``(micrograph name, int tube)
        -> 0 or 1``, 0 for the shorter repeat) and ``counts`` (filaments per
        ring); without ``periods`` and with a NaN gain if two rings could not be
        fitted.
    """
    prep = two_pitch_prepare(params, class_ids, max_filaments, seed)
    fits = [two_pitch_start(prep, k) for k in range(restarts)]
    held = [two_pitch_fold(prep, k, folds, restarts) for k in range(folds)]
    return two_pitch_finish(prep, fits, held)


def _single_threaded(fn, *args):
    """Run ``fn`` with the maths libraries on one thread: the parallel work is
    spread over processes, and the eigenproblems are too small to gain from
    threads (several processes each running many threads only slowed down)."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        return fn(*args)


def two_pitch_tests(
    jobs, max_filaments=None, restarts=4, folds=2, seed=0, workers=None, progress=None
):
    """:func:`two_pitch_test` for several groups, its pieces run in parallel.

    Each group is first prepared (:func:`two_pitch_prepare`); then its two-ring
    fits from each start and its held-out folds are independent tasks. They all
    go to one pool of single-threaded processes, the largest groups first, and a
    group's fits are queued as soon as it is prepared, so no process waits while
    another group is still running.

    Parameters
    ----------
    jobs : list of tuple
        ``(params, class_ids)`` of each group.
    workers : int, optional
        Processes; by default the CPUs free (:func:`helicon.available_cpu`,
        which respects Slurm and numba's settings) as far as the memory available
        allows 1.5 GB for each, and at most the number of tasks. 1
        runs everything here, in turn. On 12 groups, 8 processes took 53 s and 16
        took 55 s: past a point the slowest group sets the time.
    progress : callable, optional
        Called with (tasks done, tasks in all).

    Returns
    -------
    list
        The result of :func:`two_pitch_test` for each job, or the exception that
        stopped it.
    """
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    n_tasks = len(jobs) * (1 + restarts + folds)
    if workers is None:
        import helicon
        from helicon.lib.system import has_real_threads

        if has_real_threads():
            # each spawned worker holds its own Python, numpy and numba (about
            # 0.5 GB) and a group's segments and pairs
            workers = min(int(helicon.available_cpu(mem_gb_per_cpu=1.5)), n_tasks)
        else:
            # e.g. Pyodide: ProcessPoolExecutor cannot fork/spawn at all
            workers = 1
    results = [None] * len(jobs)
    if workers <= 1 or len(jobs) == 0:
        for i, (p, c) in enumerate(jobs):
            try:
                results[i] = two_pitch_test(p, c, max_filaments, seed, restarts, folds)
            except Exception as e:  # one group's failure does not stop the others
                results[i] = e
            if progress is not None:
                progress((i + 1) * (1 + restarts + folds), n_tasks)
        return results
    preps, fits, held = {}, {}, {}
    done = 0
    # spawned, not forked: a copy of a process running threads (a web server)
    # can hang
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        pending = {}
        for i in sorted(range(len(jobs)), key=lambda i: -len(jobs[i][0])):
            p, c = jobs[i]
            f = ex.submit(
                _single_threaded, two_pitch_prepare, p, c, max_filaments, seed
            )
            pending[f] = ("prepare", i, None)
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                kind, i, k = pending.pop(f)
                done += 1
                if isinstance(results[i], Exception):
                    continue
                try:
                    value = f.result()
                except Exception as e:  # one group's failure does not stop the others
                    results[i] = e
                    done += 0 if kind != "prepare" else restarts + folds
                    continue
                if kind == "prepare":
                    preps[i], fits[i], held[i] = value, [], []
                    for k in range(restarts):
                        g = ex.submit(_single_threaded, two_pitch_start, value, k)
                        pending[g] = ("start", i, k)
                    for k in range(folds):
                        g = ex.submit(
                            _single_threaded, two_pitch_fold, value, k, folds, restarts
                        )
                        pending[g] = ("fold", i, k)
                elif kind == "start":
                    fits[i].append(value)
                else:
                    held[i].append(value)
                if (
                    i in preps
                    and len(fits[i]) == restarts
                    and len(held[i]) == folds
                    and results[i] is None
                ):
                    results[i] = two_pitch_finish(preps[i], fits[i], held[i])
            if progress is not None:
                progress(min(done, n_tasks), n_tasks)
    return results


# Below this ring fit (:func:`ring_fit`) the classes of a group fit no single ring.
# Measured on twelve groups of a data set of five types solved by hand: the groups
# of one type fitted 0.6 to 0.8, the mixtures of three to five types 0.07 to 0.28;
# between 0.3 and 0.6 there were both (mostly one type, with a fifth to a third of
# another). Two types of nearly the same repeat fit one ring well (a doublet and a
# triplet together: 0.82), so a good fit does not prove one type.
MIN_RING_FIT = 0.3


def ring_fit(result, params):
    """How well the classes of an analysis fit one ring: the mean of the class
    fits (:func:`helical_pitch_phase.class_fit`, 1 when every segment is at its
    class's azimuth), each class weighted by its segments."""
    fits = np.asarray(result.class_fit, dtype=float)
    n = params["rlnClassNumber"].astype(int).value_counts()
    w = np.array([float(n.get(int(c), 0)) for c in result.class_ids])
    ok = np.isfinite(fits) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.sum(fits[ok] * w[ok]) / np.sum(w[ok]))


def split_cut(gains, floor=0.08, default=0.10):
    """The held-out gain above which a group holds two repeats.

    Groups of one type gain from a second ring too (0.02 to 0.06 on the data set
    this was measured on: the ring also fits the filaments' spread of repeats and
    class placements), and the level varies between data sets, so the cut comes
    from the groups themselves: their median plus four robust standard
    deviations, and at least ``floor``. With fewer than four groups there is no
    such spread to measure, and ``default`` is used. On that data set, the one
    group of two types (repeats 36% apart) gained 0.18 and the cut was 0.10
    to 0.12.
    """
    g = np.asarray([x for x in gains if np.isfinite(x)], dtype=float)
    if len(g) < 4:
        return default
    med = float(np.median(g))
    mad = float(np.median(np.abs(g - med)))
    return max(floor, med + 4 * 1.4826 * mad)
