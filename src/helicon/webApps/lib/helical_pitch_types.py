"""Helical types from class averages and the classes each filament uses.

Filaments of different helical types have class averages that differ in
appearance, and use different classes. Two kinds of evidence, then, neither of
which is the repeat (types can have nearly the same repeat):

* the *pixels*: two class averages of one structure, at different azimuths, can be
  registered onto each other (:func:`helical_pitch_pixel.class_pixel_offsets`),
  and the correlation at the best offset is high; that of two structures is lower;
* the *usage*: the classes of one type are used by the same filaments.

The Types card of AbInitio3D groups the filaments (:func:`marker_groups`): one
split at a time, by a set of alike classes that the filaments of one side use and
those of the other never do. The earlier grouping of the classes
(:func:`pixel_groups`, with a vote of each filament) is kept for comparison.

This is a prototype.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

from . import denovo3d_register as R
from . import helical_pitch_phase as ph


# Similarity matrices of class_similarity, by the averages' contents and the
# settings: the same Class2D job grouped again (in any session) is not registered
# again. A matrix is a few hundred kB at most; the oldest go first.
SIMILARITY_CACHE_SIZE = 16
_SIMILARITY_CACHE = {}
_SIMILARITY_LOCK = threading.Lock()


def class_similarity(
    images, classes, apix=1.0, max_width=128, progress=None, modulation=False
):
    """Correlation of the registered class averages, every pair, any orientation.

    Each pair is registered over the in-plane rotation, the shift across the
    filament, the shift along it and the four orientations (a class of the same
    structure may sit turned by 180 degrees), and the correlation at the best
    placement is kept. It is normalised over the columns the two averages overlap
    in, never less than 35% of the width (:data:`denovo3d_register.MIN_OVERLAP`).

    Measured on EMPIAR-10230 (28 classes of a known type, 231 same-type and 147
    different-type pairs):

    * A short overlap raises the correlation of two unrelated averages by chance
      (different types: 0.81 at an overlap of 40% or less, 0.73 at 90% or more;
      same type: 0.92 and 0.98), but did not blur the separation: in every overlap
      bin the same-type pairs all scored above the different-type ones (AUC 1.0 per
      bin, 0.997 overall).
    * The body of a filament correlates between any two averages, so the
      correlation of different types is not low (0.76 against 0.96). With
      ``modulation`` the mean of each row along the axis is removed first, so only
      what varies along the axis is compared: 0.55 against 0.92, a wider gap, AUC
      0.988. **It is not the default**: for grouping all the classes it did worse.
      The straighter averages (SF) vary little along the axis, and the
      cross-section shape that the plain correlation keeps is what tells the types
      apart; on EMPIAR-10230 the SF classes were no longer grouped, and on
      EMPIAR-10944 two groups of repeats 650 and 710 A were joined.
    * The averages of a structure cannot be compared far from their phases (little
      overlap), but the average linkage of :func:`pixel_groups` uses every pair.

    Returns
    -------
    np.ndarray
        ``(K, K)`` symmetric matrix in [0, 1] with ones on the diagonal.
    """
    import hashlib

    classes = [int(c) for c in classes]
    digest = hashlib.blake2b(digest_size=16)
    for c in classes:
        im = np.ascontiguousarray(images[c], dtype=np.float32)
        digest.update(str(im.shape).encode())
        digest.update(im.tobytes())
    key = (digest.hexdigest(), float(apix), int(max_width), bool(modulation))
    with _SIMILARITY_LOCK:
        if key in _SIMILARITY_CACHE:
            if progress is not None:
                progress(1, 1)
            return _SIMILARITY_CACHE[key].copy()
    S = _class_similarity(images, classes, apix, max_width, progress, modulation)
    with _SIMILARITY_LOCK:
        _SIMILARITY_CACHE[key] = S.copy()
        while len(_SIMILARITY_CACHE) > SIMILARITY_CACHE_SIZE:
            _SIMILARITY_CACHE.pop(next(iter(_SIMILARITY_CACHE)))
    return S


def _class_similarity(images, classes, apix, max_width, progress, modulation):
    """:func:`class_similarity` without its cache."""
    K = len(classes)
    prepared, _ = ph._register_prepared(images, classes, max_width)
    if modulation:
        prepared = {
            c: np.asarray(im, dtype=np.float32)
            - np.asarray(im, dtype=np.float32).mean(axis=1, keepdims=True)
            for c, im in prepared.items()
        }
    images = [prepared[c] for c in classes]
    if len({np.shape(im) for im in images}) == 1:
        # each image's turned and shifted copies made once, the pairs in bulk:
        # the same correlations as registering pair by pair, many times faster
        return R.best_corr_matrix(images, R.FLIPS, progress=progress)
    todo = [(i, j) for x, i in enumerate(classes) for j in classes[x + 1 :]]
    S = np.eye(K)
    if todo:
        found = ph._register_all(prepared, todo, R.FLIPS, progress, refine=False)
        row = {c: r for r, c in enumerate(classes)}
        for (i, j), res in zip(todo, found):
            S[row[i], row[j]] = S[row[j], row[i]] = float(np.clip(res["corr"], 0, 1))
    return S


def pixel_groups(similarity, cut=0.15, min_classes=3):
    """Groups of classes whose averages look alike: average-linkage clustering.

    The distance of two classes is one minus the correlation of their registered
    averages, and groups are cut at ``cut`` (a mean correlation of 1 - ``cut``
    between the members of a group). Averages of one structure, at different
    azimuths register at 0.9 and above; those of two structures lower (0.77 on
    average for PHF and SF). Any cut from 0.10 to 0.20 gave the same groups on
    EMPIAR-10230, and 0.25 joined the two types; the default of 0.15 gave
    coherent rings for every group on EMPIAR-10230, 10943 and 10944, and one group
    on EMPIAR-10940 and the type-A subset of 10943. The right cut depends on how
    different the structures are, so it is a setting to look at, not a constant.

    Returns
    -------
    np.ndarray
        Group of each class, 0 for the largest group, 1 for the next, and so on;
        -1 for a class in a group of fewer than ``min_classes`` (unassigned: junk,
        or a view of its own).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    S = np.asarray(similarity, dtype=float)
    if len(S) < 2:
        return np.zeros(len(S), dtype=int)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    raw = fcluster(
        linkage(squareform(D, checks=False), method="average"),
        cut,
        criterion="distance",
    )
    sizes = np.bincount(raw)
    order = [g for g in np.argsort(-sizes, kind="stable") if sizes[g] >= min_classes]
    labels = np.full(len(S), -1, dtype=int)
    for new, g in enumerate(order):
        labels[raw == g] = new
    return labels


def pixel_groups_n(similarity, n_groups, cut0=0.15, min_classes=3, low=0.02, high=0.6):
    """:func:`pixel_groups` asked for a number of groups instead of a cut.

    The tree is cut at every height at which the groups change, and the number of
    groups of ``min_classes`` classes or more is counted (the others are left
    unassigned, as in :func:`pixel_groups`). The cut that gives ``n_groups`` of them
    is used; when several do, the one closest to ``cut0``, so that the groups stay
    those of the default cut where they can. If no cut gives that many, the nearest
    number that some cut does.

    Returns
    -------
    labels : numpy.ndarray
        As :func:`pixel_groups`.
    cut : float
        The cut used.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    S = np.asarray(similarity, dtype=float)
    if len(S) < 2:
        return np.zeros(len(S), dtype=int), cut0
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    cuts = np.unique(np.clip(np.r_[Z[:, 2], cut0, low, high], low, high))
    counts = []
    for c in cuts:
        sizes = np.bincount(fcluster(Z, c, criterion="distance"))
        counts.append(int(np.sum(sizes >= min_classes)))
    counts = np.asarray(counts)
    target = int(min(counts, key=lambda k: (abs(k - n_groups), k)))
    best = np.flatnonzero(counts == target)
    cut = float(cuts[best[np.argmin(np.abs(cuts[best] - cut0))]])
    return pixel_groups(S, cut, min_classes), cut


def default_n_groups(similarity, cut0=0.15, min_classes=3):
    """The number of groups the default cut gives."""
    labels = pixel_groups(similarity, cut0, min_classes)
    return int(labels.max()) + 1


def suggest_cut(similarity, low=0.05, high=0.7, default=0.25):
    """A cut for :func:`pixel_groups` from the tree: the middle of its largest gap.

    Averages of one structure join at small distances, different structures at
    larger ones; the cut is put in the middle of the largest gap between
    successive merge heights in ``[low, high]``. Experimental: it needs more data
    sets than were available to be trusted, and the cut is better looked at (the
    tree, and whether each group's own ring is coherent) than taken.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    S = np.asarray(similarity, dtype=float)
    if len(S) < 3:
        return default
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    h = np.sort(linkage(squareform(D, checks=False), method="average")[:, 2])
    h = h[(h >= low) & (h <= high)]
    if len(h) < 2:
        return default
    i = int(np.argmax(np.diff(h)))
    return float((h[i] + h[i + 1]) / 2.0)


def neighbour_lift(params, class_numbers, window=800.0):
    """How often segments of two classes are neighbours, against the chance level.

    Segments of the same filament at most ``window`` A apart (by the track length)
    count as neighbours, whatever their polarity. For two classes the count of
    neighbouring pairs is divided by what the sizes of the two classes alone lead
    to expect (the product of their shares of all neighbour pairs), so that 1 is
    "as often as chance" and a class pair that never meet on a filament is 0.

    Measured on EMPIAR-10230 (labelled classes): about 1.1 to 1.2 for classes of
    one type and 0.07 to 0.11 for two types.

    Returns
    -------
    np.ndarray
        ``(K, K)`` symmetric matrix, rows and columns in the order of
        ``class_numbers``; 0 for a class without neighbours.
    """
    cls = [int(c) for c in class_numbers]
    index = {c: i for i, c in enumerate(cls)}
    df = params[["rlnMicrographName", "rlnHelicalTubeID", "rlnClassNumber"]].copy()
    df["pos"] = params["rlnHelicalTrackLengthAngst"].astype(float)
    df["k"] = df["rlnClassNumber"].astype(int).map(index)
    df = df[df["k"].notna()]
    key = df.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False).ngroup()
    order = np.lexsort((df["pos"].values, key.values))
    pos = df["pos"].values[order]
    fil = key.values[order]
    k = df["k"].values[order].astype(int)
    K = len(cls)
    N = np.zeros((K, K))
    shift = 1
    while shift < len(pos):
        near = (fil[shift:] == fil[:-shift]) & (pos[shift:] - pos[:-shift] <= window)
        if not near.any():
            break
        np.add.at(N, (k[:-shift][near], k[shift:][near]), 1.0)
        shift += 1
    N = N + N.T
    total = max(N.sum(), 1e-12)
    expected = np.outer(N.sum(1), N.sum(1)) / total
    return np.where(expected > 0, N / np.maximum(expected, 1e-12), 0.0)


def usage_matrix(params, class_numbers):
    """How many segments of each filament are in each class.

    Returns
    -------
    counts : np.ndarray
        ``(n_filaments, K)``, columns in the order of ``class_numbers``.
    keys : list of tuple
        ``(micrograph, tube)`` of each row.
    """
    cls = [int(c) for c in class_numbers]
    df = params[params["rlnClassNumber"].astype(int).isin(cls)]
    table = (
        df.groupby(["rlnMicrographName", "rlnHelicalTubeID", "rlnClassNumber"])
        .size()
        .unstack("rlnClassNumber", fill_value=0)
        .reindex(columns=cls, fill_value=0)
    )
    return table.values.astype(float), list(table.index)


def usage_similarity(counts):
    """Cosine similarity of the classes by the filaments that use them."""
    frac = counts / np.maximum(counts.sum(1, keepdims=True), 1e-12)
    norm = np.sqrt((frac**2).sum(0))
    gram = frac.T @ frac
    return gram / np.maximum(np.outer(norm, norm), 1e-12)


def affinity(similarity, usage, weight_pixel=0.5):
    """The classes' affinity: the pixels' similarity and the usage, mixed."""
    return weight_pixel * similarity + (1.0 - weight_pixel) * usage


def neighbour_affinity(similarity, lift, weight=0.3, saturation=1.5):
    """The classes' affinity: the correlation of their averages, with how often
    their segments are neighbours on a filament.

    ``(1 - weight) * similarity + weight * min(lift, saturation) / saturation``:
    a pair of classes whose averages correlate well but whose segments never meet
    on a filament is pulled down, and one that meets as often as chance (or more)
    is not. With ``weight = 0`` it is the similarity alone.

    Not used by the Types card of the tab, because it was not a clear gain. With a
    weight of 0.3 the grouping of EMPIAR-10230 (PHF and SF) was right for every cut
    from 0.20 to 0.40 (without it, 0.10 to 0.20), but a single-type data set
    (EMPIAR-10940, the type-A subset of EMPIAR-10943) was split into 3 or 4 groups at
    cuts of 0.15 to 0.25 (the neighbour frequency of two classes of one type
    depends on how far apart they are on the repeat), and on EMPIAR-10944 it lost
    the groups with coherent rings that the correlation alone gave (650, 710 and
    431 A; with the frequency 677 A, 660 A and an incoherent one).
    """
    L = np.clip(np.asarray(lift, dtype=float), 0.0, saturation) / saturation
    return (1.0 - weight) * np.asarray(similarity, dtype=float) + weight * L


def spectral_groups(A, n_groups=2, seed=0):
    """Group the classes by the leading eigenvectors of the normalised affinity."""
    from scipy.cluster.vq import kmeans2

    A = np.clip(np.asarray(A, dtype=float), 0, None)
    np.fill_diagonal(A, 0.0)
    d = np.maximum(A.sum(1), 1e-12)
    L = A / np.sqrt(np.outer(d, d))
    vals, vecs = np.linalg.eigh(L)
    emb = vecs[:, -n_groups:]
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    _, labels = kmeans2(emb, n_groups, minit="++", seed=seed)
    return labels


def assign_filaments(counts, class_groups, n_groups=2, min_share=0.6):
    """The group of every filament: where most of its segments are.

    Returns
    -------
    labels : np.ndarray
        Group of each filament, or -1 where no group has ``min_share`` of the
        segments.
    share : np.ndarray
        The share of the winning group.
    """
    class_groups = np.asarray(class_groups)
    # classes in no group (-1) do not vote
    by_group = np.stack(
        [counts[:, class_groups == g].sum(1) for g in range(n_groups)], axis=1
    )
    total = np.maximum(by_group.sum(1), 1e-12)
    share = by_group.max(1) / total
    labels = np.where(share >= min_share, by_group.argmax(1), -1)
    return labels, share


# ── filaments grouped by the classes only one type uses ──
#
# Measured on a Class2D job of five types solved by hand (benchmarks/multi_type):
# grouping the classes first and letting each filament vote put the filaments of
# two types that share most of their classes (a doublet and a triplet) in one group.
# What tells them apart is a set of views only one of them shows (the triplet's
# wide, three-strand views): its filaments have several segments in that set, the
# other's none. Looking for such sets, one split at a time, found it without labels
# and gave a triplet group 99% pure, where clustering the filaments' class usage
# (k-means, Ward, Gaussian mixtures, HDBSCAN, DBSCAN, OPTICS) did not. A filament
# shorter than one turn of its type may never show such a view: those stay with
# the other type, and the group is named as possibly holding them.

# A marker set used by less than this share of its filaments' segments can be
# missed by a short filament of that type
SPARSE_MARKER = 0.5

# The most groups the Types card offers
MAX_MARKER_GROUPS = 12


def marker_candidates(similarity, max_share=1.0 / 3.0):
    """Sets of classes whose averages look alike: every branch of the tree.

    The branches of the average-linkage tree of ``1 - similarity`` with at least
    two classes and at most ``max_share`` of them.

    Returns
    -------
    list of numpy.ndarray
        Class positions (rows of ``similarity``) of each set.
    """
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform

    S = np.asarray(similarity, dtype=float)
    K = len(S)
    if K < 3:
        return []
    D = np.clip(1.0 - S, 0.0, None)
    np.fill_diagonal(D, 0.0)
    out = []
    stack = [to_tree(linkage(squareform(D, checks=False), method="average"))]
    while stack:
        node = stack.pop()
        ids = node.pre_order()
        if 2 <= len(ids) <= max(2, int(max_share * K)):
            out.append(np.sort(np.asarray(ids)))
        if not node.is_leaf():
            stack += [node.left, node.right]
    return out


def _binomial_ll(k, n, p):
    """log P(k | n, p) without the binomial coefficient (it cancels)."""
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return k * np.log(p) + (n - k) * np.log1p(-p)


def binomial_mixture(k, n, weight=None, n_iter=200, tol=1e-7):
    """Two populations of filaments: rarely and often in a set of classes.

    Each filament has ``k`` of its ``n`` segments in the set. Two binomials (a low
    and a high rate) are fitted by EM, and compared with one.

    Parameters
    ----------
    k, n : numpy.ndarray
        Segments in the set, and in all compared classes, of each filament (or of
        each distinct ``(k, n)``, with ``weight`` its number of filaments).
    weight : numpy.ndarray, optional
        Defaults to 1 for each.

    Returns
    -------
    dict
        ``gain`` (log-likelihood gain of two binomials over one, per filament),
        ``low`` and ``high`` (the rates), ``share`` (of filaments in the high
        one).
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    w = np.ones_like(k) if weight is None else np.asarray(weight, dtype=float)
    total = max(float(w.sum()), 1e-12)
    p1 = float((w * k).sum() / max((w * n).sum(), 1e-12))
    ll1 = float((w * _binomial_ll(k, n, p1)).sum())
    r = np.where(k / np.maximum(n, 1.0) > p1, 0.9, 0.1)
    last = -np.inf
    for _ in range(n_iter):
        share = float((w * r).sum() / total)
        low = float((w * (1 - r) * k).sum() / max((w * (1 - r) * n).sum(), 1e-12))
        high = float((w * r * k).sum() / max((w * r * n).sum(), 1e-12))
        a = np.log(max(1.0 - share, 1e-12)) + _binomial_ll(k, n, low)
        b = np.log(max(share, 1e-12)) + _binomial_ll(k, n, high)
        m = np.maximum(a, b)
        ll = float((w * (m + np.log(np.exp(a - m) + np.exp(b - m)))).sum())
        r = np.exp(b - m) / (np.exp(a - m) + np.exp(b - m))
        if ll - last < tol * total:
            break
        last = ll
    if high < low:
        low, high, share = high, low, 1.0 - share
    return dict(gain=(ll - ll1) / total, low=low, high=high, share=share)


def in_high(fit, k, n):
    """Whether each filament belongs to the high-rate population of ``fit``."""
    a = np.log(max(1.0 - fit["share"], 1e-12)) + _binomial_ll(k, n, fit["low"])
    b = np.log(max(fit["share"], 1e-12)) + _binomial_ll(k, n, fit["high"])
    return b > a


def _best_marker_split(counts, rows, candidates, min_child):
    """The set of classes that best splits the filaments ``rows`` in two."""
    C = counts[rows]
    n = C.sum(1).astype(np.int64)
    base = int(n.max()) + 1
    best = None
    for s, members in enumerate(candidates):
        k = C[:, members].sum(1)
        if k.sum() < 50:
            continue
        # filaments with the same (k, n) are fitted once; one integer for each
        # pair (a unique of rows sorts far more slowly)
        code, weight = np.unique(k.astype(np.int64) * base + n, return_counts=True)
        fit = binomial_mixture(code // base, code % base, weight)
        high = in_high(fit, k, n)
        if min(int(high.sum()), int((~high).sum())) < min_child:
            continue
        if best is None or fit["gain"] > best["gain"]:
            best = dict(fit, set=s, high_rows=rows[high], low_rows=rows[~high])
    return best


def marker_groups(
    counts, similarity, max_groups=MAX_MARKER_GROUPS, min_segments=10, min_child=25
):
    """Filaments split, one step at a time, by sets of classes one part never uses.

    At each step, every leaf is offered every set of alike classes
    (:func:`marker_candidates`); the filaments' counts of segments in the set are
    fitted as two populations (:func:`binomial_mixture`), and the leaf and set
    with the largest gain are split: the filaments that use the set (its
    *markers*) from those that do not. The filaments with fewer than
    ``min_segments`` segments in the compared classes are not fitted, and follow
    the split they are most likely in.

    Parameters
    ----------
    counts : numpy.ndarray
        ``(n_filaments, K)`` segments of each filament in each compared class
        (:func:`usage_matrix`), columns in the order of ``similarity``.
    similarity : numpy.ndarray
        ``(K, K)`` from :func:`class_similarity`.
    max_groups : int, optional
        Splits are made until there are this many groups.
    min_segments, min_child : int, optional
        A split leaves at least ``min_child`` fitted filaments on each side.

    Returns
    -------
    list of dict
        The splits in the order they were made: ``gain``, ``low`` and ``high``
        (rates), ``markers`` (class positions of the set), ``parent`` (the group
        split: an index into the groups after the previous split, 0 at first),
        and ``labels`` (the group of every filament after this split, -1 for
        those without segments in the compared classes). The groups after split
        ``i`` are those of ``labels``; the group that used the markers keeps the
        parent's number and the other takes the next one.
    """
    counts = np.asarray(counts, dtype=float)
    n_all = counts.sum(1)
    candidates = marker_candidates(similarity)
    labels = np.where(n_all > 0, 0, -1)
    fitted = n_all >= min_segments
    pending = {0: None}  # group -> its best split (None: not looked at yet)
    splits = []
    while len(pending) < max_groups:
        for g in pending:
            if pending[g] is None:
                rows = np.flatnonzero((labels == g) & fitted)
                found = None
                if len(rows) >= 2 * min_child:
                    found = _best_marker_split(counts, rows, candidates, min_child)
                pending[g] = found or dict(gain=-np.inf)
        g = max(pending, key=lambda x: pending[x]["gain"])
        best = pending[g]
        if not np.isfinite(best["gain"]):
            break
        markers = candidates[best["set"]]
        new = len(pending)
        # every filament of the group, fitted or not, follows the split
        members = np.flatnonzero(labels == g)
        k = counts[members][:, markers].sum(1)
        labels = labels.copy()
        labels[members[~in_high(best, k, n_all[members])]] = new
        splits.append(
            dict(
                gain=float(best["gain"]),
                low=float(best["low"]),
                high=float(best["high"]),
                markers=markers,
                parent=g,
                labels=labels,
            )
        )
        pending[g] = None
        pending[new] = None
    return splits


def default_n_marker_groups(splits, min_gain=2.0):
    """Groups up to the first split whose gain is below ``min_gain``.

    The gain is a log-likelihood per filament. Splits of one type by the views a
    filament happens to cover (a filament shorter than a turn sees part of the
    ring) gained 1.2 to 1.8 on the data set it was measured on; the split of a
    doublet from a triplet by the triplet's own views 2.9, and of other types 3.7
    to 11. Fewer groups than types is the worse error (a mixed group gives no
    structure; two groups of one type can be merged later), so the cut is set
    low.
    """
    n = 1
    for s in splits:
        if s["gain"] < min_gain:
            break
        n += 1
    return n


def describe_marker_groups(splits, n_groups):
    """The groups after ``n_groups - 1`` splits, and what each may also hold.

    Returns
    -------
    labels : numpy.ndarray
        Group of each filament (-1: not grouped).
    groups : list of dict
        For each group: ``markers`` (class positions that its filaments use and
        the group split from them did not), and ``lacks`` (the groups whose sparse
        markers its filaments lack, so it may hold short filaments of theirs).
    """
    used = splits[: max(0, n_groups - 1)]
    if not used:
        return None, [dict(markers=[], lacks=[])]
    labels = used[-1]["labels"]
    n = len(used) + 1
    markers = [set() for _ in range(n)]
    # each group's descent: the splits it went through, and on which side
    path = [[] for _ in range(n)]
    for i, s in enumerate(used):
        new = i + 1
        path[new] = list(path[s["parent"]]) + [(i, False)]
        path[s["parent"]] = path[s["parent"]] + [(i, True)]
    groups = []
    for g in range(n):
        lacks = set()
        for i, high in path[g]:
            s = used[i]
            if high:
                markers[g].update(int(x) for x in s["markers"])
            elif s["high"] < SPARSE_MARKER:
                # the groups that came from the side that uses the markers
                lacks.update(h for h in range(n) if (i, True) in path[h])
        groups.append(dict(markers=sorted(markers[g]), lacks=sorted(lacks)))
    return labels, groups
