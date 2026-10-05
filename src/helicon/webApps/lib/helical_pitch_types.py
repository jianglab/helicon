"""Helical types from class averages and the classes each filament uses.

Filaments of different helical types have class averages that differ in
appearance, and use different classes. Two kinds of evidence, then, neither of
which is the repeat (types can have nearly the same repeat):

* the *pixels*: two class averages of one structure, at different azimuths, can be
  registered onto each other (:func:`helical_pitch_pixel.class_pixel_offsets`),
  and the correlation at the best offset is high; that of two structures is lower;
* the *usage*: the classes of one type are used by the same filaments.

The classes are grouped by an affinity combining the two, and each filament is
given the group most of its segments are in. A wrong class for many segments of a
filament does not matter while most of them are right, and a filament with no clear
majority is left out as unclassified.

This is a prototype.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import denovo3d_register as R
from . import helical_pitch_phase as ph


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
    classes = [int(c) for c in classes]
    K = len(classes)
    prepared, _ = ph._register_prepared(images, classes, max_width)
    if modulation:
        prepared = {
            c: np.asarray(im, dtype=np.float32)
            - np.asarray(im, dtype=np.float32).mean(axis=1, keepdims=True)
            for c, im in prepared.items()
        }
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
