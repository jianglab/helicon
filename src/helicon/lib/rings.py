"""Medians of a transform over its rings or shells.

Several estimates in helicon take the median of the pixels of a Fourier ring (or
shell): the noise power under the layer lines of a helix, and the scale of the
power spectra that HILL shows. The pixels of a ring depend only on the grid, so
which pixels they are is worked out once (:func:`group_index`), and the median of
each ring of a given transform is then one pass over its pixels
(:func:`grouped_median`).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np


class GroupIndex(NamedTuple):
    """The pixels of each group, as slices of one ordering.

    Attributes
    ----------
    order : numpy.ndarray
        The flat pixel indices sorted by group.
    starts, ends : numpy.ndarray
        Where each group begins and ends in ``order``, one entry per group number
        from 0 to the last (a number nothing falls in has an empty slice).
    """

    order: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def group_index(labels, n_groups: int | None = None) -> GroupIndex:
    """Which pixels belong to which group.

    Parameters
    ----------
    labels : array_like of int
        The group of every pixel, from 0 up. Any shape; the pixels are taken in
        C order.
    n_groups : int, optional
        The number of groups. Defaults to one more than the largest label.

    Returns
    -------
    GroupIndex
    """
    flat = np.asarray(labels).ravel()
    if n_groups is None:
        n_groups = int(flat.max()) + 1 if len(flat) else 0
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n_groups)
    ends = np.cumsum(counts)
    return GroupIndex(order, ends - counts, ends)


def grouped_median(values, index: GroupIndex, min_count: int = 1) -> np.ndarray:
    """The median of the values in each group.

    Parameters
    ----------
    values : array_like
        One value per pixel, in the order the ``index`` was made for (C order).
    index : GroupIndex
        From :func:`group_index`.
    min_count : int, optional
        A group with fewer values has no median. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        One median per group, NaN for a group with fewer than ``min_count``
        values. For an even count it is the mean of the two middle values.
    """
    ordered = np.asarray(values).ravel()[index.order]
    medians = np.full(len(index.starts), np.nan)
    for k, (start, end) in enumerate(zip(index.starts, index.ends)):
        if end - start >= max(min_count, 1):
            medians[k] = np.median(ordered[start:end])
    return medians


def rfft2_rings(shape) -> np.ndarray:
    """The ring of every pixel of an ``rfft2``: its distance from the origin, rounded.

    Parameters
    ----------
    shape : sequence of int
        ``(ny, nx2)``, the shape of the transform (``nx2 = nx // 2 + 1`` for an
        image of ``nx`` columns).

    Returns
    -------
    numpy.ndarray
        Of that shape: ring numbers, 0 at the zero frequency; shared through a
        cache and not writable.
    """
    ny, nx2 = shape
    return _rfft2_rings(int(ny), int(nx2))


@functools.lru_cache(maxsize=8)
def _rfft2_rings(ny: int, nx2: int) -> np.ndarray:
    fy = np.fft.fftfreq(ny)[:, None] * ny
    fx = np.arange(nx2)[None, :].astype(float)
    ring = np.rint(np.hypot(fy, fx)).astype(int)
    ring.setflags(write=False)
    return ring
