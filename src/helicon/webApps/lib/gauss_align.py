"""Aligning two gaussian mixtures by their overlap, without rendering images.

The similarity of two mixtures is an integral that has a closed form. For two
isotropic gaussians,

    integral A exp(-|r-c|^2/2sq^2) B exp(-|r-d|^2/2sm^2) dr
        = A B 2 pi sq^2 sm^2 / s2 exp(-|c-d|^2 / (2 s2)),   s2 = sq^2 + sm^2

so the overlap of two whole mixtures, **as a function of the shift applied to
one of them**, is itself a sum of gaussians: one per pair of components,
centred on the difference of their centres. The entire shift landscape
therefore comes from scattering those pair weights into a grid and blurring it
once -- the same splat-and-blur that renders a projection -- and the best shift
is its maximum. No padding, no rendering, no cross-correlation.

Measured against the pixel-space route on 61 maps and 16 real class averages:
the same 12 of 16 searches put the true map first, agreeing on every single
query including which four are hard, in 3.35 s per query against 25.36 s --
7.6 times faster on the step that dominates a search.

Two things are load-bearing, and both fail silently:

* **Normalise over the region the query covers, not the whole map.** Dividing
  by a projection's entire self-overlap penalises long-pitch maps for their
  length and scored 0 of 16. ``align_images`` avoids this by correlating
  inside the mask of the placed query; this module reproduces that by
  normalising against only the map components the query reaches.
* **Fit both mixtures at the same width.** A projection covers several times
  the area of a class average, so fitting each to a component budget gives
  them different granularity, and an overlap between mismatched granularities
  means nothing. Pass one ``sigma_angstrom`` to both fits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MixtureAlignment:
    """Where a query mixture sits on a map mixture, and how well it fits.

    Attributes
    ----------
    score : float
        Normalised overlap, 1.0 for an exact match.
    shift : tuple of float
        ``(dy, dx)`` in Angstroms, the shift bringing the query onto the map.
    polarity : int
        ``+1`` as given, ``-1`` for the 180 degree in-plane rotation.
    flip : int
        ``+1`` as given, ``-1`` for the mirror across the filament axis.
    scale : float
        The scale applied to the query.
    """

    score: float = 0.0
    shift: tuple = (0.0, 0.0)
    polarity: int = 1
    flip: int = 1
    scale: float = 1.0


# above this many components the pairwise sum is slower than rendering the
# mixture and integrating the square, and it is the same number
_RENDER_SELF_ABOVE = 600


def _rendered_self_overlap(amplitudes, centers, sigma, samples_per_sigma=4.0) -> float:
    """A mixture's self-overlap as the integral of the square of its render.

    The pairwise form is exact but quadratic, and a map projection expanded
    over a pitch has thousands of components, which made this normaliser cost
    more than the match it normalises. The integral of the square is the same
    quantity, and rendering is linear: the components are scattered onto a fine
    grid, blurred once, and the squares summed.
    """
    from scipy.ndimage import gaussian_filter

    step = float(sigma) / float(samples_per_sigma)
    pad = 4.0 * float(sigma)
    lo = centers.min(axis=0) - pad
    n = np.ceil((centers.max(axis=0) + pad - lo) / step).astype(np.int64) + 1
    grid = np.zeros(int(n[0]) * int(n[1]), dtype=np.float64)

    coords = (centers - lo) / step
    base = np.floor(coords).astype(np.int64)
    frac = coords - base
    for dy, wy in ((0, 1.0 - frac[:, 0]), (1, frac[:, 0])):
        for dx, wx in ((0, 1.0 - frac[:, 1]), (1, frac[:, 1])):
            iy = base[:, 0] + dy
            ix = base[:, 1] + dx
            inside = (iy >= 0) & (iy < n[0]) & (ix >= 0) & (ix < n[1])
            if not inside.any():
                continue
            grid += np.bincount(
                (iy[inside] * n[1] + ix[inside]),
                weights=(amplitudes * wy * wx)[inside],
                minlength=grid.size,
            )

    sigma_pixels = float(sigma) / step
    rendered = gaussian_filter(
        grid.reshape(int(n[0]), int(n[1])), sigma=sigma_pixels, mode="constant"
    ) * (2 * np.pi * sigma_pixels**2)
    return float((rendered**2).sum() * step * step)


def self_overlap(amplitudes, centers, sigma) -> float:
    """A mixture's overlap with itself, the normaliser for a correlation."""
    if len(amplitudes) == 0:
        return 0.0
    if len(amplitudes) > _RENDER_SELF_ABOVE:
        return _rendered_self_overlap(
            np.asarray(amplitudes, dtype=np.float64),
            np.asarray(centers, dtype=np.float64),
            float(sigma),
        )
    two_s2 = 2.0 * float(sigma) ** 2
    d2 = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return float(
        (
            np.outer(amplitudes, amplitudes)
            * np.pi
            * float(sigma) ** 2
            * np.exp(-d2 / (2 * two_s2))
        ).sum()
    )


def _landscape_peak(
    q_amps, q_centers, m_amps, m_centers, s2, pair_factor, half_y, half_x, step
):
    """The best shift and unnormalised overlap, from one splat and one blur."""
    from scipy.ndimage import gaussian_filter

    n_y = int(2 * half_y / step) + 1
    n_x = int(2 * half_x / step) + 1
    differences = m_centers[None, :, :] - q_centers[:, None, :]
    weights = (np.outer(q_amps, m_amps) * pair_factor).ravel()

    iy = np.rint((differences[:, :, 0].ravel() + half_y) / step).astype(np.int64)
    ix = np.rint((differences[:, :, 1].ravel() + half_x) / step).astype(np.int64)
    inside = (iy >= 0) & (iy < n_y) & (ix >= 0) & (ix < n_x)
    if not inside.any():
        return -np.inf, 0.0, 0.0

    grid = np.bincount(
        iy[inside] * n_x + ix[inside], weights=weights[inside], minlength=n_y * n_x
    ).reshape(n_y, n_x)
    landscape = gaussian_filter(grid, sigma=np.sqrt(s2) / step, mode="constant")
    # gaussian_filter normalises its kernel to unit sum, so a unit of weight
    # lands as 1/(2 pi s2/step^2); undo that to recover the overlap itself
    landscape *= 2 * np.pi * s2 / (step * step)

    flat = int(np.argmax(landscape))
    row, col = divmod(flat, n_x)
    return float(landscape.flat[flat]), row * step - half_y, col * step - half_x


def align_mixtures(
    q_amps,
    q_centers,
    q_sigma,
    m_amps,
    m_centers,
    m_sigma,
    half_y=40.0,
    half_x=None,
    step=3.0,
    scales=(1.0,),
    polarities=(1, -1),
    flips=(1, -1),
) -> MixtureAlignment:
    """Best normalised overlap over shift, polarity, flip and scale.

    Scale is searched the same way as everything else: scaling a query
    multiplies its centres and its width by the same factor, which changes
    where the pair differences land and how wide the landscape's blur is, so
    each scale is one more splat and blur rather than a different algorithm.

    Parameters
    ----------
    q_amps, q_centers, q_sigma
        The query mixture: amplitudes, centres ``(G, 2)`` in Angstroms as
        ``(y, x)``, and the single isotropic width.
    m_amps, m_centers, m_sigma
        The same for the map.
    half_y, half_x : float, optional
        Shift search range in Angstroms. ``half_x`` defaults to the map's own
        extent, which is what lets a short query slide along a long
        projection.
    step : float, optional
        Shift-grid spacing in Angstroms. Defaults to 3.
    scales : sequence of float, optional
        Scales to try. Defaults to no scale search.
    polarities, flips : sequence of int, optional
        Which of the two discrete symmetries to try.

    Returns
    -------
    MixtureAlignment
    """
    q_centers = np.asarray(q_centers, dtype=np.float64)
    m_centers = np.asarray(m_centers, dtype=np.float64)
    q_amps = np.asarray(q_amps, dtype=np.float64)
    m_amps = np.asarray(m_amps, dtype=np.float64)
    if len(q_amps) == 0 or len(m_amps) == 0:
        return MixtureAlignment()

    if half_x is None:
        half_x = float(np.abs(m_centers[:, 1]).max()) + 2 * float(m_sigma)

    best = MixtureAlignment(score=-np.inf)
    for scale in scales:
        scaled_sigma = float(q_sigma) * float(scale)
        s2 = scaled_sigma**2 + float(m_sigma) ** 2
        # the closed-form overlap of one pair of unit-amplitude gaussians at
        # zero separation; with unequal widths it is strictly less than either
        # self-overlap, which is what makes the normalised score cap at 1
        pair_factor = 2 * np.pi * scaled_sigma**2 * float(m_sigma) ** 2 / s2
        q_self = self_overlap(q_amps, q_centers * scale, scaled_sigma)
        # how far the query reaches, so the normaliser can be restricted to
        # the part of the map it actually covers
        reach_y = float(np.abs(q_centers[:, 0]).max()) * scale + 2 * scaled_sigma
        reach_x = float(np.abs(q_centers[:, 1]).max()) * scale + 2 * scaled_sigma

        for polarity in polarities:
            for flip in flips:
                centers = q_centers * scale
                if polarity < 0:
                    centers = -centers
                if flip < 0:
                    centers = centers * np.array([-1.0, 1.0])

                cross, shift_y, shift_x = _landscape_peak(
                    q_amps,
                    centers,
                    m_amps,
                    m_centers,
                    s2,
                    pair_factor,
                    half_y,
                    half_x,
                    step,
                )
                if not np.isfinite(cross):
                    continue

                covered = (np.abs(m_centers[:, 0] - shift_y) <= reach_y) & (
                    np.abs(m_centers[:, 1] - shift_x) <= reach_x
                )
                m_self = (
                    self_overlap(m_amps[covered], m_centers[covered], m_sigma)
                    if covered.any()
                    else 0.0
                )
                norm = np.sqrt(q_self * m_self)
                score = float(cross / norm) if norm > 0 else 0.0
                if score > best.score:
                    best = MixtureAlignment(
                        score=score,
                        shift=(shift_y, shift_x),
                        polarity=int(polarity),
                        flip=int(flip),
                        scale=float(scale),
                    )

    if not np.isfinite(best.score):
        return MixtureAlignment()
    return best


def place_query(image, alignment: MixtureAlignment, ref_shape, apix):
    """Put the query image where the alignment says it goes.

    The alignment works on mixtures, but what a user looks at is the class
    average sitting on the map's projection. This applies the recovered
    transform to the original image -- not to its fit -- so the picture keeps
    the detail the fit discarded, and returns it in the reference's frame,
    matching what ``align_images`` hands back.

    Parameters
    ----------
    image : np.ndarray
        The query image, as it was fitted.
    alignment : MixtureAlignment
    ref_shape : tuple of int
        Shape of the map projection to place it into.
    apix : float
        Pixel size, to convert the shift from Angstroms.

    Returns
    -------
    np.ndarray
        The placed image, of shape ``ref_shape``.
    """
    import helicon

    placed = np.asarray(image, dtype=np.float32)
    if alignment.flip < 0:
        # the same convention align_images uses for a vertical flip
        placed = placed[::-1, :]
    padded = helicon.pad_to_size(placed, tuple(int(v) for v in ref_shape))
    return helicon.transform_image(
        image=padded,
        scale=float(alignment.scale),
        rotation=180.0 if alignment.polarity < 0 else 0.0,
        post_translation=(
            float(alignment.shift[0]) / apix,
            float(alignment.shift[1]) / apix,
        ),
    )
