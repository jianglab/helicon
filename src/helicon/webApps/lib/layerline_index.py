"""Finding the csym, twist and rise of a helix from the layer lines of its images.

The transform of a straightened helix says three things, each answering what
the others cannot:

* **Where the lines are.** They lie at ``Y = m / rise + n / pitch`` with the
  order ``n`` a multiple of ``csym``. That fixes the rise and ``csym * twist``:
  C6 at twist 30, C4 at 45 and C8 at 22.5 put their lines at the same heights.
* **The parity of the orders.** ``J_n(-X) = (-1)**n J_n(X)``, so once the axis is
  at the origin the phases of the two sides of the meridian differ by 0 for even
  ``n`` and by 180 degrees for odd ``n``, on every line. An even csym has only
  even orders, an odd csym has odd orders on alternate lines. A shift of the image
  along the axis multiplies both sides alike and drops out. This tells C3 at twist
  59 from C6 at twist 30, which have the same line heights.
* **The radial profile of each line.** A line of order ``n`` is ``J_n(2 pi R X)``:
  nothing within an onset that grows with ``n`` (the absence of high orders near
  the meridian), then lobes whose places depend on ``n``. Among helices with the
  same line heights this is what should tell the csym. It did so for simulated
  helices (``model="bessel"``), but on real class averages the thin-shell
  ``J_n`` of one radius was less reliable than the plain mask of where each
  order starts (``model="mask"``, the default), which cannot rank symmetries
  that put their lines at the same heights, so these come back as near-equal
  scores: the answer is then among the first few candidates, not always the first.

A candidate is scored by the correlation of the transform with the model of its
lines, times the agreement of its parities with the measured phases. Everything is
a sum over the transform, so the scores of several images of the same helix add and
the noise of one image is not the noise of another.

What the search cannot tell is returned as near-equal scores: when the resolution
limit is below ``1 / rise`` the ``m = 1`` lines are out of reach and every rise gives
the same lines, and helices with the same pitch and symmetry are then the same.

The twist is returned positive: a power spectrum cannot tell the hand of a helix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# below this parity agreement a candidate is not scored further
MIN_PARITY = 0.3
# how the symmetries that put their lines at the same heights are ranked
MODELS = ("orders", "mask", "lines", "onset", "bessel")
# the share of the first pass a worker process should have, in seconds: more
# than its start-up
SECONDS_PER_WORKER = 4.0
# a predicted line is there if the transform along it is this many noise sigmas
LINE_Z_MIN = 3.0
# the share of its predicted lines the transform has, to this power, multiplies the
# score of a helix: a lattice with lines where the transform has gaps loses much
PRECISION_POWER = 2.0
# the share of the best member's order agreement a member of a lattice (the same
# line heights, another csym) reaches, to this power, is what it keeps of the score
MEMBER_POWER = 2.0
# the transform is there where it stands this many noise sigmas out of the noise
ONSET_SIGMA = 3.0
# the radius is only known roughly; these multiples of it are tried
RADIUS_FACTORS = (0.94, 1.0, 1.06)


@dataclass
class Candidate:
    """One answer of :func:`index_images`.

    Attributes
    ----------
    csym : int
    twist : float
        Degrees, positive.
    rise : float
        A.
    score : float
        ``profile * parity``, less what its extra lines cost (the share of its
        predicted lines that the transform has, squared: see
        :meth:`_Spectra.line_precision`); compare candidates of one search only.
    profile : float
        How well the lines of this helix fit the transform: the F1 of the ``"onset"``
        model, or the contrast or correlation of the others.
    parity : float
        How much of the signed layer-line signal its parities explain, from -1
        to 1 (:func:`parity_agreement`).
    radius : float
        The radius, A, at which it matched best.
    """

    csym: int
    twist: float
    rise: float
    score: float
    profile: float
    parity: float
    radius: float


class _Spectra:
    """What the search needs of a set of images, summed over them."""

    def __init__(
        self, shape, apix, cutoff_res, excess, rows, signed, n_images, density=None
    ):
        self.shape = shape
        # the radial density of the helix, (r, rho), from the images (radial_density)
        self.density = density
        self.apix = apix
        self.cutoff_res = cutoff_res
        self.excess = excess  # power / noise - 1, in the layout of rfft2, summed
        self.rows = rows  # the layer-line rows 1 .. ny/2 - 1
        self.signed = signed  # per row: the coherent sum across the meridian, summed
        self.n_images = n_images
        ny, nx = shape
        fy = np.fft.fftfreq(ny)[:, None] * ny
        fx = np.arange(nx // 2 + 1)[None, :].astype(float)
        ring = np.rint(np.hypot(fy, fx)).astype(int)
        k_max = min(0.5 / apix, 1.0 / cutoff_res) * ny * apix
        # the equator is in every candidate and is left out; so is what is beyond the limit
        self.valid = (np.abs(fy) > 3.0) & (np.hypot(fy, fx) < k_max)
        self._ring = ring[self.valid]
        self._count = np.maximum(
            np.bincount(self._ring, minlength=ring.max() + 1), 1
        ).astype(float)
        self._excess_valid = excess[self.valid]
        self._fy = np.fft.fftfreq(ny, apix)
        self._fx = np.fft.rfftfreq(nx, apix)

    def observed(self, z_on=ONSET_SIGMA):
        """Where the transform is there: its excess, smoothed over 3 x 3 pixels, is above ``z_on`` noise sigmas."""
        from scipy import ndimage

        smooth = (
            ndimage.uniform_filter(self.excess, size=(3, 3), mode="nearest")
            * 3.0
            / np.sqrt(self.n_images)
        )
        return smooth > z_on

    def onset_f1(self, heights, model, band=2):
        """How well a model of the lines agrees with where the transform is, near the lines, in both ways.

        Within ``band`` rows of every line of the lattice (whatever its order) and
        up to the resolution limit, the pixels where the transform is there and
        the model is not are missing, and those where the model is and the
        transform is not are extra; the F1 score, ``2 TP / (2 TP + missing +
        extra)``, falls with both. The lattice is the same for helices that put
        their lines at the same heights, so the pixels compared are too, and what
        decides is where along X each line starts and stops: the onset of its
        order.
        """
        ny = self.shape[0]
        rows = np.unique(np.rint(np.asarray(heights) * ny * self.apix).astype(int))
        zone = np.zeros(self.excess.shape, dtype=bool)
        n_columns = int(np.searchsorted(self._fx, 1.0 / self.cutoff_res, side="right"))
        for row in rows:
            for offset in range(-band, band + 1):
                if abs(row + offset) > 3:  # the equator is in every candidate
                    zone[(row + offset) % ny, :n_columns] = True
        seen = self.observed()
        predicted = model > 0.5
        hit = (seen & predicted & zone).sum()
        missing = (seen & ~predicted & zone).sum()
        extra = (~seen & predicted & zone).sum()
        return float(2 * hit / max(2 * hit + missing + extra, 1))

    def measured_orders(self, radius, z_min=4.0):
        """The strong layer lines of the transform, each with the Bessel order its profile shows.

        A row is a line if its excess, summed along X up to the resolution limit, is
        ``z_min`` noise sigmas or more and at least that of the rows next to it. Its
        order is the ``n`` whose ``J_n(2 pi radius X)**2`` is most like its profile
        along X (how far from the meridian it starts and where its lobes are), as
        one reads the order of a line in HILL.

        Returns
        -------
        list of tuple
            ``(row, z, order)``, in the order of the rows.
        """
        from scipy.special import jv

        key = "density" if self.density is not None else round(float(radius), 3)
        cache = self.__dict__.setdefault("_orders", {})
        if key in cache:
            return cache[key]
        ny = self.shape[0]
        n_columns = int(np.searchsorted(self._fx, 1.0 / self.cutoff_res, side="right"))
        fx = self._fx[:n_columns]
        if self.density is not None:
            r, rho = self.density
            r_out = float(r[np.flatnonzero(rho > 0)[-1]]) if (rho > 0).any() else radius
            n_max = int(2.0 * np.pi * r_out * fx[-1]) + 8
            shapes = np.array(
                [self.order_profile(n, fx) ** 2 for n in range(n_max + 1)]
            )
        else:
            n_max = int(2.0 * np.pi * radius * fx[-1]) + 8
            shapes = np.array(
                [jv(n, 2.0 * np.pi * radius * fx) ** 2 for n in range(n_max + 1)]
            )
        shapes /= np.maximum(np.linalg.norm(shapes, axis=1, keepdims=True), 1e-30)
        found = []
        last_row = min(int(ny * self.apix / self.cutoff_res), ny // 2 - 1)
        for row in range(4, last_row + 1):
            profile = self.excess[row, :n_columns]
            z = profile.sum() / np.sqrt(n_columns * self.n_images)
            if z < z_min:
                continue
            likeness = shapes @ (profile / max(np.linalg.norm(profile), 1e-30))
            found.append((row, float(z), int(np.argmax(likeness))))
        # a line spreads over the rows next to it: keep the strongest of each
        lines = [
            line
            for line in found
            if all(line[1] >= o[1] for o in found if abs(o[0] - line[0]) == 1)
        ]
        cache[key] = lines
        return lines

    def order_profile(self, n, fx):
        """The amplitude along X of a layer line of order ``n`` for the radial density of the images.

        ``G_n(X) = integral of rho(r) J_n(2 pi r X) 2 pi r dr``: what a single
        shell of radius ``R`` gives (``J_n(2 pi R X)``) summed over the radii of the
        density, so that a helix that is not a thin shell has the lobes and onsets
        its own density gives, and no radius has to be assumed.

        Parameters
        ----------
        n : int
            The Bessel order.
        fx : numpy.ndarray
            The X frequencies, 1/A.

        Returns
        -------
        numpy.ndarray
        """
        from scipy.special import jv

        r, rho = self.density
        w = np.maximum(rho, 0.0) * r * (r[1] - r[0])
        return (w[:, None] * jv(n, 2.0 * np.pi * r[:, None] * fx[None, :])).sum(axis=0)

    def order_agreement(self, heights, orders, radius):
        """How well the lowest order a helix puts on each strong line agrees with the order measured there.

        Near the meridian a layer line is the Bessel function of the lowest order on
        it. A helix whose lowest order at a line is lower than the line shows has a
        line there that is missing from the transform; one whose lowest order is
        higher allows no signal where the transform has some, an extra line. Each
        strong line counts by its significance, for the helix if the orders agree
        (within 1, or a fifth of the order) and against it if not. Helices that put
        their lines at the same heights (C4, C6 and C8 at the same ``csym * twist``)
        differ in these orders by discrete steps, which is what tells them apart,
        and the lattice is needed only to within a row.
        """
        ny = self.shape[0]
        lowest = {}
        for height, order in zip(heights, orders):
            row = int(round(height * ny * self.apix))
            if row > 0:
                lowest[row] = min(lowest.get(row, 1 << 30), abs(int(order)))
        total = 0.0
        for row, z, measured in self.measured_orders(radius):
            predicted = min(lowest.get(row + d, 1 << 30) for d in (-1, 0, 1))
            agree = abs(measured - predicted) <= max(1, 0.2 * predicted)
            total += z if agree else -z
        return total

    def line_profiles(self, heights, orders, radius, half_rows=2, flat=False):
        """The significance ``z`` of every line of a helix: how well the transform along it follows ``J_n**2``.

        Each line has its own amplitude, so the lines that are weak or strong for
        reasons the model does not have (the real density is not a shell at one
        radius) do not count against the helix; only the shape along X does, and
        that shape depends on the order ``n`` of the line. ``z`` is in units of the
        noise. Each line is read on the rows of the transform within ``half_rows`` of
        it, weighted by a Gaussian of the width of a line centred on its exact
        height, so that the score changes smoothly as the helix does. With ``flat``
        the profile along X is 1 up to the resolution limit: the significance of
        the line itself, whatever its order, which is the same for helices that put
        their lines at the same heights.
        """
        from scipy.special import jv

        ny = self.shape[0]
        n_columns = int(np.searchsorted(self._fx, 1.0 / self.cutoff_res, side="right"))
        fx = self._fx[:n_columns]
        offsets = np.arange(-half_rows, half_rows + 1)
        sigma_rows = 1.27
        profiles = {}
        out = []
        for height, order in zip(heights, orders):
            row = int(round(height * ny * self.apix))
            if row <= 3 or row >= ny // 2:
                continue
            n = -1 if flat else abs(int(order))
            if n not in profiles:
                g = (
                    np.ones(n_columns)
                    if flat
                    else jv(n, 2.0 * np.pi * radius * fx) ** 2
                )
                peak = g.max()
                profiles[n] = g / peak if peak > 0 else g * 0.0
            g = profiles[n]
            norm = (g**2).sum()
            if norm < 1e-12:
                continue
            exact = height * ny * self.apix
            weight = np.exp(-0.5 * ((row + offsets - exact) / sigma_rows) ** 2)
            along = weight @ self.excess[(row + offsets) % ny, :n_columns] @ g
            out.append(along / np.sqrt(norm * (weight**2).sum() * self.n_images))
        return np.array(out)

    def line_precision(self, heights, orders, radius, z_min=LINE_Z_MIN):
        """The share of the lines a helix predicts that the transform has.

        A helix whose lattice holds every line of another and more (C3 at 30 degrees
        has the lines of C6 at 30 and, between them, those of the odd multiples of
        3) puts the extra lines where the transform has a gap. They cost it
        nothing in the contrast of the lines it has right, so they are counted here:
        a line is there if the transform along its rows, whatever its profile along
        X, is ``z_min`` noise sigmas or more (:meth:`line_profiles` with ``flat``).
        Only the lines that can be read are counted: those whose rows are within the
        transform and the resolution limit, and that can carry signal at the radius.

        Parameters
        ----------
        heights, orders : numpy.ndarray
            The lines of the helix (``layerlines.layer_line_heights``, with the
            radius, so that the lines that cannot carry signal are left out).
        radius : float
            A.
        z_min : float, optional
            Defaults to :data:`LINE_Z_MIN`.

        Returns
        -------
        float
            From 0 to 1; 1 when no line can be read.
        """
        keep = np.asarray(heights) > 0
        z = self.line_profiles(
            np.asarray(heights)[keep], np.asarray(orders)[keep], radius, flat=True
        )
        if len(z) == 0:
            return 1.0
        return float(np.mean(z >= z_min))

    def contrast(self, model):
        """Correlation of the excess with ``model``, once the mean of the model over each ring is taken out.

        The lines have to be stronger than the transform between them at the
        same resolution, not than the smooth fall-off of the whole transform.
        """
        m = model[self.valid]
        mean = (
            np.bincount(self._ring, weights=m, minlength=len(self._count)) / self._count
        )
        centred = m - mean[self._ring]
        scale = np.sqrt((centred**2).sum())
        if scale < 1e-9:
            return 0.0
        return float((self._excess_valid * centred).sum() / scale)


def _centred_cross_meridian(image, cutoff_res, apix):
    """The excess of the power over the noise, and the signed strength of each layer-line row.

    The axis is along the rows. The signed strength of a row is the sum, over
    its pixels on one side of the meridian, of the product of the transform with
    the conjugate of that of the mirror pixel, as a phasor weighted by how far
    both stand out of the noise, with the offset of the axis from the middle of
    the box taken out. Its real part is positive for even orders and negative
    for odd ones.

    Parameters
    ----------
    image : numpy.ndarray
        2D image of a straight filament, the helical axis along the rows.
    cutoff_res : float
        Resolution limit, A.
    apix : float
        Pixel size, A.

    Returns
    -------
    excess : numpy.ndarray
        ``power / noise - 1`` in the layout of ``rfft2``.
    rows : numpy.ndarray
        The layer-line rows, 1 to ``ny // 2 - 1``.
    signed : numpy.ndarray
        The complex signed strength of each of those rows.
    quality : float
        The signal-weighted mean cosine of twice the phase difference (1 at 0 or 180
        degrees, 0 for random phases) over the strongest 5% of the pixels, from -1
        to 1: how clean the phase difference of the image is (the dark lines of the PD plot of HILL,
        against its mosaic).
    """
    from . import helix_transform

    ny, nx = image.shape
    img = np.roll(np.asarray(image, dtype=float) - np.mean(image), -(nx // 2), axis=1)
    # meridian_offset takes the filament horizontal and gives the offset across it
    offset = helix_transform.meridian_offset(img.T)
    F = np.fft.fft2(img)
    power = np.abs(np.fft.rfft2(img)) ** 2
    excess = power / helix_transform.ring_noise(power) - 1.0
    k_max = int(nx * apix / cutoff_res)
    xs = np.arange(1, min(k_max, nx // 2 - 1) + 1)
    rows = np.arange(1, ny // 2)
    mirror = (-rows) % ny
    cross = F[np.ix_(rows, xs)] * np.conj(F[np.ix_(rows, (-xs) % nx)])
    weight = np.sqrt(
        np.maximum(excess[np.ix_(rows, xs)], 0.0)
        * np.maximum(excess[np.ix_(mirror, xs)], 0.0)
    )
    phasor = weight * cross / np.maximum(np.abs(cross), 1e-30)
    phasor = phasor * np.exp(4j * np.pi * xs[None, :] / nx * offset)
    # cos(2 phase difference): 1 at 0 or 180 degrees, -1 at 90, 0 for random phases
    c2 = ((phasor / np.maximum(weight, 1e-30)) ** 2).real
    strong = weight > np.percentile(weight, 95)
    quality = float(
        (weight[strong] * c2[strong]).sum() / max(weight[strong].sum(), 1e-30)
    )
    return excess, rows, phasor.sum(axis=1), quality


def radial_density(images, apix, cutoff_res):
    """The radial density of the helix: its circular average, from the axial average of the images.

    The mean of a straightened image along the axis is the projection of the helix
    averaged along its axis, which is a tube with the density of a circular
    average. The profile across it is centred (the filament may be off the middle
    by a fraction of a pixel), made even and taken to zero at the edges. It is then
    fitted by a sum of thin cylindrical shells, one every ``apix``, each as wide as
    the resolution limit lets a shell be, with the density of each shell not
    negative (non-negative least squares on the shells' own projections). The
    inverse Abel transform would give the same density but amplify the noise into
    ripples and negative densities.

    Parameters
    ----------
    images : sequence of numpy.ndarray
        Straightened images, the helical axis along the rows.
    apix : float
        Pixel size, A.
    cutoff_res : float
        Resolution limit, A.

    Returns
    -------
    r : numpy.ndarray
        Radii, A, from 0 in steps of ``apix``, as far as half the width of the box.
    rho : numpy.ndarray
        The density at those radii (arbitrary scale, not negative).
    """
    from scipy.optimize import nnls

    nx = images[0].shape[1]
    profile = np.mean(
        [np.mean(np.asarray(i, dtype=float), axis=0) for i in images], axis=0
    )
    edge = max(nx // 16, 2)
    profile = profile - 0.5 * (profile[:edge].mean() + profile[-edge:].mean())
    x = np.arange(nx, dtype=float)
    # the centre: where the profile is most alike its own mirror image
    shifts = np.arange(-(nx // 8), nx // 8 + 1)
    match = np.array(
        [
            np.dot(
                profile, np.interp(2.0 * (nx // 2 + s) - x, x, profile, left=0, right=0)
            )
            for s in shifts
        ]
    )
    k = int(np.argmax(match))
    shift = float(shifts[k])
    if 0 < k < len(shifts) - 1:  # to a fraction of a pixel
        a, b, c = match[k - 1], match[k], match[k + 1]
        if a - 2.0 * b + c < 0:
            shift += 0.5 * (a - c) / (a - 2.0 * b + c)
    centre = nx // 2 + shift
    half = nx // 2
    xs = np.arange(0, half + 1, dtype=float)  # pixels from the centre
    even = 0.5 * (
        np.interp(centre + xs, x, profile, right=0)
        + np.interp(centre - xs, x, profile, left=0)
    )
    r = xs * apix
    # the projection of a shell of radius r_k and Gaussian thickness: its density
    # integrated along the line of sight (y), on a grid of half a pixel
    sigma = 0.5 * cutoff_res
    y = np.arange(0.0, 2.0 * half * apix, 0.5 * apix)
    wy = np.where(y == 0, 1.0, 2.0) * 0.5 * apix  # both signs of y
    radius = np.hypot(r[:, None, None], y[None, None, :])  # (x, 1, y)
    basis = np.empty((len(r), len(r)))
    for kk, rk in enumerate(r):
        basis[:, kk] = (np.exp(-0.5 * ((radius[:, 0, :] - rk) / sigma) ** 2) * wy).sum(
            axis=1
        )
    # a whisper of ridge keeps the shells from splitting a peak into spikes
    ridge = 1e-3 * np.abs(basis).max()
    A = np.vstack([basis, ridge * np.eye(len(r))])
    rho, _ = nnls(A, np.concatenate([even, np.zeros(len(r))]))
    return r, rho


def prepare(images, apix, cutoff_res, radial=False, pd_weight=None):
    """The spectra of a set of images of one helix, summed.

    Parameters
    ----------
    images : sequence of numpy.ndarray
        2D images of the same shape, each with the helical axis along the rows
        (vertical, as HILL keeps it).
    apix : float
        Pixel size, A.
    cutoff_res : float
        Resolution limit, A.
    radial : bool, optional
        Whether to take the radial density of the helix from the images
        (:func:`radial_density`) for the Bessel profile of each layer line, instead
        of a single radius given by the user. Defaults to False: on the 13 default
        class averages of HILL it was worse than the radius (see :func:`index_images`).
    pd_weight : {None, "image"}, optional
        How the phase difference across the meridian (PD) says how far each image
        is to be believed. ``"image"``: an image counts with the square of the
        coherence of its PD (the cosine of twice the phase difference over its
        strongest 5% of pixels), so that images with a clean PD (the dark lines of
        HILL's PD plot) outweigh those with a mosaic. On the 13 default class
        averages of HILL it left the right answer as it was and widened the margin
        of the best candidate over the second by 5 to 10%; weighting each pixel of
        the power by the local coherence as well was worse (C6 first on 7 of 13
        single images, from 10). Defaults to None: all alike.

    Returns
    -------
    _Spectra

    Raises
    ------
    ValueError
        For no images, or images that differ in shape.
    """
    images = [np.asarray(i, dtype=float) for i in images]
    if not images:
        raise ValueError("no images")
    if any(i.shape != images[0].shape for i in images):
        raise ValueError("the images must all have the same shape")
    excess, signed = 0.0, 0.0
    weights = []
    for image in images:
        e, rows, s, quality = _centred_cross_meridian(image, cutoff_res, apix)
        w = max(quality, 0.0) ** 2 if pd_weight == "image" else 1.0
        excess, signed = excess + w * e, signed + w * s
        weights.append(w)
    density = radial_density(images, apix, cutoff_res) if radial else None
    # noise adds in quadrature: the z-scores are scaled by the root of the sum of the
    # squares of the weights, which is the number of images when they are equal
    w = np.array(weights)
    if w.sum() <= 0:  # no image has a clean phase difference
        raise ValueError("no image has a phase difference to weigh by")
    n_effective = float((w**2).sum()) if pd_weight == "image" else len(images)
    return _Spectra(
        images[0].shape,
        apix,
        cutoff_res,
        excess,
        rows,
        signed,
        n_effective,
        density=density,
    )


def parity_agreement(spectra, heights, orders):
    """How well the parities of the orders of the lines of a helix agree with the phases.

    Parameters
    ----------
    spectra : _Spectra
        From :func:`prepare`.
    heights, orders : numpy.ndarray
        The lines of the helix (``layerlines.layer_line_heights``).

    Returns
    -------
    float
        From -1 to 1: the signed strength that the prediction explains (even
        order: 0 degrees, odd order: 180, the lowest order where lines overlap),
        as a share of all the signed strength in the rows. A helix that leaves
        strong rows without a line, or gives a line the wrong parity, scores low.
    """
    ny = spectra.shape[0]
    rows = spectra.rows
    usable = (rows > 3) & (rows / (ny * spectra.apix) <= 1.0 / spectra.cutoff_res)
    predicted = np.zeros(len(rows))
    # the highest orders first, so that the lowest, which dominates near the meridian, wins
    for i in np.argsort(-np.abs(orders)):
        centre = int(round(abs(heights[i]) * ny * spectra.apix))
        sign = 1.0 if int(orders[i]) % 2 == 0 else -1.0
        for row in (centre - 1, centre, centre + 1):
            if 1 <= row < ny // 2:
                predicted[row - 1] = sign
    total = np.abs(spectra.signed[usable]).sum()
    if total <= 0:
        return 0.0
    return float((predicted[usable] * spectra.signed[usable].real).sum() / total)


def bessel_model(spectra, heights, orders, radius, width=3.0):
    """The model of the power of the lines of a helix, in the layout of ``rfft2``.

    Each line is a Gaussian in ``Y`` times ``J_n(2 pi R X)**2`` of its order,
    scaled to a largest value of 1 within the resolution limit, so that every
    line counts alike whatever its order.

    Parameters
    ----------
    spectra : _Spectra
    heights, orders : numpy.ndarray
        The lines of the helix (``layerlines.layer_line_heights``).
    radius : float
        A.
    width : float, optional
        Full width at half maximum of each line, in pixels of the transform.

    Returns
    -------
    numpy.ndarray
    """
    from scipy.special import jv

    ny = spectra.shape[0]
    x_max = 1.0 / spectra.cutoff_res
    sigma = width / (ny * spectra.apix) / 2.3548
    fy, fx = spectra._fy, spectra._fx
    inside = fx <= x_max
    z = 2.0 * np.pi * radius * fx
    abs_orders = np.abs(orders).astype(int)
    kinds = np.unique(abs_orders)
    profiles = np.zeros((ny, len(kinds)))
    radial = np.zeros((len(kinds), len(fx)))
    for k, n in enumerate(kinds):
        ys = heights[abs_orders == n]
        for lo in range(0, len(ys), 64):
            d = fy[:, None] - ys[None, lo : lo + 64]
            profiles[:, k] = np.maximum(
                profiles[:, k], np.exp(-0.5 * (d / sigma) ** 2).max(axis=1)
            )
        j2 = jv(int(n), z) ** 2
        peak = j2[inside].max() if inside.any() else 0.0
        radial[k] = j2 / peak if peak > 0 else 0.0
    return profiles @ radial


def _score(spectra, csym, twist, rise, radii, model="mask"):
    """``(score, fit, parity, radius)`` of one helix: the best over the radii.

    ``model`` says what the lines are and how a fit is measured: ``"orders"`` the
    agreement of the lowest order the helix puts on each strong line with the order
    measured there (:meth:`_Spectra.order_agreement`), which may be negative;
    ``"lines"`` every line of the lattice with its own amplitude and the ``J_n**2``
    profile of its order along X, the fit being the sum of the squares of their
    significances; ``"mask"`` the lines kept from where each order starts
    (``layerlines.layer_line_mask``) and the contrast of the transform on them;
    ``"onset"`` the same lines, and the F1 of where they are against where the
    transform is (missing and extra lines both count against); ``"bessel"`` the
    whole ``J_n**2`` profile of each line and the correlation with the transform.
    The first is quick, for the first pass over many helices.

    Parameters
    ----------
    spectra : _Spectra
        From :func:`prepare`.
    csym : int
        Rotational symmetry of the helix.
    twist : float
        Degrees.
    rise : float
        A.
    radii : sequence of float
        The radii to try, A; only the first for the models that do not use it to rank
        them.
    model : str, optional
        One of :data:`MODELS`. Defaults to ``"mask"``.

    Returns
    -------
    score : float
        ``fit * parity``; 0 for a helix whose parities disagree with the phases.
    fit : float
        How well the lines fit the transform, by ``model``.
    parity : float
        The parity agreement (:func:`parity_agreement`), from -1 to 1.
    radius : float
        The radius, A, at which it fitted best.
    """
    from helicon.lib import layerlines

    x_max = 1.0 / spectra.cutoff_res
    masked = model in ("mask", "onset")  # the others take every line of the lattice
    heights, orders = layerlines.layer_line_heights(
        twist,
        rise,
        csym,
        y_max=x_max,
        radius=radii[0] if masked else None,
        x_max=x_max if masked else None,
    )
    parity = parity_agreement(spectra, heights, orders)
    if parity < MIN_PARITY:
        return 0.0, 0.0, parity, radii[0]
    lattice = heights
    if model == "onset":
        lattice, _ = layerlines.layer_line_heights(
            twist, rise, csym, y_max=x_max, radius=None, x_max=None
        )
    best = (0.0, 0.0, parity, radii[0])
    for radius in radii if model in ("bessel", "lines") else radii[:1]:
        if model == "bessel":
            fit = spectra.contrast(bessel_model(spectra, heights, orders, radius))
        elif model == "lines":
            z = spectra.line_profiles(heights, orders, radius)
            fit = float((np.maximum(z, 0.0) ** 2).sum())
        elif model == "orders":
            fit = spectra.order_agreement(heights, orders, radius)
            return fit * parity, fit, parity, radius
        else:
            ny, nx = spectra.shape
            mask = layerlines.layer_line_mask(
                (ny, nx),
                spectra.apix,
                twist,
                rise,
                csym,
                radius,
                cutoff_res=spectra.cutoff_res,
                width=3.0,
            )
            fit = (
                spectra.onset_f1(lattice, mask)
                if model == "onset"
                else spectra.contrast(mask**2)
            )
        if fit * parity > best[0]:
            best = (fit * parity, fit, parity, radius)
    return best


def _stacked_rings(csym, twist, tolerance):
    """Whether ``csym * twist`` is a whole turn, within ``tolerance`` degrees.

    Such a helix is a stack of rings with no turn between them: its lines are all
    at ``m / rise`` and all of order 0, which the meridian and the equator of any
    helix fill, so it says nothing about the helical symmetry.

    Parameters
    ----------
    csym : int
        Rotational symmetry.
    twist : float
        Degrees.
    tolerance : float
        Degrees.

    Returns
    -------
    bool
    """
    turn = (csym * twist) % 360.0
    return min(turn, 360.0 - turn) <= tolerance


def _first_pass(spectra, csym, twists, rises, radius, tolerance=0.0):
    """The first-pass scores of one symmetry over some twists and all rises (a worker's share).

    Parameters
    ----------
    spectra : _Spectra
        From :func:`prepare`.
    csym : int
        The symmetry.
    twists, rises : sequence of float
        Degrees and A.
    radius : float
        A.
    tolerance : float, optional
        Twists within this many degrees of a stack of rings (:func:`_stacked_rings`)
        are skipped. Defaults to 0.

    Returns
    -------
    list of tuple
        ``(score, contrast, csym, twist, rise)`` for each helix with a positive score.
    """
    from threadpoolctl import threadpool_limits

    found = []
    with threadpool_limits(1):
        for twist in twists:
            if _stacked_rings(csym, twist, tolerance):
                continue
            for rise in rises:
                score, contrast = _score(
                    spectra, csym, float(twist), float(rise), [radius], model="mask"
                )[:2]
                if score > 0:
                    found.append((score, contrast, csym, float(twist), float(rise)))
    return found


_WORKER_SPECTRA = []


def _pool_context():
    """The way to start worker processes: from a fork server that has this module loaded.

    Workers forked from a server process that imported this module once start in
    milliseconds, where a spawned one imports helicon (with numba) again, several
    seconds, and slower when many do it at once; and the server is a process of
    its own, so forking it is safe where forking a web server running threads is
    not. Where there is no fork server (Windows) they are spawned.

    Returns
    -------
    multiprocessing.context.BaseContext
    """
    from .pools import pool_context

    return pool_context()


def _init_worker(spectra):
    """Give a worker process the spectra once, not with every share of the work.

    Parameters
    ----------
    spectra : _Spectra
        From :func:`prepare`.
    """
    _WORKER_SPECTRA[:] = [spectra]


def _first_pass_in_worker(csym, twists, rises, radius, tolerance):
    return _first_pass(_WORKER_SPECTRA[0], csym, twists, rises, radius, tolerance)


def _run_first_pass(
    spectra, csyms, twists, rises, radius, workers, progress, step_twist=1.0
):
    """All of the first pass, in ``workers`` processes when it is worth it.

    Parameters
    ----------
    spectra : _Spectra
        From :func:`prepare`.
    csyms : list of int
        The symmetries to try.
    twists, rises : numpy.ndarray
        Degrees and A.
    radius : float
        A.
    workers : int or None
        Processes; None to time a sample and give each a few seconds of work.
    progress : callable or None
        ``progress(done, total)``, called as the shares finish.
    step_twist : float, optional
        Degrees between twists; a symmetry skips the twists within ``csym *
        step_twist`` of a stack of rings. Defaults to 1.

    Returns
    -------
    list of tuple
        The first-pass scores, as :func:`_first_pass` returns them.
    """
    import logging

    estimate = None
    if workers is None:
        import time

        import helicon

        # a worker holds its own copy of the spectra and its own caches, and
        # costs something to start: time a sample here and give each worker a
        # few seconds of work rather than every CPU a sliver of it
        # the first call pays for imports and caches, which the timing should not
        _first_pass(spectra, csyms[0], twists[:1], rises[:1], radius)
        # a few (csym, twist) pairs spread over both: the cost of a candidate
        # grows with its number of lines, which falls with csym and with twist
        pairs = [
            (csyms[k % len(csyms)], twists[(k * len(twists)) // 7]) for k in range(7)
        ]
        start = time.perf_counter()
        for csym, twist in pairs:
            _first_pass(spectra, csym, [twist], rises, radius)
        estimate = (time.perf_counter() - start) / len(pairs) * len(twists) * len(csyms)
        workers = min(
            int(helicon.available_cpu(mem_gb_per_cpu=0.5)),
            max(1, int(estimate / SECONDS_PER_WORKER)),
        )
    # many small shares, the costly ones (low csym, many lines) first, so that the
    # pool balances itself: a slow share started last would hold up the end
    per_share = max(1, (len(twists) * len(csyms)) // (8 * max(workers, 1)))
    tasks = [
        (csym, twists[k : k + per_share])
        for csym in sorted(csyms)
        for k in range(0, len(twists), per_share)
    ]
    workers = max(1, min(workers, len(tasks)))
    logging.getLogger(__name__).info(
        "layer-line search: %d candidates in %d shares, %d process%s%s",
        len(csyms) * len(twists) * len(rises),
        len(tasks),
        workers,
        "" if workers == 1 else "es",
        f" (about {estimate:.0f} s of work on one)" if estimate is not None else "",
    )
    found = []
    if workers == 1:
        for k, (csym, chunk) in enumerate(tasks):
            found += _first_pass(spectra, csym, chunk, rises, radius, csym * step_twist)
            if progress is not None:
                progress(k + 1, len(tasks))
        return found
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(
        workers,
        mp_context=_pool_context(),
        initializer=_init_worker,
        initargs=(spectra,),
    ) as pool:
        futures = [
            pool.submit(
                _first_pass_in_worker, csym, chunk, rises, radius, csym * step_twist
            )
            for csym, chunk in tasks
        ]
        for k, future in enumerate(as_completed(futures)):
            found += future.result()
            if progress is not None:
                progress(k + 1, len(tasks))
    return found


def index_images(
    images,
    apix,
    radius,
    cutoff_res,
    csyms=(1, 2, 3, 4, 5, 6, 8),
    twists=(2.0, 60.0),
    rises=(3.0, 40.0),
    step_twist=1.0,
    step_rise=0.5,
    top=5,
    lattices=10,
    refine=True,
    model="orders",
    workers=None,
    progress=None,
    radial_profile=False,
    pd_weight=None,
):
    """Search for the csym, twist and rise of a helix from its straightened images.

    Parameters
    ----------
    images : sequence of numpy.ndarray
        2D images of one helix, the same shape, the axis along the rows. Several
        images (different class averages of one helix, say) are better than one:
        their scores add.
    apix : float
        Pixel size, A.
    radius : float
        Outer radius of the helix, A, roughly: ``RADIUS_FACTORS`` times it are
        tried.
    cutoff_res : float
        Resolution limit, A.
    csyms : sequence of int, optional
        The symmetries tried.
    twists, rises : tuple of float, optional
        Ranges of the twist (degrees, positive) and the rise (A) of the first pass.
    step_twist, step_rise : float, optional
        Steps of the first pass.
    top : int, optional
        How many candidates to return.
    lattices : int, optional
        How many of the best places of the lines of the first pass are followed up
        by trying every symmetry that puts its lines there.
    refine : bool, optional
        Whether to place the lines of each lattice to a finer grid before the
        symmetries are ranked.
    model : {"orders", "mask", "lines", "onset", "bessel"}, optional
        How the symmetries that put their lines at the same heights are ranked.
        They differ only in the Bessel order of each line (C4, C6 and C8 at the
        same ``csym * twist`` have orders 4k, 6k and 8k on the same lines).
        ``"orders"`` (the default): the order of each strong line is read from its
        profile along X at the given radius, and a helix scores by how many of them
        equal the lowest order it puts there (:meth:`_Spectra.order_agreement`), so
        that a line it lacks or a line it has in excess counts against it. On HILL's
        default class averages and on simulated C4, C6 and C8 helices it picked the
        symmetry in 14 of 15 cases (the miss: one real class average alone), for any
        lattice within 4 degrees of ``csym * twist`` and 0.4 A of the rise. It needs
        the radius from the images (HILL's mean radius did as well as its tube);
        a radius off by the ratio of two orders swaps them, since where a line
        starts depends on ``n / radius``.
        ``"mask"``: the lines kept from where each order starts
        (``layerlines.layer_line_mask``), scored by their contrast in the transform.
        ``"lines"``, ``"onset"``, ``"bessel"``: per-line amplitudes with ``J_n**2``
        profiles, the F1 of predicted against observed lines, one ``J_n**2`` model
        for all lines; right on simulated helices, unreliable on real ones, where a
        lower symmetry can fit the profiles as well with amplitudes that leave its
        own low-order lines empty.

    workers : int, optional
        Processes for the first pass. By default a sample of it is timed and each
        process gets about :data:`SECONDS_PER_WORKER` of work, up to what
        ``helicon.available_cpu`` allows with half a GB each; the choice is logged.
        1 runs it here.
    progress : callable, optional
        ``progress(done, total)``, called as the shares of the first pass finish.
    radial_profile : bool, optional
        For the ``"orders"`` model: read the Bessel order of each strong line against
        the radial density of the helix taken from the images themselves (the axial
        average of a straightened image is the projection of its circular average:
        :func:`radial_density`), not against a single shell of the given ``radius``.
        The radius has to be estimated and decides which of C4, C6 and C8 is read
        (a radius off by the ratio of two orders swaps them), which is what this
        was to avoid. It does not help yet: on each of the 13 default class averages
        of HILL, searched alone, the radius of HILL (77.6 A) put C6 first on 10, the
        density of the image on 8 (image 11, one of the best, fell to third), and a
        single radius from the density, its centroid (80 to 86 A) on 8 and its peak
        (65 to 96 A) on 3. The density is a circular average that gives every
        layer line the same radial profile; a real helix does not. Defaults to False.

    pd_weight : {None, "image"}, optional
        Whether the coherence of the phase difference across the meridian weighs the
        images of the pooled spectra; see :func:`prepare`.

    Returns
    -------
    list of Candidate
        The best candidates, the best first, each the best of its neighbourhood.
        Check the first few: symmetries that put their lines at the same heights
        (C4 at twist 45, C6 at 30, C8 at 22.5) score within a few per cent of each
        other. On the 13 class averages of HILL's default image, pooled in sets of
        2 to 8, the lattice was the first candidate every time and the exact answer
        was among the first three every time without added noise and in 5 of 6 runs
        of 8 images with three times the noise.

    Raises
    ------
    ValueError
        For an unknown ``model``, no images, or images of different shapes.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, not {model!r}")
    from helicon.lib import layerlines

    spectra = prepare(
        images, apix, cutoff_res, radial=radial_profile, pd_weight=pd_weight
    )
    radii = (
        [radius * f for f in RADIUS_FACTORS]
        if model in ("bessel", "lines")
        else [radius]
    )

    # the first pass: where are the lines
    found = _run_first_pass(
        spectra,
        list(csyms),
        np.arange(twists[0], twists[1] + 1e-9, step_twist),
        np.arange(rises[0], rises[1] + 1e-9, step_rise),
        radius,
        workers,
        progress,
        step_twist,
    )
    found.sort(reverse=True)
    places = []
    for place in found:
        score, contrast, csym, twist, rise = place
        # lines at the same heights: the same rise and csym * twist
        if any(
            abs(rise - p[4]) <= 2.0 * step_rise
            and abs(csym * twist - p[2] * p[3]) <= 2.0 * max(csym, p[2]) * step_twist
            for p in places
        ):
            continue
        places.append(place)
        if len(places) == lattices:
            break

    # where the lines are, to a finer grid, by the contrast of the mask
    if refine:
        for k, (score, contrast, csym, twist, rise) in enumerate(places):
            best_place = places[k]
            span = step_twist * 6.0 / csym
            for t in twist + np.linspace(-span, span, 5):
                for r in rise + np.linspace(-step_rise, step_rise, 5):
                    if t <= 0 or r <= 0:
                        continue
                    sc, ct = _score(
                        spectra, csym, float(t), float(r), [radius], model="mask"
                    )[:2]
                    if sc > best_place[0]:
                        best_place = (sc, ct, csym, float(t), float(r))
            places[k] = best_place

    # the second: which symmetry, among those that put the lines there. The fits of
    # different lattices cannot be compared (a finer lattice has more lines), those
    # of the members of one can, so a lattice keeps its first-pass score and its
    # members share it out by their fits
    best = []
    for place_score, place_contrast, csym, twist, rise in places:

        def members_by(how, radii_used):
            found = []
            for other in csyms:
                t = twist * csym / other
                if not 0 < t <= 90.0:
                    continue
                score, fit, parity, rad = _score(
                    spectra, other, t, rise, radii_used, model=how
                )
                if parity >= MIN_PARITY and (score > 0 or how == "orders"):
                    found.append((fit, other, t, parity, rad))
            return found

        members = members_by(model, radii)
        if model == "orders" and members and max(m[0] for m in members) <= 0:
            # no strong line to read an order from: rank the members by their masks
            members = members_by("mask", [radius])
        if not members:
            continue
        # the best member takes the lattice's score, the others less. Members of a
        # lattice have the same line heights and differ in the order of each line,
        # which is read from where it starts along X: what a wrong member gets
        # wrong is the strong lines it gives another order, so it loses with the
        # share of the best member's agreement it reaches, squared (none at all
        # when it disagrees with the transform on net)
        fits = np.array([m[0] for m in members])
        top_fit, spread = fits.max(), max(fits.max() - fits.min(), 1e-12)
        for fit, other, t, parity, rad in members:
            if model == "orders" and top_fit > 0:
                share = min(max(fit, 0.0) / top_fit, 1.0) ** MEMBER_POWER
            else:
                share = 1.0 - 0.5 * (top_fit - fit) / spread
            # lines it predicts where the transform has none
            lines = layerlines.layer_line_heights(
                t,
                rise,
                other,
                y_max=1.0 / cutoff_res,
                radius=rad,
                x_max=1.0 / cutoff_res,
            )
            precision = spectra.line_precision(*lines, rad) ** PRECISION_POWER
            best.append(
                Candidate(
                    other,
                    t,
                    rise,
                    place_contrast * parity * share * precision,
                    fit,
                    parity,
                    rad,
                )
            )
    best.sort(key=lambda c: -c.score)
    return best[:top]
