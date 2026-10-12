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
        ``profile * parity``; compare candidates of one search only.
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

    def __init__(self, shape, apix, cutoff_res, excess, rows, signed, n_images):
        self.shape = shape
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

        key = round(float(radius), 3)
        cache = self.__dict__.setdefault("_orders", {})
        if key in cache:
            return cache[key]
        ny = self.shape[0]
        n_columns = int(np.searchsorted(self._fx, 1.0 / self.cutoff_res, side="right"))
        fx = self._fx[:n_columns]
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
    """
    from . import helix_transform

    ny, nx = image.shape
    img = np.roll(np.asarray(image, dtype=float) - np.mean(image), -(nx // 2), axis=1)
    # _meridian_offset takes the filament horizontal and gives the offset across it
    offset = helix_transform._meridian_offset(img.T)
    F = np.fft.fft2(img)
    power = np.abs(np.fft.rfft2(img)) ** 2
    excess = power / helix_transform._ring_noise(power) - 1.0
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
    return excess, rows, phasor.sum(axis=1)


def prepare(images, apix, cutoff_res):
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
    for image in images:
        e, rows, s = _centred_cross_meridian(image, cutoff_res, apix)
        excess, signed = excess + e, signed + s
    return _Spectra(
        images[0].shape, apix, cutoff_res, excess, rows, signed, len(images)
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
    ``"lines"`` every
    line of the lattice with its own amplitude and the ``J_n**2`` profile of its
    order along X, the fit being the sum of the squares of their significances;
    ``"mask"`` the
    lines kept from where each order starts (``layerlines.layer_line_mask``) and
    the contrast of the transform on them; ``"onset"`` the same lines, and the F1
    of where they are against where the transform is (missing and extra lines
    both count against); ``"bessel"`` the whole ``J_n**2`` profile of each line and
    the correlation with the transform. The first is quick, for the first pass over
    many helices.
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
    """
    turn = (csym * twist) % 360.0
    return min(turn, 360.0 - turn) <= tolerance


def _first_pass(spectra, csym, twists, rises, radius, tolerance=0.0):
    """The first-pass scores of one symmetry over some twists and all rises (a worker's share)."""
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
    """
    import multiprocessing as mp

    try:
        context = mp.get_context("forkserver")
    except ValueError:
        return mp.get_context("spawn")
    context.set_forkserver_preload(
        [__name__, "helicon.lib.layerlines", "scipy.special"]
    )
    return context


def _init_worker(spectra):
    """Give a worker process the spectra once, not with every share of the work."""
    _WORKER_SPECTRA[:] = [spectra]


def _first_pass_in_worker(csym, twists, rises, radius, tolerance):
    return _first_pass(_WORKER_SPECTRA[0], csym, twists, rises, radius, tolerance)


def _run_first_pass(
    spectra, csyms, twists, rises, radius, workers, progress, step_twist=1.0
):
    """All of the first pass, in ``workers`` processes when it is worth it."""
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
    spectra = prepare(images, apix, cutoff_res)
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
        # the best member takes the lattice's score, the others less, by how far
        # their fits fall short of it in units of the spread of the fits
        fits = np.array([m[0] for m in members])
        top_fit, spread = fits.max(), max(fits.max() - fits.min(), 1e-12)
        for fit, other, t, parity, rad in members:
            share = 1.0 - 0.5 * (top_fit - fit) / spread
            best.append(
                Candidate(
                    other, t, rise, place_contrast * parity * share, fit, parity, rad
                )
            )
    best.sort(key=lambda c: -c.score)
    return best[:top]
