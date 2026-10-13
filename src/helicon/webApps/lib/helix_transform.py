"""Estimating and applying the transform that puts a filament horizontal.

Both helical tabs need the same thing from a 2D class average: how far it is
from horizontal, how far its axis sits from the middle, and how wide it is.
denovo3D worked this out first and helicalProjection needs exactly the same
answer, so the estimators, the auto-transform decision and the per-image card
machinery live here rather than in either tab.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

import helicon

# The ways of finding a filament's orientation and centre; the first is the default.
METHODS = {
    "threshold": "Threshold and moments",
    "spectral": "Projection sharpness and phases across the meridian",
    "pd": "Phases across the meridian: rotation and centre together",
}
DEFAULT_METHOD = "threshold"


def ring_noise(power):
    """Mean noise power of every pixel of an ``rfft2``: the median of its ring over ``ln 2``.

    Signal fills few pixels of a ring, so the median is the noise; for complex
    Gaussian noise the power is exponential and its median is ``ln 2`` of its mean.

    Parameters
    ----------
    power : numpy.ndarray
        ``|rfft2(image)|**2``.

    Returns
    -------
    numpy.ndarray
        The noise power at every pixel, the shape of ``power``; rings with too few
        pixels for a median take their neighbours' by interpolation.
    """
    from helicon.lib import rings

    ring = rings.rfft2_rings(power.shape)
    medians = rings.grouped_median(power, rings.group_index(ring), min_count=4)
    noise = medians / np.log(2.0)
    ok = np.flatnonzero(np.isfinite(noise))
    noise = np.interp(np.arange(len(noise)), ok, noise[ok])
    return np.maximum(noise[ring], 1e-30)


def _sharpness_rotation(data, max_size=128):
    """The rotation (degrees, as ``helicon.transform_image`` takes it) that makes the filament horizontal.

    A horizontal filament, summed along its length, gives the narrowest and tallest
    profile across it, so the rotation that maximises the sum of squares of that
    profile is the one. It looks at the image as a whole, with no threshold, and
    does not care whether the filament is bright or dark. A coarse scan on a
    reduced image is followed by two finer ones on the full image.
    """
    import helicon

    img = np.asarray(data, dtype=float)
    img = img - img.mean()
    ny, nx = img.shape
    y, x = np.mgrid[:ny, :nx]
    # a disc, so that the corners that rotation brings in or out do not count
    img = img * (np.hypot(y - ny / 2, x - nx / 2) < min(ny, nx) / 2 - 1)
    step = max(1, int(np.ceil(max(ny, nx) / max_size)))
    small = img[: ny // step * step, : nx // step * step]
    small = small.reshape(ny // step, step, nx // step, step).mean(axis=(1, 3))

    def sharpness(image, angle):
        rotated = helicon.transform_image(image=image, rotation=angle)
        return float((rotated.sum(axis=1) ** 2).sum())

    best = float(max(np.arange(-90.0, 90.0, 2.0), key=lambda a: sharpness(small, a)))
    for half, resolution in ((2.0, 0.25), (0.25, 0.03)):
        angles = np.arange(best - half, best + half + 1e-9, resolution)
        best = float(max(angles, key=lambda a: sharpness(img, a)))
    return helicon.set_to_periodic_range(best, min=-90, max=90)


def meridian_offset(image, max_shift=None, resolution=0.01, band=0.25):
    """How far the helical axis is from the middle of the box, in pixels, across the rows.

    Along the axis (the columns of ``image``), the transform of a helix has layer
    lines whose two halves, on either side of the meridian, are mirror images of
    each other up to a sign that is the same along the whole line, when the axis
    is at the origin. Moving the axis by ``dx`` multiplies the product of the two
    halves, ``F(X, Y) F(X, -Y)``, by ``exp(-4 pi i X dx)``. The phase of that
    product is therefore a straight line in ``X`` through the same constant on
    each line: the offset is the ``dx`` that makes the phases of every line agree.
    Each pixel counts for as much as its excess over the noise, so the noise,
    which has random phases, adds little, and the equator and the layer lines
    that stand out most decide.

    Parameters
    ----------
    image : numpy.ndarray
        The filament along the rows... that is, horizontal: axis 1 along the filament.
    max_shift : float, optional
        Largest offset looked for, pixels. Defaults to a quarter of the width,
        beyond which the product cannot tell an offset from another (it repeats
        every half width).
    resolution : float, optional
        Pixels. Defaults to 0.01.
    band : float, optional
        Highest frequency used, cycles per pixel. The finest lines are the
        most easily upset by the interpolation of the rotation that came before,
        and past 0.25 they added error to the offset and no precision.

    Returns
    -------
    float
        The offset along axis 0, pixels from ``image.shape[0] // 2``.
    """
    img = np.asarray(
        image, dtype=float
    ).T  # axis along the rows, as the transform is laid out
    n = img.shape[1]
    img = np.roll(img - img.mean(), -(n // 2), axis=1)
    F = np.fft.rfft2(img)
    excess = np.maximum(np.abs(F) ** 2 / ring_noise(np.abs(F) ** 2) - 1.0, 0.0)
    ny = img.shape[0]
    rows = np.arange(0, ny // 2 + 1)
    mirror = (-rows) % ny
    product = F[rows, :] * F[mirror, :]
    weight = np.sqrt(excess[rows, :] * excess[mirror, :])
    keep = np.arange(F.shape[1]) / n < 0.5
    phasor = (weight * product / np.maximum(np.abs(product), 1e-30))[:, keep]
    freq = (np.arange(F.shape[1]) / n)[keep]
    max_shift = n / 4.0 if max_shift is None else max_shift

    height = rows / ny  # of each layer line, cycles per pixel

    def score(shifts, f_max):
        # each line's weights summed coherently along X, then over the lines,
        # within the disc of radius f_max: a small error of the rotation shears
        # the transform in proportion to the height of the line, so the lines
        # near the equator are the most reliable
        inside = (freq[None, :] ** 2 + height[:, None] ** 2) <= f_max**2
        phase = np.exp(4j * np.pi * freq[:, None] * np.asarray(shifts)[None, :])
        return np.abs((phasor * inside) @ phase).sum(axis=0)

    # Coarse to fine. Over a band 0..f the score is one lobe about 1/(4 f) wide,
    # where over the whole band it also has ghosts a few pixels away that a
    # small error of the rotation can make the highest: so the lowest
    # frequencies place the axis and each wider band refines it.
    f_max, centre, half = 0.03, 0.0, max_shift
    while True:
        f_max = min(f_max, band)
        step = max(1.0 / (16.0 * f_max), resolution)
        grid = np.arange(centre - half, centre + half + step / 2, step)
        centre = float(grid[int(np.argmax(score(grid, f_max)))])
        if f_max >= band:
            break
        half = 1.0 / (2.0 * f_max)
        f_max *= 2.0
    half = step
    while half > resolution:
        fine = np.linspace(centre - half, centre + half, 21)
        centre = float(fine[int(np.argmax(score(fine, band)))])
        half /= 5.0
    return centre


def _profile_width(image, fraction=0.2, smooth=2.0):
    """The width of a filament across a horizontal axis, in pixels.

    The profile across is the image summed along the axis. The filament is where it
    departs from the baseline (the mean of its two edges) in the same direction as
    its strongest part in the middle of the box, by more than ``fraction`` of that
    departure. Only that direction counts: class averages have a halo of the
    opposite sign around the filament, which is not part of it.
    """
    from scipy import ndimage as ndi

    profile = ndi.gaussian_filter1d(np.asarray(image, dtype=float).mean(axis=1), smooth)
    n = len(profile)
    edge = max(2, n // 10)
    departure = profile - 0.5 * (profile[:edge].mean() + profile[-edge:].mean())
    middle = slice(n // 4, n - n // 4)
    sign = 1.0 if departure[middle][np.argmax(np.abs(departure[middle]))] > 0 else -1.0
    departure = sign * departure
    above = np.flatnonzero(departure > fraction * departure.max())
    return int(above[-1] - above[0] + 1) if len(above) else n


def spectral_rotation_center_diameter(
    data, estimate_rotation=True, estimate_center=True
):
    """Rotation, vertical shift and diameter of a horizontal-ish filament, from its transform.

    The orientation is the one that makes the projection along the filament
    sharpest (:func:`_sharpness_rotation`) and the centre is where the phases
    across the meridian agree (:func:`meridian_offset`). Neither takes a
    threshold, neither assumes a bright filament on a dark background, and
    both hold up in noise where a threshold fails.

    Parameters
    ----------
    data : numpy.ndarray
        The 2D image.
    estimate_rotation, estimate_center : bool, optional
        As for :func:`refine_helix_rotation_center`; what is not estimated is 0.

    Returns
    -------
    rotation : float
        Degrees to rotate with ``helicon.transform_image`` to make it horizontal.
    shift_y : float
        Pixels to move it down (``post_translation=(shift_y, 0)``) to centre it.
    diameter : int
        Width of the filament across its axis, pixels.
    """
    import helicon

    data = np.asarray(data, dtype=float)

    def turned(image, angle):
        return helicon.transform_image(image=image, rotation=angle) if angle else image

    def moved(image, down):
        if not down:
            return image
        return helicon.transform_image(image=image, post_translation=(down, 0))

    rotation = _sharpness_rotation(data) if estimate_rotation else 0.0
    straight = turned(data, rotation)
    shift = -meridian_offset(straight) if estimate_center else 0.0
    centred = moved(straight, shift)
    if estimate_rotation:
        # the first rotation was measured with an off-centre filament, which the
        # disc it is measured in cuts unevenly; measure what is left on the
        # centred one (a turn this small moves the shift by a hundredth of a pixel)
        rotation += _sharpness_rotation(centred)
        centred = moved(turned(data, rotation), shift)
    return float(rotation), float(shift), _profile_width(centred)


def _pd_terms(image, f_max=0.3):
    """What the phase difference across the meridian (PD) of an image is made of.

    The image is laid out as for :func:`meridian_offset` (the filament horizontal).
    Returns the phase of the product of every pixel of the transform with its
    mirror across the meridian, its weight, and its frequency along the filament.

    The weight of a pixel is the geometric mean of the power of the pixel and of
    its mirror, each divided by the noise power of its ring (the median of the ring
    over ``ln 2``, :func:`ring_noise`) and with the noise taken out: the shell-median
    normalised intensity above the noise. Pixels the lines of the helix do not
    reach weigh nothing, and random phases add nothing.

    Parameters
    ----------
    image : numpy.ndarray
        The filament horizontal.
    f_max : float, optional
        Largest spatial frequency used, cycles per pixel. Defaults to 0.3.

    Returns
    -------
    phase : numpy.ndarray
        The phase of the product, radians; ``(rows, columns)`` of the pixels on one
        side of the meridian, the equator and the meridian left out.
    weight : numpy.ndarray
        The same shape.
    freq : numpy.ndarray
        The frequency along the filament of each column, cycles per pixel.
    """
    img = np.asarray(image, dtype=float).T
    ny, n = img.shape
    img = np.roll(img - img.mean(), -(n // 2), axis=1)
    F = np.fft.rfft2(img)
    power = np.abs(F) ** 2
    excess = np.maximum(power / ring_noise(power) - 1.0, 0.0)
    rows = np.arange(4, ny // 2)
    mirror = (-rows) % ny
    cols = np.arange(1, F.shape[1])
    cols = cols[cols / n <= f_max]
    product = F[np.ix_(rows, cols)] * F[np.ix_(mirror, cols)]
    weight = np.sqrt(excess[np.ix_(rows, cols)] * excess[np.ix_(mirror, cols)])
    height = rows / ny
    inside = (cols[None, :] / n) ** 2 + height[:, None] ** 2 <= f_max**2
    return np.angle(product), weight * inside, cols / n


def _pd_best_offset(phase, weight, freq, centre, half=0.6, step=0.02):
    """The offset of the axis that makes the phase difference cleanest, and its score.

    A shift ``o`` of the axis along the filament adds ``4 pi f o`` to the phase of
    the product, so ``cos(2 PD)`` of a pixel becomes ``cos(2 phase + 8 pi f o)``,
    which is 1 at 0 and at 180 degrees alike, as the phase of a line of even and
    of odd order is. The score is the weighted sum of it.

    Returns
    -------
    offset : float
        Pixels, as :func:`meridian_offset` gives it.
    score : float
        ``sum(weight * cos(2 PD))`` over the weight, from -1 to 1.
    """
    grid = np.arange(centre - half, centre + half + step / 2, step)
    two = 2.0 * phase[..., None] + 8.0 * np.pi * freq[None, :, None] * grid
    total = (weight[..., None] * np.cos(two)).sum(axis=(0, 1))
    k = int(np.argmax(total))
    offset = float(grid[k])
    if 0 < k < len(grid) - 1:  # to a fraction of a step
        a, b, c = total[k - 1], total[k], total[k + 1]
        if a - 2.0 * b + c < 0:
            offset += 0.5 * (a - c) / (a - 2.0 * b + c) * step
    return offset, float(total[k] / max(weight.sum(), 1e-30))


def pd_rotation_center_diameter(
    data, estimate_rotation=True, estimate_center=True, search=1.5, step=0.1
):
    """Rotation, vertical shift and diameter of a filament from the phases across its meridian.

    The rotation and the centre are found together by maximising the weighted sum
    of ``cos(2 PD)``, the phase difference across the meridian: for a centred helix
    whose axis is along the rows, the phase of every layer line is 0 (even Bessel
    order) or 180 degrees (odd), and ``cos(2 PD)`` is 1 for both. A tilt of the axis
    shears the lines and a shift of the axis tilts the phases along them, so the
    two are told apart by what they do (the shift is solved for exactly for every
    tilt, :func:`_pd_best_offset`). The weight of each pixel is its shell-median
    normalised intensity above the noise (:func:`_pd_terms`).

    The search starts from the estimate of ``"threshold"``, over ``search`` degrees
    either way, then narrows around the best tilt.

    Parameters
    ----------
    data : numpy.ndarray
        The 2D image.
    estimate_rotation, estimate_center : bool, optional
        Whether to estimate them; what is not estimated is 0 (the rotation of the
        threshold estimate, when only the centre is wanted).
    search : float, optional
        Degrees either way of the starting rotation. Defaults to 1.5.
    step : float, optional
        First step of the rotation, degrees. Defaults to 0.1.

    Returns
    -------
    rotation : float
        Degrees to rotate with ``helicon.transform_image`` to make it horizontal.
    shift_y : float
        Pixels to move it down (``post_translation=(shift_y, 0)``) to centre it.
    diameter : int
        Width of the filament across its axis, pixels.
    """
    import helicon

    data = np.asarray(data, dtype=float)
    start, _, _ = estimate_rotation_center_diameter(
        data, method="threshold", estimate_center=False
    )

    def turned(angle):
        return helicon.transform_image(image=data, rotation=angle) if angle else data

    def evaluate(angle):
        straight = turned(angle)
        phase, weight, freq = _pd_terms(straight)
        coarse = meridian_offset(straight)
        offset, score = _pd_best_offset(phase, weight, freq, coarse)
        return score, offset

    rotation = start
    if estimate_rotation:
        half = search
        while half >= 0.01:
            grid = np.arange(rotation - half, rotation + half + step / 2, step)
            scores = np.array([evaluate(a)[0] for a in grid])
            k = int(np.argmax(scores))
            best = float(grid[k])
            if 0 < k < len(grid) - 1:  # to a fraction of a step
                a, b, c = scores[k - 1], scores[k], scores[k + 1]
                if a - 2.0 * b + c < 0:
                    best += 0.5 * (a - c) / (a - 2.0 * b + c) * step
            rotation = best
            half = 2.0 * step
            step /= 5.0
    straight = turned(rotation)
    shift = -evaluate(rotation)[1] if estimate_center else 0.0
    centred = (
        helicon.transform_image(image=straight, post_translation=(shift, 0))
        if shift
        else straight
    )
    return float(rotation), float(shift), _profile_width(centred)


def estimate_rotation_center_diameter(
    data,
    method=DEFAULT_METHOD,
    threshold=0.0,
    estimate_rotation=True,
    estimate_center=True,
    refine=True,
):
    """``(rotation, shift_y, diameter)`` of a filament by the chosen ``method``.

    Parameters
    ----------
    data : numpy.ndarray
        The 2D image.
    method : str, optional
        One of the keys of :data:`METHODS`. Defaults to :data:`DEFAULT_METHOD`.
    threshold : float, optional
        The density floor of the ``"threshold"`` method; the ``"spectral"`` one
        takes none.
    estimate_rotation, estimate_center : bool, optional
        Whether to estimate them; what is not estimated is 0.
    refine : bool, optional
        For ``"threshold"``: whether to check and improve the estimate
        (:func:`refine_helix_rotation_center`) or take the single estimate of
        ``helicon.estimate_helix_rotation_center_diameter``. Defaults to True.

    Returns
    -------
    rotation : float
        Degrees, for ``helicon.transform_image``, to make the filament horizontal.
    shift_y : float
        Pixels, down, to centre it.
    diameter : float
        Pixels.

    Raises
    ------
    ValueError
        For a ``method`` that is not in :data:`METHODS`.
    """
    if method == "spectral":
        return spectral_rotation_center_diameter(
            data, estimate_rotation=estimate_rotation, estimate_center=estimate_center
        )
    if method == "pd":
        return pd_rotation_center_diameter(
            data, estimate_rotation=estimate_rotation, estimate_center=estimate_center
        )
    if method != "threshold":
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    if not refine:
        return helicon.estimate_helix_rotation_center_diameter(
            data,
            estimate_rotation=estimate_rotation,
            estimate_center=estimate_center,
            threshold=threshold,
        )
    return refine_helix_rotation_center(
        data,
        threshold=threshold,
        estimate_rotation=estimate_rotation,
        estimate_center=estimate_center,
    )


def refine_helix_rotation_center(
    data,
    threshold=0.0,
    max_iter=6,
    tol=0.02,
    estimate_rotation=True,
    estimate_center=True,
):
    """Refine the rotation/centre estimate by guarded iteration.

    ``_estimate_helix_rotation_center_diameter`` estimates the rotation once and
    never re-checks it after rotating, so an image that is still off-horizontal
    afterwards stays that way (measured residuals up to 2.4 deg on EMPIAR-10940
    class averages, which is enough to push the twist search to the wrong
    answer).

    Iterating it naively is not safe -- the estimator is noisy and each
    resampling degrades the next estimate, so plain iteration improved 24/42
    images but made 16/42 worse.  This version therefore *measures* the residual
    of each proposed correction and keeps it only if strictly better, which is
    monotone by construction: measured over 42 class averages it improved 25 and
    degraded none, halving the mean residual rotation (0.272 -> 0.137 deg) and
    cutting the worst case from 2.39 to 0.52 deg.

    Returns the same ``(rotation_deg, shift_y_px, diameter_px)`` triple.
    """

    def _apply(d, r, sh):
        t = d
        if r:
            t = helicon.transform_image(image=t, rotation=r)
        if sh:
            t = helicon.transform_image(image=t, rotation=0, post_translation=(sh, 0))
        return t

    def _resid(r, sh):
        t = _apply(data, r, sh)
        r2, s2, _ = estimate_helix_rotation_center_diameter(
            t,
            threshold=np.max(t) * 0.2,
            estimate_rotation=estimate_rotation,
            estimate_center=estimate_center,
        )
        # rotation dominates: a degree of tilt hurts far more than a pixel of shift
        return abs(r2) + 0.1 * abs(s2), r2, s2

    rot, shift, diameter = estimate_helix_rotation_center_diameter(
        data,
        threshold=threshold,
        estimate_rotation=estimate_rotation,
        estimate_center=estimate_center,
    )
    best_cost, r2, s2 = _resid(rot, shift)
    for _ in range(max_iter):
        if abs(r2) < tol and abs(s2) < tol:
            break
        cand_r, cand_s = rot + r2, shift + s2
        cost, nr2, ns2 = _resid(cand_r, cand_s)
        if cost >= best_cost - 1e-9:
            break  # no improvement -- keep what we have rather than risk drifting
        t = _apply(data, cand_r, cand_s)
        _, _, diameter = estimate_helix_rotation_center_diameter(
            t,
            threshold=np.max(t) * 0.2,
            estimate_rotation=estimate_rotation,
            estimate_center=estimate_center,
        )
        rot, shift, best_cost, r2, s2 = cand_r, cand_s, cost, nr2, ns2
    return rot, shift, diameter


def estimate_helix_rotation_center_diameter(
    data, estimate_rotation=True, estimate_center=True, threshold=0
):
    """Estimate the rotation, vertical center shift, and diameter of a helix.

    Returns
    -------
    tuple
        (rotation_deg, shift_y_px, diameter_px)
    """
    from skimage.morphology import closing

    ny, nx = data.shape

    def _weighted_params(mask, intensity):
        ys, xs = np.where(mask)
        if len(ys) < 2:
            return 0.0, 0.0, ny
        w = intensity[ys, xs].astype(np.float64)
        w = w - w.min() + 1e-8
        cw = w.sum()
        cy = (ys * w).sum() / cw
        cx = (xs * w).sum() / cw
        uy = ys - cy
        ux = xs - cx
        i_yy = (uy * uy * w).sum() / cw
        i_xx = (ux * ux * w).sum() / cw
        i_xy = (uy * ux * w).sum() / cw
        theta = 0.5 * np.arctan2(2.0 * i_xy, i_yy - i_xx)
        angle = np.rad2deg(theta) + 90.0
        if abs(angle) > 90.0:
            angle -= 180.0
        diameter = int(ys.max() - ys.min() + 1)
        if estimate_center:
            shift = ny // 2 - cy
        else:
            shift = 0.0
        return angle, shift, diameter

    bw = closing(data > threshold, mode="ignore")
    mask = bw > 0
    if not mask.any():
        return 0.0, 0.0, ny

    if estimate_rotation:
        rotation, _, _ = _weighted_params(mask, data)
        rotation = helicon.set_to_periodic_range(rotation, min=-180, max=180)
        data_rotated = helicon.transform_image(image=data, rotation=rotation)
    else:
        rotation = 0.0
        data_rotated = data

    bw_rot = closing(data_rotated > threshold, mode="ignore")
    mask_rot = bw_rot > 0
    if not mask_rot.any():
        return rotation, 0.0, ny

    _, shift_y, diameter = _weighted_params(mask_rot, data_rotated)
    return rotation, shift_y, diameter


@dataclass
class AutoTransform:
    """What an automatic transform works out for a set of images.

    Attributes
    ----------
    per_image : list of (float, float)
        Rotation in degrees and vertical shift in pixels, one pair per image.
    diameter : float
        The widest filament in the set, in pixels.
    crop_size : int
        A vertical crop that fits every image in the set.
    ny, nx : int
        The largest image dimensions in the set.
    """

    per_image: list
    diameter: float
    crop_size: int
    ny: int
    nx: int


def image_key(image):
    """A content-addressed key for caching one image's measurement.

    Keyed on the pixels rather than on a label or a position, so the same image
    is recognised however the selection that contains it changes, and two
    images are never confused for sharing a name.
    """
    array = np.ascontiguousarray(image)
    digest = hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest()
    return (array.shape, str(array.dtype), digest)


def auto_transform(
    images,
    estimate_rotation=True,
    estimate_center=True,
    crop_factor=2.0,
    crop_multiple=4,
    cache=None,
    method=DEFAULT_METHOD,
):
    """Work out the transform that puts each image's filament horizontal.

    The values are per image and deliberately not averaged. On ten good
    EMPIAR-10940 class averages the rotations span -20.1 to +10.2 degrees with
    a mean of -0.47, so applying that mean leaves them 5.73 degrees off
    horizontal on average -- worst case 19.6 -- where the individual values
    leave 0.06. Any caller offering a single shared rotation box should leave
    it at zero and treat it as a nudge on top of these.

    Parameters
    ----------
    images : sequence of np.ndarray
        The 2D images to measure.
    estimate_rotation, estimate_center : bool, optional
        Passed through to :func:`refine_helix_rotation_center`. Both are false
        for input that is already a projection of a 3D map.
    crop_factor : float, optional
        Vertical crop as a multiple of the measured diameter. Defaults to 2.
    crop_multiple : int, optional
        The crop is rounded down to a multiple of this. Defaults to 4.
    cache : dict, optional
        Per-image measurements, keyed by :func:`image_key`, read and written
        in place. Measuring an image is the whole cost here, and a caller that
        re-runs on every change to a selection re-measures every image that was
        already in it; with a cache only what is new is measured.
    method : str, optional
        How each image is measured: one of the keys of :data:`METHODS`. Defaults to
        :data:`DEFAULT_METHOD`.

    Returns
    -------
    AutoTransform
    """
    if not len(images):
        raise ValueError("auto_transform(): no images")
    per_image = []
    diameters = []
    for image in images:
        key = None
        if cache is not None:
            key = (image_key(image), estimate_rotation, estimate_center, method)
            measured = cache.get(key)
        else:
            measured = None
        if measured is None:
            rotation, shift, diameter = estimate_rotation_center_diameter(
                image,
                method=method,
                threshold=np.max(image) * 0.2,
                estimate_rotation=estimate_rotation,
                estimate_center=estimate_center,
            )
            measured = (float(rotation), float(shift), float(diameter))
            if key is not None:
                cache[key] = measured
        rotation, shift, diameter = measured
        per_image.append((float(rotation), float(shift)))
        diameters.append(float(diameter))
    diameter = max(diameters)
    return AutoTransform(
        per_image=per_image,
        diameter=diameter,
        crop_size=int(diameter * crop_factor) // crop_multiple * crop_multiple,
        ny=int(max(image.shape[0] for image in images)),
        nx=int(max(image.shape[1] for image in images)),
    )


@dataclass
class ImageTransform:
    """One selected image's transform, held by the server rather than a control.

    Attributes
    ----------
    rotation : float
        In-plane rotation, degrees.
    shift_y : float
        Vertical shift, pixels.
    crop_size : int
        Vertical crop, pixels.
    threshold : float
        Density floor; the image minimum means no thresholding.
    generation : int
        Part of the ids of the controls that edit this image. It changes
        whenever the values are replaced from outside those controls -- the
        image newly added, re-added, auto-transformed or reset -- because a
        control that is removed keeps its last value on the server, and a new
        control reusing its id would read that stale value until the browser
        caught up.
    """

    rotation: float
    shift_y: float
    crop_size: int
    threshold: float
    generation: int


def reconcile_transforms(previous, keys, current, fresh, previous_active=None):
    """Carry per-image transforms across a change of selection.

    The rule is the one a user expects: images that were already selected keep
    exactly what they had, manual edits included; only images that are new to
    the selection are auto-transformed; and the card shown afterwards is the
    one for the image just added. Re-deriving every image's transform on each
    change is what lost manual edits, and keying transforms by position is
    what let one image's values land on another when the order shifted.

    Parameters
    ----------
    previous : dict
        ``key -> ImageTransform`` before the change, in selection order.
    keys : list
        The new selection's keys, in the order its images are shown. A key is
        whatever identifies an image stably -- its label, not its position.
    current : dict
        ``key -> ImageTransform`` as the controls hold them now, for keys that
        were already selected. Missing keys fall back to ``previous``.
    fresh : callable
        ``fresh(new_keys) -> {key: ImageTransform}``: the auto-transform, called
        once and only with the keys that are new.
    previous_active : hashable, optional
        The key whose card was shown before the change.

    Returns
    -------
    state : dict
        ``key -> ImageTransform`` for the new selection, in its order.
    added : list
        The keys that were new.
    active : int
        Index into ``keys`` of the card to show: the last image added, else the
        image that was shown if it is still selected, else the first.
    """
    keys = list(keys)
    added = [k for k in keys if k not in previous]
    made = fresh(added) if added else {}
    state = {}
    for k in keys:
        state[k] = current.get(k, previous[k]) if k in previous else made[k]
    if added:
        active = keys.index(added[-1])
    elif previous_active in state:
        active = keys.index(previous_active)
    else:
        active = 0
    return state, added, active


def card_switching_script(galleries):
    """Script that shows only the transform card of the clicked image.

    Every card stays in the DOM and is hidden with CSS, so each image's inputs
    keep whatever was typed into them and no effect can read an input that does
    not exist -- reading one that was never created raises ``SilentException``
    and silently kills the effect that did it.

    Parameters
    ----------
    galleries : sequence of dict
        One entry per gallery, each with ``input`` (the gallery id whose
        selection drives the switch), ``cards`` (a CSS selector for the cards)
        and ``key`` (the ``data-`` attribute holding each card's index).

    Returns
    -------
    shiny.ui.Tag
        A ``<script>`` tag to place in the tab's layout.
    """
    from shiny import ui

    # Wrapped in an IIFE so each tab's configuration is its own. As a plain
    # `var` it is global, and a second tab using this helper overwrites the
    # first: with both denovo3D and helicalProjection on the page, whichever
    # script ran last owned the variable and the other tab's cards stopped
    # switching at all.
    return ui.tags.script(
        """
        (function () {
            var galleries = %s;
            $(document).on('shiny:inputchanged', function(e) {
                if (!e.name) return;
                galleries.forEach(function(g) {
                    // The id, namespaced or bare -- NOT any name containing
                    // it. Shiny's own clientdata inputs embed output ids, and
                    // HILL has one called
                    // `.clientdata_output_hill-hill_display_selected_image_bg`
                    // whose value is a CSS colour: matched loosely, that colour
                    // became the card index and every card was hidden.
                    if (e.name !== g.input &&
                        !(e.name.length > g.input.length &&
                          e.name.slice(-(g.input.length + 1)) === '-' + g.input))
                        return;
                    var v = e.value;
                    var idx = Array.isArray(v) ? v[0] : v;
                    if (idx === undefined || idx === null) return;
                    if (String(parseInt(idx, 10)) !== String(idx)) return;
                    document.querySelectorAll(g.cards).forEach(function(el) {
                        el.style.display =
                            (String(el.dataset[g.key]) === String(idx))
                                ? '' : 'none';
                    });
                });
            });
        })();
        """
        % json.dumps(list(galleries))
    )


def apply_transform(image, rotation=0.0, shift_y=0.0, crop_size=None, threshold=None):
    """Threshold, straighten and crop one image.

    Parameters
    ----------
    image : np.ndarray
        The 2D image.
    rotation : float, optional
        Rotation in degrees.
    shift_y : float, optional
        Vertical shift in pixels.
    crop_size : int, optional
        Rows to keep about the centre. ``None`` or a value that is not smaller
        than the image leaves it uncropped.
    threshold : float, optional
        Passed to :func:`helicon.threshold_data`.

    Returns
    -------
    np.ndarray

    Notes
    -----
    The crop applies whenever it is smaller than the image, including at the
    32 pixel minimum the auto transform clamps to. Requiring it to *exceed* 32
    skipped the crop for every thin filament while still rotating it -- 10 of
    42 EMPIAR-10940 class averages -- which reads as the auto transform having
    silently not run.
    """
    result = image
    if threshold is not None:
        result = helicon.threshold_data(result, thresh_value=threshold)
    if rotation or shift_y:
        result = helicon.transform_image(
            result, rotation=rotation, post_translation=(shift_y, 0)
        )
    ny, nx = result.shape
    if crop_size is not None and 0 < int(crop_size) < ny:
        result = helicon.crop_center(result, shape=(int(crop_size), nx))
    return result
