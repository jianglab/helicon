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
            key = (image_key(image), estimate_rotation, estimate_center)
            measured = cache.get(key)
        else:
            measured = None
        if measured is None:
            rotation, shift, diameter = refine_helix_rotation_center(
                image,
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
