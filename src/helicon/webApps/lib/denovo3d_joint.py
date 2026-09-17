"""Joint multi-image helical parameter search.

A single 2D class average determines the helical twist unreliably: measured on
ten good class averages of EMPIAR-10940 only 5/10 peaked at the correct value,
with score margins of 0.001-0.002, i.e. comparable to noise. Since every class
average of a given filament shares the same twist/rise, combining their score
curves resolves this -- the joint peak margin was ~50x the single-image margin
and correct for every subset size tested.

Two refinements matter and are implemented here:

* the curves are z-scored before averaging, because classes differ in both score
  scale and score spread, so a plain average is dominated by whichever image has
  the largest dynamic range (measured 65x smaller margin);
* the z-score uses a shrinkage floor, so an uninformative (flat) curve is
  downweighted instead of having its noise amplified to unit variance.

The floor is a fraction of the *median* curve spread, and that choice has a
known limitation: it protects against a loud outlier but only while flat curves
are a minority. Measured with three informative curves, flat curves get weight
0.01-0.03 and the answer stays right up to three of them, but at four the median
spread becomes the flat value, the floor collapses, flat curves get weight 0.78,
and the joint peak goes wrong. Scaling the floor to the *largest* spread instead
fixes that case and breaks the more important one -- a single class average with
wild scores then outvotes every good image. Since the user selects which classes
to include, protecting against one bad selection is worth more than tolerating a
majority of them, so this is left as is rather than made adaptive.

Refining each image's psi and dy against a reference built from the others was
tried and deliberately left out. Measured with an estimator independent of the
one that produced the transform (slope of the intensity-weighted row centroid,
so the comparison is not circular), it helps raw class averages -- mean |tilt|
5.48 -> 2.98 deg -- but the tab's own auto-transform already reaches 0.07 deg,
and refining those images pushes them back out to 0.72 deg (worst 0.16 -> 2.17),
replicated at 8 and 16 images. The cause is reference bias: the images end up
mutually consistent (match correlation 0.89) while collectively rotated away
from horizontal. So the match correlation must never be used as a convergence
criterion -- it improves while the answer gets worse.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

SHRINK = 0.25

# A two-fold about the helical axis is detected from the placements themselves,
# not from the images. The obvious test was tried first and abandoned: scoring
# how mirror-symmetric each image is about its own axis, which does not
# separate real class averages at all. Tab-prepared EMPIAR-10940 classes
# averaged 0.266 on that measure and ranged from -0.40 to 0.75, straddling what
# a synthetic control with no two-fold scored, so no threshold divided them.
#
# The symptom is a much better detector than the cause. If the structure has
# the two-fold, azimuth phi and phi + 180 describe the same placement, so a
# search over the full period is free to swap between them -- and what that
# looks like is a BIMODAL distribution of how far placements move between
# rounds. Measured on 33 tab-prepared classes: 29 moved by about 0.2 degrees,
# 4 by about 180, and none by anything in between. It is the empty middle that
# identifies the two-fold; images that merely disagree would fill it.
#
# Not "most images flipped", which was the first rule tried and is wrong: only
# a handful are ambiguous enough to swap, while the rest have a clear
# preference. Nor is the maximum movement a usable convergence criterion here,
# for the same reason -- it reads 179.9 while 31 of 33 images are settled,
# which is why the median is reported alongside it.
TWO_FOLD_FLIP_TOLERANCE = 40.0
TWO_FOLD_MIN_FLIPPED = 2


# ──────────────────────────────────────────────────────────────────────────
# Joint scoring
# ──────────────────────────────────────────────────────────────────────────


def combine_score_curves(curves, shrink=SHRINK):
    """Combine per-image score curves into one joint curve.

    Parameters
    ----------
    curves : sequence of 1-D arrays
        One score curve per image, all sampled on the same parameter grid.
    shrink : float
        Shrinkage floor, as a fraction of the median curve spread. Curves much
        flatter than the median contribute proportionally less.

    Returns
    -------
    (combined, weights)
        ``combined`` is the joint curve; ``weights`` has one value per input
        curve (near 1 for informative, near 0 for flat) and is worth reporting
        so the user can see which selections carried the answer.
    """
    C = np.asarray(curves, dtype=float)
    if C.ndim == 1:
        C = C[None, :]
    sd = np.nanstd(C, axis=1, keepdims=True)
    med = float(np.nanmedian(sd))
    eps = shrink * med if np.isfinite(med) and med > 0 else 1e-12
    Z = (C - np.nanmean(C, axis=1, keepdims=True)) / (sd + eps)
    weights = (sd.ravel() / (sd.ravel() + eps)).astype(float)
    return np.nanmean(Z, axis=0), weights


def joint_best(curves, params, shrink=SHRINK):
    """Pick the parameter combination favoured by the combined curve.

    ``params`` lists the parameter values (or tuples) matching the curve
    samples. Returns (best_param, margin_over_runner_up, combined, weights).
    """
    combined, weights = combine_score_curves(curves, shrink)
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return None, 0.0, combined, weights
    i = int(np.nanargmax(combined))
    ordered = np.sort(finite)
    margin = float(ordered[-1] - ordered[-2]) if ordered.size > 1 else 0.0
    return params[i], margin, combined, weights


def combine_results(results, shrink=SHRINK):
    """Re-rank per-image solver results into one joint ranking.

    ``results`` is the flat list produced by running every (image, twist, rise)
    task, each entry ``(score, return_data, params)`` in the shape
    ``denovo3d_pipeline.process_one_task`` returns, where ``params[2]`` is the
    image label and ``params[5:7]`` are twist and rise.

    Only (twist, rise) pairs scored by *every* image are ranked, so a task that
    was skipped or raised cannot make its pair look artificially good. Returns a
    list in the same ``(score, return_data, params)`` shape -- score replaced by
    the combined z-score, and the payload taken from the image that scored the
    pair highest -- plus the per-image weights keyed by image label.
    """
    by_pair = {}
    for r in results:
        params = r[2]
        key = (round(float(params[5]), 6), round(float(params[6]), 6))
        by_pair.setdefault(key, {})[params[2]] = r

    images = sorted({k for v in by_pair.values() for k in v})
    pairs = sorted(k for k, v in by_pair.items() if len(v) == len(images))
    if not pairs or len(images) < 2:
        return sorted(results, key=lambda x: x[0], reverse=True), {}

    curves = [[by_pair[p][im][0] for p in pairs] for im in images]
    combined, weights = combine_score_curves(curves, shrink)

    joint = []
    for p, c in zip(pairs, combined):
        best = max(by_pair[p].values(), key=lambda r: r[0])
        joint.append((float(c), best[1], best[2]))
    joint.sort(key=lambda x: x[0], reverse=True)
    return joint, dict(zip(images, weights))


# ──────────────────────────────────────────────────────────────────────────
# Joint azimuth refinement
# ──────────────────────────────────────────────────────────────────────────


def _gauge(phis):
    """Fix the arbitrary common rotation by putting the first image at zero.

    The joint solution is only defined up to one overall rotation -- rotate the
    volume and every azimuth together and nothing changes -- so a gauge has to
    be chosen or successive iterations cannot be compared.
    """
    phis = np.asarray(phis, dtype=float)
    return (phis - phis[0]) % 360.0


def _max_move(a, b):
    """Largest azimuth change between two placements, in degrees."""
    d = (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0
    return float(np.max(np.abs(d)))


def joint_refine(
    images,
    seed_volume,
    twist,
    rise,
    apix2d,
    apix3d,
    csym,
    geometry,
    iterations=3,
    tolerance=1.0,
    two_fold="auto",
    algorithm=None,
    score_metric="cosine",
    verbose=0,
    cpu=1,
):
    """Place every image along the model, rebuild from all of them, repeat.

    This is the 2D-classification loop, with a 3D model in place of the class
    average: every image is re-placed against the current model and then *all*
    of them are used to rebuild it. Folding images in one at a time was the
    obvious alternative and is not done, because it makes the answer depend on
    the order and lets an early mistake harden; a seed reconstruction from one
    image is used only to start the first placement.

    ``geometry`` carries the reconstruction dimensions the pipeline has already
    derived for these images -- the keys ``joint_reconstruct`` takes.
    ``seed_volume`` may be ``None``, in which case the first model is built
    from the images themselves at a common azimuth -- useful to a caller that
    has the geometry but no reconstruction to hand, which is the usual state
    after a twist scan, since those run without returning volumes.

    Convergence is on placement stability, never on the match correlation: the
    correlation improves while the answer gets worse when images are aligned to
    a shared reference (measured in this module's header), so it cannot be used
    to decide when to stop.

    The placements are worth as much as the volume. An azimuth is an axial
    position modulo the period, so ``phis`` is a stitching of the images onto
    the filament -- and one that needs no pairwise graph, no overlap between
    any two images, no flip synchronisation and no connectivity check, because
    every image is placed against the model rather than against its neighbours.

    Measured on 33 tab-prepared EMPIAR-10940 classes at twist 1.20, against the
    independent pairwise stitcher in ``denovo3d_register``, which works by
    correlation alone and never uses the twist:

        joint_refine    17 s   33/33 placed   converged to 0.03 deg
        auto_stitch    206 s   33/33 placed   477 pairs, 253 flip conflicts

        median difference between their placements: 0.6 px of a 142 px period,
        32 of 33 within 10 px

    Two methods sharing no machinery agreeing to sub-pixel is good evidence for
    both. Two cautions on using it, though. The placement is modulo the period,
    exactly as the pairwise stitcher's is, so neither can say which turn of the
    helix an image came from. And this one is conditional on the twist it was
    run at, where the pairwise stitcher is not -- so it is sound as an output
    but must never be fed back as independent evidence for the twist, which is
    the circularity ``denovo3d_register`` deliberately avoids.

    Returns a dict with the ``volume``, the per-image ``phis`` and ``flips``,
    the fit ``info``, and a ``history`` of what each round moved -- ``moved``
    is the largest movement and ``moved_median`` the typical one. Judge
    convergence by the median: a couple of genuinely ambiguous images can sit
    at 180 degrees while every other placement is settled.
    """
    from .denovo3d_align import (
        _rotate,
        align_to_model,
        long_side_projection,
        model_length_pixel,
    )
    from .denovo3d_jointsolve import joint_reconstruct

    images = [np.asarray(im, dtype=np.float32) for im in images]
    n = len(images)
    width = images[0].shape[1]

    # A structure with a two-fold about the helical axis -- a 2-1 screw as much
    # as a true C2 -- has a side projection that repeats every half period, so
    # the placement search only needs half the range. The images say whether it
    # does: that same two-fold is what makes a side projection mirror-symmetric
    # about its own axis at all.
    #
    # This is a change to the search range only. The cyclic symmetry stays
    # unimposed on the reconstruction, both because a 2-1 screw does not have
    # it and because leaving it free is what lets its appearance in the map
    # validate the result.
    detect_two_fold = two_fold == "auto"
    two_fold = False if detect_two_fold else bool(two_fold)
    length = model_length_pixel(twist, rise, apix2d, csym, width, two_fold)
    logger.debug("joint_refine: two_fold=%s, model %d px", two_fold, length)

    phis = np.zeros(n)
    volume = seed_volume
    if volume is None:
        # Start from the images themselves, stacked at a common azimuth. That
        # is a blurred model -- it is every view superimposed -- but it is the
        # right kind of blurred: a first pass against it separates the images
        # by azimuth, and the second round has a real model to align to. It is
        # the same opening move as building a class average from its members
        # before any of them are aligned, and it means a caller does not have
        # to supply a seed reconstruction it may not have.
        volume, _ = joint_reconstruct(
            images,
            list(phis),
            twist_degree=twist,
            rise_pixel=rise / apix3d,
            csym=csym,
            algorithm=algorithm,
            target_apix2d=apix2d,
            score_metric=score_metric,
            verbose=verbose,
            cpu=cpu,
            **geometry,
        )
    flips = [False] * n
    info = {}
    history = []

    for it in range(max(1, iterations)):
        model = long_side_projection(
            volume, apix3d, twist, rise, csym, apix2d, images[0].shape[0], length, cpu
        )
        placed = [align_to_model(im, model, twist, rise, apix2d) for im in images]
        new_phis = _gauge([p["phi"] for p in placed])
        # Bring each image into the model's frame with the in-plane rotation
        # the placement found -- a few degrees of tilt, plus 180 if the class
        # was picked from the other end. No mirror is applied, because against
        # a reference spanning a whole period none is needed; see
        # denovo3d_align for why.
        flips = [bool(p["reversed"]) for p in placed]
        oriented = [_rotate(im, p["psi"]) for im, p in zip(images, placed)]

        volume, info = joint_reconstruct(
            oriented,
            list(new_phis),
            twist_degree=twist,
            rise_pixel=rise / apix3d,
            csym=csym,
            algorithm=algorithm,
            target_apix2d=apix2d,
            score_metric=score_metric,
            verbose=verbose,
            cpu=cpu,
            **geometry,
        )
        moved = _max_move(new_phis, phis) if it else float("nan")

        # Half-turn flips across the board mean phi and phi+180 are the same
        # placement, i.e. the structure has a two-fold about the axis. Halve
        # the search range and carry on; the placements settle immediately.
        if detect_two_fold and it:
            swing = np.abs((new_phis - phis + 180.0) % 360.0 - 180.0)
            flipped = np.abs(swing - 180.0) < TWO_FOLD_FLIP_TOLERANCE
            settled = swing < TWO_FOLD_FLIP_TOLERANCE
            middle = ~flipped & ~settled
            if flipped.sum() >= TWO_FOLD_MIN_FLIPPED and not middle.any():
                two_fold = True
                detect_two_fold = False
                length = model_length_pixel(twist, rise, apix2d, csym, width, two_fold)
                logger.debug(
                    "joint_refine: %d/%d placements flipped by half a turn and "
                    "none moved in between -> two-fold about the axis, halving "
                    "the search range",
                    int(flipped.sum()),
                    len(flipped),
                )
        history.append(
            dict(
                iteration=it,
                moved=moved,
                moved_median=(
                    float(np.median(np.abs((new_phis - phis + 180.0) % 360.0 - 180.0)))
                    if it
                    else float("nan")
                ),
                score=info["score"],
                corr=float(np.mean([p["corr"] for p in placed])),
                rival=float(np.mean([p["rival"] for p in placed])),
                width=float(np.mean([p["width"] for p in placed])),
                n_reversed=sum(flips),
                psi=float(np.mean([p["psi"] for p in placed])),
            )
        )
        phis = new_phis
        if it and moved < tolerance:
            break

    # The projection of the volume we finished with, not the one the last
    # round aligned against -- that was made from the previous iteration's
    # volume, so it is one step stale. Recomputing costs a single projection
    # and makes the returned model the model of the returned answer.
    model_projection = long_side_projection(
        volume, apix3d, twist, rise, csym, apix2d, images[0].shape[0], length, cpu
    )
    return dict(
        volume=volume,
        two_fold=two_fold,
        phis=phis,
        flips=flips,
        info=info,
        placed=placed,
        history=history,
        model_projection=model_projection,
        model_length=length,
    )


def placements_as_transforms(
    phis, placed, twist, rise, apix2d, image_width, two_fold=False
):
    """Express joint placements in the manual stitch's own terms.

    The manual route already accepts a starting layout from the automatic
    stitcher, and these placements fit the same slot, so a joint search can hand
    the user a laid-out set to adjust instead of an empty canvas. The shape
    matches what ``denovo3d_tab`` builds from ``auto_stitch``: ``flip_x``,
    ``flip_y``, ``rotation``, ``shift_y``, ``shift_x`` and ``connected``.

    Three conversions are needed, and each is a place the two routes differ.

    * ``shift_x`` is a correction from an end-to-end tiled layout rather than an
      absolute position, so the tiling has to be taken back out -- hence
      ``dx - base - i * image_width``, exactly as the automatic route does it.
    * an azimuth is a position modulo the period, so every image lands inside
      one period. That is the honest layout: the placements genuinely do not say
      which turn of the helix an image came from. It does mean the images
      overlap heavily, which the manual compositor handles -- it places images
      directly rather than through a montage that assumes a grid.
    * polarity is carried as an in-plane 180 degree rotation here, but the
      manual card's rotation is bounded to +-90, so a reversed image is emitted
      as ``flip_x`` and ``flip_y`` together -- which is that same rotation --
      with only the residual few degrees left in ``rotation``.

    ``placed`` is the per-image alignment dicts ``joint_refine`` returns.
    """
    from .denovo3d_align import period_pixel, phi_to_dx

    period = period_pixel(twist, rise, apix2d, 1, two_fold)
    dx = np.array(
        [phi_to_dx(p, twist, rise, apix2d) for p in np.asarray(phis, dtype=float)]
    )
    dx = np.mod(dx, period)
    base = float(dx.min()) if dx.size else 0.0

    out = []
    for i, (d, p) in enumerate(zip(dx, placed)):
        psi = (float(p.get("psi", 0.0)) + 180.0) % 360.0 - 180.0
        reversed_ = abs(psi) > 90.0
        if reversed_:
            psi = psi - 180.0 if psi > 0 else psi + 180.0
        out.append(
            dict(
                flip_x=bool(reversed_),
                flip_y=bool(reversed_),
                rotation=float(psi),
                shift_y=0.0,
                shift_x=float(d - base - i * image_width),
                connected=True,
            )
        )
    return out


def geometry_from_result(
    result,
    sym_oversample=-1,
    interpolation="linear",
    positive_constraint=-1,
    inner_diameter_pixel=0,
):
    """Pull the reconstruction geometry out of one solver result.

    ``result`` is a ``(score, return_data, params)`` triple as
    ``denovo3d_pipeline.process_one_task`` returns. Taking the geometry from
    the result rather than rebuilding it from the UI inputs means a placement
    describes the twist that was actually solved, and cannot drift from it if
    a control is changed afterwards.

    Returns ``(twist, rise, csym, apix2d, apix3d, geometry)``, where
    ``geometry`` is the keyword set ``joint_refine`` and ``joint_reconstruct``
    take.

    The indices are not obvious and are pinned by tests: ``params`` carries
    ``target_apix3d`` and ``target_apix2d`` at 3 and 4, then twist, rise and
    csym at 5, 6 and 7; ``return_data`` carries the four reconstruction
    dimensions at 4 through 7. Note ``return_data[3]``, the volume, is None
    after a parameter scan -- the pipeline only returns volumes when a single
    pair is solved -- which is why a caller has to be able to start without one.
    """
    _score, return_data, params = result
    apix3d, apix2d = float(params[3]), float(params[4])
    geometry = dict(
        scale2d_to_3d=apix2d / apix3d,
        reconstruct_diameter_2d_pixel=return_data[4],
        reconstruct_diameter_3d_pixel=return_data[5],
        reconstruct_length_2d_pixel=return_data[6],
        reconstruct_length_3d_pixel=return_data[7],
        reconstruct_diameter_3d_inner_pixel=inner_diameter_pixel,
        sym_oversample=int(sym_oversample),
        interpolation=interpolation,
        positive_constraint=int(positive_constraint),
    )
    return (
        float(params[5]),
        float(params[6]),
        int(params[7]),
        apix2d,
        apix3d,
        geometry,
    )


def placement_composite(
    images,
    phis,
    placed,
    twist,
    rise,
    apix2d,
    two_fold=False,
    feather=8,
    canvas_length=None,
):
    """Composite the images where the joint search placed them.

    One picture of what a twist implies about the whole set: every image laid
    onto a single canvas at the axial position its azimuth corresponds to, with
    its own in-plane rotation applied.

    It is for looking at, NOT for scoring, and the difference is measured. The
    obvious summary -- how much contrast survives the averaging, on the theory
    that agreeing images reinforce and disagreeing ones blur -- does not track
    the twist. On 16 tab-prepared EMPIAR-10940 classes, against a joint fit
    that peaks correctly at 1.20:

        twist   joint fit   composite contrast
         0.90      0.7027                7.726
         1.05      0.7118                8.283
         1.20      0.7220                8.347
         1.35      0.7125                8.517   <- highest
         1.60      0.6825                7.809

    The contrast peaks a step and a half away from the answer, over a range so
    narrow that the ordering is close to arbitrary. Two plausible reasons, both
    untested: the placements are refined per twist, so a wrong twist gets
    images placed to look as consistent as they can rather than being left
    scattered; and averaging many images over one period leaves most columns
    deeply covered at any twist. Whatever the cause, do not derive a score from
    this image -- rank with the fit, and use the picture to see what the fit
    chose.

    Positions come from the gauged azimuths rather than from each alignment's
    own ``dx``, so this view and the layout handed to the manual stitch by
    :func:`placements_as_transforms` agree with each other.

    ``psi`` is passed to the compositor unchanged, including the half turn a
    reversed image carries: unlike the manual card's bounded control, the
    compositor applies it by interpolation and does not care how large it is.

    ``canvas_length`` lays the result out on the model projection's coordinates
    rather than on a canvas that starts at the leftmost image, so the composite
    and that projection line up column for column and can be shown together.

    Returns ``(image, coverage)`` as ``denovo3d_register.composite`` does, or
    ``(None, None)`` if nothing could be placed.
    """
    from .denovo3d_align import period_pixel, phi_to_dx
    from .denovo3d_register import composite

    period = period_pixel(twist, rise, apix2d, 1, two_fold)
    dx = np.mod(
        [phi_to_dx(float(p), twist, rise, apix2d) for p in np.asarray(phis, float)],
        period,
    )
    transforms = [
        dict(
            psi=float(p.get("psi", 0.0)),
            dy=float(p.get("dy", 0.0)),
            dx=float(d),
            flip_x=False,
            flip_y=False,
            connected=True,
        )
        for d, p in zip(dx, placed)
    ]
    image, coverage = composite(images, transforms, feather=feather)
    if canvas_length is None or image is None:
        return image, coverage

    # Lay it out on the model projection's own canvas instead of one that
    # starts at the leftmost image. The model spans one period plus an image
    # width, and an image placed at dx occupies columns [dx, dx + width) of
    # it, so on a canvas that size column j of this composite is column j of
    # the model and the two can be read against each other directly. The
    # compositor re-zeroes to its leftmost image, so that shift is undone here.
    canvas_length = int(canvas_length)
    x0 = int(np.floor(dx.min())) if dx.size else 0
    out = np.zeros((image.shape[0], canvas_length), dtype=image.dtype)
    cov = np.zeros(canvas_length, dtype=float)
    lo, hi = max(0, x0), min(canvas_length, x0 + image.shape[1])
    if hi > lo:
        out[:, lo:hi] = image[:, lo - x0 : hi - x0]
        cov[lo:hi] = np.asarray(coverage).ravel()[lo - x0 : hi - x0]
    return out, cov


def rank_by_projection_matching(
    results,
    images,
    algorithm=None,
    iterations=3,
    two_fold="auto",
    log=None,
    progress=None,
    display_algorithm=None,
):
    """Re-rank solver results by fitting one volume to all the images at once.

    The alternative to :func:`combine_results`, which scores each image on its
    own and combines the curves afterwards. Here each twist/rise pair is scored
    by how well a single reconstruction, built from every image at its own
    azimuth, explains all of them -- so an image's contribution depends on
    where it sits relative to the others, which a per-image score cannot see.

    Takes the same ``(score, return_data, params)`` results the pipeline
    produces and returns a list in that shape, so a caller can substitute it
    for ``combine_results`` without changing anything downstream. Also returns,
    per pair, the composite of the images at their placements -- the picture of
    what that twist implies -- keyed by ``(twist, rise)`` rounded as
    ``combine_results`` keys them, each a dict of the ``composite`` and the
    ``model`` projection it is aligned to -- the same canvas, so the two can be
    shown one above the other and compared column by column -- the ``zview``,
    and the placements that produced them, so a caller wanting the layout does
    not have to refine again to get it.

    Only pairs solved by *every* image are ranked, for the same reason
    ``combine_results`` does that: a pair scored by a subset would be compared
    against pairs scored by all of them. A pair whose refinement fails is
    dropped with a warning rather than being scored as zero, which would
    silently rank it last.

    Measured against :func:`combine_results` on all 33 good EMPIAR-10940
    classes, prepared as the tab prepares them, scanning 0.80-1.80 in 0.05
    steps with the gauss solver: both methods pick 1.20, and their top five
    agree in full -- 1.20, 1.25, 1.15, 1.30, 1.10 in that order. This route is
    therefore a second opinion rather than a correction, which is why it is
    offered beside the existing search instead of replacing it.

    It is not free. The per-image tasks took 111 s; re-ranking them cost a
    further 311 s, since each twist gets its own refinement. Roughly a
    quadrupling of search time for an answer that, so far, agrees.

    ``progress``, if given, is called as ``progress(index, total, twist)``
    before each pair and may return ``False`` to stop early.

    Returns ``(ranked, composites)``.
    """
    by_pair = {}
    for r in results:
        params = r[2]
        key = (round(float(params[5]), 6), round(float(params[6]), 6))
        by_pair.setdefault(key, {})[params[2]] = r

    labels = sorted({k for v in by_pair.values() for k in v})
    pairs = sorted(k for k, v in by_pair.items() if len(v) == len(labels))
    if not pairs or len(labels) < 2 or len(images) < 2:
        return sorted(results, key=lambda x: x[0], reverse=True), {}

    ranked, composites = [], {}
    for i, pair in enumerate(pairs):
        # Report before the work, so a caller driving a progress bar sees the
        # pair that is about to be refined rather than the one just finished,
        # and can stop us by returning False. Each pair costs a full
        # refinement, so a run with no way out is minutes long.
        if progress is not None and progress(i, len(pairs), pair[0]) is False:
            break
        group = by_pair[pair]
        representative = group[labels[0]]
        twist, rise, csym, apix2d, apix3d, geometry = geometry_from_result(
            representative
        )
        try:
            out = joint_refine(
                images,
                None,
                twist,
                rise,
                apix2d,
                apix3d,
                csym,
                geometry,
                iterations=iterations,
                algorithm=algorithm,
                two_fold=two_fold,
            )
        except Exception:  # pragma: no cover - depends on solver internals
            if log is not None:
                log.warning(
                    "projection matching failed at twist %.3f rise %.4f; "
                    "that pair is left unranked",
                    twist,
                    rise,
                )
            continue

        best = max(group.values(), key=lambda r: r[0])
        ranked.append((float(out["info"]["score"]), best[1], best[2]))
        try:
            composite, coverage = placement_composite(
                images,
                out["phis"],
                out["placed"],
                twist,
                rise,
                apix2d,
                two_fold=out["two_fold"],
                canvas_length=out.get("model_length"),
            )
            # Centre the composite first, then generate the projection over
            # the window that centring implies. The order matters: the shift has
            # to be known before the projection is made, or the only way to
            # apply it afterwards is to translate -- which cuts the projection
            # at one edge.
            composite, shift = center_on_coverage(composite, coverage)
            model, z_view = whole_set_pictures(
                images,
                out,
                twist,
                rise,
                apix2d,
                apix3d,
                csym,
                geometry,
                display_algorithm=display_algorithm,
                shift=shift,
            )
            # The z view is a cross-section, on different axes entirely, so it
            # takes no part in that shift -- it is padded to match instead.
            if z_view is not None and model is not None:
                import helicon

                z_view = helicon.pad_to_size(z_view, shape=np.shape(model))
            composites[pair] = dict(
                composite=composite,
                model=model,
                zview=z_view,
                # The placements themselves, carried out with the pictures.
                # They cost nothing extra here and they are what a manual
                # stitch needs as its starting layout, so a caller never has to
                # re-run a refinement just to find out where the images went.
                phis=out["phis"],
                placed=out["placed"],
                two_fold=out["two_fold"],
                twist=twist,
                rise=rise,
                apix2d=apix2d,
            )
        except Exception:  # pragma: no cover - a failed picture is not fatal
            composites[pair] = None

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked, composites


def center_on_coverage(composite, coverage):
    """Slide the composite so the images sit in the middle of the frame.

    Laid out on the model's canvas the images land wherever their azimuths put
    them, which for a handful of images is a clump off to one side with a wide
    empty margin -- unlike every other picture in the results, which is centred.

    Sliding is right for the composite and only for the composite: outside the
    images it is empty, so nothing is lost off the edge. The projection it is
    shown against must NOT be slid to match, because that cuts it at one end
    and leaves a blank strip at the other. The shift is returned so the
    projection can be *generated* over the displaced window instead; see
    :func:`whole_set_pictures`.

    Returns ``(composite, shift)`` in columns, positive meaning content moved
    right.
    """
    if composite is None:
        return None, 0
    cov = np.asarray(coverage).ravel() if coverage is not None else None
    if cov is None or cov.size != composite.shape[1] or not np.any(cov > 0):
        return composite, 0

    used = np.flatnonzero(cov > 0)
    middle = (int(used[0]) + int(used[-1])) // 2
    shift = composite.shape[1] // 2 - middle
    if shift == 0:
        return composite, 0

    out = np.zeros_like(composite)
    n = composite.shape[1]
    src_lo, src_hi = max(0, -shift), min(n, n - shift)
    dst_lo, dst_hi = max(0, shift), min(n, n + shift)
    if src_hi > src_lo:
        out[:, dst_lo:dst_hi] = composite[:, src_lo:src_hi]
    return out, int(shift)


def _windowed_projection(
    volume, apix3d, twist, rise, csym, apix2d, ny, length, shift=0, cpu=1
):
    """``length`` columns of the side projection, its origin moved by ``shift``.

    Generated wide and cropped, never translated. Translating a fixed-width
    projection blanks one edge and cuts the other, which is wrong twice over --
    it throws away model the user should be able to see, and it invents empty
    space where there is density. Projecting over a longer span and taking the
    window costs at most one more slab and is full everywhere.

    Column ``M // 2`` of a projection is the model's axial origin, so a window
    whose origin should sit ``shift`` columns right of centre is cropped
    ``shift`` columns left of the wide projection's centre.
    """
    from .denovo3d_align import long_side_projection

    shift = int(shift)
    if not shift:
        return long_side_projection(
            volume, apix3d, twist, rise, csym, apix2d, ny, length, cpu
        )
    pad = abs(shift)
    wide = long_side_projection(
        volume, apix3d, twist, rise, csym, apix2d, ny, length + 2 * pad, cpu
    )
    start = wide.shape[1] // 2 - length // 2 - shift
    start = max(0, min(wide.shape[1] - length, start))
    return wide[:, start : start + length]


def whole_set_pictures(
    images,
    placement,
    twist,
    rise,
    apix2d,
    apix3d,
    csym,
    geometry,
    display_algorithm=None,
    shift=0,
    cpu=1,
):
    """The model projection and z view for a finished placement.

    ``display_algorithm``, when it differs from the solver that did the search,
    rebuilds the volume with it before drawing. That follows the rule the rest
    of the results already obey: the ranking stays the search solver's, but
    every picture on screen is drawn by the reconstruction solver, because the
    user compares those against their own images and a map the reconstruction
    selector would never produce is misleading. The placements are reused as
    they are, so this costs one reconstruction rather than another refinement.

    ``shift`` displaces the projection's window so it lines up with a composite
    that has been centred, and the projection is generated over that window
    rather than translated into it, so it is full to both edges.

    The z view is one rise's worth of slices summed through the middle of the
    volume -- the same section ``denovo3d_pipeline`` shows per image -- rescaled
    to the projection's range so the two can sit side by side without one of
    them appearing blank.

    Returns ``(model_projection, z_view)``.
    """
    from .denovo3d_align import _rotate
    from .denovo3d_jointsolve import joint_reconstruct

    volume = placement.get("volume")
    length = placement.get("model_length")

    wanted = (display_algorithm or {}).get("model") if display_algorithm else None
    if wanted:
        oriented = [
            _rotate(im, p.get("psi", 0.0)) for im, p in zip(images, placement["placed"])
        ]
        volume, _info = joint_reconstruct(
            oriented,
            list(placement["phis"]),
            twist_degree=twist,
            rise_pixel=rise / apix3d,
            csym=csym,
            algorithm=display_algorithm,
            target_apix2d=apix2d,
            cpu=cpu,
            **geometry,
        )
    model = None
    if volume is not None and length:
        model = _windowed_projection(
            volume,
            apix3d,
            twist,
            rise,
            csym,
            apix2d,
            images[0].shape[0],
            int(length),
            shift,
            cpu,
        )

    z_view = None
    if volume is not None and np.ndim(volume) == 3:
        nz_per_rise = max(1, int(np.ceil(rise / max(apix3d, 1e-6))))
        z0 = max(0, volume.shape[0] // 2 - nz_per_rise // 2)
        z_view = np.sum(volume[z0 : z0 + nz_per_rise, :, :], axis=0)
        lo, hi = float(z_view.min()), float(z_view.max())
        if hi > lo and model is not None:
            tlo, thi = float(np.min(model)), float(np.max(model))
            z_view = (z_view - lo) * (thi - tlo) / (hi - lo) + tlo
    return model, z_view
