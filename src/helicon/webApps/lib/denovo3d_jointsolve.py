"""One 3D reconstruction from several 2D images, each at its own azimuth.

The single-image solver explains one image with one volume. This explains N
images with one volume and N azimuths, which is the arrangement that ought to
make the twist identifiable: a helically symmetric volume has a single
asymmetric unit of freedom, each image contributes a full 2D constraint against
it, and a wrong twist has only one number per image to absorb the disagreement.

One parameter per image, not two
--------------------------------
An image is a window of the filament at some axial position, viewed from some
direction. For a helically symmetric volume those are the same unknown -- see
``denovo3d_align`` -- so each image adds exactly one azimuth. That is also why
the azimuth enters through ``build_A_data_matrix``'s ``phi_degree``, a rotation
about the helical axis, rather than through a longer volume.

The sign is the trap. The solver rotates back-projected *coordinates*, which is
the inverse of rotating the object, so the azimuth ``denovo3d_align`` reports is
negated here rather than at every call site. Measured on a window of known
azimuth, passing the reported value reconstructs the volume the image at
azimuth zero gives (cc 0.97) while the opposite sign gives 0.08-0.66. The two
agree only at zero azimuth, which is why a sign error would survive any test
that used a single centred image.

The azimuths do real work
-------------------------
Measured on six noiseless synthetic windows spanning one period, as the cosine
similarity between the joint system's prediction and the data:

    azimuths correct     0.93
    azimuths shuffled    0.77
    all azimuths zero    0.75

so the machinery is assembling the images into one object rather than averaging
them into a blur.

What it does on real class averages
-----------------------------------
Measured on EMPIAR-10940 through the app pipeline, prepared as the tab prepares
them (median-Otsu threshold, per-image rotation from the tab's own estimator,
crop_center to the 32 px default), csym left unimposed, scanning 0.80-1.80 in
0.05 steps. Joint here means this module's azimuth method; single means the
existing per-image z-score combination:

    solver      images   joint peak   its margin   single peak   seconds
    gauss           10         1.20       0.0040          1.20        25
    gauss           10*        1.25       0.0034          1.25        25
    gauss           33         1.20       0.0159          1.20        25
    elasticnet      10         1.20       0.0921          1.25       247

    * a disjoint set of ten classes

Read that honestly. On the full 33 classes with a correct preparation the
existing per-image method already finds 1.20, and this one agrees with it
rather than rescuing it -- on every set tested the two give the same answer.
What it adds is a margin about twice as large on the full set, and on gauss a
placement that is stable at every twist (0.4-1.9 degrees of movement between
refinement iterations, against half-turn flips under the voxel solver).

The elasticnet row is included because it is the one case where the two methods
disagree, and it is not evidence for this method: elasticnet's per-image answer
is strongly sensitive to the vertical crop, giving 1.25 at 32 px and 1.05 at
48, so 1.25 there is a property of that solver's preparation sensitivity rather
than a failure the azimuth method repairs.

An earlier version of this docstring reported that the joint fit carried no
twist signal at all. That was measured on a synthetic filament so poor that the
existing method failed on it too -- picking 1.25 noiseless and 1.35 under noise
against a true 1.20 -- and no conclusion should have been drawn from it. The
lesson is cheap to state and was expensive to learn: reproduce the documented
baseline with the known-good method before judging a new one.

Two scores were tried on that synthetic data and remain untested on real data:
the mean correlation of each image to the model projection, and out-of-sample
placement agreement (rebuild without one image, re-place it, compare). Both
looked uninformative there, but so did everything else.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import vstack

import helicon

from .denovo3d_solver import (
    build_A_data_matrix,
    build_A_helical_sym_matrix,
    solve_equations,
)

logger = logging.getLogger(__name__)

MAX_EQUATIONS = 2**26


def _normalise(image):
    """Unit variance, so no image dominates the stacked system by contrast.

    The same hazard the score averaging in ``denovo3d_joint`` guards against:
    there, a curve with a large dynamic range swamped the others (measured 65x
    smaller margin); here an image with large amplitudes would simply be fitted
    first.
    """
    image = np.asarray(image, dtype=np.float32)
    sd = float(image.std())
    return image / sd if sd > 0 else image


def joint_reconstruct(
    images,
    phis,
    scale2d_to_3d,
    twist_degree,
    rise_pixel,
    csym=1,
    tilts=None,
    psis=None,
    dys=None,
    reconstruct_diameter_3d_inner_pixel=0,
    reconstruct_diameter_2d_pixel=-1,
    reconstruct_diameter_3d_pixel=-1,
    reconstruct_length_2d_pixel=-1,
    reconstruct_length_3d_pixel=-1,
    sym_oversample=1,
    interpolation="nn",
    positive_constraint=-1,
    algorithm=None,
    target_apix2d=5.0,
    score_metric="cosine",
    share_budget=True,
    verbose=0,
    cpu=1,
):
    """Solve one volume that explains every image at its own azimuth.

    ``phis`` gives one azimuth per image, in degrees, in the convention
    ``denovo3d_align.align_to_model`` reports -- the azimuth the image is a
    view of. The solver rotates back-projected *coordinates*, which is the
    inverse operation, so the sign is flipped here rather than at every call
    site: passing the reported azimuth straight through reconstructs the same
    volume as the image at azimuth zero (measured cc 0.97, against 0.08-0.66
    for the opposite sign). ``tilts``, ``psis`` and
    ``dys`` default to zero for every image: they are deliberately not fitted
    here, because aligning them to a shared reference makes the images agree
    with each other while drifting together away from horizontal (measured in
    ``denovo3d_joint``: 0.07 to 0.72 degrees of absolute tilt). Absolute
    orientation stays with the tab's auto-transform.

    Returns ``(volume, info)``. ``volume`` is the density on the cylindrical
    mask, shaped ``(nz, ny, nx)``; ``info`` carries the fit -- ``score``, the
    cosine similarity between the joint system's prediction and the data, and
    ``per_image``, the same quantity for each image's own equations.

    The joint score is the quantity worth ranking twists by. Every candidate
    twist is given the same number of unknowns and the same images, and differs
    only in the symmetry operator that ties them together, so the comparison is
    like for like -- and unlike scoring each image separately, one volume has
    to satisfy all of them with only a single azimuth each to absorb the
    disagreement.
    """
    n = len(images)
    if n == 0:
        raise ValueError("no images")

    # The Gaussian basis has its own joint form, and a far cheaper one: its
    # normal equations stay (G, G) however many images are added, where this
    # module's voxel system grows an equation block per image and has to ration
    # them. Dispatch rather than duplicate the geometry.
    if (algorithm or {}).get("model") == "gauss":
        from .solver_gauss_analytic import gauss_joint_reconstruct

        return gauss_joint_reconstruct(
            images,
            phis,
            scale2d_to_3d=scale2d_to_3d,
            twist_degree=twist_degree,
            rise_pixel=rise_pixel,
            csym=csym,
            reconstruct_diameter_2d_pixel=reconstruct_diameter_2d_pixel,
            reconstruct_diameter_3d_pixel=reconstruct_diameter_3d_pixel,
            reconstruct_length_3d_pixel=reconstruct_length_3d_pixel,
            target_apix2d=target_apix2d,
            algorithm=algorithm,
            verbose=verbose,
        )
    tilts = [0.0] * n if tilts is None else tilts
    psis = [0.0] * n if psis is None else psis
    dys = [0.0] * n if dys is None else dys

    rmin = reconstruct_diameter_3d_inner_pixel / 2
    rmax = reconstruct_diameter_3d_pixel // 2 - 1
    mask = helicon.get_cylindrical_mask(
        nz=reconstruct_length_3d_pixel,
        ny=reconstruct_diameter_3d_pixel,
        nx=reconstruct_diameter_3d_pixel,
        rmin=rmin,
        rmax=rmax,
    )
    mz, my, mx = mask.shape
    n_3d_voxels = np.count_nonzero(mask)
    n_2d_pixels = reconstruct_diameter_2d_pixel * reconstruct_length_2d_pixel

    # Equations are shared out between the images rather than granted to each,
    # so a joint solve costs about what a single-image solve costs. They are
    # symmetry copies of the same rays, so what is lost is redundancy; what is
    # gained is that the volume must satisfy every image at once.
    if sym_oversample <= 0:  # "auto", the value the tab passes by default
        ratio = 2**20 / max(
            1,
            reconstruct_length_3d_pixel
            * (
                reconstruct_diameter_3d_pixel**2
                - reconstruct_diameter_3d_inner_pixel**2
            ),
        )
        if ratio < 10:
            sym_oversample = max(1, int(round(ratio)))
        elif ratio < 100:
            sym_oversample = max(1, int(round(ratio / 10)) * 10)
        else:
            sym_oversample = max(1, int(round(ratio / 100)) * 100)

    budget = min(MAX_EQUATIONS, int(max(n_2d_pixels, n_3d_voxels) * sym_oversample))
    per_image = max(1, budget // n) if share_budget else budget

    blocks, rhs = [], []
    for i, image in enumerate(images):
        A_i, b_i, _ = build_A_data_matrix(
            image=_normalise(image),
            scale2d_to_3d=scale2d_to_3d,
            twist_degree=twist_degree,
            rise_pixel=rise_pixel,
            csym=csym,
            tilt_degree=tilts[i],
            psi_degree=psis[i],
            dy_pixel=dys[i],
            reconstruct_diameter_2d_pixel=reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel=reconstruct_length_2d_pixel,
            reconstruct_diameter_3d_pixel=reconstruct_diameter_3d_pixel,
            reconstruct_diameter_3d_inner_pixel=reconstruct_diameter_3d_inner_pixel,
            reconstruct_length_3d_pixel=reconstruct_length_3d_pixel,
            min_projection_lines=per_image,
            interpolation=interpolation,
            verbose=verbose,
            cpu=cpu,
            phi_degree=-float(phis[i]),
        )
        blocks.append(A_i)
        rhs.append(b_i)

    A_hsym, b_hsym = build_A_helical_sym_matrix(
        nz=mz,
        ny=my,
        nx=mx,
        twist_degree=twist_degree,
        rise_pixel=rise_pixel,
        csym=csym,
        rmin=rmin,
        rmax=rmax,
        min_sym_pairs=budget,
        interpolation=interpolation,
        verbose=verbose,
    )
    A_data = vstack(blocks) if len(blocks) > 1 else blocks[0]
    b_data = np.concatenate(rhs)

    # Non-negativity, for the reason recorded in denovo3d_solver: a free-sign
    # volume can explain any twist, so dropping it does not merely cost
    # precision, it removes the signal. It matters more here, not less: the
    # joint system has more images for a free-sign volume to accommodate.
    pitch_pixel = round(rise_pixel * 360 / abs(twist_degree))
    positive = positive_constraint > 0 or (
        positive_constraint < 0 and pitch_pixel > round(reconstruct_length_3d_pixel * 2)
    )

    with helicon.Timer(
        f"joint_reconstruct: {n} images, {A_data.shape[0]:,} equations, "
        f"{A_data.shape[1]:,} unknowns",
        verbose=verbose > 10,
    ):
        x, _ = solve_equations(
            A_data,
            b_data,
            A_hsym,
            b_hsym,
            positive=positive,
            algorithm=algorithm or dict(model="lsq"),
            score_metric=score_metric,
            img_shape_2d=(reconstruct_length_2d_pixel, reconstruct_diameter_2d_pixel),
            verbose=verbose,
        )

    x = np.asarray(x, dtype=np.float32)
    per_image = []
    for A_i, b_i in zip(blocks, rhs):
        per_image.append(float(helicon.cosine_similarity(A_i.dot(x), b_i)))
    score = float(helicon.cosine_similarity(A_data.dot(x), b_data))

    volume = np.zeros(mask.shape, dtype=np.float32)
    volume[np.nonzero(mask)] = x
    return volume, dict(score=score, per_image=per_image, positive=bool(positive))
