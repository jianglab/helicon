"""Helical twist solver on a Gaussian basis.

The working method fits the class average directly in pixel space: a fixed
basis of Gaussians spanning the reconstruction volume, replicated by the screw
symmetry and projected, with the coefficients solved as a non-negative
elasticnet on the pixel residual. One twist is scored by :func:`twist_score`.

Projection is analytic -- a 3D Gaussian projects to a 2D Gaussian exactly -- so
the design matrix is built from closed-form components rather than by
symmetrising and integrating a voxel volume.

Geometry matches the app's own solver: centres are (x, y, z) with z the helical
axis, projection is along x onto the (y, z) plane, and the image is
(transverse, axial). Verified directly -- projecting a structure through
:func:`helicon.apply_helical_symmetry` and through this module agree to
cc = 0.999872, with every mirrored variant far worse. Everything works in pixels.

THE SUPPORT MUST COVER THE WHOLE FIELD
--------------------------------------
Build the basis over the image half-height, not the filament radius. Earlier
versions used ``diameter/2 * 0.9``, which cuts inside a 17 px filament, and
that single choice biased the recovered twist upward by 0.6-0.7 degrees and
made this whole approach look fundamentally broken. On three classes, changing
nothing else:

    radius = 0.9 * diam/2   joint 1.90    med s@1.2/max 0.9186
    radius = diam/2         joint 1.90    med s@1.2/max 0.9641
    radius = image half     joint 1.20    med s@1.2/max 1.0000

The mechanism explains the sign. A basis function at radius r sweeps a
transverse range proportional to r as the symmetry copies rotate, so
reproducing a filament of a given apparent width needs either large enough
radii or more twist per copy; capping the support removes the first option and
the fit compensates with the second. A synthetic structure of radius 4 px
reproduces it exactly -- supports of 2.5 and 3.0 px peak at 2.0 against a true
1.2 -- and both directions are pinned in the tests.

MEASURED PERFORMANCE
--------------------
All 32 good EMPIAR-10940 class averages (true twist 1.2), prepared exactly as
the tab prepares them and scanned through the app pipeline. Measured over
0.8-1.8 deg in 0.05 steps, which resolves the peak; "prominence" is how far the
joint peak stands above the run of the curve in units of its own spread, and
"margin" how far above the best competitor more than 0.15 deg away, as a
fraction of the curve's range:

    method              joint peak   prominence   margin   per-image   ms
    gauss, old default        1.15         0.60    0.055       11/32   ~70
    gauss, current            1.20         0.62    0.065       19/32   111
    elasticnet                1.20         1.14    0.130       23/32  1024

On the tab's own defaults instead -- 0.1 to 2.0 deg in 0.1 steps -- both solvers
put the joint peak on 1.2, with prominence 0.84 for gauss against 1.24 for
elasticnet. The coarser grid also drops the per-image rate to 11/32 and 21/32,
because images whose own peak sits at 1.15 or 1.25 have nowhere to land.

Read that honestly. The current defaults fix the joint ANSWER, which the old
ones got wrong by a grid step, and nearly double the number of single class
averages that land on the truth. They do NOT make the peak stand out better
than elasticnet's: elasticnet's peak is about twice as prominent and its margin
twice as large, and it is the better solver on a single image.

What gauss is for is the search: it reaches the same joint answer about eight
times faster. Use it to scan, and elasticnet to reconstruct.

An attempt was made to beat elasticnet's prominence and it failed. Sigma across
the filament (2.5 to 7 A), sigma along it (off, 12.5 A), L1 (0.01 to 0.3) and
the score metric were swept together on the correct images; nothing reached
elasticnet's 1.13, and the best gauss configuration is the one above. Do not
assume the gap is a tuning problem.

A PREVIOUS VERSION OF THIS SECTION WAS WRONG, AND THE REASON MATTERS
---------------------------------------------------------------------
It claimed gauss scored 0.92 against elasticnet's 0.88 -- sharper than the
reference, four times the margin, twice the per-image rate. Every one of those
numbers came from a harness that prepared the images itself rather than the way
the tab does, and drifted from it by about 3% in score. Per-image twist margins
are 1e-3 to 1e-4, so a 3% shift is not a detail: it reversed the conclusion.

A harness is not the app until its numbers match the app's. Verify numerically
-- run one class average through the tab, read the score curve out of the
rendered plot, and require agreement to about four decimals -- before believing
anything a standalone harness says. The three things a reimplementation gets
wrong here are the Otsu threshold being a median over the SELECTED images, the
rotation and shift being applied in ONE transform_image call rather than two,
and both reaching the solver rounded to two decimals by the numeric widgets.

WHY THE CURVE IS SHAPED THE WAY IT IS
--------------------------------------
Two properties of the fit drive the search curve, and both are needed for the
joint answer to come out right.

The basis is narrow ACROSS the filament. Transverse width is what lets the fit
smear a wrong twist into a plausible image, so a broad basis scores well at
every twist. Widening sigma back to the old 5 A costs the joint answer (1.15,
11/32 correct) even with everything else kept.

The score is a Pearson correlation, not a raw cosine. The model is non-negative
by construction and the image is thresholded, so both vectors sit in the
positive orthant and their cosine cannot fall much below a floor near 0.72,
with the twist modulating the result on top of it. :func:`correlation_score`
centres both vectors. With raw cosine the current basis gives 1.15 and 8/32;
with centring, 1.20 and 19/32.

Neither alone is enough: centring the score on the old wide basis gives 1.25,
worse than the old default it would replace.

THE RECONSTRUCTION IS BASIS-LIMITED, AND IT IS NOT DRAWN WITH THE FIT'S BASIS
-----------------------------------------------------------------------------
By FSC against the elasticnet volume the map is good to roughly 20 A. Ample for
finding the twist, not for looking at the result; use elasticnet for the map.

The search basis and the rendering basis are deliberately different: the search
wants narrow Gaussians and the map wants broad ones, so the volume is drawn with
``DEFAULT_RENDER_SIGMA_ANGSTROM``. Rendering the narrow search basis directly
halves the fraction of voxels carrying density (0.172 to 0.083) and looks spiky.

An earlier version of this note said flatly "do not lower sigma", on the
strength of a run where sigma 4 A improved per-image counts while the joint peak
moved to 1.1 and the reconstruction grew about twice the crossovers the geometry
allows. That was a fair reading of the evidence then and is too strong as
general advice: with the score centred, the axial width held at 12.5 A, and the
map rendered from its own wider basis, sigma 3.5 A improves the joint peak, the
per-image rate, and the map together -- mean FSC against the elasticnet volume
over four classes goes from 0.528 to 0.540. What made the old result bad was
narrowing the basis in all three directions at once while scoring with a
floor-bound cosine.

The standing lesson from that episode is unchanged and is the important part:
judge this solver on the joint answer and the reconstruction together, never on
per-image counts alone -- and measure both on images prepared the way the tab
prepares them.

SHORT FILAMENTS ARE A KNOWN LIMIT
----------------------------------
Twist discrimination needs a crossover to look at, and half a pitch at twist 1.2
and rise 4.75 A is 712 A. On images much shorter than that the narrow basis is
less reliable than the broad one it replaced -- on a 320 A synthetic it returns
1.4 to 2.2 against a true 1.2 -- and the twist is weakly determined by such an
image whatever the method. At 640 A it is correct at every structure width
tested. Pinned in TestShortFilamentsAreNotReliablyIndexable.

THE RECONSTRUCTION FILLS THE AXIS -- PARTLY, AND ONLY PARTLY, FIXABLE
----------------------------------------------------------------------
Use this solver for the twist/rise search, and elasticnet for the final map.

On a hollow filament the reprojection shows a bright core where the data has a
dark lane between the strands. The strands themselves are placed correctly --
transverse peak offsets match the input column by column and there is no axial
phase error -- but the space between them is filled. On the EMPIAR-10940
classes, whose middle column sits at 0.079 of maximum, the fit gives 0.64.

Under the screw a basis function at radius r projects to y = r sin(theta(z)),
which crosses the axis only at isolated z -- the real crossovers. What fills the
lane is the function's TRANSVERSE WIDTH smearing each crossing. An earlier
version of this note claimed the crossings fill the axis regardless of the
basis, which is wrong: narrowing sigma across the filament measurably improves
it, which is why the basis is anisotropic (sigma 3.5 A across, 12.5 A along).

That only goes so far. Narrower still keeps improving the gap (sigma 3 A reaches
0.55) but once 2*sigma is below the grid spacing the basis stops overlapping and
individual functions show through as horizontal stripes -- traded one artefact
for another. Held contiguous, the gap reaches about 0.64 against the data's
0.079.

The remaining gap is non-negativity. Allowing free-sign amplitudes fits the
image almost exactly -- cc rises from 0.73 to 0.91 and the centre falls to 0.013
-- and simultaneously moves the joint twist from 1.2 to 1.4. The constraint that
spoils the picture is the one that makes the twist identifiable, because without
it the model explains any twist. The default keeps the twist.

This is a property of the problem, not of the Gaussian basis, and an earlier
version of this note implied otherwise. The voxel solver behaves the same way:
elasticnet runs with its positivity constraint on -- the auto rule turns it on
whenever the pitch exceeds twice the reconstruction length, which is every
realistic case here -- and turning it off moves its joint peak from 1.20 to 1.45
and drops its peak prominence from 1.00 to 0.65 on ten good classes. Both
solvers need non-negativity to make the twist identifiable, and both degrade to
about 1.4 without it. So the gauss/elasticnet difference in peak prominence,
whatever causes it, is not that one of them may use negative density and the
other may not; measured with both constrained, the gap is still there (0.72
against 1.00 on the same ten images).

EARLIER MEASUREMENTS ON THIS MODULE ARE VOID
---------------------------------------------
Everything measured here before the support was fixed was measured with a model
that could not represent the structure, and should not be relied on or cited.
That includes: the conclusion that this approach is "not competitive"; the claim
that sigma is a hyperparameter swinging the answer over 0.60-1.35; the reported
per-image rate of 2/8; the finding that a Gaussian basis degrades faster than
the voxel solver under added noise; the claim that arc-length basis sampling
matters for twist bias; and the note that a non-negative lasso scored worse than
the plain solve. None of these has been re-measured with the correct support.

The invariant that ran through all of them -- the score at the true twist
sitting 1-9% below the best twist -- was not evidence that Gaussians cannot
discriminate twist. It was evidence that the true twist was never representable.
With the full field that statistic is 0.9961.

Still valid from that period, because it was established synthetically,
numerically, or by direct comparison: the geometry check above; the overlap
algebra in ``gauss_mixture``; that truncating the axial sum too tightly lets the
normalised score exceed one; and that building the Gram by rasterising the basis
for BLAS is about ten times slower than the analytic pair sum.

Selectable in the tab as the "gauss" search algorithm. The tab offers a
separate reconstruction algorithm, which should stay on elasticnet for the
reason above.
"""

from __future__ import annotations

import numpy as np

from .gauss_mixture import gen_overlap, iso_overlap, SQRT_2PI

__all__ = [
    "cylindrical_grid",
    "full_field_grid",
    "design_matrix",
    "nn_elasticnet",
    "explained_variance",
    "twist_score",
    "support_radius",
    "fit_spacing_to_budget",
    "gauss_analytic_reconstruct",
    "correlation_score",
    "DEFAULT_SIGMA_Z_ANGSTROM",
    "DEFAULT_RENDER_SIGMA_ANGSTROM",
    "expand_project",
    "gram_matrix",
    "cross_vector",
    "best_score",
    "nn_lasso",
]


def cylindrical_grid(radius_px, sigma_px):
    """Basis centres on a cylinder, sampled at ~``sigma_px`` in radius and arc.

    Sampling arc length rather than using a fixed number of angular steps keeps
    the outer shells from being sampled more coarsely than sigma, which is
    sensible on its own terms. The stronger claim once made here -- that a
    constant angular count biases the recovered twist -- was measured with the
    broken support and has not been re-tested; do not rely on it.

    For the working pixel-space method prefer :func:`full_field_grid`, whose
    support covers the image rather than the filament.
    """
    pts = [[0.0, 0.0, 0.0]]
    n_r = max(1, int(round(radius_px / sigma_px)))
    for r in np.linspace(0.0, radius_px, n_r + 1)[1:]:
        n_a = max(4, int(round(2 * np.pi * r / sigma_px)))
        for th in np.linspace(0.0, 2 * np.pi, n_a, endpoint=False):
            pts.append([r * np.cos(th), r * np.sin(th), 0.0])
    centres = np.asarray(pts, float)
    return centres, np.full(len(centres), float(sigma_px))


def expand_project(
    centres,
    sigmas,
    twist_deg,
    rise_px,
    csym,
    n_repeats,
    envelope=None,
    phi_degree=0.0,
):
    """Replicate the asymmetric unit by the screw symmetry and project along x.

    ``phi_degree`` turns the whole assembly about the helical axis before
    projecting, which is what lets several images share one basis while each
    views it from its own direction. It enters exactly where the screw's own
    rotation does, since both are rotations about that axis and they commute.

    This is the raw basis rotation, positive in the usual sense: a centre on
    +x moves to +y at 90 degrees. It is NOT the azimuth
    ``denovo3d_align.align_to_model`` reports -- measured against a window of
    known azimuth, the reported value has to be negated here (cc 0.97-0.99
    against 0.07-0.40 for the other sign), the same as in the voxel solver's
    ``build_A_data_matrix``. Reasoning about which way each convention turns
    got this backwards; the calibration is what settles it. Callers should use
    :func:`gauss_joint_reconstruct`, which negates for them.

    ``envelope`` is an optional callable ``z -> weight`` applied per copy. Some
    truncation is genuinely needed: a class average is a finite segment while a
    strictly symmetric model extends uniformly, and without an envelope that
    mismatch is absorbed by whichever twist compensates for it.

    It must, however, be **twist-independent** -- a boxcar over the image
    extent. Passing the image's own axial profile injects the crossover
    modulation, which is the twist signal, so the fit then explains the image
    partly with a statistic drawn from that same image. This docstring
    previously cited such a run (fit 0.88 -> 0.96, twist 1.45 -> 1.10 on class
    2) as evidence the envelope helped; that result is a confound and should
    not be relied on. With a freer model the same envelope pins every test
    image to the top of the scan -- see ``solver_gauss_aniso``.

    The working pixel-space method needs no envelope at all: the image frame
    truncates the model, so :func:`design_matrix` passes none. An envelope is
    only relevant to the analytic-target path, which integrates over infinite
    space.

    Returns ``(mu, amp, sigma)`` shaped ``(G, C, 2)``, ``(G, C)``, ``(G, C)``.
    """
    centres = np.asarray(centres, float)
    sigmas = np.asarray(sigmas, float)
    ks = np.arange(-n_repeats, n_repeats + 1, dtype=float)
    js = np.arange(csym, dtype=float)
    K, J = np.meshgrid(ks, js, indexing="ij")
    k, j = K.ravel(), J.ravel()
    ang = (
        np.deg2rad(twist_deg) * k
        + 2 * np.pi * j / max(csym, 1)
        + np.deg2rad(phi_degree)
    )
    ca, sa = np.cos(ang), np.sin(ang)

    x, y, z = centres[:, 0:1], centres[:, 1:2], centres[:, 2:3]
    mu = np.stack(
        [x * sa[None, :] + y * ca[None, :], z + rise_px * k[None, :]], axis=-1
    )
    # Projecting an isotropic 3D Gaussian along one axis scales the amplitude
    # by sigma*sqrt(2*pi) and leaves the in-plane sigma unchanged.
    amp = np.broadcast_to((sigmas * SQRT_2PI)[:, None], mu.shape[:2]).copy()
    sig = np.broadcast_to(sigmas[:, None], mu.shape[:2]).copy()
    if envelope is not None:
        amp = amp * envelope(mu[..., 1])
    return mu, amp, sig


# Cap on one intermediate of the Gram sum. A fine basis makes G large, and the
# sum forms (G, G, pairs) arrays -- at G=800 with 137 copies that is 700 MB per
# offset. Bounding it matters: an unbounded version of the analogous sum in
# solver_gauss_aniso once exhausted the machine's memory.
GRAM_CHUNK_BYTES = 64 << 20


def _gram_rows(n_gauss, n_pairs, itemsize=8):
    per_row = max(1, n_gauss * n_pairs * itemsize)
    return max(1, min(n_gauss, GRAM_CHUNK_BYTES // per_row))


def gram_matrix(basis, dk_max):
    """``M_gh = <B_g, B_h>``, skipping copy pairs more than ``dk_max`` apart.

    Copies separated by many repeats along the axis overlap by nothing, so the
    sum truncates without meaningful error. ``dk_max`` should cover about five
    sigma, which is what :func:`_dk_for` computes. Truncating too tightly
    under-counts ``<f,f>`` and lets the normalised score exceed one, which is a
    useful tripwire -- a cosine above 1 always means the truncation is wrong,
    never that the fit is good.
    """
    mu, amp, sig = basis
    G, C = amp.shape
    M = np.zeros((G, G))
    idx = np.arange(C)
    rows = _gram_rows(G, C)
    for d in range(-dk_max, dk_max + 1):
        i = idx[max(0, -d) : C - max(0, d)]
        if len(i) == 0:
            continue
        jj = i + d
        for lo in range(0, G, rows):
            hi = min(G, lo + rows)
            M[lo:hi] += iso_overlap(
                amp[lo:hi, None, i],
                mu[lo:hi, None, i, :],
                sig[lo:hi, None, i],
                amp[None, :, jj],
                mu[None, :, jj, :],
                sig[None, :, jj],
            ).sum(-1)
    return M


def cross_vector(basis, target):
    """``u_g = <B_g, t>`` for an isotropic basis against a general target."""
    mu, amp, sig = basis
    ta, tm, tc = target
    cov = np.zeros(mu.shape + (2,))
    cov[..., 0, 0] = sig**2
    cov[..., 1, 1] = sig**2
    return gen_overlap(
        amp[:, :, None],
        mu[:, :, None, :],
        cov[:, :, None, :, :],
        ta[None, None, :],
        tm[None, None, :, :],
        tc[None, None, :, :, :],
    ).sum(axis=(1, 2))


def nn_lasso(M, u, lam, iters=300, tol=1e-10):
    """``min 0.5 a^T M a - u^T a + lam ||a||_1`` subject to ``a >= 0``.

    Coordinate descent on the Gram system. Convex, so one optimum and no
    restarts; the basis is small enough that this costs nothing next to
    building ``M``.

    Superseded by :func:`nn_elasticnet`, which adds an L2 term and, more
    importantly, is meant for the least-squares objective where penalties
    actually bind.
    """
    G = len(u)
    a = np.zeros(G)
    d = np.diag(M).copy()
    d[d <= 0] = 1e-12
    r = u.copy()
    for _ in range(iters):
        delta = 0.0
        for j in range(G):
            aj = a[j]
            new = max(0.0, (r[j] + d[j] * aj - lam) / d[j])
            if new != aj:
                r -= M[:, j] * (new - aj)
                delta = max(delta, abs(new - aj))
                a[j] = new
        if delta < tol:
            break
    return a


def best_score(basis, target, target_energy, ridge=1e-9, lam=0.0):
    """Best cosine between the projected model and the target, and the amplitudes.

    With ``lam = 0`` this is the unconstrained optimum in closed form; a
    positive ``lam`` switches to the non-negative lasso.

    Note this scores a **cosine**, which is scale-invariant, so an L1 penalty
    against it is close to meaningless -- shrinking every amplitude lowers the
    penalty without changing the score. That is why penalties appeared to do
    nothing here. For a penalty that actually binds, use the least-squares path
    (:func:`nn_elasticnet` with :func:`explained_variance`), which is what the
    working method uses. The earlier note that the lasso "scored worse on real
    data" was measured with the broken support and is void.
    """
    u = cross_vector(basis, target)
    M = gram_matrix(basis, dk_max=_dk_for(basis))
    M = M + ridge * np.trace(M) / len(M) * np.eye(len(M))
    a = nn_lasso(M, u, lam * np.abs(u).max()) if lam > 0 else np.linalg.solve(M, u)
    num = float(a @ u)
    den = float(a @ M @ a)
    if num <= 0 or den <= 0 or target_energy <= 0:
        return 0.0, a
    return num / np.sqrt(den * target_energy), a


def _dk_for(basis):
    mu, amp, sig = basis
    if mu.shape[1] < 2:
        return 0
    rise = abs(float(mu[0, 1, 1] - mu[0, 0, 1])) or 1.0
    # Five sigma, not four: at four the truncation under-counts <f,f> enough
    # to push the normalised score slightly above one (measured 1.0001).
    return int(np.ceil(5 * float(sig.max()) / rise)) + 1


def full_field_grid(radius_px, spacing_px=2.0, sigma_px=1.2):
    """Cartesian basis covering the WHOLE field, not just the filament.

    ``radius_px`` should be the image half-height, not the filament radius.
    Restricting the basis to the filament biases the recovered twist upward by
    0.6-0.7 degrees; see this module's status notes.

    A single z plane suffices at the default sigma, which exceeds the rise, so
    the symmetry copies overlap into continuous density. A version of this
    function once added z planes across one rise on the theory that a finer
    sigma leaves a comb along the axis whose aliasing explains the artefacts a
    fine basis produces. It was measured and it does not: the artefacts are
    unchanged with the extra planes, so the theory was wrong and the complexity
    is not carried.
    """
    n = int(np.ceil(radius_px / spacing_px))
    xs = np.arange(-n, n + 1) * spacing_px
    pts = [[x, y, 0.0] for x in xs for y in xs if x * x + y * y <= radius_px**2]
    centres = np.asarray(pts, float)
    return centres, np.full(len(centres), float(sigma_px))


def design_matrix(
    grid,
    twist_deg,
    rise_px,
    csym,
    n_repeats,
    ny,
    nx,
    cutoff_sigma=3.5,
    sigma_z=None,
    chunk=200000,
    phi_degree=0.0,
):
    """``(n_pixels, G)`` matrix whose column g is basis fn g with all its copies.

    Each column is dense-ish -- a basis function's copies run the length of the
    filament -- but each individual copy only reaches a few sigma, so the work
    is done per copy over a small footprint rather than per column over the
    whole image. That is the difference between ``P * G * C`` exponentials and
    ``G * C * (2*cutoff*sigma)^2``, about fifty-fold here.

    ``sigma_z`` makes the basis anisotropic: the grid's sigma then applies
    across the filament and ``sigma_z`` along it. The projection of a Gaussian
    is separable in (y, z), so this is free. It matters because a basis function
    swept by the screw traces ``y = r sin(theta(z))``, which crosses the axis
    only at isolated z; a transversely wide function smears each crossing across
    the dark lane between strands, while a narrow one does not.
    """
    centres, sigmas = grid
    mu, amp, sig = expand_project(
        centres, sigmas, twist_deg, rise_px, csym, n_repeats, phi_degree=phi_degree
    )
    sz = float(sigma_z) if sigma_z else None
    G, C = amp.shape
    g_of = np.repeat(np.arange(G), C)
    yc = mu[..., 0].ravel() + ny / 2.0
    zc = mu[..., 1].ravel() + nx / 2.0
    a = amp.ravel()
    s = sig.ravel()
    s_z = np.full_like(s, sz) if sz else s

    hy = int(np.ceil(cutoff_sigma * float(s.max())))
    hz = int(np.ceil(cutoff_sigma * float(s_z.max())))
    offy = np.arange(-hy, hy + 1)
    offz = np.arange(-hz, hz + 1)

    keep = (yc > -hy) & (yc < ny + hy) & (zc > -hz) & (zc < nx + hz) & (a != 0.0)
    yc, zc, a, s, s_z, g_of = (
        yc[keep],
        zc[keep],
        a[keep],
        s[keep],
        s_z[keep],
        g_of[keep],
    )

    A = np.zeros(ny * nx * G)
    for lo in range(0, yc.size, chunk):
        hi = min(yc.size, lo + chunk)
        r0 = np.floor(yc[lo:hi]).astype(np.int64)[:, None] + offy[None, :]
        c0 = np.floor(zc[lo:hi]).astype(np.int64)[:, None] + offz[None, :]
        ey = np.exp(-0.5 * ((r0 - yc[lo:hi, None]) / s[lo:hi, None]) ** 2)
        ez = np.exp(-0.5 * ((c0 - zc[lo:hi, None]) / s_z[lo:hi, None]) ** 2)
        vals = a[lo:hi, None, None] * ey[:, :, None] * ez[:, None, :]
        valid = ((r0 >= 0) & (r0 < ny))[:, :, None] & ((c0 >= 0) & (c0 < nx))[
            :, None, :
        ]
        flat = (
            np.clip(r0, 0, ny - 1)[:, :, None] * nx + np.clip(c0, 0, nx - 1)[:, None, :]
        ) * G + g_of[lo:hi, None, None]
        A += np.bincount(
            flat[valid].ravel(), weights=vals[valid].ravel(), minlength=ny * nx * G
        )
    return A.reshape(ny * nx, G)


def nn_elasticnet(M, u, lam1=0.0, lam2=0.0, iters=500, tol=1e-12):
    """``min a^T M a - 2 a^T u + lam1||a||_1 + lam2||a||^2``, ``a >= 0``.

    Least squares, not a cosine: a cosine is scale-invariant, so L1/L2 against
    it is degenerate -- shrinking every amplitude lowers the penalty without
    changing the fit, and the penalties simply do nothing.
    """
    G = len(u)
    a = np.zeros(G)
    d = np.diag(M).copy()
    d[d <= 0] = 1e-12
    r = u.copy()
    for _ in range(iters):
        delta = 0.0
        for j in range(G):
            aj = a[j]
            new = max(0.0, (r[j] + d[j] * aj - 0.5 * lam1) / (d[j] + lam2))
            if new != aj:
                r -= M[:, j] * (new - aj)
                delta = max(delta, abs(new - aj))
                a[j] = new
        if delta < tol:
            break
    return a


def explained_variance(M, u, tt, a):
    """``1 - ||f_a - t||^2 / ||t||^2``; comparable across twists, at most 1."""
    if tt <= 0:
        return 0.0
    return float((2.0 * (a @ u) - a @ M @ a) / tt)


def twist_score(image, grid, twist_deg, rise_px, csym=1, lam1=1e-2, lam2=1e-2):
    """Score one twist against an image: fit the basis, report explained variance."""
    ny, nx = image.shape
    n_repeats = int(nx / 2 / max(rise_px, 1e-6)) + 2
    A = design_matrix(grid, twist_deg, rise_px, csym, n_repeats, ny, nx)
    b = np.asarray(image, float).ravel()
    M = A.T @ A
    u = A.T @ b
    a = nn_elasticnet(M, u, lam1 * np.abs(u).max(), lam2 * np.trace(M) / len(M))
    return explained_variance(M, u, float(b @ b), a)


# Basis defaults, in angstroms so they follow the pixel size rather than being
# tied to the one dataset they were tuned on. At the tab's default 5 A/px these
# are 2 px spacing and 1.2 px sigma.
#
# DO NOT make the basis finer without re-checking the twist and the
# reconstruction together. A finer basis scores better on per-image twist
# counts and gives a higher-resolution-looking map, and is nonetheless worse:
# measured on the first ten good class averages, as the app runs them,
#
#     sigma 4A   joint 1.1   3/10 at the true twist   67 s
#     sigma 6A   joint 1.2   6/10                     18 s
#
# and at 4 A the reconstruction shows about five crossovers across the volume
# where the geometry allows three (pitch/2 = 142 px at twist 1.2, over a 345 px
# volume). That spurious doubling is not understood. A 4 A default shipped
# briefly on the strength of per-image counts from a standalone harness, and
# was wrong: the joint peak is what the tab reports, and it moved away from the
# truth. Per-image counts are not a sufficient criterion on their own.
DEFAULT_SPACING_ANGSTROM = 10.0
# Narrower than half the spacing, which is what it used to be. The basis is a
# sampling basis for the search, not a density model, and narrow samples keep
# the fit from reproducing the image at a wrong twist. Measured over the 32 good
# classes, prepared as the tab prepares them:
#
#                                joint peak   prominence   per-image
#     sigma 5.0 A, raw cosine          1.15         0.60       11/32
#     sigma 3.5 A, centred score       1.20         0.62       19/32
#     elasticnet                       1.20         1.13       23/32
#
# The gain is in the joint answer and the per-image rate, not in prominence --
# see the module docstring, which records the failed attempt to close that gap.
DEFAULT_SIGMA_ANGSTROM = 3.5  # across the filament, for the fit
# Along the axis the basis must stay WIDE, and for a different reason. Screw
# copies sit one rise apart -- 4.75 A here -- so a basis narrow in z leaves them
# as a comb instead of continuous density. That shows up on synthetic structures
# of known twist rather than on the class averages: dropping sigma_z to match
# sigma scores BETTER on the real images (prominence 0.83, 22/32) and gets the
# synthetic answer wrong in 3 of 9 cases. The synthetic test wins, because it
# has ground truth and the class averages only have a consensus.
DEFAULT_SIGMA_Z_ANGSTROM = 12.5  # along it
# Rendering wants the opposite of what the search wants, so the map is drawn
# with a wider Gaussian than the fit uses; rendering the search basis directly
# gives a spiky volume. Mean FSC against the elasticnet volume over four
# classes: 0.528 with the old wide basis, 0.540 now, so the map is slightly
# better rather than sacrificed.
DEFAULT_RENDER_SIGMA_ANGSTROM = 5.0
# Sparse enough to stop the fit explaining a wrong twist, loose enough to keep
# the answer. Swept from 0.01 to 0.3 on correctly prepared images; 0.3 is where
# the joint peak sits on the truth rather than a grid step above it.
DEFAULT_L1 = 0.3
DEFAULT_L2 = 1e-2

# The support has to reach about twice the filament radius; less than that and
# the recovered twist is biased upward (see the module docstring). Measured over
# eight classes, with everything else fixed:
#
#     0.75 x diam/2   joint 2.00   med s@1.2/max 0.9320
#     1.0  x diam/2   joint 1.90                 0.9586
#     1.5  x diam/2   joint 1.30                 0.9980
#     2.0  x diam/2   joint 1.20                 0.9999
#
# The margin above 2.0 guards against an under-estimated diameter. It is capped
# by the image half-height, because basis functions outside the image explain
# nothing, and by MAX_BASIS so a large box cannot blow up the design matrix:
# the basis grows as radius^2, and at a 256 px box an uncapped radius would
# reach ~12800 functions and a multi-gigabyte matrix.
SUPPORT_DIAMETER_FACTOR = 2.5
MAX_BASIS = 1500


def correlation_score(pred, target):
    """Pearson correlation between the model and the image.

    Not the plain cosine the other solvers report, and the difference matters
    here. This model is non-negative by construction and the image has been
    thresholded, so both vectors lie in the positive orthant and their cosine
    cannot fall much below a large floor -- measured around 0.72 for a filament
    that is mostly empty frame. The twist then modulates the score by a couple
    of parts in a thousand on top of that floor, which is what makes the search
    curve look flat. Centring both vectors removes the floor, which is the
    standard normalised cross-correlation used for image matching.

    It is only safe together with the narrow basis. On the old sigma 5 A basis,
    centring moves the joint peak to 1.25 on the 32 good classes -- still wrong;
    with sigma 3.5 A it lands on 1.20, and the per-image rate goes from 8/32 to
    19/32. The two changes are one change.

    Because the score is centred, it is a ranking statistic for this solver and
    is not numerically comparable with the cosine the lsq family reports.
    """
    p = np.asarray(pred, float)
    t = np.asarray(target, float)
    p = p - p.mean()
    t = t - t.mean()
    denom = np.linalg.norm(p) * np.linalg.norm(t)
    return float(p @ t / denom) if denom > 0 else 0.0


def support_radius(ny, filament_diameter_px=None):
    """Radius for the basis support, in pixels."""
    half = ny / 2.0
    if not filament_diameter_px or filament_diameter_px <= 0:
        return half
    return float(min(half, SUPPORT_DIAMETER_FACTOR * filament_diameter_px / 2.0))


def fit_spacing_to_budget(radius_px, spacing_px, sigma_px, max_basis=MAX_BASIS):
    """Coarsen the grid if the requested one would exceed the basis budget.

    Keeps sigma tied to the spacing so the basis stays able to represent a
    smooth density rather than turning into isolated spikes.
    """
    ratio = sigma_px / max(spacing_px, 1e-9)
    while True:
        n = len(full_field_grid(radius_px, spacing_px, sigma_px)[0])
        if n <= max_basis or spacing_px > radius_px:
            return spacing_px, sigma_px, n
        spacing_px *= 1.25
        sigma_px = spacing_px * ratio


def _render_volume(
    centres_px,
    sigmas_px,
    amps,
    scale2d_to_3d,
    nz,
    ny,
    nx,
    twist_deg=0.0,
    rise_px=0.0,
    csym=1,
    cutoff_sigma=3.5,
):
    """Render the fitted asymmetric unit onto a (nz, ny, nx) voxel grid.

    The screw copies that fall inside the slab are rendered too, which matters
    more than it sounds. The basis lies in one z plane, so without them the
    volume is a single Gaussian bump in z; ``helicon.apply_helical_symmetry``
    then replicates that bump every rise, and when the rise is sub-voxel -- 0.95
    px here -- the replication beats against the pixel lattice. The result is a
    strong axial modulation at the beat period (measured: 20 px, carrying 39% of
    the axial power), visible as banding along the reconstruction.

    Filling the slab with the copies first makes it axially continuous, so the
    voxel symmetrisation has nothing to alias.
    """
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    if not len(centres_px):
        return vol
    c = np.asarray(centres_px, float) * scale2d_to_3d
    s = np.asarray(sigmas_px, float) * scale2d_to_3d
    amps = np.asarray(amps, float)
    rise = abs(float(rise_px)) * scale2d_to_3d
    half = max(1, int(np.ceil(cutoff_sigma * float(s.max()))))
    off = np.arange(-half, half + 1)

    if rise > 1e-6:
        k_max = int(np.ceil((nz / 2.0 + cutoff_sigma * float(s.max())) / rise)) + 1
        ks = np.arange(-k_max, k_max + 1)
    else:
        ks = np.array([0])
    js = np.arange(max(int(csym), 1))

    for k in ks:
        for j in js:
            ang = np.deg2rad(float(twist_deg)) * k + 2 * np.pi * j / max(int(csym), 1)
            ca, sa = np.cos(ang), np.sin(ang)
            for (cx, cy, cz), sg, a in zip(c, s, amps):
                if a <= 0:
                    continue
                rx = cx * ca - cy * sa
                ry = cx * sa + cy * ca
                rz = cz + rise * k
                zi = int(round(nz // 2 + rz))
                if zi + half < 0 or zi - half > nz - 1:
                    continue
                xi = int(round(nx // 2 + rx))
                yi = int(round(ny // 2 + ry))
                xs = xi + off
                ys = yi + off
                zs = zi + off
                okx = (xs >= 0) & (xs < nx)
                oky = (ys >= 0) & (ys < ny)
                okz = (zs >= 0) & (zs < nz)
                if not (okx.any() and oky.any() and okz.any()):
                    continue
                vx = np.exp(-0.5 * ((xs - (nx // 2 + rx)) / sg) ** 2)
                vy = np.exp(-0.5 * ((ys - (ny // 2 + ry)) / sg) ** 2)
                vz = np.exp(-0.5 * ((zs - (nz // 2 + rz)) / sg) ** 2)
                blob = (
                    a * vz[okz, None, None] * vy[None, oky, None] * vx[None, None, okx]
                )
                vol[np.ix_(zs[okz], ys[oky], xs[okx])] += blob.astype(np.float32)
    return vol


def _basis_setup(
    ny,
    nx,
    target_apix2d,
    algorithm,
    reconstruct_diameter_2d_pixel,
    scale2d_to_3d,
    rise_pixel,
):
    """Basis, spacing, sigmas and repeat count for one fit.

    Shared by the single-image and joint fits so that the two cannot drift
    apart: a joint fit whose basis differed from the single-image one would not
    be comparable with it, and the comparison is the whole point.
    """
    apix2d = float(target_apix2d) if target_apix2d else 5.0
    spacing = max(
        1.0, float(algorithm.get("spacing_angstrom", DEFAULT_SPACING_ANGSTROM)) / apix2d
    )
    sigma = max(
        0.5, float(algorithm.get("sigma_angstrom", DEFAULT_SIGMA_ANGSTROM)) / apix2d
    )
    radius = support_radius(ny, reconstruct_diameter_2d_pixel)
    spacing, sigma, _ = fit_spacing_to_budget(radius, spacing, sigma)
    render_sigma = max(
        sigma,
        float(algorithm.get("render_sigma_angstrom", DEFAULT_RENDER_SIGMA_ANGSTROM))
        / apix2d,
    )
    sigma_z = max(
        sigma,
        float(algorithm.get("sigma_z_angstrom", DEFAULT_SIGMA_Z_ANGSTROM)) / apix2d,
    )
    scale = float(scale2d_to_3d) or 1.0
    rise = (abs(float(rise_pixel)) / scale) or 1.0
    return dict(
        grid=full_field_grid(radius, spacing_px=spacing, sigma_px=sigma),
        rise=rise,
        n_repeats=int(nx / 2 / rise) + 2,
        sigma=sigma,
        sigma_z=sigma_z,
        render_sigma=render_sigma,
        scale=scale,
    )


def _penalties(algorithm, M, u):
    """Elasticnet penalties, honouring the tab's alpha / l1_ratio controls."""
    rel_l1, rel_l2 = DEFAULT_L1, DEFAULT_L2
    alpha = algorithm.get("alpha", None)
    if alpha is not None and float(alpha) >= 0:
        ratio = float(algorithm.get("l1_ratio", 0.5))
        rel_l1 = float(alpha) * ratio
        rel_l2 = float(alpha) * (1.0 - ratio)
    rel_l1 = float(algorithm.get("l1", rel_l1))
    rel_l2 = float(algorithm.get("l2", rel_l2))
    return rel_l1 * np.abs(u).max(), rel_l2 * np.trace(M) / max(len(M), 1)


def gauss_analytic_reconstruct(
    projection_image,
    scale2d_to_3d,
    twist_degree,
    rise_pixel,
    csym=1,
    tilt_degree=0.0,
    psi_degree=0.0,
    dy_pixel=0.0,
    thresh_fraction=-1,
    positive_constraint=-1,
    reconstruct_diameter_3d_inner_pixel=0,
    reconstruct_diameter_2d_pixel=-1,
    reconstruct_diameter_3d_pixel=-1,
    reconstruct_length_2d_pixel=-1,
    reconstruct_length_3d_pixel=-1,
    sym_oversample=-1,
    interpolation="linear",
    fsc_test=0,
    score_metric="cosine",
    target_apix2d=5.0,
    verbose=0,
    algorithm=None,
    cpu=1,
    **_ignored,
):
    """Solve one (twist, rise) on a Gaussian basis, with the standard signature.

    A fixed basis of Gaussians spans the field, is replicated by the screw
    symmetry and projected analytically, and the coefficients are fitted as a
    non-negative elasticnet on the pixel residual. Convex, so the answer does
    not depend on initialisation.

    The support deliberately covers the whole image, not the filament: capping
    it at the filament radius biases the recovered twist upward by 0.6-0.7
    degrees (see the module docstring).
    """
    algorithm = algorithm or {}
    image = np.asarray(projection_image, dtype=float)
    ny, nx = image.shape
    setup = _basis_setup(
        ny,
        nx,
        target_apix2d,
        algorithm,
        reconstruct_diameter_2d_pixel,
        scale2d_to_3d,
        rise_pixel,
    )
    grid = setup["grid"]
    rise = setup["rise"]
    n_repeats = setup["n_repeats"]
    sigma_z = setup["sigma_z"]
    render_sigma = setup["render_sigma"]
    scale = setup["scale"]
    A = design_matrix(
        grid,
        float(twist_degree),
        rise,
        int(max(csym, 1)),
        n_repeats,
        ny,
        nx,
        sigma_z=sigma_z,
    )
    b = image.ravel()
    M = A.T @ A
    u = A.T @ b
    # Honour the tab's alpha / l1_ratio controls when the user sets them, so
    # those sliders are not silently inert for this solver; otherwise use the
    # validated defaults. Penalty strength barely moves the twist answer here
    # (unchanged over a hundredfold range), so this is about not surprising the
    # user rather than about accuracy.
    lam1, lam2 = _penalties(algorithm, M, u)
    coef = nn_elasticnet(M, u, lam1, lam2)

    pred = A @ coef
    score = correlation_score(pred, b)

    nz3 = int(reconstruct_length_3d_pixel) if reconstruct_length_3d_pixel > 0 else 4
    d3 = int(reconstruct_diameter_3d_pixel) if reconstruct_diameter_3d_pixel > 0 else ny
    rec3d = _render_volume(
        grid[0],
        np.full_like(grid[1], render_sigma),
        coef,
        scale,
        nz3,
        d3,
        d3,
        twist_deg=float(twist_degree),
        rise_px=rise,
        csym=int(max(csym, 1)),
    )
    return (rec3d, None, None), score


def gauss_joint_reconstruct(
    images,
    phis,
    scale2d_to_3d,
    twist_degree,
    rise_pixel,
    csym=1,
    reconstruct_diameter_2d_pixel=-1,
    reconstruct_diameter_3d_pixel=-1,
    reconstruct_length_3d_pixel=-1,
    target_apix2d=5.0,
    algorithm=None,
    verbose=0,
    **_ignored,
):
    """One Gaussian basis fitted to several images, each at its own azimuth.

    The joint system costs almost nothing extra. Each image contributes its own
    design matrix ``A_i``, but the normal equations only ever need
    ``M = sum_i A_i^T A_i`` and ``u = sum_i A_i^T b_i``, both ``(G, G)`` and
    ``(G,)`` whatever the number of images, so the stacked matrix is never
    formed. Adding images costs one design matrix each and leaves the solve
    unchanged -- unlike the voxel joint solver, where the equation count grows
    with the image count and has to be rationed.

    ``phis`` are azimuths in the convention ``denovo3d_align.align_to_model``
    reports. The sign is flipped here rather than at every call site, so this
    and the voxel joint solver take the same numbers and mean the same thing.

    Images are scaled to unit variance before stacking, so that one image with
    a large dynamic range cannot dominate the shared fit -- the same hazard
    ``denovo3d_joint`` guards against when combining score curves.

    Returns ``(volume, info)`` with ``info`` carrying the joint ``score``, the
    ``per_image`` scores, and the fitted ``coefficients``.
    """
    algorithm = algorithm or {}
    images = [np.asarray(im, dtype=float) for im in images]
    if not images:
        raise ValueError("no images")
    if len(phis) != len(images):
        raise ValueError("one azimuth per image is required")

    ny, nx = images[0].shape
    setup = _basis_setup(
        ny,
        nx,
        target_apix2d,
        algorithm,
        reconstruct_diameter_2d_pixel,
        scale2d_to_3d,
        rise_pixel,
    )
    grid = setup["grid"]

    mats, targets = [], []
    M = u = None
    for image, phi in zip(images, phis):
        if image.shape != (ny, nx):
            raise ValueError("all images must have the same shape")
        sd = float(image.std())
        b = (image / sd if sd > 0 else image).ravel()
        A = design_matrix(
            grid,
            float(twist_degree),
            setup["rise"],
            int(max(csym, 1)),
            setup["n_repeats"],
            ny,
            nx,
            sigma_z=setup["sigma_z"],
            phi_degree=-float(phi),
        )
        mats.append(A)
        targets.append(b)
        Mi, ui = A.T @ A, A.T @ b
        M = Mi if M is None else M + Mi
        u = ui if u is None else u + ui

    lam1, lam2 = _penalties(algorithm, M, u)
    coef = nn_elasticnet(M, u, lam1, lam2)

    per_image = [correlation_score(A @ coef, b) for A, b in zip(mats, targets)]
    score = correlation_score(
        np.concatenate([A @ coef for A in mats]), np.concatenate(targets)
    )

    nz3 = int(reconstruct_length_3d_pixel) if reconstruct_length_3d_pixel > 0 else 4
    d3 = int(reconstruct_diameter_3d_pixel) if reconstruct_diameter_3d_pixel > 0 else ny
    rec3d = _render_volume(
        grid[0],
        np.full_like(grid[1], setup["render_sigma"]),
        coef,
        setup["scale"],
        nz3,
        d3,
        d3,
        twist_deg=float(twist_degree),
        rise_px=setup["rise"],
        csym=int(max(csym, 1)),
    )
    return rec3d, dict(
        score=float(score),
        per_image=[float(v) for v in per_image],
        coefficients=coef,
        positive=True,
    )
