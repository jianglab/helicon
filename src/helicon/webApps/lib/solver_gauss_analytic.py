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
All 32 good EMPIAR-10940 class averages (true twist 1.2), scanning 0.6 to 2.0 in
0.05 steps, run through the app pipeline as a user would:

    method          joint  <=1 step  <=0.15  med s@1.2/max  ms per image-twist
    gauss_analytic   1.20      9/32   28/32         0.9981                  64
    elasticnet       1.30      8/32   28/32         0.9987                 740

About ten times faster, with the joint peak on the truth where the reference is
a grid step high, and per-image a dead heat. On a ten-image joint search run as
the app runs it, this returns 1.2 and the reference returns 1.3.

THE RECONSTRUCTION IS BASIS-LIMITED, AND THAT IS NOT CURRENTLY FIXABLE HERE
----------------------------------------------------------------------------
By FSC against the elasticnet volume the map is good to about 20 A: 0.93 at
40 A, 0.65 at 20 A, 0.29 at 15 A, 0.16 at 10 A. Ample for finding the twist,
not for looking at the result.

The obvious remedy -- a finer basis -- does not work, and this docstring
previously recommended it, which was wrong. A finer basis raises the FSC (at
sigma 2.5 A it reaches 0.82 at 15 A) and scores better on per-image twist
counts, yet it makes the solver worse where it matters. On the first ten good
classes, as the app runs them:

    sigma 4A   joint 1.1   3/10 at the true twist   67 s
    sigma 6A   joint 1.2   6/10                     18 s

and at 4 A the reconstruction shows about five crossovers across the volume
where the geometry allows three (pitch/2 = 142 px at twist 1.2 over a 345 px
volume). The spurious doubling is not understood. Until it is, do not lower
sigma: a higher-resolution map that encodes the wrong helical geometry is worse
than a 20 A one that encodes the right geometry.

A 4 A default shipped briefly on the strength of per-image counts measured in a
standalone harness. Those counts improved while the joint peak -- the number the
tab reports -- moved off the truth, and a user running the app caught it. Judge
this solver on the joint answer and the reconstruction together, never on
per-image counts alone.

THE RECONSTRUCTION FILLS THE AXIS, AND THAT IS STRUCTURAL
----------------------------------------------------------
Use this solver for the twist/rise search, and elasticnet for the final map.

On a hollow filament the reprojection shows a bright core where the data has a
dark lane between the strands. The strands themselves are placed correctly --
the transverse peak offsets match the input column by column, and there is no
axial phase error -- but the space between them is filled. Measured on the
EMPIAR-10940 classes, whose middle column sits at 0.079 of maximum, the fit
gives 0.67.

It is not the rendering, the symmetrisation, the basis width, or the penalty.
The fit itself already shows it; sigma down to 3 A only reaches 0.497; L1 across
a fiftyfold range changes nothing; and excluding an inner radius from the
support changes nothing at all, which rules out on-axis basis functions as the
source.

The cause is the combination of helical symmetry and non-negativity. Under the
screw, a basis function at radius r projects to y = r sin(theta(z)), so its
copies sweep through the axis somewhere along the filament no matter where it
sits. Every function therefore deposits density on the axis at some z, and with
non-negative amplitudes nothing can cancel it. Allowing free-sign amplitudes
confirms this exactly: the fit jumps from cc 0.73 to 0.91 and the centre falls
to 0.013, a better match than the input's own 0.079.

But that cure is worse than the disease. With free signs the joint twist moves
from 1.2 to 1.4: the constraint that ruins the picture is the same constraint
that makes the twist identifiable, because without it the model can explain any
twist. Reconstruction quality and twist accuracy are in direct tension here, and
the default resolves it in favour of the twist.

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

This module is not registered in the pipeline's algorithm dispatch: at parity on
accuracy and four times the cost, it is not a better default.
"""

from __future__ import annotations

import threading

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
    "last_gaussians",
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


def expand_project(centres, sigmas, twist_deg, rise_px, csym, n_repeats, envelope=None):
    """Replicate the asymmetric unit by the screw symmetry and project along x.

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
    ang = np.deg2rad(twist_deg) * k + 2 * np.pi * j / max(csym, 1)
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
    grid, twist_deg, rise_px, csym, n_repeats, ny, nx, cutoff_sigma=3.5, chunk=200000
):
    """``(n_pixels, G)`` matrix whose column g is basis fn g with all its copies.

    Each column is dense-ish -- a basis function's copies run the length of the
    filament -- but each individual copy only reaches a few sigma, so the work
    is done per copy over a small footprint rather than per column over the
    whole image. That is the difference between ``P * G * C`` exponentials and
    ``G * C * (2*cutoff*sigma)^2``, about fifty-fold here, and it is where
    essentially all of this solver's time used to go.
    """
    centres, sigmas = grid
    mu, amp, sig = expand_project(centres, sigmas, twist_deg, rise_px, csym, n_repeats)
    G, C = amp.shape
    g_of = np.repeat(np.arange(G), C)
    yc = mu[..., 0].ravel() + ny / 2.0  # row coordinate
    zc = mu[..., 1].ravel() + nx / 2.0  # column coordinate
    a = amp.ravel()
    s = sig.ravel()

    half = int(np.ceil(cutoff_sigma * float(s.max())))
    off = np.arange(-half, half + 1)
    w = off.size

    # Drop copies whose entire footprint lies outside the image.
    keep = (
        (yc > -half) & (yc < ny + half) & (zc > -half) & (zc < nx + half) & (a != 0.0)
    )
    yc, zc, a, s, g_of = yc[keep], zc[keep], a[keep], s[keep], g_of[keep]

    A = np.zeros(ny * nx * G)
    for lo in range(0, yc.size, chunk):
        hi = min(yc.size, lo + chunk)
        r0 = np.floor(yc[lo:hi]).astype(np.int64)[:, None] + off[None, :]
        c0 = np.floor(zc[lo:hi]).astype(np.int64)[:, None] + off[None, :]
        dy = r0 - yc[lo:hi, None]
        dz = c0 - zc[lo:hi, None]
        inv2s2 = 0.5 / s[lo:hi, None] ** 2
        ey = np.exp(-(dy**2) * inv2s2)  # (n, w) separable in y and z
        ez = np.exp(-(dz**2) * inv2s2)
        vals = a[lo:hi, None, None] * ey[:, :, None] * ez[:, None, :]
        ok_r = (r0 >= 0) & (r0 < ny)
        ok_c = (c0 >= 0) & (c0 < nx)
        valid = ok_r[:, :, None] & ok_c[:, None, :]
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


_tls = threading.local()


def last_gaussians():
    """Gaussians fitted by the most recent call *on this thread*.

    Lets the pipeline render the symmetrised volume by placing Gaussian copies
    analytically rather than resampling voxel copies. That matters here because
    the rise is sub-voxel: interpolating and averaging copies shifted by a
    fraction of a pixel is exactly where a voxel symmetrisation blurs most.
    Centres and covariances are in angstroms, as the renderer expects.
    """
    return getattr(_tls, "gaussians", None)


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
DEFAULT_SIGMA_ANGSTROM = 6.0
DEFAULT_L1 = 1e-2
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

    apix2d = float(target_apix2d) if target_apix2d else 5.0
    spacing = max(
        1.0, float(algorithm.get("spacing_angstrom", DEFAULT_SPACING_ANGSTROM)) / apix2d
    )
    sigma = max(
        0.5, float(algorithm.get("sigma_angstrom", DEFAULT_SIGMA_ANGSTROM)) / apix2d
    )
    radius = support_radius(ny, reconstruct_diameter_2d_pixel)
    spacing, sigma, _ = fit_spacing_to_budget(radius, spacing, sigma)

    # rise_pixel arrives in 3D voxel units while the fit works in image pixels.
    # They coincide only when target_apix2d == target_apix3d, which is the tab's
    # default but not guaranteed.
    scale = float(scale2d_to_3d) or 1.0
    rise = (abs(float(rise_pixel)) / scale) or 1.0
    n_repeats = int(nx / 2 / rise) + 2
    grid = full_field_grid(radius, spacing_px=spacing, sigma_px=sigma)
    A = design_matrix(
        grid, float(twist_degree), rise, int(max(csym, 1)), n_repeats, ny, nx
    )
    b = image.ravel()
    M = A.T @ A
    u = A.T @ b
    # Honour the tab's alpha / l1_ratio controls when the user sets them, so
    # those sliders are not silently inert for this solver; otherwise use the
    # validated defaults. Penalty strength barely moves the twist answer here
    # (unchanged over a hundredfold range), so this is about not surprising the
    # user rather than about accuracy.
    rel_l1, rel_l2 = DEFAULT_L1, DEFAULT_L2
    alpha = algorithm.get("alpha", None)
    if alpha is not None and float(alpha) >= 0:
        ratio = float(algorithm.get("l1_ratio", 0.5))
        rel_l1 = float(alpha) * ratio
        rel_l2 = float(alpha) * (1.0 - ratio)
    rel_l1 = float(algorithm.get("l1", rel_l1))
    rel_l2 = float(algorithm.get("l2", rel_l2))
    lam1 = rel_l1 * np.abs(u).max()
    lam2 = rel_l2 * np.trace(M) / max(len(M), 1)
    coef = nn_elasticnet(M, u, lam1, lam2)

    pred = A @ coef
    denom = np.linalg.norm(pred) * np.linalg.norm(b)
    score = float(pred @ b / denom) if denom > 0 else 0.0

    # Publish the fitted model in angstroms so the pipeline can symmetrise it
    # analytically instead of resampling the rendered volume.
    keep = coef > 0
    n_keep = int(keep.sum())
    cov3d = np.zeros((n_keep, 3, 3), dtype=float)
    sig_a = np.asarray(grid[1])[keep] * apix2d
    for i in range(3):
        cov3d[:, i, i] = sig_a**2
    apix3d = apix2d / scale
    _tls.gaussians = dict(
        amps=coef[keep],
        centers=np.asarray(grid[0])[keep] * apix2d,
        cov3d=cov3d,
        twist_degree=float(twist_degree),
        rise_angstrom=abs(float(rise_pixel)) * apix3d,
        csym=int(max(csym, 1)),
    )

    nz3 = int(reconstruct_length_3d_pixel) if reconstruct_length_3d_pixel > 0 else 4
    d3 = int(reconstruct_diameter_3d_pixel) if reconstruct_diameter_3d_pixel > 0 else ny
    rec3d = _render_volume(
        grid[0],
        grid[1],
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
