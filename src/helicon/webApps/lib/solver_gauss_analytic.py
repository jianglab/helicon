"""Grid-free helical solver: a Gaussian mixture model scored analytically.

Two stages. Stage 1 fits the class average with a 2D Gaussian mixture, which
acts as a denoiser and replaces the pixel data as the target. Stage 2 models
the structure as Gaussians in one asymmetric unit, replicates them with the
screw symmetry, and projects -- all in closed form, because projecting a
Gaussian yields a Gaussian. Nothing is ever rasterised.

Why this is worth having over the earlier iterative Gaussian solvers: the
projected model is **linear in the amplitudes**, so for fixed centres and
sigmas the best amplitudes and the best achievable score come from one linear
solve,

    score = max_a  <f_a, t> / (||f_a|| ||t||) = sqrt(u^T M^-1 u) / ||t||

with ``u_g = <B_g, t>`` and ``M_gh = <B_g, B_h>``. There are no restarts, no
learning rate and no local minima -- which is what the gradient-descent
versions kept losing to.

Geometry follows ``solver_gaussian``: centres are (x, y, z) with z the helical
axis, and the projection is along x onto the (y, z) plane, so the image is
(transverse, axial). Everything here works in pixels.

Status: this approach works, but only with the reconstruction support set
correctly, which took a long investigation to find. Over the eight good
EMPIAR-10940 classes (true twist 1.2), scored on the joint curve the tab uses:

    method                              joint  <=1 step  <=0.15  med s@1.2/max  s/img-twist
    Gaussian full-field elasticnet       1.20      9/32   29/32         0.9961         2.46
    app voxel elasticnet (reference)     1.30      8/32   28/32         0.9987         0.62

Measured over all 32 good class averages (the user's definitive list, 1-indexed
3 4 5 11 12 16 17 20 22 27 29 31 33 34 37 42 43 46 47 49 51 55 56 57 59 62 63
71 74 76 79 80), scanning 0.6 to 2.0 in steps of 0.05.

The joint answer is exact here while the reference lands a grid step high, but
per-image the two are a dead heat and the reference keeps a sharper curve and
runs four times faster. Read this as parity on accuracy, not as beating it --
and note the reference's joint peak moves with the scan range (it gives 1.20
over the wider 0.5-2.5 range), so that particular comparison is not robust.

**The support must cover the whole field, not just the filament.** Earlier
versions here built the basis over ``diameter/2 * 0.9``, which cuts inside a
17 px filament. That single choice, made early and inherited by every later
variant, produced a systematic bias of +0.6 to +0.7 in twist and made the
method look fundamentally broken. Measured on three classes, changing nothing
else:

    radius = 0.9 * diam/2   joint 1.90    med s@1.2/max 0.9186
    radius = diam/2         joint 1.90    med s@1.2/max 0.9641
    radius = image half     joint 1.20    med s@1.2/max 1.0000

The mechanism explains the sign. In projection a basis function at radius r
sweeps a transverse range proportional to r as the symmetry copies rotate, so
to reproduce a filament of a given apparent width the model needs either large
enough radii or more twist per copy. Capping the support removes the first
option and the fit compensates with the second, biasing the answer upward.

This also explains why nothing else ever helped. Regularisation, basis type,
basis resolution, sub-symmetry, adaptive growth, the objective and the target
all operate *within* the support, and none can recover radii excluded before
fitting begins. The invariant that ran through every failed experiment --
score at the true twist sitting 1-9% below the best twist -- was not evidence
that Gaussians cannot discriminate twist; it was evidence the true twist was
never representable. With the full field that statistic is 0.9999.

The axial extent, by contrast, does not matter: a single z plane scores the
same as one spanning 4 px, which makes sense when the rise (0.95 px) is below
the pixel size.

What the working configuration is: full-field Cartesian basis at 2 px spacing
with sigma 1.2 px, one z plane, projected analytically, and solved as
non-negative elasticnet on the pixel residual (least squares, NOT the cosine --
a cosine is scale-invariant, so L1/L2 penalties against it are degenerate and
simply shrink every amplitude toward zero).

Previously recorded status, now superseded:

Status: on synthetic data it recovers a known twist exactly, with a grid that
does not contain the truth. On real data it is **not** competitive with the
elasticnet voxel solver, and is deliberately not registered in the pipeline's
algorithm dispatch. Benchmarked on eight good EMPIAR-10940 class averages
(true twist 1.2 deg), scoring the joint curve the tab itself uses:

    method                 joint peak   contrast   per-image   time/step
    elasticnet (app)             1.20       1.12         6/8     ~1000 ms
    analytic sigma=2.0           1.15       0.66         2/8       363 ms
    analytic sigma=1.25..2.5  0.60-1.35        --       0-3/8   196-1071 ms

So it is roughly 3x faster and lands within one grid step of the truth in its
best configuration, but three things are wrong with it:

* It is not robust. ``sigma`` is a genuine hyperparameter and swings the answer
  from 0.60 to 1.35; elasticnet has no equivalent knob. Letting the score pick
  the scale does not work, because scores are not comparable across sigma --
  a coarser basis has fewer functions and a different attainable ceiling.
* Per-image accuracy is much worse (2/8 against 6/8), and the joint curve is
  markedly less sharp (contrast 0.66 against 1.12). Only the joint combination
  rescues the answer.
* The obvious hoped-for niche does not exist. A Gaussian basis is a smoothness
  prior, so it should degrade gracefully as SNR falls; measured against added
  noise it does the opposite, drifting 1.15 -> 1.50 while elasticnet holds
  near the truth.

Two things that do *not* explain the gap, so they need not be retried: refining
a global psi/dy of the target changes nothing (the optimum sits at zero, the
tab's auto-transform having already aligned the images), and computing the Gram
by rasterising the basis and using BLAS is ~10x slower than the analytic pair
sum as well as less accurate.

The most likely remaining explanation is expressiveness: a cylindrical grid of
isotropic Gaussians in one z-plane is strictly less expressive than the voxel
basis elasticnet solves for, so the fit at the true twist is not the best fit.
A multi-scale basis is the natural next attempt, but the Gram is O(G^2) and
several scales made it impractically slow without a smarter Gram.
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
    "expand_project",
    "gram_matrix",
    "cross_vector",
    "best_score",
    "nn_lasso",
]


def cylindrical_grid(radius_px, sigma_px):
    """Basis centres on a cylinder, sampled at ~``sigma_px`` in radius and arc.

    Sampling arc length rather than using a fixed number of angular steps
    matters: with a constant count the outer shells are sampled far more
    coarsely than sigma, leaving gaps the fit cannot cover, which biases the
    recovered twist.
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
    sum truncates without meaningful error -- ``dk_max`` should cover about
    four sigma. Truncating too tightly under-counts ``<f,f>`` and lets the
    normalised score exceed one, which is a useful tripwire.
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

    With ``lam = 0`` this is the unconstrained optimum in closed form. A
    positive ``lam`` switches to the non-negative lasso, which is the physical
    choice (densities are non-negative) but on real data scored worse here, so
    the default is the plain solve.
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
    0.6-0.7 degrees on EMPIAR-10940; see this module's status notes for the
    measurement and the mechanism.
    """
    n = int(np.ceil(radius_px / spacing_px))
    xs = np.arange(-n, n + 1) * spacing_px
    pts = [[x, y, 0.0] for x in xs for y in xs if x * x + y * y <= radius_px**2]
    centres = np.asarray(pts, float)
    return centres, np.full(len(centres), float(sigma_px))


def design_matrix(grid, twist_deg, rise_px, csym, n_repeats, ny, nx, chunk=48):
    """``(n_pixels, G)`` matrix whose column g is basis fn g with all its copies."""
    centres, sigmas = grid
    mu, amp, sig = expand_project(centres, sigmas, twist_deg, rise_px, csym, n_repeats)
    yy, xx = np.meshgrid(
        np.arange(ny) - ny / 2.0, np.arange(nx) - nx / 2.0, indexing="ij"
    )
    pts = np.stack([yy.ravel(), xx.ravel()], -1)
    G = amp.shape[0]
    A = np.empty((pts.shape[0], G))
    for lo in range(0, G, chunk):
        hi = min(G, lo + chunk)
        d2 = ((pts[None, :, None, :] - mu[lo:hi, None, :, :]) ** 2).sum(-1)
        A[:, lo:hi] = (
            (amp[lo:hi, None, :] * np.exp(-0.5 * d2 / sig[lo:hi, None, :] ** 2))
            .sum(-1)
            .T
        )
    return A


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
