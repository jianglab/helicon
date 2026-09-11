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

    ``envelope`` is an optional callable ``z -> weight`` applied per copy. It
    matters more than it looks: a class average shows a finite, tapered,
    masked segment while a strictly symmetric model extends uniformly, and
    without an envelope that mismatch is absorbed by whichever twist happens to
    compensate for it. Supplying the image's own axial profile raised the fit
    from 0.88 to 0.96 and moved the recovered twist from 1.45 to 1.10 on
    EMPIAR-10940 class 2 (ground truth 1.2).

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
    for d in range(-dk_max, dk_max + 1):
        i = idx[max(0, -d) : C - max(0, d)]
        if len(i) == 0:
            continue
        jj = i + d
        M += iso_overlap(
            amp[:, None, i],
            mu[:, None, i, :],
            sig[:, None, i],
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
