"""Free anisotropic Gaussians in the asymmetric unit, scored analytically.

Each Gaussian carries amplitude, centre (x, y, z) and a full 3x3 covariance,
all free. The screw symmetry conjugates the covariance (Sigma -> R Sigma R^T)
and rotates/translates the centre; projection along x takes the (y, z)
sub-block and scales the amplitude by sqrt(2*pi*det(Sigma)/det(Sigma_yz)).
det(Sigma) is invariant under the rotation, so it is computed once.

Nothing is rasterised and gradients are exact, unlike the older gauss solvers
which descended through a render against noisy pixels. Restricted to isotropic
covariances this reproduces ``solver_gauss_analytic`` to machine precision,
which the tests check.

ALL REAL-DATA CONCLUSIONS PREVIOUSLY RECORDED HERE ARE VOID
------------------------------------------------------------
Every real-data measurement on this module was made while the basis support was
capped at ``diameter/2 * 0.9``, inside the filament. That defect alone biases
the recovered twist upward by 0.6-0.7 degrees and is now understood and fixed
(see ``solver_gauss_analytic``, where the same pipeline with a full-field
support reaches parity with the app's voxel elasticnet). The following claims
were all drawn from that broken configuration and must not be cited or acted
on without re-measurement:

* that this solver "does not work" on real class averages, with a joint peak of
  2.00 and half the images running to the top of the scan;
* the diagnosis that the model "fits nearly as well at every twist", that the
  score curve is intrinsically flat, and that nothing which improves the fit can
  sharpen the peak;
* that more restarts, or deterministic initialisation from the convex solve,
  make matters worse and "need not be repeated";
* that a fixed bank of anisotropic basis functions improves the fit while
  collapsing discrimination, and hence that discrimination requires parsimony
  and parsimony forces non-convexity;
* the specific numbers attached to the sub-symmetry trick, and to the axial
  envelope, on real data.

The invariant underlying most of them -- the score at the true twist sitting
just below the best twist -- was not a property of Gaussian models. It was the
signature of a model that could not represent the true structure at all.

WHAT SURVIVES, BECAUSE IT WAS NOT MEASURED THAT WAY
-----------------------------------------------------
* On synthetic data, with the truth inside the model class, this recovers a
  known twist exactly. That was never support-limited, since the synthetic
  support covered the synthetic structure.
* The axial truncation must follow the fitted covariance *and* the axial spread
  of the asymmetric unit, and must be capped. Getting this wrong lets the
  optimiser inflate the cosine above 1 by elongating Gaussians out of the
  truncation, and lets an allocation size be chosen by gradient descent, which
  once exhausted the machine's memory. See :func:`axial_dk`.
* An axial envelope must be twist-independent. The image's own axial profile
  oscillates with the crossover pattern -- that is the twist signal -- so using
  it explains the image with a statistic drawn from the image. The mechanism
  stands even though the numbers once quoted for it do not.
* Modelling the sub-symmetry (n*twist, n*rise) constrains twist only modulo
  360/n, so the search range must be narrower than that. This is arithmetic and
  holds regardless. Its cost saving -- roughly n-squared, since both the copy
  count and the truncation width shrink -- is also real.
* A cosine objective is scale-invariant, so L1/L2 penalties against it are
  degenerate. Use least squares when penalising.

psi/dy REFINEMENT: RIGHT ANSWER, WRONG REASON
-----------------------------------------------
This module once recorded that refining a global psi/dy "changes nothing, the
optimum sits at zero" because the tab's auto-transform had already aligned the
images, and that it need not be retried. The advice happens to hold, but every
part of the reasoning was wrong, and the evidence for it was worthless: it was
measured with the broken support, which suppresses exactly the outer radii where
an in-plane rotation is most visible, and the test rotated the image with an
interpolating resample that pushed the filament toward the frame edge, so it was
partly measuring clipping.

Re-measured properly, the machinery is sound and the conclusion is different in
kind. The convention recovers applied psi and dy exactly on synthetic data, six
cases out of six. On real images the optimum is emphatically NOT at zero: over
twelve good classes the refined psi scatters from -3 to +3 degrees with a mean
near -0.25, i.e. no systematic tilt at all. Refining then costs accuracy rather
than nothing:

    psi = dy = 0    joint 1.25   6/12 within one step   med s@1.2/max 0.9987
    psi/dy refined  joint 1.25   4/12 within one step   med s@1.2/max 0.9970

and the refined per-image peaks move upward, five of twelve landing at 1.35.

That pattern -- a scattered optimum with no consistent sign, and a fit that
improves while the answer degrades -- is the signature of extra freedom
absorbing signal, not of a misalignment being corrected. It is consistent with
the tab's auto-transform genuinely leaving only a small residual (measured
elsewhere at about 0.14 degrees mean), which is far below the several degrees
this refinement chases. So: do not refine psi/dy here, because it fits noise,
not because the optimum sits at zero.

This module is not registered in the pipeline's algorithm dispatch.
"""

import numpy as np
import torch
import torch.utils.checkpoint

TWO_PI = 2.0 * np.pi


def _chol(raw):
    """Lower-triangular L with positive diagonal, so Sigma = L L^T is PD."""
    n = raw.shape[-1]
    L = torch.tril(raw)
    idx = torch.arange(n, device=raw.device)
    diag = torch.nn.functional.softplus(raw[..., idx, idx]) + 1e-3
    L = L.clone()
    L[..., idx, idx] = diag
    return L


def expand_project(amp, centre, raw_L, twist_deg, rise, csym, n_repeats, envelope=None):
    """Replicate by the screw and project along x. Returns 2D mixture tensors."""
    G = amp.shape[0]
    L = _chol(raw_L)  # (G,3,3)
    Sigma = L @ L.transpose(-1, -2)  # (G,3,3)
    detS = torch.det(Sigma)  # rotation-invariant

    ks = torch.arange(-n_repeats, n_repeats + 1, dtype=amp.dtype, device=amp.device)
    js = torch.arange(csym, dtype=amp.dtype, device=amp.device)
    k = ks.repeat_interleave(csym)
    j = js.repeat(len(ks))
    ang = torch.deg2rad(twist_deg) * k + 2 * np.pi * j / max(csym, 1)
    C = ang.shape[0]
    ca, sa = torch.cos(ang), torch.sin(ang)
    zero, one = torch.zeros_like(ca), torch.ones_like(ca)
    R = torch.stack(
        [
            torch.stack([ca, -sa, zero], -1),
            torch.stack([sa, ca, zero], -1),
            torch.stack([zero, zero, one], -1),
        ],
        -2,
    )  # (C,3,3)

    # centres: rotate then shift along the axis
    c = torch.einsum("cij,gj->gci", R, centre)  # (G,C,3)
    c = c + torch.stack([torch.zeros_like(k), torch.zeros_like(k), rise * k], -1)

    Srot = torch.einsum("cij,gjk,clk->gcil", R, Sigma, R)  # (G,C,3,3)
    cov2 = Srot[..., 1:, 1:]  # (y,z) block
    det2 = cov2[..., 0, 0] * cov2[..., 1, 1] - cov2[..., 0, 1] * cov2[..., 1, 0]
    amp2 = amp[:, None] * torch.sqrt(TWO_PI * detS[:, None] / det2.clamp_min(1e-12))
    mu2 = c[..., 1:]  # (y,z)
    if envelope is not None:
        amp2 = amp2 * envelope(mu2[..., 1])
    return mu2, amp2, cov2


def _overlap(a1, m1, c1, a2, m2, c2):
    """<f1,f2> for general 2D Gaussians, broadcasting over leading axes.

    Uses <f1,f2> = a1 a2 * 2*pi*sqrt(detS1 detS2/det(S1+S2))
                   * exp(-0.5 d^T (S1+S2)^-1 d),
    which is the det(P1+P2) prefactor rewritten to avoid explicit inverses.
    """
    S = c1 + c2
    dS = S[..., 0, 0] * S[..., 1, 1] - S[..., 0, 1] * S[..., 1, 0]
    d1 = c1[..., 0, 0] * c1[..., 1, 1] - c1[..., 0, 1] * c1[..., 1, 0]
    d2 = c2[..., 0, 0] * c2[..., 1, 1] - c2[..., 0, 1] * c2[..., 1, 0]
    d = m1 - m2
    # (S)^-1 applied to d, 2x2 closed form
    q = (
        S[..., 1, 1] * d[..., 0] ** 2
        - (S[..., 0, 1] + S[..., 1, 0]) * d[..., 0] * d[..., 1]
        + S[..., 0, 0] * d[..., 1] ** 2
    ) / dS.clamp_min(1e-12)
    pref = TWO_PI * torch.sqrt((d1 * d2 / dS.clamp_min(1e-12)).clamp_min(1e-30))
    return a1 * a2 * pref * torch.exp(-0.5 * q)


def cross(model, target):
    mu2, amp2, cov2 = model
    ta, tm, tc = target
    return _overlap(
        amp2[:, :, None],
        mu2[:, :, None, :],
        cov2[:, :, None, :, :],
        ta[None, None, :],
        tm[None, None, :, :],
        tc[None, None, :, :, :],
    ).sum()


_PAIR_CACHE = {}


def _pairs(C, dk_max, device):
    """Index pairs (i, i+d) for |d| <= dk_max, built once and reused.

    Looping over offsets in Python puts one tensor op per offset into the
    autograd graph, which dominated the runtime. Flattening the offsets into a
    single gather turns the whole self-energy into one batched expression.
    """
    key = (C, dk_max, str(device))
    if key not in _PAIR_CACHE:
        i, j = [], []
        for d in range(-dk_max, dk_max + 1):
            lo, hi = max(0, -d), C - max(0, d)
            if hi <= lo:
                continue
            idx = torch.arange(lo, hi, device=device)
            i.append(idx)
            j.append(idx + d)
        _PAIR_CACHE[key] = (torch.cat(i), torch.cat(j))
    return _PAIR_CACHE[key]


# Peak-memory budget for one intermediate of the self-energy sum, in bytes.
# The sum broadcasts to (G, G, P, 2, 2) and autograd retains every intermediate
# for the backward pass, so the true footprint is several times this. Exceeding
# it is not a slow path but an out-of-memory kill: an unbounded version of this
# function once took the user's machine down.
SELF_ENERGY_CHUNK_BYTES = 32 << 20


def self_energy_bytes(n_gauss, n_pairs, itemsize=8):
    """Bytes for the largest single intermediate of an unchunked self-energy."""
    return n_gauss * n_gauss * n_pairs * 4 * itemsize


def _row_chunk(n_gauss, n_pairs, itemsize):
    per_row = max(1, n_gauss * n_pairs * 4 * itemsize)
    return max(1, min(n_gauss, SELF_ENERGY_CHUNK_BYTES // per_row))


def self_energy(model, dk_max):
    """``<f, f>``, truncated in copy separation along the axis.

    Evaluated in row chunks under gradient checkpointing. Chunking the forward
    pass alone would not help: autograd would still retain every chunk's
    intermediates until the backward pass. Checkpointing recomputes each chunk
    instead, which bounds peak memory to roughly one chunk at the cost of one
    extra forward evaluation.
    """
    mu2, amp2, cov2 = model
    G, C = amp2.shape
    i, j = _pairs(C, dk_max, amp2.device)
    P = i.shape[0]
    chunk = _row_chunk(G, P, amp2.element_size())

    def _block(lo, hi):
        return _overlap(
            amp2[lo:hi, None, i],
            mu2[lo:hi, None, i, :],
            cov2[lo:hi, None, i, :, :],
            amp2[None, :, j],
            mu2[None, :, j, :],
            cov2[None, :, j, :, :],
        ).sum()

    total = amp2.new_zeros(())
    for lo in range(0, G, chunk):
        hi = min(G, lo + chunk)
        if torch.is_grad_enabled() and amp2.requires_grad:
            total = total + torch.utils.checkpoint.checkpoint(
                _block, lo, hi, use_reentrant=False
            )
        else:
            total = total + _block(lo, hi)
    return total


# Hard ceiling on the axial truncation width, independent of the fitted model.
# Five sigma of a *grown* Gaussian is unbounded, and the number of copy pairs
# rises as C*(2*dk+1) up to C^2, so leaving this to the optimiser makes memory a
# function of how wide the fit chooses to become.
MAX_AXIAL_DK = 48


def axial_dk(model, rise, n_sigma=5.0, max_dk=MAX_AXIAL_DK):
    """How many copy separations the self-overlap must span, from the CURRENT model.

    Two things set the width, and leaving either out silently breaks the score.

    The width must follow the fitted covariance rather than the initial sigma:
    with a fixed dk the optimiser can grow a Gaussian along the axis until its
    own far-separated self-overlaps fall outside the truncation, so ``<f,f>``
    loses mass the cross term still counts and the cosine climbs above 1. Adam
    finds that reward reliably (1.10 to 1.18 observed).

    It must also cover the axial SPREAD of the asymmetric unit. Copy ``k`` of
    Gaussian ``g`` and copy ``k'`` of Gaussian ``h`` are separated along the
    axis by ``(z_g - z_h) + (k - k') * rise``, so a copy-index difference only
    tracks axial distance while every Gaussian sits at a similar z. Once the
    fit spreads them out, pairs with large ``|dk|`` are still axially close,
    and truncating them throws away real self-energy -- measured at 44% of the
    true value, which inflated a 0.816 cosine to 1.228 and made a
    capacity-regularised model look like it had solved the problem.

    Both are then capped: tying an allocation size to a quantity the optimiser
    controls is how this once exhausted the machine's memory.
    """
    mu2, amp2, cov2 = model
    sig_z = float(cov2[..., 1, 1].max().detach().clamp_min(0.0).sqrt())
    z = mu2[..., 1].detach()
    # spread of the asymmetric unit along the axis, measured at one copy index
    spread = float((z[:, 0].max() - z[:, 0].min()).abs()) if z.shape[0] > 1 else 0.0
    C = amp2.shape[1]
    want = int(np.ceil((n_sigma * sig_z + spread) / max(rise, 1e-6)) + 1)
    return int(min(C - 1, max_dk, want))


def self_energy_fraction(model, dk, full_dk=None):
    """Share of the true self-energy the truncation at ``dk`` retains.

    A diagnostic for the failure above: any value well below 1 means the score
    is inflated and must not be trusted.
    """
    C = model[1].shape[1]
    with torch.no_grad():
        part = float(self_energy(model, dk))
        full = float(self_energy(model, min(C - 1, full_dk or C - 1)))
    return part / full if full > 0 else 0.0


def score(model, target, tt, rise):
    num = cross(model, target)
    den = self_energy(model, axial_dk(model, rise))
    return num / torch.sqrt(den.clamp_min(1e-30) * tt)
