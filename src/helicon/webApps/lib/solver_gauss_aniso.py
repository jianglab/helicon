"""Free anisotropic Gaussians in the asymmetric unit, scored analytically.

Each Gaussian carries amplitude, centre (x, y, z) and a full 3x3 covariance,
all free. The screw symmetry conjugates the covariance (Sigma -> R Sigma R^T)
and rotates/translates the centre; projection along x takes the (y, z)
sub-block and scales the amplitude by sqrt(2*pi*det(Sigma)/det(Sigma_yz)).
det(Sigma) is invariant under the rotation, so it is computed once.

The objective is the analytic cosine against the stage-1 target mixture, so
gradients are exact and nothing is rasterised -- unlike the older gauss
solvers, which descended through a render against noisy pixels.

This is strictly more expressive than the fixed isotropic grid in
``solver_gauss_analytic``, and on synthetic data that pays: with the truth
inside the model class it recovers a known twist exactly (1.20, error 0.00),
which no convex variant here manages.

Status: it does NOT work on real class averages, and is not registered in the
pipeline. Measured on the eight good EMPIAR-10940 classes (true twist 1.2),
six restarts per twist:

    per image (0-idx)    2     3     4    10    11    15    16    19   joint
    elasticnet        1.20  1.20  1.00  1.20  1.25  1.20  1.35  1.20    1.20
    this solver       1.25  1.25  1.20  2.00  2.00  1.95  2.00  1.20    2.00

Half the images run to the top of the scan. The reason is in the numbers
rather than the optimiser: the score at the true twist is 0.98 to 1.00 of each
curve's own maximum, i.e. the model fits nearly as well at every twist, so the
curve is almost flat (joint contrast 0.41 against elasticnet's 1.16) and the
argmax turns on a fraction of a percent of bias. Anything that makes the model
fit better -- more components, anisotropy, more restarts -- raises the fit
everywhere and does not sharpen the peak.

Two findings that changed the picture substantially, both worth applying to
any further attempt:

* **Do not use the image's own axial intensity profile as an envelope.** The
  model needs *some* truncation, because the analytic objective integrates over
  infinite space while the class average is a finite segment. But that profile
  oscillates with the crossover pattern, which is the twist signal itself, so
  imposing it explains the image with a statistic taken from the same image.
  It is a confound, not a modelling choice: with a constrained isotropic grid
  it appeared to *help* (1.45 -> 1.10) while with a free anisotropic model it
  pinned all three test images to the top of the scan. A plain boxcar
  truncates without injecting anything and is the right choice.

* **Model the sub-symmetry (n*twist, n*rise) rather than every subunit.** The
  rise here is 0.96 px, far below both the pixel size and sigma, so consecutive
  copies overlap about fourfold and the projection is a smooth continuum that
  barely moves with twist. Taking every n-th subunit separates the copies and,
  more importantly, makes the azimuthal increment per copy n*twist, so a given
  relative twist error produces an n-fold larger angular error. At n=8 the
  joint peak over the eight good classes moves from 1.95 to 1.15 (truth 1.2)
  and the joint contrast rises to 1.63, above elasticnet's 1.16 -- the first
  measure on which this line beats the reference solver -- while costing about
  nine times less per twist.

  The model then only constrains twist modulo 360/n, so t and t + 360/n are
  aliased. At n=8 that is 45 degrees, harmless against a 0.6-2.0 search range,
  but a wide-open twist search would need n bounded by 360 / (range).

  **That 1.15 is not robust, and the reason matters.** Optimising harder makes
  it worse: eight restarts instead of three moves the joint peak to 1.00, and
  doubling the iterations to 500 moves it to 0.60, with per-image answers
  scattering from 0.6 to 2.0. An objective whose answer degrades as its
  optimiser improves does not have its optimum at the true twist; the n=8
  result rests partly on under-convergence. The tell is that the score at the
  true twist stays within about one percent of each curve's maximum
  (median s@1.2/max 0.986 to 0.994) in every configuration tried, so the
  objective is nearly flat near the top and which point wins is decided by
  optimiser noise.

  Joint contrast is therefore not evidence of correctness here. It sits at
  1.63 to 1.67 across configurations whose joint peaks are 1.15, 1.00 and
  0.60: it measures how sharp the combined curve looks, not whether it peaks
  in the right place.

Three things already tried, so they need not be repeated:

* Restarts. Convergence looked solved on 0-idx 2, 3 and 4 (all settling on 1.2
  by five restarts), but those are exactly the three images this model handles;
  on the full set more restarts do not move the four that run to 2.0.
* Deterministic initialisation from the convex isotropic solve. This removes
  the seed dependence but injects that solve's own high-twist bias, and the
  joint peak moves from 1.20 to 1.95 -- worse than random restarts.
* A fixed bank of anisotropic basis functions, which keeps the amplitudes
  linear and the problem convex. With 273 basis functions the fit improves
  (0.916 -> 0.969) while discrimination collapses (contrast 1.18 -> 0.77):
  discrimination tracks parsimony, and parsimony with free shapes is
  inherently non-convex. That tension is the actual obstacle.
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
