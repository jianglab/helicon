"""Gaussian-splatting-style helical reconstruction solver.

Conceptually this is 3D Gaussian Splatting (Kerbl et al. 2023) adapted to
helical cryo-EM:

  * a small set of *anisotropic* 3D Gaussians represents one asymmetric unit,
    confined to a single rise along the helical axis;
  * they are duplicated along the whole filament by the assumed twist/rise/csym;
  * the duplicates are splatted into a 2D projection and compared to the input
    class average to give the loss.

Differences from graphics 3DGS, both in our favour except the last:

  * cryo-EM projection is a *line integral*, not alpha-blended compositing, so
    there is no occlusion, no depth sorting and no order dependence -- the
    rasteriser is a plain scatter-add.
  * the helical symmetry ties all copies to one asymmetric unit, a strong prior
    that massively reduces free parameters.
  * but we fit a *single* image with all symmetry copies superimposed, rather
    than many independent views, so the data constrains the model less well.

Key techniques carried over from 3DGS:

  * footprint culling: each Gaussian only touches the pixels within ~3 sigma of
    its projected centre, instead of the whole image (the dominant speed win);
  * anisotropic covariance parameterised as Sigma = R diag(s^2) R^T with R from
    a normalised quaternion, so orientation and elongation are learnable;
  * adaptive density control: periodically clone/split Gaussians in regions with
    large positional gradients and prune negligible ones, growing resolution
    where the residual demands it instead of using a blind dense grid;
  * per-parameter learning rates.
"""

from __future__ import annotations

import logging
import math
import threading

import numpy as np

import helicon

logger = logging.getLogger(__name__)

# Same convention as denovo3d_solver: cache expensive results under the shared
# denovo3D cache directory with a 7-day expiry.
cache_dir = helicon.cache_dir / "denovo3D"

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    F = None
    TORCH_AVAILABLE = False

EPS = 1e-12

# Per-call outputs that do not fit the fixed (rec3d, set1, set2), score return
# contract are published here.  This MUST be thread-local: the denovo3D app runs
# solvers in a ThreadPoolExecutor, so a plain module/function attribute would let
# one worker read another worker's model.
_tls = threading.local()


def last_gaussians():
    """Gaussians fitted by the most recent gauss_splat call *on this thread*."""
    return getattr(_tls, "gaussians", None)


def last_refined_params():
    """Refined tilt/psi/dy from the most recent gauss_splat call on this thread."""
    return getattr(_tls, "refined_params", None)


# Footprint cutoff in sigmas.  3.0 is the 3DGS default; 3.5 keeps the
# truncation error of the culled projection below ~1e-3 relative, which
# matters here because scores are compared against the exactly-evaluated
# lsq solver.
SIGMA_CUTOFF = 3.5


def best_device(prefer=None):
    """Pick the fastest available torch device: cuda > mps > cpu.

    The splat forward/backward pass is embarrassingly parallel (the workload
    3DGS runs on CUDA), so it benefits from a GPU; measured ~3.6x on Apple MPS
    at realistic Gaussian counts, matching CPU to float32 rounding (~4e-7).

    **MPS is only used from the main thread.**  Driving MPS from two or more
    threads at once kills the process silently -- no Python exception, no
    faulthandler traceback.  The denovo3D web app runs solvers in a
    ThreadPoolExecutor with one worker per CPU, so an unguarded MPS default
    made "Search Parameters" die instantly with nothing in the log.  Verified:
    cpu/1 ok, cpu/4 ok, mps/1 ok, mps/2 dead, mps/4 dead.  Pass device="mps"
    explicitly to override (only safe if you know the caller is single
    threaded).
    """
    if prefer:
        return prefer
    if not TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        if threading.current_thread() is threading.main_thread():
            return "mps"
        logger.debug("MPS available but caller is not the main thread; using CPU")
    return "cpu"


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("The gauss_splat solver requires PyTorch.")
    return torch


def quat_to_rotation_matrix(q):
    """Normalised-quaternion (w, x, y, z) -> rotation matrix, (N,4) -> (N,3,3).

    Implemented locally rather than imported from helicon.lib.gauss because
    this module keeps torch as an optional import, while lib.gauss requires it
    at module level.  The two implementations are identical and both are tested.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            torch.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1
            ),
            torch.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1
            ),
            torch.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        dim=1,
    )


def build_cov3d(log_scales, quats):
    """Sigma = R diag(exp(log_scales)^2) R^T  -- the 3DGS parameterisation."""
    R = quat_to_rotation_matrix(quats)
    s = torch.exp(log_scales)
    return R @ torch.diag_embed(s**2) @ R.transpose(1, 2)


def helical_expand(amps, centers, cov3d, twist_degree, rise_angstrom, csym, n_repeats):
    """Duplicate one asymmetric unit along the filament by twist/rise/csym.

    Both the centres and the covariances are rotated about the helical (z) axis:
    a rotated Gaussian has covariance R Sigma R^T.
    """
    device = amps.device
    dtype = amps.dtype
    n_range = torch.arange(-n_repeats, n_repeats + 1, device=device, dtype=dtype)
    i_range = torch.arange(csym, device=device, dtype=dtype)
    n_grid, i_grid = torch.meshgrid(n_range, i_range, indexing="ij")
    n_flat, i_flat = n_grid.flatten(), i_grid.flatten()
    N = n_flat.shape[0]

    angles = torch.deg2rad(twist_degree) * n_flat + 2 * math.pi * i_flat / csym
    ca, sa = torch.cos(angles), torch.sin(angles)
    zero = torch.zeros_like(ca)
    one = torch.ones_like(ca)
    # Rotation about z, (N,3,3)
    Rz = torch.stack(
        [
            torch.stack([ca, -sa, zero], -1),
            torch.stack([sa, ca, zero], -1),
            torch.stack([zero, zero, one], -1),
        ],
        dim=1,
    )

    # centres: rotate then translate along z
    c = torch.einsum("nij,mj->nmi", Rz, centers)
    c = c + torch.stack(
        [torch.zeros_like(n_flat), torch.zeros_like(n_flat), rise_angstrom * n_flat], -1
    ).unsqueeze(1)
    exp_centers = c.reshape(-1, 3)

    # covariances: Sigma -> R Sigma R^T
    cov = torch.einsum("nij,mjk,nlk->nmil", Rz, cov3d, Rz)
    exp_cov = cov.reshape(-1, 3, 3)

    exp_amps = amps.unsqueeze(0).expand(N, -1).reshape(-1)
    return exp_amps, exp_centers, exp_cov


def splat_project(
    amps, centers, cov3d, ny, nz, apix, patch_radius=None, max_radius=64, det3=None
):
    """Project anisotropic 3D Gaussians along x onto the (y, z) plane.

    Footprint-culled ("tile rasterisation" in 3DGS terms): each Gaussian is
    evaluated only on a small patch of pixels around its projected centre and
    scatter-added into the image, rather than broadcast against every pixel.
    Cost is O(M * patch^2) instead of O(M * ny * nz).

    The line integral along x of an unnormalised 3D Gaussian is an unnormalised
    2D Gaussian whose covariance is the (y, z) block of Sigma, with prefactor
    sqrt(2*pi) * sqrt(det(Sigma) / det(Sigma_yz)).  Note that the commonly seen
    shortcut sqrt(2*pi*Sigma_xx) is only correct when Sigma has no cross-terms
    with x; the determinant ratio (the Schur complement) is the general form.
    """
    device, dtype = amps.device, amps.dtype
    cov2 = cov3d[:, 1:, 1:]  # (M,2,2) over (y, z)
    # det(R Sigma R^T) == det(Sigma), so callers replicating one asymmetric unit
    # helically can compute det3 once per base Gaussian and pass it in rather
    # than paying a 3x3 determinant per helical copy (measured ~64x cheaper).
    det3 = torch.det(cov3d) if det3 is None else det3
    det3 = det3.clamp_min(EPS)
    det2 = torch.det(cov2).clamp_min(EPS)
    pref = amps * math.sqrt(2 * math.pi) * torch.sqrt(det3 / det2) / apix

    # inverse of a 2x2, done explicitly (cheaper and more stable than inverse())
    a, b, d = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    inv_a, inv_b, inv_d = d / det2, -b / det2, a / det2

    if patch_radius is None:
        # Per-Gaussian footprint: 3 sigma along the major axis of the projected
        # covariance.  A single global radius would be set by the largest
        # Gaussian and inflate the cost for every other one, so bucket by
        # required radius (powers of two) and splat each bucket at its own
        # size -- the cheap analogue of 3DGS's per-Gaussian tile binning.
        tr = a + d
        major = 0.5 * (tr + torch.sqrt((tr * tr - 4 * det2).clamp_min(0.0)))
        need = (
            (SIGMA_CUTOFF * torch.sqrt(major.detach()) / apix)
            .ceil()
            .clamp(1, max_radius)
        )
        buckets = torch.pow(2, torch.ceil(torch.log2(need.clamp_min(1.0)))).long()
        img = torch.zeros(ny * nz, device=device, dtype=dtype)
        for r in torch.unique(buckets).tolist():
            sel = buckets == r
            img = img + splat_project(
                amps[sel],
                centers[sel],
                cov3d[sel],
                ny,
                nz,
                apix,
                patch_radius=int(r),
                det3=det3[sel],
            ).T.reshape(-1)
        return img.reshape(nz, ny).T

    yc, zc = centers[:, 1], centers[:, 2]
    yi0 = torch.floor(yc / apix).long() + ny // 2
    zi0 = torch.floor(zc / apix).long() + nz // 2

    off = torch.arange(-patch_radius, patch_radius + 1, device=device)
    oy, oz = torch.meshgrid(off, off, indexing="ij")
    yi = yi0[:, None, None] + oy[None]  # (M,P,P)
    zi = zi0[:, None, None] + oz[None]
    valid = (yi >= 0) & (yi < ny) & (zi >= 0) & (zi < nz)

    dy = (yi.to(dtype) - ny // 2) * apix - yc[:, None, None]
    dz = (zi.to(dtype) - nz // 2) * apix - zc[:, None, None]
    q = (
        inv_a[:, None, None] * dy * dy
        + 2 * inv_b[:, None, None] * dy * dz
        + inv_d[:, None, None] * dz * dz
    )
    val = pref[:, None, None] * torch.exp(-0.5 * q) * valid.to(dtype)

    flat = (zi.clamp(0, nz - 1) * ny + yi.clamp(0, ny - 1)).reshape(-1)
    img = torch.zeros(ny * nz, device=device, dtype=dtype)
    img = img.index_add(0, flat, val.reshape(-1))
    return img.reshape(nz, ny).T


def _quat_mul_z(q, angle):
    """Compose a rotation about z (by `angle`, a per-Gaussian tensor) onto quats."""
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [c * w2 - s * z2, c * x2 - s * y2, c * y2 + s * x2, c * z2 + s * w2], dim=-1
    )


def _wrap_to_one_rise(centers, quats, rise_angstrom, twist_degree):
    """Fold Gaussians back into a single rise using the helical symmetry.

    A Gaussian at (x, y, z + k*rise) is the exact symmetry mate of one at
    (Rz(-k*twist) . (x, y), z), so folding is loss-preserving -- it just keeps
    the asymmetric unit canonical, as required ("sample a single rise, then
    duplicate").
    """
    k = torch.round(centers[:, 2] / rise_angstrom)
    # twist_degree may arrive as int, float or a 0-d tensor; float() covers all
    angle = -k * math.radians(float(twist_degree))
    ca, sa = torch.cos(angle), torch.sin(angle)
    x, y = centers[:, 0], centers[:, 1]
    new = torch.stack(
        [x * ca - y * sa, x * sa + y * ca, centers[:, 2] - k * rise_angstrom], dim=-1
    )
    return new, _quat_mul_z(quats, angle)


class SplatModel:
    """Anisotropic Gaussians for one asymmetric unit, with 3DGS-style
    adaptive density control."""

    def __init__(
        self, n_init, tube_radius, rise_angstrom, apix, device, seed=0, n_rise_slab=1
    ):
        g = torch.Generator(device="cpu").manual_seed(seed)
        r = torch.sqrt(torch.rand(n_init, generator=g)) * tube_radius * 0.85
        th = torch.rand(n_init, generator=g) * 2 * math.pi
        z = (torch.rand(n_init, generator=g) - 0.5) * rise_angstrom * n_rise_slab
        centers = torch.stack([r * torch.cos(th), r * torch.sin(th), z], dim=-1)

        self.centers = centers.to(device).requires_grad_(True)
        self.log_amp = torch.full(
            (n_init,), math.log(1e-2), device=device
        ).requires_grad_(True)
        self.log_scales = torch.full(
            (n_init, 3), math.log(apix), device=device
        ).requires_grad_(True)
        quats = torch.zeros(n_init, 4, device=device)
        quats[:, 0] = 1.0
        self.quats = quats.requires_grad_(True)
        self.device = device
        # Densification jitter must come from a seeded generator, not the global
        # torch RNG: otherwise identical arguments give different answers run to
        # run (observed 0.8476 vs 0.8523 for the same config), which also makes
        # the result unsafe to cache.
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self._reset_stats()

    def _reset_stats(self):
        n = self.centers.shape[0]
        self.grad_accum = torch.zeros(n, device=self.device)
        self.grad_count = torch.zeros(n, device=self.device)

    @property
    def n(self):
        return self.centers.shape[0]

    def params(self):
        return [self.centers, self.log_amp, self.log_scales, self.quats]

    def accumulate_grad(self):
        if self.centers.grad is not None:
            self.grad_accum += self.centers.grad.norm(dim=-1)
            self.grad_count += 1

    def _set(self, centers, log_amp, log_scales, quats):
        self.centers = centers.detach().clone().requires_grad_(True)
        self.log_amp = log_amp.detach().clone().requires_grad_(True)
        self.log_scales = log_scales.detach().clone().requires_grad_(True)
        self.quats = quats.detach().clone().requires_grad_(True)
        self._reset_stats()

    @torch.no_grad()
    def fold(self, rise_angstrom, twist_degree, tube_radius):
        c, q = _wrap_to_one_rise(self.centers, self.quats, rise_angstrom, twist_degree)
        rad = c[:, :2].norm(dim=-1, keepdim=True)
        scale = (rad.clamp(max=tube_radius) / rad.clamp_min(EPS)).clamp(max=1.0)
        c = torch.cat([c[:, :2] * scale, c[:, 2:]], dim=-1)
        self.centers.copy_(c)
        self.quats.copy_(q)

    @torch.no_grad()
    def densify_and_prune(
        self, grad_percentile, size_threshold, min_amp_frac, max_gauss, split_scale=1.6
    ):
        """Clone small high-gradient Gaussians, split large ones, prune faint ones."""
        avg_grad = self.grad_accum / self.grad_count.clamp_min(1)
        amps = torch.exp(self.log_amp)
        scales_max = torch.exp(self.log_scales).max(dim=-1).values

        keep = amps >= min_amp_frac * amps.max()
        if keep.sum() < 8:  # never prune everything away
            keep = torch.ones_like(keep)

        c, la, ls, q, ag = (
            self.centers[keep],
            self.log_amp[keep],
            self.log_scales[keep],
            self.quats[keep],
            avg_grad[keep],
        )
        sm = torch.exp(ls).max(dim=-1).values
        n_pruned = int((~keep).sum())

        room = max(0, max_gauss - c.shape[0])
        n_clone = n_split = 0
        if room > 0 and ag.numel() > 0:
            thr = torch.quantile(ag, grad_percentile)
            sel = ag > thr
            clone_sel = sel & (sm <= size_threshold)
            split_sel = sel & (sm > size_threshold)

            # cap the number added so we stay under max_gauss
            def _cap(mask, budget):
                idx = torch.nonzero(mask).flatten()
                return idx[:budget]

            ci = _cap(clone_sel, room)
            room -= ci.numel()
            si = _cap(split_sel, room)
            n_clone, n_split = ci.numel(), si.numel()

            new_c, new_la, new_ls, new_q = [c], [la], [ls], [q]
            if n_clone:
                jitter = torch.randn(n_clone, 3, generator=self.gen).to(
                    c.device
                ) * torch.exp(ls[ci]).mean(dim=-1, keepdim=True)
                new_c.append(c[ci] + jitter)
                new_la.append(la[ci])
                new_ls.append(ls[ci])
                new_q.append(q[ci])
            if n_split:
                # replace each with two smaller children sampled from its own PDF
                shrunk = ls[si] - math.log(split_scale)
                for _ in range(2):
                    jitter = torch.randn(n_split, 3, generator=self.gen).to(
                        c.device
                    ) * torch.exp(ls[si])
                    new_c.append(c[si] + jitter)
                    new_la.append(la[si] - math.log(2.0))
                    new_ls.append(shrunk)
                    new_q.append(q[si])
                keep2 = torch.ones(c.shape[0], dtype=torch.bool, device=c.device)
                keep2[si] = False
                new_c[0], new_la[0], new_ls[0], new_q[0] = (
                    c[keep2],
                    la[keep2],
                    ls[keep2],
                    q[keep2],
                )

            c = torch.cat(new_c)
            la = torch.cat(new_la)
            ls = torch.cat(new_ls)
            q = torch.cat(new_q)

        self._set(c, la, ls, q)
        return dict(pruned=n_pruned, cloned=int(n_clone), split=int(n_split), n=self.n)


def _make_opt(model, lr_center, lr_amp, lr_scale, lr_quat):
    return torch.optim.Adam(
        [
            {"params": [model.centers], "lr": lr_center},
            {"params": [model.log_amp], "lr": lr_amp},
            {"params": [model.log_scales], "lr": lr_scale},
            {"params": [model.quats], "lr": lr_quat},
        ]
    )


def splat_fit(
    projection_image,
    apix2d,
    twist_degree,
    rise_angstrom,
    csym,
    tube_radius,
    n_init=200,
    iters=1500,
    lr_center=0.5,
    lr_amp=0.05,
    lr_scale=0.02,
    lr_quat=0.005,
    lr_center_final_frac=0.02,
    densify_from=150,
    densify_until=1200,
    densify_interval=100,
    grad_percentile=0.7,
    size_threshold_frac=2.0,
    min_amp_frac=0.005,
    max_gauss=3000,
    min_scale_frac=0.25,
    max_scale_frac=8.0,
    seed=0,
    device=None,
    tilt_degree=0.0,
    psi_degree=0.0,
    dy_pixel=0.0,
    refine_tilt_psi_dy=False,
    refine_tilt=None,
    refine_psi=None,
    refine_dy=None,
    rigid_lr=0.15,
    rigid_warmup_frac=0.25,
    rigid_alternate=False,
    reg_l1=0.0,
    reg_l2=0.0,
    reg_scale=0.0,
    reg_aniso=0.0,
    reg_drift=0.0,
    reg_bg=0.0,
    reg_tv=0.0,
    rigid_block=10,
    tilt_max_degree=15.0,
    psi_max_degree=15.0,
    dy_max_pixel=None,
    init_model=None,
    n_rise_slab=1,
    compile_kernel=False,
    verbose=0,
):
    """Fit anisotropic Gaussians (one rise, helically duplicated) to a 2D image.

    Returns (model, best_cosine, history).
    """
    _require_torch()
    device = best_device(device)
    target = torch.tensor(
        np.ascontiguousarray(projection_image), dtype=torch.float32, device=device
    )
    ny, nz = target.shape
    Lz = nz * apix2d
    # Grouping n consecutive helical copies into one "super-unit" and
    # replicating that by (n*twist, n*rise) generates exactly the same set of
    # copies -- it is a re-indexing, not an approximation -- but needs n times
    # fewer replication steps.  It relaxes the symmetry constraint from
    # every-rise to every-n-rises, so the slab's interior is no longer forced
    # to be helically symmetric.  That is only safe when the rise is small
    # compared to the pixel size (here rise ~ 0.95 px), where the image cannot
    # resolve individual rises anyway.
    n_rise_slab = max(1, int(n_rise_slab))
    twist_eff = twist_degree * n_rise_slab
    rise_eff = rise_angstrom * n_rise_slab
    n_repeats = max(1, int(np.ceil(Lz / (2 * rise_eff))) + 4) if rise_eff > 0.01 else 4

    model = (
        init_model
        if init_model is not None
        else SplatModel(
            n_init,
            tube_radius,
            rise_angstrom,
            apix2d,
            device,
            seed=seed,
            n_rise_slab=n_rise_slab,
        )
    )
    opt = _make_opt(model, lr_center, lr_amp, lr_scale, lr_quat)

    # tilt/psi/dy: applied always, optimised only when refine_tilt_psi_dy.
    # Refinement is a *local* correction from the given starting value, with the
    # rigid parameters held frozen for a warmup so the density settles first
    # (same staged schedule validated for gauss_fixed on synthetic ground truth).
    # Each of tilt/psi/dy can be refined independently.  This matters because
    # *tilt* is partially degenerate with twist (tilting a filament changes its
    # apparent axial repeat in projection), so refining tilt during a twist
    # search lets a wrong twist be absorbed by tilting and flattens the twist
    # discrimination.  psi and dy show no such coupling, so refining just those
    # gets the geometry correction without damaging the search.
    refine_tilt = refine_tilt_psi_dy if refine_tilt is None else refine_tilt
    refine_psi = refine_tilt_psi_dy if refine_psi is None else refine_psi
    refine_dy = refine_tilt_psi_dy if refine_dy is None else refine_dy
    any_rigid = bool(refine_tilt or refine_psi or refine_dy)

    tilt_t = torch.tensor(float(tilt_degree), device=device)
    psi_t = torch.tensor(float(psi_degree), device=device)
    dy_t = torch.tensor(float(dy_pixel), device=device)
    opt_rigid = None
    if any_rigid:
        rigid_params = []
        if refine_tilt:
            tilt_t = tilt_t.requires_grad_(True)
            rigid_params.append(tilt_t)
        if refine_psi:
            psi_t = psi_t.requires_grad_(True)
            rigid_params.append(psi_t)
        if refine_dy:
            dy_t = dy_t.requires_grad_(True)
            rigid_params.append(dy_t)
        opt_rigid = torch.optim.Adam(rigid_params, lr=rigid_lr)
    rigid_warmup = int(rigid_warmup_frac * iters)
    # Alternating schedule (OFF by default -- measured not to help).  Free
    # Gaussians absorb a rigid misalignment by
    # rearranging -- an easier descent direction than a global rotation -- so
    # jointly optimising them starves psi/dy of gradient pressure.  Measured on
    # a real class average: a frozen model recovers an applied 5 deg rotation
    # essentially exactly (-5.00), while joint refinement recovers only -0.37.
    # Holding the Gaussians still mid-optimisation does NOT fix it (psi still
    # recovered only -0.40 for an applied 5 deg): by then they have already
    # adapted, so the frozen model explains the data and psi has no gradient.
    # What *does* work is scanning psi against a model fitted BEFORE the
    # misalignment was introduced -- i.e. a coarse psi grid search on a frozen
    # model, not gradient descent inside the joint fit.
    _rigid_only = lambda it: (
        rigid_alternate
        and opt_rigid is not None
        and it >= rigid_warmup
        and ((it - rigid_warmup) // max(1, rigid_block)) % 2 == 1
    )
    if dy_max_pixel is None:
        dy_max_pixel = 0.25 * projection_image.shape[0]

    twist_t = torch.tensor(float(twist_eff), device=device)
    rise_t = torch.tensor(float(rise_eff), device=device)
    min_scale, max_scale = math.log(min_scale_frac * apix2d), math.log(
        max_scale_frac * apix2d
    )
    size_threshold = size_threshold_frac * apix2d
    tgt_flat = target.flatten()

    # NOTE: torch.compile speeds up splat_project ~1.5x when called with a
    # fixed patch_radius, but is a net *pessimisation* in this loop (measured
    # 112s vs 92s): size bucketing calls it with several different radii, which
    # blows past dynamo's recompile limit and falls back to eager after paying
    # the compile cost.  Off by default; only worth enabling if the bucketing
    # is ever specialised per radius.
    projector = splat_project
    if compile_kernel:
        try:
            projector = torch.compile(splat_project, dynamic=False)
        except Exception as exc:  # pragma: no cover - depends on torch build
            logger.info(f"  torch.compile unavailable ({exc}); using eager")
            projector = splat_project

    _init_centers = model.centers.detach().clone() if reg_drift else None
    _bg_mask = (target <= 0) if reg_bg else None
    best_cos, best_state, history = -1.0, None, []
    best_rigid = (float(tilt_degree), float(psi_degree), float(dy_pixel))
    for it in range(iters):
        frac = it / max(1, iters - 1)
        opt.param_groups[0]["lr"] = lr_center * (
            lr_center_final_frac**frac
        )  # exp decay

        cov3d = build_cov3d(model.log_scales, model.quats)
        amps = torch.exp(model.log_amp)
        ea, ec, ecov = helical_expand(
            amps, model.centers, cov3d, twist_t, rise_t, csym, n_repeats
        )
        ec, ecov = _rigid_transform_t(ec, ecov, tilt_t, psi_t, dy_t, apix2d)
        # det3 is rotation-invariant, so the base determinants carry over to
        # every helical copy (helical_expand lays copies out n-major).
        n_copies = ea.shape[0] // model.n
        det3 = torch.det(cov3d).repeat(n_copies)
        proj = projector(ea, ec, ecov, ny, nz, apix2d, det3=det3)
        cos = F.cosine_similarity(proj.flatten(), tgt_flat, dim=0)
        loss = 1 - cos
        # Optional penalties.  Note the analogy with elasticnet is loose: lsq
        # regularises thousands of voxels on a fixed grid, whereas here the
        # model is already compact and adaptive.  The penalties that matter are
        # the ones that remove *excess capacity* -- in particular reg_scale and
        # reg_drift deny the Gaussians the ability to absorb a rigid
        # misalignment by widening or rearranging, which is what starves psi/dy
        # of gradient (see the psi diagnostic in tests/output).
        if reg_l1:
            loss = loss + reg_l1 * amps.abs().mean()  # sparsity
        if reg_l2:
            loss = loss + reg_l2 * (amps**2).mean()  # shrinkage
        if reg_scale:
            sc = torch.exp(model.log_scales)
            loss = loss + reg_scale * (sc**2).mean()  # discourage widening
        if reg_aniso:
            sc = torch.exp(model.log_scales)
            ratio = sc.max(dim=-1).values / sc.min(dim=-1).values.clamp_min(EPS)
            loss = loss + reg_aniso * (ratio**2).mean()  # discourage needles
        if reg_drift and _init_centers is not None:
            loss = loss + reg_drift * ((model.centers - _init_centers) ** 2).mean()
        if reg_bg and _bg_mask is not None:
            # The thresholded input has exactly-zero background, so any
            # predicted density there is an artefact.  This also penalises the
            # specific failure mode that starves psi: smearing the model out to
            # cover a drifted/rotated filament necessarily spills into the
            # background.
            loss = loss + reg_bg * (proj[_bg_mask] ** 2).mean()
        if reg_tv:
            tv = (proj[1:, :] - proj[:-1, :]).abs().mean() + (
                proj[:, 1:] - proj[:, :-1]
            ).abs().mean()
            loss = loss + reg_tv * tv

        opt.zero_grad()
        if opt_rigid is not None:
            opt_rigid.zero_grad()
        loss.backward()
        model.accumulate_grad()
        if not _rigid_only(it):
            opt.step()
        if opt_rigid is not None and it >= rigid_warmup:
            opt_rigid.step()
            with torch.no_grad():
                if refine_tilt:
                    tilt_t.clamp_(-tilt_max_degree, tilt_max_degree)
                if refine_psi:
                    psi_t.clamp_(-psi_max_degree, psi_max_degree)
                if refine_dy:
                    dy_t.clamp_(-dy_max_pixel, dy_max_pixel)

        with torch.no_grad():
            model.log_scales.clamp_(min_scale, max_scale)
        model.fold(rise_eff, twist_eff, tube_radius)

        c = float(cos.detach())
        history.append((it, c, model.n))
        if c > best_cos:
            best_cos = c
            best_state = (
                model.centers.detach().clone(),
                model.log_amp.detach().clone(),
                model.log_scales.detach().clone(),
                model.quats.detach().clone(),
            )
            best_rigid = (
                float(tilt_t.detach()),
                float(psi_t.detach()),
                float(dy_t.detach()),
            )

        if (
            densify_from <= it < densify_until
            and (it - densify_from) % densify_interval == 0
            and it > densify_from
        ):
            stats = model.densify_and_prune(
                grad_percentile, size_threshold, min_amp_frac, max_gauss
            )
            opt = _make_opt(model, lr_center, lr_amp, lr_scale, lr_quat)
            if verbose:
                logger.info(f"  it {it}: cos={c:.4f} {stats}")
        elif verbose and it % 200 == 0:
            logger.info(f"  it {it}: cos={c:.4f} n={model.n}")

    if best_state is not None:
        model._set(*best_state)
    if any_rigid:
        near = [
            n
            for n, v, b, on in (
                ("tilt", best_rigid[0], tilt_max_degree, refine_tilt),
                ("psi", best_rigid[1], psi_max_degree, refine_psi),
                ("dy", best_rigid[2], dy_max_pixel, refine_dy),
            )
            if on and abs(v) > 0.9 * b
        ]
        if near:
            logger.warning(
                f"  tilt/psi/dy refinement hit its bound for {near} (values "
                f"{best_rigid}). Beware: tilt is partially degenerate with twist "
                "-- a wrong twist can be absorbed by tilting, so a railed tilt "
                "usually means the twist is wrong rather than that the particle "
                "is really tilted that far."
            )
    splat_fit.refined_tilt_psi_dy = best_rigid
    return model, best_cos, history


def _rigid_transform(centers, cov3d, tilt_degree, psi_degree, dy_pixel, apix):
    """Apply the tilt/psi/dy misalignment to centres *and* covariances.

    Same convention as denovo3d_solver.build_A_data_matrix's inverse usage:
    data = Rx(psi) @ Ry(tilt) @ canonical + [0, dy, 0].  A rotated Gaussian's
    covariance transforms as Sigma -> R Sigma R^T.
    """
    if abs(tilt_degree) < 1e-9 and abs(psi_degree) < 1e-9 and abs(dy_pixel) < 1e-9:
        return centers, cov3d
    dev, dt = centers.device, centers.dtype
    t, p = math.radians(tilt_degree), math.radians(psi_degree)
    Ry = torch.tensor(
        [[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]],
        device=dev,
        dtype=dt,
    )
    Rx = torch.tensor(
        [[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]],
        device=dev,
        dtype=dt,
    )
    R = Rx @ Ry
    c = centers @ R.T
    shift = torch.tensor([0.0, dy_pixel * apix, 0.0], device=dev, dtype=dt)
    return c + shift, R @ cov3d @ R.T


def _build_volume(
    amps, centers, cov3d, nz3, ny3, nx3, apix3d, batch=256, cull=True, max_radius=32
):
    """Evaluate anisotropic Gaussians on a voxel grid (canonical frame).

    Footprint-culled in 3D for the same reason splat_project is culled in 2D:
    a Gaussian only contributes within a few sigma of its centre, so evaluating
    every Gaussian against every voxel is wasted work.  For a typical sigma of
    2.5 A at 5 A voxels that is a ~5^3 patch instead of the whole volume -- the
    difference between milliseconds and minutes on a display-sized grid.
    """
    dev = amps.device
    if amps.numel() == 0:
        return torch.zeros(nz3, ny3, nx3, device=dev, dtype=torch.float32)

    inv_all = torch.inverse(cov3d)
    if not cull:
        z = (torch.arange(nz3, device=dev, dtype=amps.dtype) - nz3 // 2) * apix3d
        y = (torch.arange(ny3, device=dev, dtype=amps.dtype) - ny3 // 2) * apix3d
        x = (torch.arange(nx3, device=dev, dtype=amps.dtype) - nx3 // 2) * apix3d
        Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")
        grid = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        vol = torch.zeros(grid.shape[0], device=dev, dtype=amps.dtype)
        for i in range(0, amps.shape[0], batch):
            d = grid.unsqueeze(1) - centers[i : i + batch].unsqueeze(0)
            e = -0.5 * torch.einsum("vbi,bij,vbj->vb", d, inv_all[i : i + batch], d)
            vol += (amps[i : i + batch].unsqueeze(0) * torch.exp(e)).sum(-1)
        return vol.reshape(nz3, ny3, nx3)

    # per-Gaussian radius from the largest principal sigma, bucketed by power of two
    var_max = torch.linalg.eigvalsh(cov3d.detach()).clamp_min(0).max(dim=-1).values
    need = (SIGMA_CUTOFF * torch.sqrt(var_max) / apix3d).ceil().clamp(1, max_radius)
    buckets = torch.pow(2, torch.ceil(torch.log2(need.clamp_min(1.0)))).long()

    vol = torch.zeros(nz3 * ny3 * nx3, device=dev, dtype=amps.dtype)
    for r in torch.unique(buckets).tolist():
        sel = buckets == r
        c, a_, inv = centers[sel], amps[sel], inv_all[sel]
        r = int(r)
        off = torch.arange(-r, r + 1, device=dev)
        oz, oy, ox = torch.meshgrid(off, off, off, indexing="ij")
        oz, oy, ox = oz.reshape(-1), oy.reshape(-1), ox.reshape(-1)
        zi0 = torch.floor(c[:, 2] / apix3d).long() + nz3 // 2
        yi0 = torch.floor(c[:, 1] / apix3d).long() + ny3 // 2
        xi0 = torch.floor(c[:, 0] / apix3d).long() + nx3 // 2
        for i in range(0, c.shape[0], batch):
            cb, ab, ib = c[i : i + batch], a_[i : i + batch], inv[i : i + batch]
            zi = zi0[i : i + batch, None] + oz
            yi = yi0[i : i + batch, None] + oy
            xi = xi0[i : i + batch, None] + ox
            valid = (
                (zi >= 0) & (zi < nz3) & (yi >= 0) & (yi < ny3) & (xi >= 0) & (xi < nx3)
            )
            dx = (xi.to(amps.dtype) - nx3 // 2) * apix3d - cb[:, 0, None]
            dy = (yi.to(amps.dtype) - ny3 // 2) * apix3d - cb[:, 1, None]
            dz = (zi.to(amps.dtype) - nz3 // 2) * apix3d - cb[:, 2, None]
            d = torch.stack([dx, dy, dz], dim=-1)
            e = -0.5 * torch.einsum("bpi,bij,bpj->bp", d, ib, d)
            val = (ab[:, None] * torch.exp(e)) * valid.to(amps.dtype)
            flat = (
                zi.clamp(0, nz3 - 1) * ny3 * nx3
                + yi.clamp(0, ny3 - 1) * nx3
                + xi.clamp(0, nx3 - 1)
            )
            vol = vol.index_add(0, flat.reshape(-1), val.reshape(-1))
    return vol.reshape(nz3, ny3, nx3)


def _gauss_splat_solve(
    projection_image,
    scale2d_to_3d,
    twist_degree,
    rise_pixel,
    csym=1,
    tilt_degree=0,
    psi_degree=0,
    dy_pixel=0,
    thresh_fraction=-1,
    positive_constraint=-1,
    reconstruct_diameter_3d_inner_pixel=0,
    reconstruct_diameter_2d_pixel=-1,
    reconstruct_diameter_3d_pixel=-1,
    reconstruct_length_2d_pixel=-1,
    reconstruct_length_3d_pixel=-1,
    sym_oversample=1,
    interpolation="nn",
    fsc_test=0,
    score_metric="cosine",
    target_apix2d=5.0,
    verbose=0,
    algorithm=None,
):
    """Cached core of gauss_splat_reconstruct.

    Returns (rec3d, score, refined_params).  Caching is only sound because the
    fit is deterministic given its arguments (densification jitter comes from a
    seeded per-model generator, not the global torch RNG).  ``verbose`` is
    excluded from the cache key; note ``device`` lives inside ``algorithm`` and
    therefore *is* part of the key, which is correct -- CPU and MPS agree only
    to float32 rounding.
    """
    _require_torch()
    algorithm = algorithm or dict(model="gauss_splat")
    ny, nx = projection_image.shape
    if reconstruct_diameter_2d_pixel <= 0:
        reconstruct_diameter_2d_pixel = ny
    if reconstruct_length_2d_pixel <= 0:
        reconstruct_length_2d_pixel = nx
    if reconstruct_diameter_3d_pixel <= 0:
        reconstruct_diameter_3d_pixel = int(np.rint(reconstruct_diameter_2d_pixel))
    if reconstruct_length_3d_pixel <= 0:
        reconstruct_length_3d_pixel = int(np.rint(reconstruct_length_2d_pixel))
    reconstruct_diameter_3d_pixel = int(np.rint(reconstruct_diameter_3d_pixel))
    reconstruct_length_3d_pixel = int(np.rint(reconstruct_length_3d_pixel))

    apix2d = target_apix2d
    apix3d = target_apix2d / scale2d_to_3d
    rise_angstrom = rise_pixel * apix3d
    tube_radius = reconstruct_diameter_3d_pixel * apix3d * 0.5

    # Optional hybrid start: solve with lsq first and seed the Gaussians from
    # its voxel reconstruction.  lsq is ~800x faster and gives a good density
    # estimate but cannot search tilt/psi/dy well (it uses finite-difference
    # Gauss-Newton, rebuilding a sparse matrix per perturbation); splat gets
    # those gradients from the same backward pass.  Handing lsq's answer to
    # splat plays to both: splat starts near the solution instead of spending
    # its first few hundred iterations discovering where the density is.
    init_model = None
    if algorithm.get("init_from_lsq", False):
        from .denovo3d_solver import lsq_reconstruct

        lsq_algo = dict(
            algorithm.get("lsq_algorithm", dict(model="elasticnet", l1_ratio=0.5))
        )
        (lsq_vol, _, _), lsq_score = lsq_reconstruct(
            projection_image=projection_image,
            scale2d_to_3d=scale2d_to_3d,
            twist_degree=twist_degree,
            rise_pixel=rise_pixel,
            csym=csym,
            tilt_degree=tilt_degree,
            psi_degree=psi_degree,
            dy_pixel=dy_pixel,
            thresh_fraction=thresh_fraction,
            positive_constraint=positive_constraint,
            reconstruct_diameter_3d_inner_pixel=reconstruct_diameter_3d_inner_pixel,
            reconstruct_diameter_2d_pixel=reconstruct_diameter_2d_pixel,
            reconstruct_diameter_3d_pixel=reconstruct_diameter_3d_pixel,
            reconstruct_length_2d_pixel=reconstruct_length_2d_pixel,
            reconstruct_length_3d_pixel=reconstruct_length_3d_pixel,
            sym_oversample=sym_oversample,
            interpolation=interpolation,
            score_metric=score_metric,
            target_apix2d=target_apix2d,
            verbose=verbose,
            algorithm=lsq_algo,
        )
        logger.info(f"  hybrid: lsq seed score={lsq_score:.6f}")
        try:
            init_model = init_from_volume(
                lsq_vol,
                apix3d=apix3d,
                rise_angstrom=rise_angstrom,
                twist_degree=twist_degree,
                tube_radius=tube_radius,
                n_gauss=algorithm.get("n_init", 150),
                device=best_device(algorithm.get("device", None)),
                seed=algorithm.get("seed", 0),
            )
        except ValueError as exc:
            logger.warning(
                f"  hybrid: lsq seed unusable ({exc}); falling back to random init"
            )

    model, cos, _ = splat_fit(
        projection_image,
        apix2d=apix2d,
        twist_degree=twist_degree,
        rise_angstrom=rise_angstrom,
        csym=csym,
        tube_radius=tube_radius,
        n_init=algorithm.get("n_init", 150),
        iters=algorithm.get("iters", 400),
        max_gauss=algorithm.get("max_gauss", 400),
        densify_from=algorithm.get("densify_from", 150),
        densify_until=algorithm.get("densify_until", 320),
        densify_interval=algorithm.get("densify_interval", 100),
        grad_percentile=algorithm.get("grad_percentile", 0.7),
        lr_center=algorithm.get("lr_center", 0.5),
        lr_amp=algorithm.get("lr_amp", 0.05),
        lr_scale=algorithm.get("lr_scale", 0.02),
        lr_quat=algorithm.get("lr_quat", 0.005),
        seed=algorithm.get("seed", 0),
        device=algorithm.get("device", None),
        tilt_degree=tilt_degree,
        psi_degree=psi_degree,
        dy_pixel=dy_pixel,
        refine_tilt_psi_dy=algorithm.get("refine_tilt_psi_dy", False),
        n_rise_slab=algorithm.get("n_rise_slab", 1),
        refine_tilt=algorithm.get("refine_tilt", None),
        refine_psi=algorithm.get("refine_psi", None),
        refine_dy=algorithm.get("refine_dy", None),
        rigid_lr=algorithm.get("rigid_lr", 0.15),
        init_model=init_model,
        verbose=verbose,
    )
    refined_params = None
    if any(
        algorithm.get(k)
        for k in ("refine_tilt_psi_dy", "refine_tilt", "refine_psi", "refine_dy")
    ):
        t, ps, dyv = splat_fit.refined_tilt_psi_dy
        logger.info(f"  refined tilt={t:.3f} psi={ps:.3f} dy={dyv:.3f}px")
        refined_params = dict(tilt=t, psi=ps, dy=dyv)
    logger.info(
        f"Gauss-splat solver: {model.n} Gaussians, cosine={cos:.6f}, "
        f"device={best_device(algorithm.get('device', None))}"
    )

    # Re-score with the requested metric.  splat_fit optimises (and returns)
    # cosine similarity, so anything else has to be evaluated on the final
    # projection -- otherwise score_metric is silently ignored and every metric
    # reports the cosine value.
    score = float(cos)
    if score_metric != "cosine":
        import helicon

        with torch.no_grad():
            cov = build_cov3d(model.log_scales, model.quats)
            n_rep_proj = (
                max(
                    1,
                    int(
                        np.ceil(
                            ny * 0
                            + projection_image.shape[1] * apix2d / (2 * rise_angstrom)
                        )
                    )
                    + 4,
                )
                if rise_angstrom > 0.01
                else 4
            )
            t, ps, dyv = getattr(
                splat_fit, "refined_tilt_psi_dy", (tilt_degree, psi_degree, dy_pixel)
            )
            ea, ec, ecov = helical_expand(
                torch.exp(model.log_amp),
                model.centers,
                cov,
                torch.tensor(float(twist_degree), device=model.device),
                torch.tensor(float(rise_angstrom), device=model.device),
                csym,
                n_rep_proj,
            )
            ec, ecov = _rigid_transform_t(
                ec,
                ecov,
                torch.tensor(float(t), device=model.device),
                torch.tensor(float(ps), device=model.device),
                torch.tensor(float(dyv), device=model.device),
                apix2d,
            )
            bp = (
                splat_project(
                    ea,
                    ec,
                    ecov,
                    projection_image.shape[0],
                    projection_image.shape[1],
                    apix2d,
                )
                .cpu()
                .numpy()
            )
        tg = np.asarray(projection_image, dtype=np.float32)
        if score_metric == "ssim":
            score = float(helicon.ssim_score(bp, tg))
        elif score_metric == "ms_ssim":
            score = float(helicon.ms_ssim_score(bp, tg))
        elif score_metric == "mutual_information":
            score = float(helicon.mutual_information_score(bp, tg))
        elif score_metric == "composite":
            score = float(
                np.mean(
                    [
                        helicon.cosine_similarity(bp.ravel(), tg.ravel()),
                        helicon.ssim_score(bp, tg),
                        helicon.ms_ssim_score(bp, tg),
                        helicon.mutual_information_score(bp, tg),
                    ]
                )
            )
        logger.info(f"  Final score ({score_metric}): {score:.6f}")

    # Final 3D volume in the canonical (untilted) frame, matching the other
    # solvers -- the pipeline applies tilt/psi/dy afterwards for display.
    with torch.no_grad():
        cov = build_cov3d(model.log_scales, model.quats)
        nz3 = reconstruct_length_3d_pixel
        ny3 = nx3 = reconstruct_diameter_3d_pixel
        n_rep_vol = (
            max(1, int(np.ceil(nz3 * apix3d / (2 * rise_angstrom))) + 2)
            if rise_angstrom > 0.01
            else 2
        )
        ea, ec, ecov = helical_expand(
            torch.exp(model.log_amp),
            model.centers,
            cov,
            torch.tensor(float(twist_degree), device=model.device),
            torch.tensor(float(rise_angstrom), device=model.device),
            csym,
            n_rep_vol,
        )
        vol = _build_volume(ea, ec, ecov, nz3, ny3, nx3, apix3d)
        rec3d = vol.cpu().numpy().astype(np.float32)

    slab = max(1, int(algorithm.get("n_rise_slab", 1)))
    with torch.no_grad():
        cov_out = build_cov3d(model.log_scales, model.quats)
        gaussians = dict(
            amps=torch.exp(model.log_amp).cpu().numpy(),
            centers=model.centers.cpu().numpy(),
            cov3d=cov_out.cpu().numpy(),
            twist_degree=float(twist_degree) * slab,
            rise_angstrom=float(rise_angstrom) * slab,
            csym=int(csym),
        )
    return rec3d, score, refined_params, gaussians


def gauss_splat_reconstruct(*args, **kwargs):
    """Gaussian-splatting solver with the standard denovo3D solver signature.

    Thin wrapper over the cached :func:`_gauss_splat_solve`.  The wrapper exists
    so that the refined tilt/psi/dy are re-published on the function attribute
    even on a cache hit -- a cached call performs no side effects, so setting
    that attribute inside the cached function would leave it stale from a
    previous run.

    Algorithm dict keys: n_init, iters, max_gauss, densify_from/until/interval,
    grad_percentile, lr_center/amp/scale/quat, seed, device, init_from_lsq,
    refine_tilt/refine_psi/refine_dy.
    """
    rec3d, score, refined_params, gaussians = _gauss_splat_solve(*args, **kwargs)
    # Thread-local, not a function attribute: the app solves many tasks
    # concurrently and a shared attribute would cross workers' results.
    _tls.refined_params = refined_params
    _tls.gaussians = gaussians
    gauss_splat_reconstruct._refined_params = refined_params  # back-compat
    return (rec3d, None, None), score


def _rigid_transform_t(centers, cov3d, tilt_t, psi_t, dy_t, apix):
    """Differentiable tilt/psi/dy applied to centres and covariances.

    Tensor version of _rigid_transform, so tilt/psi/dy can carry gradients.
    Same convention as denovo3d_solver: data = Rx(psi) @ Ry(tilt) @ canonical
    + [0, dy*apix, 0], with Sigma -> R Sigma R^T.
    """
    dev, dt = centers.device, centers.dtype
    t = torch.deg2rad(tilt_t.to(device=dev, dtype=dt))
    p = torch.deg2rad(psi_t.to(device=dev, dtype=dt))
    one = torch.ones((), device=dev, dtype=dt)
    zero = torch.zeros((), device=dev, dtype=dt)
    ct, st, cp, sp = torch.cos(t), torch.sin(t), torch.cos(p), torch.sin(p)
    Ry = torch.stack(
        [
            torch.stack([ct, zero, st]),
            torch.stack([zero, one, zero]),
            torch.stack([-st, zero, ct]),
        ]
    )
    Rx = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cp, -sp]),
            torch.stack([zero, sp, cp]),
        ]
    )
    R = Rx @ Ry
    c = centers @ R.T
    dy = dy_t.to(device=dev, dtype=dt) * apix
    c = torch.stack([c[:, 0], c[:, 1] + dy, c[:, 2]], dim=1)
    return c, R @ cov3d @ R.T


def init_from_volume(
    volume,
    apix3d,
    rise_angstrom,
    twist_degree,
    tube_radius,
    n_gauss,
    device,
    scale_init=None,
    seed=0,
    thresh_frac=0.05,
):
    """Seed Gaussians from an existing voxel reconstruction (e.g. from lsq).

    The 3DGS analogue of initialising from an SfM point cloud: importance-sample
    positions proportional to density, then fold them into a single rise using
    the helical symmetry.  Starting from a solved volume means the optimiser does
    not have to discover where the density is, which is most of what the first
    couple of hundred iterations otherwise do.

    Parameters
    ----------
    volume : ndarray (nz, ny, nx) in the canonical frame at apix3d.

    Returns
    -------
    SplatModel positioned on the sampled density.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(volume, dtype=np.float64)
    nz3, ny3, nx3 = v.shape
    w = np.clip(v, 0, None)
    if w.max() <= 0:
        raise ValueError("volume has no positive density to sample")
    w = np.where(w >= thresh_frac * w.max(), w, 0.0)
    flat = w.ravel()
    total = flat.sum()
    if total <= 0:
        raise ValueError("volume has no density above threshold")
    idx = rng.choice(flat.shape[0], size=n_gauss, p=flat / total, replace=True)
    iz, iy, ix = np.unravel_index(idx, w.shape)
    # voxel indices -> physical coordinates (x, y, z), jittered within the voxel
    x = (ix - nx3 // 2 + rng.uniform(-0.5, 0.5, n_gauss)) * apix3d
    y = (iy - ny3 // 2 + rng.uniform(-0.5, 0.5, n_gauss)) * apix3d
    z = (iz - nz3 // 2 + rng.uniform(-0.5, 0.5, n_gauss)) * apix3d
    centers = torch.tensor(
        np.stack([x, y, z], axis=1), dtype=torch.float32, device=device
    )

    model = SplatModel.__new__(SplatModel)
    model.device = device
    model.gen = torch.Generator(device="cpu").manual_seed(int(seed))
    quats = torch.zeros(n_gauss, 4, device=device)
    quats[:, 0] = 1.0
    model.quats = quats.requires_grad_(True)
    if scale_init is None:
        scale_init = apix3d
    model.log_scales = torch.full(
        (n_gauss, 3), math.log(scale_init), device=device
    ).requires_grad_(True)
    amp0 = float(np.mean(flat[idx])) / max(n_gauss, 1)
    model.log_amp = torch.full(
        (n_gauss,), math.log(max(amp0, 1e-6)), device=device
    ).requires_grad_(True)
    model.centers = centers.requires_grad_(True)
    model._reset_stats()
    # fold into one rise and keep inside the tube
    model.fold(rise_angstrom, twist_degree, tube_radius)
    return model


def apply_helical_symmetry_gauss_to_grid(
    amps,
    centers,
    cov3d,
    twist_degree,
    rise_angstrom,
    csym,
    nz,
    ny,
    nx,
    apix,
    device=None,
    sigma_pad=3.5,
):
    """Render helically-duplicated Gaussians directly onto a 3D voxel grid.

    The Gaussian analogue of ``helicon.apply_helical_symmetry``.  That function
    is voxel-centric: it takes an already-rasterised volume and rotates/resamples
    copies of it, which costs an interpolation per copy and -- more importantly
    for this solver -- imposes rise-periodicity on whatever it is given.

    Here the fitted Gaussians *are* the model, so the copies can be placed
    analytically and rasterised once, at the final grid size. That means:

    * no interpolation error (each copy is evaluated in closed form, not
      resampled from a voxel grid);
    * nothing is averaged away -- with an n-rise slab the model's symmetry is
      (n*twist, n*rise), and duplicating on *that* group reproduces exactly what
      was fitted, instead of forcing rise-periodicity the fit never assumed;
    * one pass instead of one resample per copy.

    Parameters
    ----------
    amps, centers, cov3d : Tensor or ndarray
        The fitted Gaussians of one asymmetric unit (or n-rise slab).
    twist_degree, rise_angstrom, csym :
        The symmetry the Gaussians were fitted under -- for an n-rise slab pass
        the *effective* (n*twist, n*rise).
    nz, ny, nx, apix :
        Output grid, centred on the origin.

    Returns
    -------
    ndarray (nz, ny, nx), float32
    """
    _require_torch()
    device = device or "cpu"

    def _t(v):
        if isinstance(v, torch.Tensor):
            return v.to(device=device, dtype=torch.float32)
        return torch.tensor(np.asarray(v), dtype=torch.float32, device=device)

    amps, centers, cov3d = _t(amps), _t(centers), _t(cov3d)

    # enough copies to cover the grid's z extent, plus the Gaussian tails
    half_z = 0.5 * nz * apix
    pad = sigma_pad * float(torch.sqrt(cov3d[:, 2, 2].max())) if amps.numel() else 0.0
    n_repeats = (
        max(1, int(np.ceil((half_z + pad) / max(rise_angstrom, 1e-6))) + 1)
        if rise_angstrom > 0.01
        else 2
    )
    exp_amps, exp_centers, exp_cov = helical_expand(
        amps,
        centers,
        cov3d,
        torch.tensor(float(twist_degree), device=device),
        torch.tensor(float(rise_angstrom), device=device),
        int(csym),
        n_repeats,
    )
    # drop copies whose centre is far outside the grid -- they contribute nothing
    keep = exp_centers[:, 2].abs() <= half_z + pad
    if keep.any():
        exp_amps, exp_centers, exp_cov = (
            exp_amps[keep],
            exp_centers[keep],
            exp_cov[keep],
        )

    with torch.no_grad():
        vol = _build_volume(exp_amps, exp_centers, exp_cov, nz, ny, nx, apix)
    return vol.cpu().numpy().astype(np.float32)
