from __future__ import annotations

import logging
import math
import time

import numpy as np

import helicon

logger = logging.getLogger(__name__)

# PyTorch is an optional dependency (no Intel-mac wheels exist for the project's
# supported Python versions).  Guard the import so the rest of the package can
# still be imported; the gauss/gauss_fixed models only run when torch is present.
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on torch-less installs
    torch = None
    F = None
    TORCH_AVAILABLE = False


def _require_torch():
    """Raise if torch is unavailable.

    Returns
    -------
    torch module

    Raises
    ------
    ImportError
        If torch is not installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "The gauss/gauss_fixed solvers require PyTorch. "
            "Install torch in the active environment to use them."
        )
    return torch


# Multi-worker ThreadPoolExecutor: each worker runs one twist/rise combination.
# Set PyTorch to single-threaded to avoid OpenMP thread explosion
# (N workers × M threads each would oversubscribe the CPU).
if TORCH_AVAILABLE:
    torch.set_num_threads(1)


def _to_device(verbose=0):
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _fixed_helical_expand(amplitudes, centers, sigmas, twist, rise, csym, n_repeats):
    """Differentiable helical expansion: replicate Gaussians along z with rotation.

    Unlike gauss.py's apply_helical_symmetry, this uses a fixed n_range based
    on geometry constants (not on learnable centers), keeping the gradient flow
    intact.  No masking — all copies participate in the projection.

    Parameters
    ----------
    amplitudes : Tensor (N,)
    centers : Tensor (N, 3)  — (x, y, z) with z = helical axis
    sigmas : Tensor (N,)     — isotropic sigma
    twist : Tensor scalar     — twist in degrees
    rise : Tensor scalar      — rise in angstroms
    csym : int
    n_repeats : int

    Returns
    -------
    exp_amps, exp_centers, exp_sigmas — all Tensors
    """
    device = amplitudes.device
    n_gauss = amplitudes.shape[0]
    n_range = torch.arange(
        -n_repeats, n_repeats + 1, device=device, dtype=torch.float32
    )
    n_grid, i_grid = torch.meshgrid(
        n_range, torch.arange(csym, device=device, dtype=torch.float32), indexing="ij"
    )
    n_flat = n_grid.flatten()
    i_flat = i_grid.flatten()
    N = len(n_flat)

    angles = torch.deg2rad(twist) * n_flat + 2 * math.pi * i_flat / csym
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    exp_centers = centers.unsqueeze(0).expand(N, -1, -1).clone()
    exp_centers[:, :, 2] += rise * n_flat.unsqueeze(1)
    x, y = exp_centers[:, :, 0], exp_centers[:, :, 1]
    exp_centers[:, :, 0] = x * cos_a.unsqueeze(1) - y * sin_a.unsqueeze(1)
    exp_centers[:, :, 1] = x * sin_a.unsqueeze(1) + y * cos_a.unsqueeze(1)
    exp_centers = exp_centers.reshape(-1, 3)
    exp_amps = amplitudes.unsqueeze(0).expand(N, -1).reshape(-1)
    exp_sigmas = sigmas.unsqueeze(0).expand(N, -1).reshape(-1)
    return exp_amps, exp_centers, exp_sigmas


def _project_gaussians(amplitudes, centers, sigmas, ny, nz, apix, batch_size=10000):
    """Project isotropic Gaussians along x onto the (y, z) plane.

    Analytical integration along x:
        proj(y, z) = sum_i A_i * sqrt(2*pi) * sigma_i * G(y; y_i, s_i) * G(z; z_i, s_i) / apix

    Parameters
    ----------
    amplitudes : Tensor (M,)
    centers : Tensor (M, 3)  — (x, y, z)
    sigmas : Tensor (M,)
    ny, nz : int              — grid dimensions (y = diameter, z = length)
    apix : float              — voxel size in angstroms

    Returns
    -------
    proj : Tensor (ny, nz)
    """
    device = amplitudes.device
    y_vals = (torch.arange(ny, device=device, dtype=torch.float32) - ny // 2) * apix
    z_vals = (torch.arange(nz, device=device, dtype=torch.float32) - nz // 2) * apix
    Z_grid, Y_grid = torch.meshgrid(z_vals, y_vals, indexing="ij")
    YZ = torch.stack([Y_grid.flatten(), Z_grid.flatten()], dim=-1)  # (ny*nz, 2)

    proj = torch.zeros(ny * nz, device=device, dtype=torch.float32)
    sqrt_2pi = math.sqrt(2 * math.pi)
    M = amplitudes.shape[0]
    for i in range(0, M, batch_size):
        batch_amps = amplitudes[i : i + batch_size]
        batch_centers = centers[i : i + batch_size, 1:3]  # (y, z)
        batch_sigmas = sigmas[i : i + batch_size]

        diff = YZ.unsqueeze(1) - batch_centers.unsqueeze(0)
        exponent = -0.5 * (diff / batch_sigmas.unsqueeze(0).unsqueeze(2)) ** 2
        weight = batch_amps.unsqueeze(0) * sqrt_2pi * batch_sigmas.unsqueeze(0) / apix
        proj += torch.sum(weight * torch.exp(torch.sum(exponent, dim=-1)), dim=-1)

    return proj.reshape(nz, ny).T


def _apply_tilt_psi_dy(centers, tilt_degree, psi_degree, dy_pixel):
    """Rotate/shift Gaussian centers from the canonical assembly frame into
    the observed-projection frame defined by (tilt_degree, psi_degree, dy_pixel).

    This is the forward transform whose inverse is used in
    ``denovo3d_solver.build_A_data_matrix`` to back-project data-frame ray
    coordinates into the canonical reconstruction frame:
        coords0[:, 1] -= dy_pixel
        r = R.from_euler("yx", (tilt_degree, psi_degree), degrees=True)
        coords0 = r.apply(coords0, inverse=True)
    i.e. ``data_point = R_yx(tilt, psi) @ canonical_point + [0, dy_pixel, 0]``,
    where ``R_yx(tilt, psi) = Rx(psi) @ Ry(tilt)`` (matching scipy's extrinsic
    "yx" Euler convention — verified numerically against
    ``scipy.spatial.transform.Rotation.from_euler("yx", ...)``).

    Parameters
    ----------
    centers : Tensor (N, 3)         — (x, y, z) in the canonical frame
    tilt_degree, psi_degree, dy_pixel : Tensor (0-d) — may carry gradients

    Returns
    -------
    Tensor (N, 3) — transformed (x, y, z) centers in the observed-projection frame
    """
    device = centers.device
    dtype = centers.dtype
    tilt = torch.deg2rad(tilt_degree.to(device=device, dtype=dtype))
    psi = torch.deg2rad(psi_degree.to(device=device, dtype=dtype))
    one = torch.ones((), device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    cos_t, sin_t = torch.cos(tilt), torch.sin(tilt)
    cos_p, sin_p = torch.cos(psi), torch.sin(psi)

    Ry = torch.stack(
        [
            torch.stack([cos_t, zero, sin_t]),
            torch.stack([zero, one, zero]),
            torch.stack([-sin_t, zero, cos_t]),
        ]
    )
    Rx = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cos_p, -sin_p]),
            torch.stack([zero, sin_p, cos_p]),
        ]
    )
    R = Rx @ Ry

    out = centers @ R.T
    dy = dy_pixel.to(device=device, dtype=dtype)
    out = torch.stack([out[:, 0], out[:, 1] + dy, out[:, 2]], dim=1)
    return out


def gauss_reconstruct(
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
    algorithm=dict(model="gauss"),
):
    _require_torch()
    ny, nx = projection_image.shape
    if reconstruct_diameter_2d_pixel <= 0:
        reconstruct_diameter_2d_pixel = ny
    if reconstruct_length_2d_pixel <= 0:
        reconstruct_length_2d_pixel = nx
    reconstruct_diameter_2d_pixel = int(np.rint(reconstruct_diameter_2d_pixel))
    reconstruct_length_2d_pixel = int(np.rint(reconstruct_length_2d_pixel))

    # The Gaussian solver projects directly onto a 2D grid, so the projection
    # must match the input image size.  The passed reconstruct_diameter_2d_pixel
    # / reconstruct_length_2d_pixel may differ from the image (e.g. custom
    # tube_diameter) — they're used later for the 3D volume size only.
    ny, nx = projection_image.shape
    proj_ny, proj_nz = ny, nx

    if reconstruct_diameter_3d_pixel <= 0:
        reconstruct_diameter_3d_pixel = reconstruct_diameter_2d_pixel
    if reconstruct_length_3d_pixel <= 0:
        reconstruct_length_3d_pixel = reconstruct_length_2d_pixel
    reconstruct_diameter_3d_pixel = int(np.rint(reconstruct_diameter_3d_pixel))
    reconstruct_length_3d_pixel = int(np.rint(reconstruct_length_3d_pixel))

    apix2d = target_apix2d
    apix3d = target_apix2d / scale2d_to_3d

    gauss_iter = algorithm.get("gauss_iter", 300)
    gauss_lr = algorithm.get("gauss_lr", 0.01)
    n_restarts = algorithm.get("n_restarts", 2)
    early_stop_patience = algorithm.get("early_stop_patience", 30)
    grid_sigma = algorithm.get("grid_sigma", apix3d / 2.0)

    rise_angstrom = rise_pixel * apix3d
    # n_repeats must cover the *projection image's* physical length, not the
    # (often much smaller) declared reconstruction-volume length: the model
    # explicitly enumerates helical copies to tile the target image it is
    # scored against, unlike the matrix-based lsq solver which can exploit
    # the image <-> small-unique-volume symmetry relationship implicitly.
    # Using reconstruct_length_3d_pixel here (as before) starves the forward
    # model of density for most of a real (non-cropped-to-one-repeat) image,
    # since that length is deliberately kept small (~one asymmetric unit).
    Lz = proj_nz * apix2d
    n_repeats = (
        max(1, int(np.ceil(Lz / (2 * rise_angstrom))) + 4)
        if rise_angstrom > 0.01
        else 4
    )
    tube_radius_3d = reconstruct_diameter_3d_pixel * apix3d * 0.5

    device = algorithm.get("device", _to_device(verbose))
    target = torch.tensor(projection_image, dtype=torch.float32, device=device)

    best_score = -1e9
    best_rec3d = None

    # ------------------------------------------------------------------
    # Stage 0: Build initial grid and solve amplitudes via NNLS
    # (vectorized A-matrix build, instant)
    # ------------------------------------------------------------------
    grid_nr = algorithm.get("grid_n_radial", 10)
    grid_na = algorithm.get("grid_n_angular", 24)
    grid_centers = _make_cylindrical_grid(grid_nr, grid_na, tube_radius_3d, "cpu")
    n_grid = grid_centers.shape[0]

    logger.info(
        f"Gaussian solver: {n_grid} Gauss ({grid_nr}r x {grid_na}a), "
        f"{gauss_iter} iter, lr={gauss_lr}, "
        f"sigma={grid_sigma:.1f}Å, restarts={n_restarts}, n_repeats={n_repeats}, "
        f"proj={proj_ny}x{proj_nz} @ {apix2d:.3f}Å, "
        f"vol={reconstruct_length_3d_pixel}x{reconstruct_diameter_3d_pixel}x{reconstruct_diameter_3d_pixel} @ {apix3d:.3f}Å"
    )

    target_np = projection_image.astype(np.float64)
    b = target_np.ravel()
    n_pixels = len(b)
    y_grid_np = (np.arange(proj_ny, dtype=np.float64) - proj_ny // 2) * apix2d
    z_grid_np = (np.arange(proj_nz, dtype=np.float64) - proj_nz // 2) * apix2d
    sqrt_2pi = math.sqrt(2 * math.pi)
    x_c = grid_centers[:, 0].numpy()
    y_c = grid_centers[:, 1].numpy()
    z_c = grid_centers[:, 2].numpy()

    logger.info(f"  Stage 0: Building A matrix ({n_pixels} x {n_grid})...")
    t0 = time.time()
    A = np.zeros((n_pixels, n_grid), dtype=np.float64)
    for m in range(n_repeats):
        angle = m * twist_degree
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        y_m = -x_c * sin_a + y_c * cos_a
        z_m = z_c + rise_angstrom * m
        for j in range(n_grid):
            gy = np.exp(-0.5 * ((y_grid_np - y_m[j]) / grid_sigma) ** 2)
            gz = np.exp(-0.5 * ((z_grid_np - z_m[j]) / grid_sigma) ** 2)
            A[:, j] += sqrt_2pi * grid_sigma / apix2d * np.outer(gz, gy).ravel()
    logger.info(f"  Stage 0: A matrix built in {time.time()-t0:.1f}s")

    from scipy.optimize import nnls as _nnls

    grid_amps_np, _ = _nnls(A, b)
    grid_cos = float(
        np.dot(A @ grid_amps_np, b)
        / (np.linalg.norm(A @ grid_amps_np) * np.linalg.norm(b) + 1e-30)
    )
    logger.info(
        f"  Stage 0: NNLS cosine={grid_cos:.4f}, nonzero={np.sum(grid_amps_np > 1e-6)}/{n_grid}"
    )

    # ------------------------------------------------------------------
    # Stage 1: Refine positions + amplitudes (fixed sigma)
    # ------------------------------------------------------------------
    for restart in range(n_restarts):
        if restart == 0:
            # First restart: start from grid solution
            init_centers = grid_centers.clone()
            init_log_amps = torch.log(
                torch.from_numpy(np.maximum(grid_amps_np, 1e-10).astype(np.float32))
            )
        else:
            # Subsequent restarts: random perturbation of grid centers
            init_centers = (
                grid_centers.clone() + torch.randn_like(grid_centers) * apix2d * 0.5
            )
            init_log_amps = torch.log(
                torch.from_numpy(np.maximum(grid_amps_np, 1e-10).astype(np.float32))
            )

        centers = init_centers.to(device).detach().requires_grad_(True)
        log_amps = init_log_amps.to(device).detach().requires_grad_(True)
        fixed_sigma_t = torch.full((n_grid,), grid_sigma, device=device)

        params = [log_amps, centers]
        opt = torch.optim.Adam(params, lr=gauss_lr)

        best_loss_iter = 1e9
        best_iter = 0

        for iteration in range(gauss_iter):
            amps = torch.exp(log_amps)

            exp_amps, exp_centers, exp_sigmas = _fixed_helical_expand(
                amps,
                centers,
                fixed_sigma_t,
                torch.tensor(twist_degree, device=device),
                torch.tensor(rise_angstrom, device=device),
                csym,
                n_repeats,
            )

            proj = _project_gaussians(
                exp_amps,
                exp_centers,
                exp_sigmas,
                proj_ny,
                proj_nz,
                apix2d,
            )
            cos_sim = F.cosine_similarity(proj.flatten(), target.flatten(), dim=0)
            cos_loss = 1 - cos_sim
            reg_amp = 1e-6 * (amps**2).mean()
            loss = cos_loss + reg_amp
            loss_val = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if loss_val < best_loss_iter:
                best_loss_iter = loss_val
                best_iter = iteration

            if iteration % 100 == 0 and verbose:
                logger.info(
                    f"  restart {restart+1} iter {iteration}: loss={loss_val:.6f}"
                )

            if early_stop_patience > 0 and iteration - best_iter >= early_stop_patience:
                if verbose:
                    logger.info(
                        f"  restart {restart+1}: early stop at iter {iteration}"
                    )
                break

        score = -best_loss_iter
        logger.info(
            f"  restart {restart+1} final loss={best_loss_iter:.6f} score={score:.6f}"
        )
        if score > best_score:
            best_score = score
            with torch.no_grad():
                best_amps = torch.exp(log_amps).clone()
                best_centers = centers.clone()
                best_sigmas = fixed_sigma_t.clone()

    # Re-evaluate the best restart using the chosen score metric
    with torch.no_grad():
        exp_amps, exp_centers, exp_sigmas = _fixed_helical_expand(
            best_amps,
            best_centers,
            best_sigmas,
            torch.tensor(twist_degree, device=device),
            torch.tensor(rise_angstrom, device=device),
            csym,
            n_repeats,
        )
        best_proj = _project_gaussians(
            exp_amps,
            exp_centers,
            exp_sigmas,
            proj_ny,
            proj_nz,
            apix2d,
        )
        bp = best_proj.cpu().numpy()
        tg = target.cpu().numpy()
        if score_metric == "cosine":
            best_score = helicon.cosine_similarity(bp.ravel(), tg.ravel())
        elif score_metric == "ssim":
            best_score = helicon.ssim_score(bp, tg)
        elif score_metric == "ms_ssim":
            best_score = helicon.ms_ssim_score(bp, tg)
        elif score_metric == "mutual_information":
            best_score = helicon.mutual_information_score(bp, tg)
        elif score_metric == "composite":
            best_score = float(
                np.mean(
                    [
                        helicon.cosine_similarity(bp.ravel(), tg.ravel()),
                        helicon.ssim_score(bp, tg),
                        helicon.ms_ssim_score(bp, tg),
                        helicon.mutual_information_score(bp, tg),
                    ]
                )
            )
        else:
            best_score = float(best_score)
        logger.info(f"  Final score ({score_metric}): {best_score:.6f}")

    with torch.no_grad():
        exp_amps, exp_centers, exp_sigmas = _fixed_helical_expand(
            best_amps,
            best_centers,
            best_sigmas,
            torch.tensor(twist_degree, device=device),
            torch.tensor(rise_angstrom, device=device),
            csym,
            n_repeats,
        )
        nz3 = reconstruct_length_3d_pixel
        ny3 = nx3 = reconstruct_diameter_3d_pixel
        z_vals = (
            torch.arange(nz3, device=device, dtype=torch.float32) - nz3 // 2
        ) * apix3d
        y_vals = (
            torch.arange(ny3, device=device, dtype=torch.float32) - ny3 // 2
        ) * apix3d
        x_vals = (
            torch.arange(nx3, device=device, dtype=torch.float32) - nx3 // 2
        ) * apix3d
        Z, Y, X = torch.meshgrid(z_vals, y_vals, x_vals, indexing="ij")
        grid = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

        vol = torch.zeros(nz3 * ny3 * nx3, device=device, dtype=torch.float32)
        vol_batch = 100
        M = exp_amps.shape[0]
        for i in range(0, M, vol_batch):
            diff = grid.unsqueeze(1) - exp_centers[i : i + vol_batch].unsqueeze(0)
            exponent = (
                -0.5
                * (diff / exp_sigmas[i : i + vol_batch].unsqueeze(0).unsqueeze(2)) ** 2
            )
            vol += torch.sum(
                exp_amps[i : i + vol_batch].unsqueeze(0)
                * torch.exp(torch.sum(exponent, dim=-1)),
                dim=-1,
            )
        best_rec3d = vol.reshape(nz3, ny3, nx3).cpu().numpy().astype(np.float32)

    return (best_rec3d, None, None), float(best_score)


def _make_cylindrical_grid(
    n_radial: int,
    n_angular: int,
    tube_radius_angstrom: float,
    device: torch.device,
    z_offsets: list[float] | None = None,
    dtype=None,
) -> torch.Tensor:
    """Build a set of (x, y, z) positions on a cylindrical grid.

    Parameters
    ----------
    n_radial : int
        Number of radial layers (including center).
    n_angular : int
        Number of angular points per radial layer (ignored for layer 0).
    tube_radius_angstrom : float
        Outer radius of the tube.
    device : torch.device
    z_offsets : list of float, optional
        Z-plane offsets in angstroms. If None, only z=0 is used (2D grid).
        Multiple planes (e.g. [-8, 0, 8]) create a 3D grid that is more
        sensitive to twist variations.
    dtype : torch.dtype, optional
        Tensor dtype for the returned grid. Defaults to ``torch.float32``.

    Returns
    -------
    centers : Tensor (N, 3)   — (x, y, z) positions on the grid.
    """
    if dtype is None:
        dtype = torch.float32
    if z_offsets is None:
        z_offsets = [0.0]
    points = []
    for z0 in z_offsets:
        pts_this_z = [[0.0, 0.0, z0]]
        if n_radial > 1 and n_angular > 0:
            radii = torch.linspace(
                0.0, tube_radius_angstrom * 0.85, n_radial, device=device
            )[1:]
            angles = torch.linspace(0.0, 2 * math.pi, n_angular + 1, device=device)[:-1]
            for r in radii:
                for theta in angles:
                    c = math.cos(theta)
                    s = math.sin(theta)
                    pts_this_z.append([r * c, r * s, z0])
        points.extend(pts_this_z)
    return torch.tensor(points, device=device, dtype=dtype)


def gauss_fixed_reconstruct(
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
    algorithm=dict(model="gauss_fixed"),
):
    """Gaussian reconstruction with fixed-grid positions, amplitude-only optimization.

    Unlike ``gauss_reconstruct`` which optimises all 500 parameters (100
    Gaussians × amplitude + xyz + sigma), this version places Gaussians on a
    fixed cylindrical grid at z=0 and only optimises their amplitudes.  The
    position of each Gaussian is fixed, dramatically reducing local minima and
    enabling faster, more reproducible convergence.

    Parameters
    ----------
    projection_image : ndarray (ny, nx)
        Input 2D projection image.
    scale2d_to_3d : float
        Ratio ``target_apix2d / target_apix3d``.
    … (same signature as ``gauss_reconstruct``)

    Algorithm dict keys
    -------------------
    grid_n_radial : int
        Number of radial grid layers (default 5).
    grid_n_angular : int
        Number of angular points per radial layer (default 12).
    grid_sigma : float
        Fixed isotropic sigma in angstroms (default 2.0).
    gauss_iter : int
        Number of optimisation iterations (default 200).
    gauss_lr : float
        Adam learning rate (default 0.01).
    early_stop_patience : int
        Early-stop patience (default 30).

    Returns
    -------
    (rec3d, None, None), score
        Same format as ``gauss_reconstruct``.
    """
    _require_torch()
    ny, nx = projection_image.shape
    if reconstruct_diameter_2d_pixel <= 0:
        reconstruct_diameter_2d_pixel = ny
    if reconstruct_length_2d_pixel <= 0:
        reconstruct_length_2d_pixel = nx
    reconstruct_diameter_2d_pixel = int(np.rint(reconstruct_diameter_2d_pixel))
    reconstruct_length_2d_pixel = int(np.rint(reconstruct_length_2d_pixel))

    ny, nx = projection_image.shape
    proj_ny, proj_nz = ny, nx

    if reconstruct_diameter_3d_pixel <= 0:
        reconstruct_diameter_3d_pixel = reconstruct_diameter_2d_pixel
    if reconstruct_length_3d_pixel <= 0:
        reconstruct_length_3d_pixel = reconstruct_length_2d_pixel
    reconstruct_diameter_3d_pixel = int(np.rint(reconstruct_diameter_3d_pixel))
    reconstruct_length_3d_pixel = int(np.rint(reconstruct_length_3d_pixel))

    apix2d = target_apix2d
    apix3d = target_apix2d / scale2d_to_3d

    early_stop_patience = algorithm.get("early_stop_patience", 60)
    grid_n_radial = algorithm.get("grid_n_radial", 5)
    grid_n_angular = algorithm.get("grid_n_angular", 12)
    grid_sigma = algorithm.get("grid_sigma", apix3d / 2.0)
    fine_tune = algorithm.get("fine_tune", False)
    fine_tune_iter = algorithm.get("fine_tune_iter", 300)
    fine_tune_lr = algorithm.get("fine_tune_lr", 0.003)

    rise_angstrom = rise_pixel * apix3d
    # n_repeats must cover the *projection image's* physical length, not the
    # (often much smaller) declared reconstruction-volume length: the model
    # explicitly enumerates helical copies to tile the target image it is
    # scored against, unlike the matrix-based lsq solver which can exploit
    # the image <-> small-unique-volume symmetry relationship implicitly.
    # Using reconstruct_length_3d_pixel here (as before) starves the forward
    # model of density for most of a real (non-cropped-to-one-repeat) image,
    # since that length is deliberately kept small (~one asymmetric unit).
    Lz = proj_nz * apix2d
    n_repeats = (
        max(1, int(np.ceil(Lz / (2 * rise_angstrom))) + 4)
        if rise_angstrom > 0.01
        else 4
    )
    tube_radius_3d = reconstruct_diameter_3d_pixel * apix3d * 0.5

    # 3D grid: z-planes help capture twist-dependent variations in the
    # projection.  This grid represents a single asymmetric unit and must
    # stay confined to one rise along z regardless of the image's full
    # length -- helical symmetry (n_repeats, above) is solely responsible
    # for tiling it across the full projected image.
    grid_n_z = algorithm.get("grid_n_z", 1)
    if grid_n_z > 1 and rise_angstrom > 0.01:
        z_spacing = rise_angstrom * 0.5
        max_planes_in_one_rise = max(1, int(rise_angstrom / z_spacing) + 1)
        n_planes = min(grid_n_z, max_planes_in_one_rise)
        if n_planes % 2 == 0:
            n_planes += 1  # ensure odd so one plane is at z=0
        half = (n_planes - 1) // 2
        z_offsets = [i * z_spacing for i in range(-half, half + 1)]
    else:
        z_offsets = None

    device = algorithm.get("device", _to_device(verbose))
    fixed_centers = _make_cylindrical_grid(
        grid_n_radial,
        grid_n_angular,
        tube_radius_3d,
        device,
        z_offsets=z_offsets,
    )
    n_gauss = fixed_centers.shape[0]
    fixed_sigma = torch.full((n_gauss,), grid_sigma, device=device)

    logger.info(
        f"Gauss-fixed solver: {n_gauss} Gauss (grid {grid_n_radial}r×{grid_n_angular}a), "
        f"sigma={grid_sigma}Å, n_repeats={n_repeats}, "
        f"proj={proj_ny}x{proj_nz} @ {apix2d:.3f}Å, "
        f"vol={reconstruct_length_3d_pixel}x{reconstruct_diameter_3d_pixel}x{reconstruct_diameter_3d_pixel} @ {apix3d:.3f}Å"
    )

    target = torch.tensor(projection_image, dtype=torch.float32, device=device)
    amp_scale = float(target.max() - target.min()) / max(n_gauss, 1)
    amp_scale = max(amp_scale, 1e-6)

    target_np = projection_image.astype(np.float64)
    b = target_np.ravel()
    n_pixels = len(b)

    # ------------------------------------------------------------------
    # Solve the linear system directly:  A @ amps = b
    # The forward model is linear in amplitudes, so this is exact.
    # Build A analytically using the Gaussian projection formula.
    # ------------------------------------------------------------------
    logger.info(f"  Building A matrix ({n_pixels} x {n_gauss})...")
    t0 = time.time()
    A = np.zeros((n_pixels, n_gauss), dtype=np.float64)
    y_grid_np = (np.arange(proj_ny, dtype=np.float64) - proj_ny // 2) * apix2d
    z_grid_np = (np.arange(proj_nz, dtype=np.float64) - proj_nz // 2) * apix2d
    sqrt_2pi = math.sqrt(2 * math.pi)
    x_c = fixed_centers[:, 0].numpy()
    y_c = fixed_centers[:, 1].numpy()
    z_c = fixed_centers[:, 2].numpy()
    sigma_val = fixed_sigma[0].item()

    for m in range(n_repeats):
        angle = m * twist_degree
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        y_m = -x_c * sin_a + y_c * cos_a
        z_m = z_c + rise_angstrom * m
        for j in range(n_gauss):
            gy = np.exp(-0.5 * ((y_grid_np - y_m[j]) / sigma_val) ** 2)
            gz = np.exp(-0.5 * ((z_grid_np - z_m[j]) / sigma_val) ** 2)
            A[:, j] += sqrt_2pi * sigma_val / apix2d * np.outer(gz, gy).ravel()
    logger.info(f"  A matrix built in {time.time()-t0:.1f}s")

    # Unconstrained least-squares
    amps_lsq, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    proj_lsq = A @ amps_lsq
    cos_lsq = float(
        np.dot(proj_lsq, b) / (np.linalg.norm(proj_lsq) * np.linalg.norm(b) + 1e-30)
    )
    logger.info(f"  Unconstrained LSQ: cosine={cos_lsq:.6f}, rank={rank}/{n_gauss}")

    # Non-negative least squares (NNLS)
    try:
        from scipy.optimize import nnls as _nnls

        amps_nnls, rnorm = _nnls(A, b)
        proj_nnls = A @ amps_nnls
        cos_nnls = float(
            np.dot(proj_nnls, b)
            / (np.linalg.norm(proj_nnls) * np.linalg.norm(b) + 1e-30)
        )
        logger.info(
            f"  NNLS: cosine={cos_nnls:.6f}, nonzero={np.sum(amps_nnls > 1e-6)}/{n_gauss}"
        )
    except Exception:
        amps_nnls = np.maximum(amps_lsq, 0)
        cos_nnls = 0.0

    # Use whichever is better
    if cos_nnls >= cos_lsq:
        best_amps_np = amps_nnls
        best_score = cos_nnls
    else:
        best_amps_np = amps_lsq
        best_score = cos_lsq

    best_amps = torch.from_numpy(best_amps_np.astype(np.float32)).to(device)
    best_centers = fixed_centers

    # ------------------------------------------------------------------
    # Optional Adam refinement stage: amplitude fine-tuning, and/or free
    # per-Gaussian isotropic width, and/or the rigid tilt/psi/dy transform
    # that aligns the canonical (untilted, centered) assembly with the
    # observed projection.  Any nonzero tilt/psi/dy is applied here even
    # without refinement, since the closed-form solve above ignores them
    # entirely (they never entered the A-matrix construction).
    # ------------------------------------------------------------------
    refine_width = algorithm.get("refine_width", False)
    refine_tilt_psi_dy = algorithm.get("refine_tilt_psi_dy", False)
    apply_rigid = (
        refine_tilt_psi_dy
        or abs(tilt_degree) > 1e-6
        or abs(psi_degree) > 1e-6
        or abs(dy_pixel) > 1e-6
    )
    run_adam_stage = fine_tune or refine_width or apply_rigid

    tilt_t = torch.tensor(float(tilt_degree), device=device, dtype=torch.float32)
    psi_t = torch.tensor(float(psi_degree), device=device, dtype=torch.float32)
    dy_t = torch.tensor(float(dy_pixel), device=device, dtype=torch.float32)

    if run_adam_stage:
        log_amps = (
            torch.log(
                torch.from_numpy(np.maximum(best_amps_np, 1e-8).astype(np.float32))
            )
            .to(device)
            .detach()
            .requires_grad_(True)
        )
        log_sigma = torch.log(fixed_sigma.clone()).detach().requires_grad_(refine_width)
        tilt_t = tilt_t.detach().requires_grad_(refine_tilt_psi_dy)
        psi_t = psi_t.detach().requires_grad_(refine_tilt_psi_dy)
        dy_t = dy_t.detach().requires_grad_(refine_tilt_psi_dy)

        opt_main = torch.optim.Adam([log_amps], lr=fine_tune_lr)
        opt_width = (
            torch.optim.Adam(
                [log_sigma], lr=algorithm.get("width_lr", fine_tune_lr * 0.5)
            )
            if refine_width
            else None
        )
        opt_rigid = (
            torch.optim.Adam([tilt_t, psi_t, dy_t], lr=algorithm.get("rigid_lr", 0.15))
            if refine_tilt_psi_dy
            else None
        )
        rigid_warmup_iter = algorithm.get("rigid_warmup_iter", fine_tune_iter // 6)

        # tilt/psi/dy refinement is a *local* correction (gradient descent
        # from the given starting value), not a blind global search -- same
        # assumption the lsq solver's Gauss-Newton refine_tilt_psi_dy() makes
        # with its own small default bounds/step sizes.  Bounds default to a
        # realistic residual-misalignment range; a caller expecting to
        # correct a large misalignment should pass a closer initial guess
        # via tilt_degree/psi_degree/dy_pixel rather than rely on this
        # optimizer finding it from zero (verified empirically: starting at
        # zero with 20 deg/15 deg ground truth converges to the wrong sign).
        tilt_max = algorithm.get("tilt_max_degree", 15.0)
        psi_max = algorithm.get("psi_max_degree", 15.0)
        dy_max = algorithm.get("dy_max_pixel", proj_ny * apix2d * 0.25)
        sigma_min = grid_sigma * algorithm.get("sigma_min_frac", 0.3)
        sigma_max = grid_sigma * algorithm.get("sigma_max_frac", 3.0)

        twist_t = torch.tensor(twist_degree, device=device)
        rise_t = torch.tensor(rise_angstrom, device=device)

        best_loss = 1e9
        best_iter = 0
        best_state = None
        for iteration in range(fine_tune_iter):
            amps = torch.exp(log_amps)
            sigmas = torch.exp(log_sigma) if refine_width else fixed_sigma

            exp_amps, exp_centers, exp_sigmas = _fixed_helical_expand(
                amps, fixed_centers, sigmas, twist_t, rise_t, csym, n_repeats
            )
            exp_centers = _apply_tilt_psi_dy(exp_centers, tilt_t, psi_t, dy_t)
            proj = _project_gaussians(
                exp_amps, exp_centers, exp_sigmas, proj_ny, proj_nz, apix2d
            )
            cos_sim = F.cosine_similarity(proj.flatten(), target.flatten(), dim=0)
            loss = 1 - cos_sim + 1e-6 * (amps**2).mean()

            opt_main.zero_grad()
            if opt_width is not None:
                opt_width.zero_grad()
            if opt_rigid is not None:
                opt_rigid.zero_grad()
            loss.backward()
            opt_main.step()
            if opt_width is not None:
                opt_width.step()
            if opt_rigid is not None and iteration >= rigid_warmup_iter:
                opt_rigid.step()

            with torch.no_grad():
                tilt_t.clamp_(-tilt_max, tilt_max)
                psi_t.clamp_(-psi_max, psi_max)
                dy_t.clamp_(-dy_max, dy_max)
                if refine_width:
                    log_sigma.clamp_(math.log(sigma_min), math.log(sigma_max))

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_iter = iteration
                best_state = (
                    amps.detach().clone(),
                    sigmas.detach().clone(),
                    tilt_t.detach().clone(),
                    psi_t.detach().clone(),
                    dy_t.detach().clone(),
                )
            if early_stop_patience > 0 and iteration - best_iter >= early_stop_patience:
                break

        best_amps, fixed_sigma, tilt_t, psi_t, dy_t = best_state
        logger.info(
            f"  Adam refinement (width={refine_width}, tilt/psi/dy={refine_tilt_psi_dy}): "
            f"{best_iter+1} iters, tilt={float(tilt_t):.3f}° psi={float(psi_t):.3f}° "
            f"dy={float(dy_t):.3f}px"
        )
        if refine_tilt_psi_dy:
            near_bound = [
                name
                for name, val, bound in (
                    ("tilt", float(tilt_t), tilt_max),
                    ("psi", float(psi_t), psi_max),
                    ("dy", float(dy_t), dy_max),
                )
                if abs(val) > 0.9 * bound
            ]
            if near_bound:
                logger.warning(
                    f"  tilt/psi/dy refinement hit its bound for {near_bound} -- "
                    "this usually means the true misalignment is larger than the "
                    "starting guess plus the configured *_max_degree/dy_max_pixel "
                    "range, and the optimizer has not actually converged. Consider "
                    "passing a closer initial tilt_degree/psi_degree/dy_pixel or "
                    "widening the bounds."
                )
        gauss_fixed_reconstruct._refined_params = {
            "tilt": float(tilt_t),
            "psi": float(psi_t),
            "dy": float(dy_t),
        }

    # Re-evaluate with the chosen score metric.  Always re-evaluate (even for
    # the default "cosine" metric) when the Adam stage ran: best_score above
    # was set from the closed-form solve and is stale once amplitudes/width/
    # tilt/psi/dy have since been refined.
    if score_metric != "cosine" or run_adam_stage:
        exp_amps, exp_centers, exp_sigmas = _fixed_helical_expand(
            best_amps,
            best_centers,
            fixed_sigma,
            torch.tensor(twist_degree, device=device),
            torch.tensor(rise_angstrom, device=device),
            csym,
            n_repeats,
        )
        exp_centers = _apply_tilt_psi_dy(exp_centers, tilt_t, psi_t, dy_t)
        bp = (
            _project_gaussians(
                exp_amps, exp_centers, exp_sigmas, proj_ny, proj_nz, apix2d
            )
            .cpu()
            .numpy()
        )
        tg = target.cpu().numpy()
        if score_metric == "cosine":
            best_score = helicon.cosine_similarity(bp.ravel(), tg.ravel())
        elif score_metric == "ssim":
            best_score = helicon.ssim_score(bp, tg)
        elif score_metric == "ms_ssim":
            best_score = helicon.ms_ssim_score(bp, tg)
        elif score_metric == "mutual_information":
            best_score = helicon.mutual_information_score(bp, tg)
        elif score_metric == "composite":
            best_score = float(
                np.mean(
                    [
                        helicon.cosine_similarity(bp.ravel(), tg.ravel()),
                        helicon.ssim_score(bp, tg),
                        helicon.ms_ssim_score(bp, tg),
                        helicon.mutual_information_score(bp, tg),
                    ]
                )
            )
    logger.info(f"  Final score ({score_metric}): {best_score:.6f}")

    # Build 3D volume from helically-expanded Gaussians
    exp_amps, exp_centers, exp_sigmas = _fixed_helical_expand(
        best_amps,
        best_centers,
        fixed_sigma,
        torch.tensor(twist_degree, device=device),
        torch.tensor(rise_angstrom, device=device),
        csym,
        n_repeats,
    )

    nz3 = reconstruct_length_3d_pixel
    ny3 = nx3 = reconstruct_diameter_3d_pixel
    z_vals = (torch.arange(nz3, device=device, dtype=torch.float32) - nz3 // 2) * apix3d
    y_vals = (torch.arange(ny3, device=device, dtype=torch.float32) - ny3 // 2) * apix3d
    x_vals = (torch.arange(nx3, device=device, dtype=torch.float32) - nx3 // 2) * apix3d
    Z, Y, X = torch.meshgrid(z_vals, y_vals, x_vals, indexing="ij")
    grid_3d = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

    vol = torch.zeros(nz3 * ny3 * nx3, device=device, dtype=torch.float32)
    vol_batch = 100
    M = exp_amps.shape[0]
    for i in range(0, M, vol_batch):
        diff = grid_3d.unsqueeze(1) - exp_centers[i : i + vol_batch].unsqueeze(0)
        exponent = (
            -0.5 * (diff / exp_sigmas[i : i + vol_batch].unsqueeze(0).unsqueeze(2)) ** 2
        )
        vol += torch.sum(
            exp_amps[i : i + vol_batch].unsqueeze(0)
            * torch.exp(torch.sum(exponent, dim=-1)),
            dim=-1,
        )
    best_rec3d = vol.reshape(nz3, ny3, nx3).cpu().numpy().astype(np.float32)

    return (best_rec3d, None, None), float(best_score)
