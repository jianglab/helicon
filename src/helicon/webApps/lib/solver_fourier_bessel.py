"""Fourier-Bessel solver for helical reconstruction.

Replaces the Gaussian-based solver with a helix-native Fourier-Bessel basis:
the helical symmetry is diagonal in FB space, so twist/rise shifts spectral
weight between layer lines naturally. The forward model evaluates the FB sum
directly on the 2D Fourier slice — no 3D voxel grid, no rasterization.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.special import jv

import helicon

logger = logging.getLogger(__name__)

# PyTorch is an optional dependency (no Intel-mac wheels exist for the project's
# supported Python versions).  Guard the import so the rest of the package can
# still be imported; the fb model only runs when torch is present.
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
            "The fb solver requires PyTorch. "
            "Install torch in the active environment to use it."
        )
    return torch


# Each worker runs one twist/rise combination in a ThreadPoolExecutor.
# torch.set_num_threads(1) ensures N workers do not oversubscribe N CPUs.
# This is the correct setting for the parallel grid search — see
# app.py:2523 (ThreadPoolExecutor with max_workers == cpu).
if TORCH_AVAILABLE:
    torch.set_num_threads(1)


def _to_device(verbose=0):
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Fourier-Bessel basis helpers
# ---------------------------------------------------------------------------


def _fb_linear_interp(
    x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
    """1D linear interpolation matching numpy.interp semantics.

    Parameters
    ----------
    x : Tensor (N,)
        Points at which to evaluate the interpolant.
    xp : Tensor (M,)
        Sorted 1-D array of sample points (must be increasing).
    fp : Tensor (M,)
        Sample values corresponding to *xp*.

    Returns
    -------
    Tensor (N,)
        Interpolated values, with extrapolation clamped to edge values.
    """
    # Broadcast shapes: x (N,), xp (M,), fp (M,)
    # Find the insertion indices
    indices = torch.searchsorted(xp, x)  # (N,)
    indices = indices.clamp(1, len(xp) - 1)

    x_left = xp[indices - 1]
    x_right = xp[indices]
    f_left = fp[indices - 1]
    f_right = fp[indices]

    t = (x - x_left) / (x_right - x_left + 1e-12)
    return f_left + t * (f_right - f_left)


def _fb_helical_selection(
    n_max: int,
    l_max: int,
    twist_deg: float,
    rise_A: float,
) -> list[tuple[int, int, float]]:
    """Compute active (n, l) layer lines from Klug-Crick-Wyckoff selection rule.

    For a 1-start helix with twist *twist_deg* and rise *rise_A*, the
    helical symmetry requires that the 3D Fourier transform is non-zero only
    when  n * twist_deg + l * twist_deg  is a multiple of 360 (the Klug-Crick
    selection rule).  Equivalently,  (n + l)  must be a multiple of
    ``360 / twist_deg``.  Within the band limits *n_max*, *l_max* this
    effectively selects  l = -n  for small twists.

    Parameters
    ----------
    n_max : int
        Maximum Bessel order |n| to consider.
    l_max : int
        Maximum layer-line index |l| to consider.
    twist_deg : float
        Helical twist in degrees.
    rise_A : float
        Helical rise in angstroms (used only for pitch computation).

    Returns
    -------
    list of (int, int, float)
        List of ``(n, l, k_z)`` tuples, where ``k_z = 2*pi * l / pitch``
        is the spatial frequency along the helical axis.
    """
    if abs(twist_deg) < 1e-12:
        return []

    pitch = rise_A * 360.0 / abs(twist_deg)  # angstroms
    # 360 / twist_deg is the number of subunits per turn
    n_per_turn = 360.0 / abs(twist_deg)
    active: list[tuple[int, int, float]] = []

    for n in range(-n_max, n_max + 1):
        for l_val in range(-l_max, l_max + 1):
            # Selection rule: (n + l) must be a multiple of n_per_turn.
            # Guard against floating-point fudge near zero.
            s = n + l_val
            # Check if s is approximately a multiple of n_per_turn
            if n_per_turn > 1e-6:
                ratio = s / n_per_turn
                if abs(ratio - round(ratio)) > 1e-6:
                    continue
            elif abs(s) > 1e-6:
                continue
            k_z = 2.0 * math.pi * l_val / pitch
            active.append((n, l_val, k_z))

    return active


def _fb_precompute_constants(
    grid_ky: torch.Tensor,
    grid_kz: torch.Tensor,
    active_modes: list[tuple[int, int, float]],
    layer_line_width: float | None = None,
    helix_length: float | None = None,
) -> dict:
    """Precompute invariant tensors for the FB slice evaluation loop.

    The layer-line envelope accounts for finite helix length: a short
    helix segment is the infinite helix × rectangular window, so each
    layer line is broadened from a delta into a sinc(kz·L/2π) shape
    (convolution theorem).

    Parameters
    ----------
    grid_ky : Tensor
        Shape ``(H, W)``.  Y-Fourier coordinates for each pixel.
    grid_kz : Tensor
        Shape ``(H, W)``.  Z-Fourier coordinates for each pixel.
    active_modes : list of (int, int, float)
        ``(n, l, k_z)`` from ``_fb_helical_selection``.
    layer_line_width : float, optional
        Override: Gaussian sigma for the layer-line envelope (rad/Å).
        If None, use sinc broadening based on helix_length.
    helix_length : float, optional
        Physical length of the helix segment in Angstroms.
        Used to compute sinc broadening: first zero at 2π/L.

    Returns
    -------
    dict with keys ``R_flat``, ``n_pixels``, ``H``, ``W``,
    ``kz_weights``, ``phases`` (each ``(n_active, H*W)``).
    """
    device = grid_ky.device
    H, W = grid_ky.shape
    n_pixels = H * W

    # Radial grid (flattened once)
    # The 2D Fourier slice is at kx=0 (projection along x).
    # In cylindrical coords: kR = |ky|, phi = pi/2 * sign(ky).
    R_flat = torch.abs(grid_ky).reshape(-1)  # (H*W,)
    ky_sign = torch.sign(grid_ky).reshape(-1)  # (H*W,)

    # Per-mode factors stacked into batched tensors (n_active, H*W)
    n_active = len(active_modes)
    kz_flat = grid_kz.reshape(-1)  # (H*W,)

    kz_nominal_v = torch.tensor(
        [kz for _, _, kz in active_modes], device=device
    ).reshape(
        -1, 1
    )  # (n_active, 1)
    dz = kz_flat.unsqueeze(0) - kz_nominal_v  # (n_active, H*W)

    if layer_line_width is not None:
        # Explicit Gaussian width override
        sigma_kz = layer_line_width
        kz_weights = torch.exp(-0.5 * (dz / sigma_kz) ** 2)
    elif helix_length is not None and helix_length > 0:
        # Sinc broadening from finite helix length:
        # w(z) = rect(z/L)  -->  W(kz) = L * sinc(kz*L / (2π))
        # sinc(x) = sin(πx)/(πx), normalized so W(0) = 1.
        # Use |sinc| to avoid sign-flip artifacts in the envelope.
        x = dz * helix_length / (2.0 * math.pi)  # dimensionless
        # sinc(x) = sin(πx)/(πx); handle x→0 limit
        sinc_vals = torch.where(
            x.abs() < 1e-8,
            torch.ones_like(x),
            torch.sin(math.pi * x) / (math.pi * x),
        )
        kz_weights = sinc_vals.abs()  # (n_active, H*W)
        logger.info(
            f"  Layer-line envelope: sinc broadening, L={helix_length:.0f} Å"
            f" (first zero at kz={2*math.pi/helix_length:.4f} rad/Å)"
        )
    else:
        # Fallback: Gaussian based on layer line spacing
        kz_vals = sorted({abs(kz) for _, _, kz in active_modes if abs(kz) > 1e-12})
        sigma_kz = min(kz_vals) * 0.5 if kz_vals else float("inf")
        if sigma_kz < float("inf"):
            kz_weights = torch.exp(-0.5 * (dz / sigma_kz) ** 2)
        else:
            kz_weights = torch.ones(n_active, n_pixels, device=device)

    n_v = torch.tensor([n for n, _, _ in active_modes], device=device).reshape(-1, 1)
    # At kx=0: phi = pi/2 * sign(ky), so phase = exp(i*n*pi/2*sign(ky))
    phases = torch.exp(
        1j * n_v * (math.pi / 2) * ky_sign.unsqueeze(0)
    )  # (n_active, H*W)

    return dict(
        R_flat=R_flat,
        n_pixels=n_pixels,
        H=H,
        W=W,
        kz_weights=kz_weights,
        phases=phases,
    )


def _fb_evaluate_slice(
    V_coeffs: torch.Tensor,
    k_radial: torch.Tensor,
    constants: dict,
) -> torch.Tensor:
    """Evaluate F_2D(ky, kz) from FB coefficients (batched, no Python loop).

    Uses precomputed phase and layer-line factors from *constants*.
    The radial interpolation is batched across all active modes so
    PyTorch can parallelise across modes via SIMD / internal threads.

    Parameters
    ----------
    V_coeffs : Tensor
        Shape ``(n_active, n_radial)``.  Radial samples of the FB coefficients.
    k_radial : Tensor
        Shape ``(n_radial,)``.  Radial sampling points in Fourier space.
    constants : dict
        Precomputed invariants from ``_fb_precompute_constants``.

    Returns
    -------
    Tensor
        Complex ``(H, W)`` Fourier slice.
    """
    R_flat = constants["R_flat"]
    kz_weights = constants["kz_weights"]
    phases = constants["phases"]
    H, W = constants["H"], constants["W"]

    n_active, n_radial = V_coeffs.shape
    n_pixels = R_flat.shape[0]

    # Batched radial interpolation: V_coeffs (n_active, n_radial)
    # evaluated at R_flat (n_pixels,) giving V_interp (n_active, n_pixels).
    R_clamped = R_flat.clamp(0.0, k_radial[-1])  # (n_pixels,)
    R_batch = (
        R_clamped.unsqueeze(0).expand(n_active, -1).contiguous()
    )  # (n_active, n_pixels)

    indices = torch.searchsorted(k_radial, R_batch)  # (n_active, n_pixels)
    indices = indices.clamp(1, n_radial - 1)

    x_left = k_radial[indices - 1]  # (n_active, n_pixels)
    x_right = k_radial[indices]
    f_left = torch.gather(V_coeffs, 1, (indices - 1).clamp(0, n_radial - 1))
    f_right = torch.gather(V_coeffs, 1, indices.clamp(0, n_radial - 1))

    t = (R_batch - x_left) / (x_right - x_left + 1e-12)
    V_interp = f_left + t * (f_right - f_left)  # (n_active, n_pixels)

    # Reduce over modes: sum (V * phase * kz_weight) → (n_pixels,)
    F_flat = (V_interp * phases * kz_weights).sum(dim=0)
    return F_flat.reshape(H, W)


# ---------------------------------------------------------------------------
# Hankel transform (Fourier-Bessel to real-space conversion)
# ---------------------------------------------------------------------------


def _hankel_transform_matrix(kR_grid, r_grid, n):
    """Build the nth-order Hankel transform matrix.

    Computes H[j,i] = J_n(2π kR_i r_j) · 2π kR_i · dkR_i so that
    ``rho = H @ V_kR`` evaluates:

    ``ρ_n(r) = ∫ V_n(kR) J_n(2π kR r) 2π kR dkR``

    Parameters
    ----------
    kR_grid : ndarray (n_kR,)
        Radial frequency grid (ascending, non-negative).
    r_grid : ndarray (n_r,)
        Real-space radial grid (ascending, non-negative).
    n : int
        Bessel order.

    Returns
    -------
    ndarray (n_r, n_kR)
        Transform matrix.
    """
    n_kR = len(kR_grid)

    # Trapezoidal quadrature weights for dkR
    dkR = np.zeros(n_kR, dtype=np.float64)
    if n_kR == 1:
        dkR[0] = kR_grid[0] if kR_grid[0] > 0 else 1.0
    else:
        dkR[0] = 0.5 * (kR_grid[1] - kR_grid[0])
        dkR[-1] = 0.5 * (kR_grid[-1] - kR_grid[-2])
        if n_kR > 2:
            dkR[1:-1] = 0.5 * (kR_grid[2:] - kR_grid[:-2])

    # 2π kR_i weights
    kr = 2.0 * np.pi * kR_grid  # (n_kR,)

    # J_n(2π kR_i r_j) — shape (n_r, n_kR)
    arg = np.outer(r_grid, kr)
    J = jv(n, arg)

    # Transform matrix
    H = J * (kr[np.newaxis, :] * dkR[np.newaxis, :])
    return H


def _hankel_transform_coeffs(V_coeffs_np, k_radial_np, r_grid, active_modes):
    """Compute real-space radial profiles for all modes via Hankel transform.

    Parameters
    ----------
    V_coeffs_np : ndarray (n_active, n_kR), complex
        FB coefficients in Fourier space.
    k_radial_np : ndarray (n_kR,)
        Radial frequency grid.
    r_grid : ndarray (n_r,)
        Real-space radial grid.
    active_modes : list of (n, l, kz)
        Active helical modes.

    Returns
    -------
    ndarray (n_active, n_r), complex
        Real-space radial profiles ρ_n(r).
    """
    n_active = len(active_modes)
    n_r = len(r_grid)
    rho = np.zeros((n_active, n_r), dtype=np.complex128)

    # Cache transform matrices by Bessel order (many modes share same |n|)
    H_cache = {}
    for idx, (n, l_val, kz_l) in enumerate(active_modes):
        if n not in H_cache:
            H_cache[n] = _hankel_transform_matrix(k_radial_np, r_grid, n)
        H = H_cache[n]
        # rho_n(r) = H @ V_n(kR)
        rho[idx, :] = H @ V_coeffs_np[idx, :]

    return rho


# ---------------------------------------------------------------------------
# Differentiable 3D volume construction + helical expansion
# ---------------------------------------------------------------------------


def _build_fb_volume_differentiable(
    best_V_coeffs_t,
    k_radial_t,
    active_modes,
    r_grid_np,
    nz3,
    ny3,
    nx3,
    apix3d,
    pitch,
    device,
):
    """Build 3D volume from FB coefficients (differentiable, torch tensors).

    Parameters
    ----------
    best_V_coeffs_t : Tensor (n_active, n_kR), complex
        FB coefficients.
    k_radial_t : Tensor (n_kR,)
        Radial frequency grid.
    active_modes : list of (n, l, kz)
        Active helical modes.
    r_grid_np : ndarray (n_r,)
        Real-space radial grid.
    nz3, ny3, nx3 : int
        Volume dimensions.
    apix3d : float
        3D voxel size.
    pitch : float
        Helical pitch in Angstroms.
    device : str
        Torch device.

    Returns
    -------
    Tensor (nz3, ny3, nx3)
        3D density volume.
    """
    R_max_3d = nx3 * apix3d / 2.0
    n_r = len(r_grid_np)
    r_grid_t = torch.from_numpy(r_grid_np).float().to(device)
    k_radial_np = k_radial_t.detach().cpu().numpy()

    z_vals = (torch.arange(nz3, device=device).float() - nz3 // 2) * apix3d
    y_vals = (torch.arange(ny3, device=device).float() - ny3 // 2) * apix3d
    x_vals = (torch.arange(nx3, device=device).float() - nx3 // 2) * apix3d
    Z_t, Y_t, X_t = torch.meshgrid(z_vals, y_vals, x_vals, indexing="ij")
    R_3d_t = torch.sqrt(X_t**2 + Y_t**2)
    Phi_3d_t = torch.atan2(Y_t, X_t)

    # Hankel transform matrices (cached by Bessel order)
    H_cache_t = {}

    volume = torch.zeros(nz3, ny3, nx3, dtype=torch.float32, device=device)

    for idx, (n, l_val, kz_l) in enumerate(active_modes):
        kz_l_phys = 2.0 * math.pi * l_val / pitch

        # Hankel transform: V_n(kR) → ρ_n(r)
        if n not in H_cache_t:
            H_np = _hankel_transform_matrix(k_radial_np, r_grid_np, n)
            H_cache_t[n] = torch.from_numpy(H_np).float().to(device)
        H = H_cache_t[n]
        rho_n = H @ best_V_coeffs_t[idx].real  # simplified: use real part
        rho_n_imag = H @ best_V_coeffs_t[idx].imag

        # Interpolate ρ_n(r) at each voxel's radial coordinate
        R_clamped = R_3d_t.clamp(0.0, R_max_3d)
        indices = torch.searchsorted(r_grid_t, R_clamped.reshape(-1))
        indices = indices.clamp(1, n_r - 1)
        left = (indices - 1).clamp(0, n_r - 1)
        t = (R_clamped.reshape(-1) - r_grid_t[left]) / (
            r_grid_t[indices] - r_grid_t[left] + 1e-12
        )
        rho_real_interp = rho_n[left] + t * (rho_n[indices] - rho_n[left])
        rho_imag_interp = rho_n_imag[left] + t * (
            rho_n_imag[indices] - rho_n_imag[left]
        )
        rho_3d = (rho_real_interp + 1j * rho_imag_interp).reshape(nz3, ny3, nx3)

        # Angular and axial phases
        phase = torch.exp(1j * (n * Phi_3d_t + kz_l_phys * Z_t))
        volume = volume + (rho_3d * phase).real.float()

    return volume


def _differentiable_helical_expand(volume, twist_rad, rise_voxels, n_repeats):
    """Expand 3D volume via helical symmetry (differentiable, wrap-around).

    Uses ``F.grid_sample`` for rotation and shift, enabling backprop.

    Parameters
    ----------
    volume : Tensor (nz, ny, nx)
        Unit cell volume.
    twist_rad : float
        Twist angle in radians.
    rise_voxels : float
        Rise in voxels.
    n_repeats : int
        Number of helical repeats on each side.

    Returns
    -------
    Tensor (nz, ny, nx)
        Expanded volume (wrap-around summation).
    """
    nz, ny, nx = volume.shape
    device = volume.device

    z = torch.linspace(-1, 1, nz, device=device)
    y = torch.linspace(-1, 1, ny, device=device)
    x = torch.linspace(-1, 1, nx, device=device)
    Z, Y, X = torch.meshgrid(z, y, x, indexing="ij")

    expanded = volume

    for m in range(-n_repeats, n_repeats + 1):
        if m == 0:
            continue
        angle = m * twist_rad
        shift_z = m * rise_voxels * 2.0 / max(nz - 1, 1)

        cos_a = torch.cos(torch.tensor(angle, device=device))
        sin_a = torch.sin(torch.tensor(angle, device=device))

        x_in = X * cos_a + Y * sin_a
        y_in = -X * sin_a + Y * cos_a
        z_in = Z - shift_z

        grid = torch.stack([z_in, y_in, x_in], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            volume.unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        expanded = expanded + sampled.squeeze(0).squeeze(0)

    return expanded


def _projection_cosine_score(volume_3d, target_2d_t):
    """Compute cosine similarity between x-projection and target (differentiable).

    Parameters
    ----------
    volume_3d : Tensor (nz, ny, nx)
        3D density volume.
    target_2d_t : Tensor (ny_target, nz_target)
        Target 2D projection image.

    Returns
    -------
    Tensor scalar
        Cosine similarity.
    """
    proj = volume_3d.sum(dim=2).T  # (ny, nz) → transpose → (ny, nz)

    # Resize to match target if needed
    if proj.shape != target_2d_t.shape:
        proj = (
            F.interpolate(
                proj.unsqueeze(0).unsqueeze(0),
                size=target_2d_t.shape,
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )

    # Cosine similarity
    proj_flat = proj.reshape(-1)
    tgt_flat = target_2d_t.reshape(-1)
    cos_sim = torch.dot(proj_flat, tgt_flat) / (
        torch.norm(proj_flat) * torch.norm(tgt_flat) + 1e-12
    )
    return cos_sim


# ---------------------------------------------------------------------------
# FB reconstruct solver
# ---------------------------------------------------------------------------


def fb_reconstruct(
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
    algorithm=dict(model="fb"),
):
    """Fourier-Bessel reconstruction of a helical structure from a 2D image.

    Unlike the Gaussian solvers, this method works directly in Fourier space:
    it optimises the FB coefficients so that the Fourier slice of the helical
    structure best matches the FFT of the input projection image.

    Parameters
    ----------
    projection_image : ndarray (ny, nx)
        Input 2D projection image.
    scale2d_to_3d : float
        Ratio ``target_apix2d / target_apix3d``.
    twist_degree : float
        Helical twist in degrees.
    rise_pixel : float
        Helical rise in voxels (3D grid units).
    … (same signature as ``gauss_fixed_reconstruct``)

    Algorithm dict keys
    -------------------
    fb_n_max : int
        Maximum Bessel order |n| (default 4).  Recommended ≥4 for twist
        discrimination on amyloid-class data (~1° twist).
    fb_l_max : int
        Maximum layer-line index |l| (default 4).
    fb_n_radial : int
        Number of radial sample points per ``(n, l)`` mode (default 12).
    fb_iter : int
        Number of Adam optimisation iterations (default 500).
    fb_lr : float
        Adam learning rate (default 0.01).
    fb_loss_type : str
        Loss function: ``"fourier"`` (cosine similarity in Fourier space,
        default) or ``"realspace"`` (MSE in real-space via IFFT).

    Returns
    -------
    (rec3d, None, None), score
        Same format as ``gauss_fixed_reconstruct``.
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

    rise_angstrom = rise_pixel * apix3d

    fb_n_max = algorithm.get("fb_n_max", 4)
    fb_l_max = algorithm.get("fb_l_max", 4)
    fb_n_radial = algorithm.get("fb_n_radial", 12)
    fb_iter = algorithm.get("fb_iter", 500)
    fb_lr = algorithm.get("fb_lr", 0.01)
    fb_layer_line_width = algorithm.get(
        "fb_layer_line_width", None
    )  # None = auto (min(kz)*0.5)
    fb_loss_type = algorithm.get(
        "fb_loss_type", "fourier"
    )  # "fourier" (cosine in F-space) or "realspace" (MSE in real-space)

    pitch_a = (
        rise_angstrom * 360.0 / abs(twist_degree)
        if abs(twist_degree) > 1e-12 and abs(rise_angstrom) > 1e-12
        else 0.0
    )

    # ------------------------------------------------------------------
    # NOTE: auto-raise removed. For finite-length helices, the sinc
    # broadening relaxes the selection rule, so we include ALL (n,l)
    # modes within the user's n_max/l_max range — not just the strict
    # selection-rule subset. The sinc envelope naturally suppresses modes
    # that don't contribute at the given twist.
    # ------------------------------------------------------------------
    active_modes = []
    for n in range(-fb_n_max, fb_n_max + 1):
        for l_val in range(-fb_l_max, fb_l_max + 1):
            kz_l = 2.0 * math.pi * l_val / pitch_a
            active_modes.append((n, l_val, kz_l))

    logger.info(
        f"FB solver: n_max={fb_n_max}, l_max={fb_l_max}, "
        f"n_radial={fb_n_radial}, {fb_iter} iter, lr={fb_lr}, "
        f"loss={fb_loss_type}, "
        f"proj={proj_ny}x{proj_nz} @ {apix2d:.3f}Å, "
        f"rise={rise_angstrom:.3f}Å, twist={twist_degree:.3f}°, pitch={pitch_a:.0f}Å"
    )

    device = algorithm.get("device", _to_device(verbose))

    # ------------------------------------------------------------------
    # 1. Fourier-transform target image (once, at start)
    # ------------------------------------------------------------------
    target_np = projection_image.astype(np.float32)
    target_t = torch.tensor(target_np, dtype=torch.float32, device=device)
    target_fft = torch.fft.fft2(target_t)
    # FFT2 of (ny, nx) image gives (ny, nx) where axis 0 = vertical freq (ky),
    # axis 1 = horizontal freq (kz).  The FB grid is (nz, ny) = (nx, ny) with
    # axis 0 = kz and axis 1 = ky.  Transpose to align dimensions.
    target_fft = target_fft.T  # (nx, ny) = (nz, ny) — matches FB grid
    target_fft = target_fft / math.sqrt(target_fft.numel())  # normalize

    # ------------------------------------------------------------------
    # 2. Build 2D Fourier grid (once)
    # ------------------------------------------------------------------
    # Frequency ranges (radians per angstrom)
    ky = 2.0 * math.pi * torch.fft.fftfreq(proj_ny, d=apix2d, device=device)
    kz = 2.0 * math.pi * torch.fft.fftfreq(proj_nz, d=apix2d, device=device)
    grid_kz, grid_ky = torch.meshgrid(kz, ky, indexing="ij")  # (nz, ny)

    # Radial Nyquist
    k_max = 2.0 * math.pi * 0.5 / apix2d  # radians per angstrom
    k_radial = torch.linspace(0.0, k_max, fb_n_radial, device=device)

    # ------------------------------------------------------------------
    # 3. Compute helical selection rule
    # ------------------------------------------------------------------
    active_modes = _fb_helical_selection(
        fb_n_max, fb_l_max, twist_degree, rise_angstrom
    )
    n_active = len(active_modes)
    if n_active == 0:
        nz3 = reconstruct_length_3d_pixel
        ny3 = nx3 = reconstruct_diameter_3d_pixel
        return (
            np.zeros((nz3, ny3, nx3), dtype=np.float32),
            None,
            None,
        ), 0.0

    # For logging: show modes
    if verbose:
        mode_str = ", ".join(f"(n={n}, l={l})" for n, l, _ in active_modes)
        logger.info(f"  Active modes ({n_active}): {mode_str}")

    # ------------------------------------------------------------------
    # 4b. Precompute invariant Fourier-domain tensors (once)
    # ------------------------------------------------------------------
    # Helix length in the image = horizontal extent (along helical axis)
    helix_length_A = proj_nz * apix2d  # Angstroms
    fb_constants = _fb_precompute_constants(
        grid_ky,
        grid_kz,
        active_modes,
        layer_line_width=fb_layer_line_width,
        helix_length=helix_length_A,
    )

    # Precompute target flat vector for Fourier scoring (used by both loss
    # types in the final evaluation, and needed for the Fourier loss branch).
    F_target_flat = target_fft.reshape(-1)
    norm_target = torch.sqrt((F_target_flat.conj() * F_target_flat).real.sum() + 1e-12)

    # ------------------------------------------------------------------
    # 5. Solve for FB coefficients via linear system at each ky
    # ------------------------------------------------------------------
    # Forward model at each ky:
    #   F_target(ky, kz_j) = Σ_n V_n(|ky|) × phase_n(ky) × sinc(L·(kz_j - kz_l(n))/2π)
    # For fixed ky, phase_n(ky) is known → solve for V_n(|ky|).
    ky_1d = ky  # (ny,) frequencies
    kz_1d = kz  # (nz,) frequencies

    best_V_coeffs = torch.zeros(
        n_active, fb_n_radial, dtype=torch.cfloat, device=device
    )

    # Build the sinc weight matrix: A[j, mode_idx] = |sinc(L·(kz_j - kz_l)/2π)|
    n_kz = len(kz_1d)
    A_np = np.zeros((n_kz, n_active), dtype=np.float64)
    for mode_idx, (n, l_val, k_z) in enumerate(active_modes):
        kz_l = 2.0 * math.pi * l_val / pitch_a
        dz = kz_1d.cpu().numpy() - kz_l
        x = dz * helix_length_A / (2.0 * math.pi)
        # sinc(x) = sin(πx)/(πx)
        with np.errstate(divide="ignore", invalid="ignore"):
            sinc_vals = np.where(
                np.abs(x) < 1e-10, 1.0, np.sin(np.pi * x) / (np.pi * x)
            )
        A_np[:, mode_idx] = sinc_vals  # signed sinc, not absolute value

    # Precompute damped pseudoinverse (Tikhonov regularization)
    AtA = A_np.T @ A_np  # (n_active, n_active)
    # Estimate regularization from eigenvalues
    eigvals = np.linalg.eigvalsh(AtA)
    lambda_reg = max(eigvals.max() * 1e-6, 1e-10)
    A_damped_pinv = np.linalg.solve(
        AtA + lambda_reg * np.eye(n_active), A_np.T
    )  # (n_active, n_kz)

    # Phase factors for each mode at each ky
    sign_ky_np = np.sign(ky_1d.cpu().numpy())  # (ny,)

    # Solve for each positive ky
    pos_mask = ky_1d > 0
    pos_indices = np.where(pos_mask.cpu().numpy())[0]
    ky_pos_np = ky_1d[pos_mask].cpu().numpy()

    V_all_pos = np.zeros((n_active, len(pos_indices)), dtype=np.complex128)
    target_fft_np = target_fft.cpu().numpy()  # (nz, ny) complex

    for i, ky_idx in enumerate(pos_indices):
        f = target_fft_np[:, ky_idx]  # (n_kz,) complex
        # Solve: w = A_damped_pinv @ f, then V_n = w_n / phase_n(ky)
        w = A_damped_pinv @ f  # (n_active,) complex
        phases_at_ky = np.exp(
            -1j * np.arange(n_active) * (np.pi / 2) * sign_ky_np[ky_idx]
        )
        # Map mode_idx back to n values
        for mode_idx, (n, l_val, k_z) in enumerate(active_modes):
            V_all_pos[mode_idx, i] = w[mode_idx] / (
                np.exp(-1j * n * (np.pi / 2) * sign_ky_np[ky_idx]) + 1e-30
            )

    # Interpolate each mode's radial profile from ky values to k_radial grid
    k_radial_np = k_radial.cpu().numpy()
    for mode_idx in range(n_active):
        V_profile = V_all_pos[mode_idx]
        V_interp_real = np.interp(k_radial_np, ky_pos_np, V_profile.real).astype(
            np.float32
        )
        V_interp_imag = np.interp(k_radial_np, ky_pos_np, V_profile.imag).astype(
            np.float32
        )
        best_V_coeffs[mode_idx] = torch.complex(
            torch.from_numpy(V_interp_real).to(device=device),
            torch.from_numpy(V_interp_imag).to(device=device),
        )

    # Evaluate extraction quality
    with torch.no_grad():
        F_init = _fb_evaluate_slice(best_V_coeffs, k_radial, fb_constants)
        F_init_flat = F_init.reshape(-1)
        dot_init = (F_init_flat.conj() * F_target_flat).real.sum()
        norm_init = torch.sqrt((F_init_flat.conj() * F_init_flat).real.sum() + 1e-12)
        init_cos = float((dot_init / (norm_init * norm_target + 1e-12)).item())
    logger.info(f"  Linear solve initial cosine: {init_cos:.6f}")

    # ------------------------------------------------------------------
    # Optimization: 3D projection cosine loss (twist-dependent)
    # ------------------------------------------------------------------
    # Precompute r_grid and target tensor for the 3D pipeline
    R_max_3d = reconstruct_diameter_3d_pixel * apix3d / 2.0
    n_r_hankel = max(reconstruct_diameter_3d_pixel, reconstruct_length_3d_pixel)
    r_grid_np = np.linspace(0, R_max_3d, n_r_hankel, dtype=np.float64)

    target_2d_t = torch.from_numpy(projection_image.astype(np.float32)).to(device)

    nz3 = reconstruct_length_3d_pixel
    ny3 = reconstruct_diameter_3d_pixel
    nx3 = reconstruct_diameter_3d_pixel
    rise_voxels = rise_pixel  # rise in 3D voxels
    pitch_a = (
        rise_angstrom * 360.0 / abs(twist_degree) if abs(twist_degree) > 1e-12 else 1.0
    )
    twist_rad = math.radians(twist_degree)
    n_helical_repeats = max(4, int(np.ceil(projection_image.shape[1] / nz3)) + 2)

    V_coeffs_init = best_V_coeffs.clone()
    log_amp = torch.log(V_coeffs_init.abs() + 1e-10).detach().requires_grad_(True)
    ph = torch.angle(V_coeffs_init).detach().requires_grad_(True)
    opt = torch.optim.Adam([log_amp, ph], lr=fb_lr)

    best_loss = float("inf")
    for iteration in range(fb_iter):
        amp = torch.exp(log_amp)
        V_coeffs = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))

        # Build 3D volume from FB coefficients (differentiable)
        volume_3d = _build_fb_volume_differentiable(
            V_coeffs,
            k_radial,
            active_modes,
            r_grid_np,
            nz3,
            ny3,
            nx3,
            apix3d,
            pitch_a,
            device,
        )

        # Expand via helical symmetry (differentiable)
        volume_expanded = _differentiable_helical_expand(
            volume_3d,
            twist_rad,
            rise_voxels,
            n_helical_repeats,
        )

        # Score: cosine similarity between x-projection and target
        score_3d = _projection_cosine_score(volume_expanded, target_2d_t)

        # Also include Fourier cosine as regularization for stability
        F_pred = _fb_evaluate_slice(V_coeffs, k_radial, fb_constants)
        F_pred_flat = F_pred.reshape(-1)
        dot_f = (F_pred_flat.conj() * F_target_flat).real.sum()
        norm_f = torch.sqrt((F_pred_flat.conj() * F_pred_flat).real.sum() + 1e-12)
        cos_f = dot_f / (norm_f * norm_target + 1e-12)

        # Combined loss: 3D projection cosine (primary) + Fourier cosine (regularizer)
        loss = 1.0 - score_3d + 1e-6 * (amp**2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_V_coeffs = V_coeffs.clone().detach()
            best_score_3d = float(score_3d.item())

    logger.info(f"  Final loss: {best_loss:.6f}, 3D score: {best_score_3d:.6f}")

    # ------------------------------------------------------------------
    # 6. Build 3D volume via Hankel transforms
    # ------------------------------------------------------------------
    with torch.no_grad():
        nz3 = reconstruct_length_3d_pixel
        ny3 = nx3 = reconstruct_diameter_3d_pixel

        # Real-space radial grid for the 3D volume cross-section
        R_max_3d = reconstruct_diameter_3d_pixel * apix3d / 2.0
        n_r = max(ny3, nx3)
        r_grid = np.linspace(0, R_max_3d, n_r, dtype=np.float64)

        # Compute Hankel transforms: V_n(kR) → ρ_n(r)
        rho_profiles = _hankel_transform_coeffs(
            best_V_coeffs.cpu().numpy(),
            k_radial.cpu().numpy(),
            r_grid,
            active_modes,
        )

        # 3D grid in real-space coordinates
        z_vals = (np.arange(nz3) - nz3 // 2) * apix3d
        y_vals = (np.arange(ny3) - ny3 // 2) * apix3d
        x_vals = (np.arange(nx3) - nx3 // 2) * apix3d
        Z_np, Y_np, X_np = np.meshgrid(z_vals, y_vals, x_vals, indexing="ij")
        R_3d_np = np.sqrt(X_np**2 + Y_np**2)
        Phi_3d_np = np.arctan2(Y_np, X_np)

        pitch = (
            rise_angstrom * 360.0 / abs(twist_degree)
            if abs(twist_degree) > 1e-12
            else 1.0
        )

        # Build density by summing over modes
        volume = np.zeros((nz3, ny3, nx3), dtype=np.float64)

        for idx, (n, l_val, k_z) in enumerate(active_modes):
            kz_l = 2.0 * math.pi * l_val / pitch

            # Interpolate ρ_n(r) at each voxel's radial coordinate
            rho_r = rho_profiles[idx, :]  # (n_r,) complex
            rho_real = np.interp(R_3d_np.ravel(), r_grid, rho_r.real).reshape(
                nz3, ny3, nx3
            )
            rho_imag = np.interp(R_3d_np.ravel(), r_grid, rho_r.imag).reshape(
                nz3, ny3, nx3
            )
            rho_3d = rho_real + 1j * rho_imag

            # Angular and axial phases
            phase = np.exp(1j * (n * Phi_3d_np + kz_l * Z_np))

            # Accumulate real part
            volume += (rho_3d * phase).real

        # Normalize to [0, 1]
        vmin = volume.min()
        vmax = volume.max()
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)

        best_rec3d = volume.astype(np.float32)

        # ------------------------------------------------------------------
        # 7. Score: expand via helical symmetry, project, compare with input
        # ------------------------------------------------------------------
        # The 3D volume is a small seed (nz3 voxels along helix).
        # Apply helical symmetry to expand to full-length reconstruction,
        # then project along x to get the 2D projection for scoring.
        nz_orig = projection_image.shape[1]  # nx = along helix in image
        ny_orig = projection_image.shape[0]  # ny = perpendicular
        new_length = nz_orig  # match the input image's horizontal size

        rec3d_xform = helicon.apply_helical_symmetry(
            data=best_rec3d,
            apix=apix3d,
            twist_degree=twist_degree,
            rise_angstrom=rise_angstrom,
            csym=csym,
            new_size=(new_length, ny_orig, ny_orig),
            new_apix=apix2d,
            cpu=helicon.available_cpu(),
        )
        # Project along x (axis 2) to get 2D projection
        rec3d_x_proj = np.sum(rec3d_xform, axis=2).T  # (ny_orig, new_length) = (ny, nx)

        # Rescale to match target amplitude
        tg = target_np
        s_r = np.dot(rec3d_x_proj.ravel(), tg.ravel()) / (
            np.dot(rec3d_x_proj.ravel(), rec3d_x_proj.ravel()) + 1e-30
        )
        rec3d_x_proj = rec3d_x_proj * s_r

        if score_metric == "cosine":
            best_score = helicon.cosine_similarity(rec3d_x_proj.ravel(), tg.ravel())
        elif score_metric == "ssim":
            best_score = helicon.ssim_score(rec3d_x_proj, tg)
        elif score_metric == "ms_ssim":
            best_score = helicon.ms_ssim_score(rec3d_x_proj, tg)
        elif score_metric == "mutual_information":
            best_score = helicon.mutual_information_score(rec3d_x_proj, tg)
        elif score_metric == "composite":
            best_score = float(
                np.mean(
                    [
                        helicon.cosine_similarity(rec3d_x_proj.ravel(), tg.ravel()),
                        helicon.ssim_score(rec3d_x_proj, tg),
                        helicon.ms_ssim_score(rec3d_x_proj, tg),
                        helicon.mutual_information_score(rec3d_x_proj, tg),
                    ]
                )
            )
        else:
            # Fourier cosine as fallback
            F_best = _fb_evaluate_slice(best_V_coeffs, k_radial, fb_constants)
            F_best_flat = F_best.reshape(-1)
            dot_best = (F_best_flat.conj() * F_target_flat).real.sum()
            norm_pred_best = torch.sqrt(
                (F_best_flat.conj() * F_best_flat).real.sum() + 1e-12
            )
            best_score = float(
                (dot_best / (norm_pred_best * norm_target + 1e-12)).item()
            )

        logger.info(f"  Score ({score_metric}): {best_score:.6f}")

        best_proj = rec3d_x_proj  # for debug images

        # Debug: save paired input vs reconstructed projection
        debug_dir = algorithm.get("debug_dir", "")
        if debug_dir:
            import os

            os.makedirs(debug_dir, exist_ok=True)
            tag = f"twist{twist_degree:.3f}_rise{rise_angstrom:.3f}"

            def _save_debug_img(arr, name):
                arr_norm = arr - arr.min()
                if arr_norm.max() > 0:
                    arr_norm = arr_norm / arr_norm.max()
                arr_u8 = (np.clip(arr_norm, 0, 1) * 255).astype(np.uint8)
                from PIL import Image

                Image.fromarray(arr_u8).save(
                    os.path.join(debug_dir, f"{name}_{tag}.png"),
                )
                np.save(os.path.join(debug_dir, f"{name}_{tag}.npy"), arr)

            _save_debug_img(target_np, "input")
            _save_debug_img(rec3d_x_proj, "best_proj")
            _save_debug_img(best_rec3d[best_rec3d.shape[0] // 2], "rec3d_xy_slice")
            _save_debug_img(best_rec3d.sum(axis=2).T, "rec3d_proj_x")

    return (best_rec3d, None, None), float(best_score)
