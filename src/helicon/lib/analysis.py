from __future__ import annotations

import numpy as np
import helicon

__all__ = [
    "calc_fsc",
    "calc_fsc_from_fft",
    "calc_fsc_per_shell",
    "calc_frc_2d",
    "cosine_similarity",
    "cross_correlation_coefficient",
    "estimate_helix_rotation_center_diameter",
    "find_elbow_point",
    "frc_score",
    "get_cylindrical_mask",
    "is_3d",
    "is_amyloid",
    "line_fit_projection",
    "ms_ssim_score",
    "mutual_information_score",
    "r_factor_score",
    "ssim_score",
    "twist2pitch",
    "estimate_inter_segment_distance",
    "estimate_helicalTube_length",
    "reset_inter_segment_distance",
]


def is_3d(data: np.ndarray) -> bool:
    """Check if an array represents a 3D volume (cubic or rectangular).

    Parameters
    ----------
    data : np.ndarray
        Input array.

    Returns
    -------
    bool
        True if the array is 3D and either cubic (``nz == ny == nx``) or
        rectangular (``nz > ny == nx``).
    """
    if data.ndim != 3:
        return False
    nz, ny, nx = data.shape
    if nz == ny and nz == nx:
        return True
    if nz > ny and ny == nx:
        return True
    return False


def is_amyloid(emdb_id: str) -> bool:
    """Check if an EMDB ID corresponds to an amyloid structure.

    Parameters
    ----------
    emdb_id : str
        EMDB entry identifier.

    Returns
    -------
    bool
        True if the EMDB ID is found in the amyloid atlas.
    """
    if not isinstance(emdb_id, str):
        return False
    emdb = helicon.dataset.EMDB()
    if emdb_id.split("-")[-1].split("_")[-1] in emdb.amyloid_atlas_ids():
        return True
    return False


def twist2pitch(
    twist: float, rise: float, return_pitch_for_4p75Angstrom_rise: bool = True
) -> float:
    """Convert helical twist and rise to pitch.

    When ``return_pitch_for_4p75Angstrom_rise`` is True (default), the
    calculation is adjusted to a rise near 4.75 Angstroms by scaling the
    twist and rise to match a single subunit.

    Parameters
    ----------
    twist : float
        Helical twist in degrees.
    rise : float
        Helical rise in Angstroms.
    return_pitch_for_4p75Angstrom_rise : bool, optional
        If True, adjust to a rise near 4.75 Angstroms. Defaults to True.

    Returns
    -------
    float
        Pitch in Angstroms.
    """
    if not return_pitch_for_4p75Angstrom_rise:
        return rise * 360 / abs(twist)

    rise_star = abs(rise)
    twist_star = abs(twist)
    for n in range(10, 1, -1):
        condition = (rise * n < 5) & (4.5 < rise * n)
        tmp_twist = abs(helicon.set_angle_range(twist_star * n, range=(-180, 180)))
        condition &= tmp_twist < 90
        if condition:
            twist_star = tmp_twist
            rise_star = rise_star * n
            break
    return rise_star * 360 / twist_star


# adapted from https://github.com/tdgrant1/denss/blob/3fbbefea45cb6d615e629e672d65440c46ac83da/saxstats/saxstats.py#L2185
def calc_fsc(map1, map2, apix, F1=None, F2=None, shell_flat=None, n=None):
    """Calculate Fourier Shell Correlation between two 3D maps.

    Parameters
    ----------
    map1 : np.ndarray or None
        First 3D map. Ignored if F1 is provided.
    map2 : np.ndarray or None
        Second 3D map. Ignored if F2 is provided.
    apix : float
        Pixel size in Angstroms per pixel.
    F1 : np.ndarray, optional
        Pre-computed rfftn of map1.
    F2 : np.ndarray, optional
        Pre-computed rfftn of map2.
    shell_flat : np.ndarray, optional
        Pre-computed flattened shell labels. If None, computed from n.
    n : int, optional
        Spatial dimension. Inferred from map1 or F1 if None.

    Returns
    -------
    np.ndarray
        Two-column array with spatial frequency (1/Angstrom) and FSC values.
    """
    if n is None:
        n = map1.shape[0] if F1 is None else F1.shape[0]
    df = 1.0 / (apix * n)

    if shell_flat is None:
        k2 = np.fft.fftfreq(n) ** 2
        kr2 = np.fft.rfftfreq(n) ** 2
        shell = np.round(
            np.sqrt(k2[:, None, None] + k2[None, :, None] + kr2[None, None, :]) * n
        ).astype(np.int32)
        np.clip(shell, 0, n // 2, out=shell)
        shell_flat = shell.ravel()
        del shell

    if F1 is None:
        from scipy.fft import rfftn

        F1 = rfftn(map1, workers=-1)
    if F2 is None:
        from scipy.fft import rfftn

        F2 = rfftn(map2, workers=-1)

    num = np.bincount(
        shell_flat, weights=np.real(F1 * np.conj(F2)).ravel(), minlength=n // 2 + 1
    )
    den1 = np.bincount(
        shell_flat, weights=(np.abs(F1) ** 2).ravel(), minlength=n // 2 + 1
    )
    den2 = np.bincount(
        shell_flat, weights=(np.abs(F2) ** 2).ravel(), minlength=n // 2 + 1
    )

    denom = np.sqrt(den1 * den2)
    fsc = np.ones(n // 2 + 1, dtype=np.float64)
    valid = denom > 0
    fsc[valid] = num[valid] / denom[valid]

    qx_max = np.fft.rfftfreq(n).max()
    saxis = np.arange(n // 2 + 1) * df
    idx = np.where(saxis <= qx_max)
    return np.vstack((saxis[idx], fsc[idx])).T


def calc_fsc_from_fft(F1, F2, n, apix):
    """Calculate FSC directly from pre-computed rfftn results.

    Parameters
    ----------
    F1 : np.ndarray
        rfftn result of first map.
    F2 : np.ndarray
        rfftn result of second map.
    n : int
        Original spatial dimension (maps are n×n×n).
    apix : float
        Pixel size in Angstroms per pixel.

    Returns
    -------
    np.ndarray
        Two-column array with spatial frequency (1/Angstrom) and FSC values.
    """
    df = 1.0 / (apix * n)
    k2 = np.fft.fftfreq(n) ** 2
    kr2 = np.fft.rfftfreq(n) ** 2
    shell = np.round(
        np.sqrt(k2[:, None, None] + k2[None, :, None] + kr2[None, None, :]) * n
    ).astype(np.int32)
    np.clip(shell, 0, n // 2, out=shell)
    shell_flat = shell.ravel()
    del shell

    num = np.bincount(
        shell_flat, weights=np.real(F1 * np.conj(F2)).ravel(), minlength=n // 2 + 1
    )
    den1 = np.bincount(
        shell_flat, weights=(np.abs(F1) ** 2).ravel(), minlength=n // 2 + 1
    )
    den2 = np.bincount(
        shell_flat, weights=(np.abs(F2) ** 2).ravel(), minlength=n // 2 + 1
    )

    denom = np.sqrt(den1 * den2)
    fsc = np.ones(n // 2 + 1, dtype=np.float64)
    valid = denom > 0
    fsc[valid] = num[valid] / denom[valid]

    qx_max = np.fft.rfftfreq(n).max()
    saxis = np.arange(n // 2 + 1) * df
    idx = np.where(saxis <= qx_max)
    return np.vstack((saxis[idx], fsc[idx])).T


def calc_fsc_per_shell(map1: np.ndarray, map2: np.ndarray, apix: float) -> np.ndarray:
    """Calculate FSC using per-voxel shell assignment (matching EMAN2).

    Each voxel is assigned to exactly one shell based on its integer
    distance from the origin, matching the EMAN2
    ``calc_fourier_shell_correlation`` behavior. Returns a 1D array of
    length ``n//2+1`` where index 0 is the DC term.

    Parameters
    ----------
    map1 : np.ndarray
        First 3D map.
    map2 : np.ndarray
        Second 3D map.
    apix : float
        Pixel size in Angstroms per pixel.

    Returns
    -------
    np.ndarray
        FSC values indexed by shell number (0 = DC, 1 = first shell, etc.).
        Spatial frequency for shell i is ``i / (n * apix)``.
    """
    n = map1.shape[0]
    from scipy.fft import fftn

    F1 = fftn(map1, workers=-1)
    F2 = fftn(map2, workers=-1)

    kx = np.fft.fftfreq(n)
    ky = np.fft.fftfreq(n)
    kz = np.fft.fftfreq(n)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    kr = np.sqrt(KX**2 + KY**2 + KZ**2)
    shell = np.round(kr * n).astype(np.int32)
    np.clip(shell, 0, n // 2, out=shell)

    nshells = n // 2 + 1
    num = np.zeros(nshells, dtype=np.float64)
    den1 = np.zeros(nshells, dtype=np.float64)
    den2 = np.zeros(nshells, dtype=np.float64)

    flat_shell = shell.ravel()
    flat_num = np.real(F1 * np.conj(F2)).ravel()
    flat_den1 = np.abs(F1).ravel() ** 2
    flat_den2 = np.abs(F2).ravel() ** 2

    np.add.at(num, flat_shell, flat_num)
    np.add.at(den1, flat_shell, flat_den1)
    np.add.at(den2, flat_shell, flat_den2)

    denom = np.sqrt(den1 * den2)
    fsc = np.ones(nshells, dtype=np.float64)
    valid = denom > 0
    fsc[valid] = num[valid] / denom[valid]
    return fsc


def calc_frc_2d(img1: np.ndarray, img2: np.ndarray, apix: float):
    """Compute 2D Fourier Ring Correlation curve between two images.

    Calculates the FSC as an array of correlation values across spatial
    frequency shells. This is useful for comparing a reconstructed projection
    against an input image.

    Parameters
    ----------
    img1 : np.ndarray
        First 2D image.
    img2 : np.ndarray
        Second 2D image (must have the same shape as img1).
    apix : float
        Pixel size in Angstroms per pixel.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Tuple of (spatial_frequencies, fsc_curve) where both are 1D arrays
        of the same length. spatial_frequencies is in 1/Angstrom.
        Returns (None, None) if either image is zero or computation fails.
    """
    from scipy.fft import fft2

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    img_h, img_w = img1.shape
    n_shells = min(img_h, img_w) // 2

    try:
        F1 = fft2(img1, workers=-1)
        F2 = fft2(img2, workers=-1)
    except TypeError:
        F1 = fft2(img1)
        F2 = fft2(img2)

    kx = np.fft.fftfreq(img_w) ** 2
    ky = np.fft.fftfreq(img_h) ** 2
    kr = np.sqrt(ky[:, None] + kx[None, :])
    shell = np.round(kr * n_shells).astype(np.int32)
    np.clip(shell, 0, n_shells, out=shell)
    shell_flat = shell.ravel()

    num = np.bincount(
        shell_flat,
        weights=np.real(F1 * np.conj(F2)).ravel(),
        minlength=n_shells + 1,
    )
    den1 = np.bincount(
        shell_flat, weights=(np.abs(F1) ** 2).ravel(), minlength=n_shells + 1
    )
    den2 = np.bincount(
        shell_flat, weights=(np.abs(F2) ** 2).ravel(), minlength=n_shells + 1
    )

    denom = np.sqrt(den1 * den2)
    fsc = np.ones(n_shells + 1, dtype=np.float64)
    valid = denom > 0
    fsc[valid] = num[valid] / denom[valid]

    saxis = np.arange(n_shells + 1) / (min(img_h, img_w) * apix)
    return saxis, fsc


def _fit_frc_curve(saxis, fsc, order=4):
    """Fit a Fermi or Butterworth function to the FRC curve.

    Parameters
    ----------
    saxis : np.ndarray
        Spatial frequency values (1/Angstrom).
    fsc : np.ndarray
        FRC values.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Fitted spatial frequencies and FRC values on a fine grid (500 points).
    """
    from scipy.optimize import minimize

    mask = np.isfinite(fsc) & (fsc >= -0.1) & (fsc <= 1.1)
    s_fit = saxis[mask]
    f_fit = fsc[mask]
    if len(s_fit) < 3:
        return saxis, fsc

    def fermi(mu, T, x):
        return 1.0 / (np.exp((x - mu) / T) + 1.0)

    def butterworth(omega, n, x):
        return 1.0 / (1.0 + (x / omega) ** n)

    def fit_score_fermi(params):
        mu, T = params
        if T <= 0:
            return 1e10
        a = 1.0 / fermi(mu, T, 0.0)
        pred = a * fermi(mu, T, s_fit)
        return np.mean(np.abs(f_fit - pred))

    def fit_score_butterworth(params):
        omega, n = params
        if omega <= 0 or n <= 0:
            return 1e10
        pred = butterworth(omega, n, s_fit)
        return np.mean(np.abs(f_fit - pred))

    best_error = np.inf
    best_fitted_fine = f_fit.copy()
    best_s_fine = s_fit.copy()

    mu0 = s_fit[len(s_fit) // 2]
    res_fermi = minimize(
        fit_score_fermi,
        x0=[mu0, 0.01],
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-6},
    )
    if res_fermi.fun < best_error:
        best_error = res_fermi.fun
        mu, T = res_fermi.x
        a = 1.0 / fermi(mu, T, 0.0)
        s_fine = np.linspace(saxis[1], saxis[-1], 500)
        f_fine = a * fermi(mu, T, s_fine)
        f_fine = np.clip(f_fine, -1, 1)
        best_fitted_fine = f_fine
        best_s_fine = s_fine

    omega0 = s_fit[len(s_fit) // 2]
    res_bw = minimize(
        fit_score_butterworth,
        x0=[omega0, 2.0],
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-6},
    )
    if res_bw.fun < best_error:
        omega, n = res_bw.x
        s_fine = np.linspace(saxis[1], saxis[-1], 500)
        f_fine = butterworth(omega, n, s_fine)
        f_fine = np.clip(f_fine, -1, 1)
        best_fitted_fine = f_fine
        best_s_fine = s_fine

    return best_s_fine, best_fitted_fine


def frc_score(
    img1: np.ndarray, img2: np.ndarray, apix: float, use_fit: bool = False
) -> float:
    """Compute a similarity score from the 2D FRC curve.

    Parameters
    ----------
    img1 : np.ndarray
        First 2D image.
    img2 : np.ndarray
        Second 2D image (must have the same shape as img1).
    apix : float
        Pixel size in Angstroms per pixel.
    use_fit : bool, optional
        If True, fit a Fermi or Butterworth curve to the FRC and compute
        the area under the fitted curve. If False, use the raw FRC curve.
        Defaults to False.

    Returns
    -------
    float
        FRC score. When use_fit=False, returns the mean of the raw FRC values.
        When use_fit=True, returns the normalized area under the fitted curve.
    """
    saxis, fsc = calc_frc_2d(img1, img2, apix)
    if saxis is None:
        return 0.0

    if use_fit:
        s_fine, f_fine = _fit_frc_curve(saxis, fsc)
        valid = np.isfinite(f_fine) & (f_fine >= -1) & (f_fine <= 1)
        if np.sum(valid) == 0:
            return 0.0
        area = np.trapz(f_fine[valid], s_fine[valid])
        freq_range = s_fine[valid][-1] - s_fine[valid][0]
        if freq_range <= 0:
            return 0.0
        return area / freq_range
    else:
        valid = np.isfinite(fsc) & (fsc >= -1) & (fsc <= 1)
        if np.sum(valid) == 0:
            return 0.0
        return float(np.mean(fsc[valid]))


def ssim_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two 2D images.

    Parameters
    ----------
    img1 : np.ndarray
        First 2D image.
    img2 : np.ndarray
        Second 2D image (must have the same shape as img1).

    Returns
    -------
    float
        SSIM score in ``[-1, 1]``. Returns 0 if computation fails.
    """
    from skimage.metrics import structural_similarity

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    try:
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
        if data_range == 0:
            return 0.0
        return float(structural_similarity(img1, img2, data_range=data_range))
    except Exception:
        return 0.0


def ms_ssim_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Multi-Scale SSIM (MS-SSIM) between two 2D images.

    Computes SSIM at multiple scales by successive downsampling, then
    combines the results. More robust than single-scale SSIM as it
    captures structure at different spatial resolutions.

    Parameters
    ----------
    img1 : np.ndarray
        First 2D image.
    img2 : np.ndarray
        Second 2D image (must have the same shape as img1).

    Returns
    -------
    float
        MS-SSIM score in ``[0, 1]``. Returns 0 if computation fails.
    """
    from skimage.metrics import structural_similarity
    from skimage.transform import rescale

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    try:
        data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
        if data_range == 0:
            return 0.0

        all_weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        min_size = 8

        ssim_values = []

        for i in range(len(all_weights)):
            h, w = img1.shape
            if h < min_size or w < min_size:
                break

            ssim_val = structural_similarity(
                img1,
                img2,
                data_range=data_range,
            )
            ssim_values.append(max(ssim_val, 0.0))

            if i < len(all_weights) - 1:
                img1 = rescale(img1, 0.5, anti_aliasing=True, channel_axis=None)
                img2 = rescale(img2, 0.5, anti_aliasing=True, channel_axis=None)
                data_range = max(img1.max() - img1.min(), img2.max() - img2.min())
                if data_range == 0:
                    break

        if len(ssim_values) == 0:
            return 0.0

        weights = all_weights[: len(ssim_values)]
        weights = weights / weights.sum()

        result = 1.0
        for s, w in zip(ssim_values, weights):
            result *= s**w

        return float(result)
    except Exception:
        return 0.0


def mutual_information_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute normalized mutual information between two 2D images.

    Uses skimage's normalized_mutual_information (Studholme et al. 1999),
    rescaled to [0, 1] where 1 = perfectly correlated, 0 = independent.

    Parameters
    ----------
    img1 : np.ndarray
        First 2D image.
    img2 : np.ndarray
        Second 2D image (must have the same shape as img1).

    Returns
    -------
    float
        Normalized mutual information in ``[0, 1]``. Returns 0 if
        computation fails.
    """
    from skimage.metrics import normalized_mutual_information

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    try:
        nmi = normalized_mutual_information(img1, img2, bins=64)
        return float((nmi - 1.0))
    except Exception:
        return 0.0


def r_factor_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute R-factor between two arrays.

    Crystallographic R-factor: ``sum(|img1 - img2|) / sum(|img2|)``.
    Returns ``1/(1+R)`` so that higher = better, bounded in (0, 1].

    Parameters
    ----------
    img1 : np.ndarray
        First array (predicted).
    img2 : np.ndarray
        Second array (reference, must have the same shape as img1).

    Returns
    -------
    float
        Score in ``(0, 1]`` where 1 = perfect match. Returns 0 if
        the reference array is zero.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shapes must match: {img1.shape} vs {img2.shape}")

    denom = np.sum(np.abs(img2))
    if denom == 0:
        return 0.0
    r = np.sum(np.abs(img1 - img2)) / denom
    return float(1.0 / (1.0 + r))


def estimate_helix_rotation_center_diameter(
    data: np.ndarray,
    estimate_rotation: bool = True,
    estimate_center: bool = True,
    threshold: float = 0,
) -> tuple[float, float, int]:
    """Estimate helix rotation, center, and diameter from a 2D image.

    Uses the union of all thresholded regions, then computes grayscale-weighted
    second moments for rotation and centroid.  This avoids the information loss
    of binary-only regionprops and handles fragmented masks better.

    Parameters
    ----------
    data : np.ndarray
        Input 2D image.
    estimate_rotation : bool, optional
        Whether to estimate the rotation angle. Defaults to True.
    estimate_center : bool, optional
        Whether to estimate the center from the region centroid. If False,
        the image center is used. Defaults to True.
    threshold : float, optional
        Threshold for binarization. Defaults to 0.

    Returns
    -------
    rotation : float
        Rotation (degrees) needed to align the helix to horizontal direction.
    shift_y : float
        Vertical shift (pixels) needed to center the helix in the box.
    diameter : int
        Estimated diameter (pixels) of the helix.
    """
    from skimage.morphology import closing
    import helicon

    ny, nx = data.shape

    def _weighted_params(mask, intensity):
        ys, xs = np.where(mask)
        if len(ys) < 2:
            return 0.0, 0.0, ny
        w = intensity[ys, xs].astype(np.float64)
        w = w - w.min() + 1e-8
        cw = w.sum()
        cy = (ys * w).sum() / cw
        cx = (xs * w).sum() / cw
        uy = ys - cy
        ux = xs - cx
        i_yy = (uy * uy * w).sum() / cw
        i_xx = (ux * ux * w).sum() / cw
        i_xy = (uy * ux * w).sum() / cw
        theta = 0.5 * np.arctan2(2.0 * i_xy, i_yy - i_xx)
        angle = np.rad2deg(theta) + 90.0
        if abs(angle) > 90.0:
            angle -= 180.0
        diameter = int(ys.max() - ys.min() + 1)
        if estimate_center:
            shift = ny // 2 - cy
        else:
            shift = 0.0
        return angle, shift, diameter

    bw = closing(data > threshold, mode="ignore")
    mask = bw > 0
    if not mask.any():
        return 0.0, 0.0, ny

    if estimate_rotation:
        rotation, _, _ = _weighted_params(mask, data)
        rotation = helicon.set_to_periodic_range(rotation, min=-180, max=180)
        data_rotated = helicon.transform_image(image=data, rotation=rotation)
    else:
        rotation = 0.0
        data_rotated = data

    bw_rot = closing(data_rotated > threshold, mode="ignore")
    mask_rot = bw_rot > 0
    if not mask_rot.any():
        return rotation, 0.0, ny

    _, shift_y, diameter = _weighted_params(mask_rot, data_rotated)

    return rotation, shift_y, diameter


def get_cylindrical_mask(
    nz: int, ny: int, nx: int, rmin: int = 0, rmax: int = -1, return_xyz: bool = False
) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create a cylindrical mask in a 3D grid.

    The cylinder axis runs along the Z direction.

    Parameters
    ----------
    nz : int
        Size along the Z axis.
    ny : int
        Size along the Y axis.
    nx : int
        Size along the X axis.
    rmin : int, optional
        Inner radius in pixels (excluded). Defaults to 0.
    rmax : int, optional
        Outer radius in pixels. Negative values are interpreted as
        ``ny // 2 - 1``. Defaults to -1.
    return_xyz : bool, optional
        If True, also return the meshgrid coordinates. Defaults to False.

    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape ``(nz, ny, nx)``.
    xyz : tuple of np.ndarray, optional
        Meshgrid coordinates ``(Z, Y, X)``, only returned if
        ``return_xyz`` is True.
    """
    k = np.arange(0, nz, dtype=np.int32) - nz // 2
    j = np.arange(0, ny, dtype=np.int32) - ny // 2
    i = np.arange(0, nx, dtype=np.int32) - nx // 2
    Z, Y, X = np.meshgrid(k, j, i, indexing="ij")
    if rmax < 0:
        rmax = ny // 2 - 1
    mask = X * X + Y * Y < rmax * rmax  # pixels inside a cylinder. axes order: z, y, x
    if 0 < rmin < rmax:
        mask &= X * X + Y * Y >= rmin * rmin
    if return_xyz:
        return mask, (Z, Y, X)
    else:
        return mask


def cross_correlation_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cross-correlation coefficient between two arrays.

    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.

    Returns
    -------
    float
        Cross-correlation coefficient in ``[-1, 1]``. Returns 0 if either
        array is constant (zero variance).
    """
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    norm = np.sqrt(np.sum((a - mean_a) ** 2) * np.sum((b - mean_b) ** 2))
    if norm == 0:
        return 0
    else:
        return np.sum((a - mean_a) * (b - mean_b)) / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors.

    Parameters
    ----------
    a : np.ndarray
        First vector.
    b : np.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine similarity in ``[-1, 1]``. Returns 0 if either vector is zero.
    """
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0
    else:
        return np.sum(a * b) / norm


# https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
def find_elbow_point(curve: np.ndarray) -> int:
    """Find the elbow (knee) point of a curve.

    Uses the maximum distance from the line connecting the first and last
    points of the curve.

    Parameters
    ----------
    curve : np.ndarray
        1D array of curve values.

    Returns
    -------
    int
        Index of the elbow point.
    """
    import numpy as np

    nPoints = len(curve)
    allCoord = np.vstack((range(nPoints), curve)).T
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.tile(lineVecNorm, (nPoints, 1)), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine**2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)  # should be the last point of 1st segment
    return idxOfBestPoint


def line_fit_projection(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    ref_i: int = 0,
    return_xy_fit: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Project points onto a line fitted by orthogonal distance regression.

    Uses scipy's ODR to fit a line, then projects each point onto the fitted
    line and returns signed positions relative to a reference point.

    Parameters
    ----------
    x : np.ndarray
        X coordinates of the data points.
    y : np.ndarray
        Y coordinates of the data points.
    w : np.ndarray, optional
        Weights for the ODR fit (same weights applied to x and y). Defaults
        to None (uniform weights).
    ref_i : int, optional
        Index of the reference point used as the origin for the signed
        projection. Defaults to 0.
    return_xy_fit : bool, optional
        If True, also return the fitted (x, y) coordinates. Defaults to False.

    Returns
    -------
    pos : np.ndarray
        Signed projected position along the fitted line.
    xy_fit : np.ndarray, optional
        Fitted ``(x, y)`` coordinates as an ``(N, 2)`` array, only returned
        if ``return_xy_fit`` is True.
    """
    import numpy as np
    from scipy import odr

    data = odr.Data(x, y, wd=w, we=w)
    odr_obj = odr.ODR(data, odr.unilinear)
    output = odr_obj.run()

    x2 = x + output.delta
    y2 = y + output.eps

    v0 = np.array([x2[-1] - x2[0], y2[-1] - y2[0]])
    v0 = v0 / np.linalg.norm(v0, ord=2)

    # signed, projected position on the fitted line
    pos = (x2 - x2[ref_i]) * v0[0] + (y2 - y2[ref_i]) * v0[1]  # dot product

    if return_xy_fit:
        return pos, np.vstack((x2, y2)).T
    else:
        return pos


def estimate_inter_segment_distance(
    data: pd.DataFrame,
) -> tuple[float | None, float | None, float | None, int | None]:
    """Estimate the inter-segment distance from helical particle data.

    Computes median, mean, and std of inter-segment distances from the
    ``rlnHelicalTrackLengthAngst`` column, grouped by micrograph and tube ID.

    Parameters
    ----------
    data : pd.DataFrame
        Particle data containing ``rlnImageName``, ``rlnHelicalTubeID``,
        and ``rlnHelicalTrackLengthAngst`` columns.

    Returns
    -------
    tuple of (float or None, float or None, float or None, int or None)
        ``(median, mean, std, n_max)`` where *n_max* is the estimated
        number of segments. All values are None if required columns are missing.
    """
    for attr in ["rlnImageName", "rlnHelicalTubeID", "rlnHelicalTrackLengthAngst"]:
        if attr not in data:
            return None, None, None, None

    data, data_orig = data.copy(), data
    temp = data["rlnImageName"].str.split("@", expand=True)
    data.loc[:, "pid"] = temp.iloc[:, 0].astype(int)
    filename = "micrograph"
    data.loc[:, filename] = temp.iloc[:, 1]
    data = data.sort_values([filename, "pid"], ascending=True)
    data.reset_index(drop=True, inplace=True)

    helices = data.groupby([filename, "rlnHelicalTubeID"], sort=False)

    import numpy as np

    dists_all = []
    lengths = []
    for _, particles in helices:
        lengths.append(particles["rlnHelicalTrackLengthAngst"].astype(np.float32).max())
        if len(particles) < 2:
            continue
        dists = np.sort(
            particles["rlnHelicalTrackLengthAngst"].astype(np.float32).values
        )
        dists = dists[1:] - dists[:-1]
        dists_all.append(dists)
    dists_all = np.hstack(dists_all)
    dist_seg_median = np.median(dists_all)  # Angstrom
    dist_seg_mean = np.mean(dists_all)  # Angstrom
    dist_seg_sigma = np.std(dists_all)  # Angstrom
    n_max = np.sum(np.round(np.array(lengths) / dist_seg_median) + 1).astype(int)
    return dist_seg_median, dist_seg_mean, dist_seg_sigma, n_max


def reset_inter_segment_distance(
    data: pd.DataFrame,
    new_inter_segment_distance: float,
    apix_micrograph: float,
    current_inter_segment_distance: float = -1,
    verbose: int = 0,
) -> pd.DataFrame | None:
    """Reset inter-segment distance by adding/removing particles.

    Parameters
    ----------
    data : pd.DataFrame
        Particle data with ``rlnHelicalTubeID``, ``rlnCoordinateX``,
        ``rlnCoordinateY``, and either ``rlnImageName`` or
        ``rlnMicrographName``.
    new_inter_segment_distance : float
        Desired inter-segment distance in Angstroms.
    apix_micrograph : float
        Pixel size of the micrograph in Angstroms/pixel.
    current_inter_segment_distance : float, optional
        Current inter-segment distance. If <= 0, it will be estimated.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    pd.DataFrame or None
        Updated dataframe, or None if required columns are missing.
    """
    if (
        current_inter_segment_distance > 0
        and new_inter_segment_distance == current_inter_segment_distance
    ):
        return data

    for attr in ["rlnHelicalTubeID", "rlnCoordinateX", "rlnCoordinateY"]:
        if attr not in data:
            return None
    if "rlnImageName" in data:
        tmp = data.loc[:, "rlnImageName"].str.split("@", expand=True)
        data.loc[:, "risd_pid"] = tmp.iloc[:, 0].astype(int)
        data.loc[:, "risd_filename"] = tmp.iloc[:, 1]
        filename = "risd_filename"
    else:
        return None

    if "rlnMicrographName" in data:
        filename = "rlnMicrographName"

    if current_inter_segment_distance <= 0:
        current_inter_segment_distance = estimate_inter_segment_distance(data)[0]

    if new_inter_segment_distance == current_inter_segment_distance:
        data.drop(["risd_filename", "risd_pid"], inplace=True, axis=1)
        return data

    cdist = current_inter_segment_distance / apix_micrograph
    ndist = new_inter_segment_distance / apix_micrograph

    from tqdm import tqdm

    data2 = []
    helices = data.groupby([filename, "rlnHelicalTubeID"], sort=False)
    for _, particles in tqdm(helices, unit=" helicaltubes", disable=verbose < 1):
        if len(particles) < 2:
            data2.append(particles.reset_index(drop=True))
            continue
        particles_sorted = particles.sort_values(
            ["risd_pid"], ascending=True
        ).reset_index(drop=True)
        x = particles_sorted.loc[:, "rlnCoordinateX"].astype(float).values
        y = particles_sorted.loc[:, "rlnCoordinateY"].astype(float).values
        pos, xy_fit = line_fit_projection(x, y, w=None, ref_i=0, return_xy_fit=True)
        n0 = len(pos)
        unit_vec = (xy_fit[-1] - xy_fit[0]) / (pos[-1] - pos[0])
        right = np.arange(pos[0], pos[-1] + cdist / 2 + 0.1, ndist)
        left = np.arange(pos[0] - ndist, pos[0] - cdist / 2, -ndist)
        if len(left):
            left.sort()
            pos_new = np.hstack((left, right))
        else:
            pos_new = right
        n = len(pos_new)
        xy_new = xy_fit[0] + pos_new.reshape((n, 1)) * unit_vec
        if n <= n0:
            df_tmp = particles_sorted.iloc[:n].reset_index(drop=True)
        else:
            df_tmp = particles_sorted.iloc[:n0].reset_index(drop=True)
            index_repeat = [df_tmp.index[-1]] * (n - n0)
            df_tmp = pd.concat([df_tmp, df_tmp.iloc[index_repeat]], ignore_index=True)
        df_tmp.loc[:, "rlnCoordinateX"] = xy_new[:, 0]
        df_tmp.loc[:, "rlnCoordinateY"] = xy_new[:, 1]
        if "rlnHelicalTrackLengthAngst" in df_tmp:
            df_tmp.loc[:, "rlnHelicalTrackLengthAngst"] = (
                pos_new - pos_new[0]
            ) * apix_micrograph
        data2.append(df_tmp)

    data2 = pd.concat(data2)
    data2.drop(["risd_filename", "risd_pid"], inplace=True, axis=1)

    try:
        data2.attrs = data.attrs
    except Exception:
        pass

    return data2


def estimate_helicalTube_length(
    data: pd.DataFrame,
    inter_segment_distance: float = -1,
    verbose: int = 0,
) -> pd.DataFrame | None:
    """Estimate the length of each helical filament/tube.

    Adds a ``rlnHelicalTubeLength`` column to the dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Particle data with ``rlnHelicalTubeID``, ``rlnCoordinateX``,
        ``rlnCoordinateY``, and ``rlnImageName``.
    inter_segment_distance : float, optional
        Known inter-segment distance. If <= 0, it will be estimated.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    pd.DataFrame or None
        Updated dataframe, or None if required columns are missing.
    """
    for attr in ["rlnHelicalTubeID", "rlnCoordinateX", "rlnCoordinateY"]:
        if attr not in data:
            return None
    if "rlnImageName" in data:
        tmp = data.loc[:, "rlnImageName"].str.split("@", expand=True)
        data.loc[:, "ehl_pid"] = tmp.iloc[:, 0].astype(int)
        filename = "ehl_filename"
        data.loc[:, filename] = tmp.iloc[:, 1]
    else:
        return None

    if "rlnMicrographName" in data:
        filename = "rlnMicrographName"

    if inter_segment_distance <= 0:
        inter_segment_distance = estimate_inter_segment_distance(data)[0]

    helices = data.groupby([filename, "rlnHelicalTubeID"], sort=False)

    from tqdm import tqdm

    for _, particles in tqdm(helices, unit=" helicaltubes", disable=verbose < 1):
        if "rlnHelicalTrackLengthAngst" in particles:
            data.loc[particles.index, "rlnHelicalTubeLength"] = round(
                particles["rlnHelicalTrackLengthAngst"].max(), 1
            )
        else:
            pids = particles["ehl_pid"].astype(int).values
            helical_len = (pids.max() - pids.min() + 1) * inter_segment_distance
            data.loc[particles.index, "rlnHelicalTubeLength"] = round(helical_len, 1)

    data.drop(["ehl_filename", "ehl_pid"], inplace=True, axis=1)
    return data


from .clustering import AgglomerativeClusteringWithMinSize  # noqa: F401
from .alignment import align_images  # noqa: F401
