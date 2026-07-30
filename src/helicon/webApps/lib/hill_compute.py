"""Compute functions for the HILL tab — power spectra, layer lines, Bessel orders, etc.

Adapted from HILL.git/compute.py. All functions that duplicate helicon lib
have been removed in favor of the canonical implementations.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, affine_transform
from scipy.interpolate import splrep, splev, RegularGridInterpolator
from scipy.signal import find_peaks
from scipy.special import jnp_zeros
from scipy.optimize import minimize
from itertools import product

from bokeh.events import MouseMove
from bokeh.models import ColumnDataSource, CustomJS, LinearColorMapper
from bokeh.models.tools import CrosshairTool, HoverTool
from bokeh.plotting import figure

import helicon


# ── Bessel functions ─────────────────────────────────────────────


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def bessel_1st_peak_positions(n_max: int = 100) -> np.ndarray:
    """Table of first peak positions of Bessel functions Jn for n=0..n_max."""
    ret = np.zeros(n_max + 1, dtype=np.float32)
    for i in range(1, n_max + 1):
        ret[i] = jnp_zeros(i, 1)[0]
    return ret


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def bessel_n_image(
    ny: int,
    nx: int,
    nyquist_res_x: float,
    nyquist_res_y: float,
    radius: float,
    tilt: float,
) -> np.ndarray:
    """Bessel order for each pixel in a power spectrum image."""
    table = bessel_1st_peak_positions()
    if tilt:
        dsx = 1.0 / (nyquist_res_x * nx // 2)
        dsy = 1.0 / (nyquist_res_x * ny // 2)
        Y, X = np.meshgrid(
            np.arange(ny, dtype=np.float32) - ny // 2,
            np.arange(nx, dtype=np.float32) - nx // 2,
            indexing="ij",
        )
        Y = 2 * np.pi * np.abs(Y) * dsy * radius
        X = 2 * np.pi * np.abs(X) * dsx * radius
        Y /= np.cos(np.deg2rad(tilt))
        X = np.hypot(X, Y * np.sin(np.deg2rad(tilt)))
        X = np.expand_dims(X.flatten(), axis=-1)
        indices = np.abs(table - X).argmin(axis=-1)
        return np.reshape(indices, (ny, nx)).astype(np.int16)
    else:
        ds = 1.0 / (nyquist_res_x * nx // 2)
        xs = 2 * np.pi * np.abs(np.arange(nx) - nx // 2) * ds * radius
        xs = np.expand_dims(xs, axis=-1)
        indices = np.abs(table - xs).argmin(axis=-1)
        return np.tile(indices, (ny, 1)).astype(np.int16)


# ── Layer line computation ───────────────────────────────────────


def twist2pitch(twist: float, rise: float) -> float:
    """Convert twist (deg) and rise (A) to pitch (A)."""
    if twist:
        return 360.0 * rise / abs(twist)
    return rise


def pitch2twist(pitch: float, rise: float) -> float:
    """Convert pitch (A) and rise (A) to twist (deg)."""
    if pitch > rise:
        return helicon.set_to_periodic_range(360.0 * rise / pitch)
    return 0.0


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def compute_layer_line_positions(
    twist: float,
    rise: float,
    csym: int,
    radius: float,
    tilt: float,
    cutoff_res: float,
    m_max: int = -1,
) -> dict:
    """Compute layer line positions (x, y, Bessel orders) for given helical params.

    Returns a dict keyed by m-order, each with ``{"LL": (xs, ys, bessel_n), "m": m}``.
    """
    if cutoff_res <= 0:
        return {}
    table = bessel_1st_peak_positions() / (2 * np.pi * radius)

    if m_max < 1:
        m_max = int(np.floor(abs(rise / cutoff_res))) + 3
    m_vals = list(range(-m_max, m_max + 1))
    m_vals.sort(key=lambda x: (abs(x), x))  # 0, -1, 1, -2, 2, ...

    smax = 1.0 / cutoff_res
    tf = 1.0 / np.cos(np.deg2rad(tilt))
    tf2 = np.sin(np.deg2rad(tilt))
    m_groups = {}
    for mi in range(len(m_vals)):
        d = {}
        sy0 = m_vals[mi] / rise
        p = twist2pitch(twist, rise)
        ds_p = 1.0 / p
        ll_i_top = int(abs(smax - sy0) / ds_p) * 2
        ll_i_bottom = -int(abs(-smax - sy0) / ds_p) * 2
        ll_i = np.array(
            [i for i in range(ll_i_bottom, ll_i_top + 1) if not i % csym],
            dtype=np.int32,
        )
        sy = sy0 + ll_i * ds_p
        sx = table[np.clip(np.abs(ll_i), 0, len(table) - 1)]
        if tilt:
            sy = np.array(sy, dtype=np.float32) * tf
            sx = np.sqrt(
                np.power(np.array(sx, dtype=np.float32), 2) - np.power(sy * tf2, 2)
            )
            sx[np.isnan(sx)] = 1e-6
        px = list(sx) + list(-sx)
        py = list(sy) + list(sy)
        n_vals = list(ll_i) + list(ll_i)
        d["LL"] = (px, py, n_vals)
        d["m"] = m_vals
        m_groups[m_vals[mi]] = d
    return m_groups


# ── Power spectra computation ─────────────────────────────────────


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def compute_power_spectra(
    data: np.ndarray,
    apix: float,
    cutoff_res=None,
    output_size=None,
    log: bool = True,
    square: bool = False,
    do_normalize: bool = True,
    low_pass_fraction: float = 0,
    high_pass_fraction: float = 0,
):
    """Compute power spectrum and phase from 2D image."""
    fft = fft_rescale(data, apix=apix, cutoff_res=cutoff_res, output_size=output_size)
    fft = np.fft.fftshift(fft)
    if log:
        pwr = np.log1p(np.abs(fft))
    else:
        pwr = np.abs(fft)
        if square:
            pwr *= pwr
    if 0 < low_pass_fraction < 1 or 0 < high_pass_fraction < 1:
        pwr = helicon.low_high_pass_filter(
            pwr,
            low_pass_fraction=low_pass_fraction,
            high_pass_fraction=high_pass_fraction,
        )
    if do_normalize:
        pwr = normalize(pwr, percentile=(0, 100))
    phase = np.angle(fft, deg=False)
    return pwr.astype(np.float32), phase.astype(np.float32)


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def fft_rescale(
    image: np.ndarray, apix: float = 1.0, cutoff_res=None, output_size=None
):
    """Non-uniform FFT rescaling of an image to given resolution/size."""
    if cutoff_res:
        res_limit_y, cutoff_res_x = cutoff_res
    else:
        res_limit_y = cutoff_res_x = 2 * apix
    if output_size:
        ony, onx = output_size
    else:
        ony, onx = image.shape

    from finufft import nufft2d2

    freq_y = np.fft.fftfreq(ony) * 2 * apix / res_limit_y
    freq_x = np.fft.fftfreq(onx) * 2 * apix / cutoff_res_x
    Y, X = np.meshgrid(freq_y, freq_x, indexing="ij")
    Y = (2 * np.pi * Y).flatten(order="C")
    X = (2 * np.pi * X).flatten(order="C")
    fft = nufft2d2(x=Y, y=X, f=image.astype(np.complex128), eps=1e-6)
    fft = fft.reshape((ony, onx))
    # phase shifts for real-space shifts by half the box
    phase_shift = np.ones(fft.shape)
    phase_shift[1::2, :] *= -1
    phase_shift[:, 1::2] *= -1
    fft *= phase_shift
    return fft


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def resize_rescale_power_spectra(
    data,
    nyquist_res,
    cutoff_res=None,
    output_size=None,
    log=True,
    low_pass_fraction=0,
    high_pass_fraction=0,
    norm=1,
):
    """Resize a power spectrum image to match new resolution/size."""
    ny, nx = data.shape
    ony, onx = output_size
    res_y, res_x = cutoff_res
    Y, X = np.meshgrid(
        np.arange(ony, dtype=np.float32) - (ony // 2 + 0.5),
        np.arange(onx, dtype=np.float32) - (onx // 2 + 0.5),
        indexing="ij",
    )
    Y = Y / (ony // 2 + 0.5) * nyquist_res / res_y * ny // 2 + ny // 2 + 0.5
    X = X / (onx // 2 + 0.5) * nyquist_res / res_x * nx // 2 + nx // 2 + 0.5
    pwr = map_coordinates(
        data, (Y.flatten(), X.flatten()), order=3, mode="constant"
    ).reshape(Y.shape)
    if log:
        pwr = np.log1p(np.abs(pwr))
    if 0 < low_pass_fraction < 1 or 0 < high_pass_fraction < 1:
        pwr = helicon.low_high_pass_filter(
            pwr,
            low_pass_fraction=low_pass_fraction,
            high_pass_fraction=high_pass_fraction,
        )
    if norm:
        pwr = normalize(pwr, percentile=(0, 100))
    return pwr


def compute_phase_difference_across_meridian(phase):
    """Compute phase difference across the meridian (0=even, pi=odd)."""
    phase_diff = phase * 0
    phase_diff[..., 1:] = phase[..., 1:] - phase[..., 1:][..., ::-1]
    phase_diff = np.rad2deg(np.arccos(np.cos(phase_diff)))
    return phase_diff


# ── Normalization / filtering ────────────────────────────────────


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def normalize(data: np.ndarray, percentile=(0, 100)):
    """Linear min-max normalization to [0, 1]."""
    p0, p1 = percentile
    vmin, vmax = np.percentile(data, (p0, p1))
    if vmax - vmin == 0:
        return data * 0
    return (data - vmin) / (vmax - vmin)


# ── 2D image transforms ──────────────────────────────────────────


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def transform_2d_image(data, angle, dx, dy, negate, apix):
    """Rotate, shift, and optionally negate a 2D image."""
    if negate:
        data = -data
    if angle or dx or dy:
        data = rotate_shift_image(
            data,
            angle=-angle,
            post_shift=(dy / apix, dx / apix),
            order=1,
        )
    return data


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def rotate_shift_image(
    data, angle=0, pre_shift=(0, 0), post_shift=(0, 0), rotation_center=None, order=1
):
    """Rotate and shift a 2D image."""
    ny, nx = data.shape
    if angle == 0 and list(pre_shift) == [0, 0] and list(post_shift) == [0, 0]:
        return data * 1.0
    if rotation_center is None:
        rotation_center = np.array((ny // 2, nx // 2), dtype=np.float32)
    ang = np.deg2rad(angle)
    m = np.array(
        [[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]], dtype=np.float32
    )
    pre_dy, pre_dx = pre_shift
    post_dy, post_dx = post_shift
    offset = -np.dot(m, np.array([post_dy, post_dx], dtype=np.float32).T)
    offset += np.array(rotation_center, dtype=np.float32).T - np.dot(
        m, np.array(rotation_center, dtype=np.float32).T
    )
    offset += -np.array([pre_dy, pre_dx], dtype=np.float32).T
    return affine_transform(data, matrix=m, offset=offset, order=order, mode="constant")


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def mask_2d_filament(data, mask_radius, apix, mask_len_fraction):
    """Apply a filament-shaped mask to a 2D image."""
    _, nx = data.shape
    fraction_x = mask_radius / (nx // 2 * apix)
    taper = generate_tapering_filter(
        image_size=data.shape,
        fraction_start=[mask_len_fraction, fraction_x],
        fraction_slope=(1.0 - mask_len_fraction) / 2.0,
    )
    return data * taper


def generate_tapering_filter(image_size, fraction_start=(0, 0), fraction_slope=0.1):
    """Generate a cosine-tapered filter for the image edges."""
    ny, nx = image_size
    fy, fx = fraction_start
    if not (0 < fy < 1 or 0 < fx < 1):
        return np.ones((ny, nx))
    Y, X = np.meshgrid(
        np.arange(0, ny, dtype=np.float32) - ny // 2,
        np.arange(0, nx, dtype=np.float32) - nx // 2,
        indexing="ij",
    )
    filt = np.ones_like(Y)
    if 0 < fy < 1:
        Y_abs = np.abs(Y / (ny // 2))
        inner = Y_abs < fy
        outer = Y_abs > fy + fraction_slope
        Y_w = (Y_abs - fy) / fraction_slope
        Y_w = (1.0 + np.cos(Y_w * np.pi)) / 2.0
        Y_w[inner] = 1
        Y_w[outer] = 0
        filt *= Y_w
    if 0 < fx < 1:
        X_abs = np.abs(X / (nx // 2))
        inner = X_abs < fx
        outer = X_abs > fx + fraction_slope
        X_w = (X_abs - fx) / fraction_slope
        X_w = (1.0 + np.cos(X_w * np.pi)) / 2.0
        X_w[inner] = 1
        X_w[outer] = 0
        filt *= X_w
    return filt


# ── Auto-detection ────────────────────────────────────────────────


def estimate_radial_range(data, thresh_ratio=0.1):
    """Estimate filament radial range from projection profile."""
    proj_y = np.sum(data, axis=0)
    n = len(proj_y)
    background = np.mean(proj_y[[0, 1, 2, -3, -2, -1]])
    thresh = (proj_y.max() - background) * thresh_ratio + background
    indices = np.nonzero(proj_y < thresh)[0]
    try:
        xmin = np.max(indices[indices < np.argmax(proj_y[: n // 2])])
    except Exception:
        xmin = 0
    try:
        xmax = np.min(indices[indices > np.argmax(proj_y[n // 2 :]) + n // 2])
    except Exception:
        xmax = n - 1
    mask_radius_v = max(abs(n // 2 - xmin), abs(xmax - n // 2))
    proj_y -= thresh
    proj_y[proj_y < 0] = 0

    def _fit_radial_profile(x, rad_profile):
        a, b, w, rcore, rmax = x
        n = len(rad_profile)
        xv = np.abs(np.arange(n, dtype=float) - n / 2)
        yshell = rad_profile * 0
        mask = xv <= abs(rmax)
        yshell[mask] = np.sqrt(rmax * rmax - xv[mask] * xv[mask])
        ycore = rad_profile * 0
        mask = xv <= abs(rcore)
        ycore[mask] = np.sqrt(rcore * rcore - xv[mask] * xv[mask])
        y = a * (yshell + (w - 1) * ycore) + b
        return np.linalg.norm(y - rad_profile)

    bounds = (
        (0, None),
        (None, None),
        (0, None),
        (0, mask_radius_v),
        (0, mask_radius_v),
    )
    results = []
    for val_a, val_w, val_rcore in product(
        (1, 2, 4, 8), (0, 0.5), (0, mask_radius_v / 2)
    ):
        x0 = (val_a, 0, val_w, val_rcore, mask_radius_v)
        res = minimize(
            _fit_radial_profile,
            x0,
            args=(proj_y,),
            method="Nelder-Mead",
            bounds=bounds,
            tol=1e-6,
        )
        a, b, w, rcore, rmax = res.x
        results.append((res.fun, w, rcore, rmax, val_a, val_w, val_rcore))
    result = sorted(results)[0]
    w, rcore, rmax = result[1:4]
    rmean = 0.5 * (rmax * rmax + (w - 1) * rcore * rcore) / (rmax + (w - 1) * rcore)
    return float(rmean), float(mask_radius_v)


# ── Simulation ────────────────────────────────────────────────────


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def simulate_helix(
    twist, rise, csym, helical_radius, ball_radius, ny, nx, apix, tilt=0, az0=None
):
    """Simulate a 2D projection of a helical structure."""

    def _simulate_projection(centers, sigma, ny, nx, apix):
        sigma2 = sigma * sigma
        d = np.zeros((ny, nx))
        Y, X = np.meshgrid(
            np.arange(0, ny, dtype=np.float32) - ny // 2,
            np.arange(0, nx, dtype=np.float32) - nx // 2,
            indexing="ij",
        )
        X *= apix
        Y *= apix
        for ci in range(len(centers)):
            yc, xc = centers[ci]
            x = X - xc
            y = Y - yc
            d += np.exp(-(x * x + y * y) / sigma2)
        return d

    def _helical_unit_positions(twist, rise, csym, radius, height, tilt=0, az0=0):
        imax = int(height / rise)
        centers = np.zeros(((2 * imax + 1) * csym, 3), dtype=np.float32)
        for i in range(-imax, imax + 1):
            z = rise * i
            for si in range(csym):
                angle = np.deg2rad(twist * i + si * 360.0 / csym + az0 + 90)
                x = np.cos(angle) * radius
                y = np.sin(angle) * radius
                centers[i * csym + si, 0] = x
                centers[i * csym + si, 1] = y
                centers[i * csym + si, 2] = z
        if tilt:
            from scipy.spatial.transform import Rotation as R

            rot = R.from_euler("x", tilt, degrees=True)
            centers = rot.apply(centers)
        centers = centers[:, [2, 0]]  # project along y
        return centers

    if az0 is None:
        az0 = np.random.uniform(0, 360)
    centers = _helical_unit_positions(
        twist, rise, csym, helical_radius, height=ny * apix, tilt=tilt, az0=az0
    )
    projection = _simulate_projection(centers, ball_radius, ny, nx, apix)
    return projection


# ── Filament straightening ────────────────────────────────────────


def sample_filament_axis(data, num_points=10, filament_radius_pixels=None):
    """Sample points along the central axis of a curved filament.

    Designed for low-SNR images where the filament runs roughly vertically.
    Returns (xs, ys) pixel coordinates, ordered top-to-bottom.
    """
    data = np.asarray(data, dtype=np.float64)
    ny, nx = data.shape

    # 1. Estimate filament radius if not provided
    if filament_radius_pixels is None:
        proj_x = np.sum(data, axis=0)
        bg = np.mean(np.concatenate([proj_x[:3], proj_x[-3:]]))
        signal = proj_x - bg
        peak_val = signal.max()
        if peak_val <= 0:
            filament_radius_pixels = max(nx // 4, 3)
        else:
            thresh = 0.1 * peak_val
            above = np.where(signal > thresh)[0]
            if len(above) >= 2:
                filament_radius_pixels = max((above[-1] - above[0]) // 2, 3)
            else:
                filament_radius_pixels = max(nx // 4, 3)

    r_est = int(filament_radius_pixels)
    from scipy.ndimage import median_filter, gaussian_filter, uniform_filter1d

    # 2. Two-scale denoising
    sigma_coarse = max(r_est, 3)
    coarse = gaussian_filter(data, sigma=(sigma_coarse, sigma_coarse), mode="nearest")
    sigma_y_fine = max(r_est, 3)
    sigma_x_fine = max(r_est // 3, 2)
    fine = gaussian_filter(data, sigma=(sigma_y_fine, sigma_x_fine), mode="nearest")
    for arr in [coarse, fine]:
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin > 0:
            arr -= vmin
            arr /= vmax - vmin

    # 3. Row-wise center detection
    coarse_centers = np.argmax(coarse, axis=1).astype(np.float64)
    window_half = max(int(1.5 * r_est), 5)
    refined_centers = np.copy(coarse_centers)
    for row in range(ny):
        profile = fine[row, :]
        center_guess = coarse_centers[row]
        lo = max(0, int(center_guess - window_half))
        hi = min(nx, int(center_guess + window_half + 1))
        seg = profile[lo:hi]
        seg_shifted = seg - seg.min()
        total = seg_shifted.sum()
        if total > 0:
            com = lo + np.dot(np.arange(len(seg_shifted)), seg_shifted) / total
            refined_centers[row] = com

    # 4. Outlier rejection
    med_kernel = max(ny // 15, 5)
    if med_kernel % 2 == 0:
        med_kernel += 1
    smoothed_centers = median_filter(refined_centers, size=med_kernel, mode="nearest")
    deviation = np.abs(refined_centers - smoothed_centers)
    outliers = deviation > r_est
    cleaned_centers = np.where(outliers, smoothed_centers, refined_centers)
    cleaned_centers = uniform_filter1d(
        cleaned_centers, size=max(ny // 30, 3), mode="nearest"
    )

    # 5. Cubic smoothing spline
    ys_all = np.arange(ny, dtype=np.float64)
    if ny < 4:
        ys_out = np.linspace(0, ny - 1, num_points)
        xs_out = np.full(num_points, nx / 2.0)
        return xs_out, ys_out

    s_factor = ny * (r_est * 0.2) ** 2
    tck = splrep(ys_all, cleaned_centers, s=s_factor, k=3)

    # 6. Uniform arc-length sampling
    dense_y = np.linspace(0, ny - 1, max(2000, ny * 2))
    dense_x = splev(dense_y, tck)
    dy = np.diff(dense_y)
    dx = np.diff(dense_x)
    seg_lengths = np.sqrt(dx**2 + dy**2)
    cum_arc = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_arc = cum_arc[-1]
    target_arc = np.linspace(0, total_arc, num_points)
    sample_y = np.interp(target_arc, cum_arc, dense_y)
    sample_x = splev(sample_y, tck)
    return np.asarray(sample_x, dtype=np.float64), np.asarray(
        sample_y, dtype=np.float64
    )


def fit_spline(xs, ys):
    """Fit a cubic smoothing spline to axis points."""
    if len(xs) < 4:
        return None
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    tck = splrep(ys, xs, s=20)
    return tck


def filament_straighten(data, tck, r_filament_pixel_display, y_start, y_end):
    """Straighten a curved filament using a spline fit."""
    ny, nx = data.shape
    x_coord = np.arange(0, nx, 1)
    y_coord = np.arange(0, ny, 1)
    interpol = RegularGridInterpolator(
        (x_coord, y_coord), np.transpose(data), bounds_error=False, fill_value=0
    )

    nx_out = 2 * int(r_filament_pixel_display)

    # Resample spline at uniform arc-length steps
    dense_y = np.linspace(y_start, y_end, 1000)
    dense_x = splev(dense_y, tck)
    dy = np.diff(dense_y)
    dx = np.diff(dense_x)
    dist_segments = np.sqrt(dx**2 + dy**2)
    cum_dist = np.concatenate(([0], np.cumsum(dist_segments)))
    total_length = cum_dist[-1]
    target_dists = np.arange(0, total_length, 1.0)
    uniform_arc_y = np.interp(target_dists, cum_dist, dense_y)
    uniform_arc_x = splev(uniform_arc_y, tck)

    new_im = np.zeros((len(uniform_arc_y), r_filament_pixel_display * 2))
    for row, (curr_y, curr_x) in enumerate(zip(uniform_arc_y, uniform_arc_x)):
        dxdy = splev(curr_y, tck, der=1)
        orthog_dxdy = -(1.0 / dxdy)
        rev_normal = lambda x: (x + orthog_dxdy * curr_y - curr_x) / orthog_dxdy
        new_row_xs = (
            np.arange(
                -int(r_filament_pixel_display), int(r_filament_pixel_display), 1
            ).T
            * abs(orthog_dxdy)
            / np.sqrt(1 + orthog_dxdy * orthog_dxdy)
            + curr_x
        )
        new_row_ys = rev_normal(new_row_xs)
        new_row_coords = np.vstack([new_row_xs, new_row_ys]).T
        new_row = interpol(new_row_coords)
        new_im[row, :] = new_row

    # Fill edge zeros with median
    data_median = np.median(data)
    for i in range(new_im.shape[0]):
        for j in range(new_im.shape[1]):
            if new_im[i, j] == 0:
                new_im[i, j] = data_median
            else:
                break
        for j in range(new_im.shape[1]):
            if new_im[i, -(j + 1)] == 0:
                new_im[i, -(j + 1)] = data_median
            else:
                break
    return new_im


# ── Bokeh figure helpers ─────────────────────────────────────────


def create_image_figure(
    image,
    dx,
    dy,
    title="",
    title_location="below",
    plot_width=None,
    plot_height=None,
    x_axis_label="x",
    y_axis_label="y",
    tooltips=None,
    show_axis=True,
    show_toolbar=True,
    crosshair_color="white",
    aspect_ratio=None,
    tools="box_zoom,crosshair,pan,reset,save,wheel_zoom",
):
    """Create a Bokeh figure displaying a 2D image."""
    h, w = image.shape
    if aspect_ratio is None:
        if plot_width and plot_height:
            aspect_ratio = plot_width / plot_height
        else:
            aspect_ratio = w * dx / (h * dy)

    fig = figure(
        title_location=title_location,
        frame_width=plot_width,
        frame_height=plot_height,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        x_range=(-w // 2 * dx, (w // 2 - 1) * dx),
        y_range=(-h // 2 * dy, (h // 2 - 1) * dy),
        tools=tools,
        aspect_ratio=aspect_ratio,
    )
    fig.grid.visible = False
    if title:
        fig.title.text = title
        fig.title.align = "center"
        fig.title.text_font_size = "18px"
        fig.title.text_font_style = "normal"
    if not show_axis:
        fig.axis.visible = False
    if not show_toolbar:
        fig.toolbar_location = None

    source_data = ColumnDataSource(
        data=dict(
            image=[image],
            x=[-w // 2 * dx],
            y=[-h // 2 * dy],
            dw=[w * dx],
            dh=[h * dy],
        )
    )
    color_mapper = LinearColorMapper(palette="Greys256")
    img_glyph = fig.image(
        source=source_data,
        image="image",
        color_mapper=color_mapper,
        x="x",
        y="y",
        dw="dw",
        dh="dh",
    )
    if not tooltips:
        tooltips = [("x", "$x Å"), ("y", "$y Å"), ("val", "@image")]
    img_hover = HoverTool(renderers=[img_glyph], tooltips=tooltips)
    fig.add_tools(img_hover)
    fig.hover[0].attachment = "vertical"
    crosshair = [t for t in fig.tools if isinstance(t, CrosshairTool)]
    if crosshair:
        for ch in crosshair:
            ch.line_color = crosshair_color
    return fig, source_data


def create_layerline_image_figure(
    data,
    cutoff_res_x,
    cutoff_res_y,
    helical_radius,
    tilt,
    phase=None,
    fft_top_only=False,
    pseudo_color=True,
    const_image_color="",
    title="",
    yaxis_visible=True,
    tooltips=None,
    hover_phase=False,
):
    """Create a Bokeh figure for power spectrum / phase difference layer lines."""
    ny, nx = data.shape
    dsy = 1.0 / (ny // 2 * cutoff_res_y)
    dsx = 1.0 / (nx // 2 * cutoff_res_x)
    x_range = (-(nx // 2) * dsx, (nx // 2) * dsx)
    if fft_top_only:
        y_range = (-(ny // 2 * 0.01) * dsy, (ny // 2 - 0.5) * dsy)
    else:
        y_range = (-(ny // 2 + 0.5) * dsy, (ny // 2 - 0.5) * dsy)

    bessel = bessel_n_image(
        ny, nx, cutoff_res_x, cutoff_res_y, helical_radius, tilt
    ).astype(np.int16)

    tools_str = "box_zoom,pan,reset,save,wheel_zoom"
    fig = figure(
        title_location="below",
        frame_width=nx,
        frame_height=ny,
        x_axis_label=None,
        y_axis_label=None,
        x_range=x_range,
        y_range=y_range,
        tools=tools_str,
        active_drag="box_zoom",
    )
    fig.grid.visible = False
    # Lay out responsively inside a gridplot row: fit to the column so the
    # gridplot toolbar (placed to the right) cannot push these figures past
    # the container's right edge.
    fig.width_policy = "fit"
    fig.height_policy = "fit"
    fig.title.text = title
    fig.title.align = "center"
    fig.title.text_font_size = "20px"
    fig.yaxis.visible = yaxis_visible
    fig.xaxis.major_label_text_font_size = "0pt"
    fig.yaxis.major_label_text_font_size = "0pt"
    fig.xaxis.minor_tick_line_color = None
    fig.xaxis.major_tick_line_color = None
    fig.yaxis.minor_tick_line_color = None
    fig.yaxis.major_tick_line_color = None

    source_data = ColumnDataSource(
        data=dict(
            image=[data.astype(np.float16)],
            x=[-nx // 2 * dsx],
            y=[-ny // 2 * dsy],
            dw=[nx * dsx],
            dh=[ny * dsy],
            bessel=[bessel],
        )
    )
    if phase is not None:
        source_data.add(
            data=[np.fmod(np.rad2deg(phase) + 360, 360).astype(np.float16)],
            name="phase",
        )
    if const_image_color:
        palette = (const_image_color,)
    else:
        palette = "Viridis256" if pseudo_color else "Greys256"
    color_mapper = LinearColorMapper(palette=palette)
    img_glyph = fig.image(
        source=source_data,
        image="image",
        color_mapper=color_mapper,
        x="x",
        y="y",
        dw="dw",
        dh="dh",
    )

    if tooltips is None:
        tooltips = [
            ("Res r", "Å"),
            ("Res y", "Å"),
            ("Res x", "Å"),
            ("Jn", "@bessel"),
            ("Val", "@image"),
        ]
    if hover_phase:
        tooltips.append(("Phase", "@phase °"))
    img_hover = HoverTool(
        renderers=[img_glyph], tooltips=tooltips, attachment="vertical"
    )
    fig.add_tools(img_hover)

    # MouseMove callback to compute resolution
    mousemove_code = """
    var x = cb_obj.x;
    var y = cb_obj.y;
    var resr = Math.round((1./Math.sqrt(x*x + y*y) + Number.EPSILON) * 100) / 100;
    var resy = Math.abs(Math.round((1./y + Number.EPSILON) * 100) / 100);
    var resx = Math.abs(Math.round((1./x + Number.EPSILON) * 100) / 100);
    hover.tooltips[0][1] = resr.toString() + " Å";
    hover.tooltips[1][1] = resy.toString() + " Å";
    hover.tooltips[2][1] = resx.toString() + " Å";
    """
    fig.js_on_event(
        MouseMove, CustomJS(args={"hover": fig.hover[0]}, code=mousemove_code)
    )
    return fig, source_data


def update_image_figure(fig, source_data, image_data, apix, title=""):
    """Update a Bokeh image figure with new data."""
    h, w = image_data.shape
    source_data.data = {
        "image": [image_data],
        "x": [-w // 2 * apix],
        "y": [-h // 2 * apix],
        "dw": [w * apix],
        "dh": [h * apix],
    }
    if title:
        fig.title.text = title


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def auto_correlation(data, sqrt=True, high_pass_fraction=0):
    """Compute auto-correlation of a 2D image via FFT."""
    fft = np.fft.rfft2(data)
    product = fft * np.conj(fft)
    if sqrt:
        product = np.sqrt(product)
    if 0 < high_pass_fraction <= 1:
        ny, nx = product.shape
        Y, X = np.meshgrid(
            np.arange(-ny // 2, ny // 2, dtype=float),
            np.arange(-nx // 2, nx // 2, dtype=float),
            indexing="ij",
        )
        Y /= ny // 2
        X /= nx // 2
        f2 = np.log(2) / (high_pass_fraction**2)
        filt = 1.0 - np.exp(-f2 * Y**2)
        product *= np.fft.fftshift(filt)
    corr = np.fft.fftshift(np.fft.irfft2(product))
    corr /= np.max(corr)
    return corr


@helicon.cache(cache_dir=str(helicon.cache_dir / "hill"), expires_after=7, verbose=0)
def symmetrize_transform_map(
    data,
    apix,
    twist_degree,
    rise_angstrom,
    csym=1,
    fraction=1.0,
    new_size=None,
    new_apix=None,
    axial_rotation=0,
    tilt=0,
):
    """Apply helical symmetry and transform a 3D map."""
    if new_apix is not None and new_apix > apix:
        data = helicon.low_high_pass_filter(data, low_pass_fraction=apix / new_apix)
    m = helicon.apply_helical_symmetry(
        data=data,
        apix=apix,
        twist_degree=twist_degree,
        rise_angstrom=rise_angstrom,
        csym=csym,
        new_size=new_size,
        new_apix=new_apix,
        fraction=fraction,
        cpu=helicon.available_cpu(),
    )
    if axial_rotation or tilt:
        m = helicon.transform_map(m, rot=axial_rotation, tilt=tilt)
    return m
