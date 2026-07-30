"""Core computational functions for HI3D helical indexing.

Extracted from HI3D.git/hi3d.py (Streamlit version) with all UI dependencies
removed for use in the Shiny-based Helicon Lab web app.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)


# ── Normalization ──────────────────────────────────────────────────────


def normalize(
    data: np.ndarray, percentile: tuple[float, float] = (0, 100)
) -> np.ndarray:
    """Min-max normalise *data* to [0, 1] using the given percentile range."""
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    span = vmax - vmin
    if span == 0:
        return np.zeros_like(data)
    return (data - vmin) / span


# ── Periodic range ─────────────────────────────────────────────────────


def set_to_periodic_range(v: float, vmin: float = -180, vmax: float = 180) -> float:
    """Wrap *v* into the interval [*vmin*, *vmax*)."""
    width = vmax - vmin
    tmp = math.fmod(v - vmin, width)
    if tmp >= 0:
        return tmp + vmin
    else:
        return tmp + vmax


# ── Geometric median ───────────────────────────────────────────────────


def geometric_median(X: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Geometric median of points in *X* (minimiser of sum of Euclidean distances).

    https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
    """
    from scipy.spatial.distance import cdist, euclidean

    y = np.mean(X, 0)
    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]
        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1
        y = y1


# ── 2D rotation/shift ──────────────────────────────────────────────────


def rotate_shift_image(
    data: np.ndarray,
    angle: float = 0,
    pre_shift: tuple[float, float] = (0, 0),
    post_shift: tuple[float, float] = (0, 0),
    rotation_center: np.ndarray | None = None,
    order: int = 1,
) -> np.ndarray:
    """Rotate and shift a 2D image.

    Parameters
    ----------
    data : (ny, nx) array
    angle : float
        Rotation angle in degrees.
    pre_shift : (dy, dx)
        Shift applied *before* rotation (pixels).
    post_shift : (dy, dx)
        Shift applied *after* rotation (pixels).
    rotation_center : (y, x) or None
        Centre of rotation; defaults to the image centre.
    order : int
        Spline interpolation order (passed to ``affine_transform``).

    Returns
    -------
    (ny, nx) array
    """
    if angle == 0 and pre_shift == (0, 0) and post_shift == (0, 0):
        return data * 1.0

    from scipy.ndimage import affine_transform

    ny, nx = data.shape
    if rotation_center is None:
        rotation_center = np.array((ny // 2, nx // 2), dtype=np.float32)

    ang = np.deg2rad(angle)
    m = np.array(
        [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]],
        dtype=np.float32,
    )
    pre_dy, pre_dx = pre_shift
    post_dy, post_dx = post_shift

    offset = -np.dot(m, np.array([post_dy, post_dx], dtype=np.float32).T)
    offset += np.array(rotation_center, dtype=np.float32).T - np.dot(
        m, np.array(rotation_center, dtype=np.float32).T
    )
    offset += -np.array([pre_dy, pre_dx], dtype=np.float32).T

    return affine_transform(data, matrix=m, offset=offset, order=order, mode="constant")


# ── Auto vertical-center ────────────────────────────────────────────────


def auto_vertical_center(
    image: np.ndarray, max_angle: float = 15
) -> tuple[float, float]:
    """Estimate rotation angle and X-shift to verticalise a 2D projection.

    Returns
    -------
    angle : float
        Rotation angle (degrees).
    dx : float
        X-shift (pixels) to centre the flipped image.
    """
    from scipy.optimize import fmin, minimize_scalar

    image_work = 1.0 * image

    def score_rotation(angle: float) -> float:
        tmp = rotate_shift_image(data=image_work, angle=angle)
        y_proj = tmp.sum(axis=0)
        percentiles = (100, 95, 90, 85, 80)
        y_values = np.percentile(y_proj, percentiles)
        return -np.sum(y_values)

    res = minimize_scalar(
        score_rotation, bounds=(-max_angle, max_angle), method="bounded"
    )
    res_90 = minimize_scalar(
        score_rotation, bounds=(90 - max_angle, 90 + max_angle), method="bounded"
    )

    angle = res.x if res.fun < res_90.fun else res_90.x

    # refine rotation+shift jointly
    def score_rotation_shift(x):
        a, dy, dx = x
        tmp1 = rotate_shift_image(data=image_work, angle=a, pre_shift=(dy, dx))
        tmp2 = rotate_shift_image(data=image_work, angle=a + 180, pre_shift=(dy, dx))
        tmps = [tmp1, tmp2, tmp1[::-1, :], tmp2[::-1, :], tmp1[:, ::-1], tmp2[:, ::-1]]
        tmp_mean = sum(tmps) / len(tmps)
        err = sum(np.sum(np.abs(t - tmp_mean)) for t in tmps)
        return err / (len(tmps) * image_work.size)

    res = fmin(score_rotation_shift, x0=(angle, 0, 0), xtol=1e-2, disp=0)
    angle = res[0]
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    # refine dx
    from scipy.ndimage import center_of_mass

    image_work = rotate_shift_image(data=image_work, angle=angle)
    y = np.sum(image_work, axis=0)
    y -= y.min()
    n = len(y)
    cx = int(round(center_of_mass(y)[0]))
    max_shift = abs((cx - n // 2) * 2) + 3

    import scipy.interpolate as interpolate

    xp = np.arange(3 * n)
    f = interpolate.interp1d(xp, np.tile(y, 3), kind="cubic")

    def score_shift(dx: float) -> float:
        x_tmp = xp[n : 2 * n] - dx
        tmp = f(x_tmp)
        return float(np.sum(np.abs(tmp - tmp[::-1])))

    res = minimize_scalar(score_shift, bounds=(-max_shift, max_shift), method="bounded")
    dx = res.x + (0.0 if n % 2 else 0.5)
    return angle, dx


# ── 3D map transformation ──────────────────────────────────────────────


def transform_map(
    data: np.ndarray,
    shift_x: float = 0,
    shift_y: float = 0,
    shift_z: float = 0,
    angle_x: float = 0,
    angle_y: float = 0,
) -> np.ndarray:
    """Rotate and shift a 3D map.

    Parameters
    ----------
    data : (nz, ny, nx) array
    shift_x, shift_y, shift_z : float
        Translations along each axis (pixels).
    angle_x, angle_y : float
        Rotation around X and Y axes (degrees).

    Returns
    -------
    (nz, ny, nx) array
    """
    if not (shift_x or shift_y or angle_x or angle_y):
        return data

    from scipy.ndimage import affine_transform
    from scipy.spatial.transform import Rotation as R

    # Convention: xyz in scipy ↔ zyx in cryoEM maps
    rot = R.from_euler("zy", [-angle_x, angle_y], degrees=True)
    m = rot.as_matrix()
    nx, ny, nz = data.shape
    bcenter = np.array((nx // 2, ny // 2, nz // 2), dtype=m.dtype)
    offset = bcenter.T - np.dot(m, bcenter.T) + np.array([shift_z, shift_y, -shift_x])
    return affine_transform(data, matrix=m, offset=offset, mode="nearest")


# ── Change MRC map axis order ──────────────────────────────────────────


def change_mrc_map_crs_order(
    data: np.ndarray,
    current_order: list[int],
    target_order: list[int] | None = None,
) -> np.ndarray:
    """Re-order map axes from *current_order* to *target_order*.

    Orders are MRC column/row/section numbers (mapc/mapr/maps).
    Default target is [1, 2, 3] → (x, y, z).
    """
    if target_order is None:
        target_order = [1, 2, 3]
    if current_order == target_order:
        return data
    map_crs_to_np = {1: 2, 2: 1, 3: 0}
    cur_np = [map_crs_to_np[int(i)] for i in current_order]
    tgt_np = [map_crs_to_np[int(i)] for i in target_order]
    return np.moveaxis(data, cur_np, tgt_np)


# ── Cylindrical projection ─────────────────────────────────────────────


def cylindrical_projection(
    map3d: np.ndarray,
    da: float = 1.0,
    dz: float = 1.0,
    dr: float = 1.0,
    rmin: float = 0,
    rmax: float = -1,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Compute cylindrical projection of a 3D map.

    Parameters
    ----------
    map3d : (nz, ny, nx) array
    da : float
        Angular step size (degrees).
    dz : float
        Axial step size (pixels).
    dr : float
        Radial step size (pixels).
    rmin, rmax : float
        Radial range (pixels).  ``rmax <= rmin`` → use min(nx//2, ny//2).
    interpolation_order : int
        Spline order for ``map_coordinates``.

    Returns
    -------
    cylproj : (nz_out, n_theta) array
    """
    assert map3d.shape[0] > 1
    nz, ny, nx = map3d.shape
    if rmax <= rmin:
        rmax = min(nx // 2, ny // 2)
    assert rmin < rmax

    theta = (np.arange(0, 360, da, dtype=np.float32) - 90) * np.pi / 180.0
    n_theta = len(theta)
    mid = nz // 2
    z = np.arange(
        max(0, mid - n_theta // 2 * dz),
        min(nz, mid + n_theta // 2 * dz),
        dz,
        dtype=np.float32,
    )

    cylproj = np.zeros((len(z), n_theta), dtype=np.float32)
    for r in np.arange(rmin, rmax, dr, dtype=np.float32):
        z_grid, theta_grid = np.meshgrid(z, theta, indexing="ij", copy=False)
        y_grid = ny // 2 + r * np.sin(theta_grid)
        x_grid = nx // 2 + r * np.cos(theta_grid)
        coords = np.vstack((z_grid.flatten(), y_grid.flatten(), x_grid.flatten()))
        img = map_coordinates(
            map3d, coords, order=interpolation_order, mode="nearest"
        ).reshape(z_grid.shape)
        cylproj += img * r

    cylproj = normalize(cylproj)
    return cylproj


# ── Make square shape ──────────────────────────────────────────────────


def make_square_shape(cylproj: np.ndarray) -> np.ndarray:
    """Pad or crop the cylindrical projection to make it square (for ACF)."""
    nz, na = cylproj.shape
    if nz < na:
        top = np.zeros((na // 2 - nz // 2, na))
        bot = np.zeros((na - nz - top.shape[0], na))
        ret = cylproj - cylproj[[0, -1], :].mean()
        ret = np.vstack((top, ret, bot))
    elif nz > na:
        row0 = nz // 2 - na // 2
        ret = cylproj[row0 : row0 + na]
    else:
        ret = cylproj
    return ret


# ── Auto-correlation function (ACF) ────────────────────────────────────


def auto_correlation(
    data: np.ndarray,
    sqrt_transform: bool = False,
    high_pass_fraction: float = 0,
) -> np.ndarray:
    """Compute auto-correlation function via FFT.

    Parameters
    ----------
    data : (nz, na) array
    sqrt_transform : bool
        Apply sqrt to the power spectrum before inverse FFT (SCF mode).
    high_pass_fraction : float
        Fractional high-pass filter radius (0 = no filter).

    Returns
    -------
    (nz, na) array
    """
    from scipy.signal import correlate2d

    fft = np.fft.rfft2(data)
    product = fft * np.conj(fft)
    if sqrt_transform:
        product = np.sqrt(product)
    if 0 < high_pass_fraction <= 1:
        nz, na = product.shape
        Z, A = np.meshgrid(
            np.arange(-nz // 2, nz // 2, dtype=float),
            np.arange(-na // 2, na // 2, dtype=float),
            indexing="ij",
        )
        Z /= nz // 2
        A /= na // 2
        f2 = np.log(2) / (high_pass_fraction**2)
        filt = 1.0 - np.exp(-f2 * Z**2)  # Z-direction only
        product *= np.fft.fftshift(filt)

    corr = np.fft.fftshift(np.fft.irfft2(product))
    corr -= np.median(corr, axis=1, keepdims=True)
    corr = normalize(corr)
    if sqrt_transform:
        corr = np.power(np.log1p(corr), 1.0 / 3)
        corr = normalize(corr)
    return corr


# ── Peak finding ───────────────────────────────────────────────────────


def find_peaks(
    acf: np.ndarray,
    da: float,
    dz: float,
    peak_width: float = 9.0,
    peak_height: float = 9.0,
    minmass: float = 1.0,
    max_peaks: int = 71,
) -> tuple[np.ndarray | None, Any | None]:
    """Locate peaks in the ACF using *trackpy*.

    Returns
    -------
    peaks : (N, 2) array or None
        Each row is (twist_degrees, rise_angstrom).
    masses : Series or None
        Peak quality metric from trackpy.
    """
    try:
        from trackpy import locate, refine_com
    except ImportError:
        logger.warning("trackpy not installed — cannot find peaks in ACF")
        return None, None

    diameter_height = int(peak_height / dz + 0.5) // 2 * 2 + 1
    diameter_width = int(peak_width / da + 0.5) // 2 * 2 + 1
    pad_width = diameter_width * 3
    acf2 = np.hstack((acf[:, -pad_width:], acf, acf[:, :pad_width]))

    params = []
    for hf, wf in ((1, 1), (1, 2), (0.5, 0.5), (0.5, 1)):
        h = int(diameter_height * hf + 0.5) // 2 * 2 + 1
        w = int(diameter_width * wf + 0.5) // 2 * 2 + 1
        params.append((h, w))

    results = []
    # try multiple diameter combos, reducing minmass if necessary
    while True:
        results = []
        for h, w in params:
            if h < 1 or w < 1:
                continue
            try:
                f = locate(
                    acf2, diameter=(h, w), minmass=minmass, separation=(h * 2, w * 2)
                )
                if len(f):
                    results.append((f["mass"].sum() * len(f) ** -0.5, len(f), f, h, w))
                    try:
                        fr = refine_com(
                            raw_image=acf2,
                            image=acf2,
                            radius=(h // 2, w // 2),
                            coords=f,
                        )
                        results.append(
                            (fr["mass"].sum() * len(fr) ** -0.5, len(fr), fr, h, w)
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            if len(results) and results[-1][1] > 31:
                break
        results.sort(key=lambda x: x[0], reverse=True)
        if len(results) and results[0][1] > 3:
            break
        minmass *= 0.9
        if minmass < 0.1:
            return None, None

    f = results[0][2].copy()
    f.loc[:, "x"] -= pad_width
    f = f.loc[(f["x"] >= 0) & (f["x"] < acf.shape[1])]
    f = f.sort_values("mass", ascending=False)[:max_peaks]

    peaks = np.zeros((len(f), 2), dtype=float)
    peaks[:, 0] = (f["x"].values - acf.shape[1] // 2) * da  # degrees
    peaks[:, 1] = (f["y"].values - acf.shape[0] // 2) * dz  # Angstrom
    return peaks, f["mass"]


# ── Consistent twist/rise/cn sets ──────────────────────────────────────


def _angle_difference(angle1: float, angle2: float) -> float:
    err = abs((angle1 - angle2) % 360)
    if err > 180:
        err -= 360
    return abs(err)


def _angle_mean(angle1: float, angle2: float) -> float:
    angles = np.deg2rad([angle1, angle2])
    return float(np.rad2deg(np.arctan2(np.sin(angles).sum(), np.cos(angles).sum())))


def _good_twist_rise_cn(
    twist: float, rise: float, cn: int, epsilon: float = 0.1
) -> bool:
    if abs(twist) > epsilon:
        if abs(rise) > epsilon:
            return True
        elif abs(rise * 360.0 / twist / cn) > epsilon:
            return True
        else:
            return False
    else:
        return abs(rise) > epsilon


def consistent_twist_rise_cn_pair(
    trc1: tuple[float, float, int] | None,
    trc2: tuple[float, float, int] | None,
    epsilon: float = 1.0,
) -> tuple[float, float, int] | None:
    """Check whether two (twist, rise, cn) triples are consistent.

    Returns the mean triple if they agree, else ``None``.
    """
    if trc1 is None or trc2 is None:
        return None
    twist1, rise1, cn1 = trc1
    twist2, rise2, cn2 = trc2
    if not _good_twist_rise_cn(twist1, rise1, cn1):
        return None
    if not _good_twist_rise_cn(twist2, rise2, cn2):
        return None
    if (
        cn1 == cn2
        and abs(rise2 - rise1) < epsilon
        and _angle_difference(twist1, twist2) < epsilon
    ):
        cn = cn1
        rise = (rise1 + rise2) / 2
        twist = _angle_mean(twist1, twist2)
        return twist, rise, int(cn)
    return None


def consistent_twist_rise_cn_sets(
    set1: list[tuple[float, float, int]],
    set2: list[tuple[float, float, int]],
    epsilon: float = 1.0,
) -> (
    tuple[tuple[float, float, int], tuple[float, float, int], tuple[float, float, int]]
    | None
):
    """Search two sets of (twist, rise, cn) for a consistent pair.

    Returns ``(mean_triple, triple_from_set1, triple_from_set2)`` or ``None``.
    """
    for trc1 in set1:
        for trc2 in set2:
            pair = consistent_twist_rise_cn_pair(trc1, trc2, epsilon=epsilon)
            if pair is not None:
                return (pair, trc1, trc2)
    return None


# ── Lattice fitting: helical-specific method ────────────────────────────


def get_helical_lattice(peaks: np.ndarray) -> tuple[float, float, int]:
    """Fit a helical lattice from peaks (method 1).

    Parameters
    ----------
    peaks : (N, 2) array
        Columns are (twist_degrees, rise_angstrom).

    Returns
    -------
    twist, rise, cn
    """
    if len(peaks) < 3:
        return (0.0, 0.0, 1)

    x = peaks[:, 0]
    y = peaks[:, 1]

    ys = np.sort(y)
    vys = ys[1:] - ys[:-1]
    vy = np.median(vys[np.abs(vys) > 1e-1])
    j = np.around(y / vy)
    nonzero = j != 0
    if np.count_nonzero(nonzero) == 0:
        return (0.0, 0.0, 1)
    rise = float(np.median(y[nonzero] / j[nonzero]))
    if np.isnan(rise):
        return (0.0, 0.0, 1)

    cn = 1
    js = np.rint(y / rise)
    spacing = []
    for j_val in sorted(set(js)):
        x_j = x[js == j_val]
        if len(x_j) > 1:
            x_j.sort()
            spacing += list(x_j[1:] - x_j[:-1])
    if spacing:
        best_spacing = max(0.01, float(np.median(spacing)))
        cn_f = 360.0 / best_spacing
        expected = 360.0 / round(cn_f)
        if abs(best_spacing - expected) / expected < 0.05:
            cn = int(round(cn_f))

    js = np.rint(y / rise)
    above_equator = js > 0
    if np.count_nonzero(above_equator) == 0:
        return (0.0, 0.0, 1)

    min_j = js[above_equator].min()
    vx = sorted(x[js == min_j] / min_j, key=abs)[0]
    x2 = x.reshape(-1, 1)
    xdiffs = x2 - x2.T
    y2 = y.reshape(-1, 1)
    ydiffs = y2 - y2.T
    selected = (np.rint(ydiffs / rise) == min_j) & (np.rint(xdiffs / vx) == min_j)
    best_vx = float(np.mean(xdiffs[selected]))
    if best_vx > 180:
        best_vx -= 360
    best_vx /= min_j
    twist = best_vx

    if np.isnan(twist):
        return (0.0, 0.0, 1)

    if cn > 1 and abs(twist) > 180.0 / cn:
        if twist < 0:
            twist += 360.0 / cn
        else:
            twist -= 360.0 / cn

    return (twist, rise, int(cn))


# ── Lattice fitting: generic 2D method ──────────────────────────────────


def get_generic_lattice(peaks: np.ndarray) -> tuple[float, float, int]:
    """Fit a generic 2D lattice from peaks (method 2).

    Parameters
    ----------
    peaks : (N, 2) array
        Columns are (twist_degrees, rise_angstrom).

    Returns
    -------
    twist, rise, cn
    """
    if len(peaks) < 3:
        return (0.0, 0.0, 1)

    from scipy.spatial import cKDTree as KDTree

    mindist = 10.0
    minang = 15.0
    epsilon = 0.5

    def _angle(v1, v2=None):
        p = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        p = np.clip(abs(p), 0, 1)
        return float(np.rad2deg(np.arccos(p)))

    def _distance(v1, v2):
        return math.hypot(v1[0] - v2[0], v1[1] - v2[1])

    def _on_equator(v, eps=0.5):
        return int(abs(v[1]) <= eps)

    def _pick_triplet(kdtree, index=-1):
        m = kdtree.data.shape[0]
        if index < 0:
            index = np.random.randint(0, m)
        origin = kdtree.data[index]
        distances, indices = kdtree.query(origin, k=m)
        first = None
        for i in range(1, m):
            v = kdtree.data[indices[i]]
            first = v
            break
        if first is None:
            return None, None, None
        second = None
        for j in range(i + 1, m):
            v = kdtree.data[indices[j]]
            ang = _angle(first - origin, v - origin)
            dist = _distance(first - origin, v - origin)
            if dist > mindist and ang > minang:
                second = v
                break
        return origin, first, second

    def _peaks2lattice(pts, va, vb, origin):
        A = np.vstack((va, vb)).T
        b_pts = (pts - origin).T
        x = np.linalg.solve(A, b_pts)
        NaNb = np.around(x)
        good = np.abs(x - NaNb).max(axis=0) < 0.1
        one = np.ones((1, int(NaNb[:, good].shape[1])))
        A2 = np.vstack((NaNb[:, good], one)).T
        p_out, *_ = np.linalg.lstsq(A2, pts[good, :], rcond=-1)
        va2 = p_out[0]
        vb2 = p_out[1]
        origin2 = p_out[2]
        residues = pts[good, :] - (
            NaNb[:, good].T * va2 + NaNb[:, good].T * vb2 + origin2
        )
        err = float(np.sqrt(np.sum(residues**2))) / len(pts)
        return {"NaNb": NaNb, "a": va2, "b": vb2, "origin": origin2, "err": err}

    kdt = KDTree(peaks)
    best = None
    min_err = 1e30
    for i in range(len(peaks)):
        origin, first, second = _pick_triplet(kdt, index=i)
        if first is None or second is None:
            continue
        va = first - origin
        vb = second - origin
        lattice = _peaks2lattice(peaks, va, vb, origin)
        lattice = _peaks2lattice(peaks, lattice["a"], lattice["b"], lattice["origin"])
        err = lattice["err"]
        if err < min_err:
            dist = _distance(lattice["a"], lattice["b"])
            ang = _angle(lattice["a"], lattice["b"])
            if dist > mindist and ang > minang:
                min_err = err
                best = lattice

    if best is None:
        # fallback: all peaks lie on a line
        from scipy.odr import Data, ODR, unilinear

        x = peaks[:, 0]
        y = peaks[:, 1]
        odr_data = Data(x, y)
        odr_obj = ODR(odr_data, unilinear)
        output = odr_obj.run()
        x2 = x + output.delta
        y2 = y + output.eps
        v0 = np.array([x2[-1] - x2[0], y2[-1] - y2[0]])
        n0 = np.linalg.norm(v0)
        if n0 > 0:
            v0 /= n0
        ref_i = 0
        t = (x2 - x2[ref_i]) * v0[0] + (y2 - y2[ref_i]) * v0[1]
        t.sort()
        spacings = t[1:] - t[:-1]
        a = float(np.median(spacings[np.abs(spacings) > 1e-1]))
        a_vec = v0 * a
        if a_vec[1] < 0:
            a_vec *= -1
        best = {"a": a_vec, "b": a_vec}

    a = best["a"]
    b = best["b"]
    min_len = max(1.0, min(np.linalg.norm(a), np.linalg.norm(b)) * 0.9)
    vs_on_equator = []
    vs_off_equator = []
    maxI = 10
    for i in range(-maxI, maxI + 1):
        for j in range(-maxI, maxI + 1):
            if i == 0 and j == 0:
                continue
            v = i * a + j * b
            v[0] = set_to_periodic_range(v[0], -180, 180)
            if np.linalg.norm(v) > min_len:
                if v[1] < 0:
                    v *= -1
                if _on_equator(v, eps=epsilon):
                    vs_on_equator.append(v)
                else:
                    vs_off_equator.append(v)

    twist, rise, cn = 0.0, 0.0, 1
    if vs_on_equator:
        vs_on_equator.sort(key=lambda v: abs(v[0]))
        best_spacing = abs(vs_on_equator[0][0])
        cn_f = 360.0 / best_spacing
        expected = 360.0 / round(cn_f)
        if abs(best_spacing - expected) / expected < 0.05:
            cn = int(round(cn_f))
    if vs_off_equator:
        vs_off_equator.sort(key=lambda v: (abs(round(v[1] / epsilon)), abs(v[0])))
        twist, rise = vs_off_equator[0]
        if cn > 1 and abs(twist) > 180.0 / cn:
            if twist < 0:
                twist += 360.0 / cn
            else:
                twist -= 360.0 / cn
    return twist, rise, int(cn)


# ── Refine twist/rise via Nelder-Mead ──────────────────────────────────


def refine_twist_rise(
    acf_image: np.ndarray,
    da: float,
    dz: float,
    twist: float,
    rise: float,
    cn: int,
) -> tuple[float, float]:
    """Refine *twist*/*rise* by maximising ACF values at lattice positions.

    Uses Nelder-Mead optimisation against the ACF.

    Parameters
    ----------
    acf_image : (ny, nx) array
    da, dz : float
        Step sizes (degrees, Angstrom).
    twist, rise : float
        Initial estimates.
    cn : int
        Cyclic symmetry.

    Returns
    -------
    twist_opt, rise_opt
    """
    from scipy.optimize import minimize

    if rise <= 0:
        return twist, rise
    cn = int(cn)

    ny, nx = acf_image.shape
    try:
        npeak = max(3, min(100, int(ny / 2 / abs(rise) / 2)))
    except Exception:
        npeak = 3

    i = np.repeat(range(1, npeak), cn)
    w = np.power(i, 1.0 / 2)
    x_sym = np.tile(range(cn), npeak - 1) * 360.0 / cn

    def score(x):
        t, r = x
        px = np.fmod(nx // 2 + i * t / da + x_sym + npeak * nx, nx)
        py = ny // 2 + i * r / dz
        v = map_coordinates(acf_image, (py, px))
        return -float(np.sum(v * w))

    res = minimize(
        score,
        (twist, rise),
        method="nelder-mead",
        options={"xatol": 1e-4, "adaptive": True},
    )
    twist_opt, rise_opt = res.x
    twist_opt = set_to_periodic_range(twist_opt, -180, 180)
    return twist_opt, rise_opt


# ── Fit helical lattice (orchestrator) ─────────────────────────────────


def fit_helical_lattice(
    peaks: np.ndarray,
    acf: np.ndarray,
    da: float = 1.0,
    dz: float = 1.0,
) -> tuple[tuple[float, float, int], tuple[float, float, int]]:
    """Run both lattice fitting methods and return consistent solutions.

    Returns
    -------
    (twist1, rise1, cn1), (twist2, rise2, cn2)
    """
    if len(peaks) < 3:
        return (0.0, 0.0, 1), (0.0, 0.0, 1)

    trc1s: list[tuple[float, float, int]] = []
    trc2s: list[tuple[float, float, int]] = []
    consistent = False
    nmax = len(peaks) if len(peaks) % 2 else len(peaks) - 1
    for n in range(nmax, max(min(7, nmax) - 1, 3) - 1, -2):
        trc1 = get_helical_lattice(peaks[:n])
        trc2 = get_generic_lattice(peaks[:n])
        trc1s.append(trc1)
        trc2s.append(trc2)
        if consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0):
            consistent = True
            break

    if not consistent:
        for _ in range(100):
            if len(peaks) // 2 > 5:
                n = np.random.randint(5, len(peaks) // 2)
                choices = sorted(np.random.choice(range(2 * n), size=n, replace=False))
            else:
                n = np.random.randint(3, len(peaks))
                choices = sorted(
                    np.random.choice(range(len(peaks)), size=n, replace=False)
                )
            if 0 not in choices:
                choices = [0] + choices
            p_random = peaks[choices]
            trc1 = get_helical_lattice(p_random)
            trc2 = get_generic_lattice(p_random)
            trc1s.append(trc1)
            trc2s.append(trc2)
            if consistent_twist_rise_cn_sets([trc1], [trc2], epsilon=1.0):
                consistent = True
                break

    if not consistent:
        trc_mean = consistent_twist_rise_cn_sets(trc1s, trc2s, epsilon=1.0)
        if trc_mean:
            _, trc1, trc2 = trc_mean
        else:
            arr1 = np.array(trc1s)
            arr2 = np.array(trc2s)
            trc1 = tuple(geometric_median(arr1[:, :2])) + (int(np.median(arr1[:, 2])),)
            trc2 = tuple(geometric_median(arr2[:, :2])) + (int(np.median(arr2[:, 2])),)
            # ensure they are tuples
            trc1 = (float(trc1[0]), float(trc1[1]), int(trc1[2]))
            trc2 = (float(trc2[0]), float(trc2[1]), int(trc2[2]))

    twist1, rise1, cn1 = trc1
    twist1, rise1 = refine_twist_rise(acf, da, dz, twist1, rise1, cn1)
    twist2, rise2, cn2 = trc2
    twist2, rise2 = refine_twist_rise(acf, da, dz, twist2, rise2, cn2)

    return (twist1, rise1, int(cn1)), (twist2, rise2, int(cn2))


# ── Radial profile ─────────────────────────────────────────────────────


def compute_radial_profile(data: np.ndarray) -> np.ndarray:
    """Compute the radial (azimuthally-averaged) density profile of a 3D map.

    Parameters
    ----------
    data : (nz, ny, nx) array
        The map is averaged along Z, then a polar transform is performed.

    Returns
    -------
    rad_profile : (rmax,) array
        Radial profile in pixel units.
    """
    proj = data.mean(axis=0)
    ny, nx = proj.shape
    rmax = min(nx // 2, ny // 2)

    r = np.arange(0, rmax, 1, dtype=np.float32)
    theta = np.arange(0, 360, 1, dtype=np.float32) * np.pi / 180.0
    n_theta = len(theta)

    theta_grid, r_grid = np.meshgrid(theta, r, indexing="ij", copy=False)
    y_grid = ny // 2 + r_grid * np.sin(theta_grid)
    x_grid = nx // 2 + r_grid * np.cos(theta_grid)
    coords = np.vstack((y_grid.flatten(), x_grid.flatten()))
    polar = map_coordinates(proj, coords, order=1).reshape(r_grid.shape)
    return polar.mean(axis=0)


def estimate_radial_range(
    radprofile: np.ndarray, thresh_ratio: float = 0.1
) -> tuple[float, float]:
    """Estimate rmin/rmax from a radial profile.

    The threshold is computed as:

        background + (max - background) * thresh_ratio

    where background = mean of the last 3 bins.
    """
    background = float(np.mean(radprofile[-3:]))
    thresh = (float(radprofile.max()) - background) * thresh_ratio + background
    indices = np.nonzero(radprofile > thresh)[0]
    if len(indices) == 0:
        return 0.0, float(len(radprofile) - 1)
    return float(indices.min()), float(indices.max())


# ── Minimal grids ──────────────────────────────────────────────────────


def minimal_grids(map3d: np.ndarray, max_map_dim: int = 300) -> tuple[np.ndarray, int]:
    """Bin and crop a 3D map to fit within *max_map_dim* per side.

    Returns
    -------
    small_map, bin_factor
    """
    nz, ny, nx = map3d.shape
    n_min_xy = min(ny, nx)
    n_min_z = min(nz, n_min_xy)
    bin_factor = max(1, n_min_xy // max_map_dim + 1)
    ret = map3d[
        nz // 2 - n_min_xy // 2 : nz // 2 + n_min_xy // 2 : bin_factor,
        ny // 2 - n_min_xy // 2 : ny // 2 + n_min_xy // 2 : bin_factor,
        nx // 2 - n_min_z // 2 : nx // 2 + n_min_z // 2 : bin_factor,
    ]
    return ret, bin_factor


# ── EMDB URL helpers ────────────────────────────────────────────────────


def get_emdb_map_url(emdid: str) -> str:
    """Construct the EBI FTP URL for an EMDB map."""
    num = emdid.lower().split("emd-")[-1]
    return (
        f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/"
        f"EMD-{num}/map/emd_{num}.map.gz"
    )


def extract_emd_id(text: str) -> str | None:
    """Extract EMDB numeric ID from a filename or URL."""
    import re

    match = re.search(r".*emd_([0-9]+)\.map.*", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def is_amyloid(params: dict, cutoff: float = 6) -> bool:
    """Heuristic check if the EMDB entry is likely an amyloid."""
    if "twist" in params and "rise" in params:
        r = math.hypot(params["twist"], params["rise"])
        if r < cutoff:
            return True
        twist2 = abs(params["twist"]) - 180
        if math.hypot(twist2, params["rise"]) < cutoff:
            return True
    if "sample" in params:
        sample = params["sample"].lower()
        for target in ("tau", "synuclein", "amyloid", "tdp-43"):
            if target in sample:
                return True
    return False


# ── Bokeh figure helper (for use in the tab) ──────────────────────────


def generate_bokeh_figure(
    image: np.ndarray,
    dx: float,
    dy: float,
    title: str = "",
    title_location: str = "below",
    plot_width: int | None = None,
    plot_height: int | None = None,
    x_axis_label: str | None = "x",
    y_axis_label: str | None = "y",
    tooltips: list[tuple[str, str]] | None = None,
    show_angle_tooltip: bool = False,
    show_axis: bool = True,
    show_toolbar: bool = True,
    crosshair_color: str = "white",
    aspect_ratio: float | None = None,
):
    """Create a Bokeh figure displaying a 2D image with hover and crosshair.

    This is a pure helper (no Shiny/Streamlit dependency).  Returns a
    ``bokeh.plotting.figure`` ready to be rendered by ``shinywidgets``.
    """
    from bokeh.models import CrosshairTool, HoverTool, LinearColorMapper
    from bokeh.plotting import figure

    h, w = image.shape
    if aspect_ratio is None and (plot_width and plot_height):
        aspect_ratio = plot_width / plot_height

    tools = "box_zoom,crosshair,pan,reset,save,wheel_zoom"
    fig = figure(
        title_location=title_location,
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

    source_data = dict(
        image=[image],
        x=[-w // 2 * dx],
        y=[-h // 2 * dy],
        dw=[w * dx],
        dh=[h * dy],
    )
    color_mapper = LinearColorMapper(palette="Greys256")
    r = fig.image(
        source=source_data,
        image="image",
        color_mapper=color_mapper,
        x="x",
        y="y",
        dw="dw",
        dh="dh",
    )

    if tooltips is not None and show_angle_tooltip:
        tooltips = tooltips + [("angle", "°")]

    image_hover = HoverTool(renderers=[r], tooltips=tooltips)
    fig.add_tools(image_hover)

    if tooltips is not None and show_angle_tooltip:
        from bokeh.events import MouseMove
        from bokeh.models import CustomJS

        js_code = """
        var x = cb_obj.x;
        var y = cb_obj.y;
        var angle = Math.atan2(y, x) * 180 / Math.PI - 90;
        if (angle < -180) angle += 360;
        angle = Math.round(angle * 10) / 10;
        hover.tooltips[hover.tooltips.length - 1][1] = angle.toString() + "°";
        """
        mousemove = CustomJS(args={"hover": fig.hover[0]}, code=js_code)
        fig.js_on_event(MouseMove, mousemove)

    crosshairs = [t for t in fig.tools if isinstance(t, CrosshairTool)]
    for ch in crosshairs:
        ch.line_color = crosshair_color

    return fig
