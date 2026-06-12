from __future__ import annotations

import numpy as np
import helicon

__all__ = [
    "calc_fsc",
    "calc_fsc_from_fft",
    "calc_fsc_per_shell",
    "cosine_similarity",
    "cross_correlation_coefficient",
    "estimate_helix_rotation_center_diameter",
    "find_elbow_point",
    "get_cylindrical_mask",
    "is_3d",
    "is_amyloid",
    "line_fit_projection",
    "twist2pitch",
    "estimate_inter_segment_distance",
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


def estimate_helix_rotation_center_diameter(
    data: np.ndarray,
    estimate_rotation: bool = True,
    estimate_center: bool = True,
    threshold: float = 0,
) -> tuple[float, float, int]:
    """Estimate helix rotation, center, and diameter from a 2D image.

    Uses region labelling on a thresholded image to identify the helix region,
    then estimates orientation, centroid, and bounding-box diameter.

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
    from skimage.measure import label, regionprops
    from skimage.morphology import closing
    import helicon

    if estimate_rotation:
        bw = closing(data > threshold, mode="ignore")
        label_image = label(bw)
        props = regionprops(label_image=label_image, intensity_image=data)
        props.sort(key=lambda x: x.area, reverse=True)
        angle = (
            np.rad2deg(props[0].orientation) + 90
        )  # relative to +x axis, counter-clockwise
        if abs(angle) > 90:
            angle -= 180
        rotation = helicon.set_to_periodic_range(angle, min=-180, max=180)
        data_rotated = helicon.transform_image(image=data, rotation=rotation)
    else:
        rotation = 0.0
        data_rotated = data

    bw = closing(data_rotated > threshold, mode="ignore")
    label_image = label(bw)
    props = regionprops(label_image=label_image, intensity_image=data_rotated)
    props.sort(key=lambda x: x.area, reverse=True)
    minr, minc, maxr, maxc = props[0].bbox
    diameter = maxr - minr + 1

    if estimate_center:
        center = props[0].centroid
    else:
        ny, nx = data.shape
        center = (ny // 2, nx // 2)
    shift_y = data.shape[0] // 2 - center[0]

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


from .clustering import AgglomerativeClusteringWithMinSize  # noqa: F401
from .alignment import align_images  # noqa: F401
