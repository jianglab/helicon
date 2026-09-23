import logging
from dataclasses import dataclass

import numpy as np
import helicon

logger = logging.getLogger(__name__)

__all__ = [
    "HelicalBackground",
    "background_offset",
    "helical_background",
    "calculate_structural_factor",
    "down_scale",
    "generate_tapering_filter",
    "low_high_pass_filter",
    "match_structural_factors",
    "normalize_mean_std",
    "normalize_min_max",
    "normalize_percentile",
    "randomize_phases_lowpass",
    "set_structural_factors",
    "threshold_data",
]


def calculate_structural_factor(
    data: np.ndarray,
    apix: float,
    thresh: float | None = None,
    mask: np.ndarray | None = None,
    return_fft: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the 1D structural factor, which is the rotational average of the FFT amplitude squared.

    Parameters
    ----------
    data : np.ndarray
        Input 2D or 3D data array.
    apix : float
        Pixel size.
    thresh : float, optional
        Threshold value applied via ``threshold_data`` before calculation.
    mask : np.ndarray, optional
        Mask to apply to the data before calculation.
    return_fft : bool, optional
        If True, also return the FFT of the data. Defaults to False.

    Returns
    -------
    qbins : np.ndarray
        Binned q values.
    structural_factor : np.ndarray
        Rotational average of the FFT amplitude squared.
    F : np.ndarray, optional
        FFT of the data. Only returned if ``return_fft`` is True.
    """

    if thresh:
        data_work = threshold_data(data, thresh_value=thresh)
    else:
        data_work = data
    if mask is not None:
        data_work = data_work * mask

    if data_work.ndim == 2:
        ny, nx = data_work.shape
        qy, qx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
        F = np.fft.fft2(data_work)
    elif data_work.ndim == 3:
        nz, ny, nx = data_work.shape
        qz, qy, qx = np.meshgrid(
            np.fft.fftfreq(nz), np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij"
        )
        F = np.fft.fftn(data_work)
    else:
        raise ValueError("Input data must be a 2D or 3D array.")

    amplitude_squared = F.real**2 + F.imag**2

    if data_work.ndim == 2:
        qr = np.sqrt(qx**2 + qy**2) / apix
    else:
        qr = np.sqrt(qx**2 + qy**2 + qz**2) / apix

    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0])
    nbins = int(qmax / qstep) // 2 * 2

    qbins = np.linspace(0, nbins * qstep, nbins)
    qbin_labels = np.searchsorted(qbins, qr, "right") - 1

    structural_factor = np.zeros(nbins)
    for i in range(nbins):
        structural_factor[i] = np.sum(amplitude_squared[qbin_labels == i])

    if return_fft:
        return qbins, structural_factor, F
    else:
        return qbins, structural_factor


def set_structural_factors(
    data: np.ndarray,
    apix: float,
    target_bins: np.ndarray,
    target_structural_factors: np.ndarray,
    thresh: float | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Scale the structural factors of the data array to match the target structural factors.

    Parameters
    ----------
    data : np.ndarray
        The input data array (2D or 3D) whose structural factors will be changed.
    apix : float
        The pixel size of the input data array.
    target_bins : np.ndarray
        The q-value bins for the target structural factors.
    target_structural_factors : np.ndarray
        The target structural factors to use.
    thresh : float, optional
        Threshold value applied before calculating structural factors.
    mask : np.ndarray, optional
        Mask to apply before calculating structural factors.

    Returns
    -------
    np.ndarray
        The modified data after scaling the structural factors to match the target.
    """

    qbins, structural_factor, fft = calculate_structural_factor(
        data, apix, thresh=thresh, mask=mask, return_fft=True
    )
    if mask is not None:
        fft = np.fft.fftn(data)

    from scipy import interpolate

    interp_func = interpolate.interp1d(
        target_bins, target_structural_factors, bounds_error=False, fill_value=0
    )
    structural_factor_target_interp = interp_func(qbins)

    ratio = np.zeros_like(structural_factor)
    nonzeros = np.nonzero(structural_factor)
    ratio[nonzeros] = np.sqrt(
        structural_factor_target_interp[nonzeros] / structural_factor[nonzeros]
    )

    if data.ndim == 2:
        ny, nx = data.shape
        qy, qx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
        qr = np.sqrt(qx**2 + qy**2) / apix
    elif data.ndim == 3:
        nz, ny, nx = data.shape
        qz, qy, qx = np.meshgrid(
            np.fft.fftfreq(nz), np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij"
        )
        qr = np.sqrt(qx**2 + qy**2 + qz**2) / apix
    else:
        raise ValueError("Input data must be a 2D or 3D array.")

    interp_func = interpolate.interp1d(qbins, ratio, bounds_error=False, fill_value=0)
    ratio_interp = interp_func(qr)

    modified_data = np.fft.ifftn(fft * ratio_interp)

    return np.real(modified_data)


def match_structural_factors(
    data: np.ndarray,
    apix: float,
    data_target: np.ndarray,
    apix_target: float,
    thresh: float | None = None,
    thresh_target: float | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Scale the structural factors of the data array to match those of the target data array.

    Parameters
    ----------
    data : np.ndarray
        The input data array (2D or 3D) whose structural factors will be changed.
    apix : float
        The pixel size of the input data array.
    data_target : np.ndarray
        The data array (2D or 3D) whose structural factors will be used as the target.
    apix_target : float
        The pixel size of the target data array.
    thresh : float, optional
        Threshold value applied to the input data before calculation.
    thresh_target : float, optional
        Threshold value applied to the target data before calculation.
    mask : np.ndarray, optional
        Mask to apply before calculation.

    Returns
    -------
    np.ndarray
        The modified data after scaling the structural factors to match the target array.
    """

    target_bins, target_structural_factors = calculate_structural_factor(
        data_target, apix_target, thresh=thresh_target, mask=mask, return_fft=False
    )
    return set_structural_factors(
        data, apix, target_bins, target_structural_factors, thresh=thresh, mask=mask
    )


def normalize_min_max(data: np.ndarray, min: float = 0, max: float = 1) -> np.ndarray:
    """Normalize data to a specified range using min-max scaling.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    min : float, optional
        Minimum value of the output range. Defaults to 0.
    max : float, optional
        Maximum value of the output range. Defaults to 1.

    Returns
    -------
    np.ndarray
        Data scaled to [``min``, ``max``].
    """
    data_min = data.min()
    data_max = data.max()
    if data_max == data_min:
        return data
    return (max - min) * (data - data_min) / (data_max - data_min)


def normalize_mean_std(data: np.ndarray, mean: float = 0, std: float = 1) -> np.ndarray:
    """Normalize data to a specified mean and standard deviation.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    mean : float, optional
        Desired mean of the output. Defaults to 0.
    std : float, optional
        Desired standard deviation of the output. Defaults to 1.

    Returns
    -------
    np.ndarray
        Data normalized to the specified mean and standard deviation.
    """
    data_std = data.std()
    if data_std == 0:
        return data
    data_mean = data.mean()
    return (data - data_mean) / data_std


def normalize_percentile(
    data: np.ndarray, percentile: tuple[float, float] = (0, 100)
) -> np.ndarray:
    """Normalize data to [0, 1] using percentile-based clipping.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    percentile : tuple of float, optional
        Lower and upper percentile values for clipping. Defaults to ``(0, 100)``.

    Returns
    -------
    np.ndarray
        Data scaled to [0, 1] with outliers clipped at the specified percentiles.
    """
    p0, p1 = percentile
    vmin, vmax = sorted(np.percentile(data, (p0, p1)))
    if vmax == vmin:
        return data
    return (data - vmin) / (vmax - vmin)


def background_offset(
    data: np.ndarray, clip: float = 3.0, n_iter: int = 6, min_nonzero: float = 0.5
) -> float:
    """The level the solvent sits at, so it can be moved to zero.

    Cryo-EM maps are normalised by whatever software wrote them, so zero
    means something different in every one. Anything that treats zero as
    "no density" -- thresholding, masking, a contour -- is therefore making
    an assumption the file may not honour. EMD-1427 is the cautionary case:
    its solvent sits at +1.26 and its tube interior at -3, so discarding
    everything below zero throws away the interior and keeps the solvent.

    Estimated by sigma clipping, which converges on the solvent because
    structure is the minority of a box. For a helical map prefer
    :func:`helical_background`, which reads the solvent from the geometry and
    reports when it cannot: on EMD-15538 sigma clipping gives -0.166 where
    the axial radial profile never falls below -0.09.

    Returns 0.0 for a map that has been masked, where the solvent has
    already been set to exactly zero and there is nothing left to measure:
    among 60 helical EMDB entries, 63% were masked tightly enough that under
    a fifth of the box is non-zero, and what clipping finds inside such a
    mask is structure rather than solvent.

    Parameters
    ----------
    data : np.ndarray
        The map.
    clip : float, optional
        Clipping threshold in standard deviations. Defaults to 3.
    n_iter : int, optional
        Maximum clipping iterations. Defaults to 6.
    min_nonzero : float, optional
        A map with a smaller non-zero fraction is taken to be masked, and
        already referenced to zero. Defaults to 0.5.

    Returns
    -------
    float
        The offset to subtract, or 0.0 when the map carries no measurable
        background.
    """
    work = np.asarray(data, dtype=np.float64).ravel()
    if work.size == 0:
        return 0.0
    if float((work != 0).mean()) < min_nonzero:
        return 0.0

    keep = np.ones(work.shape, dtype=bool)
    mean = float(work.mean())
    for _ in range(n_iter):
        subset = work[keep]
        if subset.size == 0:
            break
        mean = float(subset.mean())
        sigma = float(subset.std())
        if sigma <= 0:
            break
        updated = np.abs(work - mean) <= clip * sigma
        if int(updated.sum()) == int(keep.sum()):
            break
        keep = updated
    return mean


@dataclass
class HelicalBackground:
    """The solvent level of a helical map, and how it was found.

    Attributes
    ----------
    mean : float
        The background level, in the map's own units. Zero when it could not
        be determined, so subtracting it is always safe.
    sigma : float
        Standard deviation of the solvent voxels, or 0.0 when there are none
        to measure (a masked map, or an undetermined background).
    radius : int or None
        Innermost bin of the solvent region that was checked, in pixels from
        the axis; None when the level did not come from the profile.
    method : str
        ``"edge"`` (the outer bins of the radial profile are flat, so the box
        edge is solvent), ``"constant"`` (they are exactly constant, as outside
        a mask), ``"masked"`` (most of the box is exactly zero), or
        ``"undetermined"``.
    """

    mean: float
    sigma: float
    radius: int | None
    method: str


def helical_background(
    data: np.ndarray,
    tolerance: float = 0.03,
    min_bins: int = 6,
    min_fraction: float = 0.08,
) -> HelicalBackground:
    """The solvent level of a helical map, from its axial radial profile.

    Cryo-EM maps are normalised by whatever software wrote them, so zero means
    something different in every file, and anything that treats zero as "no
    density" -- a threshold, a mask, the z-extent search inside the helical
    symmetrisation -- inherits that chaos. EMD-19855 is an unmasked map whose
    solvent sits just below zero; every one of its slices sums negative, and
    that alone emptied the symmetrised map.

    The level is read exactly as the HI3D tab reads it before estimating a
    filament's radial range: :func:`helicon.compute_radial_profile` projects
    the map along the helical axis and averages it about the axis, and the
    background is the mean of the last three bins. Where the box edge is
    solvent that agrees with fitting the whole outer plateau -- -1.64e-4
    against -1.50e-4 on EMD-19855, -0.0045 against -0.0048 on EMD-4426.

    What this adds is a check that the edge *is* solvent. The outer bins must
    be flat: a straight line through them may change by at most
    ``tolerance`` of the filament's contrast, and no bin may stray from it by
    more. Otherwise no level is reported rather than a wrong one. Two kinds of
    map fail it: a box too small to reach the solvent -- EMD-1427's tube is
    wider than its box, and its edge bins sit on the tube's skirt at +2.37 --
    and a filament whose negative halo is still recovering at the box edge,
    as on EMD-1444 and EMD-15538.

    Parameters
    ----------
    data : np.ndarray
        The map, ``(nz, ny, nx)`` with the helical axis along z and through
        the centre of the box.
    tolerance : float, optional
        Allowed drift and scatter of the outer bins, as a fraction of the
        filament's contrast. Defaults to 0.03.
    min_bins, min_fraction : int, float, optional
        How many outer bins must be flat: at least this many, and at least
        this fraction of the radius. Defaults 6 and 0.08.

    Returns
    -------
    HelicalBackground
        ``mean`` is 0.0 whenever the level could not be determined, so
        ``data - helical_background(data).mean`` is always safe.
    """
    from .analysis import compute_radial_profile

    data = np.asarray(data)
    profile = compute_radial_profile(data).astype(np.float64)
    n = len(profile)
    k = max(min_bins, int(np.ceil(min_fraction * n)))
    if n < max(k, 3) + 2:
        return HelicalBackground(0.0, 0.0, None, "undetermined")

    level = float(np.mean(profile[-3:]))  # HI3D's background
    contrast = float(np.max(np.abs(profile - level)))
    tail = profile[-k:]

    def solvent_sigma():
        ny, nx = data.shape[1:]
        yy, xx = np.indices((ny, nx))
        r = np.hypot(yy - ny // 2, xx - nx // 2)
        ring = (r >= n - k) & (r < n)
        return float(data[:, ring].std())

    # Outside a mask the outer bins are exactly constant -- nearly always
    # zero -- and that constant is the solvent.
    if contrast == 0 or float(np.ptp(tail)) <= 1e-9 * contrast:
        return HelicalBackground(float(tail[-1]), 0.0, n - k, "constant")

    r = np.arange(n - k, n, dtype=np.float64)
    slope, intercept = np.polyfit(r, tail, 1)
    drift = abs(slope) * (r[-1] - r[0])
    scatter = float(np.max(np.abs(tail - (slope * r + intercept))))
    if drift <= tolerance * contrast and scatter <= tolerance * contrast:
        return HelicalBackground(level, solvent_sigma(), n - k, "edge")
    if float((data != 0).mean()) < 0.5:
        # a mask wider than the inscribed circle: the solvent is still zero
        return HelicalBackground(0.0, 0.0, None, "masked")
    return HelicalBackground(0.0, 0.0, None, "undetermined")


def threshold_data(
    data: np.ndarray,
    thresh_fraction: float | None = None,
    thresh_value: float | None = None,
) -> np.ndarray:
    """Apply a threshold to data, zeroing values below the threshold.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    thresh_fraction : float, optional
        Threshold as a fraction of the data maximum. Must be >= 0.
    thresh_value : float, optional
        Absolute threshold value.

    Returns
    -------
    np.ndarray
        Thresholded data with values below the threshold set to zero.
    """
    if thresh_fraction is not None and thresh_fraction >= 0:
        thresh = data.max() * thresh_fraction
    elif thresh_value is not None:
        thresh = thresh_value
    else:
        return data
    ret = np.clip(data, thresh, None) - thresh
    return ret


def low_high_pass_filter(
    data: np.ndarray, low_pass_fraction: float = 0, high_pass_fraction: float = 0
) -> np.ndarray:
    """Apply a low-pass and/or high-pass Gaussian filter in Fourier space.

    Parameters
    ----------
    data : np.ndarray
        Input 2D or 3D data array.
    low_pass_fraction : float, optional
        Low-pass cutoff as a fraction of the Nyquist frequency. Defaults to 0
        (no filtering).
    high_pass_fraction : float, optional
        High-pass cutoff as a fraction of the Nyquist frequency. Defaults to 0
        (no filtering).

    Returns
    -------
    np.ndarray
        Filtered data.
    """
    if data.ndim not in [2, 3]:
        raise ValueError("Input data must be a 2D or 3D array.")

    # Real-input FFT, in the precision it was handed, with the radius built by
    # broadcasting. The straightforward version -- np.fft.fftn, a meshgrid per
    # axis, np.real of the inverse -- costs about ten times the volume it is
    # filtering, which on a 384^3 map measured 2.4 GB for a 226 MB input:
    #
    #   * numpy's FFT always promotes to complex128, so the spectrum alone was
    #     four times the float32 input; scipy's keeps single precision single;
    #   * only half the spectrum is needed for real input, so rfftn halves it
    #     again;
    #   * three full meshgrid arrays were built to make one radius; broadcasting
    #     three 1-D axes costs nothing;
    #   * and np.real() returns a VIEW, so the whole complex buffer stayed alive
    #     behind the result, which was also handed downstream with a stride of
    #     two -- slow for anything that walks it.
    #
    # Same filter, same output to within float32 rounding.
    from scipy import fft as _fft

    real_dtype = np.float32 if data.dtype == np.float32 else np.float64
    work = np.asarray(data, dtype=real_dtype)

    spectrum = _fft.rfftn(work)
    axes_k = [(np.fft.fftfreq(n).astype(real_dtype) * 2.0) for n in work.shape[:-1]]
    axes_k.append(np.fft.rfftfreq(work.shape[-1]).astype(real_dtype) * 2.0)

    R2 = None
    for axis, k in enumerate(axes_k):
        shape = [1] * work.ndim
        shape[axis] = k.size
        term = (k * k).reshape(shape)
        R2 = term if R2 is None else R2 + term

    if 0 < low_pass_fraction < 1:
        f2 = np.log(2) / (low_pass_fraction**2)
        spectrum *= np.exp(-f2 * R2)
    if 0 < high_pass_fraction < 1:
        f2 = np.log(2) / (high_pass_fraction**2)
        spectrum *= 1.0 - np.exp(-f2 * R2)

    return _fft.irfftn(spectrum, s=work.shape)


def down_scale(data: np.ndarray, target_apix: float, apix_orig: float) -> np.ndarray:
    """Down-scale an image to a larger pixel size (lower resolution).

    Parameters
    ----------
    data : np.ndarray
        Input 2D image.
    target_apix : float
        Desired output pixel size.
    apix_orig : float
        Original pixel size of the input data.

    Returns
    -------
    np.ndarray
        Down-scaled image with even dimensions. Returns the input unchanged if
        ``target_apix`` <= ``apix_orig``.
    """
    if target_apix == apix_orig:
        return data
    elif target_apix > apix_orig:
        scale_factor = apix_orig / target_apix
        from skimage.transform import rescale

        ny0, nx0 = data.shape
        data = rescale(data, scale_factor, anti_aliasing=True, order=3)
        ny, nx = data.shape
        ny = ny + ny % 2
        nx = nx + nx % 2
        data = helicon.pad_to_size(data, shape=(ny, nx))
    else:
        if target_apix < apix_orig:
            logger.warning(
                "the input image pixel size (%s) is larger than --target_apix2d=%s. Down-scaling skipped",
                apix_orig,
                target_apix,
            )
    return data


def generate_tapering_filter(
    image_size: tuple[int, int],
    fraction_start: list[float] = [0.8, 0.8],
    fraction_slope: float = 0.1,
) -> np.ndarray:
    """Generate a cosine-tapering edge filter.

    Parameters
    ----------
    image_size : tuple of int
        ``(ny, nx)`` dimensions of the output filter.
    fraction_start : list of float, optional
        ``[fy, fx]`` fractional position where tapering begins along each axis.
        Defaults to ``[0.8, 0.8]``.
    fraction_slope : float, optional
        Width of the cosine falloff as a fraction of the half-axis. Defaults to
        0.1.

    Returns
    -------
    np.ndarray
        Tapering filter of shape ``image_size`` with values in [0, 1].
    """
    ny, nx = image_size
    fy, fx = fraction_start
    if not (0 < fy < 1 or 0 < fx < 1):
        return np.ones((ny, nx))
    Y, X = np.meshgrid(
        np.arange(0, ny, dtype=np.float32) - ny // 2,
        np.arange(0, nx, dtype=np.float32) - nx // 2,
        indexing="ij",
    )
    filter = np.ones_like(Y)
    if 0 < fy < 1:
        Y = np.abs(Y / (ny // 2))
        inner = Y < fy
        outer = Y > fy + fraction_slope
        Y = (Y - fy) / fraction_slope
        Y = (1.0 + np.cos(Y * np.pi)) / 2.0
        Y[inner] = 1
        Y[outer] = 0
        filter *= Y
    if 0 < fx < 1:
        X = np.abs(X / (nx // 2))
        inner = X < fx
        outer = X > fx + fraction_slope
        X = (X - fx) / fraction_slope
        X = (1.0 + np.cos(X * np.pi)) / 2.0
        X[inner] = 1
        X[outer] = 0
        filter *= X
    return filter


def randomize_phases_lowpass(
    data: np.ndarray, apix: float, cutoff_res: float, return_fft: bool = False
):
    """Randomize Fourier phases below a resolution cutoff (low-pass).

    Implements the phase randomization described in Chen et al. (2013),
    Ultramicroscopy 135:24-35, equation 4. Phases at resolutions worse
    (smaller spatial frequency) than ``cutoff_res`` are randomized while
    preserving the original amplitudes.

    Parameters
    ----------
    data : np.ndarray
        Input 3D map.
    apix : float
        Pixel size in Angstroms.
    cutoff_res : float
        Resolution cutoff in Angstroms. Phases at spatial frequencies
        >= ``apix/cutoff_res`` (normalized) are randomized.
    return_fft : bool, optional
        If True, return the rfftn result instead of the real-space map.
        Useful for computing FSC directly without ifft+fft round-trip.
        Defaults to False.

    Returns
    -------
    np.ndarray
        Phase-randomized map (if ``return_fft=False``) or its rfftn
        result (if ``return_fft=True``).
    """
    from scipy.fft import rfftn

    F = rfftn(data, workers=-1)
    amp = np.abs(F)
    phase = np.angle(F)

    cutoff_freq2 = (apix / cutoff_res) ** 2
    k = np.fft.fftfreq(data.shape[-1])
    kr = np.fft.rfftfreq(data.shape[-1])
    k2 = k * k
    kr2 = kr * kr
    mask = (k2[:, None, None] + k2[None, :, None] + kr2[None, None, :]) >= cutoff_freq2

    random_phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, size=phase.shape))
    phase[mask] = np.angle(random_phases[mask])

    F_randomized = amp * np.exp(1j * phase)
    if return_fft:
        return F_randomized
    from scipy.fft import irfftn

    return irfftn(F_randomized, workers=-1)
