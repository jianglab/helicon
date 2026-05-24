import logging
import numpy as np
import helicon

logger = logging.getLogger(__name__)

__all__ = [
    "calculate_structural_factor",
    "down_scale",
    "generate_tapering_filter",
    "low_high_pass_filter",
    "match_structural_factors",
    "normalize_mean_std",
    "normalize_min_max",
    "normalize_percentile",
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

    if data.ndim == 2:
        fft = np.fft.fft2(data)
        ny, nx = fft.shape
        Y, X = np.meshgrid(
            np.arange(ny, dtype=np.float32) - ny // 2,
            np.arange(nx, dtype=np.float32) - nx // 2,
            indexing="ij",
        )
        Y /= ny // 2
        X /= nx // 2
        R2 = X**2 + Y**2
    else:  # 3D case
        fft = np.fft.fftn(data)
        nz, ny, nx = fft.shape
        Z, Y, X = np.meshgrid(
            np.arange(nz, dtype=np.float32) - nz // 2,
            np.arange(ny, dtype=np.float32) - ny // 2,
            np.arange(nx, dtype=np.float32) - nx // 2,
            indexing="ij",
        )
        Z /= nz // 2
        Y /= ny // 2
        X /= nx // 2
        R2 = X**2 + Y**2 + Z**2

    if 0 < low_pass_fraction < 1:
        f2 = np.log(2) / (low_pass_fraction**2)
        filter_lp = np.exp(-f2 * R2)
        fft *= np.fft.fftshift(filter_lp)
    if 0 < high_pass_fraction < 1:
        f2 = np.log(2) / (high_pass_fraction**2)
        filter_hp = 1.0 - np.exp(-f2 * R2)
        fft *= np.fft.fftshift(filter_hp)
    ret = np.real(np.fft.ifftn(fft))
    return ret


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
