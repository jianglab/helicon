import numpy as np
import helicon


def calculate_structural_factor(data, apix, return_fft=False):
    """
    Calculate the 1D structural factor, which is the rotational average of the FFT amplitude squared.

    Parameters:
    data (np.ndarray): Input 2D or 3D data array.

    Returns:
    qbins (np.ndarray): Binned q values.
    structural_factor (np.ndarray): Rotational average of the FFT amplitude squared.
    """

    if data.ndim == 2:
        ny, nx = data.shape
        qy, qx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
        F = np.fft.fft2(data)
    elif data.ndim == 3:
        nz, ny, nx = data.shape
        qz, qy, qx = np.meshgrid(
            np.fft.fftfreq(nz), np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij"
        )
        F = np.fft.fftn(data)
    else:
        raise ValueError("Input data must be a 2D or 3D array.")

    amplitude_squared = F.real**2 + F.imag**2

    if data.ndim == 2:
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


def set_structural_factors(data, apix, target_bins, target_structural_factors):
    """
    Scale the structural factors of the data array to match the target structural factors.

    Parameters:
    data (np.ndarray): The input data array (2D or 3D) whose structural factors will be changed.
    apix (float): The pixel size of the input data array.
    target_bins (np.ndarray): The q-value bins for the target structural factors.
    target_structural_factors (np.ndarray): The target structural factors to use.

    Returns:
    np.ndarray: The modified data after scaling the structural factors of the target structural factors
    """

    qbins, structural_factor, fft = calculate_structural_factor(
        data, apix, return_fft=True
    )

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


def match_structural_factors(data, apix, data_target, apix_target):
    """
    Scale the structural factors of the data array to match those of the target data array.

    Parameters:
    data (np.ndarray): The input data array (2D or 3D) whose structural factors will be changed.
    apix (float): The pixel size of the input data array.
    data_target (np.ndarray): The data array (2D or 3D) whose structural factors will be used as the target.
    apix_target (float): The pixel size of the target data array.

    Returns:
    np.ndarray: The modified data after scaling the structural factors of the target array
    """

    target_bins, target_structural_factors = calculate_structural_factor(
        data_target, apix_target, return_fft=False
    )
    return set_structural_factors(data, apix, target_bins, target_structural_factors)


def threshold_data(data, thresh_fraction=-1):
    if thresh_fraction < 0:
        return data
    thresh = data.max() * thresh_fraction
    ret = np.clip(data, thresh, None) - thresh
    return ret


def low_high_pass_filter(data, low_pass_fraction=0, high_pass_fraction=0):
    fft = np.fft.fft2(data)
    ny, nx = fft.shape
    Y, X = np.meshgrid(
        np.arange(ny, dtype=np.float32) - ny // 2,
        np.arange(nx, dtype=np.float32) - nx // 2,
        indexing="ij",
    )
    Y /= ny // 2
    X /= nx // 2
    if 0 < low_pass_fraction < 1:
        f2 = np.log(2) / (low_pass_fraction**2)
        filter_lp = np.exp(-f2 * (X**2 + Y**2))
        fft *= np.fft.fftshift(filter_lp)
    if 0 < high_pass_fraction < 1:
        f2 = np.log(2) / (high_pass_fraction**2)
        filter_hp = 1.0 - np.exp(-f2 * (X**2 + Y**2))
        fft *= np.fft.fftshift(filter_hp)
    ret = np.abs(np.fft.ifft2(fft))
    return ret


def down_scale(data, target_apix, apix_orig):
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
            helicon.color_print(
                f"WARNING: the input image pixel size ({apix_orig}) is larger than --target_apix2d={target_apix}. Down-scaling skipped"
            )
    return data
