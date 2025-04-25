import platform
import numpy as np
import helicon

try:
    from numba import jit, set_num_threads, prange
except ImportError:
    helicon.color_print(
        f"WARNING: failed to load numba. The program will run correctly but will be much slower. Run 'pip install numba' to install numba and speed up the program"
    )

    def jit(*args, **kwargs):
        return lambda f: f

    def set_num_threads(n: int):
        return

    prange = range


@jit(
    nopython=True,
    cache=False,
    nogil=True,
    parallel=False if platform.system() == "Darwin" else True,
)
def apply_helical_symmetry(
    data,
    apix,
    twist_degree,
    rise_angstrom,
    csym=1,
    fraction=1.0,
    new_size=None,
    new_apix=None,
    cpu=1,
):
    if new_apix is None:
        new_apix = apix
    nz0, ny0, nx0 = data.shape
    if new_size != data.shape:
        nz1, ny1, nx1 = new_size
        nz2, ny2, nx2 = max(nz0, nz1), max(ny0, ny1), max(nx0, nx1)
        data_work = np.zeros((nz2, ny2, nx2), dtype=np.float32)
    else:
        data_work = np.zeros((nz0, ny0, nx0), dtype=np.float32)

    nz, ny, nx = data_work.shape
    w = np.zeros((nz, ny, nx), dtype=np.float32)

    hsym_max = max(1, int(nz * new_apix / rise_angstrom))
    hsyms = range(-hsym_max, hsym_max + 1)
    csyms = range(csym)

    profile_z = np.sum(np.sum(data, axis=-1), axis=-1)
    threshold = 0.01 * np.max(profile_z)
    non_zero_indices = np.where(profile_z > threshold)[0]
    z0 = non_zero_indices[0]
    z1 = non_zero_indices[-1]
    zmid = (z0 + z1) // 2 + (z0 + z1) % 2
    z0 = max(z0, zmid - int(nz0 * fraction + 0.5) // 2)
    z1 = min(z1, zmid + int(nz0 * fraction + 0.5) // 2)

    set_num_threads(cpu)

    for hi in hsyms:
        for k in prange(nz):
            k2 = ((k - nz // 2) * new_apix + hi * rise_angstrom) / apix + nz0 // 2
            if k2 < z0 or k2 >= z1:
                continue
            k2_floor, k2_ceil = int(np.floor(k2)), int(np.ceil(k2))
            wk = k2 - k2_floor

            for ci in csyms:
                rot = np.deg2rad(twist_degree * hi + 360 * ci / csym)
                m = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
                for j in prange(ny):
                    for i in prange(nx):
                        j2 = (
                            m[0, 0] * (j - ny // 2) + m[0, 1] * (i - nx / 2)
                        ) * new_apix / apix + ny0 // 2
                        i2 = (
                            m[1, 0] * (j - ny // 2) + m[1, 1] * (i - nx / 2)
                        ) * new_apix / apix + nx0 // 2

                        j2_floor, j2_ceil = int(np.floor(j2)), int(np.ceil(j2))
                        i2_floor, i2_ceil = int(np.floor(i2)), int(np.ceil(i2))
                        if j2_floor < 0 or j2_floor >= ny0 - 1:
                            continue
                        if i2_floor < 0 or i2_floor >= nx0 - 1:
                            continue

                        wj = j2 - j2_floor
                        wi = i2 - i2_floor

                        data_work[k, j, i] += (
                            (1 - wk)
                            * (1 - wj)
                            * (1 - wi)
                            * data[k2_floor, j2_floor, i2_floor]
                            + (1 - wk)
                            * (1 - wj)
                            * wi
                            * data[k2_floor, j2_floor, i2_ceil]
                            + (1 - wk)
                            * wj
                            * (1 - wi)
                            * data[k2_floor, j2_ceil, i2_floor]
                            + (1 - wk) * wj * wi * data[k2_floor, j2_ceil, i2_ceil]
                            + wk
                            * (1 - wj)
                            * (1 - wi)
                            * data[k2_ceil, j2_floor, i2_floor]
                            + wk * (1 - wj) * wi * data[k2_ceil, j2_floor, i2_ceil]
                            + wk * wj * (1 - wi) * data[k2_ceil, j2_ceil, i2_floor]
                            + wk * wj * wi * data[k2_ceil, j2_ceil, i2_ceil]
                        )
                        w[k, j, i] += 1.0
    mask = w > 0
    data_work = np.where(mask, data_work / w, data_work)
    if data_work.shape != new_size:
        nz1, ny1, nx1 = new_size
        data_work = data_work[
            nz // 2 - nz1 // 2 : nz // 2 + nz1 // 2,
            ny // 2 - ny1 // 2 : ny // 2 + ny1 // 2,
            nx // 2 - nx1 // 2 : nx // 2 + nx1 // 2,
        ]
    return data_work


def transform_map(data, scale=1.0, rot=0, tilt=0, psi=0, dx=0, dy=0, dz=0):
    """Transform a 3D volume by applying scaling, rotations and translations.

    Args:
        data (ndarray): Input 3D volume of shape (nz,ny,nx)
        scale (float): Scale factor to apply to all dimensions
        rot (float): First rotation angle around z-axis in degrees
        tilt (float): Second rotation angle around y-axis in degrees
        psi (float): Third rotation angle around new z-axis in degrees
        dx (float): Translation along x-axis
        dy (float): Translation along y-axis
        dz (float): Translation along z-axis

    Returns:
        ndarray: Transformed 3D volume
    """
    if (
        scale == 1
        and rot == 0
        and tilt == 0
        and psi == 0
        and dx == 0
        and dy == 0
        and dz == 0
    ):
        return data
    from scipy.spatial.transform import Rotation as R
    from scipy.ndimage import map_coordinates

    nz, ny, nx = data.shape
    k = np.arange(0, nz, dtype=np.int32) - nz // 2
    j = np.arange(0, ny, dtype=np.int32) - ny // 2
    i = np.arange(0, nx, dtype=np.int32) - nx // 2
    Z, Y, X = np.meshgrid(k, j, i, indexing="ij")
    if scale != 1.0:
        Z = Z * scale
        Y = Y * scale
        X = X * scale
    XYZ = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).transpose()
    xform = R.from_euler("zyz", (psi, tilt, rot), degrees=True)
    xyz = xform.apply(XYZ, inverse=False)
    xyz[:, 0] += nx // 2 - dx
    xyz[:, 1] += ny // 2 - dy
    xyz[:, 2] += nz // 2 - dz
    zyx = xyz[:, [2, 1, 0]].T
    ret = map_coordinates(data, zyx, order=3)
    ret = ret.reshape((nz, ny, nx))
    return ret


def transform_image(
    image,
    scale=1.0,
    rotation=0.0,
    rotation_center=None,
    pre_translation=(0.0, 0.0),
    post_translation=(0.0, 0.0),
    mode="constant",
    order=1,
):
    """
    Apply affine transformation with the image center as the reference point,
    with options for translation before and after the center-based transformation.

    Parameters:
    -----------
    image : ndarray
        Input image. Dimension 0 - y, 1 - x
    scale : float or tuple
        Scale factor. If float, same scale is applied to both axes.
        If tuple (sy, sx), different scales for each axis.
    rotation : float
        Rotation angle in degrees
    rotation_center : None or tuple
        (y, x) rotate around this position. default to image center (ny//2, nx//2)
    pre_translation : tuple
        (y, x) translation vector to apply BEFORE rotation/scaling
    post_translation : tuple
        (y, x) translation vector to apply AFTER rotation/scaling
    mode : str
        choice: constant, edge, symmetric, eflect, wrap
        Points outside the boundaries of the input are filled according to the given mode.
        Modes match the behaviour of numpy.pad.
    order : int
        The order of the spline interpolation, default is 1. The order has to be in the range 0-5.

    Returns:
    --------
    transform : AffineTransform
        The transformation object that can be used with skimage.transform.warp

    The transformation sequence is:
    1. Apply pre_translation
    2. Move to rotation_center
    3. Apply rotation and scaling
    4. Move back from rotation_center
    5. Apply post_translation
    """

    from skimage.transform import AffineTransform, warp

    if rotation_center is None:
        rotation_center = np.array((image.shape[0], image.shape[1])) / 2.0
    elif not isinstance(rotation_center, np.ndarray):
        rotation_center = np.array(rotation_center)

    if isinstance(scale, (int, float)):
        scale = np.array((scale, scale))

    pre_trans = AffineTransform(translation=pre_translation[::-1])
    to_center = AffineTransform(translation=-rotation_center[::-1])
    from_center = AffineTransform(translation=rotation_center[::-1])
    post_trans = AffineTransform(translation=post_translation[::-1])

    rotation_rad = np.deg2rad(rotation)
    center_transform = AffineTransform(scale=scale[::-1], rotation=rotation_rad)

    # Combine transformations in order:
    # pre_translation -> to_center -> rotation/scale -> from_center -> post_translation
    xform = pre_trans + to_center + center_transform + from_center + post_trans
    image.flags.writeable = True
    transformed = warp(image, xform.inverse, mode=mode, order=order)
    return transformed


def rotate_shift_image(
    data, angle=0, pre_shift=(0, 0), post_shift=(0, 0), rotation_center=None, order=1
):

    # pre_shift/rotation_center/post_shift: [y, x]
    if angle == 0 and pre_shift == [0, 0] and post_shift == [0, 0]:
        return data * 1.0
    ny, nx = data.shape
    if rotation_center is None:
        rotation_center = np.array((ny // 2, nx // 2), dtype=np.float32)
    ang = np.deg2rad(angle)
    m = np.array(
        [[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]], dtype=np.float32
    )
    pre_dy, pre_dx = pre_shift
    post_dy, post_dx = post_shift

    offset = -np.dot(
        m, np.array([post_dy, post_dx], dtype=np.float32).T
    )  # post_rotation shift
    offset += np.array(rotation_center, dtype=np.float32).T - np.dot(
        m, np.array(rotation_center, dtype=np.float32).T
    )  # rotation around the specified center
    offset += -np.array([pre_dy, pre_dx], dtype=np.float32).T  # pre-rotation shift

    from scipy.ndimage import affine_transform

    ret = affine_transform(data, matrix=m, offset=offset, order=order, mode="constant")
    return ret


def crop_center_z(data, n):
    assert data.ndim in [3]
    nz = data.shape[0]
    return data[nz // 2 - n // 2 : nz // 2 + n // 2 + n, :, :]


def crop_center(data, shape, center_offset=None):
    assert data.ndim in [2, 3]
    assert center_offset is None or len(center_offset) in [2, 3]
    assert data.ndim == len(shape)
    if data.shape == shape:
        return data
    if data.ndim == 2:
        ny, nx = data.shape
        my, mx = shape[-2:]
        if center_offset is not None:
            dy, dx = center_offset
        else:
            dy, dx = 0, 0
        y0 = max(0, ny // 2 + dy - my // 2)
        x0 = max(0, nx // 2 + dx - mx // 2)
        return data[y0 : min(ny, y0 + my), x0 : min(nx, x0 + mx)]
    else:
        nz, ny, nx = data.shape
        mz, my, mx = shape
        if center_offset is not None:
            dz, dy, dx = center_offset
        else:
            dz, dy, dx = 0, 0
        z0 = max(0, nz // 2 + dz - mz // 2)
        y0 = max(0, ny // 2 + dy - my // 2)
        x0 = max(0, nx // 2 + dx - mx // 2)
        return data[z0 : min(nz, z0 + mz), y0 : min(ny, y0 + my), x0 : min(nx, x0 + mx)]


def pad_to_size(data, shape):
    assert data.ndim in [2, 3]
    if data.shape == shape:
        return data
    ny, nx = data.shape[-2:]
    my, mx = shape[-2:]
    y_before = max(0, (my - ny) // 2)
    y_after = max(0, my - y_before - ny)
    x_before = max(0, (mx - nx) // 2)
    x_after = max(0, mx - x_before - nx)
    if data.ndim == 2:
        ret = np.pad(
            data, pad_width=((y_before, y_after), (x_before, x_after)), mode="constant"
        )
    else:
        nz = data.shape[0]
        mz = shape[0]
        z_before = max(0, (mz - nz) // 2)
        z_after = max(0, mz - z_before - nz)
        ret = np.pad(
            data,
            pad_width=((z_before, z_after), (y_before, y_after), (x_before, x_after)),
            mode="constant",
        )
    return ret


def get_clip(image, y0, x0, height, width):
    clip = np.zeros(shape=(height, width), dtype=image.dtype)
    y0_clipped = max(0, y0)
    x0_clipped = max(0, x0)
    y1_clipped = min(y0 + height, image.shape[0])
    x1_clipped = min(x0 + width, image.shape[1])
    clip[y0_clipped - y0 : y1_clipped - y0, x0_clipped - x0 : x1_clipped - x0] = image[
        y0_clipped:y1_clipped, x0_clipped:x1_clipped
    ]
    return clip


def get_clip3d(data, z0, y0, x0, nz, ny, nx):
    clip = np.zeros(shape=(nz, ny, nx), dtype=data.dtype)
    z0_clipped = max(0, z0)
    y0_clipped = max(0, y0)
    x0_clipped = max(0, x0)
    z1_clipped = min(z0 + nz, data.shape[0])
    y1_clipped = min(y0 + ny, data.shape[1])
    x1_clipped = min(x0 + nx, data.shape[2])
    clip[
        z0_clipped - z0 : z1_clipped - z0,
        y0_clipped - y0 : y1_clipped - y0,
        x0_clipped - x0 : x1_clipped - x0,
    ] = data[z0_clipped:z1_clipped, y0_clipped:y1_clipped, x0_clipped:x1_clipped]
    return clip


def get_rotated_clip(image, y0, x0, y1, x1, width, order=1):
    # image: numpy 2D array of shape (ny, nx)
    # y0, x0: coordinate of the starting point
    # y1, x1: coordinate of the ending point
    # order: interpolation order

    dy = y1 - y0
    dx = x1 - x0
    angle = np.atan2(dy, dx)
    length = np.sqrt(dy**2 + dx**2)
    x_steps = np.linspace(0, length, int(length))
    y_steps = np.linspace(-width / 2, width / 2, width)
    X, Y = np.meshgrid(x_steps, y_steps)
    X_rot = X * np.cos(angle) - Y * np.sin(angle) + x0
    Y_rot = X * np.sin(angle) + Y * np.cos(angle) + y0

    from scipy.ndimage import map_coordinates

    coords = np.stack([Y_rot, X_rot])
    result = map_coordinates(image, coords, order=order)

    return result


def fft_crop(data, output_size=None):
    if output_size is None or data.shape == output_size:
        return data

    assert data.ndim in (2, 3), f"ERROR: only 2-D images and 3-D maps are supported"
    assert data.ndim == len(output_size)

    if data.ndim == 2:
        ny, nx = data.shape
        ony, onx = output_size
        assert ony <= ny and onx <= nx
        fft = np.fft.rfft2(data)  # shape = (ny//2+1, nx//2+1)
        fft_truncated = np.fft.fftshift(
            np.fft.fftshift(fft, axes=0)[
                ny // 2 - ony // 2 : ny // 2 + ony // 2, : onx // 2 + 1
            ],
            axes=0,
        )
        data_downnscaled = np.fft.irfft2(fft_truncated)
        return data_downnscaled
    elif data.ndim == 3:
        nz, ny, nx = data.shape
        onz, ony, onx = output_size
        assert onz <= nz and ony <= ny and onx <= nx
        fft = np.fft.rfftn(data)  # shape = (ny//2+1, nx//2+1)
        fft_truncated = np.fft.fftshift(
            np.fft.fftshift(fft, axes=(0, 1))[
                nz // 2 - onz // 2 : nz // 2 + onz // 2,
                ny // 2 - ony // 2 : ny // 2 + ony // 2,
                : onx // 2 + 1,
            ],
            axes=(0, 1),
        )
        data_downnscaled = np.fft.irfft2(fft_truncated)
        return data_downnscaled


def fft_rescale(data, apix=1.0, cutoff_res=None, output_size=None):
    if data.ndim == 2:
        if cutoff_res:
            cutoff_res_y, cutoff_res_x = cutoff_res
        else:
            cutoff_res_y, cutoff_res_x = 2 * apix, 2 * apix
        if output_size:
            ony, onx = output_size
        else:
            ony, onx = data.shape
        freq_y = np.fft.fftfreq(ony) * 2 * apix / cutoff_res_y
        freq_x = np.fft.fftfreq(onx) * 2 * apix / cutoff_res_x
        Y, X = np.meshgrid(freq_y, freq_x, indexing="ij")
        Y = (2 * np.pi * Y).flatten(order="C")
        X = (2 * np.pi * X).flatten(order="C")

        from finufft import nufft2d2

        fft = nufft2d2(x=Y, y=X, f=data.astype(np.complex128), eps=1e-6)
        fft = fft.reshape((ony, onx))

        # phase shifts for real-space shifts by half of the image box in both directions
        phase_shift = np.ones(fft.shape)
        phase_shift[1::2, :] *= -1
        phase_shift[:, 1::2] *= -1
        fft *= phase_shift
        # now fft has the same layout and phase origin (i.e. np.fft.ifft2(fft) would obtain original image)
        return fft
    elif data.ndim == 3:
        if cutoff_res:
            cutoff_res_z, cutoff_res_y, cutoff_res_x = cutoff_res
        else:
            cutoff_res_z, cutoff_res_y, cutoff_res_x = 2 * apix, 2 * apix, 2 * apix
        if output_size:
            onz, ony, onx = output_size
        else:
            onz, ony, onx = data.shape
        freq_z = np.fft.fftfreq(onz) * 2 * apix / cutoff_res_z
        freq_y = np.fft.fftfreq(ony) * 2 * apix / cutoff_res_y
        freq_x = np.fft.fftfreq(onx) * 2 * apix / cutoff_res_x
        Z, Y, X = np.meshgrid(freq_z, freq_y, freq_x, indexing="ij")
        Z = (2 * np.pi * Z).flatten(order="C")
        Y = (2 * np.pi * Y).flatten(order="C")
        X = (2 * np.pi * X).flatten(order="C")

        from finufft import nufft3d2

        fft = nufft3d2(x=Z, y=Y, z=X, f=data.astype(np.complex128), eps=1e-6)
        fft = fft.reshape((onz, ony, onx))

        # phase shifts for real-space shifts by half of the image box in both directions
        phase_shift = np.ones(fft.shape)
        phase_shift[1::2, :, :] *= -1
        phase_shift[:, 1::2, :] *= -1
        phase_shift[:, :, 1::2] *= -1
        fft *= phase_shift
        # now fft has the same layout and phase origin (i.e. np.fft.ifft2(fft) would obtain original image)
        return fft


def compute_power_spectra(
    data,
    apix,
    cutoff_res=None,
    output_size=None,
    log=True,
    low_pass_fraction=0,
    high_pass_fraction=0,
):
    fft = fft_rescale(data, apix=apix, cutoff_res=cutoff_res, output_size=output_size)
    fft = np.fft.fftshift(fft)  # shift fourier origin from corner to center

    if log:
        pwr = np.log1p(np.abs(fft))
    else:
        pwr = np.abs(fft)
    if 0 < low_pass_fraction < 1 or 0 < high_pass_fraction < 1:
        pwr = helicon.low_high_pass_filter(
            pwr,
            low_pass_fraction=low_pass_fraction,
            high_pass_fraction=high_pass_fraction,
        )
    pwr = helicon.normalize_(pwr, percentile=(0, 100))

    phase = np.angle(fft, deg=False)
    return pwr, phase


def compute_phase_difference_across_meridian(phase):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
    phase_diff = phase * 0
    phase_diff[..., 1:] = phase[..., 1:] - phase[..., 1:][..., ::-1]
    phase_diff = np.rad2deg(
        np.arccos(np.cos(phase_diff))
    )  # set the range to [0, 180]. 0 -> even order, 180 - odd order
    return phase_diff
