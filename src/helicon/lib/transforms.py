from __future__ import annotations

import os
import platform

import numpy as np

# finufft uses a bundled FFTW for its plan stage. FFTW's planner keeps
# static/global state (wisdom cache) across plan create/destroy cycles, so
# repeated calls within the same process can corrupt FFTW internals and
# segfault.  This is especially visible when running a full test suite where
# each test creates and destroys finufft plans.  Setting OMP_NUM_THREADS=1
# prevents finufft from spawning multiple FFTW threads, avoiding the race.
os.environ.setdefault("OMP_NUM_THREADS", "1")

__all__ = [
    "apply_helical_symmetry",
    "compute_phase_difference_across_meridian",
    "compute_power_spectra",
    "crop_center",
    "crop_center_z",
    "fft_crop",
    "fft_rescale",
    "flip_hand",
    "get_clip",
    "get_clip3d",
    "get_rotated_clip",
    "pad_to_size",
    "rotate_shift_image",
    "transform_image",
    "transform_map",
]

from .filters import low_high_pass_filter, normalize_percentile

try:
    from numba import jit, set_num_threads, prange
except ImportError:
    import warnings

    warnings.warn(
        "numba not available; apply_helical_symmetry will be much slower. "
        "Install numba with 'pip install numba' or 'conda install numba'."
    )

    def jit(*args, **kwargs):
        return lambda f: f

    def set_num_threads(n: int):
        return

    prange = range


_USE_NUMBA_PARALLEL = platform.system() != "Darwin"


@jit(
    nopython=True,
    cache=False,
    nogil=True,
    parallel=_USE_NUMBA_PARALLEL,
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


def transform_map(
    data: np.ndarray,
    scale: float = 1.0,
    rot: float = 0,
    tilt: float = 0,
    psi: float = 0,
    dx: float = 0,
    dy: float = 0,
    dz: float = 0,
) -> np.ndarray:
    """Transform a 3D volume by applying scaling, rotations and translations.

    Parameters
    ----------
    data : np.ndarray
        Input 3D volume of shape (nz, ny, nx).
    scale : float, optional
        Scale factor to apply to all dimensions.
    rot : float, optional
        First rotation angle around z-axis in degrees.
    tilt : float, optional
        Second rotation angle around y-axis in degrees.
    psi : float, optional
        Third rotation angle around new z-axis in degrees.
    dx : float, optional
        Translation along x-axis.
    dy : float, optional
        Translation along y-axis.
    dz : float, optional
        Translation along z-axis.

    Returns
    -------
    np.ndarray
        Transformed 3D volume.
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
    xform = R.from_euler("ZYZ", (rot, tilt, psi), degrees=True)
    xyz = xform.apply(XYZ, inverse=False)
    xyz[:, 0] += nx // 2 - dx
    xyz[:, 1] += ny // 2 - dy
    xyz[:, 2] += nz // 2 - dz
    zyx = xyz[:, [2, 1, 0]].T
    ret = map_coordinates(data, zyx, order=3)
    ret = ret.reshape((nz, ny, nx))
    return ret


def transform_image(
    image: np.ndarray,
    scale: float | tuple[float, float] = 1.0,
    rotation: float = 0.0,
    rotation_center: tuple[float, float] | np.ndarray | None = None,
    pre_translation: tuple[float, float] = (0.0, 0.0),
    post_translation: tuple[float, float] = (0.0, 0.0),
    mode: str = "constant",
    order: int = 1,
) -> np.ndarray:
    """Apply affine transformation with the image center as the reference point,
    with options for translation before and after the center-based transformation.

    Parameters
    ----------
    image : np.ndarray
        Input image. Dimension 0 - y, 1 - x.
    scale : float or tuple, optional
        Scale factor. If float, same scale is applied to both axes.
        If tuple (sy, sx), different scales for each axis.
    rotation : float, optional
        Rotation angle in degrees.
    rotation_center : None or tuple, optional
        (y, x) rotate around this position. Defaults to image center (ny//2, nx//2).
    pre_translation : tuple, optional
        (y, x) translation vector to apply BEFORE rotation/scaling.
    post_translation : tuple, optional
        (y, x) translation vector to apply AFTER rotation/scaling.
    mode : str, optional
        Choice: constant, edge, symmetric, reflect, wrap.
        Points outside the boundaries of the input are filled according to
        the given mode. Modes match the behaviour of numpy.pad.
    order : int, optional
        The order of the spline interpolation, default is 1.
        The order has to be in the range 0-5.

    Returns
    -------
    np.ndarray
        The transformed image.

    Notes
    -----
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
    data: np.ndarray,
    angle: float = 0,
    pre_shift: tuple[int, int] | tuple[float, float] = (0, 0),
    post_shift: tuple[int, int] | tuple[float, float] = (0, 0),
    rotation_center: np.ndarray | None = None,
    order: int = 1,
) -> np.ndarray:
    """Rotate and shift a 2D image.

    Parameters
    ----------
    data : np.ndarray
        Input 2D image.
    angle : float, optional
        Rotation angle in degrees. Defaults to 0.
    pre_shift : tuple of (int, int), optional
        Pre-rotation shift (y, x). Defaults to (0, 0).
    post_shift : tuple of (int, int), optional
        Post-rotation shift (y, x). Defaults to (0, 0).
    rotation_center : np.ndarray, optional
        Center of rotation (y, x). Defaults to image center.
    order : int, optional
        Spline interpolation order. Defaults to 1.

    Returns
    -------
    np.ndarray
        Rotated and shifted image.
    """
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


def crop_center_z(data: np.ndarray, n: int) -> np.ndarray:
    """Crop a 3D volume to *n* slices along the Z axis, centered.

    Parameters
    ----------
    data : np.ndarray
        Input 3D volume.
    n : int
        Number of Z slices to keep.

    Returns
    -------
    np.ndarray
        Cropped volume.
    """
    assert data.ndim in [3]
    nz = data.shape[0]
    return data[nz // 2 - n // 2 : nz // 2 + n // 2 + n, :, :]


def crop_center(
    data: np.ndarray,
    shape: tuple[int, ...],
    center_offset: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Crop the central region of a 2D or 3D array.

    Parameters
    ----------
    data : np.ndarray
        Input 2D or 3D array.
    shape : tuple of int
        Target shape (mz, my, mx) for 3D or (my, mx) for 2D.
    center_offset : tuple of int, optional
        Offset from center (dz, dy, dx) or (dy, dx).

    Returns
    -------
    np.ndarray
        Cropped array.
    """
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
            dz, dy, dx = 0, 0, 0
        z0 = max(0, nz // 2 + dz - mz // 2)
        y0 = max(0, ny // 2 + dy - my // 2)
        x0 = max(0, nx // 2 + dx - mx // 2)
        return data[z0 : min(nz, z0 + mz), y0 : min(ny, y0 + my), x0 : min(nx, x0 + mx)]


def pad_to_size(data: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Pad a 2D or 3D array to a target size with zeros.

    Parameters
    ----------
    data : np.ndarray
        Input 2D or 3D array.
    shape : tuple of int
        Target shape.

    Returns
    -------
    np.ndarray
        Padded array.
    """
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


def get_clip(
    image: np.ndarray, y0: int, x0: int, height: int, width: int
) -> np.ndarray:
    """Extract a rectangular clip from a 2D image, zero-padding if out of bounds.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image.
    y0 : int
        Top row of the clip region.
    x0 : int
        Left column of the clip region.
    height : int
        Height of the clip.
    width : int
        Width of the clip.

    Returns
    -------
    np.ndarray
        Clipped region.
    """
    clip = np.zeros(shape=(height, width), dtype=image.dtype)
    y0_clipped = max(0, y0)
    x0_clipped = max(0, x0)
    y1_clipped = min(y0 + height, image.shape[0])
    x1_clipped = min(x0 + width, image.shape[1])
    clip[y0_clipped - y0 : y1_clipped - y0, x0_clipped - x0 : x1_clipped - x0] = image[
        y0_clipped:y1_clipped, x0_clipped:x1_clipped
    ]
    return clip


def get_clip3d(
    data: np.ndarray, z0: int, y0: int, x0: int, nz: int, ny: int, nx: int
) -> np.ndarray:
    """Extract a 3D clip from a volume, zero-padding if out of bounds.

    Parameters
    ----------
    data : np.ndarray
        Input 3D volume.
    z0 : int
        Starting Z index.
    y0 : int
        Starting Y index.
    x0 : int
        Starting X index.
    nz : int
        Number of Z slices.
    ny : int
        Height.
    nx : int
        Width.

    Returns
    -------
    np.ndarray
        Clipped 3D region.
    """
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


def get_rotated_clip(
    image: np.ndarray,
    y0: float,
    x0: float,
    y1: float,
    x1: float,
    width: int,
    order: int = 1,
) -> np.ndarray:
    """Extract a rotated rectangular clip from a 2D image.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image of shape (ny, nx).
    y0 : float
        Y coordinate of the starting point.
    x0 : float
        X coordinate of the starting point.
    y1 : float
        Y coordinate of the ending point.
    x1 : float
        X coordinate of the ending point.
    width : int
        Width of the extracted strip.
    order : int, optional
        Spline interpolation order. Defaults to 1.

    Returns
    -------
    np.ndarray
        Rotated clip of shape (width, length).
    """

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


def fft_crop(
    data: np.ndarray, output_size: tuple[int, ...] | None = None
) -> np.ndarray:
    """Crop an image or volume in Fourier space to a smaller size.

    Parameters
    ----------
    data : np.ndarray
        Input 2D or 3D array.
    output_size : tuple of int, optional
        Target size. Defaults to no cropping.

    Returns
    -------
    np.ndarray
        Fourier-cropped array.
    """
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


def fft_rescale(
    data: np.ndarray,
    apix: float = 1.0,
    cutoff_res: tuple[float, ...] | None = None,
    output_size: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Rescale an image or volume in Fourier space using NUFFT.

    Parameters
    ----------
    data : np.ndarray
        Input 2D or 3D array.
    apix : float, optional
        Pixel size in Angstroms. Defaults to 1.0.
    cutoff_res : tuple of float, optional
        Cutoff resolution(s) in Angstroms.
    output_size : tuple of int, optional
        Output size. Defaults to input size.

    Returns
    -------
    np.ndarray
        Fourier-rescaled data (Fourier coefficients).
    """
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


def flip_hand(data: np.ndarray, axis: str = "x") -> np.ndarray:
    """Flip the handedness of a 3D volume along the specified axis.

    Parameters
    ----------
    data : np.ndarray
        Input 3D volume of shape (nz, ny, nx).
    axis : str, optional
        Axis along which to flip ('x', 'y', or 'z').

    Returns
    -------
    np.ndarray
        Flipped 3D volume.
    """
    if axis == "x":
        return data[:, :, ::-1]
    elif axis == "y":
        return data[:, ::-1, :]
    elif axis == "z":
        return data[::-1, :, :]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


def compute_power_spectra(
    data: np.ndarray,
    apix: float,
    cutoff_res: tuple[float, ...] | None = None,
    output_size: tuple[int, ...] | None = None,
    log: bool = True,
    low_pass_fraction: float = 0,
    high_pass_fraction: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the power spectrum and phase of an image or volume.

    Parameters
    ----------
    data : np.ndarray
        Input 2D or 3D array.
    apix : float
        Pixel size in Angstroms.
    cutoff_res : tuple of float, optional
        Cutoff resolution(s) in Angstroms.
    output_size : tuple of int, optional
        Output size. Defaults to input size.
    log : bool, optional
        If True, apply log1p to the power spectrum. Defaults to True.
    low_pass_fraction : float, optional
        Low-pass filter fraction. Defaults to 0.
    high_pass_fraction : float, optional
        High-pass filter fraction. Defaults to 0.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (power_spectrum, phase) arrays.
    """
    fft = fft_rescale(data, apix=apix, cutoff_res=cutoff_res, output_size=output_size)
    fft = np.fft.fftshift(fft)  # shift fourier origin from corner to center

    if log:
        pwr = np.log1p(np.abs(fft))
    else:
        pwr = np.abs(fft)
    if 0 < low_pass_fraction < 1 or 0 < high_pass_fraction < 1:
        pwr = low_high_pass_filter(
            pwr,
            low_pass_fraction=low_pass_fraction,
            high_pass_fraction=high_pass_fraction,
        )
    pwr = normalize_percentile(pwr, percentile=(0, 100))

    phase = np.angle(fft, deg=False)
    return pwr, phase


def compute_phase_difference_across_meridian(phase: np.ndarray) -> np.ndarray:
    """Compute the phase difference across the meridian (Friedel symmetry check).

    Parameters
    ----------
    phase : np.ndarray
        Phase array from :func:`compute_power_spectra`.

    Returns
    -------
    np.ndarray
        Phase difference in degrees, mapped to [0, 180].
    """
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
    phase_diff = phase * 0
    phase_diff[..., 1:] = phase[..., 1:] - phase[..., 1:][..., ::-1]
    phase_diff = np.rad2deg(
        np.arccos(np.cos(phase_diff))
    )  # set the range to [0, 180]. 0 -> even order, 180 - odd order
    return phase_diff
