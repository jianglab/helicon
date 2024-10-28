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


def transform_map(data, scale=1.0, tilt=0, psi=0, dy_pixel=0):
    if scale == 1 and tilt == 0 and psi == 0 and dy_pixel == 0:
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
    ZYX = np.vstack((Z.ravel(), Y.ravel(), X.ravel())).transpose()
    xform = R.from_euler("yz", (tilt, psi), degrees=True)
    zyx = xform.apply(ZYX, inverse=False)
    zyx[:, 0] += nz // 2
    zyx[:, 1] += ny // 2 - dy_pixel
    zyx[:, 2] += nx // 2
    ret = map_coordinates(data, zyx.T, order=3)
    ret = ret.reshape((nz, ny, nx))
    return ret


def transform_image(
    image,
    scale=1.0,
    rotation=0,
    pre_translation=(0, 0),
    post_translation=(0, 0),
    mode="constant",
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
    pre_translation : tuple
        (y, x) translation vector to apply BEFORE rotation/scaling
    post_translation : tuple
        (y, x) translation vector to apply AFTER rotation/scaling
    mode : str
        choice: constant, edge, symmetric, eflect, wrap
        Points outside the boundaries of the input are filled according to the given mode.
        Modes match the behaviour of numpy.pad.

    Returns:
    --------
    transform : AffineTransform
        The transformation object that can be used with skimage.transform.warp

    The transformation sequence is:
    1. Apply pre_translation
    2. Move to center
    3. Apply rotation and scaling
    4. Move back from center
    5. Apply post_translation
    """

    from skimage.transform import AffineTransform, warp

    center = np.array((image.shape[1], image.shape[0])) / 2.0

    if isinstance(scale, (int, float)):
        scale = (float(scale), float(scale))

    pre_trans = AffineTransform(translation=pre_translation[::-1])
    to_center = AffineTransform(translation=-center)
    from_center = AffineTransform(translation=center)
    post_trans = AffineTransform(translation=post_translation[::-1])

    rotation_rad = np.deg2rad(rotation)
    center_transform = AffineTransform(scale=scale[::-1], rotation=rotation_rad)

    # Combine transformations in order:
    # pre_translation -> to_center -> rotation/scale -> from_center -> post_translation
    xform = pre_trans + to_center + center_transform + from_center + post_trans
    transformed = warp(image, xform.inverse, mode=mode)
    return transformed

def rotate_shift_image(
    data, angle=0, pre_shift=(0, 0), post_shift=(0, 0), rotation_center=None, order=1):

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


def crop_center(data, shape):
    assert data.ndim in [2, 3]
    if data.shape == shape:
        return data
    ny, nx = data.shape[-2:]
    my, mx = shape[-2:]
    y0 = max(0, ny // 2 - my // 2)
    x0 = max(0, nx // 2 - mx // 2)
    if data.ndim == 2:
        return data[y0 : min(ny, y0 + my), x0 : min(nx, x0 + mx)]
    else:
        nz = data.shape[0]
        mz = shape[0]
        z0 = max(0, nz // 2 - mz // 2)
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
