import platform
import numpy as np

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

    mask = (data != 0) * 1
    z_nonzeros = np.nonzero(mask)[0]
    z0 = np.min(z_nonzeros)
    z1 = np.max(z_nonzeros)
    z0 = max(z0, nz0 // 2 - int(nz0 * fraction + 0.5) // 2)
    z1 = min(nz0 - 1, min(z1, nz0 // 2 + int(nz0 * fraction + 0.5) // 2))

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


def get_resolution(fsc):
    from scipy import interpolate

    f = interpolate.interp1d(fsc[:, 0], fsc[:, 1], kind="cubic")
    s = np.linspace(0, fsc[-1, 0], num=10 * len(fsc) + 1, endpoint=True)
    fsc_tmp = f(s)
    try:
        res = 1 / s[np.where(fsc_tmp <= 0.143)[0][0]]
    except:
        res = 1 / s[-1]
    return res


# adapted from https://github.com/tdgrant1/denss/blob/3fbbefea45cb6d615e629e672d65440c46ac83da/saxstats/saxstats.py#L2185
def calc_fsc(map1, map2, apix):
    n = map1.shape[0]
    df = 1.0 / (apix * n)
    qx_ = np.fft.fftfreq(n) * n * df
    qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing="ij")
    qx_max = np.abs(qx).max()
    qr = np.sqrt(qx**2 + qy**2 + qz**2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0])
    nbins = int(qmax / qstep)
    qbins = np.linspace(0, nbins * qstep, nbins + 1)
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1
    F1 = np.fft.fftn(map1)
    F2 = np.fft.fftn(map2)
    from scipy import ndimage

    numerator = ndimage.sum(
        np.real(F1 * np.conj(F2)),
        labels=qbin_labels,
        index=np.arange(0, qbin_labels.max() + 1),
    )
    term1 = ndimage.sum(
        np.abs(F1) ** 2, labels=qbin_labels, index=np.arange(0, qbin_labels.max() + 1)
    )
    term2 = ndimage.sum(
        np.abs(F2) ** 2, labels=qbin_labels, index=np.arange(0, qbin_labels.max() + 1)
    )
    denominator = (term1 * term2) ** 0.5
    FSC = numerator / denominator
    qidx = np.where(qbins <= qx_max)
    return np.vstack((qbins[qidx], FSC[qidx])).T


def threshold_data(data, thresh_fraction=-1):
    if thresh_fraction < 0:
        return data
    thresh = data.max() * thresh_fraction
    ret = np.clip(data, thresh, None) - thresh
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


def get_cylindrical_mask(nz, ny, nx, rmin=0, rmax=-1, return_xyz=False):
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


def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0
    else:
        return np.sum(a * b) / norm


# https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
def find_elbow_point(curve):
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


def line_fit_projection(x, y, w=None, ref_i=0, return_xy_fit=False):
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


from sklearn.cluster import KMeans, AgglomerativeClustering


class AgglomerativeClusteringWithMinSize(AgglomerativeClustering):
    def __init__(
        self,
        min_cluster_size=2,
        n_clusters=2,
        metric="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        distance_threshold=None,
    ):
        super().__init__(
            n_clusters=n_clusters,
            metric=metric,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
        )
        self.min_cluster_size = min_cluster_size

    def fit(self, X, y=None):
        super().fit(X, y)
        labels = self.labels_

        while True:
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) < 3:
                break

            small_clusters = unique[counts < self.min_cluster_size]
            if len(small_clusters) == 0:
                break

            # If all clusters are small, merge the two smallest
            if len(small_clusters) == len(unique):
                smallest_two = unique[np.argsort(counts)[:2]]
                labels[labels == smallest_two[1]] = smallest_two[0]
                continue

            from sklearn.metrics import pairwise_distances

            distances = pairwise_distances(X)
            for small_cluster in small_clusters:
                small_cluster_points = np.where(labels == small_cluster)[0]
                for point in small_cluster_points:
                    # Find the nearest point not in a small cluster
                    valid_points = np.where(~np.isin(labels, small_clusters))[0]
                    nearest_point = valid_points[
                        np.argmin(distances[point, valid_points])
                    ]
                    labels[point] = labels[nearest_point]

        self.labels_ = labels
        self.n_clusters_ = len(np.unique(labels))

        return self
