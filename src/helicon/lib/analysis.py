import numpy as np
import helicon


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


def cross_correlation_coefficient(a, b):
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    norm = np.sqrt(np.sum((a - mean_a) ** 2) * np.sum((b - mean_b) ** 2))
    if norm == 0:
        return 0
    else:
        return np.sum((a - mean_a) * (b - mean_b)) / norm


def cosine_similarity(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0
    else:
        return np.sum(a * b) / norm


def align_images(
    image_moving,
    image_ref,
    angle_range=30,
    check_polarity=True,
    check_flip=True,
    return_aligned_moving_image=False,
):
    if check_flip:
        result = align_images(
            image_moving=image_moving,
            image_ref=image_ref,
            check_flip=False,
            return_aligned_moving_image=return_aligned_moving_image,
        )

        image_moving_flip = image_moving[::-1, :]
        result_flip = align_images(
            image_moving=image_moving_flip,
            image_ref=image_ref,
            check_flip=False,
            return_aligned_moving_image=return_aligned_moving_image,
        )
        if result_flip[2] > result[2]:
            return (True, *result_flip)
        else:
            return (False, *result)

    import numpy as np
    from skimage.registration import phase_cross_correlation
    from skimage.transform import rotate
    from scipy.ndimage import shift

    tapering_filter_moving = helicon.generate_tapering_filter(
        image_size=image_moving.shape, fraction_start=[0.8, 0.8]
    )
    tapering_filter_ref = helicon.generate_tapering_filter(
        image_size=image_ref.shape, fraction_start=[0.8, 0.8]
    )
    image_moving_work = helicon.threshold_data(
        tapering_filter_moving * image_moving, thresh_fraction=-1.0
    )
    image_ref_work = helicon.threshold_data(
        tapering_filter_ref * image_ref, thresh_fraction=0.0
    )
    padded_image_moving = helicon.pad_to_size(image_moving_work, image_ref_work.shape)

    from scipy.optimize import minimize_scalar
    from skimage.transform import rotate

    best = [1e10, 0, 0, None]

    def rotation_score(angle):
        rotated_padded_image_moving = rotate(padded_image_moving, angle)
        shift_cartesian, error, diffphase = phase_cross_correlation(
            reference_image=image_ref_work,
            moving_image=rotated_padded_image_moving,
            disambiguate=True,
        )
        shifted_rotated_padded_image_moving = shift(
            rotated_padded_image_moving, shift=shift_cartesian
        )
        score = -cross_correlation_coefficient(
            image_ref_work, shifted_rotated_padded_image_moving
        )
        if score < best[0]:
            best[0] = score
            best[1] = angle
            best[2] = shift_cartesian
            best[3] = shifted_rotated_padded_image_moving
        return score

    minimize_scalar(
        rotation_score, bounds=(-angle_range, angle_range), method="bounded"
    )
    if check_polarity:
        minimize_scalar(
            rotation_score,
            bounds=(180 - angle_range, 180 + angle_range),
            method="bounded",
        )
    (
        _,
        rotation_angle_degree,
        shift_cartesian,
        shifted_rotated_padded_image_moving,
    ) = best

    mask = shifted_rotated_padded_image_moving > 0.1 * np.max(
        shifted_rotated_padded_image_moving
    )
    similarity_score = cross_correlation_coefficient(
        shifted_rotated_padded_image_moving[mask], image_ref_work[mask]
    )

    padded_image_moving = helicon.pad_to_size(image_moving, image_ref_work.shape)
    rotated_padded_image_moving = rotate(padded_image_moving, rotation_angle_degree)
    shifted_rotated_padded_image_moving = shift(
        rotated_padded_image_moving, shift=shift_cartesian
    )

    if return_aligned_moving_image:
        return (
            rotation_angle_degree,
            shift_cartesian,
            similarity_score,
            shifted_rotated_padded_image_moving,
        )
    else:
        return rotation_angle_degree, shift_cartesian, similarity_score


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
