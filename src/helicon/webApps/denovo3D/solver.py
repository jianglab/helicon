"""Utility functions for de novo helical indexing and 3D reconstruction."""

import itertools, os, sys, pathlib, datetime, logging, joblib
from pathlib import Path
import numpy as np

import helicon

from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

try:
    from numba import jit, set_num_threads, prange
except ImportError:
    logger.warning(
        "failed to load numba. The program will run correctly but will be much slower. Run 'pip install numba' to install numba and speed up the program"
    )

    def jit(*args, **kwargs):
        return lambda f: f

    def set_num_threads(n: int):
        return

    prange = range

cache_dir = helicon.cache_dir / "denovo3D"


def lsq_reconstruct(
    projection_image,
    scale2d_to_3d,
    twist_degree,
    rise_pixel,
    csym=1,
    tilt_degree=0,
    psi_degree=0,
    dy_pixel=0,
    thresh_fraction=-1,
    positive_constraint=-1,
    reconstruct_diameter_3d_inner_pixel=0,
    reconstruct_diameter_2d_pixel=-1,
    reconstruct_diameter_3d_pixel=-1,
    reconstruct_length_2d_pixel=-1,
    reconstruct_length_3d_pixel=-1,
    sym_oversample=1,
    interpolation="nn",
    fsc_test=0,
    score_metric="cosine",
    target_apix2d=5.0,
    verbose=0,
    algorithm=dict(model="lsq"),
):
    """Build and solve the least-squares reconstruction system.

    Constructs the projection and helical symmetry matrices, solves for
    the 3D density, and optionally computes FSC by splitting the data.

    Parameters
    ----------
    projection_image : ndarray
        2D projection image.
    scale2d_to_3d : float
        Scale factor from 2D to 3D pixel size.
    twist_degree : float
        Helical twist in degrees.
    rise_pixel : float
        Helical rise in 3D pixels.
    csym : int, optional
        Cyclic symmetry. Defaults to 1.
    tilt_degree : float, optional
        Out-of-plane tilt. Defaults to 0.
    psi_degree : float, optional
        In-plane rotation. Defaults to 0.
    dy_pixel : float, optional
        Perpendicular shift in 2D pixels. Defaults to 0.
    thresh_fraction : float, optional
        Threshold fraction. Defaults to -1 (disabled).
    positive_constraint : int, optional
        Positive constraint mode. Defaults to -1 (auto).
    reconstruct_diameter_3d_inner_pixel : int, optional
        Inner diameter of 3D mask in pixels. Defaults to 0.
    reconstruct_diameter_2d_pixel : int, optional
        2D reconstruction diameter in pixels. Defaults to -1 (auto).
    reconstruct_diameter_3d_pixel : int, optional
        3D reconstruction diameter in pixels. Defaults to -1 (auto).
    reconstruct_length_2d_pixel : int, optional
        2D reconstruction length in pixels. Defaults to -1 (auto).
    reconstruct_length_3d_pixel : int, optional
        3D reconstruction length in pixels. Defaults to -1 (auto).
    sym_oversample : int, optional
        Symmetry oversampling factor. Defaults to 1.
    interpolation : str, optional
        Interpolation method. Defaults to "nn".
    fsc_test : int, optional
        FSC test mode. Defaults to 0 (disabled).
    score_metric : str, optional
        Scoring metric for ranking. Options: "cosine" for cosine similarity,
        "frc" for 2D Fourier Ring Correlation. Defaults to "cosine".
    target_apix2d : float, optional
        Target 2D pixel size in Angstroms. Used for FRC scoring. Defaults to 5.0.
    verbose : int, optional
        Verbosity level. Defaults to 0.
    algorithm : dict, optional
        Solver configuration. Defaults to ``{"model": "lsq"}``.

    Returns
    -------
    tuple of ((ndarray, ndarray or None, ndarray or None), float)
        (rec3d_full, rec3d_half1, rec3d_half2), score.
    """

    rmin = reconstruct_diameter_3d_inner_pixel / 2
    rmax = reconstruct_diameter_3d_pixel // 2 - 1

    mask = helicon.get_cylindrical_mask(
        nz=reconstruct_length_3d_pixel,
        ny=reconstruct_diameter_3d_pixel,
        nx=reconstruct_diameter_3d_pixel,
        rmin=rmin,
        rmax=rmax,
    )
    mz, my, mx = mask.shape
    assert mz == reconstruct_length_3d_pixel

    n_3d_voxels = np.count_nonzero(mask)
    n_2d_pixels = reconstruct_diameter_2d_pixel * reconstruct_length_2d_pixel
    max_equations = 2**26  # 64 million

    with helicon.Timer(f"build_A_data_matrix - {interpolation}", verbose=verbose > 10):
        A_data, b_data, b_data_pid = build_A_data_matrix(
            image=projection_image,
            scale2d_to_3d=scale2d_to_3d,
            twist_degree=twist_degree,
            rise_pixel=rise_pixel,
            csym=csym,
            tilt_degree=tilt_degree,
            psi_degree=psi_degree,
            dy_pixel=dy_pixel,
            reconstruct_diameter_2d_pixel=reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel=reconstruct_length_2d_pixel,
            reconstruct_diameter_3d_pixel=reconstruct_diameter_3d_pixel,
            reconstruct_diameter_3d_inner_pixel=reconstruct_diameter_3d_inner_pixel,
            reconstruct_length_3d_pixel=reconstruct_length_3d_pixel,
            min_projection_lines=min(
                max_equations, int(max(n_2d_pixels, n_3d_voxels) * sym_oversample)
            ),
            interpolation=interpolation,
            verbose=verbose,
        )

    with helicon.Timer(
        f"build_A_helical_sym_matrix - {interpolation}", verbose=verbose > 10
    ):
        A_hsym, b_hsym = build_A_helical_sym_matrix(
            nz=mz,
            ny=my,
            nx=mx,
            twist_degree=twist_degree,
            rise_pixel=rise_pixel,
            csym=csym,
            rmin=rmin,
            rmax=rmax,
            min_sym_pairs=min(
                max_equations, int(max(n_2d_pixels, n_3d_voxels) * sym_oversample)
            ),
            interpolation=interpolation,
            verbose=verbose,
        )

    def split_A_b(A, b, b_id, mode):
        if mode <= 0:  # no split, return the same data for both sets
            return (A, b), (A, b)

        if b_id is None:
            b_id_unique = np.arrange(len(b))
        else:
            b_id_unique = sorted(set(b_id))
        n = len(b_id_unique)
        if mode == 1:  # random split of input image pixels
            b_id_unique = list(set(b_id))
            np.random.shuffle(b_id_unique)
            b_id_unique_set_1 = b_id_unique[: n // 2]
        elif mode == 2:  # even/odd pixels
            b_id_unique_set_1 = b_id_unique[::2]
        elif mode == 3:  # left/right halves
            b_id_unique_set_1 = b_id_unique[: n // 2]
        else:  # left 1/3 + right 1/3 vs center 1/3
            b_id_unique_set_1 = b_id_unique[: n // 3] + b_id_unique[n * 2 // 3 :]

        is_set_1 = np.isin(b_id, b_id_unique_set_1)
        A_set_1 = A[is_set_1]
        b_set_1 = b[is_set_1]
        A_set_2 = A[~is_set_1]
        b_set_2 = b[~is_set_1]
        assert len(b_set_1) + len(b_set_2) == len(
            b
        ), f"ERROR: {len(b_set_1)=:,}\t{len(b_set_2)=:,}\t{len(b)=:,}"
        return (A_set_1, b_set_1), (A_set_2, b_set_2)

    def solve_equations(
        A_data,
        b_data,
        A_hsym,
        b_hsym,
        positive=False,
        algorithm="elasticnet",
        train_fraction=1.0,
        score_metric="cosine",
        img_shape_2d=None,
        target_apix2d=5.0,
        verbose=0,
    ):
        if not (A_hsym is None or b_hsym is None):
            from scipy.sparse import vstack

            A = vstack((A_data, A_hsym))
            b = np.concatenate((b_data, b_hsym))
        else:
            A = A_data
            b = b_data
        if 0 < train_fraction < 1:
            shuffled_indices = np.arange(A.shape[0])
            np.random.shuffle(shuffled_indices)
            n = int(len(shuffled_indices) * train_fraction + 0.5)
            A_train = A[shuffled_indices[:n], :]
            b_train = b[shuffled_indices[:n]]
            A_test = A[shuffled_indices[n:], :]
            b_test = b[shuffled_indices[n:]]
        else:
            A_train = A
            b_train = b
            A_test = None
            b_test = None

        tol = 1e-2
        max_iter = 200

        if (
            algorithm["model"] == "lsq"
        ):  # ordinary linear least square without regularization
            if positive:
                lb = 0.0
                ub = np.max(b_data)
                logger.info(
                    "Imposing constraint for the reconstruction: lb=%s ub=%s",
                    round(lb, 6),
                    round(ub, 6),
                )
            else:
                lb = -np.inf
                ub = np.inf

            from scipy.optimize import lsq_linear

            res = lsq_linear(
                A,
                b,
                bounds=(lb, ub),
                tol=tol,
                max_iter=max_iter,
                lsmr_maxiter=1000,
                lsmr_tol="auto",
                verbose=verbose,
            )
            return res.x.astype(np.float32), None

        if (
            algorithm["model"] == "lreg"
        ):  # ordinary linear least square without regularization
            from sklearn.linear_model import LinearRegression

            if positive:
                A_train = A_train.toarray()
                logger.warning(
                    "--algorithm=lreq with positive contraints uses very large amount memory!"
                )
            model = LinearRegression(fit_intercept=True, positive=positive)
        elif algorithm["model"] == "lasso":
            from sklearn.linear_model import Lasso

            model = Lasso(
                alpha=algorithm.get("alpha", 1e-4),
                fit_intercept=True,
                positive=positive,
                selection="random",
                tol=tol,
                max_iter=max_iter,
            )
        elif algorithm["model"] == "elasticnet":
            from sklearn.linear_model import ElasticNet

            model = ElasticNet(
                alpha=algorithm.get("alpha", 1e-4),
                l1_ratio=algorithm.get("l1_ratio", 0.5),
                fit_intercept=True,
                positive=positive,
                selection="random",
                tol=tol,
                max_iter=max_iter,
            )
        elif algorithm["model"] == "ridge":
            from sklearn.linear_model import Ridge

            model = Ridge(
                alpha=algorithm.get("alpha", 1),
                fit_intercept=True,
                positive=positive,
                tol=tol,
                max_iter=max_iter,
            )
        elif algorithm["model"] == "ard":
            from sklearn.linear_model import ARDRegression

            A_train = A_train.toarray()
            model = ARDRegression(
                alpha_1=algorithm.get("alpha", 1e-6),
                alpha_2=algorithm.get("alpha", 1e-6),
                fit_intercept=True,
                tol=tol,
                max_iter=max_iter,
            )

        model.fit(X=A_train, y=b_train)
        res = model.coef_.astype(np.float32)
        if not np.any(res):
            if algorithm["model"] in ["lreg"]:
                res[len(res) // 2] = 1
            else:
                while not np.any(res):
                    model.alpha *= 0.1
                    model.fit(X=A_train, y=b_train)
                    res = model.coef_.astype(np.float32)
        if A_test is not None and b_test is not None:
            score = helicon.cosine_similarity(A_test.dot(res), b_test)
        else:
            score = None
        return res, score

    n_eqns = A_data.shape[0]
    n_unknowns = A_data.shape[1]
    n_nonzeros = A_data.count_nonzero()
    if A_hsym is not None:
        n_eqns += A_hsym.shape[0]
        n_nonzeros += A_hsym.count_nonzero()
    sparsity = 1 - n_nonzeros / (n_eqns * n_unknowns)

    pitch_pixel = round(rise_pixel * 360 / abs(twist_degree))
    positive = positive_constraint > 0 or (
        positive_constraint < 0 and pitch_pixel > round(reconstruct_length_3d_pixel * 2)
    )
    train_fraction = 1.0

    img_shape_2d = (reconstruct_length_2d_pixel, reconstruct_diameter_2d_pixel)

    with helicon.Timer(
        f"solve_equations{' Full' if fsc_test>0 else ''}: {n_eqns:,} equations, {n_unknowns:,} unknowns, {sparsity*100:f}% sparsity",
        verbose=verbose > 10,
    ):
        x, score = solve_equations(
            A_data,
            b_data,
            A_hsym,
            b_hsym,
            positive=positive,
            algorithm=algorithm,
            train_fraction=train_fraction,
            score_metric=score_metric,
            img_shape_2d=img_shape_2d,
            target_apix2d=target_apix2d,
            verbose=2 if verbose > 10 else 0,
        )

    Abx_data_triplets = [(A_data, b_data, x)]

    xs = [x]
    scores = [score]

    if fsc_test >= 1:
        (A_data_set_1, b_data_set_1), (A_data_set_2, b_data_set_2) = split_A_b(
            A_data, b_data, b_data_pid, mode=fsc_test
        )

        Ab_pairs = [
            (A_data_set_1, A_hsym, b_data_set_1, b_hsym, "Half-1"),
            (A_data_set_2, A_hsym, b_data_set_2, b_hsym, "Half-2"),
        ]

        for pair_A_data, pair_A_hsym, pair_b_data, pair_b_hsym, tag in Ab_pairs:
            n_eqns = pair_A_data.shape[0]
            n_unknowns = pair_A_data.shape[1]
            n_nonzeros = pair_A_data.count_nonzero()
            if pair_A_hsym is not None and pair_b_hsym is not None:
                n_eqns += pair_A_hsym.shape[0]
                n_nonzeros += pair_A_hsym.count_nonzero()
            sparsity = 1 - n_nonzeros / (n_eqns * n_unknowns)
            with helicon.Timer(
                f"solve_equations {tag}: {n_eqns:,} equations, {n_unknowns:,} unknowns, {sparsity*100:f}% sparsity",
                verbose=verbose > 10,
            ):
                x, score = solve_equations(
                    pair_A_data,
                    pair_b_data,
                    pair_A_hsym,
                    pair_b_hsym,
                    positive=positive,
                    algorithm=algorithm,
                    train_fraction=train_fraction,
                    score_metric=score_metric,
                    img_shape_2d=img_shape_2d,
                    target_apix2d=target_apix2d,
                    verbose=2 if verbose > 10 else 0,
                )
                xs.append(x)
                scores.append(score)

        Abx_data_triplets += [
            (A_data_set_1, b_data_set_1, xs[1]),
            (A_data_set_2, b_data_set_2, xs[2]),
        ]

    if np.any([score is None for score in scores]):
        scores = []
        for tmp_A, tmp_b, tmp_x in Abx_data_triplets:
            pred = tmp_A.dot(tmp_x)
            if thresh_fraction >= 0:
                pred = np.clip(pred, 0, None)
            if score_metric == "frc" and img_shape_2d is not None:
                pred_2d = pred.reshape(img_shape_2d)
                b_2d = tmp_b.reshape(img_shape_2d)
                scores.append(helicon.calc_frc_2d(pred_2d, b_2d, target_apix2d))
            else:
                scores.append(helicon.cosine_similarity(pred, tmp_b))

    if len(scores) == 3:
        score = scores[0] / 2 + (scores[1] + scores[2]) / 4
    else:
        score = scores[0]

    shape = (
        reconstruct_length_3d_pixel,
        reconstruct_diameter_3d_pixel,
        reconstruct_diameter_3d_pixel,
    )
    rec3d = np.zeros(shape, dtype=np.float32)
    rec3d[mask] = xs[0]

    if len(xs) == 1:
        return (rec3d, None, None), score
    else:
        rec3d_set_1 = np.zeros(shape, dtype=np.float32)
        rec3d_set_2 = np.zeros(shape, dtype=np.float32)
        rec3d_set_1[mask] = xs[1]
        rec3d_set_2[mask] = xs[2]
        return (rec3d, rec3d_set_1, rec3d_set_2), score


@helicon.cache(
    cache_dir=cache_dir, ignore=["verbose"], expires_after=7, verbose=0
)  # 7 days
def build_A_helical_sym_matrix(
    nz: int,
    ny: int,
    nx: int,
    twist_degree: float,
    rise_pixel: float,
    csym: int,
    rmin: float,
    rmax: float,
    min_sym_pairs: int,
    interpolation: str,
    verbose: int = 0,
):
    """Build the sparse matrix enforcing helical symmetry constraints.

    Each row encodes the equation density(voxel_i) == density(voxel_j)
    for symmetry-related voxel pairs within the cylindrical mask.

    Parameters
    ----------
    nz, ny, nx : int
        Dimensions of the 3D reconstruction volume.
    twist_degree : float
        Helical twist in degrees.
    rise_pixel : float
        Helical rise in pixels.
    csym : int
        Cyclic symmetry order.
    rmin : float
        Inner radius of cylindrical mask.
    rmax : float
        Outer radius of cylindrical mask.
    min_sym_pairs : int
        Minimum number of symmetry constraint equations to generate.
    interpolation : str
        Interpolation method ("linear", "linear11", etc. or "nn").
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    tuple of (csr_matrix or None, ndarray or None)
        Symmetry constraint matrix A and zero vector b.
    """

    hcsym_pairs = sorted_hsym_csym_pairs(twist_degree, rise_pixel, csym, nz)

    mask, (Z, Y, X) = helicon.get_cylindrical_mask(
        nz, ny, nx, rmin=rmin, rmax=rmax, return_xyz=True
    )
    n_x = np.count_nonzero(mask)
    mask_nonzero_indices_zyx_tuple = np.nonzero(mask)
    mask_nonzero_indices_matrix = np.zeros(mask.shape, dtype=int) - 1
    mask_nonzero_indices_matrix[mask_nonzero_indices_zyx_tuple] = np.arange(n_x)
    xyz = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).transpose()

    # sparse A matrix
    csr_A = []

    from scipy.spatial.transform import Rotation as R

    if interpolation in ["linear", "linear01", "linear11"]:

        @jit(nopython=True, cache=True, nogil=True)
        def loop_kji(
            Zi,
            Yi,
            Xi,
            Zj,
            Yj,
            Xj,
            mask,
            mask_nonzero_indices_zyx_tuple,
            mask_nonzero_indices_matrix,
            pair_ids,
        ):
            mask_indices_Z, mask_indices_Y, mask_indices_X = (
                mask_nonzero_indices_zyx_tuple
            )
            mz, my, mx = mask.shape
            n_indices = len(mask_indices_Z)
            csr_row_tmp = np.zeros(n_indices * 16, dtype=np.float32)
            csr_col_tmp = np.zeros(n_indices * 16, dtype=np.float32)
            csr_data_tmp = np.zeros(n_indices * 16, dtype=np.float32)
            csr_rc_tmp_count = 0
            row_count_tmp = 0
            for mi in range(n_indices):
                k = mask_indices_Z[mi]
                j = mask_indices_Y[mi]
                i = mask_indices_X[mi]
                zi = int(Zi[k, j, i])
                yi = int(Yi[k, j, i])
                xi = int(Xi[k, j, i])
                zj = int(Zj[k, j, i])
                yj = int(Yj[k, j, i])
                xj = int(Xj[k, j, i])
                if zi < 0 or zi > mz - 1:
                    continue
                if zj < 0 or zj > mz - 1:
                    continue
                if yi < 0 or yi > my - 1:
                    continue
                if yj < 0 or yj > my - 1:
                    continue
                if xi < 0 or xi > mx - 1:
                    continue
                if xj < 0 or xj > mx - 1:
                    continue
                if zi + 1 > mz - 1 or yi + 1 > my - 1 or xi + 1 > mx - 1:
                    continue
                if zj + 1 > mz - 1 or yj + 1 > my - 1 or xj + 1 > mx - 1:
                    continue
                if not mask[zi, yi, xi]:
                    continue
                if not mask[zi + 1, yi, xi]:
                    continue
                if not mask[zi, yi + 1, xi]:
                    continue
                if not mask[zi + 1, yi + 1, xi]:
                    continue
                if not mask[zi, yi, xi + 1]:
                    continue
                if not mask[zi + 1, yi, xi + 1]:
                    continue
                if not mask[zi, yi + 1, xi + 1]:
                    continue
                if not mask[zi + 1, yi + 1, xi + 1]:
                    continue
                if not mask[zj, yj, xj]:
                    continue
                if not mask[zj + 1, yj, xj]:
                    continue
                if not mask[zj, yj + 1, xj]:
                    continue
                if not mask[zj + 1, yj + 1, xj]:
                    continue
                if not mask[zj, yj, xj + 1]:
                    continue
                if not mask[zj + 1, yj, xj + 1]:
                    continue
                if not mask[zj, yj + 1, xj + 1]:
                    continue
                if not mask[zj + 1, yj + 1, xj + 1]:
                    continue

                i_000 = mask_nonzero_indices_matrix[zi, yi, xi]
                if i_000 < 0 or i_000 > n_x - 1:
                    continue
                i_001 = mask_nonzero_indices_matrix[zi, yi, xi + 1]
                if i_001 < 0 or i_001 > n_x - 1:
                    continue
                i_010 = mask_nonzero_indices_matrix[zi, yi + 1, xi]
                if i_010 < 0 or i_010 > n_x - 1:
                    continue
                i_011 = mask_nonzero_indices_matrix[zi, yi + 1, xi + 1]
                if i_011 < 0 or i_011 > n_x - 1:
                    continue
                i_100 = mask_nonzero_indices_matrix[zi + 1, yi, xi]
                if i_100 < 0 or i_100 > n_x - 1:
                    continue
                i_101 = mask_nonzero_indices_matrix[zi + 1, yi, xi + 1]
                if i_101 < 0 or i_101 > n_x - 1:
                    continue
                i_110 = mask_nonzero_indices_matrix[zi + 1, yi + 1, xi]
                if i_110 < 0 or i_110 > n_x - 1:
                    continue
                i_111 = mask_nonzero_indices_matrix[zi + 1, yi + 1, xi + 1]
                if i_111 < 0 or i_111 > n_x - 1:
                    continue

                j_000 = mask_nonzero_indices_matrix[zj, yj, xj]
                if j_000 < 0 or j_000 > n_x - 1:
                    continue
                j_001 = mask_nonzero_indices_matrix[zj, yj, xj + 1]
                if j_001 < 0 or j_001 > n_x - 1:
                    continue
                j_010 = mask_nonzero_indices_matrix[zj, yj + 1, xj]
                if j_010 < 0 or j_010 > n_x - 1:
                    continue
                j_011 = mask_nonzero_indices_matrix[zj, yj + 1, xj + 1]
                if j_011 < 0 or j_011 > n_x - 1:
                    continue
                j_100 = mask_nonzero_indices_matrix[zj + 1, yj, xj]
                if j_100 < 0 or j_100 > n_x - 1:
                    continue
                j_101 = mask_nonzero_indices_matrix[zj + 1, yj, xj + 1]
                if j_101 < 0 or j_101 > n_x - 1:
                    continue
                j_110 = mask_nonzero_indices_matrix[zj + 1, yj + 1, xj]
                if j_110 < 0 or j_110 > n_x - 1:
                    continue
                j_111 = mask_nonzero_indices_matrix[zj + 1, yj + 1, xj + 1]
                if j_111 < 0 or j_111 > n_x - 1:
                    continue

                if abs(zi - zj) < 3 or abs(yi - yj) < 3 or abs(xi - xj) < 3:
                    continue

                zir = round(Zi[k, j, i])
                yir = round(Yi[k, j, i])
                xir = round(Xi[k, j, i])
                zjr = round(Zj[k, j, i])
                yjr = round(Yj[k, j, i])
                xjr = round(Xj[k, j, i])
                ir = mask_nonzero_indices_matrix[zir, yir, xir]
                jr = mask_nonzero_indices_matrix[zjr, yjr, xjr]
                pair_id = ir * n_indices + jr
                if pair_id in pair_ids:
                    continue
                pair_id2 = jr * n_indices + ir
                pair_ids.add(pair_id)
                pair_ids.add(pair_id2)

                zf = Zi[k, j, i] - zi
                yf = Yi[k, j, i] - yi
                xf = Xi[k, j, i] - xi
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_000
                csr_data_tmp[csr_rc_tmp_count] = (1 - zf) * (1 - yf) * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_001
                csr_data_tmp[csr_rc_tmp_count] = (1 - zf) * (1 - yf) * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_010
                csr_data_tmp[csr_rc_tmp_count] = (1 - zf) * yf * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_011
                csr_data_tmp[csr_rc_tmp_count] = (1 - zf) * yf * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_100
                csr_data_tmp[csr_rc_tmp_count] = zf * (1 - yf) * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_101
                csr_data_tmp[csr_rc_tmp_count] = zf * (1 - yf) * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_110
                csr_data_tmp[csr_rc_tmp_count] = xf * yf * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_111
                csr_data_tmp[csr_rc_tmp_count] = xf * yf * zf
                csr_rc_tmp_count += 1

                zf = Zj[k, j, i] - zj
                yf = Yj[k, j, i] - yj
                xf = Xj[k, j, i] - xj
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_000
                csr_data_tmp[csr_rc_tmp_count] = -(1 - zf) * (1 - yf) * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_001
                csr_data_tmp[csr_rc_tmp_count] = -(1 - zf) * (1 - yf) * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_010
                csr_data_tmp[csr_rc_tmp_count] = -(1 - zf) * yf * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_011
                csr_data_tmp[csr_rc_tmp_count] = -(1 - zf) * yf * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_100
                csr_data_tmp[csr_rc_tmp_count] = -zf * (1 - yf) * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_101
                csr_data_tmp[csr_rc_tmp_count] = -zf * (1 - yf) * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_110
                csr_data_tmp[csr_rc_tmp_count] = -xf * yf * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_111
                csr_data_tmp[csr_rc_tmp_count] = -xf * yf * zf
                csr_rc_tmp_count += 1

                row_count_tmp += 1
            return (
                csr_row_tmp[:csr_rc_tmp_count],
                csr_col_tmp[:csr_rc_tmp_count],
                csr_data_tmp[:csr_rc_tmp_count],
                row_count_tmp,
            )

    else:  # nearest neighbor

        @jit(nopython=True, cache=True, nogil=True)
        def loop_kji(
            Zi,
            Yi,
            Xi,
            Zj,
            Yj,
            Xj,
            mask,
            mask_nonzero_indices_zyx_tuple,
            mask_nonzero_indices_matrix,
            pair_ids,
        ):
            mask_indices_Z, mask_indices_Y, mask_indices_X = (
                mask_nonzero_indices_zyx_tuple
            )
            mz, my, mx = mask.shape
            n_indices = len(mask_indices_Z)
            csr_row_tmp = np.zeros(n_indices * 2, dtype=np.float32)
            csr_col_tmp = np.zeros(n_indices * 2, dtype=np.float32)
            csr_data_tmp = np.zeros(n_indices * 2, dtype=np.float32)
            csr_rc_tmp_count = 0
            row_count_tmp = 0
            for mi in range(n_indices):
                k = mask_indices_Z[mi]
                j = mask_indices_Y[mi]
                i = mask_indices_X[mi]
                zi = round(Zi[k, j, i])
                yi = round(Yi[k, j, i])
                xi = round(Xi[k, j, i])
                zj = round(Zj[k, j, i])
                yj = round(Yj[k, j, i])
                xj = round(Xj[k, j, i])
                if zi < 0 or zi > mz - 1:
                    continue
                if zj < 0 or zj > mz - 1:
                    continue
                if yi < 0 or yi > my - 1:
                    continue
                if yj < 0 or yj > my - 1:
                    continue
                if xi < 0 or xi > mx - 1:
                    continue
                if xj < 0 or xj > mx - 1:
                    continue
                if not mask[zi, yi, xi]:
                    continue
                if not mask[zj, yj, xj]:
                    continue
                index_i = mask_nonzero_indices_matrix[zi, yi, xi]
                if index_i < 0 or index_i > n_indices - 1:
                    continue
                index_j = mask_nonzero_indices_matrix[zj, yj, xj]
                if index_j < 0 or index_j > n_indices - 1:
                    continue
                pair_id = index_i * n_indices + index_j
                if pair_id in pair_ids:
                    continue
                pair_ids.add(pair_id)
                pair_id2 = index_j * n_indices + index_i
                pair_ids.add(pair_id2)

                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = index_i
                csr_data_tmp[csr_rc_tmp_count] = 1
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = index_j
                csr_data_tmp[csr_rc_tmp_count] = -1
                csr_rc_tmp_count += 1
                row_count_tmp += 1
            return (
                csr_row_tmp[:csr_rc_tmp_count],
                csr_col_tmp[:csr_rc_tmp_count],
                csr_data_tmp[:csr_rc_tmp_count],
                row_count_tmp,
            )

    with helicon.Timer(f"\tbuild csr matrix - {interpolation}", verbose=verbose > 10):
        pair_ids = set([-1])
        row_count = 0
        for pi, p in enumerate(hcsym_pairs):
            angle, ((hsym_i, csym_i), (hsym_j, csym_j)) = p[0], p[-1]
            ri = R.from_euler(
                "z", twist_degree * hsym_i + csym_i * 360 / csym, degrees=True
            )
            tmp_xyz = ri.apply(xyz, inverse=False)
            Xi = tmp_xyz[:, 0].reshape((nz, ny, nx)) + nx // 2  # axes order: z, y, x
            Yi = tmp_xyz[:, 1].reshape((nz, ny, nx)) + ny // 2  # axes order: z, y, x
            Zi = (
                tmp_xyz[:, 2].reshape((nz, ny, nx)) + nz // 2 + rise_pixel * hsym_i
            )  # axes order: z, y, x

            rj = R.from_euler(
                "z", twist_degree * hsym_j + csym_j * 360 / csym, degrees=True
            )
            tmp_xyz = rj.apply(xyz, inverse=False)
            Xj = tmp_xyz[:, 0].reshape((nz, ny, nx)) + nx // 2  # axes order: z, y, x
            Yj = tmp_xyz[:, 1].reshape((nz, ny, nx)) + ny // 2  # axes order: z, y, x
            Zj = (
                tmp_xyz[:, 2].reshape((nz, ny, nx)) + nz // 2 + rise_pixel * hsym_j
            )  # axes order: z, y, x

            try:
                from numba.core.errors import NumbaPendingDeprecationWarning
                import warnings

                warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
            except ImportError:
                pass

            csr_row_tmp, csr_col_tmp, csr_data_tmp, row_count_tmp = loop_kji(
                Zi,
                Yi,
                Xi,
                Zj,
                Yj,
                Xj,
                mask,
                mask_nonzero_indices_zyx_tuple,
                mask_nonzero_indices_matrix,
                pair_ids,
            )
            row_count += row_count_tmp
            if verbose > 20:
                logger.debug(
                    "%s/%s: %s %s %s %s",
                    pi + 1,
                    len(hcsym_pairs),
                    round(angle, 2),
                    (hsym_i, csym_i),
                    (hsym_j, csym_j),
                    f"{n_x:,}",
                    f"+{row_count_tmp:,}",
                    f"{row_count:,}",
                    f"target={min_sym_pairs:,}",
                )
            if row_count_tmp:
                csr_A_tmp = csr_matrix(
                    (csr_data_tmp, (csr_row_tmp, csr_col_tmp)),
                    shape=(row_count_tmp, n_x),
                    dtype=np.float32,
                )
                csr_A.append(csr_A_tmp)
            if row_count >= min_sym_pairs:
                break

        if len(csr_A):
            from scipy.sparse import vstack

            A = vstack(csr_A)
            b = np.zeros(row_count, dtype=np.float32)
            assert A.shape[0] == row_count
        else:
            A = None
            b = None
        return A, b


@helicon.cache(
    cache_dir=cache_dir, ignore=["verbose"], expires_after=7, verbose=0
)  # 7 days
def build_A_data_matrix(
    image,
    scale2d_to_3d,
    twist_degree,
    rise_pixel,
    csym,
    tilt_degree,
    psi_degree,
    dy_pixel,
    reconstruct_diameter_2d_pixel,
    reconstruct_length_2d_pixel,
    reconstruct_diameter_3d_pixel,
    reconstruct_diameter_3d_inner_pixel,
    reconstruct_length_3d_pixel,
    min_projection_lines,
    interpolation,
    verbose=0,
):
    """Build the sparse data matrix A and target vector b for least-squares reconstruction.

    Back-projects 2D image coordinates into 3D, applies tilt/psi/dy transforms,
    and constructs the linear system A * x = b where each equation corresponds
    to one projection ray through the 3D volume.

    Parameters
    ----------
    image : ndarray
        2D input projection image.
    scale2d_to_3d : float
        Scale factor from 2D to 3D pixel size.
    twist_degree : float
        Helical twist in degrees.
    rise_pixel : float
        Helical rise in 3D pixels.
    csym : int
        Cyclic symmetry order.
    tilt_degree : float
        Out-of-plane tilt in degrees.
    psi_degree : float
        In-plane rotation in degrees.
    dy_pixel : float
        Perpendicular shift in 2D pixels.
    reconstruct_diameter_2d_pixel : int
    reconstruct_length_2d_pixel : int
    reconstruct_diameter_3d_pixel : int
    reconstruct_diameter_3d_inner_pixel : int
    reconstruct_length_3d_pixel : int
    min_projection_lines : int
        Minimum number of projection equations to generate.
    interpolation : str
        Interpolation method ("linear", "nn", etc.).
    verbose : int, optional
        Verbosity level. Defaults to 0.

    Returns
    -------
    tuple of (csr_matrix, ndarray, ndarray)
        A (sparse matrix), b (target values), b_pid (pixel IDs).
    """
    with helicon.Timer("back_project_2d_coords_to_3d_coords", verbose=verbose > 10):
        coords_3d, pixel_vals = back_project_2d_coords_to_3d_coords(
            image=image,
            scale2d_to_3d=scale2d_to_3d,
            reconstruct_diameter_2d_pixel=reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel=reconstruct_length_2d_pixel,
        )

    X0, Y0, Z0 = coords_3d  # axes order:  z, y, x. helical axis along z
    assert X0[:, :, 0].shape[::-1] == pixel_vals.shape

    rmin = reconstruct_diameter_3d_inner_pixel / 2
    rmax = reconstruct_diameter_3d_pixel // 2 - 1

    nz, ny, nx = X0.shape  # axes order: z, y, x
    if reconstruct_length_3d_pixel <= 0:
        reconstruct_length_3d_pixel = nz

    mask = helicon.get_cylindrical_mask(
        nz=reconstruct_length_3d_pixel, ny=ny, nx=nx, rmin=rmin, rmax=rmax
    )
    n_x = np.count_nonzero(mask)
    mask_nonzero_indices_matrix = np.zeros(mask.shape, dtype=int) - 1
    mask_nonzero_indices_matrix[np.nonzero(mask)] = np.arange(n_x)

    coords0 = np.vstack((X0.ravel(), Y0.ravel(), Z0.ravel())).transpose()
    from scipy.spatial.transform import Rotation as R

    coords0[:, 1] -= dy_pixel
    r = R.from_euler("yx", (tilt_degree, psi_degree), degrees=True)
    coords0 = r.apply(coords0, inverse=True)

    # sparse A matrix
    csr_A = []
    csr_b = []  # one value for each pixel in pixel_vals
    b_pid = []
    n_b = 0
    if interpolation in ["linear", "linear10", "linear11"]:

        @jit(nopython=True, cache=True, nogil=True)
        def loop_kji(Z, Y, X, mask, mask_nonzero_indices_matrix, n_x, pixel_vals):
            nz, ny, nx = Z.shape
            mz, my, mx = mask.shape
            csr_row_tmp = np.zeros(nz * ny * nx * 8, dtype=np.float32)
            csr_col_tmp = np.zeros(nz * ny * nx * 8, dtype=np.float32)
            csr_data_tmp = np.zeros(nz * ny * nx * 8, dtype=np.float32)
            b_tmp = np.zeros(nz * ny, dtype=np.float32)
            b_pid_tmp = np.zeros(nz * ny, dtype=np.int32)
            csr_rc_tmp_count = 0
            row_count_tmp = 0
            for k in range(nz):  # old x axis before back projection
                for j in range(ny):  # same y axis before back projection
                    row_tmp = {}
                    has_projection_data = False
                    for i in range(nx):  # old z axis before back projection
                        zi = int(Z[k, j, i])
                        yi = int(Y[k, j, i])
                        xi = int(X[k, j, i])
                        if zi < 0 or zi > mz - 1:
                            continue
                        if yi < 0 or yi > my - 1:
                            continue
                        if xi < 0 or xi > mx - 1:
                            continue
                        if zi + 1 > mz - 1 or yi + 1 > my - 1 or xi + 1 > mx - 1:
                            continue
                        if not mask[zi, yi, xi]:
                            continue
                        if not mask[zi + 1, yi, xi]:
                            continue
                        if not mask[zi, yi + 1, xi]:
                            continue
                        if not mask[zi + 1, yi + 1, xi]:
                            continue
                        if not mask[zi, yi, xi + 1]:
                            continue
                        if not mask[zi + 1, yi, xi + 1]:
                            continue
                        if not mask[zi, yi + 1, xi + 1]:
                            continue
                        if not mask[zi + 1, yi + 1, xi + 1]:
                            continue
                        index_000 = mask_nonzero_indices_matrix[zi, yi, xi]
                        if index_000 < 0 or index_000 > n_x - 1:
                            continue
                        index_001 = mask_nonzero_indices_matrix[zi, yi, xi + 1]
                        if index_001 < 0 or index_001 > n_x - 1:
                            continue
                        index_010 = mask_nonzero_indices_matrix[zi, yi + 1, xi]
                        if index_010 < 0 or index_010 > n_x - 1:
                            continue
                        index_011 = mask_nonzero_indices_matrix[zi, yi + 1, xi + 1]
                        if index_011 < 0 or index_011 > n_x - 1:
                            continue
                        index_100 = mask_nonzero_indices_matrix[zi + 1, yi, xi]
                        if index_100 < 0 or index_100 > n_x - 1:
                            continue
                        index_101 = mask_nonzero_indices_matrix[zi + 1, yi, xi + 1]
                        if index_101 < 0 or index_101 > n_x - 1:
                            continue
                        index_110 = mask_nonzero_indices_matrix[zi + 1, yi + 1, xi]
                        if index_110 < 0 or index_110 > n_x - 1:
                            continue
                        index_111 = mask_nonzero_indices_matrix[zi + 1, yi + 1, xi + 1]
                        if index_111 < 0 or index_111 > n_x - 1:
                            continue
                        has_projection_data = True

                        zf = Z[k, j, i] - zi
                        yf = Y[k, j, i] - yi
                        xf = X[k, j, i] - xi
                        for index in (
                            index_000,
                            index_001,
                            index_010,
                            index_011,
                            index_100,
                            index_101,
                            index_110,
                            index_111,
                        ):
                            if index not in row_tmp:
                                row_tmp[index] = 0.0
                        row_tmp[index_000] += (1 - zf) * (1 - yf) * (1 - xf)
                        row_tmp[index_001] += (1 - zf) * (1 - yf) * (xf)
                        row_tmp[index_010] += (1 - zf) * (yf) * (1 - xf)
                        row_tmp[index_011] += (1 - zf) * (yf) * (xf)
                        row_tmp[index_100] += (zf) * (1 - yf) * (1 - xf)
                        row_tmp[index_101] += (zf) * (1 - yf) * (xf)
                        row_tmp[index_110] += (zf) * (yf) * (1 - xf)
                        row_tmp[index_111] += (zf) * (yf) * (xf)
                    if has_projection_data:
                        for index in row_tmp:
                            csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                            csr_col_tmp[csr_rc_tmp_count] = index
                            csr_data_tmp[csr_rc_tmp_count] = row_tmp[index]
                            csr_rc_tmp_count += 1
                        b_tmp[row_count_tmp] = pixel_vals[j, k]
                        b_pid_tmp[row_count_tmp] = k * ny + j
                        row_count_tmp += 1
            return (
                csr_row_tmp[:csr_rc_tmp_count],
                csr_col_tmp[:csr_rc_tmp_count],
                csr_data_tmp[:csr_rc_tmp_count],
                b_tmp[:row_count_tmp],
                b_pid_tmp[:row_count_tmp],
            )

    else:  # nearest neighbor

        @jit(nopython=True, cache=True, nogil=True)
        def loop_kji(Z, Y, X, mask, mask_nonzero_indices_matrix, n_x, pixel_vals):
            nz, ny, nx = Z.shape
            mz, my, mx = mask.shape
            csr_row_tmp = np.zeros(nz * ny * nx, dtype=np.float32)
            csr_col_tmp = np.zeros(nz * ny * nx, dtype=np.float32)
            csr_data_tmp = np.ones(nz * ny * nx, dtype=np.float32)
            b_tmp = np.zeros(nz * ny, dtype=np.float32)
            b_pid_tmp = np.zeros(nz * ny, dtype=np.int32)
            csr_rc_tmp_count = 0
            row_count_tmp = 0
            for k in range(nz):  # old x axis before back projection
                for j in range(ny):  # same y axis before back projection
                    has_projection_data = False
                    for i in range(nx):  # old z axis before back projection
                        zi = round(Z[k, j, i])
                        yi = round(Y[k, j, i])
                        xi = round(X[k, j, i])
                        if zi < 0 or zi > mz - 1:
                            continue
                        if yi < 0 or yi > my - 1:
                            continue
                        if xi < 0 or xi > mx - 1:
                            continue
                        if not mask[zi, yi, xi]:
                            continue
                        index = mask_nonzero_indices_matrix[zi, yi, xi]
                        if index < 0 or index > n_x - 1:
                            continue
                        has_projection_data = True
                        csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                        csr_col_tmp[csr_rc_tmp_count] = index
                        csr_rc_tmp_count += 1
                    if has_projection_data:
                        b_tmp[row_count_tmp] = pixel_vals[j, k]
                        b_pid_tmp[row_count_tmp] = k * ny + j
                        row_count_tmp += 1
            return (
                csr_row_tmp[:csr_rc_tmp_count],
                csr_col_tmp[:csr_rc_tmp_count],
                csr_data_tmp[:csr_rc_tmp_count],
                b_tmp[:row_count_tmp],
                b_pid_tmp[:row_count_tmp],
            )

    hsym_max = max(1, int(np.ceil(reconstruct_length_3d_pixel + nz) / 2 / rise_pixel))
    hsyms = range(-hsym_max, hsym_max + 1)
    csyms = range(csym)
    from itertools import product, combinations

    hcsyms = list(product(hsyms, csyms))
    hcsyms.sort(key=lambda x: (abs(x[0]), x[1]))
    from scipy.stats import qmc

    qmc_method = qmc.Halton(d=1, scramble=False)
    n = len(hcsyms)
    indices = qmc_method.integers(l_bounds=0, u_bounds=n, n=n)
    hcsyms = [hcsyms[int(i[0])] for i in indices]
    for hci, (hi, ci) in enumerate(hcsyms):
        angle = twist_degree * hi + 360 * ci / csym
        r = R.from_euler("z", angle, degrees=True)
        coords = r.apply(coords0, inverse=True)
        coords[:, 2] -= hi * rise_pixel
        X = coords[:, 0].reshape((nz, ny, nx)) + nx // 2  # axes order: z, y, x
        Y = coords[:, 1].reshape((nz, ny, nx)) + ny // 2  # axes order: z, y, x
        Z = (
            coords[:, 2].reshape((nz, ny, nx)) + reconstruct_length_3d_pixel // 2
        )  # axes order: z, y, x

        csr_row_tmp, csr_col_tmp, csr_data_tmp, b_tmp, b_pid_tmp = loop_kji(
            Z, Y, X, mask, mask_nonzero_indices_matrix, n_x, pixel_vals
        )
        n_b += len(b_tmp)
        if verbose > 20:
            logger.debug(
                "%s/%s: hi=%s ci=%s +%s %s target_lines=%s",
                hci + 1,
                len(hcsyms),
                hi,
                ci,
                f"{len(b_tmp):,}",
                f"{n_b:,}",
                f"{min_projection_lines:,}",
            )
        if len(b_tmp):
            b_tmp = np.array(b_tmp, dtype=np.float32)
            csr_A_tmp = csr_matrix(
                (csr_data_tmp, (csr_row_tmp, csr_col_tmp)),
                shape=(len(b_tmp), n_x),
                dtype=np.float32,
            )
            csr_A.append(csr_A_tmp)
            csr_b.append(b_tmp)
            b_pid += [b_pid_tmp]
        if min_projection_lines > 0 and n_b > min_projection_lines:
            break
    from scipy.sparse import vstack

    A = vstack(csr_A)
    b = np.concatenate(csr_b, dtype=np.float32)
    b_pid = np.concatenate(b_pid)
    return A, b, b_pid


def back_project_2d_coords_to_3d_coords(
    image,
    scale2d_to_3d,
    reconstruct_diameter_2d_pixel=-1,
    reconstruct_length_2d_pixel=-1,
):
    """Back-project 2D image coordinates to 3D coordinates.

    Maps pixel positions in the 2D input image to 3D coordinates in the
    reconstruction volume, including a 90-degree rotation that aligns
    the helical axis with the Z-axis.

    Parameters
    ----------
    image : ndarray
        2D input image.
    scale2d_to_3d : float
        Scale factor from 2D to 3D pixel size.
    reconstruct_diameter_2d_pixel : int, optional
        Diameter of the 2D reconstruction region. Defaults to -1 (auto).
    reconstruct_length_2d_pixel : int, optional
        Length of the 2D reconstruction region. Defaults to -1 (auto).

    Returns
    -------
    tuple of (tuple of ndarray, ndarray)
        ((X, Y, Z) coordinate arrays, pixel values).
    """
    ny, nx = image.shape
    if reconstruct_diameter_2d_pixel <= 0:
        reconstruct_diameter_2d_pixel = ny
    if reconstruct_length_2d_pixel <= 0:
        reconstruct_length_2d_pixel = nx  # direction of helical axis

    reconstruct_diameter_2d_pixel = int(np.rint(reconstruct_diameter_2d_pixel))
    reconstruct_length_2d_pixel = int(np.rint(reconstruct_length_2d_pixel))

    k = (
        np.arange(0, reconstruct_diameter_2d_pixel, dtype=np.int32)
        - reconstruct_diameter_2d_pixel // 2
    )
    j = (
        np.arange(0, reconstruct_diameter_2d_pixel, dtype=np.int32)
        - reconstruct_diameter_2d_pixel // 2
    )
    i = (
        np.arange(0, reconstruct_length_2d_pixel, dtype=np.int32)
        - reconstruct_length_2d_pixel // 2
    )
    region_pixel_vals = image[
        np.ix_(j + ny // 2, i + nx // 2)
    ]  # pixel values to be used for lsq solution

    from scipy.spatial.transform import Rotation as R

    r = R.from_euler("y", 90, degrees=True)
    Z, Y, X = np.meshgrid(
        k.astype(np.float32), j.astype(np.float32), i.astype(np.float32), indexing="ij"
    )
    coords = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).transpose()
    coords = r.apply(coords, inverse=True)
    if scale2d_to_3d != 1.0:
        coords *= scale2d_to_3d
    X2 = coords[:, 0].reshape(
        (
            reconstruct_diameter_2d_pixel,
            reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel,
        )
    )  # axes order: z, y, x
    Y2 = coords[:, 1].reshape(
        (
            reconstruct_diameter_2d_pixel,
            reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel,
        )
    )  # axes order: z, y, x
    Z2 = coords[:, 2].reshape(
        (
            reconstruct_diameter_2d_pixel,
            reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel,
        )
    )  # axes order: z, y, x
    # axes change after 90 degree rotation around +y axis: x -> z, y -> y, z -> x
    X2 = np.swapaxes(X2, 0, 2)  # new axes order: z', y, x'
    Y2 = np.swapaxes(Y2, 0, 2)  # new axes order: z', y, x'
    Z2 = np.swapaxes(Z2, 0, 2)  # new axes order: z', y, x'
    assert X2[:, :, 0].shape[::-1] == region_pixel_vals.shape
    return (X2, Y2, Z2), region_pixel_vals


def sorted_hsym_csym_pairs(twist, rise, csym, nz):
    """Generate sorted pairs of (helical sym, cyclic sym) for building constraints.

    Parameters
    ----------
    twist : float
        Helical twist in degrees.
    rise : float
        Helical rise in pixels.
    csym : int
        Cyclic symmetry order.
    nz : int
        Volume size in Z.

    Returns
    -------
    list
        Sorted list of (angle, additional_info, (hsym1, csym1), (hsym2, csym2)).
    """
    hsym_max = max(1, int(np.ceil(nz / (2 * rise))))
    hsyms = range(-hsym_max, hsym_max + 1)
    csyms = range(csym)
    from itertools import product, combinations

    hcsyms = product(hsyms, csyms)
    hcsym_pairs = combinations(hcsyms, r=2)
    hcsym_pair_angles = []
    for p in hcsym_pairs:
        (hsym1, csym1), (hsym2, csym2) = p
        angle1 = twist * hsym1 + csym1 * 360 / csym
        angle2 = twist * hsym2 + csym2 * 360 / csym
        angle = round(abs((angle2 - angle1 + 180) % 360 - 180), 2)  # range: [0, 180]
        hcsym_pair_angles.append(
            (angle, abs(hsym1 + hsym2), abs(hsym1 - hsym2), abs(hsym1), abs(hsym2), p)
        )
    hcsym_pair_angles.sort(key=lambda x: x[:-1])
    from scipy.stats import qmc

    qmc_method = qmc.Halton(d=1, scramble=False)
    n = len(hcsym_pair_angles)
    indices = qmc_method.integers(l_bounds=0, u_bounds=n, n=n)
    ret = [hcsym_pair_angles[int(i[0])] for i in indices]
    return ret
