"""Utility functions for de novo helical indexing and 3D reconstruction."""

import itertools, logging, os, sys, pathlib, datetime, joblib
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


def simulate_helical_projection(
    n,
    twist,
    rise,
    csym,
    helical_diameter,
    ball_radius,
    polymer,
    planarity,
    ny,
    nx,
    apix,
    tilt=0,
    rot=0,
    psi=0,
    dy=0,
):
    """Simulate a helical projection image from a set of spherical subunits.

    Parameters
    ----------
    n : int
        Number of asymmetric units per helical turn.
    twist : float
        Helical twist in degrees.
    rise : float
        Helical rise in Angstroms.
    csym : int
        Cyclic symmetry.
    helical_diameter : float
        Helical tube diameter in Angstroms.
    ball_radius : float
        Radius of each subunit in Angstroms.
    polymer : int
        Whether to use a polymer-like random walk for subunit positions.
    planarity : float
        Planarity of the polymer (0=random, 1=planar).
    ny : int
        Output image height in pixels.
    nx : int
        Output image width in pixels.
    apix : float
        Pixel size in Angstroms.
    tilt : float, optional
        Out-of-plane tilt in degrees. Defaults to 0.
    rot : float, optional
        Rotation around helical axis in degrees. Defaults to 0.
    psi : float, optional
        In-plane rotation in degrees. Defaults to 0.
    dy : float, optional
        Perpendicular shift in Angstroms. Defaults to 0.

    Returns
    -------
    ndarray
        Simulated 2D projection image.
    """
    assert helical_diameter + ball_radius < ny * apix * 0.99
    import numpy as np

    def simulate_projection(centers, sigma, ny, nx, apix):
        sigma2 = sigma * sigma / np.log(2)
        d = np.zeros((ny, nx))
        Y, X = np.meshgrid(
            np.arange(0, ny, dtype=np.float32) - ny // 2,
            np.arange(0, nx, dtype=np.float32) - nx // 2,
            indexing="ij",
        )
        X *= apix
        Y *= apix
        for ci in range(len(centers)):
            yc, xc = centers[ci]
            x = X - xc
            y = Y - yc
            d += np.exp(-(x * x + y * y) / sigma2)
        return d

    def helical_unit_positions(
        n,
        twist,
        rise,
        csym,
        diameter,
        height,
        polymer=0,
        planarity=1.0,
        tilt=0,
        rot=0,
        psi=0,
        dy=0,
    ):
        assert n >= 1
        from scipy.spatial.transform import Rotation as R

        if polymer:
            centers_0 = random_polymer(
                n_atoms=n,
                rmin=0,
                rmax=helical_diameter / 2,
                csym=csym,
                planarity=planarity,
            )
            rot = R.from_euler("y", 90, degrees=True)
            centers_0 = rot.apply(centers_0)
            centers_0 = centers_0[:, [2, 1, 0]]  # axes order: x,y,z -> z,y,x
            n = len(centers_0)  # in case that a polymer with fewer atoms is returned
        else:
            centers_0 = np.zeros((n, 3), dtype=np.float32)
            if n > 1:
                r = np.sqrt(np.random.uniform(0, diameter**2 / 4, n))
                angle = np.random.uniform(-np.pi, np.pi, n) + np.deg2rad(rot)
                centers_0[:, 0] = r * np.cos(angle)
                centers_0[:, 1] = r * np.sin(angle)
                centers_0[:, 2] = np.random.uniform(-rise / 2, rise / 2, n)
            else:
                angle = np.deg2rad(rot)  # start from +z axis
                z = np.cos(angle) * diameter / 2
                y = np.sin(angle) * diameter / 2
                centers_0[0, 0] = z
                centers_0[0, 1] = y
                centers_0[0, 2] = 0

        imax = int(np.ceil(height / rise))
        i0 = -imax
        i1 = imax
        centers = np.zeros(((2 * imax + 1) * csym * n, 3), dtype=np.float32)

        index = 0
        for i in range(i0, i1 + 1):
            for si in range(csym):
                angle = twist * i + si * 360.0 / csym
                rot = R.from_euler("z", angle, degrees=True)
                centers[index : index + n, :] = rot.apply(centers_0)
                centers[index : index + n, 2] += i * rise
                index += n
        if tilt or psi:
            rot = R.from_euler("yx", (tilt, -psi), degrees=True)
            centers = rot.apply(centers)
        if dy:
            centers[:, 1] += dy
        centers_2d = centers[:, [1, 2]]  # project along z
        return centers_2d

    centers = helical_unit_positions(
        n,
        twist,
        rise,
        csym,
        helical_diameter,
        height=nx * apix,
        polymer=polymer,
        planarity=planarity,
        tilt=tilt,
        rot=rot,
        psi=psi,
        dy=dy,
    )
    projection = simulate_projection(centers, ball_radius, ny, nx, apix)
    return projection


def random_polymer(n_atoms=100, rmin=0, rmax=100, csym=1, planarity=0.9):
    """Generate a random polymer-like chain of atoms within a cylindrical shell.

    Uses a self-avoiding random walk with cyclic symmetry and configurable planarity.

    Parameters
    ----------
    n_atoms : int, optional
        Number of atoms to attempt. Defaults to 100.
    rmin : float, optional
        Minimum radial distance. Defaults to 0.
    rmax : float, optional
        Maximum radial distance. Defaults to 100.
    csym : int, optional
        Cyclic symmetry order. Defaults to 1.
    planarity : float, optional
        Planarity factor (0=random, 1=planar). Defaults to 0.9.

    Returns
    -------
    ndarray
        Array of shape (N, 3) with atomic coordinates.
    """
    import numpy as np

    def symmetrize(p, csym=1):
        if csym <= 1:
            return np.expand_dims(p, axis=0)
        from scipy.spatial.transform import Rotation as R

        ret = [p]
        for si in range(1, csym):
            rot = R.from_euler("z", si * 360 / csym, degrees=True)
            ps = rot.apply(p)
            ret.append(ps)
        ret = np.vstack(ret)
        return ret

    def are_positions_good(new_points, existing_points, min_dist):
        def pairwise_distances(points_a, points_b):
            differences = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
            squared_distances = np.sum(differences**2, axis=-1)
            distances = np.sqrt(squared_distances)
            return distances

        if len(new_points) > 1:
            dist = pairwise_distances(new_points, new_points)
            dist[np.diag_indices_from(dist)] = 1e10
            if np.any(dist < min_dist):
                return False
        dist = pairwise_distances(new_points, existing_points)
        if new_points.shape == existing_points.shape and np.allclose(
            new_points, existing_points
        ):
            dist[np.diag_indices_from(dist)] = 1e10
        if np.any(dist < min_dist):
            return False
        return True

    def next_point(step_length, csym, rmin, rmax, planarity, existing_points):
        n_trials = 1
        while True:
            angle_out_plane_max = 90 * (1 - planarity)  # planarity should be in [0, 1]
            sigma_z = np.abs(np.random.normal(0, angle_out_plane_max / 3))
            sigma_xy = 180 / 3
            if len(existing_points) < 2:
                d0 = existing_points[-1, :] * 0
            else:
                d0 = existing_points[-1, :] - existing_points[-2, :]
                d0 /= np.linalg.norm(d0)
                d0 /= n_trials
                r = np.linalg.norm(existing_points[-1, :])
                d0 *= (rmax - r) / rmax
            d = np.random.normal(0, (sigma_xy, sigma_xy, sigma_z))
            d /= np.linalg.norm(d)
            d = (d0 + d) / np.linalg.norm(d0 + d)
            p = existing_points[-1, :] + step_length * d
            r = np.linalg.norm(p)
            if rmin <= r <= rmax or n_trials > 10:
                break
            n_trials += 1  # avoid dead loop
        p = symmetrize(p, csym)
        return p

    ca_dist = 3.8  # Angstrom

    n_good_points = 0

    max_trials = 10
    n_trials = 0
    while n_trials < max_trials:
        xyz = np.zeros([csym * n_atoms, 3], dtype=float)

        good_start_point_added = False
        ns_trials = 0
        while ns_trials < max_trials:
            r = np.sqrt(np.random.uniform(rmin**2, rmax**2))
            angle = np.random.uniform(-np.pi, np.pi)
            xyz[0, 0] = r * np.sin(angle)
            xyz[0, 1] = r * np.cos(angle)
            xyz[0, 2] = 0
            xyz[0:csym, :] = symmetrize(xyz[0, :], csym=csym)
            if are_positions_good(
                xyz[0:csym, :], xyz[0:csym, :], min_dist=ca_dist * 0.8
            ):
                good_start_point_added = True
                n_good_points = 1
                break
            ns_trials += 1

        if not good_start_point_added:
            n_trials += 1
            break

        for i in range(1, n_atoms):
            good_point_added = False
            ni_trials = 0
            while ni_trials < max_trials:
                existing_points = xyz[: i * csym, :]
                p = next_point(
                    step_length=ca_dist,
                    csym=csym,
                    rmin=rmin,
                    rmax=rmax,
                    planarity=planarity,
                    existing_points=existing_points,
                )
                if are_positions_good(p, existing_points, min_dist=ca_dist * 0.8):
                    xyz[i * csym : (i + 1) * csym, :] = p
                    good_point_added = True
                    n_good_points = i + 1
                    break
                ni_trials += 1
            if not good_point_added:
                break

        if n_good_points == n_atoms:
            break

        n_trials += 1

    return xyz[: n_good_points * csym, :]


def generate_xyz_projections(map3d, is_amyloid=False, apix=None):
    proj_xyz = [map3d.sum(axis=i) for i in [2, 1, 0]]
    if is_amyloid:
        nz = map3d.shape[0]
        nz_center = int(round(4.75 / apix))
        z0 = nz // 2 - nz_center // 2
        proj_xyz[-1] = map3d[z0 : z0 + nz_center].sum(axis=0)
    return proj_xyz


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "denovo3D"), expires_after=7, verbose=0
)
def symmetrize_transform_map(
    data,
    apix,
    twist_degree,
    rise_angstrom,
    csym=1,
    fraction=1.0,
    new_size=None,
    new_apix=None,
    axial_rotation=0,
    tilt=0,
):
    if new_apix > apix:
        data_work = helicon.low_high_pass_filter(
            data, low_pass_fraction=apix / new_apix
        )
    else:
        data_work = data
    m = helicon.apply_helical_symmetry(
        data=data_work,
        apix=apix,
        twist_degree=twist_degree,
        rise_angstrom=rise_angstrom,
        csym=csym,
        new_size=new_size,
        new_apix=new_apix,
        fraction=fraction,
        cpu=helicon.available_cpu(),
    )
    if axial_rotation or tilt:
        m = helicon.transform_map(m, rot=axial_rotation, tilt=tilt)
    return m


def auto_horizontalize(data, refine=False):
    """Automatically rotate and shift an image so the helix is horizontal.

    Parameters
    ----------
    data : ndarray
        2D input image.
    refine : bool, optional
        Whether to refine to sub-degree precision. Defaults to False.

    Returns
    -------
    tuple of (ndarray, float, float)
        Rotated/ shifted image, rotation angle (deg), shift (pixels).
    """
    from skimage.transform import radon
    from scipy.interpolate import interp1d
    from scipy.signal import correlate

    data_work = np.clip(data, 0, None)

    theta, shift_y, diameter = helicon.estimate_helix_rotation_center_diameter(data)

    if refine:  # refine to sub-degree, sub-pixel level

        def score_rotation_shift(x):
            theta, shift_y = x
            data_tmp = helicon.rotate_shift_image(
                data_work, angle=theta, post_shift=(shift_y, 0)
            )
            y = np.sum(data_tmp, axis=1)[1:]
            y += y[::-1]
            score = -np.std(y)
            return score

        from scipy.optimize import fmin

        res = fmin(score_rotation_shift, x0=(theta, shift_y), xtol=1e-2, disp=0)
        theta, shift_y = res

    rotated_shifted_data = helicon.rotate_shift_image(
        data, angle=theta, post_shift=(shift_y, 0), order=3
    )
    return rotated_shifted_data, theta, shift_y


def is_vertical(data):
    """Check if the helical structure in an image is predominantly vertical.

    Parameters
    ----------
    data : ndarray
        2D image.

    Returns
    -------
    bool
        True if vertical, False if horizontal.
    """
    py_max = np.max(np.sum(data, axis=0))
    px_max = np.max(np.sum(data, axis=1))
    if py_max > px_max:
        return True
    else:
        return False


def tilt_psi_dy_str(tilt, psi, dy, sep=" ", sep2="=", unit=True):
    """Format tilt/psi/dy values as a string for labels.

    Parameters
    ----------
    tilt : float
        Tilt angle in degrees.
    psi : float
        In-plane rotation in degrees.
    dy : float
        Shift in Angstroms.
    sep : str, optional
        Separator between parameters. Defaults to " ".
    sep2 : str, optional
        Separator between name and value. Defaults to "=".
    unit : bool, optional
        Whether to append unit symbols. Defaults to True.

    Returns
    -------
    str
        Formatted parameter string.
    """
    tpy_str = ""
    if tilt:
        tpy_str += f"{sep}tilt{sep2}{round(tilt, 2)}" + ("°" if unit else "")
    if psi:
        tpy_str += f"{sep}psi{sep2}{round(psi, 2)}" + ("°" if unit else "")
    if dy:
        tpy_str += f"{sep}dy{sep2}{round(dy, 2)}" + ("Å" if unit else "")
    return tpy_str
