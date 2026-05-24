"""Angular math utilities — Euler angles, quaternions, and angle wrapping."""

from __future__ import annotations

import math
import typing

import numpy as np

__all__ = [
    "angular_difference",
    "set_angle_range",
    "set_to_periodic_range",
    "euler_relion2eman",
    "euler_eman2relion",
    "eman_euler2quaternion",
    "relion_euler2quaternion",
    "quaternion2euler",
    "average_quaternions",
    "average_relion_eulers",
    "angular_distance",
]


def angular_difference(
    angle1: float | np.ndarray, angle2: float | np.ndarray, period: float = 360
) -> float | np.ndarray:
    """Compute the minimal angular difference between two arrays of angles.

    The result considers wrapping and is in the range [-period/2, period/2).

    Parameters
    ----------
    angle1 : float or np.ndarray
        First angle(s) in degrees or radians.
    angle2 : float or np.ndarray
        Second angle(s) in degrees or radians.
    period : float, optional
        Period of the angles. Defaults to 360.

    Returns
    -------
    float or np.ndarray
        Array or scalar of minimal angular differences.
    """
    diff = np.asarray(angle1) - np.asarray(angle2)
    diff = (diff + period / 2) % period - period / 2
    return diff


def set_angle_range(
    angle: float | np.ndarray, range: typing.Sequence[float] = (-180, 180)
) -> float | np.ndarray:
    """Wrap an angle or array of angles into a specified range.

    Parameters
    ----------
    angle : float or np.ndarray
        Input angle(s).
    range : sequence of float, optional
        Target ``[low, high]`` range. Defaults to ``[-180, 180]``.

    Returns
    -------
    float or np.ndarray
        Angle(s) wrapped into the range.
    """
    v0, v1 = range[0], range[-1]
    delta = v1 - v0
    if isinstance(angle, np.ndarray):
        pos = angle > v0
        neg = angle <= v0
        ret = np.empty_like(angle)
        ret[pos] = np.fmod(angle[pos] - v0, delta) + v0
        ret[neg] = v1 - np.fmod(v0 - angle[neg], delta)
    else:
        if angle > v0:
            ret = np.fmod(angle - v0, delta) + v0
        else:
            ret = v1 - np.fmod(v0 - angle, delta)
    return ret


def set_to_periodic_range(v: float, min: float = -180, max: float = 180) -> float:
    """Wrap a scalar value into a periodic range.

    Parameters
    ----------
    v : float
        Input value.
    min : float, optional
        Lower bound. Defaults to -180.
    max : float, optional
        Upper bound. Defaults to 180.

    Returns
    -------
    float
        Value wrapped into [min, max].
    """
    if min <= v <= max:
        return v
    tmp = math.fmod(v - min, max - min)
    if tmp >= 0:
        tmp += min
    else:
        tmp += max
    return tmp


def euler_relion2eman(
    rot: float, tilt: float, psi: float
) -> tuple[float, float, float]:
    """Convert RELION Euler angles to EMAN Euler angles.

    Parameters
    ----------
    rot : float
        RELION rot angle in degrees.
    tilt : float
        RELION tilt angle in degrees.
    psi : float
        RELION psi angle in degrees.

    Returns
    -------
    tuple of (float, float, float)
        EMAN (az, alt, phi) angles in degrees.
    """
    az = rot + 90.0
    alt = tilt
    phi = psi - 90.0
    return az, alt, phi


def euler_eman2relion(az: float, alt: float, phi: float) -> tuple[float, float, float]:
    """Convert EMAN Euler angles to RELION Euler angles.

    Parameters
    ----------
    az : float
        EMAN az angle in degrees.
    alt : float
        EMAN alt angle in degrees.
    phi : float
        EMAN phi angle in degrees.

    Returns
    -------
    tuple of (float, float, float)
        RELION (rot, tilt, psi) angles in degrees.
    """
    rot = az - 90
    tilt = alt
    psi = phi + 90
    return rot, tilt, psi


def eman_euler2quaternion(
    az: float | np.ndarray, alt: float | np.ndarray, phi: float | np.ndarray
) -> np.ndarray:
    """Convert EMAN Euler angles to a quaternion.

    Parameters
    ----------
    az : float or np.ndarray
        EMAN az (azimuth) angle in degrees.
    alt : float or np.ndarray
        EMAN alt (altitude) angle in degrees.
    phi : float or np.ndarray
        EMAN phi angle in degrees.

    Returns
    -------
    np.ndarray
        Normalized quaternion, shape (4,) or (N, 4), scalar-first (w, x, y, z).
    """
    from scipy.spatial.transform import Rotation as R

    rot, tilt, psi = az - 90.0, alt, phi + 90.0
    r = R.from_euler("zyz", np.vstack((rot, tilt, psi)).T, degrees=True)
    q = r.as_quat()  # scipy: scalar-last (x, y, z, w)
    if q.ndim == 1:
        q = q.reshape((1, 4))
    return np.hstack((q[:, 3:4], q[:, :3]))  # convert to scalar-first (w, x, y, z)


def relion_euler2quaternion(
    rot: float | np.ndarray, tilt: float | np.ndarray, psi: float | np.ndarray
) -> np.ndarray:
    """Convert RELION Euler angles to a quaternion.

    Parameters
    ----------
    rot : float or np.ndarray
        RELION rot angle in degrees.
    tilt : float or np.ndarray
        RELION tilt angle in degrees.
    psi : float or np.ndarray
        RELION psi angle in degrees.

    Returns
    -------
    np.ndarray
        NumPy array of shape (N, 4) normalized quaternion, scalar-first (w, x, y, z).
    """
    from scipy.spatial.transform import Rotation as R

    r = R.from_euler("zyz", np.vstack((rot, tilt, psi)).T, degrees=True)
    q = r.as_quat()  # scipy: scalar-last (x, y, z, w)
    if q.ndim == 1:
        q = q.reshape((1, 4))
    return np.hstack((q[:, 3:4], q[:, :3]))  # convert to scalar-first (w, x, y, z)


def quaternion2euler(
    q: np.ndarray, euler_convention: str = "relion"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a quaternion to Euler angles.

    Parameters
    ----------
    q : np.ndarray
        Quaternion(s), shape (4,) or (N, 4), ordering (w, x, y, z).
    euler_convention : str, optional
        ``"relion"`` or ``"eman"``.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        Tuple of (rot, tilt, psi) angles in degrees.
    """
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(q)
    if q.ndim == 1:
        q = q.reshape((1, 4))
    # Convert scalar-first (w, x, y, z) to scipy scalar-last (x, y, z, w)
    q_scipy = np.hstack((q[:, 1:4], q[:, 0:1]))
    r = R.from_quat(q_scipy)
    euler = r.as_euler("zyz", degrees=True)
    rot, tilt, psi = euler[:, 0], euler[:, 1], euler[:, 2]
    rot = set_angle_range(rot, range=(-180, 180))
    tilt = set_angle_range(tilt, range=(-180, 180))
    psi = set_angle_range(psi, range=(-180, 180))
    if euler_convention == "relion":
        return rot, tilt, psi
    elif euler_convention == "eman":
        return euler_relion2eman(rot, tilt, psi)
    raise ValueError(f"Unknown euler_convention: {euler_convention}")


def average_quaternions(Q: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Average a set of quaternions, handling sign ambiguity.

    Parameters
    ----------
    Q : np.ndarray
        Array of shape (N, 4) containing N quaternions (w, x, y, z).
    w : np.ndarray, optional
        Optional weight vector of length N.

    Returns
    -------
    np.ndarray
        Array of shape (4,) representing the average quaternion.
    """
    assert w is None or len(w) == Q.shape[0]

    M = Q.shape[0]
    A = np.zeros((4, 4))
    weightSum = 0

    if w is None:
        w = np.ones(M)

    for i in range(M):
        q = Q[i, :]
        A = w[i] * np.outer(q, q) + A
        weightSum += w[i]

    A = (1.0 / weightSum) * A
    eigenValues, eigenVectors = np.linalg.eig(A)
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
    ret = np.real(eigenVectors[:, 0]).ravel()
    ret = np.array(ret)
    return ret


def average_relion_eulers(
    rot: np.ndarray,
    tilt: np.ndarray,
    psi: np.ndarray,
    weights: np.ndarray | None = None,
    return_quaternion: bool = False,
) -> tuple | np.ndarray:
    """Average RELION Euler angles via quaternion averaging.

    Parameters
    ----------
    rot : np.ndarray
        Array of RELION rot angles in degrees.
    tilt : np.ndarray
        Array of RELION tilt angles in degrees.
    psi : np.ndarray
        Array of RELION psi angles in degrees.
    weights : np.ndarray, optional
        Optional weight array.
    return_quaternion : bool, optional
        If True, return the average quaternion.

    Returns
    -------
    tuple or np.ndarray
        Tuple of (rot_mean, tilt_mean, psi_mean) in degrees, or a quaternion.
    """
    assert len(rot) == len(tilt) and len(rot) == len(psi)
    if weights is not None:
        assert len(weights) == len(rot)
    Q = relion_euler2quaternion(rot, tilt, psi)
    qm = average_quaternions(Q, w=weights)
    if return_quaternion:
        return qm
    rot_mean, tilt_mean, psi_mean = quaternion2euler(qm, euler_convention="relion")
    return rot_mean, tilt_mean, psi_mean


def angular_distance(rotation_1: typing.Any, rotation_2: typing.Any) -> float:
    """Compute angular distance between two rotations in degrees.

    Parameters
    ----------
    rotation_1 : scipy.spatial.transform.Rotation
        A ``scipy.spatial.transform.Rotation`` object.
    rotation_2 : scipy.spatial.transform.Rotation
        A ``scipy.spatial.transform.Rotation`` object.

    Returns
    -------
    float
        Angular distance in degrees.
    """
    mag = (rotation_1.inv() * rotation_2).magnitude()
    return np.rad2deg(mag)
