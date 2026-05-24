"""Euler angle and quaternion conversion utilities.

Re-exports from :mod:`angular` for backward compatibility.
"""

from .angular import (  # noqa: F401
    euler_relion2eman,
    euler_eman2relion,
    eman_euler2quaternion,
    relion_euler2quaternion,
    quaternion2euler,
    average_quaternions,
    average_relion_eulers,
    angular_distance,
)
