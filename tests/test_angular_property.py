"""Property-based tests for angular math using hypothesis.

Verifies round-trip invariants for Euler angle and quaternion conversions
and range invariants for angle-wrapping functions.
"""

import math
import numpy as np
from hypothesis import given, settings, strategies as st
from helicon.lib.angular import (
    euler_relion2eman,
    euler_eman2relion,
    relion_euler2quaternion,
    eman_euler2quaternion,
    quaternion2euler,
    angular_difference,
    set_angle_range,
    set_to_periodic_range,
)

# Disable deadline: first call imports quaternionic (slow)
ANGULAR_SETTINGS = settings(deadline=None, max_examples=100)

# Reasonable range for Euler angles in degrees
euler_angle = st.floats(
    min_value=-360, max_value=360, allow_nan=False, allow_infinity=False, width=32
)


def _quaternion_diff(q1, q2):
    """Angular distance (degrees) between two quaternion arrays, handling sign."""
    # Normalize both
    q1 = np.asarray(q1).ravel()
    q2 = np.asarray(q2).ravel()
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    # q and -q represent the same rotation
    dot = abs(np.dot(q1, q2))
    dot = min(dot, 1.0)  # clamp for numerical stability
    return float(np.rad2deg(2 * math.acos(dot)))


@ANGULAR_SETTINGS
@given(euler_angle, euler_angle, euler_angle)
def test_relion_euler_quaternion_round_trip(rot, tilt, psi):
    """RELION Euler → quaternion → RELION Euler → quaternion should be the same quaternion."""
    q1 = relion_euler2quaternion(rot, tilt, psi)
    rot2, tilt2, psi2 = quaternion2euler(q1, euler_convention="relion")
    q2 = relion_euler2quaternion(rot2, tilt2, psi2)
    assert _quaternion_diff(q1, q2) < 1e-3


@ANGULAR_SETTINGS
@given(euler_angle, euler_angle, euler_angle)
def test_eman_euler_quaternion_round_trip(az, alt, phi):
    """EMAN Euler → quaternion → EMAN Euler → quaternion should be the same quaternion."""
    q1 = eman_euler2quaternion(az, alt, phi)
    az2, alt2, phi2 = quaternion2euler(q1, euler_convention="eman")
    q2 = eman_euler2quaternion(az2, alt2, phi2)
    assert _quaternion_diff(q1, q2) < 1e-3


@ANGULAR_SETTINGS
@given(euler_angle, euler_angle, euler_angle)
def test_relion_eman_convention_consistency(rot, tilt, psi):
    """RELION→quaternion and EMAN→quaternion from converted angles should match."""
    q_relion = relion_euler2quaternion(rot, tilt, psi)
    az, alt, phi = euler_relion2eman(rot, tilt, psi)
    q_eman = eman_euler2quaternion(az, alt, phi)
    assert _quaternion_diff(q_relion, q_eman) < 1e-3


@ANGULAR_SETTINGS
@given(euler_angle, euler_angle, euler_angle)
def test_relion_eman_direct_round_trip(rot, tilt, psi):
    """RELION → EMAN → RELION should be identity (numerically exact)."""
    az, alt, phi = euler_relion2eman(rot, tilt, psi)
    rot2, tilt2, psi2 = euler_eman2relion(az, alt, phi)
    assert math.isclose(rot, rot2, abs_tol=1e-10)
    assert math.isclose(tilt, tilt2, abs_tol=1e-10)
    assert math.isclose(psi, psi2, abs_tol=1e-10)


@given(euler_angle, euler_angle)
def test_angular_difference_antisymmetric(a, b):
    """angular_difference(a, b) ≈ -angular_difference(b, a), ignoring boundary at ±period/2."""
    d1 = angular_difference(a, b)
    d2 = angular_difference(b, a)
    result = d1 + d2  # should be 0 for true antisymmetry
    # At exactly ±period/2, both sides return -180 (half-open range convention)
    assert math.isclose(result, 0, abs_tol=1e-10) or math.isclose(
        abs(result), 360, abs_tol=1e-10
    )


@given(euler_angle)
def test_angular_difference_self_is_zero(a):
    """angular_difference(a, a) should be 0."""
    assert angular_difference(a, a) == 0.0


@given(euler_angle)
def test_set_angle_range_in_range(angle):
    """set_angle_range should always return a value in [-180, 180)."""
    result = set_angle_range(angle)
    assert -180 <= result < 180, f"{result} not in [-180, 180)"


@given(st.floats(min_value=-720, max_value=720, allow_nan=False, allow_infinity=False))
def test_set_to_periodic_range_in_range(v):
    """set_to_periodic_range should always return a value in [-180, 180]."""
    result = set_to_periodic_range(v)
    assert -180 <= result <= 180, f"{result} not in [-180, 180]"


@given(euler_angle)
def test_set_to_periodic_range_fixed_point(a):
    """Values already in [-180, 180] should pass through unchanged."""
    result = set_to_periodic_range(a)
    if -180 <= a <= 180:
        assert math.isclose(result, a, abs_tol=1e-10)
    else:
        assert -180 <= result <= 180
