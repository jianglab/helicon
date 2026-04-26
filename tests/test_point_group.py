import numpy as np
from scipy.spatial.transform import Rotation
import sys
import os

# Add src to path if needed
sys.path.append(os.path.abspath("src"))

from helicon.lib.point_group import PointGroup


def test_icosahedral_variants():
    print("Testing icosahedral variants...")
    variants = [
        "I",
        "I_5z2x",
        "I_5z2y",
        "I_2z2x5x",
        "I_2z2x5y",
        "I_3z2x",
        "I_3z2y",
        "I1",
        "I2",
        "I3",
        "I4",
    ]
    for v in variants:
        pg = PointGroup(v)
        assert len(pg) == 60, f"Order of {v} should be 60, got {len(pg)}"

        rots = pg.get_rotations()
        axes_5 = []
        axes_3 = []
        axes_2 = []
        for r in rots:
            vec = r.as_rotvec()
            angle = np.linalg.norm(vec)
            if angle < 1e-7:
                continue
            axis = vec / angle

            # Normalize for uniqueness
            angle = angle % (2 * np.pi)
            if angle > np.pi + 1e-7:
                angle = 2 * np.pi - angle
                axis = -axis

            if axis[2] < -1e-7:
                axis = -axis
            elif abs(axis[2]) < 1e-7:
                if axis[1] < -1e-7:
                    axis = -axis
                elif abs(axis[1]) < 1e-7:
                    if axis[0] < -1e-7:
                        axis = -axis

            if np.allclose(angle, 2 * np.pi / 5, atol=1e-5) or np.allclose(
                angle, 4 * np.pi / 5, atol=1e-5
            ):
                if not any(np.allclose(axis, a, atol=1e-4) for a in axes_5):
                    axes_5.append(axis)
            elif np.allclose(angle, 2 * np.pi / 3, atol=1e-5):
                if not any(np.allclose(axis, a, atol=1e-4) for a in axes_3):
                    axes_3.append(axis)
            elif np.allclose(angle, np.pi, atol=1e-5):
                if not any(np.allclose(axis, a, atol=1e-4) for a in axes_2):
                    axes_2.append(axis)

        assert len(axes_5) == 6
        assert len(axes_3) == 10
        assert len(axes_2) == 15

        if v in ["I", "I_5z2x"]:
            assert any(
                np.allclose(abs(a), [0, 0, 1], atol=1e-5) for a in axes_5
            ), f"{v} missing 5z"
            assert any(
                np.allclose(abs(a), [1, 0, 0], atol=1e-5) for a in axes_2
            ), f"{v} missing 2x"
        elif v == "I_5z2y":
            assert any(
                np.allclose(abs(a), [0, 0, 1], atol=1e-5) for a in axes_5
            ), f"{v} missing 5z"
            assert any(
                np.allclose(abs(a), [0, 1, 0], atol=1e-5) for a in axes_2
            ), f"{v} missing 2y"
        elif v == "I_2z2x5x":
            assert any(
                np.allclose(abs(a), [0, 0, 1], atol=1e-5) for a in axes_2
            ), f"{v} missing 2z"
            assert any(
                np.allclose(abs(a), [1, 0, 0], atol=1e-5) for a in axes_2
            ), f"{v} missing 2x"
            assert any(abs(a[1]) < 1e-5 for a in axes_5), f"{v} missing 5xz"
        elif v == "I_2z2x5y":
            assert any(
                np.allclose(abs(a), [0, 0, 1], atol=1e-5) for a in axes_2
            ), f"{v} missing 2z"
            assert any(
                np.allclose(abs(a), [1, 0, 0], atol=1e-5) for a in axes_2
            ), f"{v} missing 2x"
            assert any(abs(a[0]) < 1e-5 for a in axes_5)
        elif v == "I_3z2x":
            assert any(
                np.allclose(abs(a), [0, 0, 1], atol=1e-5) for a in axes_3
            ), f"{v} missing 3z"
            assert any(
                np.allclose(abs(a), [1, 0, 0], atol=1e-5) for a in axes_2
            ), f"{v} missing 2x"
        elif v == "I_3z2y":
            assert any(
                np.allclose(abs(a), [0, 0, 1], atol=1e-5) for a in axes_3
            ), f"{v} missing 3z"
            assert any(
                np.allclose(abs(a), [0, 1, 0], atol=1e-5) for a in axes_2
            ), f"{v} missing 2y"
    print("Icosahedral variants tested successfully.")


def test_point_distance():
    print("Testing point distance...")
    pg = PointGroup("I")
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([-1.0, 0.0, 0.0])
    # Distance should be 0 because of symmetry
    d = pg.distance_of_points(p1, p2)
    assert np.allclose(d, 0.0, atol=1e-7)
    print("Point distance tested successfully.")


def test_rotation_distance():
    print("Testing rotation distance...")
    pg = PointGroup("I")
    r1 = Rotation.identity()
    # Rotation about a 5-fold axis by 2pi/5
    r2 = Rotation.from_rotvec([0, 0, 2 * np.pi / 5])
    # Distance should be 0 because of symmetry
    d = pg.distance_of_rotations(r1, r2)
    assert np.allclose(d, 0.0, atol=1e-7)
    print("Rotation distance tested successfully.")


def test_conversion_rotation():
    print("Testing conversion rotation...")
    # Map from I_5z2x to I_2z2x5y
    rot = PointGroup.get_icosahedral_conversion_rotation("I_5z2x", "I_2z2x5y")

    # Vector [0, 0, 1] is 5-fold in I_5z2x
    v_5_5z = np.array([0, 0, 1])
    v_5_translated = rot.apply(v_5_5z)

    # Check if v_5_translated is a 5-fold axis in I_2z2x5y
    pg_2z2x5y = PointGroup("I_2z2x5y")
    found = False
    for r in pg_2z2x5y.get_rotations():
        vec = r.as_rotvec()
        ang = np.linalg.norm(vec)
        if np.allclose(ang, 2 * np.pi / 5, atol=1e-5) or np.allclose(
            ang, 4 * np.pi / 5, atol=1e-5
        ):
            ax = vec / ang
            if np.allclose(abs(ax), abs(v_5_translated), atol=1e-5):
                found = True
                break
    assert found, "Conversion rotation did not map 5-fold axis correctly"
    print("Conversion rotation tested successfully.")


def test_aliases():
    print("Testing RELION aliases...")
    aliases = {"I1": "I_2z2x5y", "I2": "I_2z2x5x", "I3": "I_5z2y", "I4": "I_5z2x"}
    for a, full in aliases.items():
        pg1 = PointGroup(a)
        pg2 = PointGroup(full)
        assert len(pg1) == len(pg2) == 60
        # Check if matrices are the same (set-wise)
        m1 = pg1.matrices
        m2 = pg2.matrices
        for mat1 in m1:
            if not any(np.allclose(mat1, mat2, atol=1e-7) for mat2 in m2):
                raise AssertionError(f"Alias {a} matrices do not match {full}")
    print("RELION aliases tested successfully.")


def test_case_insensitivity():
    print("Testing case-insensitivity...")
    # Test main types and suffixes
    test_cases = [
        ("c2", "C2"),
        ("D4", "d4"),
        ("t", "T"),
        ("O", "o"),
        ("i", "I"),
        ("i_5Z2X", "I_5z2x"),
        ("I_5Z2y", "i_5z2Y"),
        ("i1", "I1"),
        ("i4", "I4"),
    ]
    for low, high in test_cases:
        pg_low = PointGroup(low)
        pg_high = PointGroup(high)
        assert len(pg_low) == len(pg_high), f"Order mismatch for {low} and {high}"
        # Check if matrices are the same (set-wise)
        m1 = pg_low.matrices
        m2 = pg_high.matrices
        for mat1 in m1:
            if not any(np.allclose(mat1, mat2, atol=1e-7) for mat2 in m2):
                raise AssertionError(f"Matrices for {low} do not match {high}")
    print("Case-insensitivity tested successfully.")


if __name__ == "__main__":
    test_icosahedral_variants()
    test_point_distance()
    test_rotation_distance()
    test_conversion_rotation()
    test_aliases()
    test_case_insensitivity()
    print("All tests passed.")
