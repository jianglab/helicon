import numpy as np
from scipy.spatial.transform import Rotation
import re

__all__ = [
    "PointGroup",
]


class PointGroup:
    """
    Representation of a rotational point group symmetry.
    Supports generating transformation matrices, applying them to points and rotations,
    and computing symmetry-aware distances between sets of points and rotations.
    """

    def __init__(self, symbol):
        """Initialize the point group from a Schoenflies symbol.

        Parameters
        ----------
        symbol : str
            Schoenflies symmetry symbol (e.g. ``C3``, ``D5``,
            ``T``, ``O``, ``I1``, ``I2``, ``I3``, ``I4``, ``I_5z2x``).
        """
        self.symbol = symbol
        self.matrices = self._generate_matrices(symbol)

    @staticmethod
    def _get_icosahedral_orientation_rotation(symbol):
        """Return rotation to transform base RELION I1 orientation to a given symbol.

        Parameters
        ----------
        symbol : str
            Icosahedral subgroup symbol.

        Returns
        -------
        Rotation
            Rotation object mapping I1 convention to the target convention.
        """
        phi = (1 + np.sqrt(5)) / 2

        # Handle RELION aliases
        symbol_upper = symbol.upper()
        if symbol_upper == "I1":
            suffix = "_2z2x5y"
        elif symbol_upper == "I2":
            suffix = "_2z2x5x"
        elif symbol_upper == "I3":
            suffix = "_5z2y"
        elif symbol_upper == "I4" or symbol_upper == "I":
            suffix = "_5z2x"
        else:
            match = re.match(r"I(\d*)((?:_[0-9a-zA-Z]+)?)", symbol, re.IGNORECASE)
            if not match:
                raise ValueError(f"Invalid icosahedral symbol: {symbol}")
            suffix = match.group(2).lower() if match.group(2) else ""

        if suffix == "_2z2x5y":
            return Rotation.identity()
        elif suffix == "_2z2x5x":
            return Rotation.from_rotvec([0, 0, np.pi / 2])
        elif suffix == "_5z2x":
            # Map 5-fold axis (0, 1, phi) to Z
            theta = np.arctan(1 / phi)
            return Rotation.from_rotvec([theta, 0, 0])
        elif suffix == "_5z2y":
            # Map 5-fold axis (phi, 0, 1) to Z
            theta = -np.arctan(phi)
            return Rotation.from_rotvec([0, theta, 0])
        elif suffix == "_3z2x":
            # Map 3-fold axis (0, phi, 1/phi) to Z
            theta = np.arctan(phi**2)
            return Rotation.from_rotvec([theta, 0, 0])
        elif suffix == "_3z2y":
            # Generate 3z2x and rotate 90 deg around Z
            theta = np.arctan(phi**2)
            return Rotation.from_rotvec([0, 0, np.pi / 2]) * Rotation.from_rotvec(
                [theta, 0, 0]
            )
        else:
            raise ValueError(f"Unsupported icosahedral convention: {symbol}")

    @staticmethod
    def get_icosahedral_conversion_rotation(from_symbol, to_symbol):
        """Return rotation converting between icosahedral conventions.

        Parameters
        ----------
        from_symbol : str
            Source icosahedral convention.
        to_symbol : str
            Target icosahedral convention.

        Returns
        -------
        Rotation
            Rotation that maps points from *from_symbol* to *to_symbol*.
        """
        r1 = PointGroup._get_icosahedral_orientation_rotation(from_symbol)
        r2 = PointGroup._get_icosahedral_orientation_rotation(to_symbol)
        return r2 * r1.inv()

    def _generate_matrices(self, symbol):
        """Generate all symmetry operations as 3x3 matrices.

        Parameters
        ----------
        symbol : str
            Schoenflies symmetry symbol.

        Returns
        -------
        ndarray of shape (G, 3, 3)
            Array of G symmetry transformation matrices.
        """
        match = re.match(r"([A-Za-z]+)(\d*)((?:_[a-zA-Z0-9]*)?)", symbol)
        if not match:
            raise ValueError(f"Invalid Schoenflies symbol: {symbol}")

        main_type_raw, n_str, suffix_raw = match.groups()
        main_type = main_type_raw.upper()
        suffix = suffix_raw.lower()
        n = int(n_str) if n_str else 1

        # Identity is always present
        ops = [np.eye(3)]

        # 1. Add generators for the group
        if main_type == "C":
            if n > 1:
                theta = 2 * np.pi / n
                ops.append(Rotation.from_rotvec([0, 0, theta]).as_matrix())
        elif main_type == "D":
            theta = 2 * np.pi / n
            ops.append(Rotation.from_rotvec([0, 0, theta]).as_matrix())
            ops.append(Rotation.from_rotvec([np.pi, 0, 0]).as_matrix())
        elif main_type == "T":
            # Generator for T: C3 and C2
            c3 = Rotation.from_rotvec(
                np.array([1, 1, 1]) / np.sqrt(3) * (2 * np.pi / 3)
            ).as_matrix()
            c2z = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
            ops.extend([c3, c2z])
        elif main_type == "O":
            # Generator for O: C4 and C3
            c4z = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
            c3 = Rotation.from_rotvec(
                np.array([1, 1, 1]) / np.sqrt(3) * (2 * np.pi / 3)
            ).as_matrix()
            ops.extend([c4z, c3])
        elif main_type == "I":
            # Support various conventions: I_5z2x, I_5z2y, I_2z2x5x, I_2z2x5y, I_3z2x, I_3z2y
            phi = (1 + np.sqrt(5)) / 2

            # Base group: RELION I1 (2z2x5y) - 2-fold on X, Y, Z; 5-fold in YZ plane
            c2z = Rotation.from_rotvec([0, 0, np.pi])
            c2x = Rotation.from_rotvec([np.pi, 0, 0])
            axis_5y = np.array([0, 1, phi]) / np.sqrt(1 + phi**2)
            c5y = Rotation.from_rotvec(axis_5y * 2 * np.pi / 5)
            base_gens = [c2z, c2x, c5y]

            rot = self._get_icosahedral_orientation_rotation(symbol)

            for gen in base_gens:
                ops.append((rot * gen * rot.inv()).as_matrix())
        else:
            raise ValueError(f"Unsupported rotational group: {main_type}")

        # 2. Group Closure
        limit = 1
        if main_type == "T":
            limit = 12
        elif main_type == "O":
            limit = 24
        elif main_type == "I":
            limit = 60
        elif main_type == "D":
            limit = 2 * n
        elif main_type == "C":
            limit = n

        # Iterative multiplication until closure
        generators = [m for m in ops if not np.allclose(m, np.eye(3), atol=1e-7)]

        while len(ops) < limit:
            current_len = len(ops)
            new_ops = []
            for m1 in ops:
                for m2 in generators:
                    for prod in [m1 @ m2, m2 @ m1]:
                        if not any(
                            np.allclose(prod, existing, atol=1e-7) for existing in ops
                        ) and not any(
                            np.allclose(prod, existing, atol=1e-7)
                            for existing in new_ops
                        ):
                            new_ops.append(prod)
                            if len(ops) + len(new_ops) >= limit:
                                break
                    if len(ops) + len(new_ops) >= limit:
                        break
                if len(ops) + len(new_ops) >= limit:
                    break

            ops.extend(new_ops)
            if len(ops) == current_len:
                break

        return np.array(ops)

    def apply_symmetry_to_points(self, points):
        """Apply all symmetry operations to a set of points.

        Parameters
        ----------
        points : ndarray
            ``(N, 3)`` array of N points or ``(3,)`` single point.

        Returns
        -------
        ndarray
            ``(G, N, 3)`` or ``(G, 3)`` array where G is the group order.
        """
        points = np.asanyarray(points)
        if points.ndim == 1:
            return np.einsum("gij,j->gi", self.matrices, points)
        return np.einsum("gij,nj->gni", self.matrices, points)

    def distance_of_points(self, points1, points2, metric="mse"):
        """Compute symmetry-aware distance between two sets of points.

        Finds the minimum distance among all symmetry-equivalent
        configurations of points1. Assumes both arrays are ordered
        with N corresponding points.

        Parameters
        ----------
        points1 : ndarray
            ``(N, 3)`` array of N points.
        points2 : ndarray
            ``(N, 3)`` array of N points.
        metric : str, optional
            Distance metric: ``"mse"``, ``"rmse"``, or ``"max"``.
            Defaults to ``"mse"``.

        Returns
        -------
        float
            Minimum distance across symmetry operations.
        """
        points1 = np.asanyarray(points1)
        points2 = np.asanyarray(points2)

        # (G, N, 3)
        transformed_p1 = self.apply_symmetry_to_points(points1)

        # transformed_p1: (G, N, 3) or (G, 3), points2: (N, 3) or (3,)
        # Differences squared
        if points2.ndim == 1:
            diffs_sq = (transformed_p1 - points2[None, :]) ** 2
            # (G, 3)
            point_dists_sq = np.sum(diffs_sq, axis=-1)
            dists = point_dists_sq  # (G,)
        else:
            diffs_sq = (transformed_p1 - points2[None, :, :]) ** 2
            # (G, N, 3)
            point_dists_sq = np.sum(diffs_sq, axis=-1)
            # (G, N)
            if metric == "mse":
                dists = np.mean(point_dists_sq, axis=1)
            elif metric == "rmse":
                dists = np.sqrt(np.mean(point_dists_sq, axis=1))
            elif metric == "max":
                dists = np.max(
                    np.abs(transformed_p1 - points2[None, :, :]), axis=(1, 2)
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")

        return np.min(dists)

    def get_rotations(self):
        """Return all symmetry operations as Rotation objects.

        Returns
        -------
        Rotation
            ``Rotation`` object containing G symmetry operations.
        """
        return Rotation.from_matrix(self.matrices)

    def apply_symmetry_to_rotations(self, rotations):
        """Apply all symmetry operations to a set of rotations.

        Parameters
        ----------
        rotations : Rotation
            ``Rotation`` object representing N rotations.

        Returns
        -------
        Rotation
            ``Rotation`` object representing ``(G * N)`` rotations.
        """
        sym_rots = self.get_rotations()  # (G,)

        results = []
        for s in sym_rots:
            # s is a single rotation, rotations can be N rotations
            # s * rotations is (N,) rotations
            results.append(s * rotations)

        # Concatenate all (G * N) rotations
        return Rotation.concatenate(results)

    def distance_of_rotations(self, rots1, rots2, metric="geodesic"):
        """Compute symmetry-aware angular distance between two sets of rotations.

        Finds the minimum distance among all symmetry-equivalent
        configurations of rots1.

        Parameters
        ----------
        rots1 : Rotation or ndarray
            ``Rotation`` object or ``(N, 3, 3)`` matrix array.
        rots2 : Rotation or ndarray
            ``Rotation`` object or ``(N, 3, 3)`` matrix array.
        metric : str, optional
            Distance metric: ``"geodesic"`` (angle) or ``"chordal"``
            (Frobenius norm of matrix difference). Defaults to ``"geodesic"``.

        Returns
        -------
        float
            Minimum angular distance across symmetry operations.
        """
        # Ensure rots1 and rots2 are Rotation objects
        if not isinstance(rots1, Rotation):
            rots1 = Rotation.from_matrix(rots1)
        if not isinstance(rots2, Rotation):
            rots2 = Rotation.from_matrix(rots2)

        # (G, N) rotations
        transformed_r1 = self.apply_symmetry_to_rotations(rots1)

        G = len(self.matrices)
        N = len(rots1) if rots1.single is False else 1

        # We need to compare transformed_r1[g, n] with rots2[n]
        # Distance = min_g mean_n dist(S_g * R1_n, R2_n)

        distances = []
        for g in range(G):
            # Compute distance between transformed_r1[g] and rots2
            # Both are (N,) rotation objects
            r_diff = transformed_r1[g].inv() * rots2

            if metric == "geodesic":
                # Angle in radians
                angles = r_diff.magnitude()
                distances.append(np.mean(angles))
            elif metric == "chordal":
                # Frobenius norm of (R1 - R2) is sqrt(2)*chordal_dist
                # scipy doesn't have a direct chordal dist,
                # but it's related to the angle: 2 * sin(angle/2)
                # Or just use matrix differences: ||M1 - M2||_F
                m1 = transformed_r1[g].as_matrix()  # (N, 3, 3)
                m2 = rots2.as_matrix()  # (N, 3, 3)
                diff = m1 - m2
                frob_sq = np.sum(diff**2, axis=(1, 2))
                distances.append(np.mean(np.sqrt(frob_sq)))
            else:
                raise ValueError(f"Unknown metric: {metric}")

        return np.min(distances)

    def __len__(self):
        """Return the number of symmetry operations (group order).

        Returns
        -------
        int
            Number of symmetry matrices.
        """
        return len(self.matrices)
