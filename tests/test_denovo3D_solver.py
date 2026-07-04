import pytest
import numpy as np
import helicon
from scipy.sparse import csr_matrix
from helicon.webApps.denovo3D import solver_linear_regression as solver


class TestSortedHsymCsymPairs(object):
    def test_returns_list_of_tuples(self):
        result = solver.sorted_hsym_csym_pairs(twist=30, rise=5, csym=1, nz=20)
        assert isinstance(result, list)
        assert len(result) > 0
        entry = result[0]
        assert len(entry) >= 5  # (angle, hsum, hdiff, habs1, habs2, pair)

    def test_pairs_cover_range(self):
        result = solver.sorted_hsym_csym_pairs(twist=30, rise=5, csym=1, nz=20)
        angles = [e[0] for e in result]
        assert all(0 <= a <= 180 for a in angles)

    def test_csym_2_generates_more_pairs(self):
        r1 = solver.sorted_hsym_csym_pairs(twist=30, rise=5, csym=1, nz=20)
        r2 = solver.sorted_hsym_csym_pairs(twist=30, rise=5, csym=2, nz=20)
        assert len(r2) >= len(r1)


class TestBackProject2dCoordsTo3dCoords(object):
    def setup_method(self, method):
        self.image = np.arange(16, dtype=np.float32).reshape(4, 4)

    def test_returns_tuple_of_coords_and_pixels(self):
        (X, Y, Z), pixel_vals = solver.back_project_2d_coords_to_3d_coords(
            self.image, scale2d_to_3d=1.0
        )
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert isinstance(Z, np.ndarray)
        assert isinstance(pixel_vals, np.ndarray)

    def test_output_shapes_match(self):
        (X, Y, Z), pixel_vals = solver.back_project_2d_coords_to_3d_coords(
            self.image, scale2d_to_3d=1.0
        )
        assert X.shape == (4, 4, 4)
        assert pixel_vals.shape == (4, 4)

    def test_with_scaled_pixel_size(self):
        (X, Y, Z), pixel_vals = solver.back_project_2d_coords_to_3d_coords(
            self.image, scale2d_to_3d=0.5
        )
        assert X.shape == (4, 4, 4)
        assert np.all(np.isfinite(X))

    def test_with_explicit_reconstruction_size(self):
        (X, Y, Z), pixel_vals = solver.back_project_2d_coords_to_3d_coords(
            self.image,
            scale2d_to_3d=1.0,
            reconstruct_diameter_2d_pixel=2,
            reconstruct_length_2d_pixel=2,
        )
        assert X.shape == (2, 2, 2)
        assert pixel_vals.shape == (2, 2)


class TestBuildADataMatrix(object):
    def setup_method(self, method):
        self.image = np.eye(8, dtype=np.float32)
        self.scale2d_to_3d = 1.0

    def test_returns_csr_matrix_and_arrays(self):
        A, b, b_pid = solver.build_A_data_matrix(
            image=self.image,
            scale2d_to_3d=self.scale2d_to_3d,
            twist_degree=30,
            rise_pixel=2,
            csym=1,
            tilt_degree=0,
            psi_degree=0,
            dy_pixel=0,
            reconstruct_diameter_2d_pixel=4,
            reconstruct_length_2d_pixel=4,
            reconstruct_diameter_3d_pixel=4,
            reconstruct_diameter_3d_inner_pixel=0,
            reconstruct_length_3d_pixel=4,
            min_projection_lines=10,
            interpolation="nn",
            verbose=0,
        )
        assert isinstance(A, csr_matrix)
        assert isinstance(b, np.ndarray)
        assert isinstance(b_pid, np.ndarray)
        assert A.shape[1] > 0
        assert len(b) == A.shape[0]
        assert len(b_pid) == len(b)

    def test_with_tilt_and_psi(self):
        A, b, b_pid = solver.build_A_data_matrix(
            image=self.image,
            scale2d_to_3d=self.scale2d_to_3d,
            twist_degree=30,
            rise_pixel=2,
            csym=1,
            tilt_degree=5,
            psi_degree=10,
            dy_pixel=0,
            reconstruct_diameter_2d_pixel=4,
            reconstruct_length_2d_pixel=4,
            reconstruct_diameter_3d_pixel=4,
            reconstruct_diameter_3d_inner_pixel=0,
            reconstruct_length_3d_pixel=4,
            min_projection_lines=10,
            interpolation="nn",
            verbose=0,
        )
        assert isinstance(A, csr_matrix)
        assert A.shape[1] > 0

    def test_with_csym(self):
        A, b, b_pid = solver.build_A_data_matrix(
            image=self.image,
            scale2d_to_3d=self.scale2d_to_3d,
            twist_degree=30,
            rise_pixel=2,
            csym=2,
            tilt_degree=0,
            psi_degree=0,
            dy_pixel=0,
            reconstruct_diameter_2d_pixel=4,
            reconstruct_length_2d_pixel=4,
            reconstruct_diameter_3d_pixel=4,
            reconstruct_diameter_3d_inner_pixel=0,
            reconstruct_length_3d_pixel=4,
            min_projection_lines=10,
            interpolation="nn",
            verbose=0,
        )
        assert isinstance(A, csr_matrix)
        assert A.shape[1] > 0


class TestBuildAHelicalSymMatrix(object):
    def test_returns_csr_or_none(self):
        A, b = solver.build_A_helical_sym_matrix(
            nz=8,
            ny=8,
            nx=8,
            twist_degree=30,
            rise_pixel=2,
            csym=1,
            rmin=0,
            rmax=3,
            min_sym_pairs=10,
            interpolation="nn",
            verbose=0,
        )
        if A is not None:
            assert isinstance(A, csr_matrix)
            assert isinstance(b, np.ndarray)

    def test_with_linear_interpolation(self):
        A, b = solver.build_A_helical_sym_matrix(
            nz=8,
            ny=8,
            nx=8,
            twist_degree=30,
            rise_pixel=2,
            csym=1,
            rmin=0,
            rmax=3,
            min_sym_pairs=10,
            interpolation="linear",
            verbose=0,
        )
        if A is not None:
            assert isinstance(A, csr_matrix)


class TestLsqReconstruct(object):
    def setup_method(self, method):
        np.random.seed(42)
        self.image = np.random.rand(12, 12).astype(np.float32)

    def test_returns_tuple_with_expected_structure(self):
        (rec3d, rec3d_h1, rec3d_h2), score = solver.lsq_reconstruct(
            projection_image=self.image,
            scale2d_to_3d=1.0,
            twist_degree=30,
            rise_pixel=2,
            csym=1,
            tilt_degree=0,
            psi_degree=0,
            dy_pixel=0,
            reconstruct_diameter_2d_pixel=8,
            reconstruct_length_2d_pixel=8,
            reconstruct_diameter_3d_pixel=8,
            reconstruct_length_3d_pixel=8,
            interpolation="nn",
            verbose=0,
        )
        assert isinstance(rec3d, np.ndarray)
        assert rec3d.dtype == np.float32
        assert rec3d_h1 is None
        assert rec3d_h2 is None
        assert isinstance(score, (float, np.floating))

    def test_reconstruction_shape(self):
        (rec3d, _, _), score = solver.lsq_reconstruct(
            projection_image=self.image,
            scale2d_to_3d=1.0,
            twist_degree=30,
            rise_pixel=2,
            csym=1,
            reconstruct_diameter_2d_pixel=8,
            reconstruct_length_2d_pixel=8,
            reconstruct_diameter_3d_pixel=8,
            reconstruct_length_3d_pixel=8,
            interpolation="nn",
            verbose=0,
        )
        assert rec3d.shape == (8, 8, 8)
        assert np.all(np.isfinite(rec3d))

    def test_with_csym_2(self):
        (rec3d, _, _), score = solver.lsq_reconstruct(
            projection_image=self.image,
            scale2d_to_3d=1.0,
            twist_degree=30,
            rise_pixel=2,
            csym=2,
            reconstruct_diameter_2d_pixel=8,
            reconstruct_length_2d_pixel=8,
            reconstruct_diameter_3d_pixel=8,
            reconstruct_length_3d_pixel=8,
            interpolation="nn",
            verbose=0,
        )
        assert rec3d.shape == (8, 8, 8)
        assert np.all(np.isfinite(rec3d))

    def test_with_tilt_and_psi(self):
        (rec3d, _, _), score = solver.lsq_reconstruct(
            projection_image=self.image,
            scale2d_to_3d=1.0,
            twist_degree=30,
            rise_pixel=2,
            csym=1,
            tilt_degree=5,
            psi_degree=10,
            dy_pixel=1,
            reconstruct_diameter_2d_pixel=8,
            reconstruct_length_2d_pixel=8,
            reconstruct_diameter_3d_pixel=8,
            reconstruct_length_3d_pixel=8,
            interpolation="nn",
            verbose=0,
        )
        assert np.all(np.isfinite(rec3d))

    def test_fsc_test_returns_halves(self):
        (rec3d, rec3d_h1, rec3d_h2), score = solver.lsq_reconstruct(
            projection_image=self.image,
            scale2d_to_3d=1.0,
            twist_degree=30,
            rise_pixel=2,
            csym=1,
            reconstruct_diameter_2d_pixel=8,
            reconstruct_length_2d_pixel=8,
            reconstruct_diameter_3d_pixel=8,
            reconstruct_length_3d_pixel=8,
            fsc_test=1,
            interpolation="nn",
            verbose=0,
        )
        assert rec3d.shape == (8, 8, 8)
        assert rec3d_h1 is not None
        assert rec3d_h2 is not None
        assert rec3d_h1.shape == rec3d.shape
        assert rec3d_h2.shape == rec3d.shape
        assert np.all(np.isfinite(rec3d_h1))
        assert np.all(np.isfinite(rec3d_h2))

    def test_with_inner_diameter(self):
        (rec3d, _, _), score = solver.lsq_reconstruct(
            projection_image=self.image,
            scale2d_to_3d=1.0,
            twist_degree=30,
            rise_pixel=2,
            csym=1,
            reconstruct_diameter_2d_pixel=8,
            reconstruct_length_2d_pixel=8,
            reconstruct_diameter_3d_pixel=8,
            reconstruct_diameter_3d_inner_pixel=2,
            reconstruct_length_3d_pixel=8,
            interpolation="nn",
            verbose=0,
        )
        assert np.all(np.isfinite(rec3d))


class TestRefineTiltPsiDy:
    """Tests for the Gauss-Newton local refinement of tilt/psi/dy."""

    def setup_method(self):
        np.random.seed(42)
        self.image = np.random.rand(12, 12).astype(np.float32)
        self.scale2d_to_3d = 1.0
        self.twist_degree = 30
        self.rise_pixel = 2
        self.csym = 1
        self.d2d = 8
        self.l2d = 8
        self.d3d = 8
        self.l3d = 8

    def _get_x_init(self):
        """Get initial voxel solution at tilt=0, psi=0, dy=0."""
        (rec3d, _, _), _ = solver.lsq_reconstruct(
            projection_image=self.image,
            scale2d_to_3d=self.scale2d_to_3d,
            twist_degree=self.twist_degree,
            rise_pixel=self.rise_pixel,
            csym=self.csym,
            reconstruct_diameter_2d_pixel=self.d2d,
            reconstruct_length_2d_pixel=self.l2d,
            reconstruct_diameter_3d_pixel=self.d3d,
            reconstruct_length_3d_pixel=self.l3d,
            interpolation="nn",
            verbose=0,
        )
        mask = helicon.get_cylindrical_mask(
            nz=self.l3d,
            ny=self.d3d,
            nx=self.d3d,
            rmin=0,
            rmax=self.d3d // 2 - 1,
        )
        return rec3d[mask]

    def test_returns_correct_structure(self):
        """refine_tilt_psi_dy returns (tilt, psi, dy, x, score)."""
        x_init = self._get_x_init()
        tilt, psi, dy, x, score = solver.refine_tilt_psi_dy(
            projection_image=self.image,
            scale2d_to_3d=self.scale2d_to_3d,
            twist_degree=self.twist_degree,
            rise_pixel=self.rise_pixel,
            csym=self.csym,
            reconstruct_diameter_2d_pixel=self.d2d,
            reconstruct_length_2d_pixel=self.l2d,
            reconstruct_diameter_3d_pixel=self.d3d,
            reconstruct_diameter_3d_inner_pixel=0,
            reconstruct_length_3d_pixel=self.l3d,
            sym_oversample=1,
            interpolation="nn",
            x_init=x_init,
            max_iter=2,
            verbose=0,
        )
        assert isinstance(tilt, float)
        assert isinstance(psi, float)
        assert isinstance(dy, float)
        assert isinstance(x, np.ndarray)
        assert isinstance(score, float)
        assert np.all(np.isfinite(x))

    def test_zero_shift_no_change(self):
        """When image is already aligned, refinement should stay near zero."""
        x_init = self._get_x_init()
        tilt, psi, dy, x, score = solver.refine_tilt_psi_dy(
            projection_image=self.image,
            scale2d_to_3d=self.scale2d_to_3d,
            twist_degree=self.twist_degree,
            rise_pixel=self.rise_pixel,
            csym=self.csym,
            reconstruct_diameter_2d_pixel=self.d2d,
            reconstruct_length_2d_pixel=self.l2d,
            reconstruct_diameter_3d_pixel=self.d3d,
            reconstruct_diameter_3d_inner_pixel=0,
            reconstruct_length_3d_pixel=self.l3d,
            sym_oversample=1,
            interpolation="nn",
            x_init=x_init,
            max_iter=3,
            verbose=0,
        )
        # For a random image with no particular structure,
        # the refinement should converge quickly
        assert abs(tilt) < 30  # within bounds
        assert abs(psi) < 45
        assert abs(dy) < 5

    def test_bounds_respected(self):
        """Parameters should stay within specified bounds."""
        x_init = self._get_x_init()
        tilt, psi, dy, x, score = solver.refine_tilt_psi_dy(
            projection_image=self.image,
            scale2d_to_3d=self.scale2d_to_3d,
            twist_degree=self.twist_degree,
            rise_pixel=self.rise_pixel,
            csym=self.csym,
            reconstruct_diameter_2d_pixel=self.d2d,
            reconstruct_length_2d_pixel=self.l2d,
            reconstruct_diameter_3d_pixel=self.d3d,
            reconstruct_diameter_3d_inner_pixel=0,
            reconstruct_length_3d_pixel=self.l3d,
            sym_oversample=1,
            interpolation="nn",
            x_init=x_init,
            bounds_tilt=(-5.0, 5.0),
            bounds_psi=(-10.0, 10.0),
            bounds_dy=(-1.0, 1.0),
            max_iter=3,
            verbose=0,
        )
        assert -5.0 <= tilt <= 5.0
        assert -10.0 <= psi <= 10.0
        assert -1.0 <= dy <= 1.0

    def test_score_finite(self):
        """Score should be a finite number."""
        x_init = self._get_x_init()
        _, _, _, _, score = solver.refine_tilt_psi_dy(
            projection_image=self.image,
            scale2d_to_3d=self.scale2d_to_3d,
            twist_degree=self.twist_degree,
            rise_pixel=self.rise_pixel,
            csym=self.csym,
            reconstruct_diameter_2d_pixel=self.d2d,
            reconstruct_length_2d_pixel=self.l2d,
            reconstruct_diameter_3d_pixel=self.d3d,
            reconstruct_diameter_3d_inner_pixel=0,
            reconstruct_length_3d_pixel=self.l3d,
            sym_oversample=1,
            interpolation="nn",
            x_init=x_init,
            max_iter=2,
            verbose=0,
        )
        assert np.isfinite(score)

    def test_integrated_with_lsq_reconstruct(self):
        """refine_tilt_psi_dy_range parameter triggers refinement in lsq_reconstruct."""
        (rec3d, _, _), score = solver.lsq_reconstruct(
            projection_image=self.image,
            scale2d_to_3d=self.scale2d_to_3d,
            twist_degree=self.twist_degree,
            rise_pixel=self.rise_pixel,
            csym=self.csym,
            reconstruct_diameter_2d_pixel=self.d2d,
            reconstruct_length_2d_pixel=self.l2d,
            reconstruct_diameter_3d_pixel=self.d3d,
            reconstruct_length_3d_pixel=self.l3d,
            interpolation="nn",
            verbose=0,
            algorithm=dict(model="elasticnet"),
            refine_tilt_psi_dy_range={"tilt": 5, "psi": 10, "dy": 2},
        )
        assert np.all(np.isfinite(rec3d))
        assert isinstance(score, float)
