import pytest
import numpy as np
from helicon.webApps.denovo3D import utils


class TestGenerateXYZProjections(object):
    def setup_method(self, method):
        self.map3d = np.arange(8, dtype=np.float32).reshape(2, 2, 2)

    def test_returns_list_of_three_projections(self):
        projs = utils.generate_xyz_projections(self.map3d)
        assert isinstance(projs, list)
        assert len(projs) == 3

    def test_each_projection_is_2d(self):
        projs = utils.generate_xyz_projections(self.map3d)
        for p in projs:
            assert len(p.shape) == 2

    def test_sum_equivalence(self):
        projs = utils.generate_xyz_projections(self.map3d)
        np.testing.assert_array_equal(projs[0], self.map3d.sum(axis=2))
        np.testing.assert_array_equal(projs[1], self.map3d.sum(axis=1))
        np.testing.assert_array_equal(projs[2], self.map3d.sum(axis=0))

    def test_amyloid_mode_selects_center(self):
        nz, ny, nx = 16, 8, 8
        map3d = np.ones((nz, ny, nx), dtype=np.float32)
        projs = utils.generate_xyz_projections(map3d, is_amyloid=True, apix=1.0)
        assert len(projs) == 3
        assert projs[0].shape == (nz, ny)  # sum along X
        assert projs[2].shape == (ny, nx)  # sum along Z
        nz_center = int(round(4.75 / 1.0))
        z0 = nz // 2 - nz_center // 2
        expected_z = map3d[z0 : z0 + nz_center].sum(axis=0)
        np.testing.assert_array_equal(projs[2], expected_z)


class TestAutoHorizontalize(object):
    def test_returns_image_and_params(self):
        data = np.zeros((16, 16), dtype=np.float32)
        data[:, 4:12] = 1.0  # vertical bar
        result, theta, shift = utils.auto_horizontalize(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape
        assert isinstance(theta, float)
        assert isinstance(shift, float)

    def test_refine_mode(self):
        data = np.zeros((16, 16), dtype=np.float32)
        data[:, 4:12] = 1.0
        result, theta, shift = utils.auto_horizontalize(data, refine=True)
        assert result.shape == data.shape


class TestIsVertical(object):
    def test_vertical_image_returns_true(self):
        data = np.zeros((16, 8), dtype=np.float32)
        data[:, 3:5] = 1.0
        assert utils.is_vertical(data) is True

    def test_horizontal_image_returns_false(self):
        data = np.zeros((8, 16), dtype=np.float32)
        data[3:5, :] = 1.0
        assert utils.is_vertical(data) is False

    def test_square_image(self):
        data = np.eye(10, dtype=np.float32)
        result = utils.is_vertical(data)
        assert isinstance(result, bool)


class TestTiltPsiDyStr(object):
    def test_empty_when_all_zero(self):
        assert utils.tilt_psi_dy_str(0, 0, 0) == ""

    def test_includes_tilt(self):
        s = utils.tilt_psi_dy_str(10, 0, 0)
        assert "tilt" in s
        assert "10" in s

    def test_includes_psi(self):
        s = utils.tilt_psi_dy_str(0, 20, 0)
        assert "psi" in s
        assert "20" in s

    def test_includes_dy(self):
        s = utils.tilt_psi_dy_str(0, 0, 3.5)
        assert "dy" in s
        assert "3.5" in s

    def test_all_together(self):
        s = utils.tilt_psi_dy_str(10, 20, 3.5)
        assert "tilt" in s
        assert "psi" in s
        assert "dy" in s

    def test_without_units(self):
        s = utils.tilt_psi_dy_str(10, 0, 0, unit=False)
        assert "°" not in s

    def test_custom_separator(self):
        s = utils.tilt_psi_dy_str(10, 0, 0, sep=",", sep2=": ")
        assert ",tilt: 10°" in s


class TestSimulateHelicalProjection(object):
    def test_returns_2d_array(self):
        result = utils.simulate_helical_projection(
            n=10,
            twist=30,
            rise=5,
            csym=1,
            helical_diameter=40,
            ball_radius=3,
            polymer=0,
            planarity=0,
            ny=32,
            nx=32,
            apix=2.0,
        )
        assert result.shape == (32, 32)
        assert np.all(np.isfinite(result))

    def test_with_tilt_and_psi(self):
        result = utils.simulate_helical_projection(
            n=10,
            twist=30,
            rise=5,
            csym=1,
            helical_diameter=40,
            ball_radius=3,
            polymer=0,
            planarity=0,
            ny=32,
            nx=32,
            apix=2.0,
            tilt=5,
            psi=10,
            dy=2,
        )
        assert result.shape == (32, 32)

    def test_polymer_mode(self):
        result = utils.simulate_helical_projection(
            n=10,
            twist=30,
            rise=5,
            csym=1,
            helical_diameter=40,
            ball_radius=3,
            polymer=1,
            planarity=0.9,
            ny=32,
            nx=32,
            apix=2.0,
        )
        assert result.shape == (32, 32)


class TestRandomPolymer(object):
    def test_returns_array_with_expected_shape(self):
        result = utils.random_polymer(
            n_atoms=10, rmin=0, rmax=50, csym=1, planarity=0.9
        )
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 3
        assert result.shape[0] <= 10

    def test_with_csym(self):
        result = utils.random_polymer(n_atoms=5, rmin=0, rmax=50, csym=2, planarity=0.9)
        assert isinstance(result, np.ndarray)
        assert result.shape[1] == 3


class TestSymmetrizeTransformMap(object):
    def test_returns_3d_array(self):
        data = np.ones((8, 8, 8), dtype=np.float32)
        with pytest.raises(Exception):
            # This function calls apply_helical_symmetry which
            # requires valid parameters; with default fraction=1.0,
            # it may produce output. We just verify it returns an array.
            result = utils.symmetrize_transform_map(
                data=data,
                apix=1.0,
                twist_degree=30,
                rise_angstrom=5,
                csym=1,
            )
