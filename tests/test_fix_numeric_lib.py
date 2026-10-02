"""Regression tests for numeric fixes in the web-app compute libraries."""

from itertools import combinations, permutations
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import helicon
from helicon.webApps.lib import denovo3d_pipeline as pipeline
from helicon.webApps.lib import denovo3d_solver as solver
from helicon.webApps.lib import helical_pitch_compute
from helicon.webApps.lib import helical_projection_compute
from helicon.webApps.lib import hi3d_core
from helicon.webApps.lib import hill_compute


def _mock_mrc(data, apix=1.5):
    mrc = MagicMock()
    mrc.voxel_size.x = apix
    mrc.data = data
    mrc.__enter__.return_value = mrc
    return mrc


class TestMapInfoKeepsMapOrientation:
    def test_non_cubic_map_is_returned_unchanged(self, tmp_path):
        data = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)  # nx < ny
        fake = tmp_path / "map.mrc"
        fake.write_bytes(b"")
        with patch("mrcfile.open", return_value=_mock_mrc(data)):
            got, apix = helical_projection_compute.MapInfo(
                filename=str(fake)
            ).get_data()
        assert got.shape == (4, 6, 3)
        np.testing.assert_array_equal(got, data)
        assert apix == 1.5

    def test_2d_image_loader_still_transposes(self):
        data = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
        with patch("mrcfile.open", return_value=_mock_mrc(data)):
            got, _ = helical_projection_compute.get_images_from_file("x.mrcs")
        assert got.shape == (1, 3, 6)


class TestChangeMrcMapCrsOrder:
    @pytest.mark.parametrize("order", list(permutations([1, 2, 3])))
    def test_all_orders(self, order):
        rng = np.random.default_rng(0)
        ref = rng.random((3, 4, 5))  # physical (z, y, x)
        # The file array is (section, row, column); entry k of the order is the
        # physical axis on file axis 2 - k. Physical axis p is ref axis 3 - p.
        perm = [3 - order[2 - a] for a in range(3)]
        file_arr = np.transpose(ref, perm)
        got = hi3d_core.change_mrc_map_crs_order(file_arr, list(order), [1, 2, 3])
        assert got.shape == ref.shape
        np.testing.assert_array_equal(got, ref)


class TestHelicalSymMatrixLinear:
    def _build(self, interpolation):
        A, _ = solver.build_A_helical_sym_matrix(
            nz=16,
            ny=24,
            nx=24,
            twist_degree=-1.2,
            rise_pixel=1.6,
            csym=1,
            rmin=0,
            rmax=11,
            min_sym_pairs=100000,
            interpolation=interpolation,
        )
        return A.tocsr()

    def test_weights_sum_to_one_and_rows_comparable_to_nn(self):
        A_lin = self._build("linear")
        A_nn = self._build("nn")
        pos = A_lin.multiply(A_lin > 0).sum(axis=1).A1
        neg = A_lin.multiply(A_lin < 0).sum(axis=1).A1
        np.testing.assert_allclose(pos, 1.0, atol=1e-5)
        np.testing.assert_allclose(neg, -1.0, atol=1e-5)
        assert A_lin.min() >= -1.0 - 1e-5 and A_lin.max() <= 1.0 + 1e-5
        assert A_lin.shape[0] > 0.4 * A_nn.shape[0]


class TestHaltonPermutation:
    def test_is_permutation(self):
        for n in range(1, 60):
            p = solver.halton_permutation(n)
            assert sorted(p.tolist()) == list(range(n))
        assert solver.halton_permutation(0).size == 0

    def test_sym_pairs_used_once(self):
        ret = solver.sorted_hsym_csym_pairs(30.0, 2.0, 2, 8)
        pairs = [r[-1] for r in ret]
        hsym_max = max(1, int(np.ceil(8 / (2 * 2.0))))
        hcsyms = [(h, c) for h in range(-hsym_max, hsym_max + 1) for c in range(2)]
        assert len(pairs) == len(set(pairs)) == len(list(combinations(hcsyms, 2)))


class TestGenericLattice:
    def test_perfect_lattice_has_zero_error(self):
        a = np.array([30.0, 4.75])
        b = np.array([-10.0, 9.5])
        origin = np.array([0.0, 0.0])
        pts = np.array(
            [i * a + j * b + origin for i in range(-2, 3) for j in range(-2, 3)]
        )
        lat = hi3d_core.peaks_to_lattice(pts, a, b, origin)
        assert lat["err"] < 1e-9


class TestFitHelicalLattice:
    def _acf(self):
        return np.random.default_rng(1).random((64, 64))

    def test_three_peaks(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            peaks = rng.uniform(-50, 50, size=(3, 2))
            hi3d_core.fit_helical_lattice(peaks, self._acf())

    def test_reproducible(self):
        rng = np.random.default_rng(5)
        peaks = rng.uniform(-50, 50, size=(15, 2))
        r1 = hi3d_core.fit_helical_lattice(peaks, self._acf())
        r2 = hi3d_core.fit_helical_lattice(peaks, self._acf())
        assert r1 == r2


class TestRefineTwistRiseSymmetryOffset:
    def test_symmetry_copies_are_360_over_cn_degrees_apart(self, monkeypatch):
        da, cn = 2.0, 2
        acf = np.zeros((64, int(360 / da)))
        seen = []
        real = hi3d_core.map_coordinates

        def spy(img, coords, *args, **kwargs):
            seen.append(np.array(coords[1]))
            return real(img, coords, *args, **kwargs)

        monkeypatch.setattr(hi3d_core, "map_coordinates", spy)
        hi3d_core.refine_twist_rise(acf, da, 1.0, 10.0, 4.0, cn)
        px = seen[0]
        nx = acf.shape[1]
        offset = np.mod(px[1::cn] - px[0::cn], nx)
        np.testing.assert_allclose(offset, 360.0 / cn / da)


class TestBesselNImage:
    def test_tilted_uses_y_nyquist_and_sin_tilt(self):
        ny, nx, rx, ry, radius, tilt = 64, 32, 4.0, 6.0, 50.0, 20.0
        img = hill_compute.bessel_n_image(ny, nx, rx, ry, radius, tilt)
        table = hill_compute.bessel_1st_peak_positions()
        for k in (5, 10, 20, 30):
            s = 2 * np.pi * k / (ny // 2 * ry) * radius * np.sin(np.deg2rad(tilt))
            assert img[ny // 2 + k, nx // 2] == np.abs(table - s).argmin()

    def test_consistent_with_layer_lines(self):
        ny, nx, res, radius, tilt = 128, 128, 4.0, 40.0, 15.0
        img = hill_compute.bessel_n_image(ny, nx, res, res, radius, tilt)
        lls = hill_compute.compute_layer_line_positions(
            -1.2, 4.75, 1, radius, tilt, res
        )
        ds = 1.0 / (nx // 2 * res)
        n_checked = 0
        for d in lls.values():
            xs, ys, ns = d["LL"]
            for x, y, n in zip(xs, ys, ns):
                if 3 <= abs(n) <= 15 and abs(x) > 1e-3:
                    col = int(round(x / ds)) + nx // 2
                    row = int(round(y / ds)) + ny // 2
                    if 0 <= col < nx and 0 <= row < ny:
                        assert abs(int(img[row, col]) - abs(int(n))) <= 2
                        n_checked += 1
        assert n_checked > 0


class TestSelectHelicesByLength:
    def test_max_len_none(self):
        helices = [("a", [1, 2]), ("b", [3])]
        ret = helical_pitch_compute.select_helices_by_length(
            helices, [10.0, 100.0], 50.0, None
        )
        assert [gn for gn, _ in ret[0]] == ["b"]


class TestAutoCorrelationOddSize:
    @staticmethod
    def _reference(data, high_pass_fraction, sqrt):
        fft = np.fft.fft2(data)
        product = fft * np.conj(fft)
        if sqrt:
            product = np.sqrt(product)
        nz = data.shape[0]
        Z = np.fft.fftfreq(nz) * nz / (nz // 2)
        f2 = np.log(2) / high_pass_fraction**2
        product = product * (1.0 - np.exp(-f2 * Z**2))[:, None]
        return np.fft.fftshift(np.real(np.fft.ifft2(product)))

    @pytest.mark.parametrize("shape", [(15, 21), (16, 21), (15, 20)])
    def test_hill(self, shape):
        data = np.random.default_rng(2).random(shape)
        got = hill_compute.auto_correlation(data, sqrt=False, high_pass_fraction=0.3)
        ref = self._reference(data, 0.3, False)
        assert got.shape == shape
        np.testing.assert_allclose(got, ref / ref.max(), atol=1e-10)

    @pytest.mark.parametrize("shape", [(15, 21), (16, 21)])
    def test_hi3d(self, shape):
        data = np.random.default_rng(2).random(shape)
        got = hi3d_core.auto_correlation(data, high_pass_fraction=0.3)
        ref = self._reference(data, 0.3, False)
        ref -= np.median(ref, axis=1, keepdims=True)
        ref = hi3d_core.normalize(ref)
        assert got.shape == shape
        np.testing.assert_allclose(got, ref, atol=1e-10)


class TestTransformMapAndMinimalGrids:
    def test_shift_z_only(self):
        data = np.zeros((16, 16, 16), dtype=np.float32)
        data[8, 8, 8] = 1
        out = hi3d_core.transform_map(data, shift_z=2)
        assert not np.array_equal(out, data)

    def test_minimal_grids_short_z(self):
        m = np.random.default_rng(0).random((10, 20, 20))
        small, bin_factor = hi3d_core.minimal_grids(m)
        assert bin_factor == 1
        assert small.shape == (10, 20, 20)


def _task_params(**kw):
    params = dict(
        ti=0,
        ntasks=1,
        data=None,
        imageFile="test.mrc",
        imageIndex=1,
        twist=30,
        rise=10,
        rise_range=(5, 15),
        csym=1,
        tilt=0,
        tilt_range=(0, 0),
        psi=0,
        psi_range=0,
        dy=0,
        dy_range=0,
        apix2d_orig=1.0,
        denoise="",
        low_pass=0,
        transpose=0,
        horizontalize=0,
        target_apix3d=2.0,
        target_apix2d=1.0,
        thresh_fraction=-1,
        positive_constraint=-1,
        tube_length=-1,
        tube_diameter=40,
        tube_diameter_inner=0,
        reconstruct_length=20,
        sym_oversample=1,
        interpolation="nn",
        fsc_test=0,
        return_3d=False,
        score_metric="cosine",
        algorithm=dict(model="lsq"),
        verbose=0,
    )
    params.update(kw)
    return params


class TestProcessOneTaskInputs:
    def test_caller_array_not_thresholded(self):
        data = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        before = data.copy()
        result = pipeline.process_one_task(
            **_task_params(data=data, thresh_fraction=0.5)
        )
        np.testing.assert_array_equal(data, before)
        data_orig = result[2][0]
        np.testing.assert_array_equal(data_orig, before)


class TestRefinedParamsReturned:
    def test_lsq_reconstruct_returns_refined_params(self):
        image = np.random.default_rng(42).random((12, 12)).astype(np.float32)
        ret = solver.lsq_reconstruct(
            projection_image=image,
            scale2d_to_3d=1.0,
            twist_degree=30,
            rise_pixel=2,
            reconstruct_diameter_2d_pixel=8,
            reconstruct_length_2d_pixel=8,
            reconstruct_diameter_3d_pixel=8,
            reconstruct_length_3d_pixel=8,
            interpolation="nn",
            refine_tilt_psi_dy_range={"psi": 2.0, "max_iter": 1},
            return_refined_params=True,
        )
        assert len(ret) == 3
        assert isinstance(ret[2], dict)
        assert not hasattr(solver.lsq_reconstruct, "_refined_params")

    def test_pipeline_uses_returned_params(self, monkeypatch):
        def stub(**kwargs):
            nz = int(kwargs["reconstruct_length_3d_pixel"])
            d = int(kwargs["reconstruct_diameter_3d_pixel"])
            vol = np.zeros((nz, d, d), dtype=np.float32)
            vol[nz // 2, d // 2, d // 2] = 1.0
            assert kwargs["return_refined_params"]
            return (vol, None, None), 0.5, {"tilt": 3.0, "psi": 4.0, "dy": 0.0}

        seen = {}
        real = helicon.transform_map

        def spy(m, **kwargs):
            if "psi" in kwargs:
                seen.update(kwargs)
            return real(m, **kwargs)

        monkeypatch.setattr(pipeline, "lsq_reconstruct", stub)
        monkeypatch.setattr(helicon, "transform_map", spy)
        data = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        pipeline.process_one_task(**_task_params(data=data, psi_range=5))
        assert seen["tilt"] == 3.0 and seen["psi"] == 4.0


class TestRefineTiltPsiDyRowCountChange:
    def test_resolve_with_fewer_rows(self, monkeypatch):
        image = np.random.default_rng(42).random((12, 12)).astype(np.float32)
        real = solver.build_A_data_matrix
        calls = {"n": 0}

        def fewer_rows(**kwargs):
            A, b, pid = real(**kwargs)
            calls["n"] += 1
            if calls["n"] > 1:  # every geometry after the first loses rows
                return A[:-5], b[:-5], pid[:-5]
            return A, b, pid

        monkeypatch.setattr(solver, "build_A_data_matrix", fewer_rows)
        n_x = np.count_nonzero(
            helicon.get_cylindrical_mask(nz=8, ny=8, nx=8, rmin=0, rmax=3)
        )
        t, p, d, x, score = solver.refine_tilt_psi_dy(
            projection_image=image,
            scale2d_to_3d=1.0,
            twist_degree=30,
            rise_pixel=2,
            csym=1,
            reconstruct_diameter_2d_pixel=8,
            reconstruct_length_2d_pixel=8,
            reconstruct_diameter_3d_pixel=8,
            reconstruct_diameter_3d_inner_pixel=0,
            reconstruct_length_3d_pixel=8,
            sym_oversample=1,
            interpolation="nn",
            x_init=np.zeros(n_x, dtype=np.float32),
            max_iter=2,
            tol_tilt=-1,
            tol_psi=-1,
            tol_dy=-1,
        )
        assert np.isfinite(score)
