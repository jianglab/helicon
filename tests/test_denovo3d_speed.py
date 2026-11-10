"""Tests for denovo3d Tier A no-Numba / single-CPU speedups and parity."""

import sys
import time

import numpy as np
import pytest

import helicon
from helicon.lib import transforms
from helicon.webApps.lib import denovo3d_pipeline as pipeline
from helicon.webApps.lib import denovo3d_solver as solver
from helicon.webApps.lib import solver_gauss_analytic as sga


class TestApplyHelicalSymmetryNpParityAndCulling:
    @pytest.mark.skipif(not transforms._HAS_NUMBA, reason="Numba required for parity")
    @pytest.mark.parametrize(
        "twist,rise,csym",
        [
            (1.2, 4.75, 1),
            (0.1, 4.75, 1),
            (-1.2, 4.75, 1),
            (1.2, -4.75, 1),
            (35.0, 4.75, 2),
        ],
    )
    def test_matches_numba_jit_across_signs_and_small_twist(self, twist, rise, csym):
        rng = np.random.default_rng(42)
        vol = rng.normal(size=(4, 16, 16)).astype(np.float32)
        args = (vol, 5.0, twist, rise, csym, 0.5, (96, 16, 16), 5.0)

        out_jit = transforms._apply_helical_symmetry_jit(*args)
        t0 = time.perf_counter()
        out_np = transforms._apply_helical_symmetry_np(*args)
        elapsed = time.perf_counter() - t0

        assert out_np.dtype == np.float32
        assert out_np.shape == out_jit.shape
        np.testing.assert_allclose(out_np, out_jit, atol=1e-5, rtol=1e-5)
        # At twist=0.1 deg, un-culled hi iterated ~3600 symmetry copies; analytical
        # Z-window culling keeps only ~4 copies per output slice.
        assert elapsed < 1.0


class TestCacheEmscriptenBypass:
    def test_emscripten_platform_uses_in_memory_dummy_backend(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(sys, "platform", "emscripten")
        calls = []

        @helicon.cache(cache_dir=str(tmp_path / "disk_cache"), verbose=0)
        def add_one(x):
            calls.append(x)
            return x + len(calls)

        assert add_one(10) == 11
        assert add_one(10) == 12
        assert not add_one.is_cached(10)
        assert not (tmp_path / "disk_cache").exists()
        add_one.clear_cache()
        assert calls == [10, 10]


class TestDenovo3DSolverSpeedAndParity:
    def test_helical_sym_matrix_linear_thin_slab_early_returns_none(self, monkeypatch):
        monkeypatch.setattr(solver, "_HAS_NUMBA", False)
        fn = solver.build_A_helical_sym_matrix.__wrapped__
        # Warm up lazy scipy.stats.qmc import once before timing
        fn(4, 8, 8, 1.2, 0.95, 1, 0, 3, 10, "linear", 0)
        t0 = time.perf_counter()
        A, b = fn(
            nz=4,
            ny=32,
            nx=32,
            twist_degree=1.2,
            rise_pixel=0.95,
            csym=1,
            rmin=0,
            rmax=15,
            min_sym_pairs=1000,
            interpolation="linear",
            verbose=0,
        )
        elapsed = time.perf_counter() - t0
        assert A is None and b is None
        assert elapsed < 0.05

    @pytest.mark.skipif(not solver._HAS_NUMBA, reason="Numba required for parity")
    @pytest.mark.parametrize("interpolation", ["linear", "nn"])
    @pytest.mark.parametrize(
        "tilt,psi,dy,csym,phi", [(0, 0, 0, 1, 0), (6, 9, 0.5, 2, 12)]
    )
    def test_data_matrix_z_sliced_matches_numba(
        self, monkeypatch, interpolation, tilt, psi, dy, csym, phi
    ):
        rng = np.random.default_rng(7)
        image = rng.random((16, 48), dtype=np.float32)
        kw = dict(
            image=image,
            scale2d_to_3d=1.0,
            twist_degree=2.5,
            rise_pixel=1.1,
            csym=csym,
            tilt_degree=tilt,
            psi_degree=psi,
            dy_pixel=dy,
            phi_degree=phi,
            reconstruct_diameter_2d_pixel=12,
            reconstruct_length_2d_pixel=40,
            reconstruct_diameter_3d_pixel=12,
            reconstruct_diameter_3d_inner_pixel=0,
            reconstruct_length_3d_pixel=6,
            min_projection_lines=200,
            interpolation=interpolation,
            verbose=0,
        )
        solver.build_A_data_matrix.clear_cache()
        A_jit, b_jit, pid_jit = solver.build_A_data_matrix(**kw)

        solver.build_A_data_matrix.clear_cache()
        monkeypatch.setattr(solver, "_HAS_NUMBA", False)
        A_np, b_np, pid_np = solver.build_A_data_matrix(**kw)

        assert A_jit.shape == A_np.shape
        assert np.array_equal(pid_jit, pid_np)
        np.testing.assert_allclose(b_jit, b_np, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            A_jit.toarray(), A_np.toarray(), atol=1e-5, rtol=1e-4
        )

    def test_np_kernels_use_intp_indices(self):
        shape = (3, 4, 4)
        Z, Y, X = np.meshgrid(
            np.linspace(0.2, 1.8, shape[0]),
            np.linspace(0.2, 2.8, shape[1]),
            np.linspace(0.2, 2.8, shape[2]),
            indexing="ij",
        )
        mask = np.ones(shape, dtype=bool)
        idx_mat = np.arange(np.prod(shape), dtype=np.intp).reshape(shape)
        pv = np.ones((shape[1], shape[0]), dtype=np.float32)
        for fn in (
            solver._build_A_data_matrix_loop_kji_linear_np,
            solver._build_A_data_matrix_loop_kji_nn_np,
        ):
            cr, cc, cd, bt, bpid = fn(Z, Y, X, mask, idx_mat, mask.size, pv)
            assert cr.dtype == np.intp
            assert cc.dtype == np.intp
            assert cd.dtype == np.float32
            assert bt.dtype == np.float32
            assert bpid.dtype == np.int32


class TestGaussAnalyticAndScoreOnly:
    def test_nn_elasticnet_cython_matches_python_and_preserves_inputs(self):
        rng = np.random.default_rng(11)
        A = rng.normal(size=(80, 24))
        b = rng.normal(size=80)
        M_orig = np.asfortranarray(A.T @ A)
        M_copy = M_orig.copy()
        u_orig = A.T @ b
        u_copy = u_orig.copy()

        lam1 = 0.05 * float(np.abs(u_orig).max())
        lam2 = 0.05 * float(np.trace(M_orig) / len(M_orig))

        w_fast_yy = sga.nn_elasticnet(M_orig, u_orig, lam1, lam2, yy=float(b @ b))
        w_fast_no_yy = sga.nn_elasticnet(M_orig, u_orig, lam1, lam2)
        w_py = sga._nn_elasticnet_py(M_orig, u_orig, lam1, lam2)

        assert np.array_equal(M_orig, M_copy)
        assert np.array_equal(u_orig, u_copy)
        np.testing.assert_allclose(w_fast_yy, w_py, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(w_fast_no_yy, w_py, atol=1e-5, rtol=1e-4)

    def test_render_volume_vectorized_matches_scalar_reference(self):
        rng = np.random.default_rng(19)
        centres = rng.uniform(-5.0, 5.0, size=(18, 3))
        centres[:, 2] = 0.0
        sigmas = rng.uniform(0.9, 1.5, size=18)
        amps = rng.uniform(-0.5, 2.0, size=18)

        nz, ny, nx = 6, 20, 20
        scale = 1.1
        twist_deg = 12.5
        rise_px = 0.95
        csym = 2
        cutoff_sigma = 3.5

        got = sga._render_volume(
            centres,
            sigmas,
            amps,
            scale,
            nz,
            ny,
            nx,
            twist_deg=twist_deg,
            rise_px=rise_px,
            csym=csym,
            cutoff_sigma=cutoff_sigma,
        )

        # Scalar reference matching the original triple-nested loop
        ref = np.zeros((nz, ny, nx), dtype=np.float32)
        c = centres * scale
        s = sigmas * scale
        rise = abs(rise_px) * scale
        half = max(1, int(np.ceil(cutoff_sigma * float(s.max()))))
        off = np.arange(-half, half + 1)
        k_max = int(np.ceil((nz / 2.0 + cutoff_sigma * float(s.max())) / rise)) + 1
        for k in range(-k_max, k_max + 1):
            for j in range(csym):
                ang = np.deg2rad(twist_deg) * k + 2 * np.pi * j / csym
                ca, sa = np.cos(ang), np.sin(ang)
                for (cx, cy, cz), sg, a in zip(c, s, amps):
                    if a <= 0:
                        continue
                    rx = cx * ca - cy * sa
                    ry = cx * sa + cy * ca
                    rz = cz + rise * k
                    zi = int(round(nz // 2 + rz))
                    if zi + half < 0 or zi - half > nz - 1:
                        continue
                    xi = int(round(nx // 2 + rx))
                    yi = int(round(ny // 2 + ry))
                    xs, ys, zs = xi + off, yi + off, zi + off
                    okx = (xs >= 0) & (xs < nx)
                    oky = (ys >= 0) & (ys < ny)
                    okz = (zs >= 0) & (zs < nz)
                    if not (okx.any() and oky.any() and okz.any()):
                        continue
                    vx = np.exp(-0.5 * ((xs - (nx // 2 + rx)) / sg) ** 2)
                    vy = np.exp(-0.5 * ((ys - (ny // 2 + ry)) / sg) ** 2)
                    vz = np.exp(-0.5 * ((zs - (nz // 2 + rz)) / sg) ** 2)
                    blob = (
                        a
                        * vz[okz, None, None]
                        * vy[None, oky, None]
                        * vx[None, None, okx]
                    )
                    ref[np.ix_(zs[okz], ys[oky], xs[okx])] += blob.astype(np.float32)

        assert got.dtype == np.float32
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5)

        empty = sga._render_volume(centres, sigmas, np.zeros_like(amps), scale, 4, 8, 8)
        assert empty.shape == (4, 8, 8)
        assert np.count_nonzero(empty) == 0

    def test_gauss_score_only_and_pipeline_process_one_task_contract(self):
        rng = np.random.default_rng(23)
        img = rng.random((24, 64), dtype=np.float32)

        (vol3d, s1, s2), score_full = sga.gauss_analytic_reconstruct(
            projection_image=img,
            scale2d_to_3d=1.0,
            twist_degree=1.2,
            rise_pixel=0.95,
            csym=1,
            target_apix2d=5.0,
            score_only=False,
        )
        (x_proj, s1_so, s2_so), score_so = sga.gauss_analytic_reconstruct(
            projection_image=img,
            scale2d_to_3d=1.0,
            twist_degree=1.2,
            rise_pixel=0.95,
            csym=1,
            target_apix2d=5.0,
            score_only=True,
        )
        assert vol3d.ndim == 3
        assert x_proj.shape == img.shape
        assert x_proj.dtype == np.float32
        assert s1 is None and s2 is None and s1_so is None and s2_so is None
        assert score_so == pytest.approx(score_full, abs=1e-12)

        common_args = dict(
            ti=0,
            ntasks=1,
            data=img,
            imageFile="synthetic.mrc",
            imageIndex=0,
            twist=1.2,
            rise=4.75,
            rise_range=(4.75, 4.75),
            csym=1,
            tilt=0.0,
            tilt_range=(0.0, 0.0),
            psi=0.0,
            psi_range=0.0,
            dy=0.0,
            dy_range=0.0,
            apix2d_orig=5.0,
            denoise="",
            low_pass=0.0,
            transpose=0,
            horizontalize=0,
            target_apix3d=5.0,
            target_apix2d=5.0,
            thresh_fraction=0.0,
            positive_constraint=True,
            tube_length=-1.0,
            tube_diameter=100.0,
            tube_diameter_inner=0.0,
            reconstruct_length=20.0,
            sym_oversample=1,
            interpolation="linear",
            fsc_test=False,
            score_metric="cosine",
            verbose=0,
            n_cpu=1,
        )
        score_p_so, rdata_so, params_so = pipeline.process_one_task(
            **common_args,
            return_3d=False,
            algorithm={"model": "gauss", "score_only": True},
        )
        score_p_full, rdata_full, params_full = pipeline.process_one_task(
            **common_args,
            return_3d=False,
            algorithm={"model": "gauss", "score_only": False},
        )
        assert score_p_so == pytest.approx(score_p_full, abs=1e-12)
        assert len(rdata_so) == 9
        assert rdata_so[0].ndim == 2
        assert rdata_so[1:] == (None,) * 8
        assert len(params_so) == len(params_full) == 11

        # return_3d=True must override score_only=True so top-1 / 3D export works
        _, rdata_3d, _ = pipeline.process_one_task(
            **common_args,
            return_3d=True,
            algorithm={"model": "gauss", "score_only": True},
        )
        assert rdata_3d[3] is not None
        assert rdata_3d[8] is not None
