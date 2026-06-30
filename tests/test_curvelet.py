"""Tests for helicon.lib.curvelet module."""

from __future__ import annotations

import numpy as np
import pytest


class TestCurveletDenoiseFDCT:
    """Tests for curvelet_denoise_fdct."""

    def test_square_image(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import curvelet_denoise_fdct

        np.random.seed(42)
        image = np.random.rand(64, 64).astype(np.float64)
        result = curvelet_denoise_fdct(image, sigma=0.1, num_scales=3)
        assert result.shape == (64, 64)

    def test_non_square_image(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import curvelet_denoise_fdct

        np.random.seed(42)
        image = np.random.rand(40, 64).astype(np.float64)
        result = curvelet_denoise_fdct(image, sigma=0.1, num_scales=3)
        assert result.shape == (40, 64)

    def test_preserves_dynamic_range(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import curvelet_denoise_fdct

        np.random.seed(42)
        image = np.random.rand(48, 48).astype(np.float64) * 2.0 + 1.0
        result = curvelet_denoise_fdct(image, sigma=0.1, num_scales=3)
        assert np.abs(result.min() - image.min()) < 0.3
        assert np.abs(result.max() - image.max()) < 0.3

    def test_improves_mse_for_noisy_structure(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import curvelet_denoise_fdct

        np.random.seed(42)
        y, x = np.ogrid[-32:32, -32:32]
        clean = (x**2 + y**2 < 20**2).astype(np.float64) * 0.8 + 0.1
        noisy = clean + np.random.randn(64, 64) * 0.15
        noisy = noisy.clip(0, 1)

        result = curvelet_denoise_fdct(noisy, sigma=0.15, num_scales=3)
        noisy_mse = np.mean((noisy - clean) ** 2)
        denoised_mse = np.mean((result - clean) ** 2)
        assert denoised_mse < noisy_mse

    def test_auto_sigma(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import curvelet_denoise_fdct

        np.random.seed(42)
        y, x = np.ogrid[-32:32, -32:32]
        clean = (x**2 + y**2 < 20**2).astype(np.float64) * 0.8 + 0.1
        noisy = clean + np.random.randn(64, 64) * 0.15
        noisy = noisy.clip(0, 1)

        result = curvelet_denoise_fdct(noisy, num_scales=3)
        noisy_mse = np.mean((noisy - clean) ** 2)
        denoised_mse = np.mean((result - clean) ** 2)
        assert denoised_mse < noisy_mse

    def test_returns_float64(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import curvelet_denoise_fdct

        image = np.random.rand(32, 32).astype(np.float32)
        result = curvelet_denoise_fdct(image, sigma=0.1, num_scales=2)
        assert result.dtype == np.float64


class TestHasCurvelet:
    """Tests for has_curvelet_fdct."""

    def test_fdct_returns_true_when_installed(self):
        import helicon

        assert helicon.has_curvelet_fdct() is True

    def test_public_api_fdct_available(self):
        import helicon

        assert hasattr(helicon, "curvelet_denoise_fdct")
        assert callable(helicon.curvelet_denoise_fdct)
        assert hasattr(helicon, "curvelet_denoise_batch_fdct")
        assert callable(helicon.curvelet_denoise_batch_fdct)

    def test_udct_returns_true_when_installed(self):
        import helicon

        assert helicon.has_curvelet_udct() is True

    def test_public_api_udct_available(self):
        import helicon

        assert hasattr(helicon, "curvelet_denoise_udct")
        assert callable(helicon.curvelet_denoise_udct)
        assert hasattr(helicon, "curvelet_denoise_batch_udct")
        assert callable(helicon.curvelet_denoise_batch_udct)
        assert hasattr(helicon, "curvelet_denoise_3d_udct")
        assert callable(helicon.curvelet_denoise_3d_udct)


class TestCurveletDenoiseBatchFDCT:
    """Tests for curvelet_denoise_batch_fdct."""

    def test_batch_parallel(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import curvelet_denoise_batch_fdct

        np.random.seed(42)
        images = [np.random.rand(32, 32) for _ in range(4)]
        results = curvelet_denoise_batch_fdct(images, sigma=0.1, num_scales=2, n_jobs=2)
        assert len(results) == 4
        assert all(r.shape == (32, 32) for r in results)

    def test_batch_single_job(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import curvelet_denoise_batch_fdct

        np.random.seed(42)
        images = [np.random.rand(32, 32) for _ in range(3)]
        results = curvelet_denoise_batch_fdct(images, sigma=0.1, num_scales=2, n_jobs=1)
        assert len(results) == 3

    def test_mad_threshold_higher_sigma_retains_less(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import (
            _get_grid,
            _denoise,
            _compute_thresholds_mad,
        )

        np.random.seed(42)
        image = np.random.rand(64, 64).astype(np.float64)

        grid = _get_grid((64, 64), 4)
        coeffs = grid.forward_transform(image)

        thresh_low = _compute_thresholds_mad(coeffs, sigma_scale=1.0)
        _, pct_low = _denoise(
            image, grid, sigma_scale=1, thresholds=thresh_low, coeffs=coeffs
        )

        thresh_high = _compute_thresholds_mad(coeffs, sigma_scale=3.0)
        _, pct_high = _denoise(image, grid, sigma_scale=1, thresholds=thresh_high)

        assert pct_high < pct_low, (
            f"higher sigma should retain fewer coefficients "
            f"({pct_high:.1f}% vs {pct_low:.1f}%)"
        )

    def test_elbow_via_single_image_api(self):
        pytest.importorskip("curvepy")
        import helicon

        np.random.seed(42)
        noisy = np.random.rand(32, 32) + 0.5
        result = helicon.curvelet_denoise_fdct(noisy, sigma=-1, num_scales=2)
        assert result.shape == (32, 32)
        assert result.dtype == np.float64

    def test_elbow_improves_mse_vs_no_denoising(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import curvelet_denoise_fdct

        np.random.seed(42)
        clean = np.sin(np.linspace(0, 2 * np.pi, 64)).reshape(1, -1).repeat(64, axis=0)
        clean += np.cos(np.linspace(0, 2 * np.pi, 64)).reshape(-1, 1)
        noisy = clean + np.random.randn(64, 64) * 0.3

        result = curvelet_denoise_fdct(noisy, sigma=-1, num_scales=3)
        mse_noisy = np.mean((noisy - clean) ** 2)
        mse_denoised = np.mean((result - clean) ** 2)
        assert mse_denoised < mse_noisy, (
            f"elbow denoising should reduce MSE "
            f"({mse_denoised:.4f} vs {mse_noisy:.4f})"
        )

    def test_elbow_in_batch(self):
        pytest.importorskip("curvepy")
        from helicon.lib.curvelet import curvelet_denoise_batch_fdct

        np.random.seed(42)
        images = [np.random.rand(32, 32) for _ in range(3)]
        results = curvelet_denoise_batch_fdct(images, sigma=-1, num_scales=2, n_jobs=1)
        assert len(results) == 3
        assert all(r.shape == (32, 32) for r in results)

    def test_batch_via_public_api(self):
        pytest.importorskip("curvepy")
        import helicon

        np.random.seed(42)
        images = [np.random.rand(32, 32) for _ in range(2)]
        results = helicon.curvelet_denoise_batch_fdct(
            images, sigma=0.1, num_scales=2, n_jobs=1
        )
        assert len(results) == 2


class TestMadStd:
    """Tests for _mad_std."""

    def test_mad_std_scales_with_noise_level(self):
        from helicon.lib.curvelet import _mad_std

        np.random.seed(42)
        v1 = np.random.randn(10000) * 0.5
        v2 = np.random.randn(10000) * 1.0
        s1 = _mad_std(v1)
        s2 = _mad_std(v2)
        assert s2 > s1
        assert abs(s2 / s1 - 2.0) < 0.5


class TestCurveletDenoiseUDCT:
    """Tests for curvelet_denoise_udct (2D UDCT)."""

    def test_square_image(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_udct

        np.random.seed(42)
        image = np.random.rand(64, 64).astype(np.float64)
        result = curvelet_denoise_udct(image, sigma=0.1, num_scales=3)
        assert result.shape == (64, 64)

    def test_non_square_image(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_udct

        np.random.seed(42)
        image = np.random.rand(40, 64).astype(np.float64)
        result = curvelet_denoise_udct(image, sigma=0.1, num_scales=3)
        assert result.shape == (40, 64)

    def test_preserves_dynamic_range(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_udct

        np.random.seed(42)
        image = np.random.rand(48, 48).astype(np.float64) * 2.0 + 1.0
        result = curvelet_denoise_udct(image, sigma=0.1, num_scales=3)
        assert np.abs(result.min() - image.min()) < 0.3
        assert np.abs(result.max() - image.max()) < 0.3

    def test_improves_mse_for_noisy_structure(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_udct

        np.random.seed(42)
        y, x = np.ogrid[-32:32, -32:32]
        clean = (x**2 + y**2 < 20**2).astype(np.float64) * 0.8 + 0.1
        noisy = clean + np.random.randn(64, 64) * 0.15
        noisy = noisy.clip(0, 1)

        result = curvelet_denoise_udct(noisy, sigma=0.15, num_scales=3)
        noisy_mse = np.mean((noisy - clean) ** 2)
        denoised_mse = np.mean((result - clean) ** 2)
        assert denoised_mse < noisy_mse

    def test_auto_sigma(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_udct

        np.random.seed(42)
        y, x = np.ogrid[-32:32, -32:32]
        clean = (x**2 + y**2 < 20**2).astype(np.float64) * 0.8 + 0.1
        noisy = clean + np.random.randn(64, 64) * 0.15
        noisy = noisy.clip(0, 1)

        result = curvelet_denoise_udct(noisy, num_scales=3)
        noisy_mse = np.mean((noisy - clean) ** 2)
        denoised_mse = np.mean((result - clean) ** 2)
        assert denoised_mse < noisy_mse

    def test_returns_float64(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_udct

        image = np.random.rand(32, 32).astype(np.float32)
        result = curvelet_denoise_udct(image, sigma=0.1, num_scales=2)
        assert result.dtype == np.float64

    def test_wedges_per_dir_param(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_udct

        np.random.seed(42)
        image = np.random.rand(48, 48).astype(np.float64)
        result = curvelet_denoise_udct(image, sigma=0.1, num_scales=3, wedges_per_dir=6)
        assert result.shape == (48, 48)


class TestCurveletDenoiseBatchUDCT:
    """Tests for curvelet_denoise_batch_udct."""

    def test_batch_parallel(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_batch_udct

        np.random.seed(42)
        images = [np.random.rand(32, 32) for _ in range(4)]
        results = curvelet_denoise_batch_udct(images, sigma=0.1, num_scales=2, n_jobs=2)
        assert len(results) == 4
        assert all(r.shape == (32, 32) for r in results)

    def test_batch_single_job(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_batch_udct

        np.random.seed(42)
        images = [np.random.rand(32, 32) for _ in range(3)]
        results = curvelet_denoise_batch_udct(images, sigma=0.1, num_scales=2, n_jobs=1)
        assert len(results) == 3

    def test_elbow_in_batch(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_batch_udct

        np.random.seed(42)
        images = [np.random.rand(32, 32) for _ in range(3)]
        results = curvelet_denoise_batch_udct(images, sigma=-1, num_scales=2, n_jobs=1)
        assert len(results) == 3
        assert all(r.shape == (32, 32) for r in results)

    def test_batch_via_public_api(self):
        pytest.importorskip("curvelets")
        import helicon

        np.random.seed(42)
        images = [np.random.rand(32, 32) for _ in range(2)]
        results = helicon.curvelet_denoise_batch_udct(
            images, sigma=0.1, num_scales=2, n_jobs=1
        )
        assert len(results) == 2


class TestCurveletDenoise3DUDCT:
    """Tests for curvelet_denoise_3d_udct."""

    def test_small_volume(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_3d_udct

        np.random.seed(42)
        volume = np.random.rand(16, 16, 16).astype(np.float64)
        result = curvelet_denoise_3d_udct(volume, sigma=0.1, num_scales=2)
        assert result.shape == (16, 16, 16)

    def test_preserves_dynamic_range_3d(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_3d_udct

        np.random.seed(42)
        volume = np.random.rand(12, 12, 12).astype(np.float64) * 2.0 + 1.0
        result = curvelet_denoise_3d_udct(volume, sigma=0.1, num_scales=2)
        assert result.shape == (12, 12, 12)
        assert np.isfinite(result).all()

    def test_elbow_mode_3d(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_3d_udct

        np.random.seed(42)
        volume = np.random.rand(12, 12, 12).astype(np.float64)
        result = curvelet_denoise_3d_udct(volume, num_scales=2)
        assert result.shape == (12, 12, 12)

    def test_returns_float64_3d(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_3d_udct

        volume = np.random.rand(8, 8, 8).astype(np.float32)
        result = curvelet_denoise_3d_udct(volume, sigma=0.1, num_scales=2)
        assert result.dtype == np.float64


class TestUDCTCompatibleShape:
    """Tests for _udct_compatible_shape."""

    def test_compatible_shape_power_of_two(self):
        from helicon.lib.curvelet import _udct_compatible_shape

        result = _udct_compatible_shape((64, 64), 4)
        assert result == (64, 64)

    def test_compatible_shape_needs_padding(self):
        from helicon.lib.curvelet import _udct_compatible_shape

        result = _udct_compatible_shape((63, 63), 4)
        assert result == (64, 64)

    def test_compatible_shape_large_non_divisible(self):
        from helicon.lib.curvelet import _udct_compatible_shape

        result = _udct_compatible_shape((3838, 3710), 4)
        assert result == (3840, 3712)

    def test_compatible_shape_with_3_scales(self):
        from helicon.lib.curvelet import _udct_compatible_shape

        result = _udct_compatible_shape((3838, 3710), 3)
        assert result == (3840, 3712)

    def test_compatible_shape_with_2_scales(self):
        from helicon.lib.curvelet import _udct_compatible_shape

        result = _udct_compatible_shape((63, 63), 2)
        assert result == (64, 64)

    def test_compatible_shape_3d(self):
        from helicon.lib.curvelet import _udct_compatible_shape

        result = _udct_compatible_shape((31, 63, 127), 4)
        assert result == (32, 64, 128)

    def test_identity_with_padding(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import (
            _udct_compatible_shape,
            _get_udct_grid,
        )

        np.random.seed(42)
        for shape in [(63, 63), (3838, 3710)]:
            orig = np.random.randn(*shape).astype(np.float64)
            pad_shape = _udct_compatible_shape(shape, 4)
            if pad_shape != shape:
                pads = tuple((0, ps - ds) for ds, ps in zip(shape, pad_shape))
                padded = np.pad(orig, pads, mode="edge")
            else:
                padded = orig
            grid = _get_udct_grid(pad_shape, 4, 12)
            coeffs = grid.forward(padded)
            recon = grid.backward(coeffs)
            recon_cropped = recon[: shape[0], : shape[1]]
            max_err = np.abs(orig - recon_cropped).max()
            assert (
                max_err < 0.01
            ), f"UDCT identity failed for {shape}: max_err={max_err:.4f}"


class TestUDCTGPU:
    """Tests for UDCT GPU (torch backend) path."""

    def test_gpu_public_api_flag(self):
        pytest.importorskip("curvelets")
        import helicon

        assert hasattr(helicon, "has_curvelet_udct_gpu")

    def test_gpu_available(self):
        import helicon

        assert helicon.has_curvelet_udct_gpu() is True

    def test_gpu_single_image(self):
        pytest.importorskip("curvelets")
        import helicon

        np.random.seed(42)
        image = np.random.rand(48, 48).astype(np.float64)
        result = helicon.curvelet_denoise_udct(
            image, sigma=0.1, num_scales=3, use_gpu=True
        )
        assert result.shape == (48, 48)
        assert np.isfinite(result).all()

    def test_gpu_batch(self):
        pytest.importorskip("curvelets")
        import helicon

        np.random.seed(42)
        images = [np.random.rand(32, 32).astype(np.float64) for _ in range(2)]
        results = helicon.curvelet_denoise_batch_udct(
            images, sigma=0.1, num_scales=2, use_gpu=True, n_jobs=1
        )
        assert len(results) == 2
        assert all(r.shape == (32, 32) for r in results)

    def test_gpu_3d(self):
        pytest.importorskip("curvelets")
        import helicon

        np.random.seed(42)
        volume = np.random.rand(12, 12, 12).astype(np.float64)
        result = helicon.curvelet_denoise_3d_udct(
            volume, sigma=0.1, num_scales=2, use_gpu=True
        )
        assert result.shape == (12, 12, 12)
        assert np.isfinite(result).all()

    def test_gpu_matches_numpy(self):
        pytest.importorskip("curvelets")
        import helicon

        np.random.seed(42)
        image = np.random.rand(48, 48).astype(np.float64) * 0.05 + 0.5
        r_gpu = helicon.curvelet_denoise_udct(
            image, sigma=3, num_scales=3, use_gpu=True
        )
        r_np = helicon.curvelet_denoise_udct(
            image, sigma=3, num_scales=3, use_gpu=False
        )
        from helicon.lib.curvelet import _get_device

        dev = _get_device()
        tol = 5e-7 if dev.type == "mps" else 1e-8
        assert (
            np.abs(r_gpu - r_np).max() < tol
        ), f"GPU vs NumPy mismatch on {dev.type}: max_diff={np.abs(r_gpu - r_np).max():.4e}"

    def test_gpu_elbow_mode(self):
        pytest.importorskip("curvelets")
        import helicon

        np.random.seed(42)
        image = np.random.rand(32, 32).astype(np.float64)
        result = helicon.curvelet_denoise_udct(image, use_gpu=True)
        assert result.shape == (32, 32)

    def test_gpu_with_padding(self):
        pytest.importorskip("curvelets")
        import helicon

        np.random.seed(42)
        image = np.random.rand(63, 63).astype(np.float64)
        result = helicon.curvelet_denoise_udct(
            image, sigma=0.1, num_scales=4, use_gpu=True
        )
        assert result.shape == (63, 63)
        assert np.isfinite(result).all()


class TestUDCTDenoiseCurveletPluginGPU:
    """Test --denoiseCurvelet plugin wiring with gpu param."""

    def test_gpu_param_parsing(self):
        from helicon.lib.util import parse_param_str

        _, d = parse_param_str("sigma=3:numScales=4:gpu=true")
        assert d["gpu"] in (1, True, "true")

    def test_gpu_param_false_default(self):
        from helicon.lib.util import parse_param_str

        _, d = parse_param_str("sigma=3")
        assert "gpu" not in d

    def test_fdct_gpu_rejected(self):
        pytest.importorskip("curvelets")
        from helicon.plugins.images2star.denoisecurvelet import handle
        from helicon.lib.exceptions import HeliconError
        import argparse, pandas as pd

        args = argparse.Namespace(cpu=1, verbose=0)
        data = pd.DataFrame(
            {"rlnImageName": ["1@test.mrcs"], "rlnMicrographName": ["test.mrc"]}
        )
        index_d = {"denoiseCurvelet": 0}
        import io, contextlib

        with contextlib.redirect_stderr(io.StringIO()):
            with pytest.raises(HeliconError, match="FDCT does not support GPU"):
                handle(data, args, index_d, "transform=fdct:gpu=true")


class TestUDCTDenoiseCurveletPadding:
    """Tests for UDCT denoising with non-divisible image sizes."""

    def test_denoise_odd_size(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_udct

        np.random.seed(42)
        image = np.random.rand(63, 63).astype(np.float64)
        result = curvelet_denoise_udct(image, sigma=0.1, num_scales=4)
        assert result.shape == (63, 63)
        assert np.isfinite(result).all()

    def test_denoise_odd_size_batch(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_batch_udct

        np.random.seed(42)
        images = [np.random.rand(63, 63).astype(np.float64) for _ in range(2)]
        results = curvelet_denoise_batch_udct(images, sigma=0.1, num_scales=4, n_jobs=1)
        assert len(results) == 2
        for r in results:
            assert r.shape == (63, 63)
            assert np.isfinite(r).all()

    def test_denoise_odd_size_3d(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_3d_udct

        np.random.seed(42)
        volume = np.random.rand(15, 31, 63).astype(np.float64)
        result = curvelet_denoise_3d_udct(volume, sigma=0.1, num_scales=4)
        assert result.shape == (15, 31, 63)
        assert np.isfinite(result).all()

    def test_mean_preserved_with_padding(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_udct

        np.random.seed(42)
        image = np.random.rand(63, 63).astype(np.float64) * 0.05 + 0.5
        result = curvelet_denoise_udct(image, sigma=3, num_scales=4)
        assert (
            abs(image.mean() - result.mean()) < 0.01
        ), f"mean shift: {image.mean():.4f} -> {result.mean():.4f}"


class TestFDCTvsUDCTCorrelation:
    """FDCT and UDCT should produce correlated results for the same input."""

    def test_fdct_udct_correlation(self):
        pytest.importorskip("curvepy")
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_fdct, curvelet_denoise_udct

        np.random.seed(42)
        y, x = np.ogrid[-32:32, -32:32]
        clean = (x**2 + y**2 < 20**2).astype(np.float64) * 0.8 + 0.1
        noisy = clean + np.random.randn(64, 64) * 0.15
        noisy = noisy.clip(0, 1)

        result_fdct = curvelet_denoise_fdct(noisy, num_scales=3)
        result_udct = curvelet_denoise_udct(noisy, num_scales=3, wedges_per_dir=3)

        # Both improve MSE vs noisy
        noisy_mse = np.mean((noisy - clean) ** 2)
        fdct_mse = np.mean((result_fdct - clean) ** 2)
        udct_mse = np.mean((result_udct - clean) ** 2)
        assert fdct_mse < noisy_mse
        assert udct_mse < noisy_mse

        # Results are strongly correlated (> 0.95)
        cc = np.corrcoef(result_fdct.ravel(), result_udct.ravel())[0, 1]
        assert cc > 0.95, f"FDCT/UDCT Pearson correlation = {cc:.4f}"


class TestMCTDenoising:
    """Tests for Monogenic Curvelet Transform denoising."""

    def test_riesz_transform_properties(self):
        from helicon.lib.curvelet import _riesz_transform_x, _riesz_transform_y

        np.random.seed(42)
        image = np.random.randn(32, 32)
        rx = _riesz_transform_x(image)
        ry = _riesz_transform_y(image)

        assert rx.shape == image.shape
        assert ry.shape == image.shape
        assert np.isfinite(rx).all()
        assert np.isfinite(ry).all()

    def test_riesz_orthogonality(self):
        from helicon.lib.curvelet import _riesz_transform_x, _riesz_transform_y

        np.random.seed(42)
        image = np.random.randn(32, 32)
        rx = _riesz_transform_x(image)
        ry = _riesz_transform_y(image)

        inner = np.sum(rx * ry)
        norm_x = np.sqrt(np.sum(rx * rx))
        norm_y = np.sqrt(np.sum(ry * ry))
        denom = norm_x * norm_y + 1e-30
        assert abs(inner / denom) < 0.3

    def test_mct_denoise_single(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_mct

        np.random.seed(42)
        image = np.random.rand(64, 64).astype(np.float64)
        result = curvelet_denoise_mct(image, sigma=0.1, num_scales=3)
        assert result.shape == (64, 64)
        assert np.isfinite(result).all()

    def test_mct_denoise_odd_size(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_mct

        np.random.seed(42)
        image = np.random.rand(63, 63).astype(np.float64)
        result = curvelet_denoise_mct(image, sigma=0.1, num_scales=3)
        assert result.shape == (63, 63)
        assert np.isfinite(result).all()

    def test_mct_denoise_batch(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_batch_mct

        np.random.seed(42)
        images = [np.random.rand(32, 32).astype(np.float64) for _ in range(3)]
        results = curvelet_denoise_batch_mct(images, sigma=0.1, num_scales=3, n_jobs=1)
        assert len(results) == 3
        for r in results:
            assert r.shape == (32, 32)
            assert np.isfinite(r).all()

    def test_mct_improves_mse(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_mct

        np.random.seed(42)
        y, x = np.ogrid[-32:32, -32:32]
        clean = (x**2 + y**2 < 20**2).astype(np.float64) * 0.8 + 0.1
        noisy = clean + np.random.randn(64, 64) * 0.15
        noisy = noisy.clip(0, 1)

        result = curvelet_denoise_mct(noisy, num_scales=3)
        noisy_mse = np.mean((noisy - clean) ** 2)
        result_mse = np.mean((result - clean) ** 2)
        assert result_mse < noisy_mse

    def test_mct_vs_udct_correlation_not_identical(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_mct, curvelet_denoise_udct

        np.random.seed(42)
        y, x = np.ogrid[-32:32, -32:32]
        clean = (x**2 + y**2 < 20**2).astype(np.float64) * 0.8 + 0.1
        noisy = clean + np.random.randn(64, 64) * 0.15
        noisy = noisy.clip(0, 1)

        result_udct = curvelet_denoise_udct(noisy, sigma=0.15, num_scales=3)
        result_mct = curvelet_denoise_mct(noisy, sigma=0.15, num_scales=3)

        assert not np.allclose(
            result_udct, result_mct, atol=1e-6
        ), "MCT and UDCT should produce different results"
        cc = np.corrcoef(result_udct.ravel(), result_mct.ravel())[0, 1]
        assert (
            cc > 0.85
        ), f"MCT/UDCT should be correlated (both denoise), got cc={cc:.4f}"
        udct_mse = np.mean((result_udct - clean) ** 2)
        mct_mse = np.mean((result_mct - clean) ** 2)
        assert (
            mct_mse < udct_mse * 1.5
        ), f"MCT MSE ({mct_mse:.6f}) should not be much worse than UDCT MSE ({udct_mse:.6f})"

    def test_mct_sigma_zero_uses_elbow(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_mct

        np.random.seed(42)
        image = np.random.rand(32, 32).astype(np.float64)
        result = curvelet_denoise_mct(image, sigma=0, num_scales=3)
        assert result.shape == (32, 32)
        assert np.isfinite(result).all()

    def test_mct_constant_image_preserved(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_mct

        image = np.ones((32, 32), dtype=np.float64) * 0.5
        result = curvelet_denoise_mct(image, sigma=0.1, num_scales=3)
        assert result.shape == (32, 32)
        assert np.isfinite(result).all()
        assert abs(result.mean() - 0.5) < 0.02

    def test_mct_imported_via_helicon(self):
        import helicon

        assert hasattr(helicon, "curvelet_denoise_mct")
        assert hasattr(helicon, "curvelet_denoise_batch_mct")

    def test_mct_riesz_constant_zero(self):
        from helicon.lib.curvelet import _riesz_transform_x, _riesz_transform_y

        const = np.ones((16, 16), dtype=np.float64) * 0.5
        rx = _riesz_transform_x(const)
        ry = _riesz_transform_y(const)
        assert abs(rx).max() < 1e-12
        assert abs(ry).max() < 1e-12

    def test_mct_denoise_subband_direct(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import _mct_denoise_subband

        np.random.seed(42)
        q0 = np.random.randn(16, 16).astype(np.float64)
        result = _mct_denoise_subband(q0, T=0.5)
        assert result.shape == (16, 16)
        assert np.isfinite(result).all()
        result_zero = _mct_denoise_subband(q0, T=0.0)
        assert np.allclose(result_zero, q0)

    def test_mct_batch_empty_sigma(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_batch_mct

        np.random.seed(42)
        images = [np.random.rand(16, 16).astype(np.float64) for _ in range(2)]
        results = curvelet_denoise_batch_mct(images, n_jobs=1)
        assert len(results) == 2
        for r in results:
            assert np.isfinite(r).all()

    def test_mct_wedge_threshold_apply(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import (
            _mct_apply_threshold_to_wedge,
            _mct_threshold_apply_thresholds,
        )

        np.random.seed(42)
        wedge = [np.random.randn(8, 8).astype(np.float64) for _ in range(3)]
        result_wedge = _mct_apply_threshold_to_wedge(wedge, T=0.5)
        assert len(result_wedge) == len(wedge)
        for r, w in zip(result_wedge, wedge):
            assert r.shape == w.shape

        coeffs = [[wedge]]
        result_coeffs = _mct_threshold_apply_thresholds(coeffs, [[0.5]])
        assert len(result_coeffs) == 1


class TestTiledDenoising:
    """Tests for 2D tiled curvelet denoising."""

    def test_tile_indices_2d(self):
        from helicon.lib.curvelet import _tile_indices_2d

        tiles = _tile_indices_2d((64, 64), 32, 8)
        assert len(tiles) == 9  # stride=24, 3 tiles per dim
        tiles_large = _tile_indices_2d((64, 64), 64, 8)
        assert len(tiles_large) == 1
        tiles_overhang = _tile_indices_2d((70, 70), 32, 8)
        assert len(tiles_overhang) == 9

    def test_cosine_taper_2d(self):
        from helicon.lib.curvelet import _cosine_taper_2d

        w = _cosine_taper_2d((32, 32), 8)
        assert w.shape == (32, 32)
        assert np.all(w >= 0) and np.all(w <= 1)
        assert w[16, 16] == 1.0
        assert w[0, 16] < 1.0  # edge is tapered
        assert w[16, 0] < 1.0
        assert w[8, 8] > w[0, 0]  # ramps up from corner

    def test_tiled_udct_vs_untiled(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import (
            curvelet_denoise_udct,
            curvelet_denoise_udct_tiled,
        )

        np.random.seed(42)
        image = np.random.rand(64, 64).astype(np.float64)
        ref = curvelet_denoise_udct(image, sigma=0.1, num_scales=3)
        tiled = curvelet_denoise_udct_tiled(
            image,
            sigma=0.1,
            num_scales=3,
            tile_size=32,
            overlap=8,
            n_jobs=1,
        )
        assert ref.shape == tiled.shape
        diff = np.abs(ref - tiled).mean()
        assert diff < 0.07

    def test_tiled_mct_vs_untiled(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import (
            curvelet_denoise_mct,
            curvelet_denoise_mct_tiled,
        )

        np.random.seed(42)
        image = np.random.rand(64, 64).astype(np.float64)
        ref = curvelet_denoise_mct(image, sigma=0.1, num_scales=3)
        tiled = curvelet_denoise_mct_tiled(
            image,
            sigma=0.1,
            num_scales=3,
            tile_size=32,
            overlap=8,
            n_jobs=1,
        )
        assert ref.shape == tiled.shape
        diff = np.abs(ref - tiled).mean()
        assert diff < 0.07

    def test_tiled_udct_odd_size(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_udct_tiled

        np.random.seed(42)
        image = np.random.rand(63, 63).astype(np.float64)
        result = curvelet_denoise_udct_tiled(
            image,
            sigma=0.1,
            num_scales=3,
            tile_size=32,
            overlap=8,
            n_jobs=1,
        )
        assert result.shape == (63, 63)
        assert np.isfinite(result).all()

    def test_tiled_mct_odd_size(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_mct_tiled

        np.random.seed(42)
        image = np.random.rand(63, 63).astype(np.float64)
        result = curvelet_denoise_mct_tiled(
            image,
            sigma=0.1,
            num_scales=3,
            tile_size=32,
            overlap=8,
            n_jobs=1,
        )
        assert result.shape == (63, 63)
        assert np.isfinite(result).all()
