"""Tests for helicon.lib.curvelet module."""

from __future__ import annotations

import numpy as np
import pytest


class TestHasCurvelet:
    """Tests for has_curvelet_udct."""

    def test_udct_returns_true_when_installed(self):
        pytest.importorskip("curvelets")
        import helicon

        assert helicon.has_curvelet_udct() is True

    def test_public_api_udct_available(self):
        pytest.importorskip("curvelets")
        import helicon

        for name in (
            "curvelet_denoise_3d_udct",
            "curvelet_denoise_3d_udct_tiled",
            "curvelet_denoise_3d_mct",
            "curvelet_denoise_3d_mct_tiled",
        ):
            assert callable(getattr(helicon, name))

    def test_the_2d_denoisers_are_gone(self):
        import helicon

        assert not hasattr(helicon, "curvelet_denoise_udct")
        assert not hasattr(helicon, "has_curvelet_fdct")


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
        # Shapes that need padding, small enough to fit in memory. The pair
        # (3838, 3710) used to be here and made this test able to take down the
        # machine it ran on: the UDCT window builder allocates a
        # (12, 3840, 3712) float64 array per direction -- 1.27 GiB each -- and
        # the process reached 23 GB before the kernel killed it, taking the
        # whole test session with it. What is being checked is that padding
        # survives a forward/backward round trip, which does not depend on the
        # image being detector sized; the padding arithmetic at that size is
        # covered by test_compatible_shape_large_non_divisible, which allocates
        # nothing.
        for shape in [(63, 63), (250, 246)]:
            orig = np.random.randn(*shape).astype(np.float64)
            pad_shape = _udct_compatible_shape(shape, 4)
            if pad_shape != shape:
                pads = tuple((0, ps - ds) for ds, ps in zip(shape, pad_shape))
                padded = np.pad(orig, pads, mode="edge")
            else:
                padded = orig
            # 3 wedges per direction, not 12: see
            # test_non_square_round_trip_is_broken_upstream below. What this
            # test is for is helicon's padding, so it uses a setting where the
            # transform underneath it is exact.
            grid = _get_udct_grid(pad_shape, 4, 3)
            coeffs = grid.forward(padded)
            recon = grid.backward(coeffs)
            recon_cropped = recon[: shape[0], : shape[1]]
            max_err = np.abs(orig - recon_cropped).max()
            assert (
                max_err < 0.01
            ), f"UDCT identity failed for {shape}: max_err={max_err:.4f}"


class TestUDCTNonSquareUpstream:
    """The UDCT round trip is not exact for non-square 2D input.

    Found when the identity test was made small enough to finish: with its
    original (3838, 3710) image it exhausted memory and was killed before ever
    reaching its assertion, so the failure underneath had never been seen.

    The defect is in curvelets 1.1 itself -- reproduced below with no helicon
    code in the loop -- and depends on the scale and wedge counts rather than
    on the padding: (256, 248) round trips to 4e-15 with 3 wedges per
    direction and to 3.99 with 12. Square shapes are exact at every setting
    tried, and 3D volumes are exact even when their axes differ, so this is
    narrow. helicon's own defaults (numScales=3, wedgesPerDir=3) are inside
    the sound region.

    When curvelets fixes this, the xfail below turns into an unexpected pass
    and says so.
    """

    @pytest.mark.xfail(
        reason="curvelets 1.1: non-square 2D UDCT round trip is not exact "
        "at higher wedge counts",
        strict=False,
    )
    def test_non_square_round_trip_is_broken_upstream(self):
        curvelets_numpy = pytest.importorskip("curvelets.numpy")

        rng = np.random.default_rng(0)
        image = rng.standard_normal((256, 248))
        udct = curvelets_numpy.UDCT(image.shape, num_scales=4, wedges_per_direction=12)
        recovered = udct.backward(udct.forward(image))
        assert np.abs(image - recovered).max() < 0.01

    def test_the_same_shape_is_exact_with_fewer_wedges(self):
        curvelets_numpy = pytest.importorskip("curvelets.numpy")

        rng = np.random.default_rng(0)
        image = rng.standard_normal((256, 248))
        udct = curvelets_numpy.UDCT(image.shape, num_scales=4, wedges_per_direction=3)
        recovered = udct.backward(udct.forward(image))
        assert np.abs(image - recovered).max() < 0.01


class TestUDCTGPU:
    """Tests for UDCT GPU (torch backend) path."""

    def test_gpu_public_api_flag(self):
        pytest.importorskip("curvelets")
        import helicon

        assert hasattr(helicon, "has_curvelet_udct_gpu")

    def test_gpu_available(self):
        pytest.importorskip("curvelets")
        import helicon

        if not helicon.has_curvelet_udct_gpu():
            pytest.skip("no GPU backend for the curvelet transform here")
        assert helicon.has_curvelet_udct_gpu() is True

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


class TestParseParamStr:
    """Option values of the form name:key=value:..."""

    def test_a_value_may_contain_an_equals_sign(self):
        from helicon.lib.util import parse_param_str

        _, d = parse_param_str("rlnMicrographName=/data/a=b.mrc")
        assert d == {"rlnMicrographName": "/data/a=b.mrc"}

    def test_a_leading_name_still_separates_from_the_pairs(self):
        from helicon.lib.util import parse_param_str

        name, d = parse_param_str("noEqualsHere:sigma=3")
        assert name == "noEqualsHere"
        assert d == {"sigma": 3}


class TestUDCTDenoiseCurveletPadding:
    """Tests for UDCT denoising with non-divisible image sizes."""

    def test_denoise_odd_size_3d(self):
        pytest.importorskip("curvelets")
        from helicon.lib.curvelet import curvelet_denoise_3d_udct

        np.random.seed(42)
        volume = np.random.rand(15, 31, 63).astype(np.float64)
        result = curvelet_denoise_3d_udct(volume, sigma=0.1, num_scales=4)
        assert result.shape == (15, 31, 63)
        assert np.isfinite(result).all()


class TestBlendTile:
    """Tiles are accumulated as they arrive, not kept for a final blend pass.

    Holding every tile gave peak memory of the output plus every denoised tile
    and its weight; the tiled path exists for the volumes where that does not
    fit.
    """

    def test_a_tile_at_the_volume_edge_is_cropped(self):
        from helicon.lib.curvelet import _blend_tile

        output = np.zeros((4, 4, 4))
        weights = np.zeros((4, 4, 4))
        tile = np.full((3, 3, 3), 2.0)
        ones = np.ones((3, 3, 3))
        _blend_tile(output, weights, tile, ones, slice(2, 5), slice(2, 5), slice(2, 5))
        assert output[2, 2, 2] == 2.0
        assert output[2, 2, 1] == 0.0
        assert output.sum() == 8 * 2.0  # only the 2x2x2 part inside
        assert weights.sum() == 8

    def test_overlapping_tiles_are_averaged(self):
        from helicon.lib.curvelet import _blend_tile

        output = np.zeros((2, 2, 2))
        weights = np.zeros((2, 2, 2))
        once = np.ones((2, 2, 2))
        box = (slice(0, 2), slice(0, 2), slice(0, 2))
        _blend_tile(output, weights, np.full((2, 2, 2), 1.0), once, *box)
        _blend_tile(output, weights, np.full((2, 2, 2), 3.0), once, *box)
        assert np.allclose(output / weights, 2.0)
