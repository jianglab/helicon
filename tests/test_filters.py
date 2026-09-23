import numpy as np
import pytest

import helicon

from helicon.lib import filters


class TestFilters(object):
    def test_normalize_min_max(self):
        data = np.array([1, 2, 3, 4, 5])
        normalized_data = filters.normalize_min_max(data)
        np.testing.assert_allclose(normalized_data, [0, 0.25, 0.5, 0.75, 1])

    def test_normalize_mean_std(self):
        data = np.array([1, 2, 3, 4, 5])
        normalized_data = filters.normalize_mean_std(data)
        np.testing.assert_allclose(
            normalized_data, [-1.41421356, -0.70710678, 0, 0.70710678, 1.41421356]
        )

    def test_normalize_percentile(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        normalized_data = filters.normalize_percentile(data, percentile=(10, 90))
        np.testing.assert_allclose(
            normalized_data,
            [
                -0.125,
                0.01388889,
                0.15277778,
                0.29166667,
                0.43055556,
                0.56944444,
                0.70833333,
                0.84722222,
                0.98611111,
                1.125,
            ],
            atol=1e-6,
        )

    def test_threshold_data(self):
        data = np.array([1, 2, 3, 4, 5])
        thresholded_data = filters.threshold_data(data, thresh_value=3)
        np.testing.assert_allclose(thresholded_data, [0, 0, 0, 1, 2])
        thresholded_data = filters.threshold_data(data, thresh_fraction=0.5)
        np.testing.assert_allclose(thresholded_data, [0, 0, 0.5, 1.5, 2.5])

    def test_low_high_pass_filter(self):
        data = np.zeros((10, 10))
        data[5, 5] = 1
        filtered_data = filters.low_high_pass_filter(data, low_pass_fraction=0.1)
        assert filtered_data.max() < 1.0
        assert abs(filtered_data.sum() - 1.0) < 1e-5

    def test_generate_tapering_filter(self):
        tapering_filter = filters.generate_tapering_filter((10, 10))
        assert tapering_filter.shape == (10, 10)
        assert tapering_filter.min() == 0
        assert tapering_filter.max() == 1


class TestBackgroundOffset:
    """Where the solvent sits, so density can be referenced to it.

    Every package normalises its maps differently, so zero means something
    different in each one and nothing that treats zero as "no density" can be
    trusted across maps. EMD-1427 is the cautionary case: its solvent sits at
    +1.26 and the inside of its tube at -3, so discarding everything below
    zero keeps the solvent and throws away the structure.
    """

    def _map(self, offset=0.0, seed=0):
        rng = np.random.default_rng(seed)
        data = rng.normal(offset, 1.0, (32, 32, 32)).astype(np.float32)
        data[12:20, 12:20, 12:20] += 30.0  # structure, the minority of the box
        return data

    def test_it_recovers_a_shifted_background(self):
        assert filters.background_offset(self._map(offset=5.0)) == pytest.approx(
            5.0, abs=0.1
        )

    def test_structure_does_not_drag_it(self):
        plain = filters.background_offset(self._map())
        assert plain == pytest.approx(0.0, abs=0.1)

    def test_a_masked_map_reports_no_offset(self):
        # most deposited maps are masked: their solvent is already exactly
        # zero, and what sigma clipping finds inside such a mask is structure
        data = self._map(offset=5.0)
        keep = np.zeros(data.shape, dtype=bool)
        keep[8:24, 8:24, 8:24] = True
        assert filters.background_offset(np.where(keep, data, 0.0)) == 0.0

    def test_an_empty_array_is_not_an_error(self):
        assert filters.background_offset(np.zeros(0, dtype=np.float32)) == 0.0


class TestHelicalBackground:
    """The solvent level of a helical map, read from its axial radial profile.

    Projected along the helical axis and averaged about it, the filament sits
    in the middle and the box edge is solvent -- at whatever level the writing
    software chose. The level is read as HI3D reads it, from the last bins of
    the same profile; what is added is the check that the edge really is
    solvent, and a refusal to guess when it is not.
    """

    def _tube(self, n=64, radius=8.0, width=2.5, offset=0.0, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        yy, xx = np.indices((n, n)) - n // 2
        r = np.hypot(yy, xx)
        slab = np.exp(-((r - radius) ** 2) / (2 * width**2))
        vol = np.repeat(slab[None, :, :], 32, axis=0).astype(np.float32)
        vol += rng.normal(0, noise, vol.shape).astype(np.float32) if noise else 0
        return vol + np.float32(offset), r

    def test_it_recovers_a_shifted_solvent(self):
        vol, _ = self._tube(offset=-0.37, noise=0.02)
        bg = filters.helical_background(vol)
        assert bg.method == "edge"
        assert bg.mean == pytest.approx(-0.37, abs=0.01)
        assert bg.sigma == pytest.approx(0.02, rel=0.25)

    def test_it_is_indifferent_to_the_sign_of_the_offset(self):
        for offset in (-5.0, 0.0, 3.0):
            vol, _ = self._tube(offset=offset, noise=0.01)
            assert filters.helical_background(vol).mean == pytest.approx(
                offset, abs=0.01
            )

    def test_outside_a_mask_the_solvent_is_exactly_zero(self):
        vol, r = self._tube(offset=0.4, noise=0.02)
        vol[:, r > 20] = 0.0
        bg = filters.helical_background(vol)
        assert bg.method == "constant"
        assert bg.mean == 0.0

    def test_a_filament_that_fills_the_box_gives_no_level(self):
        # the profile is still changing at the box edge; there is no solvent
        # in view, so no number is better than a wrong one
        vol, _ = self._tube(n=40, radius=16.0, width=6.0, offset=1.0)
        bg = filters.helical_background(vol)
        assert bg.method == "undetermined"
        assert bg.mean == 0.0

    def test_a_negative_halo_still_recovering_gives_no_level(self):
        vol, r = self._tube(offset=0.0)
        halo = -0.3 * np.exp(-((r - 18.0) ** 2) / (2 * 8.0**2))
        vol = vol + halo[None].astype(np.float32)
        assert filters.helical_background(vol).method == "undetermined"

    def test_subtracting_it_is_always_safe(self):
        vol, _ = self._tube(n=40, radius=16.0, width=6.0, offset=1.0)
        bg = filters.helical_background(vol)
        assert np.array_equal(vol - bg.mean, vol)

    def test_it_shares_hi3d_s_profile_rather_than_copying_it(self):
        from helicon.webApps.lib import hi3d_core

        assert hi3d_core.compute_radial_profile is helicon.compute_radial_profile

    def test_the_level_is_hi3d_s_last_three_bins(self):
        vol, _ = self._tube(offset=0.25, noise=0.02)
        profile = helicon.compute_radial_profile(vol)
        bg = filters.helical_background(vol)
        assert bg.mean == pytest.approx(float(np.mean(profile[-3:])))
