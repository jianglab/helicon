import numpy as np
import pytest

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
