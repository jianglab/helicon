import unittest
import numpy as np
from helicon.lib import filters

class TestFilters(unittest.TestCase):
    def test_normalize_min_max(self):
        data = np.array([1, 2, 3, 4, 5])
        normalized_data = filters.normalize_min_max(data)
        np.testing.assert_allclose(normalized_data, [0, 0.25, 0.5, 0.75, 1])

    def test_normalize_mean_std(self):
        data = np.array([1, 2, 3, 4, 5])
        normalized_data = filters.normalize_mean_std(data)
        np.testing.assert_allclose(normalized_data, [-1.41421356, -0.70710678, 0, 0.70710678, 1.41421356])

    def test_normalize_percentile(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        normalized_data = filters.normalize_percentile(data, percentile=(10, 90))
        np.testing.assert_allclose(normalized_data, [-0.125, 0.01388889, 0.15277778, 0.29166667, 0.43055556, 0.56944444, 0.70833333, 0.84722222, 0.98611111, 1.125], atol=1e-6)

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
        self.assertLess(filtered_data.max(), 1.0)
        self.assertAlmostEqual(filtered_data.sum(), 1.0, places=5)

    def test_generate_tapering_filter(self):
        tapering_filter = filters.generate_tapering_filter((10, 10))
        self.assertEqual(tapering_filter.shape, (10, 10))
        self.assertEqual(tapering_filter.min(), 0)
        self.assertEqual(tapering_filter.max(), 1)

if __name__ == '__main__':
    unittest.main()
