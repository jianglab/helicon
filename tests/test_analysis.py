import unittest
from unittest.mock import patch
import numpy as np
from helicon.lib import analysis

class TestAnalysis(unittest.TestCase):
    def test_is_3d(self):
        self.assertTrue(analysis.is_3d(np.zeros((10, 10, 10))))
        self.assertTrue(analysis.is_3d(np.zeros((12, 10, 10))))
        self.assertFalse(analysis.is_3d(np.zeros((10, 12, 10))))
        self.assertFalse(analysis.is_3d(np.zeros((10, 10))))

    @patch('helicon.lib.dataset.EMDB')
    def test_is_amyloid(self, mock_emdb):
        mock_emdb.return_value.amyloid_atlas_ids.return_value = ['1234', '5678']
        self.assertTrue(analysis.is_amyloid('EMD-1234'))
        self.assertFalse(analysis.is_amyloid('EMD-9999'))

    def test_twist2pitch(self):
        self.assertAlmostEqual(analysis.twist2pitch(10, 1, return_pitch_for_4p75Angstrom_rise=False), 36)
        self.assertAlmostEqual(analysis.twist2pitch(10, 1), 36.0)

    def test_calc_fsc(self):
        map1 = np.random.rand(10, 10, 10)
        map2 = np.random.rand(10, 10, 10)
        fsc = analysis.calc_fsc(map1, map2, apix=1.0)
        self.assertEqual(fsc.shape[1], 2)
        self.assertAlmostEqual(fsc[0, 1], 1.0, places=5)

        # test with identical maps
        fsc_identical = analysis.calc_fsc(map1, map1, apix=1.0)
        np.testing.assert_allclose(fsc_identical[:, 1], 1.0, atol=1e-6)

    def test_get_cylindrical_mask(self):
        mask = analysis.get_cylindrical_mask(10, 10, 10, rmin=2, rmax=4)
        self.assertEqual(mask.shape, (10, 10, 10))
        self.assertFalse(mask[5, 5, 5])
        self.assertTrue(mask[5, 7, 5])
        self.assertFalse(mask[5, 9, 5])

    def test_cross_correlation_coefficient(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        self.assertAlmostEqual(analysis.cross_correlation_coefficient(a, b), 1.0)
        c = np.array([3, 2, 1])
        self.assertAlmostEqual(analysis.cross_correlation_coefficient(a, c), -1.0)
        d = np.array([1, 1, 1])
        self.assertAlmostEqual(analysis.cross_correlation_coefficient(a, d), 0.0)

    def test_cosine_similarity(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        self.assertAlmostEqual(analysis.cosine_similarity(a, b), 1.0)
        c = np.array([-1, -2, -3])
        self.assertAlmostEqual(analysis.cosine_similarity(a, c), -1.0)
        d = np.array([3, -1, -1/3])
        self.assertAlmostEqual(analysis.cosine_similarity(a, d), 0)

    def test_find_elbow_point(self):
        curve = np.array([10, 8, 6, 4, 2, 1, 0.5, 0.2, 0.1, 0.05])
        elbow_point = analysis.find_elbow_point(curve)
        self.assertEqual(elbow_point, 4)

    def test_agglomerative_clustering_with_min_size(self):
        X = np.array([[1, 2], [1, 4], [1, 0],
                      [4, 2], [4, 4], [4, 0],
                      [10, 2], [10, 4], [10, 0]])
        clustering = analysis.AgglomerativeClusteringWithMinSize(n_clusters=3, min_cluster_size=2)
        clustering.fit(X)
        self.assertEqual(clustering.n_clusters_, 3)

        # Test with a small cluster
        X = np.array([[1, 2], [1, 4], [1, 0],
                      [4, 2], [4, 4], [4, 0],
                      [10, 2]])
        clustering = analysis.AgglomerativeClusteringWithMinSize(n_clusters=3, min_cluster_size=3)
        clustering.fit(X)
        self.assertEqual(clustering.n_clusters_, 2)

if __name__ == '__main__':
    unittest.main()
