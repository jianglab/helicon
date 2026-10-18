from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from helicon.lib import analysis


class TestAnalysis(object):
    def test_is_3d(self):
        assert analysis.is_3d(np.zeros((10, 10, 10)))
        assert analysis.is_3d(np.zeros((12, 10, 10)))
        assert not analysis.is_3d(np.zeros((10, 12, 10)))
        assert not analysis.is_3d(np.zeros((10, 10)))

    @patch("helicon.lib.dataset.EMDB")
    def test_is_amyloid(self, mock_emdb):
        mock_emdb.return_value.amyloid_atlas_ids.return_value = ["1234", "5678"]
        assert analysis.is_amyloid("EMD-1234")
        assert not analysis.is_amyloid("EMD-9999")

    def test_twist2pitch(self):
        assert (
            abs(
                analysis.twist2pitch(10, 1, return_pitch_for_4p75Angstrom_rise=False)
                - 36
            )
            < 1e-7
        )
        assert abs(analysis.twist2pitch(10, 1) - 36.0) < 1e-7

    def test_calc_fsc(self):
        map1 = np.random.rand(10, 10, 10)
        map2 = np.random.rand(10, 10, 10)
        fsc = analysis.calc_fsc(map1, map2, apix=1.0)
        assert fsc.shape[1] == 2
        assert abs(fsc[0, 1] - 1.0) < 1e-5

        # test with identical maps
        fsc_identical = analysis.calc_fsc(map1, map1, apix=1.0)
        np.testing.assert_allclose(fsc_identical[:, 1], 1.0, atol=1e-6)

    def test_calc_fsc_per_shell_matches_reference(self):
        """Optimized calc_fsc_per_shell must match the np.add.at reference
        bit-for-bit, both with and without a precomputed shell grid."""
        rng = np.random.default_rng(0)
        for n in (16, 32):
            m1 = rng.normal(size=(n, n, n))
            m2 = rng.normal(size=(n, n, n))
            # Reference: original implementation (meshgrid + np.add.at).
            from scipy.fft import fftn

            F1 = fftn(m1, workers=-1)
            F2 = fftn(m2, workers=-1)
            kx = np.fft.fftfreq(n)
            KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
            kr = np.sqrt(KX**2 + KY**2 + KZ**2)
            shell = np.round(kr * n).astype(np.int32)
            np.clip(shell, 0, n // 2, out=shell)
            nshells = n // 2 + 1
            num = np.zeros(nshells)
            den1 = np.zeros(nshells)
            den2 = np.zeros(nshells)
            flat_shell = shell.ravel()
            flat_num = np.real(F1 * np.conj(F2)).ravel()
            flat_den1 = np.abs(F1).ravel() ** 2
            flat_den2 = np.abs(F2).ravel() ** 2
            np.add.at(num, flat_shell, flat_num)
            np.add.at(den1, flat_shell, flat_den1)
            np.add.at(den2, flat_shell, flat_den2)
            denom = np.sqrt(den1 * den2)
            ref = np.ones(nshells)
            valid = denom > 0
            ref[valid] = num[valid] / denom[valid]

            # Default path (shell recomputed internally).
            out_default = analysis.calc_fsc_per_shell(m1, m2, 1.0)
            np.testing.assert_array_equal(out_default, ref)

            # Precomputed-shell path (used by the mask-slope optimiser).
            out_precomputed = analysis.calc_fsc_per_shell(
                m1, m2, 1.0, shell_flat=flat_shell, n=n
            )
            np.testing.assert_array_equal(out_precomputed, ref)

    def test_fsc_from_rfft_matches_bincount(self):
        """_fsc_from_rfft must match the pure-numpy np.bincount reference."""
        rng = np.random.default_rng(1)
        for n in (16, 32):
            F1 = rng.normal(size=(n, n, n // 2 + 1)) + 1j * rng.normal(
                size=(n, n, n // 2 + 1)
            )
            F2 = rng.normal(size=(n, n, n // 2 + 1)) + 1j * rng.normal(
                size=(n, n, n // 2 + 1)
            )
            k2 = np.fft.fftfreq(n) ** 2
            kr2 = np.fft.rfftfreq(n) ** 2
            shell = np.round(
                np.sqrt(k2[:, None, None] + k2[None, :, None] + kr2[None, None, :]) * n
            ).astype(np.int32)
            np.clip(shell, 0, n // 2, out=shell)
            shell_flat = shell.ravel()
            nshells = n // 2 + 1
            num = np.bincount(
                shell_flat, weights=np.real(F1 * np.conj(F2)).ravel(), minlength=nshells
            )
            den1 = np.bincount(
                shell_flat, weights=(np.abs(F1) ** 2).ravel(), minlength=nshells
            )
            den2 = np.bincount(
                shell_flat, weights=(np.abs(F2) ** 2).ravel(), minlength=nshells
            )
            denom = np.sqrt(den1 * den2)
            ref = np.ones(nshells)
            valid = denom > 0
            ref[valid] = num[valid] / denom[valid]

            out = analysis._fsc_from_rfft(F1, F2, shell_flat, n)
            np.testing.assert_allclose(out, ref, atol=1e-12, rtol=1e-12)

    def test_get_cylindrical_mask(self):
        mask = analysis.get_cylindrical_mask(10, 10, 10, rmin=2, rmax=4)
        assert mask.shape == (10, 10, 10)
        assert not mask[5, 5, 5]
        assert mask[5, 7, 5]
        assert not mask[5, 9, 5]

    def test_get_cylindrical_mask_matches_a_full_coordinate_grid(self):
        # the mask is built from one (ny, nx) plane and broadcast over z; a
        # full meshgrid of coordinates costs GBs on a 512^3 map
        nz, ny, nx = 4, 9, 11
        mask = analysis.get_cylindrical_mask(nz, ny, nx, rmin=2, rmax=4)
        k = np.arange(nz) - nz // 2
        j = np.arange(ny) - ny // 2
        i = np.arange(nx) - nx // 2
        _z, y, x = np.meshgrid(k, j, i, indexing="ij")
        ref = ((x * x + y * y) < 16) & ((x * x + y * y) >= 4)
        assert mask.dtype == bool
        assert np.array_equal(mask, ref)

    def test_get_cylindrical_mask_returns_xyz_on_request(self):
        mask, (z, y, x) = analysis.get_cylindrical_mask(
            3, 4, 5, rmax=1, return_xyz=True
        )
        assert mask.shape == (3, 4, 5)
        assert z.shape == y.shape == x.shape == (3, 4, 5)

    def test_cross_correlation_coefficient(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        assert abs(analysis.cross_correlation_coefficient(a, b) - 1.0) < 1e-7
        c = np.array([3, 2, 1])
        assert abs(analysis.cross_correlation_coefficient(a, c) - (-1.0)) < 1e-7
        d = np.array([1, 1, 1])
        assert abs(analysis.cross_correlation_coefficient(a, d) - 0.0) < 1e-7

    def test_cosine_similarity(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        assert abs(analysis.cosine_similarity(a, b) - 1.0) < 1e-7
        c = np.array([-1, -2, -3])
        assert abs(analysis.cosine_similarity(a, c) - (-1.0)) < 1e-7
        d = np.array([3, -1, -1 / 3])
        assert abs(analysis.cosine_similarity(a, d) - 0) < 1e-7

    def test_find_elbow_point(self):
        curve = np.array([10, 8, 6, 4, 2, 1, 0.5, 0.2, 0.1, 0.05])
        elbow_point = analysis.find_elbow_point(curve)
        assert elbow_point == 4

    def test_agglomerative_clustering_with_min_size(self):
        X = np.array(
            [[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0], [10, 2], [10, 4], [10, 0]]
        )
        clustering = analysis.AgglomerativeClusteringWithMinSize(
            n_clusters=3, min_cluster_size=2
        )
        clustering.fit(X)
        assert clustering.n_clusters_ == 3

        # Test with a small cluster
        X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0], [10, 2]])
        clustering = analysis.AgglomerativeClusteringWithMinSize(
            n_clusters=3, min_cluster_size=3
        )
        clustering.fit(X)
        assert clustering.n_clusters_ == 2


class TestInterSegmentDistance:
    def _data(self, track_lengths):
        return pd.DataFrame(
            {
                "rlnImageName": [f"{i + 1}@m1.mrc" for i in range(len(track_lengths))],
                "rlnHelicalTubeID": [1] * len(track_lengths),
                "rlnCoordinateX": [float(i) for i in range(len(track_lengths))],
                "rlnCoordinateY": [0.0] * len(track_lengths),
                "rlnHelicalTrackLengthAngst": track_lengths,
            }
        )

    def test_missing_columns_give_none(self):
        assert analysis.estimate_inter_segment_distance(pd.DataFrame({"x": [1]})) == (
            None,
            None,
            None,
            None,
        )

    def test_a_single_segment_helix_gives_none(self):
        # no pair of segments: there is no distance to estimate
        assert analysis.estimate_inter_segment_distance(self._data([100.0])) == (
            None,
            None,
            None,
            None,
        )

    def test_a_real_helix_reports_its_distance(self):
        median, mean, sigma, n_max = analysis.estimate_inter_segment_distance(
            self._data([0.0, 10.0, 20.0])
        )
        assert median == pytest.approx(10.0)
        assert mean == pytest.approx(10.0)
        assert sigma == pytest.approx(0.0)
        assert n_max == 3

    def test_reset_gives_none_when_the_distance_cannot_be_estimated(self):
        data = self._data([100.0])
        assert analysis.reset_inter_segment_distance(data, 50.0, 1.0) is None
        assert analysis.estimate_helicalTube_length(data) is None
