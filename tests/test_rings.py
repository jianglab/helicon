"""Medians of a transform over its rings or shells."""

import numpy as np
import pytest

from helicon.lib import rings


def _brute(values, labels, n):
    return np.array(
        [
            np.median(values[labels == k]) if (labels == k).any() else np.nan
            for k in range(n)
        ]
    )


class TestGroupedMedian:
    def test_it_agrees_with_numpys_median_of_each_group(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 9, size=500)
        values = rng.standard_normal(500)
        index = rings.group_index(labels)
        np.testing.assert_allclose(
            rings.grouped_median(values, index), _brute(values, labels, 9)
        )

    def test_the_middle_two_are_averaged_for_an_even_group(self):
        index = rings.group_index([0, 0, 0, 0])
        assert rings.grouped_median([1.0, 2.0, 3.0, 10.0], index)[0] == 2.5

    def test_any_shape_in_c_order(self):
        labels = np.array([[0, 1, 1], [0, 1, 0]])
        values = np.array([[5.0, 1.0, 3.0], [7.0, 2.0, 9.0]])
        got = rings.grouped_median(values, rings.group_index(labels))
        np.testing.assert_allclose(got, [7.0, 2.0])

    def test_a_group_nothing_falls_in_has_no_median(self):
        index = rings.group_index([0, 0, 3, 3])  # 1 and 2 are empty
        got = rings.grouped_median([1.0, 3.0, 5.0, 7.0], index)
        assert got[0] == 2.0 and got[3] == 6.0
        assert np.isnan(got[1]) and np.isnan(got[2])

    def test_a_small_group_has_no_median_below_min_count(self):
        index = rings.group_index([0, 0, 0, 1])
        got = rings.grouped_median([1.0, 2.0, 3.0, 9.0], index, min_count=2)
        assert got[0] == 2.0 and np.isnan(got[1])

    def test_the_number_of_groups_can_be_given(self):
        index = rings.group_index([0, 1], n_groups=5)
        assert len(index.starts) == 5
        assert np.isnan(rings.grouped_median([1.0, 2.0], index)[4])

    def test_one_index_serves_any_values(self):
        labels = np.arange(60) % 6
        index = rings.group_index(labels)
        for seed in (1, 2):
            values = np.random.default_rng(seed).standard_normal(60)
            np.testing.assert_allclose(
                rings.grouped_median(values, index), _brute(values, labels, 6)
            )

    def test_no_pixels_means_no_groups(self):
        index = rings.group_index(np.array([], dtype=int))
        assert len(rings.grouped_median([], index)) == 0


class TestRfft2Rings:
    def test_the_layout_and_the_origin(self):
        ring = rings.rfft2_rings((8, 4))
        assert ring.shape == (8, 4) and ring[0, 0] == 0

    def test_it_is_symmetric_in_the_rows(self):
        ring = rings.rfft2_rings((16, 6))
        np.testing.assert_array_equal(ring[1:], ring[1:][::-1])

    def test_a_ring_is_the_rounded_distance_from_the_origin(self):
        ring = rings.rfft2_rings((16, 9))
        assert ring[3, 4] == 5 and ring[0, 8] == 8 and ring[-4, 3] == 5

    def test_it_is_the_shape_of_the_transform_so_odd_widths_work(self):
        for nx in (9, 10):
            transform = np.fft.rfft2(np.zeros((8, nx)))
            assert rings.rfft2_rings(transform.shape).shape == transform.shape

    def test_it_is_cached_and_not_writable(self):
        a = rings.rfft2_rings((12, 7))
        assert rings.rfft2_rings((12, 7)) is a
        assert not a.flags.writeable


def _old_ring_noise(power):
    """helix_transform.ring_noise as it was: sort once, a median per ring."""
    ny, nx2 = power.shape
    fy = np.fft.fftfreq(ny)[:, None] * ny
    fx = np.arange(nx2)[None, :].astype(float)
    ring = np.rint(np.hypot(fy, fx)).astype(int)
    order = np.argsort(ring, axis=None, kind="stable")
    sorted_ring = ring.ravel()[order]
    sorted_power = power.ravel()[order]
    bounds = np.searchsorted(sorted_ring, np.arange(ring.max() + 2))
    noise = np.full(ring.max() + 1, np.nan)
    for r in range(len(noise)):
        lo, hi = bounds[r], bounds[r + 1]
        if hi - lo > 3:
            noise[r] = np.median(sorted_power[lo:hi]) / np.log(2.0)
    ok = np.flatnonzero(np.isfinite(noise))
    noise = np.interp(np.arange(len(noise)), ok, noise[ok])
    return np.maximum(noise[ring], 1e-30)


def _old_noise_power(power, mask, floor=0.02, min_count=12):
    """layerlines.noise_power as it was: a full-size comparison for every ring."""
    ny, nx2 = power.shape
    fy = np.fft.fftfreq(ny)[:, None] * ny
    fx = np.arange(nx2)[None, :].astype(float)
    ring = np.rint(np.hypot(fy, fx)).astype(int)
    n_ring = int(ring.max()) + 1
    off = mask < floor
    est = np.full(n_ring, np.nan)
    for r in range(n_ring):
        sel = off & (ring == r)
        if sel.sum() >= min_count:
            est[r] = np.median(power[sel]) / np.log(2.0)
    ok = np.flatnonzero(np.isfinite(est))
    if len(ok) == 0:
        return np.zeros_like(power)
    est = np.interp(np.arange(n_ring), ok, est[ok])
    return est[ring]


class TestTheRefactoredEstimatesAgreeWithTheirOldCode:
    @pytest.mark.parametrize("shape", [(64, 33), (48, 24), (65, 32), (40, 41)])
    def test_ring_noise(self, shape):
        from helicon.webApps.lib import helix_transform as ht

        rng = np.random.default_rng(shape[0])
        power = rng.exponential(1.0, size=shape) * (1 + 5 * rng.random(shape) ** 8)
        np.testing.assert_allclose(ht.ring_noise(power), _old_ring_noise(power))

    @pytest.mark.parametrize("shape", [(64, 33), (48, 24), (65, 32)])
    def test_noise_power(self, shape):
        from helicon.lib import layerlines

        rng = np.random.default_rng(shape[1])
        power = rng.exponential(1.0, size=shape)
        mask = rng.random(shape)
        mask[::4] = 1.0  # lines, which are left out of the estimate
        np.testing.assert_allclose(
            layerlines.noise_power(power, mask), _old_noise_power(power, mask)
        )

    def test_noise_power_with_every_pixel_on_a_line(self):
        from helicon.lib import layerlines

        power = np.random.default_rng(0).exponential(1.0, size=(32, 17))
        out = layerlines.noise_power(power, np.ones_like(power))
        assert not out.any()  # nothing to estimate from
        assert not _old_noise_power(power, np.ones_like(power)).any()

    def test_the_three_estimates_share_the_ring_median(self):
        import inspect

        from helicon.lib import layerlines
        from helicon.webApps.lib import helix_transform as ht, hill_compute

        for fn in (
            ht.ring_noise,
            layerlines.noise_power,
            hill_compute.shell_median_ratio,
        ):
            assert "grouped_median" in inspect.getsource(fn)


class TestRfft2RingsShapeTypes:
    def test_a_list_or_numpy_ints_give_the_same_cached_rings(self):
        a = rings.rfft2_rings((12, 7))
        assert rings.rfft2_rings([12, 7]) is a
        assert rings.rfft2_rings(np.array([12, 7])) is a
