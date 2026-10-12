"""The filament radius estimate of HILL, and its memory."""

import numpy as np
import pytest

from helicon.webApps.lib import hill_compute as hill


def _filament(radius=20, n=96, rows=96, seed=0, noise=0.02):
    """A tube along the rows, a projection of a hollow cylinder: a known radius."""
    x = np.arange(n) - n // 2
    inside = np.abs(x) <= radius
    profile = np.zeros(n)
    profile[inside] = np.sqrt(radius**2 - x[inside] ** 2)
    profile /= profile.max()
    rng = np.random.default_rng(seed)
    return np.tile(profile, (rows, 1)) + noise * rng.standard_normal((rows, n))


class TestEstimateRadialRange:
    def test_it_finds_the_radius_of_a_filament_of_known_size(self):
        for radius in (14, 20, 28):
            rmean, mask_radius = hill.estimate_radial_range(_filament(radius))
            # the mask range encloses the filament and the mean radius is inside it
            assert radius <= mask_radius <= radius + 4
            assert 0.4 * radius < rmean <= mask_radius

    def test_the_same_profile_is_remembered(self):
        image = _filament(18, seed=1)
        hill.estimate_radial_range(image)
        before = hill._radial_range_of_profile.cache_info()
        again = hill.estimate_radial_range(image.copy())  # another array, same sums
        after = hill._radial_range_of_profile.cache_info()
        assert after.hits == before.hits + 1 and after.misses == before.misses
        assert again == hill.estimate_radial_range(image)

    def test_a_different_profile_is_worked_out(self):
        a, b = _filament(14, seed=2), _filament(28, seed=2)
        before = hill._radial_range_of_profile.cache_info()
        ra, rb = hill.estimate_radial_range(a), hill.estimate_radial_range(b)
        assert hill._radial_range_of_profile.cache_info().misses >= before.misses + 1
        assert rb[1] > ra[1]

    def test_rows_that_do_not_change_the_column_sums_do_not_matter(self):
        # a shift along the filament's axis leaves the profile alone
        image = _filament(18, seed=3)
        rolled = np.roll(image, 7, axis=0)
        assert hill.estimate_radial_range(image) == hill.estimate_radial_range(rolled)

    def test_the_threshold_is_part_of_what_is_remembered(self):
        image = _filament(18, seed=4)
        a = hill.estimate_radial_range(image, thresh_ratio=0.1)
        b = hill.estimate_radial_range(image, thresh_ratio=0.4)
        assert a != b

    def test_it_does_not_change_the_image(self):
        image = _filament(18, seed=5)
        before = image.copy()
        hill.estimate_radial_range(image)
        np.testing.assert_array_equal(image, before)
