"""Automatic radial range for HI3D.

``estimate_radial_range`` was ported from the original HI3D along with the rest
of hi3d_core, and imported into the tab, but never called -- so the range
defaulted to 0..half-box and every map began by including the hollow middle and
the empty corners.

The profile is CYLINDRICAL: the map is averaged along z and then transformed to
polar, so these fixtures are tubes, not spheres. Testing it with a spherical
shell gives a misleading rmin of 0, because the column through the axis still
passes through the shell at high |z|.
"""

import numpy as np
import pytest

from helicon.webApps.lib.hi3d_core import compute_radial_profile, estimate_radial_range


def _tube(inner, outer, n=80):
    _z, y, x = np.mgrid[:n, :n, :n]
    r = np.sqrt((x - n / 2) ** 2 + (y - n / 2) ** 2)
    return ((r > inner) & (r < outer)).astype(np.float32)


class TestEstimateRadialRange:
    @pytest.mark.parametrize("inner,outer", [(10, 25), (5, 12), (18, 30)])
    def test_it_recovers_a_tube_of_known_radii(self, inner, outer):
        lo, hi = estimate_radial_range(compute_radial_profile(_tube(inner, outer)))
        assert lo == pytest.approx(inner, abs=1.5)
        assert hi == pytest.approx(outer, abs=1.5)

    def test_a_solid_core_starts_at_the_axis(self):
        lo, hi = estimate_radial_range(compute_radial_profile(_tube(0, 20)))
        assert lo <= 1.5
        assert hi == pytest.approx(20, abs=1.5)

    def test_the_range_is_narrower_than_the_whole_box(self):
        """The point of the exercise: the old default was 0..half-box."""
        n = 80
        rp = compute_radial_profile(_tube(15, 25, n=n))
        lo, hi = estimate_radial_range(rp)
        assert lo > 0 and hi < n // 2

    def test_a_higher_threshold_gives_a_tighter_range(self):
        rp = compute_radial_profile(_tube(10, 25))
        loose = estimate_radial_range(rp, thresh_ratio=0.05)
        tight = estimate_radial_range(rp, thresh_ratio=0.5)
        assert tight[0] >= loose[0] and tight[1] <= loose[1]

    def test_a_flat_profile_falls_back_to_the_whole_range(self):
        """Nothing above threshold means nothing to choose, so take everything
        rather than returning an empty range the caller would have to guess at."""
        lo, hi = estimate_radial_range(np.zeros(40))
        assert (lo, hi) == (0.0, 39.0)


class TestTheTabUsesIt:
    def test_the_tab_calls_the_estimator(self):
        """It was imported and never called for the life of the port."""
        import pathlib

        src = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src/helicon/webApps/tabs/hi3d_tab.py"
        ).read_text()
        assert (
            "estimate_radial_range(" in src.split("import", 1)[1]
        ), "estimate_radial_range is imported but never called"
