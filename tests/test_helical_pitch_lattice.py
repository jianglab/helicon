"""The subunit lattice of a helix from its class averages, and the twist it implies.

The class averages are simulated: a helix of known twist, rise and symmetry seen at
several azimuths, as the averages of one filament type would be.
"""

import inspect

import numpy as np
import pytest

from helicon.webApps.lib import helical_pitch_lattice as hl
from helicon.webApps.lib import helical_pitch_map as maps
from helicon.webApps.lib import hill_compute

APIX = 3.0


def _averages(twist=29.4, rise=21.8, csym=6, n=4):
    """Side views of one helix at ``n`` azimuths, filament horizontal."""
    out = []
    for k in range(n):
        img = hill_compute.simulate_helix(
            twist, rise, csym, 77.6, 8.0, 160, 96, APIX, az0=17.0 + 11.0 * k
        )
        out.append(np.ascontiguousarray(img.T, dtype=np.float32))
    return out


class TestDetect:
    def test_a_resolved_helix_shows_its_rise_and_near_repeat(self):
        lat = hl.detect(_averages(), APIX)
        assert lat is not None
        assert lat.rise == pytest.approx(21.8, abs=0.6)
        # two subunits turn 58.8 degrees, close to a C6 period
        assert lat.k_s == 2
        assert lat.m == 1
        assert [p[2] for p in lat.peaks][:2] == [1, 2]

    def test_nothing_that_repeats_along_the_axis_is_no_lattice(self):
        y = np.arange(96.0) - 48
        rod = np.exp(-(y**2) / 200.0)[:, None] * np.ones((1, 160))
        assert hl.detect([rod, rod * 0.9], APIX) is None

    def test_the_curve_is_kept_for_display(self):
        lat = hl.detect(_averages(n=2), APIX)
        assert lat.shifts.shape == lat.correlation.shape
        assert lat.shifts[0] > 0


class TestTwist:
    VIPAB = hl.Lattice(rise=21.8, k_s=2, m=1)

    def test_vipab_turns_25_times_60_degrees_over_51_subunits(self):
        j, n = hl.periods_per_repeat(1116.0, 21.8, self.VIPAB)
        assert (j, n) == (25, 51.0)
        assert hl.twist(1116.0, 21.8, 6, self.VIPAB) == pytest.approx(29.41, abs=0.01)
        # every symmetry with the same csym * twist
        assert hl.twist(1116.0, 21.8, 1, self.VIPAB) == pytest.approx(176.47, abs=0.01)
        assert hl.twist(1116.0, 21.8, 2, self.VIPAB) == pytest.approx(88.24, abs=0.01)

    def test_without_a_lattice_it_is_one_turn_as_for_an_amyloid(self):
        j, n = hl.periods_per_repeat(850.0, 4.75, None)
        assert j == 1
        assert n == pytest.approx(850.0 / 4.75)
        assert hl.twist(850.0, 4.75, 1) == pytest.approx(360.0 * 4.75 / 850.0)
        assert hl.twist(850.0, 4.75, 2) == pytest.approx(360.0 * 4.75 / 1700.0)

    def test_a_tie_is_the_hand_and_the_lower_count_is_taken(self):
        # 51 subunits, half a period per subunit: 25.5, between 25 and its mirror 26
        j, _ = hl.periods_per_repeat(51 * 21.8, 21.8, self.VIPAB)
        assert j == 25

    def test_the_turns_in_a_longer_near_repeat_follow_the_peak_heights(self):
        # k_s 5: with m 2, shifts of 2 and 3 subunits come closer to a period
        # than shifts of 1 and 4
        heights = {
            1: (21.8, 0.3),
            2: (43.6, 0.6),
            3: (65.4, 0.6),
            4: (87.2, 0.3),
            5: (109.0, 0.9),
        }
        assert hl._turns_in_near_repeat(heights, 5) == 2
        heights = {
            1: (21.8, 0.6),
            2: (43.6, 0.3),
            3: (65.4, 0.3),
            4: (87.2, 0.6),
            5: (109.0, 0.9),
        }
        assert hl._turns_in_near_repeat(heights, 5) == 1


class TestTheMap:
    def test_with_many_turns_the_twist_counts_whole_subunits(self):
        images = [np.zeros((32, 48), np.float32) for _ in range(3)]
        for im in images:
            im[12:20] = 1.0
        out = maps.reconstruct_map(
            images,
            [0.0, 120.0, 240.0],
            1116.0,
            6,
            21.8,
            left_handed=False,
            apix=5.0,
            method="backprojection",
            turns=25,
        )
        assert out["twist"] == pytest.approx(25 * 360.0 / (6 * 51))

    def test_one_turn_keeps_the_amyloid_formula(self):
        images = [np.zeros((32, 48), np.float32) for _ in range(3)]
        out = maps.reconstruct_map(
            images,
            [0.0, 120.0, 240.0],
            850.0,
            1,
            4.75,
            apix=5.0,
            method="backprojection",
        )
        assert out["twist"] == pytest.approx(-360.0 * 4.75 / 850.0)

    def test_projection_matching_places_the_averages_itself(self):
        straight = maps.straighten_classes(
            _averages(n=3), np.zeros(3), target_apix=6.0, apix=APIX
        )
        out = maps.reconstruct_map(
            straight,
            [0.0, 0.0, 0.0],  # ring angles it does not use
            1116.0,
            6,
            21.8,
            left_handed=False,
            apix=6.0,
            method="projection",
            turns=25,
            cpu=2,
        )
        assert out["twist"] == pytest.approx(29.41, abs=0.01)
        assert len(out["phis"]) == 3
        assert out["volume"].ndim == 3
        assert out["projection"].shape[0] == straight[0].shape[0]


class TestTheTab:
    def test_the_pitch_job_measures_the_lattice_and_the_map_uses_it(self):
        from helicon.webApps.tabs import abinitio3d_tab

        src = inspect.getsource(abinitio3d_tab)
        assert "result.lattice = _lattice(job, result)" in src
        assert "lattice.twist(r.period, rise, c, lat)" in src
        assert '"projection"\n                if job["turns"] > 1' in src
        assert 'ui.update_numeric("rise", value=round(float(lat.rise), 2))' in src
