"""The twist search: one map's projection, stretched along the filament axis.

At low resolution a filament with the same packing and another twist is the
same filament stretched along its axis, and so is its side projection. The
search matches each map's projection at a range of stretches and keeps the
best, reporting the stretch and the twist it stands for.
"""

import math

import numpy as np
import pytest

from helicon.webApps.lib import helical_projection_compute as compute

RISE = 4.75


BLOBS = (-30, -18, -6, 6, 18, 30)
SIGMA = 4.0


def _gauss(coords, centre):
    return np.exp(-((coords - centre) ** 2) / (2 * SIGMA**2)).astype(np.float32)


def _ribbon(nz, n, apix, twist, rise=RISE):
    """A flat amyloid-like ribbon: six blobs in a row, stacked by the screw."""
    z = (np.arange(nz, dtype=np.float32) - nz // 2) * apix
    yx = (np.arange(n, dtype=np.float32) - n // 2) * apix
    vol = np.zeros((nz, n, n), np.float32)
    kmax = int(nz * apix / 2 / rise) + 3
    for k in range(-kmax, kmax + 1):
        a = np.deg2rad(twist * k)
        gz = _gauss(z, k * rise)
        for r in BLOBS:
            gy = _gauss(yx, r * np.sin(a))
            gx = _gauss(yx, r * np.cos(a))
            vol += gz[:, None, None] * (gy[:, None] * gx[None, :])[None]
    return vol


def _side_view(nz, n, apix, twist, rise=RISE):
    """The ribbon's side projection, helical axis along x, drawn in 2D.

    Summing an isotropic 3D gaussian along x leaves a 2D gaussian at its
    (y, z), so the projection needs no volume.
    """
    z = (np.arange(nz, dtype=np.float32) - nz // 2) * apix
    y = (np.arange(n, dtype=np.float32) - n // 2) * apix
    image = np.zeros((n, nz), np.float32)
    kmax = int(nz * apix / 2 / rise) + 3
    for k in range(-kmax, kmax + 1):
        a = np.deg2rad(twist * k)
        gz = _gauss(z, k * rise)
        for r in BLOBS:
            image += _gauss(y, r * np.sin(a))[:, None] * gz[None, :]
    return image


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


class TestStretchFactors:
    def test_no_range_is_no_search(self):
        assert compute.axial_stretch_factors(0, 5) == (1.0,)
        assert compute.axial_stretch_factors(None, 5) == (1.0,)
        assert compute.axial_stretch_factors(20, 0) == (1.0,)

    def test_the_range_reaches_both_ways_equally(self):
        factors = compute.axial_stretch_factors(20, 5)
        assert factors[0] == pytest.approx(1 / 1.2)
        assert factors[-1] == pytest.approx(1.2)
        assert 1.0 in factors
        logs = np.log(factors)
        assert np.allclose(logs, -logs[::-1])

    def test_the_steps_are_even_in_log_and_no_coarser_than_asked(self):
        factors = compute.axial_stretch_factors(30, 5)
        ratios = np.array(factors[1:]) / np.array(factors[:-1])
        assert np.allclose(ratios, ratios[0])
        assert ratios[0] <= 1.05 + 1e-9


class TestStretchedTwist:
    def test_a_plain_amyloid_twist_is_divided(self):
        assert compute.stretched_twist(-1.2, RISE, 2.0) == pytest.approx(-0.6)
        assert compute.stretched_twist(-1.2, RISE, 0.5) == pytest.approx(-2.4)

    def test_a_pseudo_two_fold_screw_divides_its_deviation_from_180(self):
        # 179.4 at 2.38 A is 1.2 degrees per 4.75 A rung; stretched twice as
        # long that is 0.6 degrees, i.e. 179.7 per subunit
        assert compute.stretched_twist(179.4, 2.378, 2.0) == pytest.approx(179.7)

    def test_no_stretch_keeps_the_twist(self):
        for twist, rise in ((-1.2, RISE), (179.4, 2.378), (25.0, 6.0)):
            assert compute.stretched_twist(twist, rise, 1.0) == pytest.approx(twist)


class TestStretchAlongAxis:
    def test_no_stretch_is_the_image(self):
        image = np.random.default_rng(0).normal(size=(8, 30)).astype(np.float32)
        out = compute.stretch_along_axis(image, 1.0, 30)
        assert np.array_equal(out, image)

    def test_a_stretch_scales_distances_from_the_centre(self):
        image = np.zeros((4, 101), np.float32)
        image[:, 50 + 10] = 1.0
        out = compute.stretch_along_axis(image, 2.0, 101)
        assert int(np.argmax(out[0])) == 50 + 20

    def test_the_output_has_the_length_asked_for(self):
        image = np.ones((6, 40), np.float32)
        assert compute.stretch_along_axis(image, 0.8, 32).shape == (6, 32)
        assert compute.stretch_along_axis(image, 1.3, 52).shape == (6, 52)

    def test_it_is_the_projection_of_the_filament_with_the_other_twist(self):
        """The physics the search rests on: stretch = smaller twist."""
        reference = _side_view(400, 40, 2.0, -3.0)
        truth = _side_view(400, 40, 2.0, -2.4)
        stretched = compute.stretch_along_axis(reference, 1.25, 400)
        centre = slice(100, 300)
        assert _ncc(stretched[:, centre], truth[:, centre]) > 0.999
        # and it is the stretch that does it, not a lenient measure
        assert _ncc(reference[:, centre], truth[:, centre]) < 0.99


class TestStretchMixture:
    def test_centres_move_along_the_axis_and_the_window_is_cut(self):
        from helicon.webApps.lib.map_gauss_fit import ImageGaussians

        mixture = ImageGaussians(
            amplitudes=np.ones(3, np.float32),
            centers=np.array([[1.0, -40.0], [2.0, 10.0], [3.0, 60.0]], np.float32),
            sigma=2.0,
            apix=1.0,
        )
        out = compute._stretch_mixture(mixture, 2.0, 200, 1.0)
        # 60 A stretched to 120 A is outside a 200 A window, even with the
        # margin of four widths
        assert out.centers[:, 1].tolist() == [-80.0, 20.0]
        assert out.centers[:, 0].tolist() == [1.0, 2.0]
        assert out.sigma == mixture.sigma


@pytest.fixture(scope="module")
def ribbon_search():
    """A map at twist -3 and a query cut from the same ribbon at twist -2.4."""
    apix = 4.0
    map_info = compute.MapInfo(
        data=_ribbon(96, 48, 2.0, -3.0),
        label="ribbon",
        apix=2.0,
        twist=-3.0,
        rise=RISE,
        csym=1,
    )
    full = _side_view(300, 48, 2.0, -2.4)[::2, ::2]
    rng = np.random.default_rng(0)
    query = np.ascontiguousarray(full[:, 25:125])
    query = query + rng.normal(0, 0.3 * query.std(), query.shape).astype("f4")
    return map_info, query, apix


def _search(map_info, query, apix, factors, method="volume", fits=None):
    _, result = compute.symmetrize_project_align_one_map(
        map_info,
        [query],
        ["q"],
        apix,
        True,
        1.2,
        False,
        0.0,
        0.05,
        method,
        fits,
        factors,
    )
    return result


class TestTheSearch:
    def test_without_a_range_the_result_is_the_map_s_own_twist(self, ribbon_search):
        map_info, query, apix = ribbon_search
        result = _search(map_info, query, apix, (1.0,))
        assert result[9] == 1.0
        assert result[10] == pytest.approx(map_info.twist)

    def test_it_finds_the_twist_of_the_query(self, ribbon_search):
        map_info, query, apix = ribbon_search
        factors = compute.axial_stretch_factors(40, 10)
        result = _search(map_info, query, apix, factors)
        # the truth is 1.25; the grid steps by 10%
        assert abs(math.log(result[9] / 1.25)) <= math.log(1.1) + 1e-6
        assert -2.75 < result[10] < -2.15
        assert result[4] > _search(map_info, query, apix, (1.0,))[4]

    def test_the_projection_is_a_stretched_pitch_long(self, ribbon_search):
        map_info, query, apix = ribbon_search
        plain = _search(map_info, query, apix, (1.0,))
        longer = _search(map_info, query, apix, (1.25,))
        pitch = 360 / 3.0 * RISE / apix
        width = plain[7].shape[1]
        assert longer[7].shape[1] - width == pytest.approx(0.25 * pitch, abs=2)
        # the query is placed in the frame of the projection it was matched to
        assert longer[5].shape == longer[7].shape

    def test_the_gaussian_mode_finds_it_too(self, ribbon_search):
        from helicon.webApps.lib import map_gauss_fit as mgf

        map_info, query, apix = ribbon_search
        fits = mgf.fit_queries([query], apix)
        factors = compute.axial_stretch_factors(40, 10)
        result = _search(map_info, query, apix, factors, "gaussian", fits)
        assert 1.1 <= result[9] <= 1.45
        assert result[5].shape == result[7].shape


class TestDisplayKeepsTheStretch:
    def test_a_re_placed_result_keeps_the_stretch_it_was_found_at(self, monkeypatch):
        rng = np.random.default_rng(0)
        proj = rng.normal(0, 1, (32, 96)).astype(np.float32)
        good = np.zeros_like(proj)
        good[:, 30:60] = proj[:, 30:60]
        bad = np.zeros_like(proj)
        bad[:, 30:60] = proj[:, 50:80]
        monkeypatch.setattr(
            compute,
            "align_images",
            lambda **k: (False, 1.0, 0.0, (0.0, 0.0), 0.5, good),
        )
        result = (False, 1.0, 0.0, (0.0, 0.0), 0.5, bad, "q", proj, "map", 1.2, -2.5)
        shown = compute.refine_placement_for_display(result, [good], 0.05)
        assert shown is not result
        assert np.array_equal(shown[5], good)
        assert shown[9:] == (1.2, -2.5)


class TestTheTab:
    def test_the_label_says_what_the_search_found(self):
        from helicon.webApps.tabs.helical_projection_tab import _stretch_label

        found = (False, 1.0, 0.0, (0, 0), 0.9, None, "q", None, "m", 1.25, -2.4)
        assert _stretch_label(found) == "|×1.25 twist=-2.400°"
        unstretched = found[:9] + (1.0, -3.0)
        assert _stretch_label(unstretched) == ""
        assert _stretch_label(found[:9]) == ""

    def test_a_best_stretch_at_the_end_of_the_range_is_marked(self):
        from helicon.webApps.tabs.helical_projection_tab import _stretch_label

        factors = compute.axial_stretch_factors(25, 5)
        at_edge = (False, 1.0, 0.0, (0, 0), 0.9, None, "q", None, "m")
        at_edge += (factors[-1], -2.4)
        assert _stretch_label(at_edge, factors).endswith("(range edge)")
        inside = at_edge[:9] + (factors[-2], -2.5)
        assert "range edge" not in _stretch_label(inside, factors)

    def test_the_controls_sit_together_and_default_to_no_search(self):
        from _ui import page
        from helicon.webApps.tabs import helical_projection_tab as tab

        shown = page(tab.helical_projection_tab_ui)
        assert shown.same_row("twist_search_range", "twist_search_step")
        assert shown.comes_before("scale_range", "twist_search_range")
        assert tab.BOOKMARK_DEFAULTS["twist_range"] == ("twist_search_range", 0)
        assert tab.BOOKMARK_DEFAULTS["twist_step"] == ("twist_search_step", 5)
