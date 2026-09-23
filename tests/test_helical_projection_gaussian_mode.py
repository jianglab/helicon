"""The gaussian mode: map and query both as gaussians, matched analytically.

The app offers two modes. One is voxels and pixels throughout -- symmetrise the
volume, project it, correlate the images. The other never builds either: the
map is a few hundred fitted gaussians, its projection is a mixture derived from
them, the query is fitted too, and the match is the closed-form overlap of the
two mixtures. This covers the second one, from the wiring that reaches it to
the placement it hands the display.
"""

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import helicon
from helicon.webApps.lib import helical_projection_compute as compute
from helicon.webApps.lib import map_gauss_fit as mgf


def _helical_map(nz=64, ny=48, nx=48, apix=2.0, radius=9.0, sigma=3.5):
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    z = (z - nz // 2) * apix
    y = (y - ny // 2) * apix
    x = (x - nx // 2) * apix
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    rise, twist = 6.0, 25.0
    for k in range(-nz, nz):
        cz = k * rise
        if abs(cz) > nz * apix / 2 + 3 * sigma:
            continue
        ang = np.deg2rad(twist * k)
        vol += np.exp(
            -(
                (x - radius * np.cos(ang)) ** 2
                + (y - radius * np.sin(ang)) ** 2
                + (z - cz) ** 2
            )
            / (2 * sigma**2)
        )
    return vol, apix, twist, rise


@pytest.fixture(scope="module")
def setup():
    vol, apix, twist, rise = _helical_map()
    map_info = compute.MapInfo(
        data=vol, label="synthetic", apix=apix, twist=twist, rise=rise, csym=1
    )
    fit = mgf.fit_map(vol, apix, twist, rise, 1, fit_apix=apix, n_components=200)
    proj = mgf.side_projection(fit, twist, rise, 1, 128, 48, apix)
    rng = np.random.default_rng(0)
    w = 32
    queries = [
        np.ascontiguousarray(proj[:, c - w // 2 : c + w // 2])
        + rng.normal(0, 0.2, (48, w)).astype("f4")
        for c in (40, 70, 96)
    ]
    return map_info, queries, apix


def _run(map_info, queries, apix, query_fits=None, method="gaussian"):
    labels = ["q%d" % i for i in range(len(queries))]
    _, result = compute.symmetrize_project_align_one_map(
        map_info,
        queries,
        labels,
        apix,
        True,
        1.2,
        False,
        0.0,
        0.1,
        method,
        query_fits,
    )
    return result


class TestTheGaussianModeRuns:
    def test_it_returns_a_placement_in_the_projection_frame(self, setup):
        map_info, queries, apix = setup
        fits = mgf.fit_queries(queries, apix)
        result = _run(map_info, queries, apix, fits)
        assert result is not None
        composite, projection = result[5], result[7]
        assert composite.shape == projection.shape
        assert np.isfinite(composite).all()

    def test_the_score_is_a_correlation(self, setup):
        map_info, queries, apix = setup
        fits = mgf.fit_queries(queries, apix)
        result = _run(map_info, queries, apix, fits)
        assert 0.0 < result[4] <= 1.0

    def test_it_finds_where_the_query_was_cut_from(self, setup):
        """The placement has to land the query back on its own stretch."""
        map_info, queries, apix = setup
        fits = mgf.fit_queries(queries, apix)
        for query, fit in zip(queries, fits):
            result = _run(map_info, [query], apix, [fit])
            placed, projection = result[5], result[7]
            covered = placed != 0
            assert covered.sum() > 0.5 * query.size
            a = placed[covered] - placed[covered].mean()
            b = projection[covered] - projection[covered].mean()
            ncc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
            assert ncc > 0.5, ncc

    def test_the_score_is_the_mean_over_the_selected_images(self, setup):
        map_info, queries, apix = setup
        fits = mgf.fit_queries(queries, apix)
        singles = [_run(map_info, [q], apix, [f]) for q, f in zip(queries, fits)]
        joint = _run(map_info, queries, apix, fits)
        assert joint[4] == pytest.approx(np.mean([s[4] for s in singles]), rel=1e-6)

    def test_without_query_fits_it_falls_back_to_the_pixel_aligner(self, setup):
        """A query that cannot be fitted must not take the search down."""
        map_info, queries, apix = setup
        result = _run(map_info, queries, apix, query_fits=None)
        assert result is not None
        assert result[5].shape == result[7].shape

    def test_the_volume_mode_ignores_the_fits(self, setup):
        map_info, queries, apix = setup
        fits = mgf.fit_queries(queries, apix)
        with_fits = _run(map_info, queries, apix, fits, method="volume")
        without = _run(map_info, queries, apix, None, method="volume")
        assert with_fits[4] == without[4]


class TestDisplayRefinement:
    """The hybrid: rank by gaussians, place the top matches by pixels."""

    def test_it_keeps_the_score_and_replaces_the_placement(self, setup):
        map_info, queries, apix = setup
        fits = mgf.fit_queries(queries, apix)
        result = _run(map_info, [queries[0]], apix, [fits[0]])
        refined = compute.refine_placement_for_display(result, [queries[0]], 0.1)
        assert refined[4] == result[4]
        assert refined[6] == result[6]
        assert refined[7] is result[7]
        assert refined[5].shape == result[5].shape

    def test_the_refinement_does_not_degrade_the_picture(self, setup):
        """Not "improves": on this clean synthetic map the two placements are
        equally good to within a few parts in a thousand, and which one wins
        depends on the measure. What the refinement is for shows on real class
        averages, where the analytic placement scored 0.47 to 0.64 by the pixel
        aligner's own measure and its refinement 0.65 to 0.68. The check that
        travels is that re-placing never throws the picture away."""
        map_info, queries, apix = setup
        fits = mgf.fit_queries(queries, apix)

        def ncc(placed, projection):
            covered = placed != 0
            a = placed[covered] - placed[covered].mean()
            b = projection[covered] - projection[covered].mean()
            return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

        for query, fit in zip(queries, fits):
            result = _run(map_info, [query], apix, [fit])
            refined = compute.refine_placement_for_display(result, [query], 0.1)
            assert ncc(refined[5], refined[7]) > 0.95 * ncc(result[5], result[7])


class TestTheTabReachesIt:
    """Wiring, so the mode cannot be present in the library and absent in the UI."""

    def _source(self):
        from helicon.webApps.tabs import helical_projection_tab as tab

        return inspect.getsource(tab), tab

    def test_the_search_fits_the_queries_and_passes_them_down(self):
        source, _ = self._source()
        assert "map_gauss_fit.fit_queries(query_imgs, query_apix)" in source
        assert "query_fits," in source

    def test_the_top_matches_are_re_placed_for_display(self):
        source, tab = self._source()
        assert "refine_placement_for_display" in source
        assert tab.POLISHED_PAIRS_FOR_DISPLAY > 0
        # counted in image-map pairs, so a joint search does not pay ten times
        assert "// max(1, len(query_imgs))" in source

    def test_the_gaussian_mode_is_the_default(self):
        source, tab = self._source()
        assert tab.BOOKMARK_DEFAULTS["projection_method"][1] == "gaussian"
        assert 'selected="gaussian"' in source


class TestTheDisplayedPlacementIsScaleCorrected:
    """The analytic search never varies scale, so the picture must be re-placed.

    On the default query against EMD-14046 the gaussian route reports a scale
    of exactly 1.0000 while the pixel aligner finds 0.9886 -- the query really
    does sit about 1% larger than the projection it is drawn against. The
    re-placement is what puts them at one scale.
    """

    def test_one_result_is_re_placed_too(self):
        """A single selected map is when the placement is studied, not ranked."""
        import inspect

        from helicon.webApps.tabs import helical_projection_tab as tab

        body = inspect.getsource(tab)
        body = body[body.index("def _compare_projections") :]
        assert "if query_fits is not None and len(good):" in body
        assert "len(good) > 1" not in body

    def test_the_analytic_route_reports_no_scale_of_its_own(self, setup):
        map_info, queries, apix = setup
        fits = mgf.fit_queries(queries, apix)
        result = _run(map_info, [queries[0]], apix, [fits[0]])
        assert result[1] == 1.0

    def test_the_re_placement_can_change_the_scale(self, setup):
        map_info, queries, apix = setup
        fits = mgf.fit_queries(queries, apix)
        result = _run(map_info, [queries[0]], apix, [fits[0]])
        refined = compute.refine_placement_for_display(result, [queries[0]], 0.05)
        # it may or may not move on synthetic data, but it must be free to
        assert 0.95 <= refined[1] <= 1.05
        assert refined[4] == result[4]


class TestTheDisplayKeepsTheBetterPlacement:
    """Neither aligner's placement is reliably the better one to show.

    On the default query against EMD-14046 the pixel aligner's placement fits
    the projection far better (NCC 0.930 against 0.738 over the query's
    footprint); on a class average against EMD-60539 it lands on a poor
    optimum (0.740 against the gaussian placement's 0.907) and the query was
    drawn visibly off its crossover. So both are judged by a measure neither
    optimises, and the better is kept.
    """

    def _result(self, placed, proj):
        return (False, 1.0, 0.0, (0.0, 0.0), 0.5, placed, "q", proj, "map")

    def _scene(self):
        rng = np.random.default_rng(0)
        proj = rng.normal(0, 1, (32, 96)).astype(np.float32)
        good = np.zeros_like(proj)
        good[:, 30:60] = proj[:, 30:60]
        bad = np.zeros_like(proj)
        bad[:, 30:60] = proj[:, 50:80]
        return proj, good, bad

    def test_the_agreement_measure(self):
        proj, good, bad = self._scene()
        assert compute.placement_agreement(good, proj) == pytest.approx(1.0)
        assert compute.placement_agreement(bad, proj) < 0.5
        assert compute.placement_agreement(np.zeros_like(proj), proj) == -1.0

    def test_a_worse_pixel_placement_is_not_shown(self, monkeypatch):
        proj, good, bad = self._scene()
        monkeypatch.setattr(
            compute,
            "align_images",
            lambda **k: (True, 1.02, 180.0, (0.0, 20.0), 0.99, bad),
        )
        result = self._result(good, proj)
        shown = compute.refine_placement_for_display(result, [good], 0.05)
        assert shown is result

    def test_a_better_pixel_placement_is_shown(self, monkeypatch):
        proj, good, bad = self._scene()
        monkeypatch.setattr(
            compute,
            "align_images",
            lambda **k: (True, 1.02, 180.0, (0.0, 20.0), 0.10, good),
        )
        result = self._result(bad, proj)
        shown = compute.refine_placement_for_display(result, [good], 0.05)
        assert np.array_equal(shown[5], good)
        # and it carries the transform that produced it
        assert shown[1] == pytest.approx(1.02)
        assert shown[2] == pytest.approx(180.0)
        # while the score is still the search's
        assert shown[4] == result[4]

    def test_the_aligner_s_own_score_does_not_decide(self, monkeypatch):
        # align_images rates its own placement 0.99 here; the footprint NCC
        # says otherwise, and that is what is trusted
        proj, good, bad = self._scene()
        monkeypatch.setattr(
            compute,
            "align_images",
            lambda **k: (False, 1.0, 0.0, (0.0, 0.0), 0.99, bad),
        )
        result = self._result(good, proj)
        assert compute.refine_placement_for_display(result, [good], 0.05) is result
