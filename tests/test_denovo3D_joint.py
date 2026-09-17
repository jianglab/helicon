"""Tests for the joint multi-image helical parameter search.

The point of the joint search is that a single 2D class average picks the twist
unreliably while several together do not, so the tests here are mostly about the
*combination* being robust to the ways individual curves go wrong: different
score scales, different spreads, and curves that carry no information at all.
"""

import numpy as np
import pytest

from helicon.webApps.lib import denovo3d_joint as J
from helicon.webApps.tabs.denovo3d_tab import _rank


def _curve(twists, peak, height=0.1, offset=0.5, width=0.15):
    """A score curve peaking at ``peak``."""
    t = np.asarray(twists, dtype=float)
    return offset + height * np.exp(-0.5 * ((t - peak) / width) ** 2)


TWISTS = np.arange(0.6, 1.81, 0.1)
TRUE = 1.2


def test_combined_curve_beats_a_wrong_majority_vote():
    """Curves that individually miss can still combine to the right answer.

    This is the whole justification for the feature: on real data only 5/10 good
    class averages peaked at the correct twist on their own.
    """
    curves = [
        _curve(TWISTS, 1.0, height=0.08),  # wrong peak
        _curve(TWISTS, 1.4, height=0.08),  # wrong peak, other side
        _curve(TWISTS, TRUE, height=0.12),
        _curve(TWISTS, TRUE, height=0.10),
    ]
    best, margin, _, _ = J.joint_best(curves, list(TWISTS))
    assert best == pytest.approx(TRUE, abs=0.05)
    assert margin > 0


def test_z_scoring_defeats_a_dominant_scale():
    """One curve with a huge dynamic range must not outvote everyone else.

    A plain average of raw scores is dominated by whichever image has the widest
    score range, which measured a 65x smaller margin on real data.
    """
    curves = [
        _curve(TWISTS, 0.8, height=5.0, offset=100.0),  # loud and wrong
        _curve(TWISTS, TRUE, height=0.05),
        _curve(TWISTS, TRUE, height=0.05),
        _curve(TWISTS, TRUE, height=0.05),
    ]
    assert np.mean(curves, axis=0).argmax() != np.argmin(np.abs(TWISTS - TRUE))
    best, _, _, _ = J.joint_best(curves, list(TWISTS))
    assert best == pytest.approx(TRUE, abs=0.05)


@pytest.mark.parametrize("n_flat", [1, 2, 3])
def test_a_minority_of_flat_curves_is_downweighted(n_flat):
    """Uninformative curves must not have their noise scaled up to unit variance.

    Plain z-scoring gives a flat, noisy curve the same say as an informative one.
    The shrinkage floor is what stops a few good images being outvoted by noise.
    """
    rng = np.random.default_rng(0)
    curves = [_curve(TWISTS, TRUE, height=0.1) for _ in range(3)]
    curves += [0.5 + rng.normal(0, 1e-4, TWISTS.size) for _ in range(n_flat)]
    best, _, _, weights = J.joint_best(curves, list(TWISTS))
    assert best == pytest.approx(TRUE, abs=0.05)
    assert min(weights[:3]) > 5 * max(
        weights[3:]
    ), f"informative curves not favoured: {weights}"


def test_a_majority_of_flat_curves_is_a_known_limitation():
    """Pin the documented failure so it is a known boundary, not a surprise.

    The floor is a fraction of the median spread, so once flat curves are the
    majority the median *is* the flat value and the floor stops protecting
    anything. Scaling to the largest spread instead would fix this and break
    the loud-outlier case, which matters more -- see the module docstring.
    """
    rng = np.random.default_rng(0)
    curves = [_curve(TWISTS, TRUE, height=0.1) for _ in range(3)]
    curves += [0.5 + rng.normal(0, 1e-4, TWISTS.size) for _ in range(4)]
    _, _, _, weights = J.joint_best(curves, list(TWISTS))
    assert max(weights[3:]) > 0.5, (
        "flat curves are now downweighted when in the majority -- if this was "
        "deliberate, drop this test and the caveat in the module docstring"
    )


def test_weights_report_which_images_carried_the_answer():
    curves = [_curve(TWISTS, TRUE, height=0.1), np.full(TWISTS.size, 0.5)]
    _, weights = J.combine_score_curves(curves)
    assert weights[0] > 0.8
    assert weights[1] < 0.05


def test_single_curve_is_just_its_own_argmax():
    best, _, _, _ = J.joint_best([_curve(TWISTS, TRUE)], list(TWISTS))
    assert best == pytest.approx(TRUE, abs=0.05)


def test_all_flat_does_not_crash_or_divide_by_zero():
    curves = [np.full(TWISTS.size, 0.5) for _ in range(3)]
    best, margin, combined, weights = J.joint_best(curves, list(TWISTS))
    assert best is not None
    assert np.all(np.isfinite(combined))
    assert margin == pytest.approx(0.0, abs=1e-9)


# ── result re-ranking, the shape the tab actually uses ──────────────────


def _result(score, label, twist, rise=4.75):
    """Mimic one ``process_one_task`` return value."""
    payload = (np.zeros((4, 4)), None, np.zeros((4, 4)), np.zeros((4, 4, 4)))
    params = (np.zeros((4, 4)), "file", label, 1.0, 1.0, twist, rise, 1, 0, 0, 0)
    return (score, payload, params)


def _results_for(curves, labels, twists):
    out = []
    for c, lab in zip(curves, labels):
        out += [_result(s, lab, t) for s, t in zip(c, twists)]
    return out


def test_rank_joint_can_overturn_the_first_image():
    """Ranking must be joint, not "first image wins"."""
    curves = [
        _curve(TWISTS, 1.0, height=0.08),
        _curve(TWISTS, TRUE, height=0.12),
        _curve(TWISTS, TRUE, height=0.10),
    ]
    results = _results_for(curves, ["A", "B", "C"], TWISTS)
    top = _rank(results, n_images=3)[0]
    assert top[2][5] == pytest.approx(TRUE, abs=0.05)
    # image A alone would have chosen 1.0
    solo = _rank([r for r in results if r[2][2] == "A"], n_images=1)[0]
    assert solo[2][5] == pytest.approx(1.0, abs=0.05)


def test_rank_single_image_is_unchanged():
    """The single-image path must behave exactly as before: sort by score."""
    results = [_result(s, "A", t) for s, t in zip(_curve(TWISTS, TRUE), TWISTS)]
    ranked = _rank(results, n_images=1)
    assert [r[0] for r in ranked] == sorted((r[0] for r in results), reverse=True)


def test_rank_ignores_pairs_not_scored_by_every_image():
    """A pair only one image managed to score must not win by default.

    Tasks can be skipped or raise, and a pair scored by one lucky image would
    otherwise be ranked against a single sample rather than a curve.
    """
    curves = [_curve(TWISTS, TRUE, height=0.1) for _ in range(2)]
    results = _results_for(curves, ["A", "B"], TWISTS)
    # a pair scored only by A, with an implausibly good score
    results.append(_result(99.0, "A", 5.0))
    ranked = _rank(results, n_images=2)
    assert all(r[2][5] != 5.0 for r in ranked), "orphan pair leaked into the ranking"
    assert ranked[0][2][5] == pytest.approx(TRUE, abs=0.05)


def test_rank_payload_comes_from_the_best_scoring_image():
    """The displayed reconstruction should be the best one for that pair."""
    results = [
        _result(0.4, "A", TRUE),
        _result(0.9, "B", TRUE),
        _result(0.3, "A", 1.0),
        _result(0.2, "B", 1.0),
    ]
    ranked = _rank(results, n_images=2)
    top = [r for r in ranked if r[2][5] == pytest.approx(TRUE)][0]
    assert top[2][2] == "B"


def test_rank_handles_empty_results():
    assert _rank([], n_images=3) == []


class TestPlacementsAsTransforms:
    """Joint placements handed to the manual stitch as a starting layout."""

    def _placed(self, psis):
        return [dict(psi=p) for p in psis]

    def test_shift_x_undoes_the_tiled_layout(self):
        from helicon.webApps.lib.denovo3d_align import phi_to_dx, period_pixel

        twist, rise, apix, width = 1.2, 4.75, 4.944, 128
        phis = [0.0, 40.0, 100.0, 250.0]
        out = J.placements_as_transforms(
            phis, self._placed([0.0] * 4), twist, rise, apix, width
        )
        period = period_pixel(twist, rise, apix, 1, False)
        want = np.mod([phi_to_dx(p, twist, rise, apix) for p in phis], period)
        base = want.min()
        got = [t["shift_x"] + base + i * width for i, t in enumerate(out)]
        assert np.allclose(got, want)

    def test_polarity_becomes_two_flips_not_a_big_rotation(self):
        """The manual card's rotation is bounded to +-90, so a reversed image
        has to travel as flip_x and flip_y -- which together are the same 180
        degree in-plane rotation."""
        out = J.placements_as_transforms(
            [0.0, 0.0], self._placed([178.0, -175.0]), 1.2, 4.75, 4.944, 128
        )
        for t in out:
            assert t["flip_x"] and t["flip_y"]
            assert abs(t["rotation"]) <= 5.0

    def test_an_upright_image_carries_no_flips(self):
        out = J.placements_as_transforms(
            [0.0], self._placed([3.5]), 1.2, 4.75, 4.944, 128
        )
        assert not out[0]["flip_x"] and not out[0]["flip_y"]
        assert out[0]["rotation"] == pytest.approx(3.5)

    def test_every_rotation_fits_the_manual_control(self):
        rng = np.random.default_rng(0)
        psis = list(rng.uniform(-180, 180, 40))
        out = J.placements_as_transforms(
            [0.0] * 40, self._placed(psis), 1.2, 4.75, 4.944, 128
        )
        assert all(abs(t["rotation"]) <= 90.0 for t in out)

    def test_every_image_is_placed(self):
        """Unlike the pairwise stitcher, nothing can be left unconnected: each
        image is placed against the model, not against its neighbours."""
        out = J.placements_as_transforms(
            [0.0, 90.0, 200.0], self._placed([0.0] * 3), 1.2, 4.75, 4.944, 128
        )
        assert all(t["connected"] for t in out)

    def test_two_fold_keeps_the_layout_inside_the_halved_period(self):
        from helicon.webApps.lib.denovo3d_align import period_pixel

        twist, rise, apix, width = 1.2, 4.75, 4.944, 128
        half = period_pixel(twist, rise, apix, 1, True)
        out = J.placements_as_transforms(
            [0.0, 90.0, 179.0],
            self._placed([0.0] * 3),
            twist,
            rise,
            apix,
            width,
            two_fold=True,
        )
        base_dx = [t["shift_x"] + i * width for i, t in enumerate(out)]
        assert max(base_dx) - min(base_dx) <= half + 1e-6


class TestGeometryFromResult:
    """The indices into a solver result, which are not self-evident.

    Wrong ones here would not raise; they would quietly place the images using
    some other pair of numbers, so each is pinned.
    """

    def _result(self):
        params = ("data", "f.mrcs", 3, 5.0, 4.944, 1.2, 4.746, 2, 0.0, 0.0, 0.0)
        return_data = ("xproj", "yproj", "zsec", None, 32, 30, 126, 4, None)
        return (0.87, return_data, params)

    def test_reads_the_helical_parameters(self):
        twist, rise, csym, apix2d, apix3d, _ = J.geometry_from_result(self._result())
        assert (twist, rise, csym) == (1.2, 4.746, 2)
        assert (apix2d, apix3d) == (4.944, 5.0)

    def test_reads_the_reconstruction_dimensions(self):
        *_, geometry = J.geometry_from_result(self._result())
        assert geometry["reconstruct_diameter_2d_pixel"] == 32
        assert geometry["reconstruct_diameter_3d_pixel"] == 30
        assert geometry["reconstruct_length_2d_pixel"] == 126
        assert geometry["reconstruct_length_3d_pixel"] == 4

    def test_scale_is_the_ratio_of_the_two_pixel_sizes(self):
        *_, geometry = J.geometry_from_result(self._result())
        assert geometry["scale2d_to_3d"] == pytest.approx(4.944 / 5.0)

    def test_solver_options_are_passed_through(self):
        *_, geometry = J.geometry_from_result(
            self._result(),
            sym_oversample=20,
            interpolation="nn",
            positive_constraint=1,
        )
        assert geometry["sym_oversample"] == 20
        assert geometry["interpolation"] == "nn"
        assert geometry["positive_constraint"] == 1

    def test_the_geometry_is_what_joint_reconstruct_accepts(self):
        import inspect

        from helicon.webApps.lib import denovo3d_jointsolve

        *_, geometry = J.geometry_from_result(self._result())
        accepted = set(
            inspect.signature(denovo3d_jointsolve.joint_reconstruct).parameters
        )
        assert set(geometry) <= accepted

    def test_a_scanned_result_carries_no_volume(self):
        """Which is why joint_refine has to accept seed_volume=None: the
        pipeline returns volumes only when a single pair is solved."""
        assert self._result()[1][3] is None


class TestPlacementComposite:
    def _placed(self, n):
        return [dict(psi=0.0) for _ in range(n)]

    def test_places_every_image_on_one_canvas(self):
        imgs = [np.random.default_rng(i).random((32, 128)) for i in range(4)]
        img, cov = J.placement_composite(
            imgs, [0.0, 90.0, 180.0, 270.0], self._placed(4), 1.2, 4.75, 5.0
        )
        assert img.ndim == 2
        assert np.asarray(cov).max() > 1  # they overlap, so some columns stack

    def test_layout_agrees_with_the_manual_stitch_transfer(self):
        """Both derive positions from the gauged azimuths, so the picture the
        search shows and the layout the manual stitch receives must match."""
        from helicon.webApps.lib.denovo3d_align import period_pixel, phi_to_dx

        phis = [0.0, 70.0, 150.0]
        twist, rise, apix = 1.2, 4.75, 5.0
        period = period_pixel(twist, rise, apix, 1, False)
        want = np.mod([phi_to_dx(p, twist, rise, apix) for p in phis], period)
        transforms = J.placements_as_transforms(
            phis, self._placed(3), twist, rise, apix, 128
        )
        got = [t["shift_x"] + want.min() + i * 128 for i, t in enumerate(transforms)]
        assert np.allclose(got, want)

    def test_a_two_fold_halves_the_spread(self):
        imgs = [np.random.default_rng(i).random((32, 128)) for i in range(3)]
        wide, _ = J.placement_composite(
            imgs, [0.0, 90.0, 180.0], self._placed(3), 1.2, 4.75, 5.0
        )
        half, _ = J.placement_composite(
            imgs, [0.0, 90.0, 180.0], self._placed(3), 1.2, 4.75, 5.0, two_fold=True
        )
        assert half.shape[1] < wide.shape[1]


class TestRankByProjectionMatching:
    def _results(self, twists, labels):
        out = []
        for t in twists:
            for j, label in enumerate(labels):
                params = ("img", "f.mrcs", label, 5.0, 5.0, t, 4.75, 1, 0.0, 0.0, 0.0)
                rd = ("x", "y", "z", None, 32, 32, 126, 4, None)
                out.append((0.5 + 0.01 * j, rd, params))
        return out

    def test_falls_back_to_a_plain_sort_with_one_image(self):
        results = self._results([1.1, 1.2], [3])
        ranked, composites = J.rank_by_projection_matching(results, [np.zeros((8, 8))])
        assert composites == {}
        assert [r[0] for r in ranked] == sorted([r[0] for r in results], reverse=True)

    def test_only_pairs_every_image_solved_are_ranked(self, monkeypatch):
        """Same rule combine_results uses: a pair scored by a subset would be
        compared against pairs scored by all of them."""
        results = self._results([1.1, 1.2], [3, 4])
        results = [r for r in results if not (r[2][5] == 1.2 and r[2][2] == 4)]
        seen = []

        def fake_refine(images, seed, twist, *a, **k):
            seen.append(round(twist, 3))
            return dict(
                phis=np.zeros(len(images)),
                placed=[dict(psi=0.0)] * len(images),
                two_fold=False,
                info=dict(score=0.9),
            )

        monkeypatch.setattr(J, "joint_refine", fake_refine)
        monkeypatch.setattr(J, "placement_composite", lambda *a, **k: (None, None))
        J.rank_by_projection_matching(results, [np.zeros((8, 8))] * 2)
        assert seen == [1.1]

    def test_a_failed_pair_is_dropped_not_scored_zero(self, monkeypatch):
        """Scoring it zero would rank it last, which is a claim about the twist
        rather than an admission that it was not measured."""
        results = self._results([1.1, 1.2], [3, 4])

        def fake_refine(images, seed, twist, *a, **k):
            if round(twist, 3) == 1.2:
                raise RuntimeError("solver blew up")
            return dict(
                phis=np.zeros(len(images)),
                placed=[dict(psi=0.0)] * len(images),
                two_fold=False,
                info=dict(score=0.9),
            )

        monkeypatch.setattr(J, "joint_refine", fake_refine)
        monkeypatch.setattr(J, "placement_composite", lambda *a, **k: (None, None))
        ranked, composites = J.rank_by_projection_matching(
            results, [np.zeros((8, 8))] * 2
        )
        assert [r[2][5] for r in ranked] == [1.1]
        assert 1.2 not in {p[0] for p in composites}

    def test_results_keep_the_pipeline_shape(self, monkeypatch):
        """So a caller can swap this in for combine_results unchanged."""
        results = self._results([1.1], [3, 4])

        monkeypatch.setattr(
            J,
            "joint_refine",
            lambda images, seed, twist, *a, **k: dict(
                phis=np.zeros(len(images)),
                placed=[dict(psi=0.0)] * len(images),
                two_fold=False,
                info=dict(score=0.77),
            ),
        )
        monkeypatch.setattr(J, "placement_composite", lambda *a, **k: (None, None))
        ranked, _ = J.rank_by_projection_matching(results, [np.zeros((8, 8))] * 2)
        assert len(ranked) == 1
        score, return_data, params = ranked[0]
        assert score == pytest.approx(0.77)
        assert len(params) == 11 and len(return_data) == 9


class TestCompositeAlignsToTheModel:
    """The composite and the model projection must share one canvas, or showing
    them together is misleading rather than useful."""

    def _placed(self, n):
        return [dict(psi=0.0) for _ in range(n)]

    def test_canvas_length_is_honoured(self):
        from helicon.webApps.lib.denovo3d_align import model_length_pixel

        imgs = [np.random.default_rng(i).random((32, 128)) for i in range(3)]
        twist, rise, apix = 1.2, 4.75, 5.0
        length = model_length_pixel(twist, rise, apix, 1, 128)
        img, cov = J.placement_composite(
            imgs,
            [0.0, 90.0, 200.0],
            self._placed(3),
            twist,
            rise,
            apix,
            canvas_length=length,
        )
        assert img.shape[1] == length
        assert np.asarray(cov).size == length

    def test_an_image_lands_where_its_azimuth_says(self):
        """Column dx of the composite must hold the image placed at dx, which
        is what makes it comparable with the model column for column."""
        from helicon.webApps.lib.denovo3d_align import (
            model_length_pixel,
            period_pixel,
            phi_to_dx,
        )

        twist, rise, apix = 1.2, 4.75, 5.0
        length = model_length_pixel(twist, rise, apix, 1, 64)
        one = np.ones((16, 64))
        phi = 120.0
        img, cov = J.placement_composite(
            [one],
            [phi],
            self._placed(1),
            twist,
            rise,
            apix,
            canvas_length=length,
        )
        # Modulo the period, as the placement itself is: an azimuth names a
        # position within one turn, not which turn.
        period = period_pixel(twist, rise, apix, 1, False)
        dx = int(round(phi_to_dx(phi, twist, rise, apix) % period))
        covered = np.flatnonzero(np.asarray(cov).ravel() > 0)
        assert abs(int(covered.min()) - dx) <= 2

    def test_without_a_canvas_length_it_is_unchanged(self):
        imgs = [np.random.default_rng(i).random((32, 64)) for i in range(2)]
        a, _ = J.placement_composite(imgs, [0.0, 90.0], self._placed(2), 1.2, 4.75, 5.0)
        assert a.shape[1] < 400  # tight to the images, not the whole period


class TestCentringDoesNotCutTheProjection:
    """Centring the composite must not be paid for by cutting the projection.

    Sliding a fixed-width projection blanks one edge and cuts the other; the
    projection has to be generated over the displaced window instead.
    """

    def test_centring_reports_a_shift_and_moves_only_the_composite(self):
        comp = np.zeros((8, 100))
        comp[:, 10:20] = 1.0
        cov = np.zeros(100)
        cov[10:20] = 1
        out, shift = J.center_on_coverage(comp, cov)
        assert shift == 50 - 14  # occupied midpoint 14 -> canvas midpoint 50
        moved = np.flatnonzero(out.any(axis=0))
        assert abs(int(moved.mean()) - 50) <= 1

    def test_no_coverage_means_no_shift(self):
        comp = np.zeros((8, 40))
        out, shift = J.center_on_coverage(comp, np.zeros(40))
        assert shift == 0 and out is comp

    def test_windowed_projection_is_full_width_at_both_edges(self, monkeypatch):
        """The window is cropped out of a wider projection, so every column
        holds real model -- unlike a translation, which leaves a blank strip."""
        made = {}

        def fake_long_side(
            vol, apix3d, twist, rise, csym, out_apix, ny, length_pixel, cpu=1
        ):
            made["length"] = length_pixel
            # a ramp, so a blank strip or a cut is detectable
            return np.tile(np.arange(1, length_pixel + 1, dtype=float), (ny, 1))

        import helicon.webApps.lib.denovo3d_align as A

        monkeypatch.setattr(A, "long_side_projection", fake_long_side)
        win = J._windowed_projection(
            np.zeros((4, 4, 4)), 5.0, 1.2, 4.75, 1, 5.0, 6, 100, shift=20
        )
        assert win.shape == (6, 100)
        assert made["length"] == 140  # widened by 2 * |shift|
        assert (win > 0).all()  # no blank strip anywhere

    def test_a_zero_shift_asks_for_exactly_the_width_wanted(self, monkeypatch):
        made = {}

        def fake_long_side(
            vol, apix3d, twist, rise, csym, out_apix, ny, length_pixel, cpu=1
        ):
            made["length"] = length_pixel
            return np.ones((ny, length_pixel))

        import helicon.webApps.lib.denovo3d_align as A

        monkeypatch.setattr(A, "long_side_projection", fake_long_side)
        J._windowed_projection(np.zeros((4, 4, 4)), 5.0, 1.2, 4.75, 1, 5.0, 6, 100)
        assert made["length"] == 100

    def test_the_window_follows_the_shift(self, monkeypatch):
        """A positive shift moves content right, so the window must be taken
        from further left -- the opposite direction."""

        def fake_long_side(
            vol, apix3d, twist, rise, csym, out_apix, ny, length_pixel, cpu=1
        ):
            return np.tile(np.arange(length_pixel, dtype=float), (ny, 1))

        import helicon.webApps.lib.denovo3d_align as A

        monkeypatch.setattr(A, "long_side_projection", fake_long_side)
        right = J._windowed_projection(
            np.zeros((4, 4, 4)), 5.0, 1.2, 4.75, 1, 5.0, 4, 60, shift=10
        )
        left = J._windowed_projection(
            np.zeros((4, 4, 4)), 5.0, 1.2, 4.75, 1, 5.0, 4, 60, shift=-10
        )
        assert right[0, 0] < left[0, 0]
