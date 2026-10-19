"""A class average that does not repeat at half a candidate repeat rules it out."""

import numpy as np
import pytest

import synthetic_pitch as syn

from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402

P, K = 400.0, 12


@pytest.fixture(scope="module")
def data():
    images, _ = syn.class_averages(P, K, apix=4.0, noise=0.02, nx=96)
    params = syn.class2d_table(P, K, n_fil=150, length=1200.0, seed=1)
    return params, {c + 1: images[c] for c in range(K)}


def _analyze(data, classes, use):
    params, imgs = data
    return ph.analyze(
        params,
        class_ids=classes,
        n_boot=2,
        image_apix=4.0,
        class_images=imgs if use else None,
        use_refined_path=False,
        max_sep=1500.0,
    )


class TestGate:
    def test_a_period_the_gate_zeroes_cannot_win(self, data):
        params, _ = data
        pairs = ph.prepare_pairs(
            params, class_ids=list(range(1, 7)), use_refined_path=False, max_sep=1500.0
        )
        plain = ph.estimate_period(pairs)
        assert plain["period"] == pytest.approx(P / 2, rel=0.03)  # half the views
        gated = ph.estimate_period(pairs, gate=lambda p: np.where(p < 300.0, 0.0, 1.0))
        assert gated["period"] > 300.0
        assert gated["ungated_period"] == pytest.approx(plain["period"], rel=0.01)

    def test_without_a_gate_nothing_changes(self, data):
        params, _ = data
        pairs = ph.prepare_pairs(params, use_refined_path=False, max_sep=1500.0)
        est = ph.estimate_period(pairs)
        assert est["ungated_period"] == pytest.approx(est["period"], rel=0.01)


class TestAnalyze:
    def test_half_the_views_make_half_the_repeat_without_the_veto(self, data):
        r = _analyze(data, list(range(1, 7)), use=False)
        assert r.period == pytest.approx(P / 2, rel=0.03)
        assert r.double_ratio > 0.8 and r.vetoed_period is None

    def test_the_veto_gives_the_repeat_back_and_says_what_it_ruled_out(self, data):
        r = _analyze(data, list(range(1, 7)), use=True)
        assert r.period == pytest.approx(P, rel=0.02)
        assert r.vetoed_period == pytest.approx(P / 2, rel=0.03)

    def test_the_scan_without_the_gate_is_kept_for_the_plot(self, data):
        r = _analyze(data, list(range(1, 7)), use=True)
        short = r.scan_periods < P * 0.6
        assert r.scan_scores[short].max() < r.scan_scores_ungated[short].max()
        assert r.scan_scores_ungated[short].max() > 0.5 * r.scan_scores.max()
        assert _analyze(data, list(range(1, 7)), use=False).scan_scores_ungated is None

    def test_a_complete_selection_is_left_alone(self, data):
        plain = _analyze(data, list(range(1, 13)), use=False)
        gated = _analyze(data, list(range(1, 13)), use=True)
        assert gated.period == pytest.approx(plain.period, rel=0.002)
        assert gated.vetoed_period is None

    def test_a_selection_with_gaps_in_the_ring_is_left_alone_when_it_was_right(
        self, data
    ):
        sel = [1, 2, 3, 4, 5, 6, 8, 10]
        plain = _analyze(data, sel, use=False)
        gated = _analyze(data, sel, use=True)
        assert plain.period == pytest.approx(P, rel=0.02)
        assert gated.period == pytest.approx(plain.period, rel=0.002)

    def test_a_failing_check_does_not_stop_the_estimate(self, data):
        params, imgs = data
        bad = {c: np.zeros((4, 4), np.float32) for c in imgs}  # too small to register
        r = ph.analyze(
            params,
            class_ids=list(range(1, 13)),
            n_boot=2,
            image_apix=4.0,
            class_images=bad,
            use_refined_path=False,
            max_sep=1500.0,
        )
        assert r.period == pytest.approx(P, rel=0.02)


class TestShortRepeat:
    """The gate is the averages' own self-similarity, not a fixed length: a fast
    twisting helix with a short repeat keeps its repeat."""

    @pytest.mark.parametrize("period", [150.0, 250.0])
    def test_a_short_repeat_is_not_ruled_out(self, period):
        images, _ = syn.class_averages(period, K, apix=4.0, noise=0.02, nx=96)
        params = syn.class2d_table(period, K, n_fil=150, length=1200.0, seed=1)
        r = ph.analyze(
            params,
            class_ids=list(range(1, K + 1)),
            n_boot=2,
            image_apix=4.0,
            class_images={c + 1: images[c] for c in range(K)},
            use_refined_path=False,
            max_sep=1500.0,
        )
        assert r.period == pytest.approx(period, rel=0.03)
        assert r.vetoed_period is None
