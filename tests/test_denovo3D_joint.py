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
