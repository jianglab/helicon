"""Tests for the two-ring fit (helical_pitch_mixture) on synthetic mixtures.

Two helical types that share their classes: each filament's segments are labelled
by the sector of its own type's repeat they sit at.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from joint_pixel import synthetic as syn  # noqa: E402

from helicon.webApps.lib import helical_pitch_mixture as mx  # noqa: E402
from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402

K = 12


def _pairs(periods, share=0.5, length=1200.0, mislabel=0.0, n_fil=160, seed=0):
    params, types = syn.class2d_table_mixture(
        periods, (1 - share, share), K, n_fil, length, mislabel=mislabel, seed=seed
    )
    pairs = ph.prepare_pairs(params, use_refined_path=False, max_sep=2000.0)
    truth = np.array(
        [types[int(k[0][3:6]) * 10 + k[1] - 1] for k in pairs.filament_keys]
    )
    return pairs, truth


def _fit(pairs, seed=0, restarts=4):
    est = ph.estimate_period(pairs)
    est, _ = ph.sync_filament_directions(pairs, est)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        lab = rng.integers(0, 2, pairs.n_filaments)
        two = mx.fit_two_rings(pairs, lab, (est["period"] * 0.93, est["period"] * 1.07))
        if two is not None and (best is None or two["objective"] > best["objective"]):
            best = two
    return est, best


class TestFilamentScores:
    def test_a_filament_agrees_with_its_own_ring_and_not_with_another(self):
        pairs, truth = _pairs((400.0, 400.0), length=800.0)
        est = ph.estimate_period(pairs)
        keep, swap, weight = mx.filament_scores(pairs, est["period"], est["phases"])
        assert keep.shape == swap.shape == weight.shape == (pairs.n_filaments,)
        assert np.mean(np.maximum(keep, swap)) > 0.8
        # a repeat that is not this one agrees much less
        keep2, swap2, _ = mx.filament_scores(pairs, est["period"] * 1.25, est["phases"])
        assert np.mean(np.maximum(keep2, swap2)) < np.mean(np.maximum(keep, swap))


class TestTwoTypes:
    def test_two_types_sharing_their_classes_are_found(self):
        pairs, truth = _pairs((400.0, 440.0))
        est, two = _fit(pairs)
        assert est["period"] == pytest.approx(420.0, rel=0.06)  # a blend
        lo, hi = np.sort(two["periods"])
        assert lo == pytest.approx(400.0, rel=0.02)
        assert hi == pytest.approx(440.0, rel=0.02)
        accuracy = max(np.mean(two["labels"] == truth), np.mean(two["labels"] != truth))
        assert accuracy > 0.95

    def test_they_fit_far_better_than_one_ring(self):
        pairs, _ = _pairs((400.0, 480.0))
        est, two = _fit(pairs)
        assert mx.objective_gain(pairs, two, est["period"], est["phases"]) > 0.1

    def test_a_minor_type_of_a_fifth_is_found_too(self):
        pairs, _ = _pairs((400.0, 460.0), share=0.2, n_fil=240)
        _, two = _fit(pairs)
        assert np.sort(two["periods"])[1] == pytest.approx(460.0, rel=0.03)

    def test_filaments_are_given_to_a_ring_either_way_round(self):
        pairs, truth = _pairs((400.0, 440.0))
        pairs.D = np.where(
            np.random.default_rng(1).random(pairs.n_filaments)[pairs.pair_filament]
            < 0.5,
            -pairs.D,
            pairs.D,
        )
        _, two = _fit(pairs)
        lo, hi = np.sort(two["periods"])
        assert lo == pytest.approx(400.0, rel=0.03)
        assert hi == pytest.approx(440.0, rel=0.03)


class TestOneType:
    def test_one_type_is_not_split_into_two(self):
        pairs, _ = _pairs((400.0, 400.0))
        est, two = _fit(pairs)
        lo, hi = np.sort(two["periods"])
        assert hi / lo - 1.0 < 0.01
        assert mx.objective_gain(pairs, two, est["period"], est["phases"]) < 0.01


class TestHeldOutGain:
    def _gain(self, periods):
        pairs, _ = _pairs(periods, length=1600.0)
        est = ph.estimate_period(pairs)
        est, _ = ph.sync_filament_directions(pairs, est)
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, pairs.n_filaments)
        return mx.heldout_gain(
            pairs,
            labels,
            (est["period"] * 0.93, est["period"] * 1.07),
            est["period"],
        )

    def test_two_types_gain_on_filaments_the_rings_were_not_fitted_to(self):
        assert self._gain((400.0, 480.0)) > 0.1

    def test_one_type_gains_nothing_on_held_out_filaments(self):
        assert abs(self._gain((400.0, 400.0))) < 0.01

    def test_train_filaments_alone_fit_the_rings(self):
        pairs, truth = _pairs((400.0, 480.0), length=1600.0)
        train = np.arange(pairs.n_filaments) % 2 == 0
        labels = truth.copy()
        two = mx.fit_two_rings(pairs, labels, (400.0, 480.0), train=train)
        assert sorted(np.round(two["periods"] / 10) * 10) == [400.0, 480.0]
        # the held-out filaments were still given a ring
        assert len(two["best"]) == pairs.n_filaments and np.all(two["best"] > 0.5)


class TestStartingLabels:
    def test_the_filaments_are_split_at_the_median_repeat(self):
        pairs, _ = _pairs((400.0, 440.0), n_fil=40)
        pitch = np.where(np.arange(pairs.n_filaments) % 2 == 0, 400.0, 440.0)
        labels = mx.starting_labels(pairs, pitch, np.random.default_rng(0))
        assert np.array_equal(labels, (pitch > np.median(pitch)).astype(int))

    def test_without_repeats_it_starts_at_random(self):
        pairs, _ = _pairs((400.0, 440.0), n_fil=40)
        labels = mx.starting_labels(
            pairs, np.full(pairs.n_filaments, np.nan), np.random.default_rng(0)
        )
        assert set(labels) <= {0, 1} and 0 < labels.mean() < 1

    def test_a_ring_needs_filaments(self):
        pairs, _ = _pairs((400.0, 440.0), n_fil=40)
        assert (
            mx.fit_two_rings(pairs, np.zeros(pairs.n_filaments, int), (400.0, 440.0))
            is None
        )


def _switching(n_fil=60, seed=0, noise=0.0):
    """Filaments of two known rings (400 and 440 A); half switch type midway."""
    rng = np.random.default_rng(seed)
    import pandas as pd

    rows, truth, switching = [], [], []
    step, n_seg = 20.0, 80
    for f in range(n_fil):
        track = np.arange(n_seg) * step
        moves = f % 2 == 1
        switch = rng.integers(25, 55) if moves else n_seg + 1
        first = int(rng.integers(0, 2))
        typ = np.where(np.arange(n_seg) < switch, first, 1 - first)
        off = rng.random(2)
        periods = np.where(typ == 0, 400.0, 440.0)
        phase = (track / periods + off[typ]) % 1.0
        cls = np.minimum((phase * K).astype(int), K - 1)
        bad = rng.random(n_seg) < noise
        cls[bad] = rng.integers(0, K, int(bad.sum()))
        for i in range(n_seg):
            rows.append(
                dict(
                    rlnMicrographName=f"mic{f // 10:03d}.mrc",
                    rlnHelicalTubeID=f % 10 + 1,
                    rlnHelicalTrackLengthAngst=track[i],
                    rlnClassNumber=int(cls[i]) + 1,
                    rlnAnglePsi=0.0,
                    rlnAnglePsiPrior=0.0,
                    rlnCoordinateX=1000.0 + track[i],
                    rlnCoordinateY=500.0,
                    rlnOriginXAngst=0.0,
                    rlnOriginYAngst=0.0,
                    rlnMicrographOriginalPixelSize=1.0,
                )
            )
        truth.append(typ)
        switching.append(np.full(n_seg, moves))
    pairs = ph.prepare_pairs(pd.DataFrame(rows), use_refined_path=False, max_sep=2000.0)
    sector = (np.arange(K) + 0.5) / K
    rings = [(400.0, -2 * np.pi * sector), (440.0, -2 * np.pi * sector)]
    return (
        pairs,
        rings,
        np.concatenate(truth)[pairs.seg_row],
        np.concatenate(switching)[pairs.seg_row],
    )


class TestSegmentStates:
    @pytest.fixture(scope="class")
    def data(self):
        pairs, rings, truth, switching = _switching()
        scores, _ = mx.segment_scores(pairs, rings, window=300.0)
        return pairs, rings, truth, switching, scores

    def test_the_scores_have_one_row_per_ring(self, data):
        pairs, rings, _, _, scores = data
        assert scores.shape == (2, len(pairs.seg_class))
        assert np.all(np.abs(scores) <= 1.0 + 1e-9)

    def test_a_whole_filament_is_one_state_when_switching_is_too_costly(self, data):
        pairs, _, _, _, scores = data
        state = mx.assign_segments(pairs, scores, switch_cost=1e9, junk=None)
        for f in range(pairs.n_filaments):
            assert len(set(state[pairs.seg_filament == f])) == 1

    def test_filaments_that_never_switch_get_their_ring(self, data):
        pairs, _, truth, switching, scores = data
        state = mx.assign_segments(pairs, scores, switch_cost=1e9, junk=None)
        steady = ~switching
        assert np.mean(state[steady] == truth[steady]) > 0.95

    def test_a_filament_that_switches_is_followed_when_switching_is_cheap(self, data):
        pairs, _, truth, switching, scores = data
        whole = mx.assign_segments(pairs, scores, switch_cost=1e9, junk=None)
        follow = mx.assign_segments(pairs, scores, switch_cost=0.5, junk=None)
        moving = switching
        assert (
            np.mean(follow[moving] == truth[moving])
            > np.mean(whole[moving] == truth[moving]) + 0.1
        )

    def test_a_stretch_that_no_ring_explains_goes_to_junk(self):
        pairs, rings, truth, _ = _switching(n_fil=20, noise=1.0)
        scores, _ = mx.segment_scores(pairs, rings, window=800.0)
        state = mx.assign_segments(pairs, scores, switch_cost=2.0, junk=0.3)
        assert np.mean(state == 2) > 0.7  # labels at random: the junk state wins

    def test_the_states_are_rings_or_junk(self, data):
        pairs, _, _, _, scores = data
        state = mx.assign_segments(pairs, scores, switch_cost=2.0, junk=0.3)
        assert set(np.unique(state)) <= {0, 1, 2}
