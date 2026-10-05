"""Tests for the pixel term of the class ring (helical_pitch_pixel).

Everything is built in memory from a known helix: class averages are windows of
its side view, and segments are labelled by the sector of the repeat they sit
at, so the repeat and every class phase are known.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from joint_pixel import synthetic as syn  # noqa: E402

from helicon.webApps.lib import helical_pitch_phase as ph  # noqa: E402
from helicon.webApps.lib import helical_pitch_pixel as px  # noqa: E402

PERIOD, K, APIX = 400.0, 8, 4.0


@pytest.fixture(scope="module")
def averages():
    return syn.class_averages(PERIOD, K, apix=APIX, noise=0.02)


@pytest.fixture(scope="module")
def offsets(averages):
    images, _ = averages
    return px.class_pixel_offsets(images, list(range(K)), APIX)


@pytest.fixture(scope="module")
def pairs():
    params = syn.class2d_table(PERIOD, K, n_fil=60, length=600.0, seed=1)
    return ph.prepare_pairs(params, use_refined_path=False)


def _wrap(x, period=PERIOD):
    return (x + period / 2.0) % period - period / 2.0


class TestPairWeight:
    def test_a_weak_correlation_gets_no_weight(self):
        w = px.pair_weight([0.2, 0.9], [5.0, 5.0], min_corr=0.3)
        assert w[0] == 0.0 and w[1] == pytest.approx(0.81)

    def test_a_flat_profile_is_dropped_when_asked(self):
        w = px.pair_weight([0.9, 0.9], [1.0, 5.0], min_prominence=3.0)
        assert w[0] == 0.0 and w[1] > 0.0

    def test_the_prominence_is_not_used_by_default(self):
        assert px.pair_weight([0.9], [0.0])[0] > 0.0


class TestRegisteredOffsets:
    def test_every_pair_is_registered_once(self, offsets):
        assert len(offsets.dx) == K * (K - 1) // 2
        assert np.all(offsets.a < offsets.b)

    def test_the_offset_is_the_phase_difference_modulo_the_repeat(
        self, averages, offsets
    ):
        _, phase_true = averages
        expected = phase_true[offsets.b] - phase_true[offsets.a]
        residual = _wrap(offsets.dx - expected)
        # the best sign convention is a plain +1: shifting b by dx brings it onto a
        assert np.max(np.abs(residual)) < 0.03 * PERIOD
        assert np.median(np.abs(residual)) < 0.01 * PERIOD

    def test_a_single_class_has_nothing_to_register(self, averages):
        images, _ = averages
        found = px.class_pixel_offsets(images, [0], APIX)
        assert len(found.dx) == 0 and found.n_classes == 1


class TestPixelTerm:
    def test_it_is_hermitian(self, offsets):
        H = px.pixel_matrices(offsets, np.array([380.0, 400.0]))
        assert np.allclose(H, np.conj(np.transpose(H, (0, 2, 1))))

    def test_alone_it_cannot_tell_the_repeat_from_its_half(self, pairs, offsets):
        # offsets are known modulo the repeat, and what is consistent modulo P
        # is consistent modulo P / 2
        hist, centers = ph.class_pair_histogram(pairs)
        both = np.array([PERIOD / 2, PERIOD])
        scores, _ = px.joint_scan(hist, centers, offsets, both, alpha=1.0)
        assert scores[0] == pytest.approx(scores[1], abs=0.05)

    def test_the_phases_are_those_of_the_classes(self, averages, pairs, offsets):
        _, phase_true = averages
        est = px.estimate_period_joint(pairs, offsets, alpha=1.0)
        assert syn.phase_error(est["phases"], phase_true, est["period"]) < 0.02


class TestMetadataTermAgreesWithPhaseScan:
    def test_alpha_zero_is_the_existing_ring(self, pairs, offsets):
        hist, centers = ph.class_pair_histogram(pairs)
        periods = np.arange(300.0, 500.0, 5.0)
        mine, _ = px.joint_scan(hist, centers, offsets, periods, alpha=0.0)
        theirs, _ = ph.phase_scan(hist, centers, periods)
        assert np.allclose(mine, theirs, atol=1e-9)


class TestJointEstimate:
    def test_the_repeat_is_found(self, pairs, offsets):
        est = px.estimate_period_joint(pairs, offsets, alpha=0.5)
        assert est["period"] == pytest.approx(PERIOD, rel=0.02)


class TestSelfSimilarity:
    @pytest.fixture(scope="class")
    def similarity(self, averages):
        images, _ = averages
        return px.self_similarity(images, list(range(K)), APIX)

    def test_the_averages_do_not_repeat_at_half_the_repeat(self, similarity):
        assert similarity.gate([PERIOD / 2])[0] < 0.5

    def test_a_lag_too_long_to_observe_is_not_judged(self, similarity):
        assert similarity.lag[-1] < PERIOD
        assert similarity.gate([PERIOD])[0] == 1.0

    def test_no_classes_gate_nothing(self):
        empty = px.SelfSimilarity(np.zeros(0), np.zeros(0))
        assert np.all(empty.gate([100.0, 400.0]) == 1.0)


@pytest.fixture(scope="module")
def similarity(averages):
    images, _ = averages
    return px.self_similarity(images, list(range(K)), APIX)


class TestShortFilaments:
    @pytest.fixture(scope="class")
    def short(self):
        # filaments of 0.4 of a repeat: no pair spans a repeat, so the segment
        # pairs cannot tell the repeat from its half, and what they do give is
        # biased low
        params = syn.class2d_table(PERIOD, K, n_fil=120, length=160.0, seed=2)
        found = ph.prepare_pairs(params, use_refined_path=False)
        assert found.W[np.abs(found.D) > PERIOD].sum() == 0
        return found

    def test_the_metadata_alone_is_far_off(self, short, offsets):
        est = px.estimate_period_joint(short, offsets, alpha=0.0)
        assert abs(est["period"] / PERIOD - 1.0) > 0.1

    def test_the_joint_estimate_alone_can_take_the_half(self, short, offsets):
        est = px.estimate_period_joint(short, offsets, alpha=0.5)
        assert est["period"] == pytest.approx(PERIOD / 2, rel=0.05)

    def test_the_self_similarity_gate_finds_the_basin_of_the_repeat(
        self, short, offsets, similarity
    ):
        est = px.estimate_period_joint(short, offsets, 0.5, similarity=similarity)
        assert est["period"] == pytest.approx(PERIOD, rel=0.08)

    def test_the_adaptive_estimate_falls_back_to_the_joint_one(
        self, short, offsets, similarity
    ):
        est = px.estimate_period_adaptive(short, offsets, similarity)
        assert est["stage"] == "joint"
        assert est["period"] == pytest.approx(PERIOD, rel=0.08)


class TestLongFilaments:
    def test_gated_estimate_is_as_sharp_as_the_metadata(
        self, pairs, offsets, similarity
    ):
        meta = px.estimate_period_joint(pairs, offsets, alpha=0.0)
        est = px.estimate_period_gated(pairs, offsets, similarity)
        assert est["period"] == pytest.approx(meta["period"], rel=0.002)
        assert est["coarse_period"] == pytest.approx(PERIOD, rel=0.03)

    def test_gating_keeps_a_sub_multiple_from_winning(self, pairs, offsets, similarity):
        # filaments of 1.5 repeats leave the half repeat a close second
        params = syn.class2d_table(PERIOD, K, n_fil=60, length=600.0, seed=3)
        found = ph.prepare_pairs(params, use_refined_path=False)
        est = px.estimate_period_gated(found, offsets, similarity)
        assert est["period"] == pytest.approx(PERIOD, rel=0.03)

    def test_the_adaptive_estimate_takes_the_metadata_when_it_agrees(
        self, pairs, offsets, similarity
    ):
        meta = px.estimate_period_joint(pairs, offsets, alpha=0.0)
        est = px.estimate_period_adaptive(pairs, offsets, similarity)
        assert est["stage"] == "metadata"
        assert est["period"] == pytest.approx(meta["period"], rel=0.005)


class TestPoorAssignments:
    def test_the_class_phases_hold_up_when_most_labels_are_wrong(self):
        # 60% of the segments in a random class, and the averages smeared by them
        images, phase_true = syn.class_averages(
            PERIOD, K, apix=APIX, noise=0.02, contamination=0.6
        )
        params = syn.class2d_table(PERIOD, K, n_fil=100, length=1200.0, mislabel=0.6)
        found = ph.prepare_pairs(params, use_refined_path=False)
        offs = px.class_pixel_offsets(images, list(range(K)), APIX)
        sim = px.self_similarity(images, list(range(K)), APIX)
        meta = px.estimate_period_joint(found, offs, alpha=0.0)
        both = px.estimate_period_gated(found, offs, sim)
        e_meta = syn.phase_error(meta["phases"], phase_true, meta["period"])
        e_both = syn.phase_error(both["phases"], phase_true, both["period"])
        assert e_both < 0.01
        assert e_both < e_meta


class TestPixelSign:
    def test_the_offsets_run_the_same_way_as_the_ring(self, pairs, offsets):
        est = px.estimate_period_joint(pairs, offsets, alpha=0.5)
        sign, (plus, minus) = px.pixel_sign(offsets, est["phases"], est["period"])
        assert sign == 1.0 and plus < minus

    def test_offsets_in_the_opposite_direction_are_found_out(self, pairs, offsets):
        est = px.estimate_period_joint(pairs, offsets, alpha=0.5)
        flipped = px.PixelOffsets(
            offsets.a,
            offsets.b,
            -offsets.dx,
            offsets.corr,
            offsets.prominence,
            offsets.weight,
            offsets.n_classes,
        )
        sign, _ = px.pixel_sign(flipped, est["phases"], est["period"])
        assert sign == -1.0


class TestAgreementDiagnostic:
    def test_a_consistent_set_has_small_residuals(self, pairs, offsets):
        est = px.estimate_period_joint(pairs, offsets, alpha=0.5)
        residual, per_class = px.pixel_residuals(offsets, est["phases"], est["period"])
        assert np.nanmax(np.abs(residual)) < 0.05 * PERIOD
        assert np.all(per_class < 0.02 * PERIOD)

    def test_a_class_with_a_wrong_offset_is_singled_out(self, pairs, offsets):
        est = px.estimate_period_joint(pairs, offsets, alpha=0.5)
        bad = px.PixelOffsets(
            offsets.a,
            offsets.b,
            np.where(
                (offsets.a == 3) | (offsets.b == 3), offsets.dx + 90.0, offsets.dx
            ),
            offsets.corr,
            offsets.prominence,
            offsets.weight,
            offsets.n_classes,
        )
        _, per_class = px.pixel_residuals(bad, est["phases"], est["period"])
        assert int(np.nanargmax(per_class)) == 3


class TestClassPixelCheck:
    @pytest.fixture(scope="class")
    def ring(self, pairs, offsets):
        return px.estimate_period_joint(pairs, offsets, alpha=0.5)

    def test_classes_that_sit_where_the_ring_puts_them_agree(self, averages, ring):
        images, _ = averages
        check = px.class_pixel_check(
            images, np.zeros(K), ring["phases"], ring["period"], APIX
        )
        assert check is not None
        assert check.agree().all() and not check.disagree().any()
        assert check.sign == 1.0

    def test_a_class_that_does_not_sit_where_the_ring_puts_it_is_flagged(
        self, averages, ring
    ):
        images, _ = averages
        # swap two classes' averages: their segments say one thing, the pixels another
        moved = list(images)
        moved[1], moved[5] = images[5], images[1]
        check = px.class_pixel_check(
            moved, np.zeros(K), ring["phases"], ring["period"], APIX
        )
        assert check.disagree()[1] and check.disagree()[5]
        assert check.agree().sum() >= K - 4

    def test_fewer_than_three_classes_have_no_ring_to_check(self, averages, ring):
        images, _ = averages
        assert (
            px.class_pixel_check(
                images[:2], np.zeros(2), ring["phases"][:2], ring["period"], APIX
            )
            is None
        )

    def test_the_cutoff_is_a_share_of_the_repeat(self, averages, ring):
        images, _ = averages
        check = px.class_pixel_check(
            images, np.zeros(K), ring["phases"], ring["period"], APIX
        )
        assert check.cutoff == pytest.approx(px.DISAGREE_FRACTION * ring["period"])

    def test_a_class_without_a_registered_pair_is_neither(self):
        check = px.PixelCheck(np.array([np.nan, 10.0, 90.0]), 1.0, 3, 50.0)
        assert list(check.agree()) == [False, True, False]
        assert list(check.disagree()) == [False, False, True]


class TestPixelAgreement:
    def test_it_is_one_where_every_offset_is_where_the_ring_puts_it(
        self, averages, offsets, pairs
    ):
        est = px.estimate_period_joint(pairs, offsets, alpha=0.5)
        ag = px.pixel_agreement(offsets, est["phases"], est["period"])
        assert ag.shape == (K,)
        assert np.all((ag >= 0) & (ag <= 1))
        assert ag.min() > 0.9

    def test_a_class_with_a_wrong_offset_scores_low(self, offsets, pairs):
        est = px.estimate_period_joint(pairs, offsets, alpha=0.5)
        bad = px.PixelOffsets(
            offsets.a,
            offsets.b,
            np.where(
                (offsets.a == 3) | (offsets.b == 3), offsets.dx + 100.0, offsets.dx
            ),
            offsets.corr,
            offsets.prominence,
            offsets.weight,
            offsets.n_classes,
        )
        ag = px.pixel_agreement(bad, est["phases"], est["period"])
        assert int(np.nanargmin(ag)) == 3 and ag[3] < ag[np.arange(K) != 3].min() - 0.1

    def test_a_class_without_a_pair_is_nan_and_the_scale_is_clipped(self):
        offs = px.PixelOffsets(
            np.array([0]),
            np.array([1]),
            np.array([0.0]),
            np.array([0.9]),
            np.array([5.0]),
            np.array([0.81]),
            3,
        )
        ag = px.pixel_agreement(offs, np.zeros(3), 400.0)
        assert ag[0] == pytest.approx(1.0) and ag[1] == pytest.approx(1.0)
        assert np.isnan(ag[2])
        # a residual of half the repeat is -1 for a cos, clipped to 0
        far = px.PixelOffsets(
            np.array([0]),
            np.array([1]),
            np.array([200.0]),
            np.array([0.9]),
            np.array([5.0]),
            np.array([0.81]),
            2,
        )
        assert px.pixel_agreement(far, np.zeros(2), 400.0)[0] == 0.0

    def test_the_check_carries_it(self, averages):
        images, _ = averages
        est = px.estimate_period_joint(
            ph.prepare_pairs(
                syn.class2d_table(PERIOD, K, n_fil=60, length=600.0, seed=1),
                use_refined_path=False,
            ),
            px.class_pixel_offsets(images, list(range(K)), APIX),
            alpha=0.5,
        )
        check = px.class_pixel_check(
            images, np.zeros(K), est["phases"], est["period"], APIX
        )
        assert check.agreement.shape == (K,) and np.all(check.agreement > 0.9)


class TestAgreementReason:
    def _result(self, agreement):
        from types import SimpleNamespace

        import numpy as np

        check = SimpleNamespace(agreement=np.array(agreement))
        return SimpleNamespace(
            class_ids=[1, 2, 3, 4],
            class_fit=np.array([0.9, 0.9, 0.9, 0.9]),
            class_evidence=None,
            merged_into=None,
            pixel_check=check,
        )

    def test_a_class_below_the_threshold_is_picked_and_nan_is_not_judged(self):
        from helicon.webApps.lib import helical_pitch_phase as phase

        r = self._result([0.9, 0.3, float("nan"), 0.6])
        out = phase.diagnose_classes(r, agreement_threshold=0.5)
        assert out == {2: phase.PIXELS_DISAGREE}

    def test_zero_picks_none(self):
        from helicon.webApps.lib import helical_pitch_phase as phase

        r = self._result([0.0, 0.3, 0.5, 0.6])
        assert phase.diagnose_classes(r, agreement_threshold=0.0) == {}


class TestCandidateAgreement:
    """The agreement an unselected class would have, as the suggestions use it."""

    @pytest.fixture(scope="class")
    def ring(self, averages):
        images, phase_a = averages
        phases = 2 * np.pi * np.asarray(phase_a) / PERIOD
        n = K - 2  # the last two are the candidates
        off = px.class_pixel_offsets(images[:n], list(range(n)), APIX)
        sign, _ = px.pixel_sign(off, phases[:n], PERIOD)
        return images, phases, n, sign

    def _agree(self, ring, cand_phases):
        images, phases, n, sign = ring
        return px.candidate_agreement(
            images[:n],
            [0.0] * n,
            phases[:n],
            PERIOD,
            sign,
            APIX,
            images[n:],
            [0.0] * (K - n),
            cand_phases,
        )

    def test_a_candidate_at_its_true_azimuth_agrees(self, ring):
        a = self._agree(ring, ring[1][ring[2] :])
        assert a.shape == (2,) and np.all(a > 0.8)

    def test_a_candidate_half_a_repeat_away_does_not(self, ring):
        a = self._agree(ring, ring[1][ring[2] :] + np.pi)
        assert np.all(a < 0.3)

    def test_only_the_asked_pairs_are_registered(self, averages):
        images, _ = averages
        off = px.class_pixel_offsets(
            images, list(range(K)), APIX, pairs=[(0, 1), (0, 2)]
        )
        assert list(off.a) == [0, 0] and list(off.b) == [1, 2]

    def test_without_candidates_it_is_empty(self, ring):
        images, phases, n, sign = ring
        out = px.candidate_agreement(
            images[:n], [0.0] * n, phases[:n], PERIOD, sign, APIX, [], [], []
        )
        assert len(out) == 0
