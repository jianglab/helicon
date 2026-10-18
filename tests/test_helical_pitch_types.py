"""Tests for the grouping of classes into helical types (helical_pitch_types).

Two helices of the same repeat but different structure: the class averages of one
are windows of its side view, so they register on each other well, and not so
well on those of the other. Filaments use the classes of their own type.
"""

import numpy as np
import pandas as pd
import pytest

import synthetic_pitch as syn

from helicon.webApps.lib import helical_pitch_types as ty  # noqa: E402

PERIOD, K, APIX = 400.0, 6, 4.0


@pytest.fixture(scope="module")
def two_types():
    a, _ = syn.class_averages(PERIOD, K, apix=APIX, noise=0.03, structure_seed=0)
    b, _ = syn.class_averages(PERIOD, K, apix=APIX, noise=0.03, structure_seed=7)
    return a + b  # classes 0..K-1 are type A, K..2K-1 type B


@pytest.fixture(scope="module")
def similarity(two_types):
    # these synthetic structures differ only a little, and the body of the filament
    # hides it from the plain correlation, so the modulation is compared
    return ty.class_similarity(two_types, list(range(2 * K)), APIX, modulation=True)


class TestClassSimilarity:
    def test_it_is_a_symmetric_matrix_with_ones_on_the_diagonal(self, similarity):
        assert similarity.shape == (2 * K, 2 * K)
        assert np.allclose(similarity, similarity.T)
        assert np.allclose(np.diag(similarity), 1.0)
        assert similarity.min() >= 0.0 and similarity.max() <= 1.0

    def test_classes_of_one_structure_look_more_alike_than_those_of_two(
        self, similarity
    ):
        same = np.concatenate(
            [
                similarity[:K, :K][np.triu_indices(K, 1)],
                similarity[K:, K:][np.triu_indices(K, 1)],
            ]
        )
        different = similarity[:K, K:].ravel()
        assert same.mean() > different.mean() + 0.1
        assert same.min() > different.max() - 0.05  # nearly separable


class TestModulation:
    def test_removing_the_axial_mean_widens_the_gap_but_is_not_the_default(
        self, two_types
    ):
        import inspect

        assert (
            inspect.signature(ty.class_similarity).parameters["modulation"].default
            is False
        )
        plain = ty.class_similarity(two_types, list(range(2 * K)), APIX)
        mod = ty.class_similarity(two_types, list(range(2 * K)), APIX, modulation=True)
        upper = np.triu_indices(K, 1)
        gap_plain = plain[:K, :K][upper].mean() - plain[:K, K:].mean()
        gap_mod = mod[:K, :K][upper].mean() - mod[:K, K:].mean()
        # the body of the filament correlates between any two averages
        assert gap_mod > gap_plain


class TestSuggestedCut:
    def test_it_goes_between_the_two_scales_of_distance(self):
        S = np.eye(8)
        S[:4, :4] = 0.93
        S[4:, 4:] = 0.9
        S[:4, 4:] = S[4:, :4] = 0.5
        np.fill_diagonal(S, 1.0)
        cut = ty.suggest_cut(S)
        assert 0.1 < cut < 0.5
        g = ty.pixel_groups(S, cut, min_classes=2)
        assert len(set(g[:4])) == 1 and len(set(g[4:])) == 1 and g[0] != g[4]

    def test_with_too_few_merges_it_returns_the_default(self):
        assert ty.suggest_cut(np.eye(2), default=0.3) == 0.3


class TestPixelGroups:
    def test_two_structures_make_two_groups(self, similarity):
        # the right cut depends on how different the structures are: these differ
        # little, so 0.2; the default of 0.15 is for the plain correlation of
        # real types (PHF and SF)
        g = ty.pixel_groups(similarity, cut=0.2, min_classes=2)
        assert len(set(g[:K])) == 1 and len(set(g[K:])) == 1
        assert set(g[:K]) != set(g[K:])

    def test_the_default_cut_is_that_which_worked_on_real_types(
        self,
    ):
        import inspect

        assert inspect.signature(ty.pixel_groups).parameters["cut"].default == 0.15

    def test_a_cut_that_is_too_loose_joins_them(self, similarity):
        g = ty.pixel_groups(similarity, cut=0.9, min_classes=2)
        assert len(set(g)) == 1

    def test_a_small_group_is_left_unassigned(self):
        S = np.eye(5)
        S[:4, :4] = 0.95
        np.fill_diagonal(S, 1.0)
        g = ty.pixel_groups(S, cut=0.15, min_classes=3)
        assert list(g) == [0, 0, 0, 0, -1]

    def test_the_largest_group_is_numbered_first(self):
        S = np.eye(7)
        S[:2, :2] = 0.95
        S[2:, 2:] = 0.95
        np.fill_diagonal(S, 1.0)
        g = ty.pixel_groups(S, cut=0.15, min_classes=2)
        assert list(g[2:]) == [0] * 5 and list(g[:2]) == [1, 1]


class TestFilamentTypes:
    @pytest.fixture(scope="class")
    def usage(self):
        # 100 filaments of each type use their own classes, 30% of segments in
        # any class at random (poor assignments)
        rng = np.random.default_rng(0)
        parts = []
        for t in (0, 1):
            df = syn.class2d_table(PERIOD, K, n_fil=100, length=800.0, seed=t + 1)
            df["rlnClassNumber"] += t * K
            df["rlnMicrographName"] = f"{'AB'[t]}_" + df["rlnMicrographName"]
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        bad = rng.random(len(df)) < 0.3
        df.loc[bad, "rlnClassNumber"] = rng.integers(1, 2 * K + 1, int(bad.sum()))
        U, keys = ty.usage_matrix(df, list(range(1, 2 * K + 1)))
        truth = np.array([0 if k[0].startswith("A_") else 1 for k in keys])
        return U, truth

    def test_the_usage_matrix_counts_the_segments(self, usage):
        U, truth = usage
        assert U.shape == (200, 2 * K)
        assert U.sum() == pytest.approx(200 * 41)

    def test_most_poorly_assigned_segments_do_not_change_the_filament(self, usage):
        U, truth = usage
        groups = np.array([0] * K + [1] * K)
        lab, share = ty.assign_filaments(U, groups, 2, min_share=0.6)
        called = lab >= 0
        assert called.mean() > 0.95
        assert np.mean(lab[called] == truth[called]) > 0.97

    def test_a_filament_without_a_majority_is_left_out(self):
        U = np.array([[5.0, 5.0], [9.0, 1.0]])
        lab, share = ty.assign_filaments(U, np.array([0, 1]), 2, min_share=0.6)
        assert list(lab) == [-1, 0]
        assert share[1] == pytest.approx(0.9)

    def test_classes_in_no_group_do_not_vote(self):
        U = np.array([[2.0, 8.0, 100.0]])
        lab, share = ty.assign_filaments(U, np.array([0, 1, -1]), 2, min_share=0.6)
        assert lab[0] == 1 and share[0] == pytest.approx(0.8)


class TestUsageSimilarity:
    def test_classes_used_by_the_same_filaments_are_alike(self):
        U = np.array([[5, 5, 0, 0], [4, 6, 0, 0], [0, 0, 5, 5], [0, 0, 6, 4.0]])
        C = ty.usage_similarity(U)
        assert C[0, 1] > 0.9 and C[2, 3] > 0.9
        assert C[0, 2] == pytest.approx(0.0)


class TestNeighbourFrequency:
    @pytest.fixture(scope="class")
    def params(self):
        # filaments of type A use classes 1-4, those of type B classes 5-8;
        # segments every 20 A, classes advancing along the filament
        rows = []
        rng = np.random.default_rng(0)
        for f in range(60):
            t = f % 2
            for i in range(40):
                rows.append(
                    dict(
                        rlnMicrographName=f"m{f // 10}.mrc",
                        rlnHelicalTubeID=f % 10 + 1,
                        rlnHelicalTrackLengthAngst=20.0 * i,
                        rlnClassNumber=1 + 4 * t + (i // 3 + rng.integers(0, 2)) % 4,
                    )
                )
        df = pd.DataFrame(rows)
        # the two types must not share a filament: make the ids distinct
        df["rlnHelicalTubeID"] = df["rlnHelicalTubeID"] + 100 * (
            df["rlnClassNumber"] > 4
        )
        return df

    def test_classes_of_one_type_meet_on_filaments_and_those_of_two_do_not(
        self, params
    ):
        lift = ty.neighbour_lift(params, list(range(1, 9)), window=400.0)
        assert lift.shape == (8, 8)
        assert np.allclose(lift, lift.T)
        assert lift[:4, :4].mean() > 1.0
        assert lift[:4, 4:].max() == pytest.approx(0.0)

    def test_it_counts_neighbours_within_the_window_only(self, params):
        near = ty.neighbour_lift(params, list(range(1, 9)), window=20.0)
        far = ty.neighbour_lift(params, list(range(1, 9)), window=800.0)
        assert np.count_nonzero(near[:4, :4]) <= np.count_nonzero(far[:4, :4])

    def test_a_class_absent_from_the_data_has_no_neighbours(self, params):
        lift = ty.neighbour_lift(params, [1, 2, 99], window=400.0)
        assert np.all(lift[2] == 0.0) and np.all(lift[:, 2] == 0.0)

    def test_weight_zero_is_the_similarity_alone(self):
        S = np.array([[1.0, 0.9], [0.9, 1.0]])
        lift = np.array([[0.0, 0.2], [0.2, 0.0]])
        assert np.allclose(ty.neighbour_affinity(S, lift, weight=0.0), S)

    def test_classes_that_never_meet_are_pulled_apart(self):
        S = np.full((2, 2), 0.9)
        met = ty.neighbour_affinity(S, np.full((2, 2), 1.5), weight=0.3)
        never = ty.neighbour_affinity(S, np.zeros((2, 2)), weight=0.3)
        assert met[0, 1] > never[0, 1] + 0.2

    def test_the_lift_saturates(self):
        S = np.full((2, 2), 0.5)
        a = ty.neighbour_affinity(S, np.full((2, 2), 1.5), 0.3, saturation=1.5)
        b = ty.neighbour_affinity(S, np.full((2, 2), 50.0), 0.3, saturation=1.5)
        assert np.allclose(a, b)


def _blocks(sizes, within=0.95, between=0.72, seed=0, noise=0.01):
    """A class similarity of groups of the given sizes on a common baseline."""
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(len(sizes)), sizes)
    S = np.where(lab[:, None] == lab[None, :], within, between)
    S = S + noise * rng.standard_normal(S.shape)
    S = np.clip(0.5 * (S + S.T), 0, 1)
    np.fill_diagonal(S, 1.0)
    return S, lab


class TestGroupsByNumber:
    def _three(self):
        # three blocks; the first two are more alike than either is to the third
        S, lab = _blocks([12, 8, 6], within=0.95, between=0.7, noise=0.0)
        S[np.ix_(lab == 0, lab == 1)] = 0.88
        S[np.ix_(lab == 1, lab == 0)] = 0.88
        return S, lab

    def test_the_default_number_is_what_the_default_cut_gives(self):
        S, _ = self._three()
        assert ty.default_n_groups(S) == int(ty.pixel_groups(S).max()) + 1

    def test_the_groups_at_the_default_number_are_those_of_the_default_cut(self):
        S, _ = self._three()
        n = ty.default_n_groups(S)
        labels, cut = ty.pixel_groups_n(S, n)
        assert list(labels) == list(ty.pixel_groups(S))

    def test_fewer_groups_join_the_most_alike_ones(self):
        S, lab = self._three()
        labels, _ = ty.pixel_groups_n(S, 2)
        assert len(set(labels)) == 2 and len(set(labels[lab == 2])) == 1
        assert len(set(labels[lab != 2])) == 1  # blocks 0 and 1 are joined

    def test_one_group_holds_every_class_the_tree_joins(self):
        S, _ = self._three()
        labels, _ = ty.pixel_groups_n(S, 1)
        assert set(labels) == {0}

    def test_a_number_no_cut_gives_falls_back_to_the_nearest_one(self):
        S, _ = self._three()
        labels, _ = ty.pixel_groups_n(S, 8)
        assert int(labels.max()) + 1 == 3

    def test_junk_classes_stay_unassigned_when_they_form_no_group(self):
        S, lab = _blocks([12, 10, 1, 1])
        labels, _ = ty.pixel_groups_n(S, 2)
        assert (labels[lab >= 2] == -1).all() and set(labels[lab < 2]) == {0, 1}


def _marker_data(n_a=300, n_b=300, n_seg=40, marker_rate=0.3, seed=0):
    """Two types sharing classes 0-5; type A also shows classes 6-8 (its markers)
    at ``marker_rate`` of its segments, and type B never does. Classes 0-5 and
    6-8 look alike among themselves."""
    rng = np.random.default_rng(seed)
    K = 9
    counts = np.zeros((n_a + n_b, K))
    for i in range(n_a + n_b):
        n = rng.integers(n_seg // 2, n_seg * 2)
        if i < n_a:
            k = rng.binomial(n, marker_rate)
            np.add.at(counts[i], rng.integers(6, 9, k), 1)
            np.add.at(counts[i], rng.integers(0, 6, n - k), 1)
        else:
            np.add.at(counts[i], rng.integers(0, 6, n), 1)
    S = np.full((K, K), 0.7)
    S[:6, :6] = 0.95
    S[6:, 6:] = 0.95
    np.fill_diagonal(S, 1.0)
    return counts, S, np.r_[np.zeros(n_a, int), np.ones(n_b, int)]


class TestBinomialMixture:
    def test_two_populations_are_found_with_their_rates(self):
        rng = np.random.default_rng(0)
        n = rng.integers(20, 80, 2000)
        high = rng.random(2000) < 0.4
        k = rng.binomial(n, np.where(high, 0.3, 0.01))
        fit = ty.binomial_mixture(k, n)
        assert abs(fit["low"] - 0.01) < 0.01 and abs(fit["high"] - 0.3) < 0.02
        assert abs(fit["share"] - 0.4) < 0.03
        assert fit["gain"] > 2.0
        assert np.mean(ty.in_high(fit, k, n) == high) > 0.97

    def test_one_population_gains_little(self):
        rng = np.random.default_rng(1)
        n = rng.integers(20, 80, 2000)
        k = rng.binomial(n, 0.2)
        assert ty.binomial_mixture(k, n)["gain"] < 0.05

    def test_weights_count_filaments_with_the_same_counts_once(self):
        k, n = np.array([0, 0, 5, 5, 5]), np.array([20, 20, 20, 20, 20])
        a = ty.binomial_mixture(k, n)
        b = ty.binomial_mixture(np.array([0, 5]), np.array([20, 20]), np.array([2, 3]))
        assert a["gain"] == pytest.approx(b["gain"], abs=1e-6)
        assert a["high"] == pytest.approx(b["high"], abs=1e-6)


class TestMarkerCandidates:
    def test_the_sets_of_alike_classes_are_offered(self):
        _, S, _ = _marker_data()
        sets = [tuple(x) for x in ty.marker_candidates(S, max_share=0.7)]
        assert (6, 7, 8) in sets and (0, 1, 2, 3, 4, 5) in sets

    def test_no_set_is_larger_than_its_share_of_the_classes(self):
        _, S, _ = _marker_data()
        assert all(len(x) <= 3 for x in ty.marker_candidates(S))


class TestMarkerGroups:
    def test_the_type_with_markers_is_split_from_the_one_without(self):
        counts, S, truth = _marker_data()
        splits = ty.marker_groups(counts, S, max_groups=2)
        assert len(splits) == 1
        assert list(splits[0]["markers"]) == [6, 7, 8]
        labels = splits[0]["labels"]
        # the group that uses the markers keeps number 0
        assert np.mean(labels[truth == 0] == 0) > 0.95
        assert np.mean(labels[truth == 1] == 1) > 0.95

    def test_a_clear_split_is_in_the_default_number_of_groups(self):
        counts, S, _ = _marker_data()
        splits = ty.marker_groups(counts, S, max_groups=4)
        assert splits[0]["gain"] > 2.0
        assert ty.default_n_marker_groups(splits) >= 2

    def test_one_type_has_no_clear_split(self):
        counts, S, _ = _marker_data(n_a=600, n_b=0)
        splits = ty.marker_groups(counts, S, max_groups=4)
        assert ty.default_n_marker_groups(splits) == 1

    def test_filaments_without_compared_segments_are_not_grouped(self):
        counts, S, _ = _marker_data()
        counts[:3] = 0
        labels = ty.marker_groups(counts, S, max_groups=2)[0]["labels"]
        assert np.all(labels[:3] == -1)

    def test_short_filaments_follow_the_split(self):
        counts, S, truth = _marker_data()
        counts[0] = 0
        counts[0, 7] = 3  # three marker segments: type A, though too short to fit
        labels = ty.marker_groups(counts, S, max_groups=2)[0]["labels"]
        assert labels[0] == 0


class TestDescribeMarkerGroups:
    def test_one_group_before_any_split(self):
        counts, S, _ = _marker_data()
        splits = ty.marker_groups(counts, S, max_groups=2)
        labels, groups = ty.describe_marker_groups(splits, 1)
        assert labels is None and len(groups) == 1

    def test_the_group_without_sparse_markers_may_hold_the_other_type(self):
        counts, S, _ = _marker_data(marker_rate=0.3)
        splits = ty.marker_groups(counts, S, max_groups=2)
        _, groups = ty.describe_marker_groups(splits, 2)
        assert groups[0]["markers"] == [6, 7, 8] and groups[0]["lacks"] == []
        assert groups[1]["markers"] == [] and groups[1]["lacks"] == [0]

    def test_abundant_markers_are_not_missed_by_short_filaments(self):
        counts, S, _ = _marker_data(marker_rate=0.8)
        splits = ty.marker_groups(counts, S, max_groups=2)
        _, groups = ty.describe_marker_groups(splits, 2)
        assert groups[1]["lacks"] == []


class TestSimilarityCache:
    def test_the_same_averages_are_not_registered_again(self, monkeypatch):
        rng = np.random.default_rng(5)
        images = [rng.standard_normal((32, 48)).astype(np.float32) for _ in range(4)]
        first = ty.class_similarity(images, range(4), 2.0)
        calls = []
        monkeypatch.setattr(
            ty, "_class_similarity", lambda *a, **k: calls.append(1) or np.eye(4)
        )
        assert np.array_equal(ty.class_similarity(images, range(4), 2.0), first)
        assert not calls
        # other averages, or other settings, are computed
        images[2] = images[2] + 1.0
        ty.class_similarity(images, range(4), 2.0)
        ty.class_similarity(images, range(4), 2.0, modulation=True)
        assert len(calls) == 2

    def test_a_cached_matrix_cannot_be_changed_from_outside(self):
        rng = np.random.default_rng(6)
        images = [rng.standard_normal((32, 48)).astype(np.float32) for _ in range(3)]
        S = ty.class_similarity(images, range(3), 2.0)
        S[0, 1] = -5
        assert ty.class_similarity(images, range(3), 2.0)[0, 1] != -5
