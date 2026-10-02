"""Tests for the class-phase pitch estimate of the AbInitio3D tab.

Everything is built in memory: filaments of known period whose segments are
labelled by the azimuth sector they sit at, which is what a 2D classification
of a helix does to within noise.
"""

import numpy as np
import pandas as pd
import pytest

from helicon.webApps.lib import helical_pitch_phase as ph

APIX_MIC = 1.0


def make_params(
    n_fil=200,
    period=300.0,
    n_classes=12,
    n_seg=80,
    step=10.0,
    noise=0.2,
    period_sd=0.0,
    class_offset=0,
    tube_offset=0,
    seed=0,
    reverse_fraction=0.0,
):
    """Class2D parameters for straight filaments of known period.

    Each filament gets a random phase, and ``period_sd`` a period of its own.
    Coordinates run along x, so the refined path is the track itself. A
    filament picked the other way (``reverse_fraction`` of them) has psi turned
    by 180 degrees and its phase running down its track.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fil):
        p = period * (1.0 + period_sd * rng.standard_normal())
        offset = rng.random()
        backwards = reverse_fraction > 0 and rng.random() < reverse_fraction
        track = np.arange(n_seg) * step
        phase = ((-track if backwards else track) / p + offset) % 1.0
        cls = np.minimum((phase * n_classes).astype(int), n_classes - 1)
        flip = rng.random(n_seg) < noise
        cls[flip] = rng.integers(0, n_classes, int(flip.sum()))
        for t, c, ph_true in zip(track, cls, phase):
            rows.append(
                dict(
                    rlnMicrographName=f"mic{f // 10:03d}.mrc",
                    rlnHelicalTubeID=tube_offset + f % 10 + 1,
                    rlnHelicalTrackLengthAngst=t,
                    rlnClassNumber=class_offset + c + 1,
                    rlnAnglePsi=180.0 if backwards else 0.0,
                    rlnAnglePsiPrior=0.0,
                    truePhase=ph_true,
                    rlnCoordinateX=(1000.0 + t) / APIX_MIC,
                    rlnCoordinateY=500.0 / APIX_MIC,
                    rlnOriginXAngst=0.0,
                    rlnOriginYAngst=0.0,
                    rlnMicrographOriginalPixelSize=APIX_MIC,
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def homogeneous():
    return make_params()


class TestRefinedPath:
    def test_straight_path_is_the_track(self):
        t = np.arange(50) * 10.0
        pos, perp = ph.refined_path_positions(t, 100.0 + t, np.full(50, 20.0))
        assert np.allclose(pos - pos[0], t, atol=1e-6)
        assert np.allclose(perp, 0.0, atol=1e-6)

    def test_curved_path_measures_the_arc_not_the_chord(self):
        # a filament on a circle of radius R, boxed along its chord: the track
        # says the chord length, the refined centres lie on the arc
        R, half_angle = 1500.0, 0.5
        chord = 2 * R * np.sin(half_angle)
        track = np.linspace(0.0, chord, 120)
        angle = np.arcsin((track - chord / 2) / R)
        x, y = R * np.sin(angle), R * np.cos(angle)
        pos, _ = ph.refined_path_positions(track, x, y)
        arc = 2 * R * half_angle
        assert (pos[-1] - pos[0]) == pytest.approx(arc, rel=2e-3)
        assert (track[-1] - track[0]) < 0.96 * arc  # the chord really is short

    def test_axial_shift_moves_the_position(self):
        t = np.arange(20) * 10.0
        x = t.copy()
        x[10] += 3.0  # this segment sits 3 A further along the filament
        pos, _ = ph.refined_path_positions(t, x, np.zeros(20))
        assert pos[10] - t[10] == pytest.approx(3.0, abs=0.5)


class TestPrepare:
    def test_uses_refined_path_when_columns_exist(self, homogeneous):
        assert ph.prepare_pairs(homogeneous).refined_path

    def test_falls_back_to_track_length_without_coordinates(self, homogeneous):
        # the cryoSPARC loader provides no coordinates or origins
        df = homogeneous.drop(columns=["rlnCoordinateX", "rlnOriginXAngst"])
        pairs = ph.prepare_pairs(df)
        assert not pairs.refined_path
        assert len(pairs.D) > 0

    def test_opposite_polarity_pairs_are_left_out(self):
        # twenty consistent filaments pin down each class's axis convention;
        # in one of them the second half then runs the other way
        df = make_params(n_fil=20, n_seg=20)
        df["rlnClassNumber"] = np.tile(np.arange(20) % 3 + 1, 20)
        first = df.index[:20]
        df.loc[first[10:], "rlnAnglePsi"] = 180.0
        pairs = ph.prepare_pairs(df)
        assert np.all(pairs.seg_pol[pairs.I] == pairs.seg_pol[pairs.J])
        per_fil = np.bincount(pairs.pair_filament, minlength=pairs.n_filaments)
        flipped = pairs.filament_keys.index(("mic000.mrc", 1))
        assert per_fil[flipped] == 2 * (10 * 9 // 2)
        assert np.all(np.delete(per_fil, flipped) == 20 * 19 // 2)

    def test_no_segments_is_a_clear_error(self, homogeneous):
        with pytest.raises(ValueError, match="no segments"):
            ph.prepare_pairs(homogeneous.iloc[:0])

    def test_offsets_respect_max_sep(self, homogeneous):
        pairs = ph.prepare_pairs(homogeneous, max_sep=200.0)
        assert np.all(np.abs(pairs.D) <= 200.0)


class TestPeriod:
    def test_recovers_the_period(self, homogeneous):
        r = ph.estimate_period(ph.prepare_pairs(homogeneous))
        assert r["period"] == pytest.approx(300.0, rel=0.01)

    def test_more_than_two_classes_get_ordered_phases(self, homogeneous):
        pairs = ph.prepare_pairs(homogeneous)
        r = ph.estimate_period(pairs)
        # class k covers sector k of the period, so the phases step around the
        # circle in class order, one way or the other
        steps = np.angle(np.exp(1j * np.diff(r["phases"])))
        expected = 2 * np.pi / pairs.n_classes
        assert np.allclose(np.abs(steps), expected, atol=0.25 * expected)
        assert np.all(np.sign(steps) == np.sign(steps[0]))

    def test_calibration_removes_short_filament_bias(self):
        # filaments a little shorter than the period: the raw peak is biased low
        df = make_params(n_fil=400, period=600.0, n_seg=55, step=10.0, seed=1)
        pairs = ph.prepare_pairs(df)
        raw = ph.estimate_period(pairs, length_scale=400.0)["period"]
        cal = ph.calibrate_period(pairs, raw, length_scale=400.0)["period"]
        assert abs(cal - 600.0) < abs(raw - 600.0)
        assert cal == pytest.approx(600.0, rel=0.01)

    def test_filaments_shorter_than_half_the_period_are_flagged(self):
        # no pair is further apart than 290 A, so 600 A and 300 A fit alike
        df = make_params(n_fil=400, period=600.0, n_seg=30, step=10.0, seed=1)
        r = ph.estimate_period(ph.prepare_pairs(df))
        assert r["support"] < ph.MIN_SUPPORT

    def test_long_filaments_are_not_flagged(self, homogeneous):
        r = ph.estimate_period(ph.prepare_pairs(homogeneous))
        assert r["support"] > ph.MIN_SUPPORT

    def test_bootstrap_spread_is_small_for_clean_data(self, homogeneous):
        pairs = ph.prepare_pairs(homogeneous)
        boot = ph.bootstrap_period(pairs, 300.0, n_boot=5)
        assert np.std(boot) < 3.0


class TestHeterogeneity:
    def test_homogeneous_filaments_show_little_real_spread(self, homogeneous):
        pairs = ph.prepare_pairs(homogeneous)
        r = ph.estimate_period(pairs)
        het = ph.filament_heterogeneity(pairs, r["period"], r["phases"], n_boot=200)
        assert het["n"] > 100
        assert het["sd_between"] < 0.02 * 300.0

    def test_heterogeneous_filaments_are_measured(self):
        df = make_params(period_sd=0.05, seed=2)
        pairs = ph.prepare_pairs(df)
        r = ph.estimate_period(pairs)
        het = ph.filament_heterogeneity(pairs, r["period"], r["phases"], n_boot=200)
        assert het["sd_between"] == pytest.approx(0.05 * 300.0, rel=0.35)
        assert het["corr"] > 0.5

    def test_fits_at_the_edge_of_the_range_are_rejected(self, homogeneous):
        pairs = ph.prepare_pairs(homogeneous)
        r = ph.estimate_period(pairs)
        # a range that excludes the true period: every fit runs into its edge
        fits = ph.filament_periods(pairs, r["phases"], 500.0, rel_range=0.1)
        assert np.all(np.isnan(fits["period"]))


class TestGroups:
    def test_one_type_is_one_group(self, homogeneous):
        groups = ph.class_groups(ph.prepare_pairs(homogeneous))
        assert len(groups) == 1

    def test_two_types_with_their_own_classes_are_separated(self):
        a = make_params(n_fil=100, period=300.0, seed=3)
        b = make_params(
            n_fil=100, period=400.0, class_offset=100, tube_offset=100, seed=4
        )
        pairs = ph.prepare_pairs(pd.concat([a, b], ignore_index=True))
        groups = ph.class_groups(pairs)
        assert len(groups) == 2
        ids = [set(pairs.class_ids[g] > 100) for g in groups]
        assert all(len(s) == 1 for s in ids)  # each group is one type only


class TestAnalyze:
    def test_end_to_end(self, homogeneous):
        messages = []
        r = ph.analyze(homogeneous, n_boot=5, progress=messages.append)
        assert r.period == pytest.approx(300.0, rel=0.01)
        assert r.period_sd < 5.0
        assert len(r.groups) == 1 and r.group_periods == []
        assert len(r.phases) == len(r.class_ids) == 12
        assert messages

    def test_filament_table_selects_a_pitch_band(self):
        df = make_params(period_sd=0.05, seed=5)
        r = ph.analyze(df, n_boot=3)
        table = r.filaments
        assert len(table) > 100
        assert {"rlnMicrographName", "rlnHelicalTubeID", "pitch"} <= set(table)
        low, high = np.nanpercentile(table["pitch"], [25, 75])
        sub = ph.select_segments(df, table, low, high)
        chosen = set(zip(sub["rlnMicrographName"], sub["rlnHelicalTubeID"]))
        in_band = table[(table["pitch"] >= low) & (table["pitch"] <= high)]
        assert chosen == set(
            zip(in_band["rlnMicrographName"], in_band["rlnHelicalTubeID"])
        )
        assert 0 < len(sub) < len(df)

    def test_selection_carries_pitch_and_length_and_filters_by_length(self):
        df = make_params(period_sd=0.05, seed=5)
        r = ph.analyze(df, n_boot=3)
        table = r.filaments
        everything = ph.select_segments(df, table)
        finite = table[table["pitch"].notna()]
        assert set(
            zip(everything["rlnMicrographName"], everything["rlnHelicalTubeID"])
        ) == set(zip(finite["rlnMicrographName"], finite["rlnHelicalTubeID"]))
        row = finite.iloc[0]
        mine = everything[
            (everything["rlnMicrographName"] == row["rlnMicrographName"])
            & (everything["rlnHelicalTubeID"] == row["rlnHelicalTubeID"])
        ]
        assert (mine["heliconFilamentPitch"] == row["pitch"]).all()
        assert (mine["heliconFilamentLength"] == row["span"]).all()
        cut = float(finite["span"].max())
        assert len(ph.select_segments(df, table, min_length=cut)) > 0
        assert len(ph.select_segments(df, table, min_length=cut + 1.0)) == 0
        assert len(ph.select_segments(df, table, max_length=cut)) == len(everything)
        assert len(ph.select_segments(df, table, max_length=0.0)) == 0
        assert everything.attrs == df.attrs

    def test_the_export_can_be_limited_to_the_selected_classes(self):
        df = make_params(n_fil=40, n_classes=8, seed=11)
        r = ph.analyze(df, class_ids=range(1, 5), n_boot=3)
        chosen = [1, 2, 3, 4]
        out = ph.select_segments(df, r.filaments, class_numbers=chosen)
        assert len(out) > 0
        assert out["rlnClassNumber"].astype(int).isin(chosen).all()
        everything = ph.select_segments(df, r.filaments)
        assert len(everything) > len(out)
        angles = ph.segment_angles(df, r)
        both = ph.select_segments(df, r.filaments, angles=angles, class_numbers=chosen)
        assert len(both) == len(out) and both["rlnAngleRot"].notna().all()

    def test_the_most_populated_class_is_at_zero_azimuth(self, homogeneous):
        r = ph.analyze(homogeneous, n_boot=3)
        assert r.phases[np.argmax(r.class_weight)] == pytest.approx(0.0, abs=1e-9)
        assert np.all(np.abs(r.phases) <= np.pi + 1e-9)

    def test_every_filament_gets_a_pitch_and_each_segment_its_own_angle(self):
        df = make_params(n_fil=60, n_seg=8, n_classes=12, seed=9)  # short filaments
        r = ph.analyze(df, n_boot=3)
        assert len(r.filaments) == 60 and r.filaments["pitch"].notna().all()
        assert (r.filaments.loc[~r.filaments["fitted"], "pitch"] == r.period).all()
        out = ph.select_segments(df, r.filaments, angles=ph.segment_angles(df, r))
        assert len(out) == len(df)
        one = out[out["rlnHelicalTubeID"] == out["rlnHelicalTubeID"].iloc[0]]
        one = one[one["rlnMicrographName"] == one["rlnMicrographName"].iloc[0]]
        assert one["rlnAngleRot"].nunique() == len(one)  # not one angle per class
        assert (out["rlnAngleTilt"] == 90.0).all()

    def test_no_angles_leaves_the_table_as_it_was(self):
        df = make_params(n_fil=20, n_classes=6, seed=8)
        r = ph.analyze(df, n_boot=3)
        out = ph.select_segments(df, r.filaments)
        assert "rlnAngleRot" not in out
        assert (out["rlnAnglePsi"] == 0.0).all()

    def test_scan_covers_the_full_range(self, homogeneous):
        r = ph.analyze(homogeneous, n_boot=3)
        assert r.scan_periods.min() <= 150.0 and r.scan_periods.max() >= 1400.0

    def test_no_pairs_is_an_error(self):
        df = make_params(n_fil=3, n_seg=1)
        with pytest.raises(ValueError):
            ph.analyze(df)


class TestClassPolarity:
    def test_rotated_classes_with_flipped_axes_are_reconciled(self):
        # cryoSPARC-like: every class carries its own rotation, and half the
        # classes label the opposite end of their axis as "+"; every segment
        # of a filament still runs the same way
        rng = np.random.default_rng(6)
        n_fil, n_seg, K = 60, 30, 8
        seg_class = rng.integers(0, K, n_fil * n_seg)
        seg_fil = np.repeat(np.arange(n_fil), n_seg)
        truth = np.repeat(rng.choice([-1.0, 1.0], n_fil), n_seg)
        rotation = rng.uniform(-90, 90, K)
        flipped = np.where(np.arange(K) % 2 == 0, 0.0, 180.0)
        delta = (
            rotation[seg_class] + flipped[seg_class] + np.where(truth > 0, 0.0, 180.0)
        )
        delta += rng.normal(0, 5, len(delta))
        pol = ph.class_polarity(delta, seg_class, seg_fil, K)
        agree = np.mean(pol == truth)
        assert max(agree, 1 - agree) == 1.0

    def test_relion_style_polarity_is_unchanged(self):
        seg_class = np.array([0, 1, 0, 1, 2, 2])
        seg_fil = np.array([0, 0, 0, 1, 1, 1])
        delta = np.array([0.0, 0.0, 0.0, 180.0, 180.0, 180.0])
        pol = ph.class_polarity(delta, seg_class, seg_fil, 3)
        assert np.all(pol[:3] == pol[0]) and np.all(pol[3:] == -pol[0])


class TestMaxClasses:
    def test_keeps_the_most_populated_classes(self, homogeneous):
        df = homogeneous.copy()
        df.loc[df.index[:5], "rlnClassNumber"] = 99  # a nearly empty class
        pairs = ph.prepare_pairs(df, max_classes=12)
        assert 99 not in set(pairs.class_ids)
        assert len(pairs.class_ids) == 12


class TestSelectionDiagnostics:
    def test_a_junk_class_fits_the_ring_worst(self, homogeneous):
        df = homogeneous.copy()
        rng = np.random.default_rng(11)
        junk = rng.random(len(df)) < 0.08  # segments sent to a class at random
        df.loc[junk, "rlnClassNumber"] = 99
        pairs = ph.prepare_pairs(df)
        r = ph.estimate_period(pairs)
        fit = ph.class_fit(pairs, r["phases"], r["period"])
        k = list(pairs.class_ids).index(99)
        assert fit[k] < 0.5 * np.nanmedian(np.delete(fit, k))

    def test_phase_gap_of_a_full_ring_is_small(self, homogeneous):
        r = ph.estimate_period(ph.prepare_pairs(homogeneous))
        assert ph.largest_phase_gap(r["phases"]) < 60.0

    def test_half_the_views_is_flagged_as_possibly_half_the_repeat(self, homogeneous):
        # classes 1-6 cover half of the 12 sectors of the 300 A period: they
        # fit a 150 A ring at least as well, and the estimate lands there
        df = homogeneous[homogeneous["rlnClassNumber"] <= 6]
        r = ph.estimate_period(ph.prepare_pairs(df))
        assert r["period"] == pytest.approx(150.0, rel=0.02)
        assert r["double_ratio"] >= ph.DOUBLE_RATIO_WARN

    def test_a_full_selection_is_not_flagged(self, homogeneous):
        r = ph.estimate_period(ph.prepare_pairs(homogeneous))
        assert r["double_ratio"] < ph.DOUBLE_RATIO_WARN

    def test_a_missing_quarter_leaves_a_gap(self, homogeneous):
        df = homogeneous[homogeneous["rlnClassNumber"] <= 9]
        r = ph.estimate_period(ph.prepare_pairs(df))
        assert r["period"] == pytest.approx(300.0, rel=0.01)
        assert ph.largest_phase_gap(r["phases"]) > 90.0

    def test_analyze_uses_only_the_selected_classes(self, homogeneous):
        r = ph.analyze(homogeneous, class_ids=[1, 2, 3, 4, 5, 6, 7, 8], n_boot=3)
        assert list(r.class_ids) == [1, 2, 3, 4, 5, 6, 7, 8]
        assert len(r.class_fit) == 8


class TestFilamentDirection:
    def test_unknown_picking_direction_is_recovered(self):
        # half the filaments run the other way along their track, and nothing
        # in psi says which: their offsets are negated
        df = make_params(seed=12)
        pairs = ph.prepare_pairs(df)
        rng = np.random.default_rng(13)
        reversed_ = rng.random(pairs.n_filaments) < 0.5
        pairs.D = np.where(reversed_[pairs.pair_filament], -pairs.D, pairs.D)
        plain = ph.estimate_period(pairs)
        est, flipped = ph.sync_filament_directions(pairs, plain)
        assert est["period"] == pytest.approx(300.0, rel=0.01)
        assert est["score"] > plain["score"]
        agree = np.mean(flipped == reversed_)
        assert max(agree, 1 - agree) > 0.95

    def test_consistent_directions_are_left_alone(self, homogeneous):
        pairs = ph.prepare_pairs(homogeneous)
        est = ph.estimate_period(pairs)
        again, flipped = ph.sync_filament_directions(pairs, est)
        assert again["period"] == pytest.approx(est["period"], abs=1.0)
        assert flipped.mean() < 0.05 or flipped.mean() > 0.95


def _filament_image(phase, ny=48, nx=128, period=64.0, seed=0):
    """A side-view-like image: two strands crossing, starting at ``phase``."""
    x = np.arange(nx)
    y = np.arange(ny)[:, None] - ny / 2
    img = np.zeros((ny, nx))
    for sign in (1, -1):
        centre = sign * 10 * np.sin(2 * np.pi * (x / period + phase))
        width = 2.5 + 1.5 * sign * np.cos(2 * np.pi * (x / period + phase))
        img += np.exp(-0.5 * ((y - centre) / width) ** 2)
    rng = np.random.default_rng(seed)
    return (img + 0.05 * rng.standard_normal(img.shape)).astype(np.float32)


class TestCounterparts:
    def test_the_rotated_view_is_suggested_and_an_unrelated_one_is_not(self):
        a = _filament_image(0.0, seed=1)
        partner = np.rot90(_filament_image(0.0, seed=2), 2)  # a, picked the other way
        blob = np.zeros_like(a)
        blob[20:28, 40:80] = 1.0  # not a filament view at all
        images = [a, partner, blob]
        found = ph.suggest_counterparts(
            images, selected=[0], candidates=[1, 2], min_corr=0.5
        )
        assert [d["candidate"] for d in found] == [1]
        assert found[0]["selected"] == 0

    def test_on_all_cpus_the_same_as_on_one(self, monkeypatch):
        # 3 selected x 14 candidates: enough registrations for the process pool
        import helicon

        views = [_filament_image(k / 17.0, seed=k) for k in range(17)]
        images = views[:3] + [np.rot90(v, 2) for v in views[3:]]
        args = dict(selected=[0, 1, 2], candidates=list(range(3, 17)), min_corr=0.3)
        monkeypatch.setattr(helicon, "available_cpu", lambda *a, **k: 4)
        pooled = ph.suggest_counterparts(images, **args)
        monkeypatch.setattr(helicon, "available_cpu", lambda *a, **k: 1)
        serial = ph.suggest_counterparts(images, **args)
        assert pooled == serial and pooled

    def test_nothing_to_suggest_without_candidates(self):
        a = _filament_image(0.0)
        assert ph.suggest_counterparts([a], selected=[0], candidates=[0]) == []

    def test_wide_images_are_binned(self):
        img = np.ones((64, 512), np.float32)
        assert ph._bin_to_width(img, 128).shape == (16, 128)


class TestSuggestExpansion:
    def _two_types(self):
        a = make_params(n_fil=100, period=300.0, seed=3)
        b = make_params(
            n_fil=100, period=400.0, class_offset=100, tube_offset=100, seed=4
        )
        b["rlnMicrographName"] = "other_" + b["rlnMicrographName"]
        return pd.concat([a, b], ignore_index=True)

    def test_classes_of_the_same_filaments_are_suggested_and_the_other_type_is_not(
        self,
    ):
        df = self._two_types()
        found = ph.suggest_expansion(df, selected=range(1, 9))
        suggested = {d["candidate"] for d in found}
        assert suggested == {9, 10, 11, 12}
        assert all(d["share"] > d["baseline"] and d["z"] >= 3.0 for d in found)
        assert [d["z"] for d in found] == sorted((d["z"] for d in found), reverse=True)

    def test_candidates_restrict_the_suggestions(self):
        df = self._two_types()
        found = ph.suggest_expansion(df, selected=range(1, 9), candidates=[9, 101])
        assert [d["candidate"] for d in found] == [9]

    def test_nothing_is_suggested_when_all_classes_are_selected_or_none_fit(self):
        df = self._two_types()
        assert (
            ph.suggest_expansion(
                df, selected=list(range(1, 13)) + list(range(101, 113))
            )
            == []
        )
        assert ph.suggest_expansion(df, selected=[9999]) == []


class TestClassDiagnosis:
    """Which classes to take out of a fit, from the segments alone."""

    def _result(self, fit, evidence, merged_into=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            class_ids=np.arange(1, len(fit) + 1),
            class_fit=np.asarray(fit, dtype=float),
            class_evidence=np.asarray(evidence, dtype=float),
            merged_into=merged_into or {},
        )

    def test_too_few_off_the_ring_and_their_copies(self):
        fit = [0.8, 0.82, 0.79, 0.81, 0.02, 0.5, 0.78]
        evidence = [100, 120, 90, 110, 80, 1, 100]
        r = self._result(fit, evidence, merged_into={7: 5})
        out = ph.diagnose_classes(r)
        assert out == {
            6: ph.TOO_FEW,
            5: ph.AT_CHANCE,
            7: "180° copy of class 5",
        }

    def test_good_classes_are_left_alone(self):
        r = self._result([0.8, 0.7, 0.75, 0.9], [100, 50, 80, 120])
        assert ph.diagnose_classes(r) == {}

    def test_the_cut_follows_a_poorer_dataset(self):
        # median 0.2: a class at 0.08 is still within reach of the rest
        ids = [1, 2, 3, 4]
        assert ph.poorly_fitting(ids, [0.2, 0.22, 0.18, 0.08]) == []
        assert ph.poorly_fitting(ids, [0.2, 0.22, 0.18, 0.02]) == [4]

    def test_near_neighbours_do_not_vouch_for_a_class(self, homogeneous):
        # pairs closer than a quarter period agree with any phase; a class
        # with only those has no evidence at all
        pairs = ph.prepare_pairs(homogeneous)
        r = ph.estimate_period(pairs)
        fit = ph.class_fit(pairs, r["phases"], r["period"], min_separation=10.0)
        assert np.all(np.isnan(fit))


class TestLongRepeats:
    """A slowly twisting filament: a repeat beyond the old fixed 1500 A."""

    @pytest.fixture(scope="class")
    def long_repeat(self):
        # 2400 A repeat (0.71 degree twist at 4.75 A rise), 3600 A filaments
        return make_params(n_fil=120, period=2400.0, n_seg=180, step=20.0, seed=5)

    def test_found_when_the_search_reaches_it(self, long_repeat):
        r = ph.analyze(long_repeat, n_boot=2, max_sep=360.0 * 4.75 / 0.3)
        assert r.period == pytest.approx(2400.0, rel=0.02)

    def test_missed_by_the_old_fixed_limit(self, long_repeat):
        r = ph.analyze(long_repeat, n_boot=2, max_sep=1500.0)
        assert r.period < 1500.0 * 1.15


class TestFilamentSlices:
    """The per-filament walk over numpy columns gives what the per-filament
    pandas walk gave, without its cost."""

    def test_filaments_in_groupby_order_and_rows_by_track(self):
        df = make_params(n_fil=30, n_seg=12, seed=7).sample(frac=1.0, random_state=1)
        order, starts, ends, keys = ph._filament_slices(df)
        groups = list(df.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False))
        assert keys == [k for k, _ in groups]
        track = df["rlnHelicalTrackLengthAngst"].to_numpy()
        for (a, b), (_, h) in zip(zip(starts, ends), groups):
            rows = order[a:b]
            assert sorted(rows.tolist()) == sorted(df.index.get_indexer(h.index))
            assert np.all(np.diff(track[rows]) >= 0)

    def test_pairs_do_not_depend_on_the_frames_attrs(self):
        # pandas copies attrs on every column access: the cost the walk avoids
        df = make_params(n_fil=40, seed=8)
        plain = ph.prepare_pairs(df)
        heavy = df.copy()
        heavy.attrs["optics"] = pd.DataFrame(np.zeros((2000, 20)))
        loaded = ph.prepare_pairs(heavy)
        for f in ("I", "J", "D", "W", "seg_pos", "seg_class", "seg_row"):
            assert np.array_equal(getattr(plain, f), getattr(loaded, f))

    @pytest.mark.parametrize("flip", [1.0, -1.0])
    def test_direction_convention(self, flip):
        df = make_params(n_fil=20, n_seg=60, seed=9)
        # filaments at 30 degrees, the prior agreeing with +y (or -y)
        t = df["rlnHelicalTrackLengthAngst"].to_numpy()
        a = np.deg2rad(30.0)
        df["rlnCoordinateX"] = (1000.0 + t * np.cos(a)) / APIX_MIC
        df["rlnCoordinateY"] = (500.0 + flip * t * np.sin(a)) / APIX_MIC
        df["rlnAnglePsiPrior"] = 30.0
        assert ph._direction_convention(df) == flip


class TestMergeTrials:
    def test_the_winning_trial_pairs_are_the_ones_used(self, monkeypatch):
        df = make_params(n_fil=60, seed=10)
        calls = []
        real = ph.prepare_pairs
        monkeypatch.setattr(
            ph, "prepare_pairs", lambda *a, **k: calls.append(1) or real(*a, **k)
        )
        cps = [dict(a=1, b=7, corr=0.9, dx=0.0)]
        r = ph.analyze(df, n_boot=2, counterparts=cps, image_apix=1.0)
        # one per sign; the final collection reuses the better one
        assert len(calls) == 2
        assert np.isfinite(r.period)


class TestFilamentCurves:
    def _direct(self, fi, n_fil, d, w, dphi, grid):
        out = np.zeros((len(grid), n_fil))
        for k, p in enumerate(grid):
            np.add.at(out[k], fi, w * np.cos(2 * np.pi * d / p - dphi))
        return out

    def test_matches_the_sum_over_pairs(self):
        rng = np.random.default_rng(11)
        n_fil = 40
        reach = rng.uniform(50, 3000, n_fil)  # short and long filaments
        fi = rng.integers(0, n_fil, 4000)
        d = rng.uniform(-1, 1, len(fi)) * reach[fi]
        # separations on the bins themselves, so binning moves nothing
        d = np.round(d / 0.5) * 0.5
        w, dphi = rng.random(len(fi)), rng.uniform(-np.pi, np.pi, len(fi))
        grid = np.arange(600.0, 800.0, 7.0)
        got = ph._filament_curves(fi, n_fil, d, w, dphi, grid)
        assert np.allclose(got, self._direct(fi, n_fil, d, w, dphi, grid))

    def test_the_longest_pair_stays_in_its_own_filament(self):
        # 0.3 A past the first pair rounds up a bin: it used to spill into the
        # next filament's first bin, or past the array for the last one
        fi = np.array([0, 1, 1])
        d = np.array([0.0, 0.0, 0.3])
        w, dphi = np.ones(3), np.zeros(3)
        grid = np.array([700.0])
        got = ph._filament_curves(fi, 2, d, w, dphi, grid)
        assert np.allclose(got, self._direct(fi, 2, d, w, dphi, grid), atol=1e-6)
