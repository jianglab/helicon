"""trueFSC: the automatic mask (central Z, threshold and edge search, density
kept), the area under the curve, and the options that reach them."""

import argparse
from unittest.mock import patch

import numpy as np
import pytest

import _truefsc_synthetic as syn
from helicon.commands import trueFSC


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch):
    """Every test runs the search: a cached answer would test nothing."""
    search = getattr(trueFSC._optimal_mask, "__wrapped__", None)
    if search is not None:
        monkeypatch.setattr(trueFSC, "_optimal_mask", search)


class TestCentralZ:
    def test_only_the_central_sections_are_kept(self):
        keep = trueFSC._central_z((10, 4, 4), 4).ravel()
        assert keep.tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]

    def test_a_length_or_a_fraction(self):
        assert trueFSC._z_pixels(100, 2.0, mask_z_len=50) == 25
        assert trueFSC._z_pixels(100, 2.0, mask_z_fraction=0.3) == 30
        assert trueFSC._z_pixels(100, 2.0) == 0
        assert trueFSC._z_pixels(100, 2.0, mask_z_fraction=1) == 0

    def test_the_automatic_mask_is_cut_to_the_central_sections(self, tmp_path):
        out = syn.run(
            tmp_path, syn.helix(n=48), mask_z_fraction=0.5, mask_soft=3, refine_mask=0
        )
        # the central half, and the 2-pixel soft edge (3 Angstrom) beyond it
        profile = out["mask"].max(axis=(1, 2))
        assert profile[:9].max() == 0 and profile[-9:].max() == 0
        assert profile[24] == 1

    def test_it_helps_a_helix_with_poorer_ends(self, tmp_path):
        maps = syn.helix(n=48)
        (tmp_path / "all").mkdir()
        (tmp_path / "central").mkdir()
        whole = syn.run(tmp_path / "all", maps)
        central = syn.run(tmp_path / "central", maps, mask_z_fraction=0.5)
        assert central["truth"] < whole["truth"]
        assert abs(central["measured"] - central["truth"]) < 0.15

    def test_the_command_line_reaches_the_computation(self, tmp_path):
        parser = argparse.ArgumentParser()
        trueFSC.add_args(parser)
        # the log goes next to the plot file
        plot = str(tmp_path / "fsc.pdf")
        args = parser.parse_args(["a.mrc", "b.mrc", plot, "--maskZfraction", "0.3"])
        args = trueFSC.check_args(args, parser)
        with patch.object(
            trueFSC, "compute_truefsc", return_value={"plot_file": None}
        ) as compute:
            trueFSC.main(args)
        kw = compute.call_args.kwargs
        assert kw["mask_z_fraction"] == 0.3 and kw["mask_z_len"] == -1
        args = parser.parse_args(["a.mrc", "b.mrc", "--maskZlen", "200"])
        assert args.maskZlen == 200

    def test_a_length_and_a_fraction_together_are_refused(self):
        parser = argparse.ArgumentParser()
        trueFSC.add_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["a.mrc", "b.mrc", "--maskZlen", "200", "--maskZfraction", "0.3"]
            )


class TestMaskSearch:
    def test_the_answer_matches_the_truth_and_keeps_the_weak_domain(self, tmp_path):
        out = syn.run(tmp_path, syn.particle(n=48))
        choice = out["result"]["mask"]
        assert choice["kept"] >= trueFSC.MIN_DENSITY_KEPT
        assert out["signal_kept"] > 0.98
        assert abs(out["measured"] - out["truth"]) < 0.15

    def test_an_automatic_threshold_is_searched_and_a_given_one_is_not(self, tmp_path):
        f1, f2 = tmp_path / "1.mrc", tmp_path / "2.mrc"
        S, n1, n2 = syn.particle(n=32)
        syn.write(f1, S + n1, 1.5)
        syn.write(f2, S + n2, 1.5)
        seen = []

        def fake_search(input_files, params, *args):
            thresholds = args[5]
            seen.append(len(thresholds))
            return dict(thresh=thresholds[0], width=2.0, kept=1.0, score=0.0)

        with patch.object(trueFSC, "_optimal_mask", fake_search):
            trueFSC.compute_truefsc(f1, f2, tmp_path / "a.pdf")
            trueFSC.compute_truefsc(
                f1, f2, tmp_path / "b.pdf", mask_fraction_thresh=0.2
            )
        assert seen == [len(trueFSC.THRESHOLD_FACTORS), 1]

    def test_force_deletes_the_cached_searches(self, tmp_path):
        f1, f2 = tmp_path / "1.mrc", tmp_path / "2.mrc"
        S, n1, n2 = syn.particle(n=32)
        syn.write(f1, S + n1, 1.5)
        syn.write(f2, S + n2, 1.5)
        cleared = []

        def fake_search(input_files, params, *args):
            return dict(thresh=args[5][0], width=2.0, kept=1.0, score=0.0)

        fake_search.clear_cache = lambda: cleared.append(True)
        with patch.object(trueFSC, "_optimal_mask", fake_search):
            trueFSC.compute_truefsc(f1, f2, tmp_path / "a.pdf")
            assert cleared == []
            trueFSC.compute_truefsc(f1, f2, tmp_path / "a.pdf", force=True)
        assert cleared == [True]

    def test_a_flattened_outside_is_not_taken_for_background(self):
        # RELION flattens what lies outside its spherical mask to a constant
        # near zero; counted as background, it would make noise pass for density
        n = 48
        noise = np.random.default_rng(0).normal(size=(n, n, n))
        z, y, x = np.indices(noise.shape)
        r = np.sqrt((x - n / 2) ** 2 + (y - n / 2) ** 2 + (z - n / 2) ** 2)
        noise[r > n / 2] = 0.01
        loose = np.zeros_like(noise)
        loose[20:28, 20:28, 20:28] = 1
        density = trueFSC._density_above_background(noise, loose, noise)
        # E[max(X - 2, 0)] of a standard normal is 0.0085
        assert density[r <= n / 2].mean() < 0.015

    def test_a_shell_without_a_correction_is_left_out(self):
        fsc_t = np.array([0.9, 0.5, 0.3])
        fsc_n = np.array([0.1, 1.0, 0.05])
        weights = np.array([1.0, 1.0, 1.0])
        with_nan = trueFSC._mask_score(fsc_t, fsc_n, weights)
        without = trueFSC._mask_score(fsc_t[[0, 2]], fsc_n[[0, 2]], weights[:2])
        assert with_nan == pytest.approx(without)

    def test_the_scored_shells_end_where_the_maps_stop_correlating(self):
        n = 64
        k2 = np.fft.fftfreq(n) ** 2
        kr2 = np.fft.rfftfreq(n) ** 2
        shell = np.round(
            np.sqrt(k2[:, None, None] + k2[None, :, None] + kr2[None, None, :]) * n
        ).astype(np.int32)
        shell = np.clip(shell, 0, n // 2).ravel()
        fsc = np.where(np.arange(n // 2 + 1) < 16, 0.9, 0.0)
        lo, hi, weights = trueFSC._score_shells(fsc, shell, 4)
        assert lo == 4 and hi == 20  # 16, and a quarter more
        assert len(weights) == hi - lo and np.all(np.diff(weights) > 0)


class TestThresholds:
    def test_a_pixel_threshold_is_applied_to_the_map_as_it_is(self):
        v = np.random.default_rng(0).normal(size=(16, 16, 16))
        assert np.array_equal(trueFSC._mask_volume(v, 1.0, 8.0, [0.5]), v)
        assert not np.array_equal(trueFSC._mask_volume(v, 1.0, 8.0, None), v)

    def test_separate_masks_take_their_own_pixel_threshold(self, tmp_path):
        f1, f2 = tmp_path / "1.mrc", tmp_path / "2.mrc"
        S, n1, n2 = syn.particle(n=32)
        syn.write(f1, S + n1, 1.5)
        syn.write(f2, S + n2, 1.5)
        used = []
        real = trueFSC._seeded_mask

        def spy(volume, thresh, z_pixels=0):
            used.append(thresh)
            return real(volume, thresh, z_pixels)

        with patch.object(trueFSC, "_seeded_mask", spy):
            trueFSC.compute_truefsc(
                f1,
                f2,
                tmp_path / "a.pdf",
                one_mask=0,
                mask_thresh=[0.5, 0.7],
                mask_soft=2,
                refine_mask=0,
            )
        assert used == [0.5, 0.7]


class TestRepeatable:
    def test_the_same_maps_give_the_same_answer(self, tmp_path):
        maps = syn.particle(n=32)
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        first = syn.run(tmp_path / "a", maps, mask_soft=3, refine_mask=0)
        np.random.seed(123)  # whatever the global random state
        second = syn.run(tmp_path / "b", maps, mask_soft=3, refine_mask=0)
        assert first["measured"] == second["measured"]


class TestArea:
    def test_a_step_curve_reads_as_its_edge(self):
        saxis = np.arange(51) / 100  # up to 0.5 1/A
        fsc = np.where(saxis <= 0.2, 1.0, 0.0)
        area, res = trueFSC._fsc_area(saxis, fsc)
        assert area == pytest.approx(0.205, abs=0.006)
        assert res == pytest.approx(1 / area)

    def test_it_is_reported(self, tmp_path):
        out = syn.run(tmp_path, syn.particle(n=32), mask_soft=3, refine_mask=0)
        r = out["result"]
        assert r["fsc_area"] > 0
        # a curve that falls gradually reaches less far on average than its
        # 0.143 crossing
        assert r["resolution_area"] > r["resolution"]
