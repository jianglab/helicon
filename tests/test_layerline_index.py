"""Finding the csym, twist and rise of a helix from the layer lines of its images.

The checks use a simulated helix (its true parameters are known) on a small grid,
so each takes a few seconds: the parity of the orders of the layer lines, the
way images add, and the search itself.
"""

import numpy as np
import pytest

from helicon.lib import layerlines
from helicon.webApps.lib import hill_compute
from helicon.webApps.lib import layerline_index as li
from _source import tab_source

APIX = 4.6876  # a box of 128 pixels as wide as 256 at the pixel size of HILL's default
CUTOFF = 14.0


def _helix(twist=29.4, rise=21.92, csym=6, radius=77.6, noise=0.0, seed=0):
    img = hill_compute.simulate_helix(
        twist, rise, csym, radius, 8.0, 128, 128, APIX, az0=17.0
    )
    img = img - img.mean()
    if noise:
        img = img + np.random.default_rng(seed).normal(0, noise * img.std(), img.shape)
    return img


def _lines(twist, rise, csym):
    return layerlines.layer_line_heights(
        twist, rise, csym, y_max=1.0 / CUTOFF, radius=None, x_max=None
    )


class TestParity:
    def test_an_even_csym_has_all_its_lines_at_zero_degrees(self):
        spectra = li.prepare([_helix(csym=6)], APIX, CUTOFF)
        heights, orders = _lines(29.4, 21.92, 6)
        assert li.parity_agreement(spectra, heights, orders) > 0.8

    def test_the_same_line_heights_with_odd_orders_are_refused(self):
        # C3 at twice the twist puts its lines at the same heights, with odd orders
        # on alternate lines, which the phases of a C6 helix do not have
        spectra = li.prepare([_helix(csym=6)], APIX, CUTOFF)
        heights, orders = _lines(58.8, 21.92, 3)
        assert li.parity_agreement(spectra, heights, orders) < li.MIN_PARITY

    def test_an_odd_csym_has_lines_at_180_degrees(self):
        spectra = li.prepare([_helix(twist=14.0, rise=21.9, csym=3)], APIX, CUTOFF)
        heights, orders = _lines(14.0, 21.9, 3)
        right = li.parity_agreement(spectra, heights, orders)
        # taking every order as even loses the lines that are at 180 degrees
        wrong = li.parity_agreement(spectra, heights, 2 * orders)
        assert right > 0.7
        assert wrong < right - 0.3

    def test_a_shift_along_the_axis_makes_no_difference(self):
        img = _helix(csym=6)
        shifted = np.roll(img, 17, axis=0)
        heights, orders = _lines(29.4, 21.92, 6)
        a = li.parity_agreement(li.prepare([img], APIX, CUTOFF), heights, orders)
        b = li.parity_agreement(li.prepare([shifted], APIX, CUTOFF), heights, orders)
        assert b == pytest.approx(a, abs=0.05)

    def test_a_shift_across_the_axis_is_taken_out(self):
        img = _helix(csym=6)
        moved = np.roll(img, 9, axis=1)
        heights, orders = _lines(29.4, 21.92, 6)
        assert (
            li.parity_agreement(li.prepare([moved], APIX, CUTOFF), heights, orders)
            > 0.8
        )


class TestPrepare:
    def test_it_refuses_nothing(self):
        with pytest.raises(ValueError, match="no images"):
            li.prepare([], APIX, CUTOFF)

    def test_it_refuses_images_of_different_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            li.prepare([np.zeros((64, 64)), np.zeros((64, 32))], APIX, CUTOFF)

    def test_images_add(self):
        one = li.prepare([_helix(noise=2.0, seed=1)], APIX, CUTOFF)
        both = li.prepare(
            [_helix(noise=2.0, seed=1), _helix(noise=2.0, seed=2)], APIX, CUTOFF
        )
        assert both.n_images == 2
        assert both.excess.shape == one.excess.shape
        # the signal adds and the noise partly cancels: the lines stand out more
        assert (
            both.signed[np.argmax(np.abs(both.signed))].real
            > 1.3 * np.abs(one.signed).max()
        )


class TestSearch:
    SEARCH = dict(
        csyms=(3, 6),
        twists=(10.0, 50.0),
        rises=(15.0, 30.0),
        step_twist=2.0,
        step_rise=1.0,
        top=3,
        lattices=3,
        refine=False,
    )

    def test_it_finds_the_helix(self):
        found = li.index_images([_helix(noise=1.0)], APIX, 77.6, CUTOFF, **self.SEARCH)
        best = found[0]
        assert best.csym == 6
        assert best.twist == pytest.approx(29.4, abs=2.0)
        assert best.rise == pytest.approx(21.92, abs=1.0)
        assert best.score > 0
        assert best.parity > li.MIN_PARITY

    def test_the_odd_csym_with_the_same_line_spacing_is_not_the_answer(self):
        found = li.index_images([_helix(noise=1.0)], APIX, 77.6, CUTOFF, **self.SEARCH)
        assert not any(c.csym == 3 and c.twist > 50 for c in found[:1])

    def test_the_candidates_are_best_first_and_the_twist_is_positive(self):
        found = li.index_images([_helix(noise=1.0)], APIX, 77.6, CUTOFF, **self.SEARCH)
        scores = [c.score for c in found]
        assert scores == sorted(scores, reverse=True)
        assert all(c.twist > 0 and c.rise > 0 for c in found)

    def test_the_models_are_the_documented_ones(self):
        assert set(li.MODELS) == {"orders", "mask", "lines", "onset", "bessel"}

    def test_the_model_is_chosen_by_name(self):
        images = [_helix(noise=1.0)]
        with pytest.raises(ValueError, match="model"):
            li.index_images(images, APIX, 77.6, CUTOFF, model="magic", **self.SEARCH)
        for model in li.MODELS:
            found = li.index_images(
                images, APIX, 77.6, CUTOFF, model=model, **self.SEARCH
            )
            assert found and found[0].csym in (3, 6), model

    def test_several_images_are_accepted(self):
        images = [_helix(noise=3.0, seed=s) for s in range(3)]
        found = li.index_images(images, APIX, 77.6, CUTOFF, **self.SEARCH)
        assert found[0].csym == 6

    def test_without_real_threads_several_workers_still_search_serially(
        self, monkeypatch
    ):
        """As under Pyodide: ProcessPoolExecutor cannot fork/spawn there, so
        asking for several workers must still give the one-process result."""
        from helicon.lib import system

        images = [_helix(noise=1.0)]
        serial = li.index_images(images, APIX, 77.6, CUTOFF, workers=1, **self.SEARCH)
        monkeypatch.setattr(system, "_HAS_REAL_THREADS", False)
        many = li.index_images(images, APIX, 77.6, CUTOFF, workers=4, **self.SEARCH)
        assert [(c.csym, c.twist, c.rise) for c in many] == [
            (c.csym, c.twist, c.rise) for c in serial
        ]


class TestSymmetriesOfOneLattice:
    """C4, C6 and C8 at the same csym * twist put their lines at the same heights;
    only the Bessel order of each line (4k, 6k, 8k) tells them apart."""

    @staticmethod
    def _family(images, apix=2.3438, rise=21.92, product=176.4, radius=77.6):
        spectra = li.prepare(images, apix, 12.0)
        scores = {}
        for csym in (2, 4, 6, 8, 12):
            scores[csym] = li._score(
                spectra, csym, product / csym, rise, [radius], model="orders"
            )[0]
        return scores

    def test_the_lines_are_at_the_same_heights(self):
        heights = {}
        for csym in (4, 6, 8):
            h, _ = _lines(176.4 / csym, 21.92, csym)
            heights[csym] = np.unique(np.round(h, 6))
        assert np.array_equal(heights[4], heights[6])
        assert np.array_equal(heights[6], heights[8])

    @pytest.mark.parametrize("csym", [4, 6, 8])
    def test_the_measured_orders_pick_the_symmetry(self, csym):
        img = hill_compute.simulate_helix(
            176.4 / csym, 21.92, csym, 77.6, 8.0, 256, 256, 2.3438, az0=17.0
        )
        img = img - img.mean()
        noisy = [
            img + np.random.default_rng(s).normal(0, img.std(), img.shape)
            for s in range(2)
        ]
        scores = self._family(noisy)
        assert max(scores, key=scores.get) == csym, scores


class TestHillCandidates:
    """What HILL's 'Suggest csym, twist and rise' shows and uses."""

    def test_the_table_lists_the_candidates_best_first_in_the_users_hand(self):
        found = [
            li.Candidate(6, 30.0, 21.75, 10.0, 5.0, 0.9, 77.6),
            li.Candidate(8, 22.5, 21.75, 8.0, 4.0, 0.9, 77.6),
        ]
        right = hill_compute.candidates_table(found, twist_sign=1.0)
        left = hill_compute.candidates_table(found, twist_sign=-1.0)
        assert list(right["Rank"]) == [1, 2]
        assert list(right["Csym"]) == [6, 8]
        assert list(right["Twist (°)"]) == [30.0, 22.5]
        assert list(left["Twist (°)"]) == [-30.0, -22.5]
        assert list(right["Score (%)"]) == [100, 80]
        assert right["Pitch (Å)"][0] == pytest.approx(360 * 21.75 / 30.0, abs=0.1)
        assert list(right["Group"]) == ["A", "A"]

    @staticmethod
    def _found():
        c = li.Candidate
        return [
            c(6, 30.0, 21.75, 10.0, 5.0, 0.9, 77.6),  # A
            c(2, 60.0, 14.5, 9.0, 4.0, 0.9, 77.6),  # another lattice: B
            c(8, 22.5, 21.75, 8.0, 4.0, 0.9, 77.6),  # A: 8 x 22.5 = 6 x 30
            c(4, 45.0, 21.8, 7.0, 4.0, 0.9, 77.6),  # A, within the tolerances
            c(3, 60.0, 21.75, 6.0, 4.0, 0.5, 77.6),  # A: 3 x 60 = 180
            c(4, 60.0, 14.5, 5.0, 4.0, 0.9, 77.6),  # 240: not B
        ]

    def test_candidates_with_the_same_lines_share_a_letter(self):
        assert hill_compute.same_line_groups(self._found()) == list("ABAAAC")
        assert hill_compute.same_line_groups([]) == []

    def test_the_ambiguity_names_the_alternatives_of_the_best(self):
        assert hill_compute.csym_ambiguity(self._found()) == [
            (1, 6),
            (3, 8),
            (4, 4),
            (5, 3),
        ]
        alone = [li.Candidate(6, 30.0, 21.75, 10.0, 5.0, 0.9, 77.6)]
        assert hill_compute.csym_ambiguity(alone) is None

    def test_the_alternatives_of_the_best_are_always_listed(self):
        c = li.Candidate
        found = [c(6, 30.0, 21.75, 10.0, 5.0, 0.9, 77.6)]
        found += [c(2, 60.0 + k, 14.5, 9.0 - k * 0.1, 4.0, 0.9, 77.6) for k in range(8)]
        found += [c(8, 22.5, 21.75, 5.0, 4.0, 0.9, 77.6)]  # rank 10, same lines
        ranks, shown = hill_compute.candidates_to_show(found, n=3)
        assert ranks == [1, 2, 3, 10]
        assert shown[-1].csym == 8
        table = hill_compute.candidates_table(shown, ranks=ranks)
        assert list(table["Rank"]) == [1, 2, 3, 10]
        assert list(table["Group"])[0] == list(table["Group"])[-1] == "A"
        assert hill_compute.csym_ambiguity(shown, ranks) == [(1, 6), (10, 8)]

    def test_the_tab_warns_and_says_how_to_evaluate(self):
        from helicon.webApps.tabs import hill_tab

        group = hill_compute.csym_ambiguity(self._found())
        text = hill_tab._csym_warning_text(group)
        assert "C6 (rank 1)" in text and "C8 (rank 3)" in text and "C4 (rank 4)" in text
        assert "same heights" in text and "Evaluate each" in text
        assert "does not rule one out" in text
        # an odd member: the phases across the meridian are the better check
        assert "phases across the meridian" in text
        even = hill_tab._csym_warning_text([(1, 6), (2, 8)])
        assert "phases across the meridian" not in even
        src = tab_source(hill_tab)
        assert 'ui.output_ui("hill_symmetry_warning")' in src
        assert 'class_="alert alert-warning"' in src

    def test_no_candidates_is_an_empty_table(self):
        assert len(hill_compute.candidates_table([])) == 0

    def test_images_are_stood_upright_by_their_record_or_automatically(self):
        upright = _helix(noise=0.0)  # axis along the rows already
        turned = np.ascontiguousarray(upright.T)  # lying down
        out = hill_compute.images_for_indexing(
            [upright, turned],
            [dict(angle=0.0, dx=0.0, dy=0.0, negate=False), None],
            APIX,
        )
        assert out[0] is not None and np.allclose(out[0], upright)
        # the one without a record was turned upright: its projection along the
        # rows is sharp again
        sharp = lambda im: float((im.sum(axis=0) ** 2).sum())  # noqa: E731
        assert sharp(out[1]) > 0.8 * sharp(upright)
        assert sharp(turned) < 0.5 * sharp(upright)

    def test_the_tab_offers_it_and_applies_a_choice(self):
        import re

        from helicon.webApps.tabs import hill_tab

        src = tab_source(hill_tab)
        assert re.search(r'ui\.input_action_button\(\s*"hill_suggest_symmetry"', src)
        assert 'ui.output_data_frame("hill_symmetry_candidates")' in src
        # the search uses the diameter and the resolution limit the user sees
        assert "input.hill_diameter() / 2" in src
        assert "input.hill_cutoff_res_x()" in src
        # a chosen row sets the three values HILL draws its layer lines from
        for name in ("hill_csym", "hill_rise", "hill_twist"):
            assert f'ui.update_numeric("{name}"' in src


class TestExtraLines:
    """A lattice with the lines of the helix and others between them is not the helix."""

    RADIUS = 77.6

    def _precision(self, spectra, twist, rise, csym):
        heights, orders = layerlines.layer_line_heights(
            twist,
            rise,
            csym,
            y_max=1.0 / CUTOFF,
            radius=self.RADIUS,
            x_max=1.0 / CUTOFF,
        )
        return spectra.line_precision(heights, orders, self.RADIUS)

    def test_the_lattice_of_the_helix_has_all_its_lines_and_a_finer_one_does_not(self):
        # the same twist and rise with half the symmetry: C6's lines and, between
        # them, the odd multiples of 3
        spectra = li.prepare([_helix()], APIX, CUTOFF)
        assert self._precision(spectra, 30.0, 21.92, 6) == pytest.approx(1.0)
        assert self._precision(spectra, 30.0, 21.92, 3) < 0.6

    def test_noise_hides_weak_lines_of_both_alike(self):
        spectra = li.prepare([_helix(noise=1.0)], APIX, CUTOFF)
        right = self._precision(spectra, 30.0, 21.92, 6)
        finer = self._precision(spectra, 30.0, 21.92, 3)
        assert right > 1.5 * finer

    def test_a_helix_with_no_readable_line_loses_nothing(self):
        spectra = li.prepare([_helix()], APIX, CUTOFF)
        assert spectra.line_precision(np.array([]), np.array([]), self.RADIUS) == 1.0

    def test_the_finer_lattice_is_not_a_candidate_with_a_score_near_the_helix(self):
        found = li.index_images(
            [_helix(noise=0.5)],
            APIX,
            self.RADIUS,
            CUTOFF,
            csyms=(3, 6),
            twists=(10.0, 50.0),
            rises=(15.0, 30.0),
            step_twist=2.0,
            step_rise=1.0,
            top=8,
            lattices=6,
            refine=False,
            workers=1,
        )
        best = found[0]
        assert (best.csym, round(best.twist)) == (6, 30)
        for c in found[1:]:
            near = abs(c.rise - best.rise) < 1.5 and abs(c.csym * c.twist - 180) < 8
            if c.csym == 3 and near:
                assert c.score < 0.5 * best.score

    def test_the_cost_is_documented_and_squared(self):
        assert li.PRECISION_POWER == 2.0
        assert li.LINE_Z_MIN == 3.0


class TestRadialDensity:
    """The radial density of the helix from the axial average of its images."""

    @staticmethod
    def _tube(radius=60.0, nx=128, ny=64, apix=4.0, offset=0.0):
        x = (np.arange(nx) - nx // 2 - offset) * apix
        # the projection of a thin shell: limb brightening at +-radius
        inside = np.abs(x) < radius
        profile = np.where(
            inside, 1.0 / np.sqrt(np.maximum(radius**2 - x**2, 1.0)), 0.0
        )
        profile = np.minimum(profile, 3.0 / radius)
        return np.tile(profile, (ny, 1))

    def test_a_shell_has_its_density_at_its_radius(self):
        r, rho = li.radial_density([self._tube(60.0)], 4.0, 8.0)
        assert r[0] == 0.0 and r[1] == pytest.approx(4.0)
        assert r[np.argmax(rho)] == pytest.approx(60.0, abs=8.0)
        assert rho[r > 90.0].max() < 0.1 * rho.max()

    def test_a_thick_tube_is_recovered_with_its_edges(self):
        # a uniform tube between 40 and 70 A, projected along y: the inverse of that
        # is the density, as far as the resolution limit allows
        apix, nx, cut = 2.0, 160, 6.0
        x = (np.arange(nx) - nx // 2) * apix
        y = np.arange(-200.0, 200.0, 0.25)
        rr = np.hypot(x[:, None], y[None, :])
        profile = ((rr >= 40.0) & (rr <= 70.0)).sum(axis=1) * 0.25
        r, rho = li.radial_density([np.tile(profile, (32, 1))], apix, cut)
        mass = np.cumsum(np.maximum(rho, 0.0) * r)
        mass /= mass[-1]
        assert np.interp(0.05, mass, r) == pytest.approx(41.0, abs=3.0)
        assert np.interp(0.95, mass, r) == pytest.approx(69.0, abs=3.0)
        inside = (r > 45.0) & (r < 65.0)
        assert rho[inside].min() > 0.8 * rho[inside].max()
        assert np.abs(rho[r < 30.0]).max() < 0.1 * rho.max()

    def test_an_off_centre_filament_is_centred_first(self):
        _, rho = li.radial_density([self._tube(60.0, offset=3.0)], 4.0, 8.0)
        r = np.arange(len(rho)) * 4.0
        assert r[np.argmax(rho)] == pytest.approx(60.0, abs=8.0)

    def test_the_profile_of_an_order_follows_the_density(self):
        from scipy.special import jv

        spectra = li.prepare([_helix()], APIX, CUTOFF)
        spectra.density = (
            np.arange(0, 160.0, 2.0),
            np.exp(-0.5 * ((np.arange(0, 160.0, 2.0) - 70.0) / 2.0) ** 2),
        )
        fx = np.linspace(0.005, 0.07, 30)
        g = spectra.order_profile(6, fx)
        shell = jv(6, 2.0 * np.pi * 70.0 * fx)
        assert np.corrcoef(g, shell)[0, 1] > 0.98

    def test_the_radius_is_what_the_search_uses_unless_asked(self):
        assert li.prepare([_helix()], APIX, CUTOFF).density is None
        assert li.prepare([_helix()], APIX, CUTOFF, radial=True).density is not None


class TestMembersOfALattice:
    """C4, C6 and C8 at one csym * twist have the same lines: the wrong ones keep little."""

    def test_the_share_a_member_keeps_is_the_square_of_its_agreement(self):
        assert li.MEMBER_POWER == 2.0
        found = li.index_images(
            [_helix(noise=0.5)],
            APIX,
            77.6,
            CUTOFF,
            csyms=(3, 4, 6, 8),
            twists=(10.0, 50.0),
            rises=(15.0, 30.0),
            step_twist=2.0,
            step_rise=1.0,
            top=8,
            lattices=3,
            refine=False,
            workers=1,
        )
        best = found[0]
        assert best.csym == 6
        same_lines = [
            c
            for c in found[1:]
            if abs(c.rise - best.rise) < 1.0
            and abs(c.csym * c.twist - 6 * best.twist) < 6
        ]
        assert same_lines
        for c in same_lines:
            ratio = max(c.profile, 0.0) / best.profile
            # score = lattice score x parity x ratio**2: far below the best
            assert c.score <= best.score * min(ratio, 1.0) ** 2 * 1.05


class TestPhaseDifferenceQuality:
    """How clean the phase difference across the meridian is says how far an image is believed."""

    def test_a_clean_image_has_a_cleaner_phase_difference_than_a_noisy_one(self):
        clean = li._centred_cross_meridian(_helix(), CUTOFF, APIX)[3]
        noisy = li._centred_cross_meridian(_helix(noise=3.0), CUTOFF, APIX)[3]
        assert -1.0 <= noisy < clean <= 1.0
        assert clean > 0.5

    def test_unweighted_pooling_counts_every_image_alike(self):
        images = [_helix(noise=0.5, seed=s) for s in range(3)]
        spectra = li.prepare(images, APIX, CUTOFF)
        assert spectra.n_images == 3

    def test_weighting_by_the_phase_difference_lets_the_clean_image_outweigh(self):
        clean, noisy = _helix(noise=0.3), _helix(noise=4.0, seed=1)
        flat = li.prepare([clean, noisy], APIX, CUTOFF)
        weighted = li.prepare([clean, noisy], APIX, CUTOFF, pd_weight="image")
        # the effective number of images falls when one is believed more
        assert weighted.n_images < flat.n_images
        assert weighted.n_images > 0

    def test_search_takes_the_option(self):
        found = li.index_images(
            [_helix(noise=0.5), _helix(noise=0.5, seed=1)],
            APIX,
            77.6,
            CUTOFF,
            csyms=(6,),
            twists=(20.0, 40.0),
            rises=(18.0, 26.0),
            step_twist=2.0,
            step_rise=1.0,
            top=2,
            lattices=2,
            refine=False,
            workers=1,
            pd_weight="image",
        )
        assert found and found[0].csym == 6


class TestVectorizedFirstPass:
    """The vectorized first pass matches per-candidate mask scoring and runs fast on one core."""

    @pytest.mark.parametrize(
        "shape,apix,cutoff",
        [
            ((128, 128), APIX, CUTOFF),
            ((127, 125), 2.5, 10.0),
            ((64, 64), 2.5, 5.0),
        ],
    )
    def test_first_pass_matches_single_candidate_mask_scores(self, shape, apix, cutoff):
        ny, nx = shape
        img = hill_compute.simulate_helix(
            29.4, 21.92, 6, 60.0, 8.0, ny, nx, apix, az0=17.0
        )
        spectra = li.prepare([img - img.mean()], apix, cutoff)
        assert spectra._ring.dtype == np.dtype(np.intp)

        twists = [5.0, 18.0, 29.4, 45.0]
        rises = [8.0, 15.0, 21.92, 30.0]
        for csym in (1, 3, 6):
            fast = {
                (c, t, r): (sc, ct)
                for sc, ct, c, t, r in li._first_pass(
                    spectra, csym, twists, rises, 60.0
                )
            }
            ref = {}
            for t in twists:
                for r in rises:
                    sc, ct = li._score(spectra, csym, t, r, [60.0], model="mask")[:2]
                    if sc > 0:
                        ref[(csym, t, r)] = (sc, ct)
            assert set(fast) == set(ref)
            for key in ref:
                assert fast[key][0] == pytest.approx(ref[key][0], rel=1e-10, abs=1e-10)
                assert fast[key][1] == pytest.approx(ref[key][1], rel=1e-10, abs=1e-10)

    def test_spectra_with_cached_mask_tables_is_picklable(self):
        import pickle

        spectra = li.prepare([_helix()], APIX, CUTOFF)
        li._first_pass(spectra, 6, [29.4], [21.92], 77.6)
        restored = pickle.loads(pickle.dumps(spectra))
        assert li._first_pass(restored, 6, [29.4], [21.92], 77.6) == pytest.approx(
            li._first_pass(spectra, 6, [29.4], [21.92], 77.6)
        )

    def test_default_hill_grid_runs_under_four_seconds_on_one_worker(self):
        import time

        apix = 2.3438
        img = hill_compute.simulate_helix(
            29.4, 21.92, 6, 77.6, 8.0, 256, 256, apix, az0=17.0
        )
        img = img - img.mean()
        noisy = [
            img + np.random.default_rng(s).normal(0, 0.8 * img.std(), img.shape)
            for s in range(2)
        ]
        t0 = time.perf_counter()
        found = li.index_images(noisy, apix, 77.6, 7.03, top=40, workers=1)
        elapsed = time.perf_counter() - t0
        assert elapsed < 4.0, f"Single-worker HILL search took {elapsed:.2f}s"
        assert found[0].csym == 6
        assert found[0].twist == pytest.approx(29.4, abs=1.5)
        assert found[0].rise == pytest.approx(21.92, abs=1.0)
