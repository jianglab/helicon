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
        import inspect
        import re

        from helicon.webApps.tabs import hill_tab

        src = inspect.getsource(hill_tab)
        assert re.search(r'ui\.input_action_button\(\s*"hill_suggest_symmetry"', src)
        assert 'ui.output_data_frame("hill_symmetry_candidates")' in src
        # the search uses the diameter and the resolution limit the user sees
        assert "input.hill_diameter() / 2" in src
        assert "input.hill_cutoff_res_x()" in src
        # a chosen row sets the three values HILL draws its layer lines from
        for name in ("hill_csym", "hill_rise", "hill_twist"):
            assert f'ui.update_numeric("{name}"' in src
