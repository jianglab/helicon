"""The pictures shown after a search must come from the reconstruction solver.

A twist/rise search scores with the search solver, but what it puts on screen is
a reconstruction, and the user judges it by comparing it against the input
image. Drawing it with the search solver shows a map the reconstruction selector
was never going to produce -- gauss, for instance, fills the hollow core that
elasticnet carves out, so the comparison looks wrong for a reason that has
nothing to do with the twist being searched for.

So when the two selectors differ, the displayed pairs are re-solved with the
reconstruction solver and only their images are kept. These tests pin the part
that decides what to re-solve and where each answer goes back; the executor that
runs them is a thin loop around denovo3d_pipeline.process_one_task.
"""

import pytest

from helicon.webApps.tabs.denovo3d_tab import _display_key, _display_redraw_plan


def _params(twist, rise, image_index):
    """A solver result's third element: (data, file, imageIndex, apix3d, apix2d,
    twist, rise, csym, tilt, psi, dy)."""
    return (None, "f.mrc", image_index, 5.0, 5.0, twist, rise, 1, 0, 0, 0)


def _result(twist, rise, image_index, score):
    return (score, f"images({twist},{image_index})", _params(twist, rise, image_index))


def _task(twist, rise, image_index, model="gauss"):
    """A process_one_task argument tuple: only indices 4, 5, 6 and -3 matter
    here, so the rest is padding of the right length."""
    t = [None] * 38
    t[4] = image_index
    t[5] = twist
    t[6] = rise
    t[-3] = {"model": model, "l1_ratio": 0.5}
    return tuple(t)


RISE = 4.75


class TestDisplayKey:
    def test_matches_a_task_built_from_the_same_pair(self):
        assert _display_key(_params(1.2, RISE, "7")) == (1.2, RISE, "7")

    def test_survives_a_float_round_trip(self):
        # Twist reaches the solver as a rounded float and comes back through
        # the result; keying on raw equality would be fragile.
        assert _display_key(_params(1.2000000001, RISE, "7")) == _display_key(
            _params(1.2, RISE, "7")
        )

    def test_separates_images_at_the_same_twist(self):
        assert _display_key(_params(1.2, RISE, "7")) != _display_key(
            _params(1.2, RISE, "8")
        )


class TestRedrawPlan:
    def setup_method(self):
        self.twists = [1.0, 1.1, 1.2, 1.3]
        self.images = ["7", "8"]
        self.results = [
            _result(t, RISE, im, score=(1.0 if t == 1.2 else 0.5))
            for t in self.twists
            for im in self.images
        ]
        self.tasks = [_task(t, RISE, im) for t in self.twists for im in self.images]
        # Ranked best-first; one entry per pair, as _rank produces.
        self.ranked = [
            _result(t, RISE, self.images[0], score=(1.0 if t == 1.2 else 0.5))
            for t in [1.2, 1.1, 1.0, 1.3]
        ]

    def test_redraws_only_the_displayed_pairs(self):
        redo, _ = _display_redraw_plan(
            self.tasks, self.results, self.ranked, top_n=2, display_model="elasticnet"
        )
        # Top 2 pairs (1.2 and 1.1), both images each.
        assert len(redo) == 4
        assert {round(t[5], 6) for t in redo} == {1.2, 1.1}
        assert {t[4] for t in redo} == {"7", "8"}

    def test_swaps_the_model_and_leaves_other_settings_alone(self):
        redo, _ = _display_redraw_plan(
            self.tasks, self.results, self.ranked, top_n=1, display_model="elasticnet"
        )
        assert redo
        for t in redo:
            assert t[-3]["model"] == "elasticnet"
            assert t[-3]["l1_ratio"] == 0.5

    def test_does_not_mutate_the_original_tasks(self):
        _display_redraw_plan(
            self.tasks, self.results, self.ranked, top_n=4, display_model="elasticnet"
        )
        assert all(t[-3]["model"] == "gauss" for t in self.tasks)

    def test_top_n_zero_means_every_pair(self):
        redo, _ = _display_redraw_plan(
            self.tasks, self.results, self.ranked, top_n=0, display_model="elasticnet"
        )
        assert len(redo) == len(self.tasks)

    def test_top_n_beyond_the_ranking_is_harmless(self):
        redo, _ = _display_redraw_plan(
            self.tasks, self.results, self.ranked, top_n=99, display_model="elasticnet"
        )
        assert len(redo) == len(self.tasks)

    def test_skips_tasks_whose_result_is_missing(self):
        # A task can be discarded by the pipeline (returns None), and then there
        # is no slot to put a redrawn image into.
        results = [r for r in self.results if _display_key(r[2]) != (1.2, RISE, "8")]
        redo, slot = _display_redraw_plan(
            self.tasks, results, self.ranked, top_n=1, display_model="elasticnet"
        )
        assert len(redo) == 1
        assert redo[0][4] == "7"
        assert (1.2, RISE, "8") not in slot

    def test_slot_points_at_the_matching_result(self):
        _, slot = _display_redraw_plan(
            self.tasks, self.results, self.ranked, top_n=1, display_model="elasticnet"
        )
        for key, i in slot.items():
            assert _display_key(self.results[i][2]) == key

    def test_splicing_keeps_the_search_score(self):
        # This is the contract the caller relies on: the ranking the user sees
        # stays the one the search computed; only the images change.
        redo, slot = _display_redraw_plan(
            self.tasks, self.results, self.ranked, top_n=1, display_model="elasticnet"
        )
        results = list(self.results)
        for t in redo:
            fresh = (0.123, "REDRAWN", _params(t[5], t[6], t[4]))
            i = slot[_display_key(fresh[2])]
            results[i] = (results[i][0], fresh[1], results[i][2])
        redrawn = [r for r in results if r[1] == "REDRAWN"]
        assert len(redrawn) == 2
        assert all(r[0] == 1.0 for r in redrawn)
        assert all(r[1] != "REDRAWN" for r in results if r[2][5] != 1.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
