"""The candidate twist/rise solutions of Denovo3D: each says what a click does, and a
click reconstructs a 3D map for it."""

import inspect

import numpy as np
import pytest

from helicon.webApps.tabs import denovo3d_tab as dn
from _source import contains


def _result(twist, rise, score):
    """A solver result as the tab keeps it: (score, return_data, params)."""
    params = [None] * 8
    params[5], params[6] = twist, rise
    return (score, None, tuple(params))


def _grid(twists, rises):
    return [
        _result(t, r, 0.1 * i + 0.01 * j)
        for i, t in enumerate(twists)
        for j, r in enumerate(rises)
    ]


class TestScoresFigure:
    def test_one_candidate_or_none_has_no_plot(self):
        assert dn._scores_figure([]) is None
        assert dn._scores_figure(_grid([1.0], [4.75])) is None

    def test_a_fixed_twist_and_rise_pair_repeated_has_no_plot(self):
        assert dn._scores_figure(_grid([1.0, 1.0], [4.75, 4.75])) is None

    def test_several_twists_make_a_line(self):
        fig = dn._scores_figure(_grid([1.0, 1.2, 1.4], [4.75]))
        assert fig.data[0].type == "scatter"
        assert "Twist" in fig.data[0].hovertemplate

    def test_several_rises_make_a_line_too(self):
        fig = dn._scores_figure(_grid([1.2], [4.7, 4.75, 4.8]))
        assert "Rise" in fig.data[0].hovertemplate

    def test_both_make_a_map(self):
        fig = dn._scores_figure(_grid([1.0, 1.2, 1.4], [4.7, 4.75, 4.8]))
        assert fig.data[0].type == "heatmap"

    @pytest.mark.parametrize(
        "twists,rises",
        [([1.0, 1.2], [4.75]), ([1.2], [4.7, 4.8]), ([1.0, 1.2, 1.4], [4.7, 4.8])],
    )
    def test_every_candidate_says_what_a_click_does(self, twists, rises):
        fig = dn._scores_figure(_grid(twists, rises))
        hover = fig.data[0].hovertemplate
        assert dn.CLICK_HINT in hover
        assert "3D map" in dn.CLICK_HINT and "click" in dn.CLICK_HINT.lower()

    def test_the_points_of_a_line_carry_their_twist_and_rise(self):
        fig = dn._scores_figure(_grid([1.4, 1.0, 1.2], [4.75]))
        cd = np.asarray(fig.data[0].customdata)
        assert cd.shape == (3, 2)
        # in the order of the plotted x (sorted), each with its own pair
        assert list(cd[:, 0]) == [1.0, 1.2, 1.4] and set(cd[:, 1]) == {4.75}

    def test_the_cells_of_the_map_carry_their_twist_and_rise(self):
        fig = dn._scores_figure(_grid([1.0, 1.2, 1.4], [4.7, 4.75]))
        cd = np.asarray(fig.data[0].customdata)
        assert cd.shape == (2, 3, 2)  # rows are rises, columns twists
        assert tuple(cd[1, 2]) == (1.4, 4.75)
        assert tuple(cd[0, 0]) == (1.0, 4.7)

    def test_the_candidate_that_was_reconstructed_is_marked(self):
        res = _grid([1.0, 1.2, 1.4], [4.7, 4.75])
        plain = dn._scores_figure(res)
        marked = dn._scores_figure(res, picked=(1.2, 4.75))
        assert len(marked.data) == len(plain.data) + 1
        assert marked.data[-1].mode == "markers"
        assert list(marked.data[-1].x) == [1.2] and list(marked.data[-1].y) == [4.75]

    def test_the_marker_of_a_line_sits_on_that_point(self):
        res = _grid([1.0, 1.2, 1.4], [4.75])
        marked = dn._scores_figure(res, picked=(1.2, 4.75))
        assert list(marked.data[-1].x) == [1.2]

    def test_the_marker_does_not_change_what_a_click_on_it_does(self):
        marked = dn._scores_figure(_grid([1.0, 1.2, 1.4], [4.7, 4.75]), (1.2, 4.75))
        assert marked.data[-1].hoverinfo == "skip"


class TestPickedPair:
    def test_a_click_gives_its_pair(self):
        assert dn._picked_pair({"twist": 1.21, "rise": 4.75, "nonce": 1}) == (
            1.21,
            4.75,
        )

    def test_numbers_as_text_are_taken(self):
        assert dn._picked_pair({"twist": "-1.2", "rise": "4.8"}) == (-1.2, 4.8)

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            {},
            {"twist": 1.0},
            {"twist": "x", "rise": 4.75},
            {"twist": float("nan"), "rise": 4.75},
            {"twist": 1.0, "rise": float("inf")},
            {"twist": 1.0, "rise": 0},
            {"twist": 1.0, "rise": -4.75},
            {"twist": 270.0, "rise": 4.75},
        ],
    )
    def test_anything_else_is_refused(self, payload):
        assert dn._picked_pair(payload) is None


class TestClickReachesTheApp:
    def test_a_figure_without_a_click_input_sends_nothing(self):
        html = str(dn._fig_to_html(dn._scores_figure(_grid([1.0, 1.2], [4.75]))))
        assert (
            "Shiny.setInputValue('" not in html and "gd.on('plotly_click'" not in html
        )

    def test_the_click_sends_the_twist_and_rise_of_the_point(self):
        html = str(
            dn._fig_to_html(
                dn._scores_figure(_grid([1.0, 1.2], [4.75])),
                click_input="denovo3d-dn_pick_candidate",
            )
        )
        assert "gd.on('plotly_click'" in html
        assert "Shiny.setInputValue('denovo3d-dn_pick_candidate'" in html
        assert "p.customdata[0]" in html and "p.customdata[1]" in html
        # every click is an event, also on the same point twice
        assert "priority: 'event'" in html

    def test_the_plot_is_drawn_with_the_click_input_of_the_module(self):
        src = inspect.getsource(dn)
        plot = src[
            src.index("def dn_scores_plot") : src.index(
                "def dn_reconstructed_projections"
            )
        ]
        assert 'click_input=session.ns("dn_pick_candidate")' in plot


class TestClickStartsTheReconstruction:
    def _src(self):
        return inspect.getsource(dn)

    def test_the_click_reconstructs_the_pair_without_touching_the_ranges(self):
        src = self._src()
        effect = src[
            src.index("def _reconstruct_picked_candidate") : src.index(
                "@reactive.extended_task"
            )
        ]
        assert "_picked_pair(input.dn_pick_candidate())" in effect
        assert "_start_reconstruction(pair)" in effect
        assert "dn_twist_range" not in effect and "update_slider" not in effect

    def test_a_running_calculation_is_not_started_over(self):
        src = self._src()
        effect = src[src.index("def _reconstruct_picked_candidate") :][:900]
        assert '_reconstruction_task.status() == "running"' in effect

    def test_a_pair_replaces_the_ranges_in_the_run(self):
        src = self._src()
        run = src[
            src.index("def _start_reconstruction") : src.index(
                "def _reconstruct_picked_candidate"
            )
        ]
        assert contains(run, "if pair is not None:\n            tr_pairs = [pair]")
        # one pair is what asks the solver for the map
        assert "return_3d = n_pairs == 1" in run

    def test_the_search_button_still_runs_the_ranges(self):
        src = self._src()
        effect = src[src.index("def _run_denovo3D_reconstruction") :][:200]
        assert "_start_reconstruction()" in effect

    def test_the_candidates_of_a_search_are_kept_while_one_is_reconstructed(self):
        src = self._src()
        assert "scan_results = reactive.value([])" in src
        run = src[
            src.index("def _start_reconstruction") : src.index(
                "def _reconstruct_picked_candidate"
            )
        ]
        # only a new search forgets them
        assert contains(run, "if pair is None:\n            # a new search")
        assert "is_scan=pair is None and n_pairs > 1" in run
        task = src[src.index("async def _reconstruction_task") :]
        assert contains(task, "if is_scan:\n                scan_results.set(")

    def test_the_plot_shows_the_kept_candidates_with_the_picked_one_marked(self):
        src = self._src()
        plot = src[
            src.index("def dn_scores_plot") : src.index(
                "def dn_reconstructed_projections"
            )
        ]
        assert "scan_results()" in plot and "picked = (" in plot

    def test_a_new_selection_forgets_the_candidates(self):
        src = self._src()
        assert contains(
            src, "reconstruction_results.set([])\n        scan_results.set([])"
        )
