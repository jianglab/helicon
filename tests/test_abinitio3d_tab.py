"""The AbInitio3D tab, split out of HelicalPitch, and how it is reached.

The class-azimuth workflow (ring, repeat, star export, 3D maps) was the
"Inter-class" view of HelicalPitch; it is now a tab of its own with its own
inputs, and HelicalPitch is back to the same-class pair-distance histogram.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from helicon.webApps.tabs import abinitio3d_tab, helical_pitch_tab

TABS = Path(__file__).resolve().parents[1] / "src/helicon/webApps/tabs"


def _declared(module):
    src = Path(module.__file__).read_text()
    return set(
        re.findall(
            r'(?:ui\.input_\w+|helicon\.shiny\.(?:range_)?slider)\(\s*"(\w+)"', src
        )
    )


def _read(module):
    return set(re.findall(r"input\.(\w+)\b", Path(module.__file__).read_text()))


class TestTheSplit:
    def test_helical_pitch_has_no_inter_class_controls(self):
        src = Path(helical_pitch_tab.__file__).read_text()
        for name in ("phase_run", "map_run", "relion_run", "pitch_band", "rot_fold"):
            assert f'"{name}"' not in src
        assert "Inter-class" not in src
        assert "helical_pitch_phase" not in src

    def test_helical_pitch_keeps_its_histogram_and_auto_run(self):
        declared = _declared(helical_pitch_tab)
        for name in ("min_len", "max_len", "bins", "rise", "run"):
            assert name in declared
        assert "_auto_run_at_start" in Path(helical_pitch_tab.__file__).read_text()

    def test_abinitio3d_has_its_own_inputs(self):
        declared = _declared(abinitio3d_tab)
        for name in (
            "url_params",
            "url_classes",
            "run",
            "phase_run",
            "map_run",
            "rot_fold",
            "map_hand",
            "pitch_band",
            "length_range",
        ):
            assert name in declared
        # the same-class histogram stays in HelicalPitch
        for name in ("min_len", "max_len", "bins", "auto_min_len"):
            assert name not in declared

    @pytest.mark.parametrize("module", [abinitio3d_tab, helical_pitch_tab])
    def test_every_declared_input_is_read(self, module):
        # buttons and gallery picks are read too; download buttons are outputs
        declared = _declared(module) - {"input_files", "class-selection"}
        unread = {
            n for n in declared - _read(module) if not n.startswith(("download_",))
        }
        # a select-all button acts in the browser only
        unread -= {"accepted_select_all"}
        assert not unread, sorted(unread)

    @pytest.mark.parametrize("module", [abinitio3d_tab, helical_pitch_tab])
    def test_bookmark_defaults_name_real_inputs(self, module):
        declared = _declared(module)
        from helicon.webApps import bookmark

        for short, entry in bookmark.tab_entries(module).items():
            assert entry.input_id in declared, (short, entry.input_id)

    def test_select_all_targets_its_own_gallery(self):
        # both tabs have a select_classes_inner gallery on the page; a plain
        # suffix match would find HelicalPitch's first
        from shiny import ui

        html = str(ui.page_fluid(abinitio3d_tab.abinitio3d_tab_ui("ab")))
        assert "getElementById(&apos;ab-select_classes_inner_image_1&apos;)" in html


class TestRegistration:
    def test_the_app_registers_the_tab(self):
        from helicon.webApps import app

        assert app._TAB_MODULE_MAP["AbInitio3D"] == ("abinitio3d", abinitio3d_tab)
        src = Path(app.__file__).read_text()
        assert 'ui.nav_panel("AbInitio3D", abinitio3d_tab_ui("abinitio3d"))' in src
        assert '"AbInitio3D": lambda: abinitio3d_tab_server("abinitio3d")' in src
        # the navbar puts it right after Denovo3D
        assert src.index('ui.nav_panel("AbInitio3D"') > src.index(
            'ui.nav_panel("Denovo3D"'
        )

    def test_the_bookmark_table_reaches_the_page(self):
        import json

        from helicon.webApps import app, bookmark

        table = json.loads(bookmark.js_table(app._TAB_MODULE_MAP))["AbInitio3D"]
        for short, entry in bookmark.tab_entries(abinitio3d_tab).items():
            assert table[short] == [
                f"abinitio3d-{entry.input_id}",
                entry.default,
                entry.derived,
            ]

    def test_home_shows_it(self):
        from helicon.webApps.tabs.home_tab import HOME_APPS

        app = next(a for a in HOME_APPS if a.name == "AbInitio3D")
        assert app.is_tab


class TestFileBrowserLaunch:
    def test_apps_menu_lists_it_after_denovo3d(self):
        from helicon.lib.gui.file_browser import _APP_LAUNCH_TABLE

        names = [row[0] for row in _APP_LAUNCH_TABLE]
        assert names.index("AbInitio3D") == names.index("Denovo3D") + 1
        assert _APP_LAUNCH_TABLE[names.index("AbInitio3D")][1] == "AbInitio3D"

    def test_action_button_follows_denovo3d(self):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        modes = [m for _, m in FolderBrowserWidget._DISPLAY_BUTTONS]
        assert modes.index("abInitio3D") == modes.index("denovo3D") + 1

    def test_relion_star_finds_its_classes(self, tmp_path):
        from helicon.lib.gui.webapps import _class2d_bookmark

        star = tmp_path / "run_it020_data.star"
        star.write_text("dummy")
        mrcs = tmp_path / "run_it020_classes.mrcs"
        mrcs.write_text("dummy")
        expected = dict(
            mode_params="server",
            mode_classes="server",
            server_params=str(star.resolve()),
            server_classes=str(mrcs.resolve()),
        )
        assert _class2d_bookmark(str(star)) == expected
        assert _class2d_bookmark(str(mrcs)) == expected

    def test_a_missing_companion_is_left_out(self, tmp_path):
        from helicon.lib.gui.webapps import _class2d_bookmark

        star = tmp_path / "run_it020_data.star"
        star.write_text("dummy")
        assert "server_classes" not in _class2d_bookmark(str(star))

    def test_cryosparc_class_averages_find_their_particles(self, tmp_path):
        from helicon.lib.gui.webapps import _class2d_bookmark

        averages = tmp_path / "J63_020_class_averages.mrc"
        particles = tmp_path / "J63_020_particles.cs"
        averages.write_text("x")
        particles.write_text("x")
        bookmark = _class2d_bookmark(str(averages))
        assert bookmark["server_classes"] == str(averages)
        assert bookmark["server_params"] == str(particles)

    def test_the_display_dispatches_the_mode(self):
        import helicon.commands.display as display

        src = Path(display.__file__).read_text()
        assert 'if mode == "abInitio3D":\n            _launch_abinitio3d(' in src


class TestRelionSymmetry:
    """relion_reconstruct imposes the C symmetry only when Impose C symmetry is
    ticked, as the map from the class averages does."""

    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    def test_the_symmetry_follows_impose_c(self):
        src = self._src()
        helper = src[src.index("def _imposed_csym") :]
        helper = helper[: helper.index("\n\n")]
        assert "input.map_csym()" in helper and "else 1" in helper

    def test_the_run_and_the_star_file_hint_use_it(self):
        src = self._src()
        run = src[src.index("def run_relion") :]
        run = run[: run.index("@render.ui")]
        assert "csym=_imposed_csym()" in run
        hint = src[src.index("def pitch_band_download") :]
        hint = hint[: hint.index("return ui.tooltip(")]
        assert "csym = _imposed_csym()" in hint


class TestLongWorkInTheBackground:
    """The pitch estimate, the suggestions and the two reconstructions run in
    worker threads, so one visitor's run does not stall every other session."""

    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    @pytest.mark.parametrize(
        "button, task",
        [
            ("phase_run", "phase_task"),
            ("suggest_run", "suggest_task"),
            ("relion_run", "relion_task"),
            ("map_run", "map_task"),
        ],
    )
    def test_each_button_starts_a_background_task(self, button, task):
        src = self._src()
        assert re.search(
            rf'{task} = helicon\.shiny\.background_task\(\s*"{button}"', src
        )
        start = src[src.index(f"@reactive.event(input.{button})") :]
        start = start[: start.index("\n\n    ")]
        assert f"{task}.invoke(" in start

    def test_no_work_blocks_the_event_loop(self):
        # a ui.Progress block in an effect is the sign of work done in place
        assert "ui.Progress(" not in self._src()


class TestRelionOnAHost:
    def _body(self):
        import inspect

        src = inspect.getsource(abinitio3d_tab)
        body = src[src.index("def relion_ui") :]
        return body[: body.index("@render.ui")]

    def test_the_controls_are_not_offered_on_a_host(self):
        body = self._body()
        cloud = body.index("deployment.is_cloud()")
        assert (
            body[cloud:]
            .lstrip()
            .startswith("deployment.is_cloud():\n            return None")
        )
        assert cloud < body.index("find_relion_reconstruct()")
        assert cloud < body.index('"relion_run"')

    def test_nothing_is_offered_without_relion_and_no_message_is_shown(self):
        body = self._body()
        missing = body[body.index("exe is None") :]
        assert missing.lstrip("exe is None:\n ").startswith("return None")
        # no note on the page: the controls are just not there
        assert "not available" not in body
        assert "text-muted" not in body

    def test_the_run_is_refused_on_the_server_too(self):
        import inspect

        src = inspect.getsource(abinitio3d_tab)
        run = src[src.index("def run_relion") :]
        run = run[: run.index("relion_task.invoke(")]
        assert "deployment.refuse_server_mode()" in run


class TestRepeatSearchRange:
    """The longest repeat searched comes from the rise and the smallest
    twist, not a fixed 1500 A."""

    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    def test_the_input_sits_left_of_the_estimate_button(self):
        src = self._src()
        row = src[
            src.index('class_="ab-pitch-row"')
            - 3000 : src.index('class_="ab-pitch-row"')
        ]
        row = row[row.rindex("ui.div(") :]
        assert row.index('"min_twist"') < row.index('"phase_run"')
        assert re.search(r'"min_twist",.*?value=0\.3,', row, re.S)
        params = src[src.index('"Parameters"') :]
        params = params[: params.index('"3D map from classes"')]
        assert '"min_twist"' not in params
        assert abinitio3d_tab.BOOKMARK_DEFAULTS["min_twist"] == ("min_twist", 0.3)

    def test_the_range_follows_rise_and_twist(self):
        src = self._src()
        helper = src[src.index("def _max_repeat") :]
        helper = helper[: helper.index("def run_phase_pitch")]
        assert "input.rise()" in helper and "input.min_twist()" in helper
        assert "360.0 * rise / twist" in helper
        run = src[src.index("@reactive.event(input.phase_run)") :]
        assert (
            "max_repeat=_max_repeat()" in run[: run.index("phase_task.invoke(") + 600]
        )
        work = src[src.index("def _phase_work") :]
        work = work[: work.index("def _phase_apply")]
        assert 'max_sep=job["max_repeat"]' in work
        assert "1500" not in src


class TestRingColours:
    def test_the_fit_colour_always_spans_0_to_1(self):
        import inspect

        src = inspect.getsource(abinitio3d_tab)
        plot = src[src.index("def phase_circle_plot") :]
        plot = plot[: plot.index("@render.")]
        assert "cmin=0.0" in plot and "cmax=1.0" in plot


class TestRankedFits:
    """The class fits of the ring, ranked, below the ring and beside the
    per-filament histogram."""

    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    def test_it_sits_below_the_ring_right_of_the_histogram(self):
        src = self._src()
        ring = src.index('ui.output_ui("phase_circle_plot")')
        hist = src.index('ui.output_ui("filament_pitch_plot")')
        rank = src.index('ui.output_ui("class_fit_rank_plot")')
        assert ring < hist < rank
        # the same two columns as the scan and the ring above
        row = src[hist : rank + 2000]
        assert "col_widths=(7, 5)" in row

    def test_the_fallback_ranks_the_fits_best_first_sized_by_segments(self):
        src = self._src()
        plot = src[src.index("def _rank_figure") : src.index("def _agreement_figure")]
        assert "np.argsort(-np.nan_to_num(fit" in plot  # best first
        assert "r.class_count" in plot  # size by segments
        assert "cmax=1.0" in plot and "1.05]" in plot  # the ring's fixed scale
        assert "phase.poorly_fitting_cut(fit)" in plot

    def test_the_plot_is_the_fit_against_the_pixel_agreement(self):
        src = self._src()
        plot = src[
            src.index("def _agreement_figure") : src.index("def class_fit_rank_plot")
        ]
        assert "x=x," in plot and "y=y," in plot
        assert 'title="Pixel agreement"' in plot and 'title="Fit"' in plot
        assert "r.class_count" in plot  # size by segments
        assert "cmin=0.0" in plot and "cmax=1.0" in plot  # the fixed colour scale
        assert "range=[-0.04, 1.06]" in plot  # the 0 to 1 agreement axis
        assert "phase.poorly_fitting_cut(fit)" in plot  # the chance level
        assert "fit_threshold" in plot  # the threshold line
        assert "good classes" in plot  # where the good ones are

    def test_without_the_check_the_ranked_fits_are_shown(self):
        src = self._src()
        pick = src[src.index("def class_fit_rank_plot") :]
        pick = pick[: pick.index("def _length_range")]
        assert "_agreement_figure(r, agreement)" in pick
        assert "_rank_figure(r)" in pick
        assert "np.isfinite(agreement)" in pick


class TestResultsAppearInOrder:
    """The controls under the plots wait for the plots."""

    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    def test_the_results_controls_wait_for_the_plots(self):
        css = str(abinitio3d_tab._RESULTS_ORDER_CSS)
        assert ":has(.ab-wait > .shiny-html-output:empty) .ab-after" in css
        # opacity: a child cannot undo it, as the sliders' labels undid
        # visibility
        assert "opacity: 0" in css and "pointer-events: none" in css
        src = self._src()
        assert "_RESULTS_ORDER_CSS," in src
        # every result has these three; the histogram may be empty, so it is
        # not waited for
        for name in ("phase_scan_plot", "phase_circle_plot", "class_fit_rank_plot"):
            box = src[src.index(f'ui.output_ui("{name}")') :][:200]
            assert 'class_="ab-plot-box ab-wait"' in box, name
        box = src[src.index('ui.output_ui("filament_pitch_plot")') :][:200]
        assert "ab-wait" not in box
        # the controls that wait: both range sliders with the download, the two
        # thresholds, and everything below the plots
        assert src.count('class_="ab-after"') == 3


class TestHiddenGalleryStillUpdates:
    def test_the_class_gallery_is_drawn_while_hidden(self):
        """Remove picked and the suggestions' Add buttons redraw it; with the
        Parameters tab in front a suspended output made them do nothing."""
        import inspect

        src = inspect.getsource(abinitio3d_tab)
        at = src.index("def select_classes_gallery")
        assert "@output(suspend_when_hidden=False)" in src[at - 200 : at]


class TestHiddenUntilThereIsSomethingToShow:
    """The results and the suggestions start hidden in the page itself: hiding
    them from the server left them on screen while the tab started."""

    def test_the_page_starts_with_both_hidden(self):
        from htmltools import TagList

        html = TagList(abinitio3d_tab.abinitio3d_tab_ui("abinitio3d")).render()["html"]
        rule = "#ab_results_box, #abinitio3d-ab_suggestions_box { display: none; }"
        assert rule in html
        # the rule is in the page before the output that shows them
        assert html.index(rule) < html.index('id="abinitio3d-results_visibility"')

    def test_the_server_only_shows_them(self):
        import inspect

        src = inspect.getsource(abinitio3d_tab)
        body = src[src.index("def results_visibility") :]
        body = body[: body.index("@render")]
        assert "display: none" not in body
        assert "display: block" in body and "display: flex" in body


class TestClassAverageCheck:
    """The ring against the registered class averages (helical_pitch_pixel)."""

    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    def test_it_runs_after_the_pitch_estimate_and_never_breaks_it(self):
        src = self._src()
        work = src[src.index("def _phase_work") : src.index("def _phase_apply")]
        assert "result.pixel_check = _pixel_check(job, result, progress)" in work
        check = work[work.index("def _pixel_check") :]
        assert "pixel.class_pixel_check(" in check
        # a diagnostic: a failure is logged, the result is still shown
        assert "except Exception" in check and "return None" in check

    def test_the_images_are_always_passed_to_it(self):
        src = self._src()
        call = src[src.index("phase_task.invoke(") :]
        call = call[: call.index("\n        )\n")]
        assert "images=list(displayed_class_images())," in call
        assert "ids=list(displayed_class_ids())," in call

    def test_the_gallery_marks_the_classes_that_disagree(self):
        src = self._src()
        gallery = src[src.index("def accepted_gallery") :]
        gallery = gallery[: gallery.index("@reactive")]
        assert "\\u25c6" in gallery

    def test_the_agreement_comes_from_the_check(self):
        src = self._src()
        assert "check.agreement" in src[src.index("def class_fit_rank_plot") :][:900]

    def test_nothing_is_removed_because_of_it(self):
        # the flag is shown; the classes to take out come only from the segments
        src = self._src()
        poor = src[src.index("def _poorly_fitting") : src.index("first_result =")]
        assert "pixel" not in poor.split("def _pixel_flags")[0]

    def test_a_class_with_too_few_segments_is_not_called_sound_by_the_pixels(self):
        src = self._src()
        kept = src[src.index("kept = [") :]
        kept = kept[: kept.index("]\n") + 1]
        assert "phase.AT_CHANCE" in kept and "phase.BELOW_THRESHOLD" in kept
        assert "TOO_FEW" not in kept


class TestTypesGrouping:
    """The filaments grouped by the classes they use (helical_pitch_types)."""

    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    def test_the_card_sits_with_the_selection_and_the_suggestions(self):
        src = self._src()
        sug = src.index("id=_SUGGESTIONS_BOX")
        card = src.index('ui.card_header("Types")')
        assert sug < card
        assert (
            'ui.input_task_button(\n                                "types_run"' in src
        )

    def test_the_grouping_runs_in_the_background(self):
        src = self._src()
        assert 'helicon.shiny.background_task(\n        "types_run"' in src
        work = src[src.index("def _types_work") : src.index("def _types_apply")]
        assert "types.class_similarity(" in work

    def test_a_result_is_dropped_when_other_data_were_loaded_meanwhile(self):
        src = self._src()
        apply_ = src[src.index("def _types_apply") : src.index("def _types_error")]
        assert 'params() is not job["params"]' in apply_

    def test_the_comparison_is_forgotten_with_the_data(self):
        src = self._src()
        forget = src[src.index("def _forget_types") : src.index("def _types_work")]
        assert "types_state.set(None)" in forget
        assert "types_pitch.set(None)" in forget
        assert "type_group_used.set(None)" in forget
        assert 'ui.update_checkbox("types_restrict", value=False)' in forget

    def test_the_number_of_groups_is_a_bookmarked_input_that_follows_the_classes(self):
        d = abinitio3d_tab.BOOKMARK_DEFAULTS
        assert d["types_n"][:2] == ("types_n", 1)
        assert d["types_n"][2] is abinitio3d_tab.bookmark.DERIVED
        assert "types_cut" not in d

    def test_the_number_slider_is_part_of_the_card_not_of_what_needs_it(self):
        # a slider drawn by the output that needs its value never gets drawn: the
        # groups wait for the number, and the number for the groups
        src = self._src()
        card = src.index('ui.card_header("Types")')
        slider = src.index('"types_n"', card)
        controls = src.index('ui.output_ui("types_controls")', card)
        assert slider < controls
        assert 'helicon.shiny.slider(\n                            "types_n"' in src
        dynamic = src[src.index("def types_controls") : src.index("def _chosen_group")]
        assert "types_n" not in dynamic

    def test_the_groups_are_those_of_the_number_asked_for(self):
        src = self._src()
        groups = src[src.index("def _marker_groups") : src.index("def _group_entries")]
        assert "types.describe_marker_groups(" in groups
        assert "_types_number()" in groups
        number = src[src.index("def _types_number") : src.index("def _marker_groups")]
        assert "input.types_n()" in number

    def test_each_grouping_starts_at_the_number_of_its_clear_splits(self):
        src = self._src()
        work = src[src.index("def _types_work") : src.index("def _types_apply")]
        assert "types.marker_groups(usage, similarity)" in work
        assert "types.default_n_marker_groups(splits)" in work
        effect = src[src.index("def _default_types_n") :][:900]
        assert 'ui.update_slider("types_n"' in effect
        assert "types_first" in effect  # a bookmarked number is kept at first

    def test_the_slider_offers_the_groups_the_splits_can_make(self):
        src = self._src()
        assert "max=types.MAX_MARKER_GROUPS" in src

    def test_using_a_group_restricts_the_analysis_to_its_filaments(self):
        src = self._src()
        use = src[src.index("def use_type_group") : src.index("def add_type_group")]
        assert "type_group_used.set(" in use and 'g["keys"]' in use
        assert 'ui.update_checkbox("types_restrict", value=True)' in use
        calc = src[
            src.index("def analysis_params") : src.index("# ── pitch from the phases")
        ]
        assert "type_filter()" in calc and "out.attrs = dict(p.attrs)" in calc

    def test_everything_after_the_selection_uses_the_restricted_table(self):
        src = self._src()
        for start, end in (
            ("def _phase_apply", "def _phase_error"),
            ("def run_phase_pitch", "# ── the classes in the computation"),
            ("def _suggest_apply", "def run_suggestions"),
            ("def _angles", "def _download_label"),
        ):
            part = src[src.index(start) : src.index(end)]
            assert "analysis_params()" in part, start
            # (a check that data are loaded may read the whole table)
            rest = part.replace("analysis_params()", "")
            assert "params()" not in rest.replace(
                "req(params() is not None)", ""
            ), start
        assert src.count("phase.select_segments(\n") == src.count(
            "phase.select_segments(\n            analysis_params()"
        ) + src.count("phase.select_segments(\n                    analysis_params()")

    def test_the_restricted_table_is_defined_before_anything_uses_it(self):
        # a decorator that names it runs when the server function reaches it
        import re

        src = self._src()
        uses = [m.start() for m in re.finditer(r"(?<!def )\banalysis_params\b", src)]
        assert src.index("def analysis_params") < min(uses)
        assert src.index("type_filter = reactive.value(None)") < src.index(
            "def analysis_params"
        )

    def test_a_new_restriction_clears_the_pitch_estimate(self):
        src = self._src()
        clear = src[src.index("def _clear_phase_result") - 120 :][:200]
        assert "@reactive.event(analysis_params, input.select_classes_inner)" in clear

    def test_the_restriction_is_shown_and_can_be_undone(self):
        src = self._src()
        assert 'ui.output_ui("type_filter_status")' in src
        # a checkbox of the Parameters tab switches it off and on
        params_tab = src[
            src.index('"Parameters",') : src.index('ui.tags.b("3D map from classes")')
        ]
        assert '"types_restrict"' in params_tab
        follow = src[
            src.index("def _restrict_to_group") : src.index("def types_restrict_note")
        ]
        assert "input.types_restrict()" in follow
        assert "type_filter.set(after)" in follow
        # what happened is said: all filaments again, and a pitch estimate cleared
        assert "ui.notification_show(" in follow and "was cleared" in follow
        alone = src[
            src.index("def _restrict_without_group") : src.index(
                "def _restrict_to_group"
            )
        ]
        assert "type_group_used() is None" in alone

    def test_the_two_repeat_test_runs_in_the_background_on_the_shown_groups(self):
        src = self._src()
        assert 'helicon.shiny.background_task(\n        "types_pitch_run"' in src
        work = src[
            src.index("def _types_pitch_work") : src.index("def _types_pitch_apply")
        ]
        assert "mixture.two_pitch_tests(" in work
        assert "mixture.split_cut(" in work
        apply_ = src[
            src.index("def _types_pitch_apply") : src.index("def _types_pitch_error")
        ]
        assert 'types_state() is not job["state"]' in apply_
        assert '_types_number() != job["n"]' in apply_

    def test_groups_whose_classes_fit_no_ring_are_merged_as_mixed(self):
        src = self._src()
        work = src[
            src.index("def _types_pitch_work") : src.index("def _types_pitch_apply")
        ]
        assert 't["fit"] < mixture.MIN_RING_FIT' in work
        assert 'not t["split"]' in work  # a group split by repeat is not mixed
        shown = src[src.index("def _type_groups") : src.index("def _not_grouped")]
        assert "labels[np.isin(labels, poor)] = poor[0]" in shown
        assert "mixed=poor[0] if poor else None" in shown
        entries = src[src.index("def _group_entries") : src.index("def _type_groups")]
        assert '"Mixed" if x["mixed"]' in entries

    def test_the_groups_keep_their_numbers_when_some_are_merged(self):
        # the note names groups by the numbers they were first shown with
        src = self._src()
        shown = src[src.index("def _type_groups") : src.index("def _not_grouped")]
        assert "numbers=numbers" in shown
        entries = src[src.index("def _group_entries") : src.index("def _type_groups")]
        assert "i != mixed and i in present" in entries

    def test_classes_that_look_like_no_other_stay_out_of_the_split(self):
        # junk classes split filaments by their junk segments, not by their type
        src = self._src()
        work = src[src.index("def _types_work") : src.index("def _types_apply")]
        assert "types.pixel_groups(similarity) >= 0" in work
        assert work.index("loners = ") < work.index("types.usage_matrix(")
        assert "loners=loners" in work
        shown = src[src.index("def _type_groups") : src.index("def _group_text")]
        assert shown.count("+ _not_grouped()") == 2

    def test_using_the_classes_in_no_group_restricts_no_filaments(self):
        src = self._src()
        use = src[src.index("def use_type_group") : src.index("def add_type_group")]
        assert 'if g["id"] is None:' in use
        assert 'ui.update_checkbox("types_restrict", value=False)' in use
        add = src[
            src.index("def _add_type_classes") : src.index(
                "def _restrict_without_group"
            )
        ]
        assert 'g["id"] is not None' in add

    def test_all_groups_are_evaluated_together_in_parallel(self):
        src = self._src()
        run = src[src.index("def run_types_pitch") : src.index("def types_pitch_ui")]
        assert "groups=groups" in run and "_marker_groups()" in run
        work = src[
            src.index("def _types_pitch_work") : src.index("def _types_pitch_apply")
        ]
        assert "mixture.two_pitch_tests(" in work
        # one cut from the gains of all the groups
        assert "mixture.split_cut(gains)" in work
        assert '"Evaluate one vs two ring fits for each group"' in src

    def test_a_group_with_two_repeats_is_shown_as_two(self):
        src = self._src()
        shown = src[src.index("def _type_groups") : src.index("def _group_text")]
        assert "_pitch_tests()" in shown
        tests = src[src.index("def _pitch_tests") : src.index("def run_types_pitch")]
        assert 'test["n"] != _types_number()' in tests
        assert 'for r, letter in ((0, "a"), (1, "b"))' in shown

    def test_moving_the_number_does_not_redraw_the_slider(self):
        # the controls follow the comparison, not the number: read the groups isolated
        src = self._src()
        controls = src[src.index("def types_controls") : src.index("def _chosen_group")]
        assert "reactive.isolate()" in controls
        assert "update_select" in src[src.index("def _types_choices") :][:400]

    def test_the_group_replaces_or_adds_to_the_selection(self):
        src = self._src()
        assert (
            "_set_selection(positions)"
            in src[src.index("def use_type_group") : src.index("def add_type_group")]
        )
        add = src[
            src.index("def add_type_group") : src.index("def _restrict_without_group")
        ]
        assert "input.select_classes_inner()" in add and "_set_selection(" in add

    def test_add_group_adds_all_and_add_selected_the_picked_classes(self):
        src = self._src()
        assert '"types_add_picked",\n                    "Add selected"' in src
        add = src[src.index("def add_type_group") : src.index("def add_picked_type")]
        assert "_type_positions(picked_only=False)" in add
        picked = src[
            src.index("def add_picked_type") : src.index("def _add_type_classes")
        ]
        assert "_type_positions(picked_only=True)" in picked
        buttons = src[src.index("def types_buttons") : src.index("def _type_positions")]
        assert "col_widths=(4, 4, 4)" in buttons

    def test_nothing_is_selected_or_estimated_without_the_user(self):
        # the grouping only offers; the selection changes on Use / Add group
        src = self._src()
        apply_ = src[src.index("def _types_apply") : src.index("def _types_error")]
        assert "_set_selection" not in apply_


class TestVetoOfAHalfRepeat:
    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    def test_the_class_averages_are_given_to_the_pitch_estimate(self):
        src = self._src()
        work = src[src.index("def _phase_work") : src.index("def _pixel_check")]
        assert "class_images=" in work

    def test_a_repeat_that_was_ruled_out_is_said_so(self):
        src = self._src()
        assert "r.vetoed_period is not None" in src
        assert "ruled out as the repeat" in src


class TestPixelAgreementThreshold:
    """The second removal threshold, on the pixel agreement of each class."""

    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    def test_it_is_off_by_default_and_bookmarked(self):
        assert abinitio3d_tab.BOOKMARK_DEFAULTS["agree_cut"] == ("agree_threshold", 0.0)
        src = self._src()
        slider = src[src.index('"agree_threshold",\n') :]
        assert "value=0.0" in slider[:1500]

    def test_it_is_drawn_as_a_vertical_line_and_the_hover_counts_pairs(self):
        src = self._src()
        plot = src[
            src.index("def _agreement_figure") : src.index(
                "@render.ui\n    def class_fit_rank_plot"
            )
        ]
        assert "add_vline" in plot and "agree_threshold" in plot
        assert "registered pairs" in plot and "not judged" in plot

    def test_it_is_passed_to_the_diagnosis_with_a_label(self):
        src = self._src()
        assert "agreement_threshold=" in src
        assert (
            abinitio3d_tab._SHORT_REASON[abinitio3d_tab.phase.PIXELS_DISAGREE]
            == "pixels disagree"
        )

    def test_the_download_is_right_of_the_two_range_sliders(self):
        src = self._src()
        band = src.index('"pitch_band",')
        length = src.index('"length_range",')
        download = src.index('ui.output_ui("pitch_band_download")')
        assert band < length < download
        assert "col_widths=(7, 5)" in src[download : download + 200]

    def test_the_plots_column_is_the_wider_one(self):
        assert "col_widths=(4, 8)" in self._src()


class TestSuggestionsAndAgreement:
    """Suggest classes holds back what either threshold would remove."""

    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    def test_the_shown_suggestions_follow_both_thresholds(self):
        src = self._src()
        shown = src[src.index("def _shown_suggestions") :]
        shown = shown[: shown.index("@render.ui")]
        assert "input.fit_threshold()" in shown and "input.agree_threshold()" in shown
        assert 'd["agree"] >= agreement' in shown

    def test_the_agreement_is_predicted_from_the_placement_on_the_ring(self):
        src = self._src()
        work = src[src.index("def _suggest_agreement") :]
        work = work[: work.index("def _suggest_apply")]
        assert "pixel.candidate_agreement" in work and "check.sign" in work
        assert "placement=True" in src

    def test_a_failing_check_leaves_the_suggestions_standing(self):
        src = self._src()
        work = src[src.index("def _suggest_agreement") :]
        assert "except Exception" in work[: work.index("def _suggest_apply")]


class TestPlotsFollowTheirBox:
    """A figure is resized when its box changes, so its edges are not cut off."""

    def test_the_html_resizes_the_figure_with_its_box(self):
        import plotly.graph_objects as go

        html = str(
            abinitio3d_tab._fig_to_html(go.Figure(go.Scatter(x=[0, 1], y=[0, 1])))
        )
        assert "ResizeObserver" in html and "Plotly.Plots.resize" in html
        # the placeholder is filled with the figure's own id
        assert "{plot_id}" not in html


class TestFusedReconstruction:
    """The reconstruction can cut the segments from the micrographs, at another
    pixel size and in another box, when relion_reconstruct has --fused_extract."""

    def _src(self):
        import inspect

        return inspect.getsource(abinitio3d_tab)

    def _fused_ui(self):
        src = self._src()
        return src[
            src.index("def _fused_state") : src.index("def _relion_fused_request")
        ]

    def test_the_boxes_need_the_program_the_micrographs_and_the_coordinates(self):
        ui_src = self._fused_ui()
        state = ui_src[: ui_src.index("def relion_fused_ui")]
        assert "relion.fused_extract_available(exe)" in state
        assert "relion.micrographs_found(params(), input.relion_project_dir())" in state
        # nothing is drawn without them: the layout is what it was
        assert 'if state == "no":\n            return None' in ui_src
        for name in ("relion_apix", "relion_diameter", "relion_length"):
            assert f'"{name}"' in ui_src

    def test_the_boxes_come_and_go_with_the_micrographs_not_with_each_keystroke(self):
        ui_src = self._fused_ui()
        # a calc: what depends on it is redrawn only when its value changes
        src = self._src()
        before = src[: src.index("def _fused_state")].rstrip()
        assert before.endswith("@reactive.calc")
        assert "no micrographs" in ui_src  # a hint when only they are missing

    def test_the_defaults_are_the_class_averages_pixel_size_and_box(self):
        src = self._src()
        geometry = src[src.index("def _relion_default_geometry") :][:500]
        assert "found[1]" in geometry and "images[0].shape[0]" in geometry
        ui_src = self._fused_ui()
        assert "value=round(geometry[0], 4)" in ui_src
        assert ui_src.count("value=geometry[1]") == 2

    def test_the_defaults_reconstruct_from_the_stored_images_as_before(self):
        src = self._src()
        request = src[
            src.index("def _relion_fused_request") : src.index("def relion_fused_note")
        ]
        assert 'if _fused_state() != "yes"' in request
        assert "return None" in request
        assert "abs(apix - geometry[0]) < 1e-6" in request
        assert (
            "diameter == geometry[1]" in request and "length == geometry[1]" in request
        )
        run = src[src.index("def run_relion") : src.index("def relion_display")]
        assert "fused=fused" in run
        work = src[src.index("def _relion_work") : src.index("def _relion_apply")]
        assert 'fused=job["fused"]' in work

    def test_a_new_pixel_size_keeps_the_area_the_boxes_cover(self):
        src = self._src()
        effect = src[src.index("def _rescale_relion_boxes") :][:900]
        assert (
            "input.relion_apix"
            in src[src.index("def _rescale_relion_boxes") - 120 :][:200]
        )
        assert "old / new" in effect and "ui.update_numeric(name" in effect

    def test_the_segments_need_their_coordinates(self):
        src = self._src()
        run = src[src.index("def run_relion") : src.index("def relion_display")]
        assert '"rlnCoordinateX" not in subset' in run

    def test_a_cuboid_map_is_titled_with_its_three_sizes(self):
        assert abinitio3d_tab._box_text((4, 4, 4)) == "4³"
        assert abinitio3d_tab._box_text((8, 4, 4)) == "4 × 4 × 8"


class TestProjectFolderGuess:
    def test_it_is_the_folder_with_the_pipeline_file_above_the_star_file(self):
        import inspect

        src = inspect.getsource(abinitio3d_tab)
        guess = src[
            src.index("def _relion_project_guess") : src.index("relion_prev_apix =")
        ]
        assert "class2d_files.relion_project_root(star)" in guess
        # tried before anything that looks for the stacks
        assert guess.index("relion_project_root") < guess.index("rlnImageName")

    def test_the_root_is_two_folders_above_the_star_file(self, tmp_path):
        from helicon.webApps import class2d_files

        (tmp_path / "default_pipeline.star").touch()
        job = tmp_path / "Class2D" / "job010"
        job.mkdir(parents=True)
        (job / "default_pipeline.star").touch()  # RELION 5: each job has a copy
        star = job / "run_it020_data.star"
        star.touch()
        assert class2d_files.relion_project_root(star) == str(tmp_path)

    def test_no_stacks_or_micrographs_are_needed_to_find_it(self, tmp_path):
        from helicon.webApps import class2d_files

        (tmp_path / "default_pipeline.star").touch()
        star = tmp_path / "Select" / "job005" / "particles.star"
        star.parent.mkdir(parents=True)
        star.touch()
        assert class2d_files.relion_project_root(star) == str(tmp_path)

    def test_a_job_folder_that_is_a_link_stays_in_its_project(self, tmp_path):
        from helicon.webApps import class2d_files

        project = tmp_path / "project"
        (project / "Class2D").mkdir(parents=True)
        (project / "default_pipeline.star").touch()
        elsewhere = tmp_path / "archive" / "job010"
        elsewhere.mkdir(parents=True)
        (elsewhere / "default_pipeline.star").touch()
        (project / "Class2D" / "job010").symlink_to(elsewhere)
        star = project / "Class2D" / "job010" / "run_it020_data.star"
        (elsewhere / "run_it020_data.star").touch()
        assert class2d_files.relion_project_root(star) == str(project)

    def test_a_file_at_another_depth_gets_the_outermost_pipeline_folder(self, tmp_path):
        from helicon.webApps import class2d_files

        (tmp_path / "default_pipeline.star").touch()
        deep = tmp_path / "Extract" / "job001" / "sub" / "p.star"
        deep.parent.mkdir(parents=True)
        (deep.parent.parent / "default_pipeline.star").touch()
        deep.touch()
        assert class2d_files.relion_project_root(deep) == str(tmp_path)

    def test_a_file_outside_a_project_gives_nothing(self, tmp_path):
        from helicon.webApps import class2d_files

        star = tmp_path / "a" / "b" / "c.star"
        star.parent.mkdir(parents=True)
        star.touch()
        assert class2d_files.relion_project_root(star) == ""

    def test_a_stack_or_a_micrograph_is_enough_to_reconstruct(self, tmp_path):
        import pandas as pd
        import pytest

        from helicon.webApps.lib import helical_pitch_relion as relion

        mic = tmp_path / "MotionCorr" / "m.mrc"
        mic.parent.mkdir()
        mic.touch()
        seg = pd.DataFrame(
            dict(
                rlnImageName=["1@Extract/none/a.mrcs"],  # no stacks in this project
                rlnMicrographName=["MotionCorr/m.mrc"],
                rlnCoordinateX=[1.0],
                rlnCoordinateY=[1.0],
                rlnAnglePsiPrior=[0.0],
                rlnAnglePsi=[0.0],
            )
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(relion, "find_relion_reconstruct", lambda: "/bin/true")
            mp.setattr(relion, "fused_extract_available", lambda exe=None: True)
            mp.setattr(relion, "find_relion_reconstruct_mpi", lambda: None)

            def reached(*a, **k):
                raise RuntimeError("reached relion")

            mp.setattr(relion.subprocess, "run", reached)
            with pytest.raises(RuntimeError, match="reached relion"):
                relion.reconstruct(
                    seg, str(tmp_path), fused=dict(angpix=2, length=64, diameter=64)
                )
        # the stored-image route still needs the stack
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(relion, "find_relion_reconstruct", lambda: "/bin/true")
            with pytest.raises(FileNotFoundError, match="image paths"):
                relion.reconstruct(seg, str(tmp_path))


class TestProjectRootMarker:
    """RELION marks a project root with an empty .gui_projectdir."""

    def test_the_root_is_the_folder_with_the_gui_marker(self, tmp_path):
        from helicon.webApps import class2d_files

        (tmp_path / ".gui_projectdir").touch()
        job = tmp_path / "Class2D" / "job010"
        job.mkdir(parents=True)
        (job / "default_pipeline.star").touch()  # RELION 5: each job has a copy
        star = job / "run_it020_data.star"
        star.touch()
        assert class2d_files.relion_project_root(star) == str(tmp_path)

    def test_the_marker_wins_whatever_the_depth(self, tmp_path):
        from helicon.webApps import class2d_files

        (tmp_path / ".gui_projectdir").touch()
        deep = tmp_path / "Extract" / "job001" / "sub" / "p.star"
        deep.parent.mkdir(parents=True)
        (deep.parent.parent / "default_pipeline.star").touch()
        deep.touch()
        assert class2d_files.relion_project_root(deep) == str(tmp_path)

    def test_the_marker_is_found_through_a_linked_job_folder(self, tmp_path):
        from helicon.webApps import class2d_files

        project = tmp_path / "project"
        (project / "Class2D").mkdir(parents=True)
        (project / ".gui_projectdir").touch()
        archive = tmp_path / "archive" / "job010"
        archive.mkdir(parents=True)
        (archive / "run_it020_data.star").touch()
        (project / "Class2D" / "job010").symlink_to(archive)
        star = project / "Class2D" / "job010" / "run_it020_data.star"
        assert class2d_files.relion_project_root(star) == str(project)


class TestMicrographsFound:
    def _segments(self, tmp_path, **kw):
        import pandas as pd

        mic = tmp_path / "MotionCorr" / "m.mrc"
        mic.parent.mkdir(exist_ok=True)
        mic.touch()
        cols = dict(
            rlnMicrographName=["MotionCorr/m.mrc"],
            rlnCoordinateX=[1.0],
            rlnCoordinateY=[1.0],
            rlnAnglePsiPrior=[0.0],
        )
        cols.update(kw)
        return pd.DataFrame(cols)

    def test_the_micrograph_is_under_the_project(self, tmp_path):
        from helicon.webApps.lib import helical_pitch_relion as relion

        assert relion.micrographs_found(self._segments(tmp_path), str(tmp_path))

    def test_not_under_another_folder(self, tmp_path):
        from helicon.webApps.lib import helical_pitch_relion as relion

        other = tmp_path / "other"
        other.mkdir()
        assert not relion.micrographs_found(self._segments(tmp_path), str(other))

    def test_an_absolute_micrograph_path_needs_no_project(self, tmp_path):
        from helicon.webApps.lib import helical_pitch_relion as relion

        seg = self._segments(tmp_path)
        seg["rlnMicrographName"] = str(tmp_path / "MotionCorr" / "m.mrc")
        assert relion.micrographs_found(seg, str(tmp_path / "anywhere"))

    def test_without_a_project_folder_or_coordinates_there_is_nothing_to_cut_from(
        self, tmp_path
    ):
        from helicon.webApps.lib import helical_pitch_relion as relion

        seg = self._segments(tmp_path)
        assert not relion.micrographs_found(seg, "")
        assert not relion.micrographs_found(
            seg.drop(columns="rlnCoordinateX"), str(tmp_path)
        )
        assert not relion.micrographs_found(
            seg.drop(columns="rlnAnglePsiPrior"), str(tmp_path)
        )
        assert not relion.micrographs_found(None, str(tmp_path))


class TestPlotSizing:
    def test_the_plot_boxes_follow_the_height_of_the_window(self):
        assert "vh" in abinitio3d_tab._PLOT_BOX
        assert "340px" not in abinitio3d_tab._PLOT_BOX
        assert "clamp(" in abinitio3d_tab._PLOT_BOX

    def test_a_figure_is_fitted_to_its_box_whenever_it_changes(self):
        js = abinitio3d_tab._RESIZE_WITH_BOX
        # the figure's size is compared with its box's, not only the window's
        assert "_fullLayout" in js and "offsetWidth" in js and "offsetHeight" in js
        assert "ResizeObserver" in js and "IntersectionObserver" in js
        assert "addEventListener('resize'" in js
        # and again as the page settles
        assert "setTimeout(fit" in js

    def test_the_figure_html_carries_the_script_for_its_own_div(self):
        import plotly.graph_objects as go

        html = str(abinitio3d_tab._fig_to_html(go.Figure(), plot_id="abc"))
        assert "getElementById('abc')" in html


class TestFilamentJointFigure:
    """The per-filament repeat against the length, with a histogram of each."""

    @pytest.fixture(scope="class")
    def table(self):
        import pandas as pd

        rng = np.random.default_rng(1)
        n = 300
        return pd.DataFrame(
            dict(
                span=rng.uniform(100, 2000, n),
                pitch=np.where(rng.random(n) < 0.5, 700.0, 780.0) + rng.normal(0, 5, n),
                rlnMicrographName=[
                    f"MotionCorr/job001/mic{i % 30}.mrc" for i in range(n)
                ],
                rlnHelicalTubeID=[i % 5 for i in range(n)],
            )
        )

    def _fig(self, table, **kw):
        args = dict(length_range=(300, 1500), repeat_band=(680, 720), n_unfitted=12)
        args.update(kw)
        return abinitio3d_tab._filament_joint_figure(
            table, 705.0, args["length_range"], args["repeat_band"], args["n_unfitted"]
        )

    @staticmethod
    def _scatters(fig):
        return [d for d in fig.data if d.type in ("scatter", "scattergl")]

    def _points(self, fig):
        return [d for d in self._scatters(fig) if d.mode == "markers"]

    def _lines(self, fig, colour):
        return [
            d
            for d in self._scatters(fig)
            if d.mode == "lines" and d.line is not None and d.line.color == colour
        ]

    def _selected(self, table, lo=300, hi=1500, low=680, high=720):
        s, p = table["span"].to_numpy(), table["pitch"].to_numpy()
        return (s >= lo) & (s <= hi) & (p >= low) & (p <= high)

    def test_the_scatter_has_length_across_and_repeat_up(self, table):
        fig = self._fig(table)
        points = self._points(fig)
        x = np.concatenate([np.asarray(d.x) for d in points])
        y = np.concatenate([np.asarray(d.y) for d in points])
        assert len(x) == len(table)
        assert sorted(x) == pytest.approx(sorted(table["span"]))
        assert sorted(y) == pytest.approx(sorted(table["pitch"]))
        assert fig.layout.xaxis3.title.text.startswith("Filament length")
        assert fig.layout.yaxis3.title.text.startswith("Repeat")

    def test_the_selection_is_what_the_sliders_set(self, table):
        fig = self._fig(table)
        chosen = [
            d
            for d in self._points(fig)
            if "selected" in d.hovertemplate and "not selected" not in d.hovertemplate
        ]
        assert len(chosen) == 1
        assert len(chosen[0].x) == int(self._selected(table).sum())
        assert f"{int(self._selected(table).sum())} selected" in fig.layout.title.text

    def test_no_bounds_select_every_filament(self, table):
        fig = self._fig(table, length_range=(None, None), repeat_band=(None, None))
        assert f"{len(table)} selected" in fig.layout.title.text
        assert not [d for d in self._points(fig) if "not selected" in d.hovertemplate]

    def test_each_variable_has_a_histogram_beside_the_scatter(self, table):
        fig = self._fig(table)
        hists = [d for d in fig.data if d.type == "histogram"]
        top = [d for d in hists if d.x is not None]
        right = [d for d in hists if d.y is not None]
        assert sum(len(d.x) for d in top) == len(table)
        assert sum(len(d.y) for d in right) == len(table)
        assert all(d.orientation == "h" for d in right)
        # a bar is every filament of its bin: the others stacked on the selected
        assert fig.layout.barmode == "stack"

    def test_the_histograms_share_an_axis_with_the_scatter(self, table):
        fig = self._fig(table)
        assert fig.layout.xaxis.matches == "x3" or fig.layout.xaxis3.matches == "x"
        assert fig.layout.yaxis4.matches == "y3" or fig.layout.yaxis3.matches == "y4"

    def test_the_pooled_repeat_is_a_green_line_in_the_scatter_and_the_histogram(
        self, table
    ):
        fig = self._fig(table)
        green = self._lines(fig, "green")
        assert len(green) == 2
        assert all(list(d.y) == [705.0, 705.0] for d in green)

    def test_the_selection_marks_are_traces_not_shapes(self, table):
        # the page script of the crosshairs takes over the shapes of every plot
        fig = self._fig(table)
        assert len(fig.layout.shapes) == 0
        orange = self._lines(fig, "orange")
        assert len(orange) == 4  # two edges in each histogram
        filled = [d for d in self._scatters(fig) if d.fill == "toself"]
        assert len(filled) == 1  # the rectangle in the scatter

    def test_the_title_counts_the_filaments_without_a_repeat(self, table):
        fig = self._fig(table, n_unfitted=1740)
        assert "1,740 without a clear repeat" in fig.layout.title.text
        assert f"{len(table):,} filaments" in fig.layout.title.text

    def test_many_filaments_are_drawn_with_webgl(self, table):
        import pandas as pd

        big = pd.concat([table] * 12, ignore_index=True)
        points = self._points(self._fig(big))
        assert {d.type for d in points} == {"scattergl"}
        small = self._points(self._fig(table))
        assert {d.type for d in small} == {"scatter"}

    def test_the_plot_uses_the_figure_in_its_box(self):
        import inspect

        src = inspect.getsource(abinitio3d_tab)
        plot = src[src.index("def filament_pitch_plot") : src.index("def _write_star")]
        assert "_filament_joint_figure(" in plot
        assert "_fig_to_html(" in plot


class TestMapFigure:
    """The classes and the map projection, one above the other, with no gap."""

    @staticmethod
    def _figure(ny=46, nx=330):
        m = dict(
            tiles=np.zeros((ny, nx)),
            projection=np.zeros((ny, nx)),
            z_view=np.zeros((60, 60)),
        )
        return abinitio3d_tab._map_figure(m, "title")

    def test_the_strips_meet_and_the_z_section_spans_both(self):
        fig = self._figure()
        lay = fig.layout
        # row 1 shrinks toward row 2 and row 2 toward row 1
        assert lay.yaxis.constraintoward == "bottom"
        assert lay.yaxis3.constraintoward == "top"
        assert lay.xaxis.constraintoward == lay.xaxis3.constraintoward == "right"
        # the z section (axis 2) spans the height of both strips
        assert lay.yaxis2.domain[0] == pytest.approx(lay.yaxis3.domain[0])
        assert lay.yaxis2.domain[1] == pytest.approx(lay.yaxis.domain[1])

    def test_rows_and_height_follow_the_shape_of_the_strips(self):
        flat, tall = self._figure(ny=30), self._figure(ny=90)
        assert flat.layout.height < tall.layout.height
        # the strips of the screenshot that had a gap: no more than ~300 px tall
        assert self._figure().layout.height < 320

    def test_each_title_sits_on_its_own_image(self):
        refs = {a.text: (a.xref, a.yref) for a in self._figure().layout.annotations}
        assert refs["Classes at their azimuths"] == ("x domain", "y domain")
        assert refs["Map projection"] == ("x3 domain", "y3 domain")
        assert refs["z section"] == ("x2 domain", "y2 domain")


class TestMapMethodWhenTheHelixTurnsManyTimes:
    """Back-projection has only the ring angles to go on, which cannot place the
    averages of a helix that turns many times per repeat: it is not used, and the
    map's title says so."""

    def _source(self):
        from pathlib import Path

        from helicon.webApps.tabs import abinitio3d_tab

        return Path(abinitio3d_tab.__file__).read_text(encoding="utf-8")

    def test_the_names_of_the_methods_are_in_one_place(self):
        from helicon.webApps.tabs import abinitio3d_tab as tab

        assert set(tab._MAP_METHODS) == {"joint", "gauss", "backprojection"}
        # "joint" stays the key of elastic net, for older bookmarks
        assert tab._MAP_METHODS["joint"].endswith("elasticnet")
        assert "_MAP_METHODS,\n" in self._source()  # the radio takes them from here

    def test_the_title_names_the_method_that_was_not_used(self):
        src = self._source()
        title = src[src.index("turns per repeat: placed by projection") :][:700]
        assert 'm.get("method_asked") == "backprojection"' in title
        assert "_MAP_METHODS['backprojection']" in title

    def test_the_job_records_the_method_that_was_asked_for(self):
        assert 'out["method_asked"] = job["method"]' in self._source()


class TestClassPositions:
    def test_class_numbers_are_one_based_and_ids_zero_based(self):
        from helicon.webApps.tabs import abinitio3d_tab as tab

        # displayed ids 5, 7, 9 sit at positions 0, 1, 2; class numbers are id + 1
        assert tab._class_positions([6, 10, 8], [5, 7, 9]) == [0, 2, 1]

    def test_a_class_that_is_not_displayed_is_an_error(self):
        from helicon.webApps.tabs import abinitio3d_tab as tab

        with pytest.raises(KeyError):
            tab._class_positions([99], [5, 7, 9])
