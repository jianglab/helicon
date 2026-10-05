"""The AbInitio3D tab, split out of HelicalPitch, and how it is reached.

The class-azimuth workflow (ring, repeat, star export, 3D maps) was the
"Inter-class" view of HelicalPitch; it is now a tab of its own with its own
inputs, and HelicalPitch is back to the same-class pair-distance histogram.
"""

import re
from pathlib import Path

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
    """relion_reconstruct imposes the C symmetry only when Impose C is ticked,
    as the map from the class averages does."""

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
    def test_the_controls_are_replaced_by_a_note(self):
        import inspect

        src = inspect.getsource(abinitio3d_tab)
        body = src[src.index("def relion_ui") :]
        body = body[: body.index("@render.ui")]
        cloud = body.index("deployment.is_cloud()")
        assert cloud < body.index("find_relion_reconstruct()")
        assert cloud < body.index('"relion_run"')
        assert "not available on the hosted web site" in body

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
    """The classes grouped by how their averages look (helical_pitch_types)."""

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

    def test_the_cut_is_a_bookmarked_input_at_its_default(self):
        assert abinitio3d_tab.BOOKMARK_DEFAULTS["types_cut"] == ("types_cut", 0.15)

    def test_the_cut_slider_is_part_of_the_card_not_of_what_needs_the_cut(self):
        # a slider drawn by the output that needs its value never gets drawn: the
        # groups wait for the cut, and the cut for the groups
        src = self._src()
        card = src.index('ui.card_header("Types")')
        slider = src.index('"types_cut"', card)
        controls = src.index('ui.output_ui("types_controls")', card)
        assert slider < controls
        assert 'helicon.shiny.slider(\n                            "types_cut"' in src
        dynamic = src[src.index("def types_controls") : src.index("def _chosen_group")]
        assert "types_cut" not in dynamic

    def test_moving_the_cut_does_not_redraw_the_slider(self):
        # the controls follow the comparison, not the cut: read the groups isolated
        src = self._src()
        controls = src[src.index("def types_controls") : src.index("def _chosen_group")]
        assert "reactive.isolate()" in controls
        assert "update_select" in src[src.index("def _types_choices") :][:400]

    def test_the_group_replaces_or_adds_to_the_selection(self):
        src = self._src()
        assert (
            "_set_selection(positions)" in src[src.index("def use_type_group") :][:300]
        )
        add = src[src.index("def add_type_group") :][:400]
        assert "input.select_classes_inner()" in add and "_set_selection(" in add

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
