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
        assert '"AbInitio3D": lambda: abinitio3d_tab_server("abinitio3d", project)' in (
            src
        )
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
            mode_params="url",
            mode_classes="url",
            url_params=str(star.resolve()),
            url_classes=str(mrcs.resolve()),
        )
        assert _class2d_bookmark(str(star)) == expected
        assert _class2d_bookmark(str(mrcs)) == expected

    def test_a_missing_companion_is_left_out(self, tmp_path):
        from helicon.lib.gui.webapps import _class2d_bookmark

        star = tmp_path / "run_it020_data.star"
        star.write_text("dummy")
        assert "url_classes" not in _class2d_bookmark(str(star))

    def test_cryosparc_class_averages_find_their_particles(self, tmp_path):
        from helicon.lib.gui.webapps import _class2d_bookmark

        averages = tmp_path / "J63_020_class_averages.mrc"
        particles = tmp_path / "J63_020_particles.cs"
        averages.write_text("x")
        particles.write_text("x")
        bookmark = _class2d_bookmark(str(averages))
        assert bookmark["url_classes"] == str(averages)
        assert bookmark["url_params"] == str(particles)

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
        run = run[: run.index("relion_map.set(result)")]
        assert "csym=_imposed_csym()" in run
        hint = src[src.index("def pitch_band_download") :]
        hint = hint[: hint.index("return ui.tooltip(")]
        assert "csym = _imposed_csym()" in hint
