"""Bookmark URLs: one scheme for every tab of the web app.

Each tab declares its parameters once, in ``BOOKMARK_DEFAULTS``; the URL
carries only those that differ from their defaults, under short keys, and
opening it restores them through Shiny's own input restoration.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from helicon.webApps import bookmark


@pytest.fixture(scope="module")
def tabs():
    from helicon.webApps import app

    return app._TAB_MODULE_MAP


class TestValues:
    @pytest.mark.parametrize(
        "value, default, text",
        [
            (True, False, "1"),
            (False, True, "0"),
            (4.8, 4.75, "4.8"),
            (5.0, 4.75, "5"),
            (2, 1, "2"),
            (-81.1, 0.0, "-81.1"),
            ("right", "left", "right"),
            ("lime cyan", "", "lime cyan"),
            ("/data/run_it020_data.star", "", "/data/run_it020_data.star"),
            ((10.0, 120.5), (0.0, 0.0), "10,120.5"),
            (["x", "z"], ["x", "y", "z"], "x,z"),
        ],
    )
    def test_round_trip(self, value, default, text):
        assert bookmark.encode_value(value) == text
        decoded = bookmark.decode_value(text, default)
        if isinstance(value, tuple):
            value = list(value)
        assert decoded == value
        assert type(decoded) is type(value) or isinstance(value, float)

    def test_booleans_also_read_as_words(self):
        assert bookmark.decode_value("true", False) is True
        assert bookmark.decode_value("false", True) is False

    def test_a_number_where_an_int_was_declared(self):
        assert bookmark.decode_value("4.5", 1) == 4.5

    def test_bad_values_raise(self):
        with pytest.raises(ValueError):
            bookmark.decode_value("maybe", True)
        with pytest.raises(ValueError):
            bookmark.decode_value("abc", 1.0)

    def test_derived_flag(self):
        module = SimpleNamespace(
            BOOKMARK_DEFAULTS={
                "a": ("a_id", 1),
                "b": ("b_id", 2.0, bookmark.DERIVED),
            }
        )
        entries = bookmark.tab_entries(module)
        assert entries["a"] == bookmark.Entry("a_id", 1, False)
        assert entries["b"] == bookmark.Entry("b_id", 2.0, True)


class TestParse:
    def test_short_keys_become_full_input_ids(self, tabs):
        tab, inputs = bookmark.parse(
            "?tab=AbInitio3D&rise=4.8&csym=2&hand=right&merge_counterparts=0", tabs
        )
        assert tab == "AbInitio3D"
        assert inputs == {
            "abinitio3d-rise": 4.8,
            "abinitio3d-rot_fold": 2,
            "abinitio3d-map_hand": "right",
            "abinitio3d-merge_counterparts": False,
        }

    def test_unknown_keys_and_unreadable_values_are_dropped(self, tabs):
        tab, inputs = bookmark.parse(
            "tab=AbInitio3D&nonsense=1&rise=abc&csym=3&helicon_token=t&helicon_theme=Dark",
            tabs,
        )
        assert inputs == {"abinitio3d-rot_fold": 3}

    def test_an_unknown_tab_opens_nothing(self, tabs):
        assert bookmark.parse("tab=Nope&rise=4.8", tabs) == (None, {})
        assert bookmark.parse("", tabs) == (None, {})

    def test_the_earlier_form_still_opens(self, tabs):
        p = json.dumps({"url_params": "/a/run_it020_data.star", "ignore_blank": False})
        query = '_inputs_&helicon_tab="HelicalPitch"&_values_&p=' + __import__(
            "urllib.parse"
        ).parse.quote(p)
        tab, inputs = bookmark.parse(query, tabs)
        assert tab == "HelicalPitch"
        assert inputs == {
            "helical_pitch-url_params": "/a/run_it020_data.star",
            "helical_pitch-ignore_blank": False,
        }

    def test_ranges_and_lists(self, tabs):
        _, inputs = bookmark.parse("tab=HI3D&radius=10,120.5", tabs)
        assert inputs == {"hi3d-hi3d_radius": [10, 120.5]}
        _, inputs = bookmark.parse("tab=HelicalProjection&proj_xyz=x,z", tabs)
        assert inputs == {"helical_projection-map_projection_xyz_choices": ["x", "z"]}


class TestTheTables:
    """The declared tables are the only source; check them against the tabs."""

    def test_every_tab_has_one(self, tabs):
        for name, (_ns, module) in tabs.items():
            assert bookmark.tab_entries(module), name

    def test_every_entry_names_an_input_of_its_tab(self, tabs):
        for name, (_ns, module) in tabs.items():
            src = Path(module.__file__).read_text()
            declared = set(
                re.findall(
                    r"(?:ui\.input_\w+|helicon\.shiny\.(?:range_)?slider)"
                    r'\(\s*"(\w+)"',
                    src,
                )
            )
            for key, entry in bookmark.tab_entries(module).items():
                assert entry.input_id in declared, (name, key, entry.input_id)

    def test_short_keys_do_not_collide_with_reserved_ones(self, tabs):
        for name, (_ns, module) in tabs.items():
            assert not set(bookmark.tab_entries(module)) & bookmark.RESERVED_KEYS

    def test_the_page_script_gets_the_same_table(self, tabs):
        table = json.loads(bookmark.js_table(tabs))
        assert table["HILL"]["apix"] == ["hill-hill_apix", 2.3438, True]
        assert table["AbInitio3D"]["csym"] == ["abinitio3d-rot_fold", 1, False]
        assert set(table) == set(tabs)


class TestRestore:
    def test_the_page_is_built_with_the_bookmarked_values(self, tabs):
        from shiny import ui

        from helicon.webApps.tabs.abinitio3d_tab import abinitio3d_tab_ui

        _, inputs = bookmark.parse("tab=AbInitio3D&rise=4.8&hand=right", tabs)
        with bookmark.ui_restore_context(inputs):
            html = str(ui.page_fluid(abinitio3d_tab_ui("abinitio3d")))
        rise = re.search(r'<input id="abinitio3d-rise"[^>]*>', html).group(0)
        assert 'value="4.8"' in rise
        right = re.search(
            r'<input[^>]*name="abinitio3d-map_hand"[^>]*value="right"[^>]*>', html
        )
        assert right and "checked" in right.group(0)

    def test_without_a_bookmark_the_defaults_show(self):
        from shiny import ui

        from helicon.webApps.tabs.abinitio3d_tab import abinitio3d_tab_ui

        with bookmark.ui_restore_context({}):
            html = str(ui.page_fluid(abinitio3d_tab_ui("abinitio3d")))
        rise = re.search(r'<input id="abinitio3d-rise"[^>]*>', html).group(0)
        assert 'value="4.75"' in rise

    def test_inputs_built_later_restore_once(self):
        # what a render.ui does when a tab's data arrives: the first build of
        # the input gets the bookmarked value, later re-renders their own
        from shiny.bookmark import RestoreContext

        ctx = RestoreContext()
        session = SimpleNamespace(
            bookmark=SimpleNamespace(
                _restore_context=ctx, _set_restore_context=lambda c: None
            )
        )
        bookmark.restore_in_session(session, {"hi3d-hi3d_npeaks": 7})
        assert ctx.active
        from shiny.module import ResolvedId

        key = ResolvedId("hi3d-hi3d_npeaks")
        assert ctx.input.get(key) == 7
        ctx.input.flush_pending()
        assert ctx.input.get(key) is None

    def test_a_session_without_a_context_gets_one(self):
        made = []
        session = SimpleNamespace(
            bookmark=SimpleNamespace(
                _restore_context=None, _set_restore_context=made.append
            )
        )
        bookmark.restore_in_session(session, {"x-y": 1})
        assert made and made[0].input.as_dict() == {"x-y": 1}


class TestLaunchers:
    def test_query_names_the_tab_and_encodes_values(self):
        assert bookmark.query("HI3D", {"input_mode": "url", "url": "/a/b.map"}) == {
            "tab": "HI3D",
            "input_mode": "url",
            "url": "/a/b.map",
        }

    def test_the_file_browser_builds_the_same_urls(self):
        from helicon.lib.gui.webapps import _make_bookmark_query
        from helicon.lib.shiny import encode_query_params

        q = _make_bookmark_query(
            "HelicalPitch", {"url_params": "/a/run_it020_data.star"}
        )
        assert (
            encode_query_params(q)
            == "tab=HelicalPitch&url_params=/a/run_it020_data.star"
        )

    def test_a_launch_url_restores_what_it_says(self, tabs):
        from helicon.lib.gui.webapps import _make_bookmark_query
        from helicon.lib.shiny import encode_query_params

        q = _make_bookmark_query("HILL", {"input_mode": "2", "url": "/a/b c.mrcs"})
        tab, inputs = bookmark.parse(encode_query_params(q), tabs)
        assert tab == "HILL"
        assert inputs == {
            "hill-hill_input_mode_params": "2",
            "hill-hill_img_file_url": "/a/b c.mrcs",
        }
