"""The row of actions under the EMDB table.

The data grid selects one row at a time, or a run of them with shift; it has
no select-all. Filtering the table down to a family of structures and then
searching against all of them is how this tab is used, so two buttons under
the table do that. A third makes the x/y/z previews, which no longer happen
merely because maps were selected -- 751 selected maps is 751 downloads, and
that is a decision rather than a surprise.
"""

import inspect

from helicon.webApps.tabs import helical_projection_tab as tab


def _source():
    return inspect.getsource(tab)


class TestTheButtonsAreThere:
    def test_they_sit_under_the_table(self):
        source = _source()
        panel = source[source.index("amyloid_atlas' || input.input_mode_maps") :][:500]
        # one conditional panel covers amyloid_atlas, EMDB-helical and EMDB
        assert "EMDB-helical" in panel and "'EMDB'" in panel
        # below the table, where the eye ends up after reading it
        assert panel.index("display_emdb_dataframe") < panel.index("map_actions_ui")

    def test_all_three_share_one_row(self):
        body = _source()
        body = body[body.index("def map_actions_ui") :]
        body = body[: body.index("@reactive.effect")]
        assert body.index("select_all_emdb_rows") < body.index("clear_emdb_rows")
        assert body.index("clear_emdb_rows") < body.index("generate_xyz_projections")
        assert "display: flex" in body

    def test_the_selection_buttons_are_only_for_the_table_modes(self):
        body = _source()
        body = body[body.index("def map_actions_ui") :]
        body = body[: body.index("@reactive.effect")]
        assert "table_mode" in body
        # the previews apply to a single uploaded or downloaded map as well
        assert body.index("if n_maps:") > body.index("if table_mode")

    def test_generating_the_previews_is_a_separate_button(self):
        """Selecting maps must not start downloading them.

        The previews are a convenience the search does not need, and a
        filtered table can hand over hundreds of maps at a click.
        """
        source = _source()
        assert "map_actions_ui" in source
        body = source[source.index("def _get_map_xyz_projections") :][:400]
        assert "@reactive.event(input.generate_xyz_projections)" in source
        assert "MAPS_NEEDING_CONFIRMATION" in body
        assert "confirm_xyz_projections" in source

    def test_the_label_carries_no_count(self):
        """A number here went stale, and the grid already prints one.

        The count would have to come from the grid's filtered view. Read as a
        reactive.event dependency it suppressed the whole row of buttons in
        the url and upload modes, where the grid is not rendered; read inside
        the renderer it established no dependency and changed only after the
        button was pressed. The grid's own footer says "Viewing rows 1 through
        N of M" directly above these buttons, so the number is on screen
        anyway.
        """
        body = _source()
        body = body[body.index("def map_actions_ui") :]
        body = body[: body.index("@reactive.effect")]
        assert '"Select all displayed"' in body
        assert "%d displayed" not in body
        assert "data_view_rows" not in body

    def test_the_click_reads_the_view_as_it_is_then(self):
        # no count to go stale, and the selection is taken from the filtered
        # view at the moment of the click, which is always current
        body = _source()
        body = body[body.index("def _select_all_emdb_rows") :][:600]
        assert "display_emdb_dataframe.data_view_rows()" in body


class TestWhatTheButtonsDo:
    def test_select_all_selects_the_displayed_rows(self):
        body = _source()
        body = body[body.index("def _select_all_emdb_rows") :][:600]
        assert "data_view_rows()" in body
        assert "update_cell_selection" in body
        assert '"type": "row"' in body
        assert "await" in body

    def test_an_empty_table_selects_nothing(self):
        body = _source()
        body = body[body.index("def _select_all_emdb_rows") :][:600]
        assert "if not rows:" in body

    def test_clear_empties_the_selection(self):
        body = _source()
        body = body[body.index("def _clear_emdb_rows") :][:400]
        assert '"rows": []' in body

    def test_the_selection_still_drives_the_map_list(self):
        # the same effect serves a click and a programmatic selection
        body = _source()
        assert "@reactive.event(display_emdb_dataframe.cell_selection)" in body
        body = body[body.index("def _get_map_from_emdb") :]
        body = body[: body.index("# -- Map XYZ projections")]
        assert "maps.set(maps_tmp)" in body
