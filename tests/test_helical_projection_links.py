"""The EMDB table's entry ids, and where they take you."""

import inspect

from helicon.webApps.tabs import helical_projection_tab as tab


def _source():
    return inspect.getsource(tab)


class TestTheEntryIdsAreLinks:
    """Each EMDB id opens its entry, without the column losing its filter.

    The link is added to the rendered cells by a script rather than put into
    the frame. An anchor in the frame makes the grid treat the column as HTML,
    and everything that reads the table back -- the row a user selects, the
    rank written against an id -- would be reading markup instead of an id.
    """

    def test_the_frame_stays_plain_text(self):
        body = _source()
        body = body[body.index("def display_emdb_dataframe") :]
        body = body[: body.index("def map_actions_ui")]
        assert "ui.a(" not in body
        assert "ui.HTML(" not in body

    def test_the_script_is_on_the_page(self):
        source = _source()
        assert "_emdb_link_script()" in source
        assert "def _emdb_link_script()" in source

    def test_it_builds_the_entry_url_from_the_id(self):
        body = _source()
        body = body[body.index("def _emdb_link_script") :]
        assert "https://www.ebi.ac.uk/emdb/" in body
        assert "^EMD-\\d+$" in body
        assert "_blank" in body

    def test_following_a_link_does_not_select_the_row(self):
        body = _source()
        body = body[body.index("def _emdb_link_script") :]
        assert "stopPropagation" in body

    def test_it_survives_the_grid_redrawing_itself(self):
        # the grid draws only the rows in view and redraws on scroll, sort
        # and filter
        body = _source()
        body = body[body.index("def _emdb_link_script") :]
        assert "MutationObserver" in body
