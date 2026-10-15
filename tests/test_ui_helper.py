"""The helper that tests use to look at a rendered page (tests/_ui.py)."""

import pytest

from _ui import Page

HTML = """
<div class="card"><div class="card-header">Types</div>
  <div class="card-body">
    <bslib-layout-columns col-widths-sm="7,5">
      <div><select id="t-method"></select></div>
      <div><button id="t-run">Go</button><input id="t-n" value="3"></div>
    </bslib-layout-columns>
  </div></div>
<p>text<br>more</p>
<bslib-layout-columns col-widths-sm="4,8"><div id="t-left"></div><div><div id="t-right"></div></div></bslib-layout-columns>
"""


@pytest.fixture
def page():
    return Page(HTML)


class TestOrder:
    def test_ids_are_found_with_or_without_the_namespace(self, page):
        assert page.element("run") is page.element("t-run")
        assert page.has("run") and not page.has("nothing")

    def test_the_order_is_the_order_on_the_page(self, page):
        assert page.comes_before("method", "run", "left")
        assert not page.comes_before("run", "method")

    def test_a_missing_element_is_an_error_not_false(self, page):
        with pytest.raises(AssertionError):
            page.comes_before("method", "nothing")


class TestLayout:
    def test_columns_say_their_widths_and_which_one_it_is(self, page):
        assert page.columns("method") == ("7,5", 0)
        assert page.columns("n") == ("7,5", 1)
        assert page.columns("right") == ("4,8", 1)

    def test_side_by_side_is_one_row(self, page):
        assert page.same_row("method", "run")
        assert not page.same_row("method", "left")

    def test_a_card_is_found_by_its_header(self, page):
        card = page.card_of(page.heading("Types"))
        assert page.inside("run", card) and not page.inside("left", card)

    def test_text_and_void_tags_do_not_break_the_tree(self, page):
        assert page.element("run").text() == "Go"
        assert page.element("right").parent.parent.tag == "bslib-layout-columns"
