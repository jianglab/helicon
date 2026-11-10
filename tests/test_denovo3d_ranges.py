"""The twist and rise sliders of the Denovo3D tab cover the usual range."""

import re

import pytest

from helicon.webApps.tabs import denovo3d_tab, denovo3d_ui
from _source import tab_source
from _ui import page


def _slider(input_id):
    """The keyword arguments of the range slider ``input_id``, as written."""
    src = tab_source(denovo3d_tab)
    call = src[src.index(f'"{input_id}"') :]
    call = call[: call.index("width=")]  # the arguments before the width
    args = dict(re.findall(r"\b(min|max|step)=([-\d.]+)", call))
    lo, hi = re.search(r"\bvalue=\(([-\d.]+),\s*([-\d.]+)\)", call).groups()
    args["value"] = (float(lo), float(hi))
    return args


class TestSliderBounds:
    @pytest.mark.parametrize(
        "input_id, low, high",
        [("dn_twist_range", 0.1, 2.0), ("dn_rise_range", 0.1, 10.0)],
    )
    def test_the_ends_of_the_slider(self, input_id, low, high):
        args = _slider(input_id)
        assert float(args["min"]) == low and float(args["max"]) == high

    @pytest.mark.parametrize("input_id", ["dn_twist_range", "dn_rise_range"])
    def test_the_selected_range_is_within_the_ends(self, input_id):
        args = _slider(input_id)
        lo, hi = args["value"]
        assert float(args["min"]) <= lo <= hi <= float(args["max"])


class TestSearchRowLayout:
    def test_cards_and_buttons_share_an_unclipped_stretch_row(self):
        dn_page = page(denovo3d_ui.denovo3d_tab_ui)
        ids = (
            "dn_twist_card",
            "dn_rise_card",
            "dn_csym_card",
            "dn_show_run_button",
            "dn_stop_denovo3D",
        )
        assert dn_page.comes_before(*ids)
        rows = [
            next(
                up
                for up in dn_page.element(name).ancestors()
                if "dn-search-row" in up.classes()
            )
            for name in ids
        ]
        assert all(r is rows[0] for r in rows)
        assert not any("card" in node.classes() for node in rows[0].walk())
        for node in rows[0].walk():
            assert "height:" not in node.attrs.get("style", "")

    def test_parameter_cards_do_not_fix_their_height(self):
        src = tab_source(denovo3d_tab)
        block = src[
            src.index("def dn_csym_card") : src.index("def _dn_map_for_download")
        ]
        assert "height:" not in block
