"""The twist and rise sliders of the Denovo3D tab cover the usual range."""

import inspect
import re

import pytest

from helicon.webApps.tabs import denovo3d_tab


def _slider(input_id):
    """The keyword arguments of the range slider ``input_id``, as written."""
    src = inspect.getsource(denovo3d_tab)
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
