"""Shiny names its reactive values the same way, without inspecting the whole stack."""

import time

from shiny import reactive
from shiny.reactive import _reactives

from helicon.webApps import shiny_speedups


class _Holder:
    pass


def _make_values():
    counter = reactive.value(0)
    typed: reactive.Value[int] = reactive.Value(1)
    holder = _Holder()
    holder.selected = reactive.value([])
    spread = reactive.value(3)
    unnamed = [reactive.value(4)]
    return counter, typed, holder.selected, spread, unnamed[0]


def _names():
    return [v._name for v in _make_values()]


class TestFastNames:
    def test_the_names_are_the_ones_shiny_finds(self):
        original = _reactives.Value._try_infer_name
        shiny_speedups.apply()
        try:
            fast = _names()
            _reactives.Value._try_infer_name = shiny_speedups._original or original
            slow = _names()
        finally:
            shiny_speedups._applied = False
            shiny_speedups.apply()
        assert fast == slow
        assert fast[:4] == ["counter", "typed", "selected", "spread"]

    def test_it_is_applied_once_and_is_quick(self):
        assert shiny_speedups.apply() is True
        assert shiny_speedups.apply() is True
        assert _reactives.Value._try_infer_name is shiny_speedups._fast_try_infer_name
        start = time.perf_counter()
        for _ in range(200):
            _make_values()
        assert (time.perf_counter() - start) / 1000 < 0.005  # each value

    def test_the_app_applies_it(self):
        from pathlib import Path

        import helicon.webApps.app as app

        assert "shiny_speedups.apply()" in Path(app.__file__).read_text()
