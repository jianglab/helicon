"""Lattice lines on the HI3D indexing plot.

Each C-symmetry copy's markers sit at ``(twist * n + phase, rise * n)``, colinear
in unwrapped twist. The plot shows only one turn, so a line through them leaves
one edge and returns at the other, and must do so exactly -- a segment ending at
+180 and the next starting at -180 at the SAME rise -- so the family reads as one
continuous line if the plot were tiled.
"""

import numpy as np
import pytest

from helicon.webApps.tabs.hi3d_tab import lattice_line_segments as segments

Y = 256.0


class TestEdgeContinuity:
    def test_a_segment_ends_on_an_edge_and_the_next_starts_on_the_other(self):
        xs, ys = segments(60.0, 0.0, 20.0, -Y, Y)
        assert len(xs) > 1
        for (x0, x1), (y0, y1), (nx0, _), (ny0, _) in zip(
            xs[:-1], ys[:-1], xs[1:], ys[1:]
        ):
            assert abs(abs(x1) - 180.0) < 1e-9, "segment must end exactly on an edge"
            assert abs(abs(nx0) - 180.0) < 1e-9, "next must start exactly on an edge"
            assert x1 == pytest.approx(-nx0), "it must reappear on the opposite edge"
            assert y1 == pytest.approx(ny0), "and at the same rise, or it jumps"

    def test_negative_twist_wraps_the_other_way(self):
        xs, _ = segments(-29.4, 0.0, 15.0, -Y, Y)
        assert len(xs) > 1
        assert xs[0][1] == pytest.approx(-180.0)
        assert xs[1][0] == pytest.approx(180.0)

    def test_the_line_spans_the_whole_rise_range(self):
        _, ys = segments(60.0, 0.0, 20.0, -Y, Y)
        assert ys[0][0] == pytest.approx(-Y)
        assert ys[-1][-1] == pytest.approx(Y)

    def test_segments_tile_the_range_without_gaps(self):
        _, ys = segments(47.3, 12.0, 11.0, -Y, Y)
        for (_, end), (start, _) in zip(ys[:-1], ys[1:]):
            assert end == pytest.approx(start)


class TestGeometry:
    def test_a_small_twist_needs_no_wrap_at_all(self):
        xs, _ = segments(1.2, 0.0, 4.75, -Y, Y)
        assert len(xs) == 1

    def test_points_lie_on_the_twist_rise_direction(self):
        """Every segment must have the slope the arrow shows."""
        twist, rise = 37.0, 9.0
        xs, ys = segments(twist, 0.0, rise, -Y, Y)
        for (x0, x1), (y0, y1) in zip(xs, ys):
            assert (x1 - x0) / (y1 - y0) == pytest.approx(twist / rise, rel=1e-6)

    def test_the_phase_offsets_the_line(self):
        """A C-symmetry copy is the same line shifted along twist."""
        a, _ = segments(20.0, 0.0, 10.0, -50, 50)
        b, _ = segments(20.0, 90.0, 10.0, -50, 50)
        assert a[0][0] + 90.0 == pytest.approx(b[0][0])

    def test_zero_twist_is_a_vertical_line(self):
        xs, ys = segments(0.0, 45.0, 10.0, -Y, Y)
        assert len(xs) == 1
        assert xs[0][0] == pytest.approx(xs[0][1]) == pytest.approx(45.0)
        assert ys[0] == [pytest.approx(-Y), pytest.approx(Y)]

    def test_every_x_stays_inside_the_axis(self):
        for twist in (-170.0, -29.4, 1.2, 60.0, 179.0):
            xs, _ = segments(twist, 0.0, 12.0, -Y, Y)
            flat = [v for seg in xs for v in seg]
            assert min(flat) >= -180.0 - 1e-9 and max(flat) <= 180.0 + 1e-9


class TestDegenerateInput:
    def test_a_nonpositive_rise_draws_nothing(self):
        assert segments(30.0, 0.0, 0.0, -Y, Y) == ([], [])

    def test_an_empty_range_draws_nothing(self):
        assert segments(30.0, 0.0, 10.0, 5.0, 5.0) == ([], [])
