"""Aligning gaussian mixtures by their overlap.

Two properties here are load-bearing and fail *silently* -- they produce
plausible scores and a wrong answer -- so they are pinned deliberately:
normalising over the region the query covers rather than the whole map, and
fitting both mixtures at the same width.
"""

import numpy as np
import pytest

from helicon.webApps.lib import gauss_align as ga


def _mixture(n=80, seed=0, extent=(60.0, 300.0)):
    rng = np.random.default_rng(seed)
    centers = (rng.random((n, 2)) - 0.5) * np.array(extent)
    amps = rng.random(n) + 0.5
    return amps, centers


class TestRecovery:
    """A mixture against a transformed copy of itself."""

    def test_it_finds_the_shift(self):
        amps, centers = _mixture()
        applied = np.array([6.0, -42.0])
        result = ga.align_mixtures(
            amps,
            centers,
            6.0,
            amps,
            centers + applied,
            6.0,
            half_y=80,
            half_x=400,
            step=2.0,
        )
        assert result.shift[0] == pytest.approx(applied[0], abs=2.0)
        assert result.shift[1] == pytest.approx(applied[1], abs=2.0)
        assert result.score > 0.95

    def test_it_finds_the_polarity(self):
        amps, centers = _mixture()
        result = ga.align_mixtures(
            amps, centers, 6.0, amps, -centers, 6.0, half_y=80, half_x=400, step=2.0
        )
        assert result.polarity == -1
        assert result.score > 0.95

    def test_it_finds_the_scale(self):
        amps, centers = _mixture()
        k = 1.08
        searched = ga.align_mixtures(
            amps,
            centers,
            6.0,
            amps,
            centers * k,
            6.0 * k,
            half_y=80,
            half_x=400,
            step=2.0,
            scales=np.linspace(0.92, 1.12, 11),
        )
        assert searched.scale == pytest.approx(k, abs=0.02)
        unsearched = ga.align_mixtures(
            amps,
            centers,
            6.0,
            amps,
            centers * k,
            6.0 * k,
            half_y=80,
            half_x=400,
            step=2.0,
        )
        assert searched.score > unsearched.score + 0.05

    def test_an_empty_mixture_is_not_an_error(self):
        amps, centers = _mixture()
        empty = np.zeros((0, 2))
        result = ga.align_mixtures(
            np.zeros(0), empty, 6.0, amps, centers, 6.0, half_y=80, half_x=400
        )
        assert result.score == 0.0


class TestLocalNormalisation:
    """Normalise over what the query covers, not over the whole map.

    Dividing by a long projection's entire self-overlap makes the score fall
    with the map's length rather than with how well it matches -- it scored
    0 of 16 real searches where the local form scored 12 of 16, which is what
    the pixel-space route achieves.
    """

    def test_a_longer_map_does_not_score_lower_for_being_longer(self):
        amps, centers = _mixture(n=40, extent=(40.0, 100.0))
        # the same match, embedded in maps of very different length
        far = centers + np.array([0.0, 900.0])
        short = (
            np.concatenate([amps, amps]),
            np.concatenate([centers, centers + np.array([0.0, 300.0])]),
        )
        long = (
            np.concatenate([amps, amps, amps]),
            np.concatenate([centers, centers + np.array([0.0, 300.0]), far]),
        )

        s_short = ga.align_mixtures(
            amps,
            centers,
            6.0,
            short[0],
            short[1],
            6.0,
            half_y=60,
            half_x=1200,
            step=3.0,
        ).score
        s_long = ga.align_mixtures(
            amps, centers, 6.0, long[0], long[1], 6.0, half_y=60, half_x=1200, step=3.0
        ).score
        assert s_long == pytest.approx(s_short, rel=0.2)

    def test_self_overlap_grows_with_the_mixture(self):
        amps, centers = _mixture(n=40, extent=(40.0, 100.0))
        one = ga.self_overlap(amps, centers, 6.0)
        two = ga.self_overlap(
            np.concatenate([amps, amps]),
            np.concatenate([centers, centers + np.array([0.0, 400.0])]),
            6.0,
        )
        # which is exactly why it cannot be the normaliser for a local match
        assert two > 1.8 * one


class TestMatchedWidths:
    """Both mixtures have to be fitted at the same granularity."""

    def test_a_mismatched_width_costs_the_score(self):
        amps, centers = _mixture()
        matched = ga.align_mixtures(
            amps, centers, 6.0, amps, centers, 6.0, half_y=60, half_x=400, step=2.0
        ).score
        mismatched = ga.align_mixtures(
            amps, centers, 6.0, amps, centers, 18.0, half_y=60, half_x=400, step=2.0
        ).score
        assert matched > mismatched


class TestPlaceQuery:
    """The picture a user looks at, from the transform the alignment found."""

    def _image(self, ny=32, nx=32):
        image = np.zeros((ny, nx), dtype=np.float32)
        image[ny // 2 - 2 : ny // 2 + 2, 4 : nx - 4] = 1.0
        image[ny // 2 - 6, 6] = 2.0  # something asymmetric to track
        return image

    def test_it_returns_the_reference_shape(self):
        image = self._image()
        placed = ga.place_query(image, ga.MixtureAlignment(), (64, 200), 2.0)
        assert placed.shape == (64, 200)

    def test_a_shift_moves_the_density(self):
        image = self._image()
        apix = 2.0
        still = ga.place_query(image, ga.MixtureAlignment(), (64, 200), apix)
        moved = ga.place_query(
            image, ga.MixtureAlignment(shift=(0.0, 40.0)), (64, 200), apix
        )
        centre_still = (still * np.arange(still.shape[1])[None, :]).sum() / still.sum()
        centre_moved = (moved * np.arange(moved.shape[1])[None, :]).sum() / moved.sum()
        assert centre_moved - centre_still == pytest.approx(40.0 / apix, abs=2.0)

    def test_a_flip_mirrors_the_rows(self):
        image = self._image()
        plain = ga.place_query(image, ga.MixtureAlignment(), (64, 200), 2.0)
        flipped = ga.place_query(image, ga.MixtureAlignment(flip=-1), (64, 200), 2.0)
        assert not np.allclose(plain, flipped)
        assert np.allclose(np.flip(plain, axis=0), flipped, atol=1e-5)


class TestScoreBounds:
    """A normalised overlap is a correlation: it cannot exceed 1."""

    @pytest.mark.parametrize("m_sigma", [3.0, 6.0, 12.0, 30.0])
    def test_no_configuration_scores_above_one(self, m_sigma):
        amps, centers = _mixture()
        result = ga.align_mixtures(
            amps,
            centers,
            6.0,
            amps,
            centers,
            m_sigma,
            half_y=60,
            half_x=400,
            step=2.0,
        )
        assert result.score <= 1.0 + 1e-9


class TestRenderedNormaliser:
    """The self-overlap of a large mixture, by rendering rather than pairwise.

    A map projection expanded over a pitch runs to thousands of components,
    where the exact pairwise sum costs more than the match it normalises; the
    integral of the square of the render is the same quantity in linear time.
    """

    def test_it_agrees_with_the_exact_sum(self):
        amps, centers = _mixture(n=400, seed=3)
        exact = (
            np.outer(amps, amps)
            * np.pi
            * 6.4**2
            * np.exp(
                -((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
                / (4 * 6.4**2)
            )
        ).sum()
        assert ga._rendered_self_overlap(amps, centers, 6.4) == pytest.approx(
            exact, rel=0.01
        )

    def test_a_big_mixture_takes_the_rendered_route_and_still_scores_one(self):
        amps, centers = _mixture(n=ga._RENDER_SELF_ABOVE + 200, seed=4)
        result = ga.align_mixtures(
            amps, centers, 6.0, amps, centers, 6.0, half_y=60, half_x=400, step=2.0
        )
        assert result.score == pytest.approx(1.0, abs=0.03)
