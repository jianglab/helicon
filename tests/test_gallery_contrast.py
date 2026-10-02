"""Black and white points of the image gallery's tiles."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from helicon.lib.gui.gallery_widget import _tile_contrast


def _class_average(n=64):
    """A bright particle on a flat masked background, like a 2D class."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[:n, :n] - n / 2
    r = np.hypot(x, y)
    image = np.zeros((n, n), dtype=np.float32)
    inside = r < n * 0.4
    image[inside] = rng.normal(0, 0.1, inside.sum())
    particle = r < n * 0.2
    image[particle] += 2 + rng.normal(0, 0.3, particle.sum())
    return image


class TestTileContrast:
    def test_a_class_average_keeps_its_particle(self):
        image = _class_average()
        black, white = _tile_contrast(image)
        assert (image > white).mean() <= 0.01
        assert (image < black).mean() <= 0.01

    def test_a_flat_image(self):
        black, white = _tile_contrast(np.full((8, 8), 3.0))
        assert black == 3.0 and white == 4.0

    def test_non_finite_pixels_are_ignored(self):
        image = _class_average()
        image[0, :4] = (np.nan, np.inf, -np.inf, np.nan)
        black, white = _tile_contrast(image)
        assert np.isfinite(black) and np.isfinite(white) and white > black
