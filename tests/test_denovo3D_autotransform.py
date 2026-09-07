"""Tests for the denovo3D auto-transform (horizontalise + centre).

Getting this right matters downstream: a class average left a couple of degrees
off horizontal is enough to push the twist search to the wrong answer, and the
solvers do not currently compensate for it.
"""

import numpy as np
import pytest

import helicon
from helicon.webApps.tabs.denovo3d_tab import (
    _estimate_helix_rotation_center_diameter as est_plain,
    _refine_helix_rotation_center as est_guarded,
)


def _synthetic_filament(
    ny=64, nx=128, rotation=0.0, shift_y=0.0, diameter=10.0, seed=0, noise=0.0
):
    """A horizontal filament with periodic density, optionally rotated/shifted."""
    rng = np.random.default_rng(seed)
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    Y, X = np.meshgrid(y, x, indexing="ij")
    img = np.exp(-0.5 * (Y / (diameter / 2.0)) ** 2)
    img = img * (1.0 + 0.5 * np.cos(2 * np.pi * X / 9.0))  # axial repeats
    img = img.astype(np.float32)
    if noise:
        img = img + rng.normal(0, noise, img.shape).astype(np.float32)
        img = np.clip(img, 0, None)
    if rotation:
        img = helicon.transform_image(image=img, rotation=rotation)
    if shift_y:
        img = helicon.transform_image(
            image=img, rotation=0, post_translation=(shift_y, 0)
        )
    return img


def _residual_rotation(img, est):
    """Apply the estimated transform, then measure what is left over."""
    r, s, _ = est(img, threshold=np.max(img) * 0.2)
    t = img
    if r:
        t = helicon.transform_image(image=t, rotation=r)
    if s:
        t = helicon.transform_image(image=t, rotation=0, post_translation=(s, 0))
    r2, s2, _ = est_plain(t, threshold=np.max(t) * 0.2)
    return abs(r2), abs(s2)


@pytest.mark.parametrize("rotation", [0.0, 1.5, -4.0, 12.0, -20.0])
def test_guarded_refinement_is_never_worse(rotation):
    """The guard exists so refinement cannot degrade a good estimate.

    Plain iteration of the (noisy) estimator improved some images and made
    others worse; the guarded version only accepts a candidate that measurably
    reduces the residual, so it must be no worse than the plain estimate.
    """
    img = _synthetic_filament(rotation=rotation, shift_y=2.0, noise=0.02, seed=1)
    r_plain, s_plain = _residual_rotation(img, est_plain)
    r_guard, s_guard = _residual_rotation(img, est_guarded)
    assert (
        r_guard <= r_plain + 1e-6
    ), f"guarded residual rotation {r_guard} worse than plain {r_plain}"


@pytest.mark.parametrize("rotation", [3.0, -7.5, 15.0])
def test_recovers_known_rotation(rotation):
    """A cleanly synthetic filament should end up horizontal."""
    img = _synthetic_filament(rotation=rotation, noise=0.0, seed=2)
    r_guard, _ = _residual_rotation(img, est_guarded)
    assert r_guard < 0.5, f"left {r_guard} deg off horizontal"


def test_centres_the_filament():
    img = _synthetic_filament(rotation=0.0, shift_y=6.0, noise=0.0, seed=3)
    _, s_guard = _residual_rotation(img, est_guarded)
    assert s_guard < 1.0, f"left {s_guard} px off centre"


def test_returns_same_triple_shape_as_plain():
    """Drop-in compatibility: same (rotation, shift, diameter) contract."""
    img = _synthetic_filament(rotation=2.0, seed=4)
    a = est_plain(img, threshold=np.max(img) * 0.2)
    b = est_guarded(img, threshold=np.max(img) * 0.2)
    assert len(a) == len(b) == 3
    assert all(np.isscalar(v) or isinstance(v, (int, float, np.floating)) for v in b)
