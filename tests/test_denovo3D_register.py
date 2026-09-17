"""Tests for automatic multi-image registration (virtual stitching).

The interesting failures here are not crashes but plausible-looking wrong
answers: a pair with no overlap still registers confidently, a flip choice can
be inconsistent around a loop, and composing two transforms can silently
introduce a half-pixel shift. Each of those has a test.
"""

import itertools

import numpy as np
import pytest

import helicon
from helicon.webApps.lib import denovo3d_register as R


def _long_filament(nx=1200, ny=40, seed=3):
    """A filament with unique axial features, as one pitch of a real one has.

    Deliberately not periodic: a short repeat would make the offsets
    recoverable only modulo that period, which is a property of the test rather
    than of the algorithm.
    """
    rng = np.random.default_rng(seed)
    prof = np.fft.irfft(
        np.fft.rfft(rng.normal(size=nx)) * np.exp(-((np.fft.rfftfreq(nx) / 0.06) ** 2)),
        n=nx,
    )
    prof = (prof - prof.min()) / (prof.max() - prof.min())
    y = np.arange(ny) - ny // 2
    img = (
        np.exp(-0.5 * (y[:, None] / 6.0) ** 2)
        * (0.4 + prof[None, :])
        * (
            1
            + 0.35
            * np.cos(2 * np.pi * np.arange(nx)[None, :] / 9.0 + 0.8 * y[:, None] / 6.0)
        )
    )
    return img.astype(np.float32)


def _windows(offsets, width=200, psi_amp=0.4, dy_amp=0.7, noise=0.02, seed=3):
    """Cut windows at known offsets, with small residual misalignments.

    Small on purpose: these images have already been through the tab's
    auto-transform, which leaves well under a degree of tilt.
    """
    long_img = _long_filament(seed=seed)
    rng = np.random.default_rng(seed + 1)
    imgs, truth = [], []
    for n, x0 in enumerate(offsets):
        psi = psi_amp * (1 if n % 2 else -1)
        dy = dy_amp * (1 if n % 3 else -1)
        fx, fy = (n % 2 == 1), (n % 3 == 2)
        w = long_img[:, x0 : x0 + width].copy()
        w = helicon.transform_image(image=w, rotation=-psi, post_translation=(-dy, 0))
        if fx:
            w = w[:, ::-1]
        if fy:
            w = w[::-1, :]
        w = np.ascontiguousarray(w + rng.normal(0, noise, w.shape).astype(np.float32))
        imgs.append(w)
        truth.append(dict(x0=x0, psi=psi, dy=dy, flip_x=fx, flip_y=fy))
    return imgs, truth


# ── the axial correlation core ──────────────────────────────────────────


@pytest.mark.parametrize("true_dx", [0, 7, -13, 40, -55])
def test_dx_profile_matches_brute_force(true_dx):
    """The FFT search must be exact, not approximate.

    ``_dx_profile`` returns a sub-pixel estimate, so the comparison is against
    the integer sample it refines from; the refinement itself is pinned by
    test_dx_profile_recovers_a_subpixel_offset.
    """
    a = _long_filament(nx=300)[:, 40:200]
    b = _long_filament(nx=300)[:, 40 - true_dx : 200 - true_dx]
    dx, corr, _prom = R._dx_profile(a, b)

    av, bv = a - a.mean(), b - b.mean()
    nx = a.shape[1]
    best = (-2.0, 0)
    for k in range(-(nx // 2), nx // 2):
        lo_a, hi_a = max(0, k), min(nx, nx + k)
        lo_b, hi_b = max(0, -k), min(nx, nx - k)
        if hi_a - lo_a < int(0.35 * nx):
            continue
        pa, pb = av[:, lo_a:hi_a], bv[:, lo_b:hi_b]
        c = float((pa * pb).sum() / np.sqrt((pa**2).sum() * (pb**2).sum()))
        if c > best[0]:
            best = (c, k)
    assert round(dx) == best[1]
    assert abs(dx - best[1]) <= 0.5
    assert corr == pytest.approx(best[0], abs=1e-6)


def test_dx_profile_sign_convention():
    """Pin the sign, which nothing else did.

    ``_dx_profile(a, b)`` returns the offset to apply to ``b`` to bring it onto
    ``a``. Slicing ``b`` from ``shift`` pixels earlier in the same long image
    therefore gives ``-shift``. The brute-force test compares only against its
    own search, so it would pass with the sign inverted.
    """
    long_img = _long_filament(nx=400)
    a = long_img[:, 60:260]
    for shift in (11, -17, 33):
        b = long_img[:, 60 - shift : 260 - shift]
        dx, corr, _prom = R._dx_profile(a, b)
        assert corr > 0.9
        assert dx == pytest.approx(-shift, abs=0.5), f"shift {shift} gave {dx}"


@pytest.mark.parametrize("frac", [0.25, 0.5, -0.3, 0.75])
def test_dx_profile_recovers_a_subpixel_offset(frac):
    """Axial offsets are not whole numbers of pixels, and rounding them is lossy.

    At the tab's sampling one pixel is about one helical rise, so quantising the
    pairwise offset discards the very scale the composite exists to preserve.
    The correlation profile is smooth in the offset, so a parabola through the
    three samples around the maximum recovers the remainder.
    """
    import helicon

    long_img = _long_filament(nx=400)
    a = long_img[:, 60:260]
    shifted = helicon.transform_image(
        image=np.ascontiguousarray(long_img),
        rotation=0.0,
        post_translation=(0.0, float(frac)),
    )
    b = shifted[:, 40:240]
    dx, corr, _prom = R._dx_profile(a, b)
    assert corr > 0.9
    assert dx == pytest.approx(-(20 + frac), abs=0.15), f"got {dx}"


def test_dx_profile_subpixel_is_a_refinement_not_a_jump():
    """The estimate never leaves the sampling interval around the argmax."""
    long_img = _long_filament(nx=400)
    a = long_img[:, 60:260]
    for shift in (0, 11, -17, 33):
        b = long_img[:, 60 - shift : 260 - shift]
        dx, _c, _p = R._dx_profile(a, b)
        assert abs(dx - round(dx)) <= 0.5


def test_dx_profile_accepts_unequal_widths():
    """The refinement pass registers each image against the whole composite."""
    long_img = _long_filament(nx=600)
    dx, corr, _ = R._dx_profile(long_img[:, :500], long_img[:, 120:320])
    assert corr > 0.9
    assert abs(dx - 120) <= 1


def test_dx_profile_flags_a_flat_profile():
    """No real overlap still yields a best offset; prominence is what tells.

    Without this the global solve is handed confident nonsense.
    """
    rng = np.random.default_rng(0)
    a = _long_filament(nx=400)[:, :200]
    b = rng.normal(0, 1, (40, 200)).astype(np.float32)
    _dx_a, _c_a, prom_a = R._dx_profile(a, a.copy())
    _dx_b, _c_b, prom_b = R._dx_profile(a, b)
    assert prom_a > prom_b


# ── flips ───────────────────────────────────────────────────────────────


def test_the_four_flips_reduce_to_one_mirror_and_a_rotation():
    """x-flip then y-flip is exactly a 180 degree rotation.

    This is what makes it possible to fold the flips into a single-step
    transform, so it is worth pinning rather than assuming.
    """
    a = np.arange(48, dtype=np.float32).reshape(6, 8)
    assert np.array_equal(a[:, ::-1][::-1, :], np.rot90(a, 2))
    assert R.normalize_flip(False, False) == (False, 0.0)
    assert R.normalize_flip(True, False) == (True, 0.0)
    assert R.normalize_flip(True, True) == (False, 180.0)
    assert R.normalize_flip(False, True) == (True, 180.0)


@pytest.mark.parametrize("flip_x", [False, True])
@pytest.mark.parametrize("flip_y", [False, True])
@pytest.mark.parametrize("auto", [(3.0, 2.0), (-5.0, -3.0), (0.0, 0.0), (8.0, 4.0)])
def test_composed_transform_equals_the_two_step_chain(flip_x, flip_y, auto):
    """One interpolation must reproduce auto-transform followed by registration.

    Folding the 180 degree part into the rotation angle instead of keeping it
    as an array operation costs real accuracy -- it measured 0.95 here against
    0.999, because an array reversal and a 180 degree rotation differ by half a
    pixel on an even-sized image.
    """
    img = _long_filament(nx=160)
    r, s = auto
    reg = dict(flip_x=flip_x, flip_y=flip_y, psi=0.7, dy=-1.0)

    step1 = helicon.transform_image(image=img, rotation=r, post_translation=(s, 0))
    two_step = R._apply(R._flip(step1, flip_x, flip_y), reg["psi"], reg["dy"])
    one_step = R.apply_composed(img, R.compose_transforms(r, s, reg))

    a, b = two_step - two_step.mean(), one_step - one_step.mean()
    corr = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    assert corr > 0.99, f"composed transform differs from the two-step chain: {corr}"


def test_sync_flips_is_consistent_around_a_loop():
    """Pairwise flip choices can disagree; the global assignment may not."""
    pairs = [
        dict(i=0, j=1, flip_x=True, flip_y=False, corr=0.9),
        dict(i=1, j=2, flip_x=True, flip_y=False, corr=0.9),
        dict(i=0, j=2, flip_x=False, flip_y=False, corr=0.9),
    ]
    flips, conflicts = R.sync_flips(3, pairs)
    assert flips[0] == (False, False)
    assert conflicts == 0
    # mirroring twice returns to the original
    assert flips[2] == (False, False)


# ── end to end ──────────────────────────────────────────────────────────


def test_recovers_a_known_layout():
    offsets = [0, 40, 80, 120]
    imgs, truth = _windows(offsets)
    stitched, tf, diag = R.auto_stitch(
        imgs, rot_range=2.0, dy_range=3.0, coarse_step=0.5
    )
    assert diag["n_connected"] == len(offsets)
    got = np.array([t["dx"] - tf[0]["dx"] for t in tf])
    want = np.array([-(t["x0"] - truth[0]["x0"]) for t in truth])
    # the gauge may come out mirrored, which flips the sign of every offset
    err = min(np.abs(got - want).max(), np.abs(got + want).max())
    assert err <= 2, f"offsets off by {err} px"
    assert diag["closure"]["dx"] < 2.0
    assert stitched.shape[1] == pytest.approx(
        max(offsets) - min(offsets) + imgs[0].shape[1], abs=4
    )


def test_unregistrable_pairs_are_either_solved_or_flagged():
    """At low twist many pairs share no overlap. That must never fail silently.

    With well-overlapping images the offsets come out exactly; with images
    spread so that most pairs cannot be registered, some wrong pairs get in and
    the answer can be badly wrong (228 px on this instance). What is required
    is that it be *reported*: the closure error and flip conflicts are what tell
    the user the composite is not trustworthy. A quietly wrong stitch would be
    far worse than a refusal.
    """
    offsets = [0, 70, 150, 250, 340]
    imgs, truth = _windows(offsets)
    _stitched, tf, diag = R.auto_stitch(
        imgs, rot_range=2.0, dy_range=3.0, coarse_step=0.5
    )
    got = np.array([t["dx"] - tf[0]["dx"] for t in tf])
    want = np.array([-(t["x0"] - truth[0]["x0"]) for t in truth])
    err = min(np.abs(got - want).max(), np.abs(got + want).max())

    if err > 4:
        assert diag["closure"]["dx"] > 2.0 or diag["flip_conflicts"] > 0, (
            f"registration was wrong by {err} px and nothing flagged it: "
            f"closure={diag['closure']}, conflicts={diag['flip_conflicts']}"
        )
    else:
        assert diag["n_connected"] == len(offsets)


def test_a_good_stitch_reports_clean_diagnostics():
    """The flip side: a correct result must not cry wolf.

    Diagnostics that fire on good data would train the user to ignore them.
    """
    imgs, truth = _windows([0, 40, 80, 120])
    _s, tf, diag = R.auto_stitch(imgs, rot_range=2.0, dy_range=3.0, coarse_step=0.5)
    got = np.array([t["dx"] - tf[0]["dx"] for t in tf])
    want = np.array([-(t["x0"] - truth[0]["x0"]) for t in truth])
    assert min(np.abs(got - want).max(), np.abs(got + want).max()) <= 2
    assert diag["closure"]["dx"] < 2.0
    assert diag["flip_conflicts"] == 0


def test_span_reports_the_lever_arm_gain():
    """The span is the whole point, so it has to be reported honestly."""
    offsets = [0, 40, 80, 120]
    imgs, _ = _windows(offsets)
    _s, _tf, diag = R.auto_stitch(imgs, rot_range=2.0, dy_range=3.0, coarse_step=0.5)
    expected = (max(offsets) - min(offsets) + imgs[0].shape[1]) / imgs[0].shape[1]
    assert diag["span_gain"] == pytest.approx(expected, abs=0.1)


def test_isolated_image_is_reported_not_stacked_at_the_origin():
    """An image nothing registers to must be excluded, not silently overlaid."""
    imgs, _ = _windows([0, 40])
    rng = np.random.default_rng(7)
    imgs.append(rng.normal(0, 1, imgs[0].shape).astype(np.float32))
    _s, tf, diag = R.auto_stitch(imgs, rot_range=2.0, dy_range=3.0, coarse_step=0.5)
    assert (
        diag["n_connected"] < len(imgs) or not tf[2]["connected"]
    ), "pure noise should not register to a filament"


def test_single_image_is_a_no_op():
    imgs, _ = _windows([0])
    stitched, tf, diag = R.auto_stitch(imgs)
    assert np.allclose(stitched, imgs[0])
    assert tf[0]["dx"] == 0.0
    assert diag["n_pairs"] == 0


def test_empty_input():
    stitched, tf, diag = R.auto_stitch([])
    assert stitched is None
    assert tf == []


def test_never_silently_wrong():
    """The safety property: a wrong stitch must never report trustworthy.

    Layouts vary in how many pairs can be registered at all, and some cannot be
    solved -- that is expected and fine. What is not acceptable is returning a
    confident wrong answer, so every layout must either come out right or be
    marked untrustworthy.
    """
    layouts = [
        [0, 40, 80, 120],
        [0, 70, 150, 250, 340],
        [0, 50, 100, 150, 200, 250],
        [0, 30, 60, 90, 120, 150, 180],
        [0, 20, 40, 60],
    ]
    for offsets in layouts:
        imgs, truth = _windows(offsets)
        _s, tf, diag = R.auto_stitch(imgs, rot_range=2.0, dy_range=3.0, coarse_step=0.5)
        got = np.array([t["dx"] - tf[0]["dx"] for t in tf])
        want = np.array([-(t["x0"] - truth[0]["x0"]) for t in truth])
        err = min(np.abs(got - want).max(), np.abs(got + want).max())
        if err > 4:
            assert not diag["trustworthy"], (
                f"layout {offsets} was wrong by {err} px but reported trustworthy: "
                f"redundancy={diag['redundancy']} closure={diag['closure']} "
                f"conflicts={diag['flip_conflicts']}"
            )


def test_closure_is_not_claimed_meaningful_without_loops():
    """A tree fits its own measurements exactly, so closure proves nothing there."""
    pairs = [
        dict(i=0, j=1, psi=0.0, dy=0.0, dx=10.0, corr=0.9),
        dict(i=1, j=2, psi=0.0, dy=0.0, dx=10.0, corr=0.9),
    ]
    _tf, diag = R.solve_global(3, pairs, image_width=100)
    assert diag["redundancy"] == 0
    assert diag["closure_meaningful"] is False
    assert diag["trustworthy"] is False
