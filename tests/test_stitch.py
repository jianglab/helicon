"""Tests for automatic multi-image registration (virtual stitching).

The interesting failures here are not crashes but plausible-looking wrong
answers: a pair with no overlap still registers confidently, a flip choice can
be inconsistent around a loop, and composing two transforms can silently
introduce a half-pixel shift. Each of those has a test.
"""

import numpy as np
import pytest

import helicon
from helicon.lib import stitch as R


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


class TestBestCorrMatrix:
    """The correlations of every pair at once, as register_pair finds them."""

    @pytest.fixture(scope="class")
    def images(self):
        rng = np.random.default_rng(0)
        y, x = np.mgrid[:40, :64]
        out = []
        for k in range(7):
            band = np.exp(-((y - 20 - 3 * np.sin(x / (6 + k) + k)) ** 2) / 20.0)
            out.append(
                (band + 0.3 * rng.standard_normal(band.shape)).astype(np.float32)
            )
        return out

    def test_it_agrees_with_registering_pair_by_pair(self, images):
        S = R.best_corr_matrix(images, workers=1)
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                one = R.register_pair(images[i], images[j], flips=R.FLIPS, refine=False)
                assert S[i, j] == pytest.approx(np.clip(one["corr"], 0, 1), abs=1e-9)
        assert np.allclose(S, S.T) and np.allclose(np.diag(S), 1.0)

    def test_processes_give_the_same_matrix(self, images):
        assert np.allclose(
            R.best_corr_matrix(images, workers=1), R.best_corr_matrix(images, workers=3)
        )

    def test_blocks_give_the_same_matrix(self, images, monkeypatch):
        whole = R.best_corr_matrix(images, workers=1)
        monkeypatch.setattr(R, "CORR_BLOCK", 2)
        assert np.allclose(R.best_corr_matrix(images, workers=1), whole)

    def test_progress_counts_the_pairs(self, images):
        seen = []
        R.best_corr_matrix(images, workers=1, progress=lambda k, n: seen.append((k, n)))
        assert seen[-1] == (21, 21)

    def test_images_of_different_shapes_are_refused(self, images):
        with pytest.raises(ValueError):
            R.best_corr_matrix([images[0], images[1][:, :50]])


class TestRegisterMany:
    """register_pair for many pairs at once."""

    @pytest.fixture(autouse=True)
    def fresh(self):
        # each test computes its registrations, not the cache of an earlier one
        R.clear_pair_cache()
        yield
        R.clear_pair_cache()

    @pytest.fixture(scope="class")
    def images(self):
        return TestBestCorrMatrix.images.__wrapped__(None)

    @pytest.mark.parametrize(
        "flips,rot,dy", [(R.FLIPS, 6.0, 5.0), (((False, False),), 3.0, 3.0)]
    )
    def test_it_gives_what_register_pair_gives(self, images, flips, rot, dy):
        pairs = [(0, 3), (1, 3), (2, 5), (4, 1), (6, 0)]
        many = R.register_many(images, pairs, flips, rot, dy, workers=1)
        for (i, j), got in zip(pairs, many):
            one = R.register_pair(
                images[i],
                images[j],
                rot_range=rot,
                dy_range=dy,
                flips=flips,
                refine=False,
            )
            for key in ("corr", "dx", "prominence", "psi", "dy"):
                assert got[key] == pytest.approx(one[key], abs=1e-9), key
            assert (got["flip_x"], got["flip_y"]) == (one["flip_x"], one["flip_y"])

    def test_processes_give_the_same_results(self, images):
        pairs = [(i, j) for i in range(7) for j in range(7) if i != j]
        one = R.register_many(images, pairs, workers=1)
        R.clear_pair_cache()
        many = R.register_many(images, pairs, workers=3)
        assert all(a == pytest.approx(b) for a, b in zip(one, many))

    def test_refining_from_the_bulk_coarse_search_is_register_pair(self, images):
        pairs = [(0, 3), (2, 5), (6, 1)]
        coarse = R.register_many(images, pairs, workers=1)
        for (i, j), c in zip(pairs, coarse):
            full = R.register_pair(images[i], images[j], refine=True)
            fast = R.register_pair(images[i], images[j], refine=True, coarse=c)
            assert fast == pytest.approx(full, abs=1e-9)

    def test_a_repeat_is_read_from_the_cache(self, images, monkeypatch):
        pairs = [(0, 3), (2, 5)]
        first = R.register_many(images, pairs, workers=1)

        def no(*a, **k):
            raise AssertionError("registered again")

        monkeypatch.setattr(R, "_register_column", no)
        assert R.register_many(images, pairs, workers=1) == first
        # the order and a pair asked for twice do not matter
        again = R.register_many(images, [(2, 5), (0, 3), (2, 5)], workers=1)
        assert again == [first[1], first[0], first[1]]

    def test_other_images_or_another_search_are_not_read_from_it(self, images):
        pairs = [(0, 3)]
        first = R.register_many(images, pairs, workers=1)[0]
        other = R.register_many(images, pairs, workers=1, rot_range=2.0)[0]
        changed = [im.copy() for im in images]
        changed[3] = changed[3][::-1].copy()
        turned = R.register_many(changed, pairs, workers=1)[0]
        fresh = R.register_pair(images[0], changed[3], refine=False)
        assert turned["corr"] == pytest.approx(fresh["corr"], abs=1e-9)
        assert (other["psi"], other["corr"]) != (first["psi"], first["corr"]) or abs(
            first["psi"]
        ) <= 2.0


# ── extending the stitch by whole periods ───────────────────────────────

PERIOD = 180.5


def _periodic_window(x0, width=200, ny=40, seed=0):
    """A window of a filament that repeats every ``PERIOD`` columns (no noise but a little)."""
    x = np.arange(width) + x0
    t = 2 * np.pi * x / PERIOD
    axial = 1 + 0.6 * np.cos(t + 0.3) + 0.4 * np.cos(2 * t + 1.1) + 0.3 * np.cos(3 * t)
    axial = axial + 0.3 * np.cos(2 * np.pi * x / 9.0)
    y = np.arange(ny) - ny // 2
    img = np.exp(-0.5 * (y[:, None] / 6.0) ** 2) * axial[None, :]
    rng = np.random.default_rng(seed)
    return (img + rng.normal(0, 0.01, img.shape)).astype(np.float32)


class TestGrowingAStitch:
    def _tiles(self, offsets, width=128):
        return [
            _periodic_window(x0, width=width, seed=n) for n, x0 in enumerate(offsets)
        ]

    def test_images_are_placed_end_to_end_into_the_long_filament(self):
        # six windows of a filament, in an order the growth has to find itself
        offsets = [0, 60, 120, 180, 240, 300]
        imgs = self._tiles([offsets[i] for i in (2, 5, 0, 3, 1, 4)])
        placed = R.grow_stitch(imgs, first=0, min_corr=0.9)
        assert len(placed) == 6
        long_img, _ = R.composite(
            [imgs[k] for k in placed],
            [dict(psi=0.0, dy=0.0, dx=dx) for dx in placed.values()],
            match_intensity=False,
        )
        assert long_img.shape[1] > 300
        width = long_img.shape[1]
        truth = _periodic_window(0, width=width + int(PERIOD), seed=9)
        best = max(
            np.corrcoef(long_img.ravel(), truth[:, s : s + width].ravel())[0, 1]
            for s in range(int(PERIOD))
        )
        assert best > 0.97

    def test_every_image_extends_the_stitch_by_a_fair_part_of_its_width(self):
        imgs = self._tiles([0, 20, 40, 60, 80, 100])
        placed = R.grow_stitch(imgs, first=0, min_corr=0.9)
        dxs = sorted(placed.values())
        assert all(b - a >= 0 for a, b in zip(dxs, dxs[1:]))
        # images only 20 apart cannot each add 32 columns: some are left out
        assert len(placed) < 6

    def test_images_that_fit_nowhere_are_left_out(self):
        imgs = self._tiles([0, 40])
        rng = np.random.default_rng(3)
        imgs.append(rng.normal(0, 1, imgs[0].shape).astype(np.float32))
        placed = R.grow_stitch(imgs, first=0, min_corr=0.9)
        assert set(placed) == {0, 1}


class TestRefiningOneImage:
    def test_an_image_at_the_end_keeps_its_rotation_and_its_repeat(self):
        offsets = [0, 60, 120]
        imgs = [_periodic_window(x0, width=128, seed=n) for n, x0 in enumerate(offsets)]
        true = [
            dict(
                psi=0.0,
                dy=0.0,
                dx=float(x0),
                flip_x=False,
                flip_y=False,
                connected=True,
            )
            for x0 in offsets
        ]
        guess = [dict(t) for t in true]
        guess[2].update(psi=0.3, dx=121.0)
        out = R._refine_one(imgs, guess, 2, dict(try_flips=False, rot_range=1.0))
        assert abs(out["psi"]) < 0.5
        assert abs(out["dx"] - 120.0) < R.REFINE_DX_WINDOW


class TestStitchImages:
    def _tiles(self):
        return [
            _periodic_window(x0, width=128, seed=n) for n, x0 in enumerate([0, 50, 100])
        ]

    def test_stitches_images_with_the_helix_along_x(self):
        res = R.stitch_images(self._tiles(), axis="x", rot_range=1.0, dy_range=1.0)
        assert res.image.shape[0] == 40 and res.image.shape[1] > 128
        assert len(res.composed) == 3 and res.diagnostics["n_connected"] == 3

    def test_the_helix_along_y_gives_the_same_stitch_turned(self):
        tiles = self._tiles()
        along_x = R.stitch_images(tiles, axis="x", rot_range=1.0, dy_range=1.0)
        along_y = R.stitch_images(
            [t.T for t in tiles], axis="y", rot_range=1.0, dy_range=1.0
        )
        assert along_y.image.shape == along_x.image.T.shape
        assert np.allclose(along_y.image, along_x.image.T, atol=1e-4)

    def test_extending_is_reported_and_never_shortens(self):
        tiles = self._tiles()
        plain = R.stitch_images(tiles, rot_range=1.0, dy_range=1.0)
        longer = R.stitch_images(tiles, extend=True, rot_range=1.0, dy_range=1.0)
        assert longer.image.shape[1] >= plain.image.shape[1]
        assert longer.extension["grown_of"] == 3
        assert plain.extension == {}

    def test_a_wrong_axis_is_refused(self):
        with pytest.raises(ValueError):
            R.stitch_images(self._tiles(), axis="z")

    def test_the_auto_transform_of_a_helix_along_y_is_folded_in(self):
        import helicon

        straight = [t.T.copy() for t in self._tiles()]
        auto = [(2.0, 1.5), (-1.5, -1.0), (1.0, 2.0)]
        # what the auto-transform undoes: each image tilted and off-centre
        tilted = [
            helicon.transform_image(im, rotation=-r, post_translation=(0, -sy))
            for im, (r, sy) in zip(straight, auto)
        ]
        restored = [
            helicon.transform_image(im, rotation=r, post_translation=(0, sy))
            for im, (r, sy) in zip(tilted, auto)
        ]
        kw = dict(axis="y", rot_range=1.0, dy_range=1.0)
        res = R.stitch_images(restored, originals=tilted, auto=auto, **kw)
        ref = R.stitch_images(straight, **kw)
        assert res.image.shape == ref.image.shape
        assert np.corrcoef(res.image.ravel(), ref.image.ravel())[0, 1] > 0.95
