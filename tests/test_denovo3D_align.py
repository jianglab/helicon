"""Tests for placing a 2D image along a projection of the 3D model.

The claims that matter here are geometric, and each is checked against a
synthetic helix rather than against the code's own output: that rotating a
helically symmetric volume about its axis is the same as sliding it along that
axis, that the long projection is assembled without a seam however many slabs
it takes, and that the azimuth reported for a placement is the rotation which
actually reproduces the image. A sign error in any of them would be invisible
in ordinary use and wrong everywhere.
"""

import numpy as np
import pytest

import helicon
from helicon.webApps.lib import denovo3d_align as A

TWIST, RISE, CSYM = 1.2, 4.75, 1
APIX = 4.944
DIAM, BALL, NSUB = 150.0, 22.0, 6


def _helix_volume(nz=48, ny=40, twist=TWIST, rise=RISE, csym=CSYM, seed=11):
    """Gaussian subunits on a helical lattice, axis along z (index 0)."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform((DIAM / 4) ** 2, (DIAM / 2) ** 2, NSUB))
    th = rng.uniform(0, 2 * np.pi, NSUB)
    z0 = rng.uniform(-rise / 2, rise / 2, NSUB)
    imax = int(np.ceil((nz * APIX / 2 + BALL) / rise)) + 1
    cz, cy, cx = [], [], []
    for i in range(-imax, imax + 1):
        for si in range(csym):
            a = np.deg2rad(twist * i + 360.0 * si / csym)
            cz.append(z0 + i * rise)
            cy.append(r * np.sin(th + a))
            cx.append(r * np.cos(th + a))
    cz, cy, cx = np.concatenate(cz), np.concatenate(cy), np.concatenate(cx)
    Z = (np.arange(nz) - nz // 2) * APIX
    Y = (np.arange(ny) - ny // 2) * APIX
    vol = np.zeros((nz, ny, ny), dtype=np.float32)
    s2 = BALL * BALL / np.log(2)
    for k, zz in enumerate(Z):
        sel = np.abs(cz - zz) < 3 * BALL
        if not sel.any():
            continue
        dz2 = (cz[sel] - zz) ** 2
        dy = Y[:, None, None] - cy[sel][None, None, :]
        dx = Y[None, :, None] - cx[sel][None, None, :]
        vol[k] = np.exp(-(dz2[None, None, :] + dy**2 + dx**2) / s2).sum(axis=2)
    return vol


def _cc(a, b):
    a = np.asarray(a, float).ravel() - np.mean(a)
    b = np.asarray(b, float).ravel() - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


@pytest.fixture(scope="module")
def volume():
    return _helix_volume()


@pytest.fixture(scope="module")
def model(volume):
    length = A.model_length_pixel(TWIST, RISE, APIX, CSYM, 128)
    return A.long_side_projection(volume, APIX, TWIST, RISE, CSYM, APIX, 40, length)


# ──────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────


def test_period_is_a_whole_turn_of_the_helix():
    # rise 4.75 A per 1.2 degrees, so 360 degrees is 1425 A, or 288 px at 4.944.
    assert A.period_pixel(1.2, 4.75, 4.944) == pytest.approx(288.23, abs=0.01)


def test_csym_divides_the_period():
    # csym n makes azimuths 360/n apart identical, so the projection repeats n
    # times as often. Searching a whole turn would re-test the same azimuths.
    assert A.period_pixel(1.2, 4.75, 4.944, csym=3) == pytest.approx(
        A.period_pixel(1.2, 4.75, 4.944) / 3
    )


def test_dx_and_phi_are_inverses():
    for dx in (-100.0, 0.0, 37.5, 120.0):
        phi = A.dx_to_phi(dx, TWIST, RISE, APIX)
        assert A.phi_to_dx(phi, TWIST, RISE, APIX) == pytest.approx(dx)


def test_model_covers_one_period_plus_the_image():
    # Short of this, some azimuth cannot be tested with the image lying wholly
    # inside the model, and that offset would be scored on a partial overlap.
    m = A.model_length_pixel(TWIST, RISE, APIX, CSYM, 128)
    assert m >= A.period_pixel(TWIST, RISE, APIX, CSYM) + 128


# ──────────────────────────────────────────────────────────────────────────
# The long projection
# ──────────────────────────────────────────────────────────────────────────


def test_a_chunked_projection_has_no_seam(volume):
    """Assembling from many slabs must equal assembling from one.

    Slabs after the first are produced by rotating the model, which is only
    the same thing as moving along it because of the equivalence this module
    rests on -- so this also tests that, and the sign of it.
    """
    length = A.model_length_pixel(TWIST, RISE, APIX, CSYM, 128)
    one = A.long_side_projection(volume, APIX, TWIST, RISE, CSYM, APIX, 40, length)
    saved = A.MAX_SLAB_VOXELS
    try:
        A.MAX_SLAB_VOXELS = 40 * 40 * 100  # forces several slabs
        many = A.long_side_projection(volume, APIX, TWIST, RISE, CSYM, APIX, 40, length)
    finally:
        A.MAX_SLAB_VOXELS = saved
    assert many.shape == one.shape
    assert _cc(one, many) > 0.999


def test_projection_is_periodic_in_one_period(volume):
    """Content a whole period apart is the same content, which is why the
    search range is one period and not the whole model."""
    period = int(round(A.period_pixel(TWIST, RISE, APIX, CSYM)))
    length = 2 * period + 128
    proj = A.long_side_projection(volume, APIX, TWIST, RISE, CSYM, APIX, 40, length)
    a = proj[:, 100:228]
    b = proj[:, 100 + period : 228 + period]
    assert _cc(a, b) > 0.99


# ──────────────────────────────────────────────────────────────────────────
# Placement
# ──────────────────────────────────────────────────────────────────────────


def test_sliding_correlation_finds_an_exact_window():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(8, 300))
    for lo in (0, 7, 101, 172):
        k, c = A.sliding_ncc(m, m[:, lo : lo + 64])
        assert k[int(np.argmax(c))] == lo
        assert c.max() == pytest.approx(1.0)


def test_image_wider_than_the_model_is_refused():
    with pytest.raises(ValueError):
        A.sliding_ncc(np.zeros((4, 10)), np.zeros((4, 20)))


@pytest.mark.parametrize("dx_true", [-100.0, -40.0, 0.0, 37.0, 120.0])
def test_a_window_is_placed_back_where_it_came_from(model, dx_true):
    m = model.shape[1]
    lo = int(round(m // 2 + dx_true - 64))
    res = A.align_to_model(model[:, lo : lo + 128], model, TWIST, RISE, APIX)
    assert res["dx"] == pytest.approx(dx_true, abs=0.5)
    assert res["corr"] > 0.999
    assert not res["reversed"]


@pytest.mark.parametrize("dx_true", [-100.0, 37.0, 120.0])
def test_the_reported_azimuth_is_a_real_rotation(volume, model, dx_true):
    """The azimuth reported for a placement must be the rotation that, applied
    to the volume, reproduces the image -- not its negative. Both signs match
    at dx=0 and differ everywhere else, so this is the test that pins it."""
    m = model.shape[1]
    lo = int(round(m // 2 + dx_true - 64))
    window = model[:, lo : lo + 128]
    phi = A.align_to_model(window, model, TWIST, RISE, APIX)["phi"]

    def projected(angle):
        rotated = helicon.transform_map(volume, rot=angle)
        return A.long_side_projection(rotated, APIX, TWIST, RISE, CSYM, APIX, 40, 128)

    assert _cc(projected(phi), window) > 0.99
    assert _cc(projected(-phi), window) < _cc(projected(phi), window)


def test_a_y_mirror_is_a_half_period_shift(volume):
    """The identity the whole design rests on, measured rather than assumed.

    Projecting along the viewing axis, a 180 degree rotation about the helical
    axis mirrors the projection in y; that same rotation is half a period of
    axial shift. So for any helical structure M(-y, z) = M(y, z + period/2),
    which is why no mirror branch is needed.
    """
    period = A.period_pixel(TWIST, RISE, APIX, CSYM)
    length = int(2 * period + 256)
    proj = A.long_side_projection(volume, APIX, TWIST, RISE, CSYM, APIX, 40, length)
    half = int(round(period / 2))
    lo = length // 4
    plain = proj[:, lo : lo + 128]
    shifted = proj[:, lo + half : lo + half + 128]
    assert _cc(plain[::-1, :], shifted) > 0.97
    # ...and it is specific: a quarter period does not do it.
    quarter = int(round(period / 4))
    other = proj[:, lo + quarter : lo + quarter + 128]
    assert _cc(plain[::-1, :], other) < _cc(plain[::-1, :], shifted)


def test_polarity_is_an_in_plane_rotation_not_a_mirror(model):
    """A class average picked from the other end of the filament is related to
    the model by an in-plane 180 degree rotation, and must be recognised as
    that rather than needing a mirror branch."""
    m = model.shape[1]
    window = model[:, m // 2 - 64 : m // 2 + 64]
    reversed_view = window[::-1, ::-1]  # in-plane 180 degrees
    res = A.align_to_model(reversed_view, model, TWIST, RISE, APIX)
    assert res["reversed"]
    assert res["corr"] > 0.95


def test_an_upright_window_is_not_reported_reversed(model):
    m = model.shape[1]
    window = model[:, m // 2 - 64 : m // 2 + 64]
    res = A.align_to_model(window, model, TWIST, RISE, APIX)
    assert not res["reversed"]
    assert abs(res["psi"]) <= A.PSI_RANGE


def test_placement_reports_a_rival(model):
    """Over one period the profile is a single broad hump, so prominence --
    the peak against the whole profile -- reads low even when the placement is
    certain. The rival, the best peak outside this one's shoulders, is what
    says whether a second placement competes."""
    m = model.shape[1]
    res = A.align_to_model(
        model[:, m // 2 - 64 : m // 2 + 64], model, TWIST, RISE, APIX
    )
    assert res["corr"] > res["rival"]
    assert res["width"] > 0


# ──────────────────────────────────────────────────────────────────────────
# Two-fold about the helical axis
# ──────────────────────────────────────────────────────────────────────────


def test_two_fold_halves_the_search_range():
    """A structure with a two-fold about the axis has a projection that
    repeats twice as often, so the placement search covers half as much."""
    full = A.period_pixel(TWIST, RISE, APIX, csym=1)
    half = A.period_pixel(TWIST, RISE, APIX, csym=1, two_fold=True)
    assert half == pytest.approx(full / 2)
    assert A.model_length_pixel(
        TWIST, RISE, APIX, 1, 128, two_fold=True
    ) < A.model_length_pixel(TWIST, RISE, APIX, 1, 128)


def test_two_fold_is_not_csym():
    """Halving the search range must not be confused with imposing C2, which
    is a claim about the density and moved the recovered twist from 1.20 to
    1.05 on EMPIAR-10940 (a 2-1 screw, which has no C2)."""
    assert A.period_pixel(TWIST, RISE, APIX, csym=2) == pytest.approx(
        A.period_pixel(TWIST, RISE, APIX, csym=1, two_fold=True)
    )
    # ...they agree on the period and on nothing else: two_fold is a parameter
    # of the placement search, csym a parameter of the reconstruction.


@pytest.mark.parametrize("csym", [2, 4, 6])
def test_even_symmetry_needs_no_shift_with_the_mirror(csym):
    assert A.mirror_shift_pixel(TWIST, RISE, APIX, csym) == 0.0


@pytest.mark.parametrize("csym", [1, 3, 5])
def test_odd_symmetry_pairs_the_mirror_with_a_shift(csym):
    """For odd n, pitch/2 is not a whole number of pitch/n periods; the
    remainder is pitch/(2n), a shift along the helical axis."""
    period = A.period_pixel(TWIST, RISE, APIX, csym)
    assert A.mirror_shift_pixel(TWIST, RISE, APIX, csym) == pytest.approx(period / 2)


def test_a_plain_helix_cannot_match_its_own_mirror():
    """csym=1 is odd, and its mirror shift is half a pitch -- far longer than
    any window, which is why a filament with no two-fold shows no self-mirror
    symmetry however the image is shifted."""
    shift = A.mirror_shift_pixel(TWIST, RISE, APIX, 1)
    assert shift == pytest.approx(A.period_pixel(TWIST, RISE, APIX, 1) / 2)
    assert shift > 128  # wider than the images this is used on
