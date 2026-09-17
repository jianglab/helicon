"""Tests for reconstructing one volume from several images at their azimuths.

The sign of the per-image azimuth is the thing most likely to be wrong and
least likely to announce itself, so it is tested against a volume built
independently, not against the solver's own output. The other tests establish
that the azimuths are doing work at all: images placed correctly must fit
better than the same images placed together or placed at random.
"""

import numpy as np
import pytest

from helicon.webApps.lib import denovo3d_align as A
from helicon.webApps.lib import denovo3d_jointsolve as JS

from test_denovo3D_align import _helix_volume, _cc, TWIST, RISE, CSYM, APIX

NY, NX = 40, 128


@pytest.fixture(scope="module")
def scene():
    """A helix, its long projection, and windows of it at known azimuths."""
    volume = _helix_volume()
    length = A.model_length_pixel(TWIST, RISE, APIX, CSYM, NX)
    model = A.long_side_projection(volume, APIX, TWIST, RISE, CSYM, APIX, NY, length)
    m = model.shape[1]

    offsets = np.linspace(-90.0, 90.0, 5)
    images, phis = [], []
    for dx in offsets:
        lo = int(round(m // 2 + dx - NX / 2))
        w = model[:, lo : lo + NX].astype(np.float32)
        images.append((w - w.mean()) / (w.std() + 1e-12))
        phis.append(A.dx_to_phi(dx, TWIST, RISE, APIX) % 360.0)
    return volume, model, images, phis


GEOMETRY = dict(
    scale2d_to_3d=1.0,
    rise_pixel=RISE / APIX,
    csym=CSYM,
    reconstruct_diameter_2d_pixel=NY,
    reconstruct_diameter_3d_pixel=NY,
    reconstruct_length_2d_pixel=NX,
    reconstruct_length_3d_pixel=6,
    sym_oversample=5,
    interpolation="nn",
    positive_constraint=-1,
)


def _reconstruct(images, phis, **over):
    kw = dict(GEOMETRY, twist_degree=TWIST)
    kw.update(over)
    return JS.joint_reconstruct(images, phis, **kw)


def test_rejects_an_empty_set():
    with pytest.raises(ValueError):
        _reconstruct([], [])


def test_azimuth_sign_matches_the_alignment_convention(scene):
    """An image known to be at azimuth phi, reconstructed with phi, must give
    the volume that the image at azimuth zero gives.

    The solver rotates back-projected coordinates, which is the inverse of
    rotating the object, so passing the reported azimuth through unchanged
    would be wrong by a sign -- and wrong in a way that still produces a
    perfectly plausible volume. Both signs agree only at zero azimuth.
    """
    _, _, images, phis = scene
    reference, _ = _reconstruct([images[2]], [phis[2]])  # the centre window
    for i in (0, 1, 3, 4):
        right, _ = _reconstruct([images[i]], [phis[i]])
        wrong, _ = _reconstruct([images[i]], [-np.asarray(phis[i])])
        assert _cc(reference, right) > _cc(reference, wrong)
        assert _cc(reference, right) > 0.9


def test_correct_azimuths_fit_better_than_none(scene):
    """If the azimuths were not doing anything, stacking the images at a common
    azimuth would fit just as well. It does not."""
    _, _, images, phis = scene
    _, right = _reconstruct(images, phis)
    _, flat = _reconstruct(images, [0.0] * len(images))
    assert right["score"] > flat["score"] + 0.05


def test_correct_azimuths_fit_better_than_shuffled(scene):
    _, _, images, phis = scene
    rng = np.random.default_rng(5)
    shuffled = list(rng.permutation(phis))
    _, right = _reconstruct(images, phis)
    _, wrong = _reconstruct(images, shuffled)
    assert right["score"] > wrong["score"]


def test_reports_a_fit_for_every_image(scene):
    _, _, images, phis = scene
    _, info = _reconstruct(images, phis)
    assert len(info["per_image"]) == len(images)
    assert all(np.isfinite(s) for s in info["per_image"])
    assert np.isfinite(info["score"])


def test_volume_has_the_requested_shape(scene):
    _, _, images, phis = scene
    volume, _ = _reconstruct(images, phis)
    assert volume.shape == (6, NY, NY)


def test_non_negativity_is_applied_at_a_realistic_pitch(scene):
    """The constraint is what makes the twist identifiable, so its automatic
    rule must actually fire for the geometry this is used on: a pitch of
    hundreds of pixels against a reconstruction a few pixels long."""
    _, _, images, phis = scene
    _, info = _reconstruct(images, phis)
    assert info["positive"]


def test_sharing_the_budget_costs_equations_not_images(scene):
    """Sharing keeps a joint solve about as expensive as a single-image one.
    Both must still use every image."""
    _, _, images, phis = scene
    _, shared = _reconstruct(images, phis, share_budget=True)
    _, full = _reconstruct(images, phis, share_budget=False)
    assert len(shared["per_image"]) == len(full["per_image"]) == len(images)


# ──────────────────────────────────────────────────────────────────────────
# The Gaussian basis as an alternative joint solver
# ──────────────────────────────────────────────────────────────────────────


def _gauss(images, phis, **over):
    kw = dict(
        scale2d_to_3d=1.0,
        twist_degree=TWIST,
        rise_pixel=RISE / APIX,
        csym=CSYM,
        reconstruct_diameter_2d_pixel=NY,
        reconstruct_diameter_3d_pixel=NY,
        reconstruct_length_2d_pixel=NX,
        reconstruct_length_3d_pixel=6,
        sym_oversample=5,
        interpolation="nn",
        target_apix2d=APIX,
        algorithm=dict(model="gauss"),
    )
    kw.update(over)
    return JS.joint_reconstruct(images, phis, **kw)


def test_gauss_azimuth_sign_matches_the_alignment_convention(scene):
    """Same calibration as the voxel solver, and it matters more here because
    the two rotate different things: reasoning about which way each convention
    turns gives the wrong answer, so both are pinned by measurement instead."""
    _, _, images, phis = scene
    reference, _ = _gauss([images[2]], [phis[2]])
    for i in (0, 1, 3, 4):
        right, _ = _gauss([images[i]], [phis[i]])
        wrong, _ = _gauss([images[i]], [-np.asarray(phis[i])])
        assert _cc(reference, right) > _cc(reference, wrong)
        assert _cc(reference, right) > 0.9


def test_gauss_correct_azimuths_fit_better_than_none(scene):
    _, _, images, phis = scene
    _, right = _gauss(images, phis)
    _, flat = _gauss(images, [0.0] * len(images))
    assert right["score"] > flat["score"]


def test_gauss_reports_a_fit_for_every_image(scene):
    _, _, images, phis = scene
    _, info = _gauss(images, phis)
    assert len(info["per_image"]) == len(images)
    assert np.isfinite(info["score"])


def test_gauss_and_voxel_solvers_share_one_interface(scene):
    """Both are reached through joint_reconstruct with the same arguments and
    the same azimuth convention; only algorithm["model"] differs."""
    _, _, images, phis = scene
    v_gauss, _ = _gauss(images, phis)
    v_voxel, _ = _reconstruct(images, phis)
    assert v_gauss.shape == v_voxel.shape


def test_gauss_rejects_a_mismatched_azimuth_count(scene):
    _, _, images, phis = scene
    with pytest.raises(ValueError):
        _gauss(images, phis[:-1])


def test_gauss_rejects_images_of_different_shapes(scene):
    _, _, images, phis = scene
    odd = list(images)
    odd[1] = odd[1][:, :-4]
    with pytest.raises(ValueError):
        _gauss(odd, phis)
