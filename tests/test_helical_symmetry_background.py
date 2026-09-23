"""Symmetrising a map whose solvent is not zero.

apply_helical_symmetry finds the structure along z from the sum of each
slice, which assumes the solvent contributes nothing -- true of a masked map.
An unmasked map normalised to a slightly negative background breaks that:
every slice of EMD-19855 sums negative, no slice passed the threshold, and
the empty index array was read past its end in compiled code, producing a
window of garbage and an output of exact zeros. The gaussian search then
reported "no density above the threshold" and skipped the map; the voxel
search scored it 0.000.
"""

import numpy as np
import pytest

import helicon


def _tube(nz=48, ny=40, nx=40, apix=2.0, radius=8.0, sigma=2.5):
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    z = (z - nz // 2) * apix
    y = (y - ny // 2) * apix
    x = (x - nx // 2) * apix
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    rise, twist = 6.0, 30.0
    for k in range(-nz, nz):
        cz = k * rise
        if abs(cz) > nz * apix / 2 + 3 * sigma:
            continue
        ang = np.deg2rad(twist * k)
        vol += np.exp(
            -(
                (x - radius * np.cos(ang)) ** 2
                + (y - radius * np.sin(ang)) ** 2
                + (z - cz) ** 2
            )
            / (2 * sigma**2)
        )
    return vol, apix, twist, rise


def _symmetrize(vol, apix, twist, rise):
    return helicon.apply_helical_symmetry(
        data=vol,
        apix=apix,
        twist_degree=twist,
        rise_angstrom=rise,
        csym=1,
        fraction=5 * rise / (vol.shape[0] * apix),
        new_size=(32, vol.shape[1], vol.shape[2]),
        new_apix=apix,
        cpu=1,
    )


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


class TestAnUnmaskedMapWithANegativeBackground:
    @pytest.fixture
    def maps(self):
        vol, apix, twist, rise = _tube()
        # the solvent shifted below zero until every slice sums negative, the
        # condition EMD-19855 is in
        offset = 2.0 * float(vol.sum(axis=(1, 2)).max()) / (vol.shape[1] * vol.shape[2])
        unmasked = vol - offset
        return vol, unmasked, apix, twist, rise

    def test_the_condition_holds(self, maps):
        _, unmasked, *_ = maps
        assert (unmasked.sum(axis=(1, 2)) < 0).all()

    def test_it_is_not_symmetrised_into_zeros(self, maps):
        _, unmasked, apix, twist, rise = maps
        out = _symmetrize(unmasked, apix, twist, rise)
        assert np.count_nonzero(out) > 0.5 * out.size

    def test_it_gives_the_same_structure_as_the_masked_map(self, maps):
        vol, unmasked, apix, twist, rise = maps
        masked_out = _symmetrize(vol, apix, twist, rise)
        unmasked_out = _symmetrize(unmasked, apix, twist, rise)
        # symmetrising is linear, so a constant offset changes nothing but the
        # constant -- up to the slab each is sampled from
        assert _ncc(masked_out, unmasked_out) > 0.9


class TestDegenerateMaps:
    def test_an_all_zero_map_gives_zeros_rather_than_garbage(self):
        vol = np.zeros((24, 20, 20), dtype=np.float32)
        out = _symmetrize(vol, 2.0, 30.0, 6.0)
        assert out.shape == (32, 20, 20)
        assert not np.any(out)

    def test_a_masked_map_is_untouched_by_the_fallback(self):
        # the fallback runs only when the original test finds nothing, so a
        # masked map must still come out with its structure
        vol, apix, twist, rise = _tube()
        out = _symmetrize(vol, apix, twist, rise)
        assert out.max() > 0.5 * vol.max()
