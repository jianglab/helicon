"""apply_helical_symmetry without numba (as under Pyodide, where numba cannot
run at all): the vectorised numpy/scipy kernel agrees with the numba-fused
one it stands in for.

Without this fallback, the only alternative is lazy_jit's own: the *same*
nested Python loops, un-jitted, measured about 500x slower than the compiled
kernel (tools/shinylive/README.md) -- unusable for interactive use. This
kernel's vectorised numpy version costs a real but far smaller multiple (see
TestPerformance below), paid only when numba is unavailable.
"""

import numpy as np
import pytest

import helicon
from helicon.lib import transforms


def _tube(nz=32, ny=28, nx=28, apix=2.0, radius=7.0, sigma=2.2, rise=6.0, twist=30.0):
    """A synthetic helical tube, the same shape of fixture
    test_helical_symmetry_background.py uses."""
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    z = (z - nz // 2) * apix
    y = (y - ny // 2) * apix
    x = (x - nx // 2) * apix
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
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


class TestKernelAgreement:
    """The two kernels directly, bypassing apply_helical_symmetry's own
    numba/numpy dispatch (which TestDispatch below exercises instead)."""

    @pytest.mark.parametrize(
        "csym, fraction, new_size_factor, new_apix_factor",
        [
            (1, 1.0, 1, 1.0),
            (2, 1.0, 1, 1.0),
            (1, 0.6, 1, 1.0),
            (3, 1.0, 2, 1.0),  # new_size != data.shape
            (1, 1.0, 1, 1.25),  # new_apix != apix
        ],
    )
    def test_agrees_on_a_random_volume(
        self, csym, fraction, new_size_factor, new_apix_factor
    ):
        rng = np.random.default_rng(0)
        n = 24
        vol = (rng.random((n, n, n)).astype(np.float32) - 0.5) * 2
        apix = 2.0
        new_size = (n * new_size_factor, n, n)
        new_apix = apix * new_apix_factor
        kw = dict(
            apix=apix,
            twist_degree=37.0,
            rise_angstrom=5.2,
            csym=csym,
            fraction=fraction,
            new_size=new_size,
            new_apix=new_apix,
        )
        out_numba = transforms._apply_helical_symmetry_jit(vol, **kw)
        out_np = transforms._apply_helical_symmetry_np(vol, **kw)
        np.testing.assert_allclose(out_numba, out_np, rtol=1e-4, atol=1e-6)

    def test_agrees_on_an_unmasked_negative_background_map(self):
        # the edge case test_helical_symmetry_background.py documents: no
        # slice sum passes the threshold, so the (zero-size) fallback branch
        # of both kernels runs
        vol, apix, twist, rise = _tube()
        offset = 2.0 * float(vol.sum(axis=(1, 2)).max()) / (vol.shape[1] * vol.shape[2])
        unmasked = vol - offset
        assert (unmasked.sum(axis=(1, 2)) < 0).all()
        kw = dict(
            apix=apix,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=1,
            fraction=1.0,
            new_size=unmasked.shape,
            new_apix=apix,
        )
        out_numba = transforms._apply_helical_symmetry_jit(unmasked, **kw)
        out_np = transforms._apply_helical_symmetry_np(unmasked, **kw)
        np.testing.assert_allclose(out_numba, out_np, rtol=1e-4, atol=1e-6)

    def test_agrees_on_an_all_zero_map(self):
        vol = np.zeros((16, 16, 16), dtype=np.float32)
        kw = dict(
            apix=1.0,
            twist_degree=30.0,
            rise_angstrom=3.0,
            csym=1,
            fraction=1.0,
            new_size=vol.shape,
            new_apix=1.0,
        )
        out_numba = transforms._apply_helical_symmetry_jit(vol, **kw)
        out_np = transforms._apply_helical_symmetry_np(vol, **kw)
        np.testing.assert_array_equal(out_numba, out_np)
        assert not np.any(out_np)


class TestFftRescaleWithoutFinufft:
    """finufft has no Pyodide/wasm wheel (tools/shinylive/README.md): fft_rescale
    falls back to transforms._nonuniform_grid_dft, a direct separable DFT. It is
    exact, not an eps-approximation like finufft, since fft_rescale always asks
    for a uniform tensor-product query grid (one independently-uniform 1D
    frequency array per axis)."""

    @pytest.mark.parametrize(
        "shape, output_size",
        [
            ((16, 20), None),
            ((16, 20), (18, 14)),  # output_size != data.shape
            ((15, 15), (15, 15)),  # odd sizes
        ],
    )
    def test_2d_matches_finufft(self, monkeypatch, shape, output_size):
        pytest.importorskip("finufft")
        rng = np.random.default_rng(0)
        data = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex128
        )
        ref = transforms.fft_rescale(data, apix=1.0, output_size=output_size)
        monkeypatch.setattr(transforms, "_HAS_FINUFFT", False)
        out = transforms.fft_rescale(data, apix=1.0, output_size=output_size)
        # finufft itself is only accurate to its own eps=1e-6, so this is a
        # check against *that* tolerance, not against the numpy kernel's own
        # precision (which is exact to float64 rounding, see its docstring)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize(
        "shape, output_size",
        [
            ((8, 10, 12), None),
            ((8, 10, 12), (9, 11, 7)),  # output_size != data.shape, odd and even
        ],
    )
    def test_3d_matches_finufft(self, monkeypatch, shape, output_size):
        pytest.importorskip("finufft")
        rng = np.random.default_rng(1)
        data = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex128
        )
        ref = transforms.fft_rescale(data, apix=1.0, output_size=output_size)
        monkeypatch.setattr(transforms, "_HAS_FINUFFT", False)
        out = transforms.fft_rescale(data, apix=1.0, output_size=output_size)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


class TestDispatch:
    """The public helicon.apply_helical_symmetry, through its numba/numpy switch."""

    def test_without_numba_it_still_matches_the_numba_path(self, monkeypatch):
        vol, apix, twist, rise = _tube()
        kw = dict(
            data=vol,
            apix=apix,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=1,
            fraction=1.0,
            new_size=vol.shape,
            new_apix=apix,
            cpu=1,
        )
        out_numba = helicon.apply_helical_symmetry(**kw)
        monkeypatch.setattr(transforms, "_HAS_NUMBA", False)
        out_np = helicon.apply_helical_symmetry(**kw)
        np.testing.assert_allclose(out_numba, out_np, rtol=1e-4, atol=1e-6)

    def test_without_numba_a_structure_is_still_found(self, monkeypatch):
        # the regression test_helical_symmetry_background.py exists for,
        # through the numpy path too
        vol, apix, twist, rise = _tube()
        offset = 2.0 * float(vol.sum(axis=(1, 2)).max()) / (vol.shape[1] * vol.shape[2])
        unmasked = vol - offset
        monkeypatch.setattr(transforms, "_HAS_NUMBA", False)
        out = helicon.apply_helical_symmetry(
            data=unmasked,
            apix=apix,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=1,
            fraction=5 * rise / (vol.shape[0] * apix),
            new_size=(16, vol.shape[1], vol.shape[2]),
            new_apix=apix,
            cpu=1,
        )
        assert np.count_nonzero(out) > 0.5 * out.size
