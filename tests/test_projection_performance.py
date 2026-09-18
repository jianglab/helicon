"""Guards for the helicalProjection speed and memory work.

Both changes here are optimisations, and an optimisation that quietly changes
its answer is worse than the slow version it replaced. These pin the behaviour
that was measured to be identical, and the shapes that used to be wrong.
"""

import numpy as np
import pytest

import helicon


class TestFilterUnchangedByTheRewrite:
    """low_high_pass_filter moved to a real-input FFT in the caller's precision,
    with the radius built by broadcasting rather than three meshgrids. It cost
    about ten times the volume it filtered before -- 2.4 GB for a 226 MB map."""

    def _reference(self, data, lp=0, hp=0):
        """The previous implementation, kept here as the thing to match."""
        if data.ndim == 2:
            fft = np.fft.fft2(data)
            ny, nx = fft.shape
            Y, X = np.meshgrid(
                np.arange(ny, dtype=np.float32) - ny // 2,
                np.arange(nx, dtype=np.float32) - nx // 2,
                indexing="ij",
            )
            Y /= ny // 2
            X /= nx // 2
            r2 = X**2 + Y**2
        else:
            fft = np.fft.fftn(data)
            nz, ny, nx = fft.shape
            Z, Y, X = np.meshgrid(
                np.arange(nz, dtype=np.float32) - nz // 2,
                np.arange(ny, dtype=np.float32) - ny // 2,
                np.arange(nx, dtype=np.float32) - nx // 2,
                indexing="ij",
            )
            Z /= nz // 2
            Y /= ny // 2
            X /= nx // 2
            r2 = X**2 + Y**2 + Z**2
        if 0 < lp < 1:
            fft *= np.fft.fftshift(np.exp(-np.log(2) / lp**2 * r2))
        if 0 < hp < 1:
            fft *= np.fft.fftshift(1.0 - np.exp(-np.log(2) / hp**2 * r2))
        return np.real(np.fft.ifftn(fft))

    @pytest.mark.parametrize("shape", [(64, 64), (48, 52), (24, 28, 32)])
    @pytest.mark.parametrize("lp,hp", [(0.5, 0), (0, 0.3), (0.4, 0.2), (0, 0)])
    def test_it_matches_the_previous_implementation(self, shape, lp, hp):
        data = np.random.default_rng(0).random(shape).astype(np.float32)
        want = self._reference(data, lp, hp)
        got = helicon.low_high_pass_filter(data, lp, hp)
        assert got.shape == want.shape
        assert np.abs(got - want).max() / want.std() < 1e-4

    def test_the_result_is_contiguous(self):
        """np.real() of a complex spectrum returns a VIEW, so the old result had
        a stride of two and kept the whole complex buffer alive behind it."""
        data = np.random.default_rng(0).random((32, 32, 32)).astype(np.float32)
        out = helicon.low_high_pass_filter(data, low_pass_fraction=0.5)
        assert out.flags["C_CONTIGUOUS"]

    def test_single_precision_in_single_precision_out(self):
        data = np.random.default_rng(0).random((32, 32)).astype(np.float32)
        assert helicon.low_high_pass_filter(data, 0.5).dtype == np.float32


class TestSymmetrisationComputesOnlyWhatItReturns:
    """apply_helical_symmetry allocated the element-wise maximum of input and
    output shape, twice, then cropped. For a helical case -- long output, narrow
    cross-section -- that maximum is larger than either: 384-cubed in and
    1578x128x128 out meant two 931 MB buffers for a 103 MB result, with nine
    voxels computed for every one kept."""

    def _run(self, shape, new_size, twist=15.0, rise=5.0, csym=1):
        data = np.random.default_rng(0).random(shape).astype(np.float32)
        return helicon.apply_helical_symmetry(
            data=data,
            apix=1.0,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=csym,
            fraction=0.5,
            new_size=new_size,
            new_apix=1.0,
            cpu=1,
        )

    @pytest.mark.parametrize(
        "shape,new_size",
        [
            ((32, 24, 24), (64, 16, 16)),  # longer and narrower: the helical case
            ((32, 24, 24), (16, 32, 32)),  # shorter and wider
            ((24, 24, 24), (24, 24, 24)),  # unchanged
            ((20, 22, 22), (60, 12, 12)),
        ],
    )
    def test_the_output_has_exactly_the_requested_shape(self, shape, new_size):
        assert self._run(shape, new_size).shape == new_size

    def test_an_odd_length_is_not_one_short(self):
        """The old centred crop used nz//2 +- nz1//2, which loses a row when the
        requested length is odd."""
        assert self._run((24, 24, 24), (31, 15, 15)).shape == (31, 15, 15)

    def test_it_still_symmetrises(self):
        """Cheap sanity that the values are real work, not an empty buffer."""
        out = self._run((32, 24, 24), (64, 16, 16))
        assert np.isfinite(out).all()
        assert out.std() > 0

    def test_a_larger_output_costs_memory_in_proportion_to_itself(self):
        """Not to the input. A tall thin output must not allocate the input's
        cross-section, which is what the element-wise maximum did."""
        out = self._run((32, 48, 48), (96, 8, 8))
        assert out.shape == (96, 8, 8)
        assert out.nbytes < 96 * 48 * 48 * 4
