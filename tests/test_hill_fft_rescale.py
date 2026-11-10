"""hill_compute.fft_rescale without finufft (as under Pyodide, where finufft
has no wasm wheel): the numpy fallback (helicon.lib.transforms._nonuniform_grid_dft)
agrees with the finufft path it stands in for -- see
tests/test_transforms_numpy_fallback.py::TestFftRescaleWithoutFinufft for the
same check against the canonical helicon.lib.transforms.fft_rescale.
"""

import numpy as np
import pytest

from helicon.lib import transforms
from helicon.webApps.lib import hill_compute as hill


class TestFftRescaleWithoutFinufft:
    @pytest.mark.parametrize(
        "shape, output_size",
        [
            ((16, 20), None),
            ((15, 15), (18, 22)),  # output_size != image.shape, odd input size
        ],
    )
    def test_matches_finufft(self, monkeypatch, shape, output_size):
        pytest.importorskip("finufft")
        rng = np.random.default_rng(2)
        image = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
            np.complex128
        )
        fft_rescale = hill.fft_rescale.__wrapped__
        ref = fft_rescale(image, apix=1.0, output_size=output_size)
        monkeypatch.setattr(transforms, "_HAS_FINUFFT", False)
        out = fft_rescale(image, apix=1.0, output_size=output_size)
        # finufft itself is only accurate to its own eps=1e-6
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
