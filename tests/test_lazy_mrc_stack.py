"""A stack of MRC planes is shown without building anything for each plane."""

import time

import mrcfile
import numpy as np
import pytest

from helicon.lib.gui.viewer import _LazyMrcStack


@pytest.fixture
def stack(tmp_path):
    data = np.random.default_rng(0).normal(size=(7, 5, 6)).astype(np.float32)
    path = tmp_path / "s.mrcs"
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(data)
    return data, np.memmap(
        path, dtype=np.float32, mode="r", offset=1024, shape=data.shape
    )


class TestLazyMrcStack:
    def test_it_has_what_napari_asks_of_an_array(self, stack):
        data, mm = stack
        lazy = _LazyMrcStack(mm)
        assert lazy.shape == (7, 5, 6) and lazy.ndim == 3 and lazy.dtype == np.float32
        assert len(lazy) == 7 and lazy.nbytes == data.nbytes

    def test_it_gives_the_planes_that_are_asked_for(self, stack):
        data, mm = stack
        lazy = _LazyMrcStack(mm)
        np.testing.assert_array_equal(lazy[3], data[3])
        np.testing.assert_array_equal(lazy[2:5], data[2:5])
        np.testing.assert_array_equal(lazy[4, 1:3, ::2], data[4, 1:3, ::2])
        np.testing.assert_array_equal(lazy[[0, 6]], data[[0, 6]])
        np.testing.assert_array_equal(np.asarray(lazy), data)
        assert isinstance(lazy[1], np.ndarray) and not isinstance(lazy[1], np.memmap)

    def test_building_it_costs_nothing_per_plane(self, tmp_path):
        # 120,000 planes (a sparse 2 GB file): the dask graph this replaces took 21 s
        path = tmp_path / "big.mrcs"
        with mrcfile.new_mmap(path, shape=(120000, 64, 64), mrc_mode=2, overwrite=True):
            pass
        mm = np.memmap(
            path, dtype=np.float32, mode="r", offset=1024, shape=(120000, 64, 64)
        )
        start = time.perf_counter()
        lazy = _LazyMrcStack(mm)
        _ = lazy[60000]
        assert time.perf_counter() - start < 0.5
        assert lazy.shape[0] == 120000
