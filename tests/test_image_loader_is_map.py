"""A stack of 2D classes is not taken for a 3D map."""

import numpy as np

from helicon.webApps.lib import image_loader


def _write(path, shape, ispg):
    import mrcfile

    data = np.zeros(shape, dtype=np.float32)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.header.ispg = ispg
    return data


class TestIsMap:
    def test_more_classes_than_pixels_is_a_stack_by_its_header(self, tmp_path):
        path = str(tmp_path / "classes.mrc")
        data = _write(path, (150, 32, 32), ispg=0)
        assert image_loader.is_map(path, data) is False

    def test_the_same_shape_with_the_header_of_a_map_is_a_map(self, tmp_path):
        path = str(tmp_path / "long_map.mrc")
        data = _write(path, (150, 32, 32), ispg=1)
        assert image_loader.is_map(path, data) is True

    def test_mrcs_is_a_stack_by_its_name_also_for_an_upload_and_a_url(self, tmp_path):
        data = np.zeros((150, 32, 32), dtype=np.float32)
        assert (
            image_loader.is_map("https://example.org/run_classes.mrcs", data) is False
        )
        assert (
            image_loader.is_map(str(tmp_path / "tmpab12"), data, "run_classes.mrcs")
            is False
        )

    def test_a_cubic_map_is_a_map(self, tmp_path):
        path = str(tmp_path / "cube.mrc")
        data = _write(path, (16, 16, 16), ispg=1)
        assert image_loader.is_map(path, data) is True

    def test_a_url_without_a_header_falls_back_on_the_shape(self):
        assert (
            image_loader.is_map("https://example.org/m.map", np.zeros((8, 8, 8)))
            is True
        )
        assert (
            image_loader.is_map("https://example.org/m.map", np.zeros((8, 8))) is False
        )
