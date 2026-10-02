"""Smaller fixes made alongside the survey fixes."""

import argparse
import gzip
import shutil

import mrcfile
import numpy as np
import pandas as pd
import pytest

from helicon.lib import io
from helicon.lib.exceptions import HeliconError


class TestAstigmatismConversion:
    @pytest.mark.parametrize(
        "u, v, angle",
        [(20000.0, 18000.0, 30.0), (18000.0, 20000.0, 30.0), (20000.0, 18000.0, 100.0)],
    )
    def test_relion_to_eman_and_back_is_the_same_ctf(self, u, v, angle):
        eman = io.relion_astigmatism_to_eman(u, v, angle)
        u2, v2, angle2 = io.eman_astigmatism_to_relion(*eman)

        # the same astigmatism: the defocus along any direction agrees
        def defocus(du, dv, a, theta):
            return (
                du * np.cos(np.deg2rad(theta - a)) ** 2
                + dv * np.sin(np.deg2rad(theta - a)) ** 2
            )

        for theta in (0.0, 37.0, 90.0, 145.0):
            assert defocus(u, v, angle, theta) == pytest.approx(
                defocus(u2, v2, angle2, theta)
            )


class TestProcessOption:
    def test_it_stops_with_a_clear_message(self):
        from helicon.plugins.images2star import process

        parser = argparse.ArgumentParser()
        process.add_args(parser)
        args = parser.parse_args(["--process", "normalize"])
        with pytest.raises(HeliconError, match="EMAN2"):
            process.handle(pd.DataFrame(), args, {"process": 0}, args.process[0])

    def test_nothing_happens_without_it(self):
        from helicon.plugins.images2star import process

        data = pd.DataFrame({"a": [1]})
        out, index_d = process.handle(data, None, {"process": 0}, None)
        assert out is data and index_d == {"process": 0}


class TestGalleryOpensCompressedFiles:
    def test_a_gzipped_stack_is_read_whole(self, tmp_path):
        from helicon.lib.gui.gallery_backends import _open_mrc

        data = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
        plain = tmp_path / "c.mrcs"
        mrcfile.new(plain, data=data, overwrite=True).close()
        packed = tmp_path / "c.mrcs.gz"
        with open(plain, "rb") as src, gzip.open(packed, "wb") as dst:
            shutil.copyfileobj(src, dst)
        for path in (plain, packed):
            with _open_mrc(path) as mrc:
                assert np.array_equal(mrc.data[1], data[1])
