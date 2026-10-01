"""A 3D map from class averages at their ring angles."""

import numpy as np
import pytest

from helical_sim import Helix
from helicon.webApps.lib import helical_pitch_map as maps


def _classes(n=6, pixel=5.0):
    hx = Helix(pitch=1400.0, fold=2)
    az = np.arange(n) * (180.0 / n)
    images = [hx.render(0.0, a, 0.0, size=48, pixel=pixel) for a in az]
    return hx, az, images


class TestStraighten:
    def test_a_class_whose_axial_direction_points_left_is_turned_over(self):
        _, _, images = _classes(n=4)
        # an asymmetric picture: the same average twice, the second one's
        # axial direction pointing the other way
        img = images[1] * (np.linspace(0.3, 1.0, images[1].shape[1])[None, :])
        out = maps.straighten_classes([img, img], [0.0, 180.0], 5.0, 5.0)
        a, b = out
        assert a.shape == b.shape
        assert np.corrcoef(a.ravel(), b[::-1, ::-1].ravel())[0, 1] > 0.9
        assert np.corrcoef(a.ravel(), b.ravel())[0, 1] < 0.9


class TestReconstruct:
    def _prepared(self):
        _, az, images = _classes()
        return az, maps.straighten_classes(images, np.zeros(len(images)), 5.0, 5.0)

    def test_negative_twist_is_left_handed_and_the_pitch_uses_csym(self):
        az, prepared = self._prepared()
        left = maps.reconstruct_map(
            prepared, az * 2, 700.0, 2, 4.75, left_handed=True, apix=5.0
        )
        right = maps.reconstruct_map(
            prepared,
            az * 2,
            700.0,
            2,
            4.75,
            left_handed=False,
            apix=5.0,
            method="backprojection",
        )
        assert left["twist"] == pytest.approx(-360.0 * 4.75 / 1400.0)
        assert right["twist"] == pytest.approx(360.0 * 4.75 / 1400.0)
        assert left["volume"].ndim == 3 and np.isfinite(left["volume"]).all()
        # about 1.2 pitches of side projection, the tiles on the same canvas
        assert left["projection"].shape[1] == left["tiles"].shape[1]
        assert left["projection"].shape[1] * 5.0 == pytest.approx(
            1.2 * 1400.0, rel=0.05
        )

    def test_the_export_has_the_box_and_pixel_size_asked_for(self):
        az, prepared = self._prepared()
        m = maps.reconstruct_map(
            prepared,
            az * 2,
            700.0,
            2,
            4.75,
            apix=5.0,
            method="backprojection",
            helical_sym_order=9,
            output_box=64,
            output_apix=4.0,
        )
        assert m["volume_out"].shape == (64, 64, 64)
        assert m["apix_out"] == 4.0
        assert np.isfinite(m["volume_out"]).all()

    def test_helical_symmetry_order_and_c_symmetry_change_the_map(self):
        az, prepared = self._prepared()
        # not mirror symmetric across the axis, as real averages are not
        ramp = np.linspace(0.5, 1.0, prepared[0].shape[0])[:, None]
        prepared = [im * ramp for im in prepared]
        kw = dict(apix=5.0, method="backprojection")
        once = maps.reconstruct_map(
            prepared, az * 2, 700.0, 2, 4.75, helical_sym_order=1, **kw
        )
        many = maps.reconstruct_map(
            prepared, az * 2, 700.0, 2, 4.75, helical_sym_order=15, **kw
        )
        with_c = maps.reconstruct_map(
            prepared,
            az * 2,
            700.0,
            2,
            4.75,
            helical_sym_order=1,
            impose_csym=True,
            **kw,
        )
        assert not np.allclose(once["volume"], many["volume"])
        assert not np.allclose(once["volume"], with_c["volume"])
        # the twist and pitch do not depend on whether the symmetry is imposed
        assert once["twist"] == with_c["twist"] == many["twist"]

    def test_the_joint_fit_is_the_default(self):
        az, prepared = self._prepared()
        m = maps.reconstruct_map(prepared, az * 2, 700.0, 2, 4.75, apix=5.0)
        assert 0.0 < m["score"] <= 1.0 and m["volume"].ndim == 3
        assert m["volume"].shape[0] < 20  # a few rises: the symmetry does the rest
