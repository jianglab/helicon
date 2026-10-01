"""The Gaussian-basis joint fit builds its design matrices in parallel.

They are the bulk of its time (84 of 93 s for 26 class averages, run one after
another), and each image's is independent of the others'. The sums over the
images are taken in image order, so the fit must not depend on the threads.
"""

import numpy as np
import pytest

from helicon.webApps.lib import solver_gauss_analytic as gauss


def _images(n=4, ny=40, nx=80):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:ny, :nx]
    return [
        np.exp(-((yy - ny / 2 - 6 * np.sin(2 * np.pi * (xx + 10 * k) / 40)) ** 2) / 8)
        + 0.05 * rng.standard_normal((ny, nx))
        for k in range(n)
    ]


_KW = dict(
    phis=[0, 90, 180, 270],
    scale2d_to_3d=1.0,
    twist_degree=-9.0,
    rise_pixel=1.0,
    csym=1,
    reconstruct_diameter_2d_pixel=40,
    reconstruct_diameter_3d_pixel=40,
    reconstruct_length_3d_pixel=8,
    target_apix2d=5.0,
)


class TestParallelDesignMatrices:
    def test_threads_do_not_change_the_fit(self):
        images = _images()
        one, info_one = gauss.gauss_joint_reconstruct(images, cpu=1, **_KW)
        four, info_four = gauss.gauss_joint_reconstruct(images, cpu=4, **_KW)
        np.testing.assert_allclose(four, one, rtol=1e-6, atol=1e-9)
        assert info_four["score"] == pytest.approx(info_one["score"], abs=1e-9)
        assert info_four["per_image"] == pytest.approx(info_one["per_image"])

    def test_workers_are_capped_by_images_and_memory(self, monkeypatch):
        import helicon

        monkeypatch.setattr(helicon, "available_cpu", lambda mem_gb_per_cpu=None: 3)
        assert gauss._workers(16, 26) == 3  # memory allows 3
        assert gauss._workers(16, 2) == 2  # only 2 images
        assert gauss._workers(1, 26) == 1
        assert gauss._workers(None, 26) == 1

    def test_the_voxel_joint_solver_passes_its_cpu_on(self, monkeypatch):
        from helicon.webApps.lib import denovo3d_jointsolve

        seen = {}

        def fake(images, phis, cpu=1, **kw):
            seen["cpu"] = cpu
            return np.zeros((1, 1, 1)), {}

        monkeypatch.setattr(gauss, "gauss_joint_reconstruct", fake)
        denovo3d_jointsolve.joint_reconstruct(
            _images(1), [0.0], 1.0, -9.0, 1.0, algorithm=dict(model="gauss"), cpu=7
        )
        assert seen["cpu"] == 7
