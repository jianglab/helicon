import numpy as np
import pytest

from helicon.lib import denoise


def filaments(shape=(192, 208), noise_sd=1.0, seed=0):
    """Straight filaments with a repeat along them, plus white noise.

    Returns
    -------
    clean, noisy : numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]].astype(float)
    clean = np.zeros(shape)
    for _ in range(4):
        angle = rng.uniform(0, np.pi)
        c, s = np.cos(angle), np.sin(angle)
        y0, x0 = rng.uniform(0.2, 0.8) * shape[0], rng.uniform(0.2, 0.8) * shape[1]
        across = (x - x0) * s - (y - y0) * c
        along = (x - x0) * c + (y - y0) * s
        clean += np.exp(-0.5 * (across / 3.0) ** 2) * (1 + 0.5 * np.cos(along / 2.5))
    return clean, clean + rng.normal(scale=noise_sd, size=shape)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestWindows:
    def test_curvelet_windows_and_low_pass_add_up_to_one(self):
        low2, windows = denoise.curvelet_windows((70, 83), n_bands=3)
        total = low2 + sum(w**2 for _, w in windows)
        assert np.abs(total - 1).max() < 1e-12

    def test_wavelet_windows_and_approximation_add_up_to_one(self):
        low2, windows = denoise.wavelet_windows((70, 83), levels=3)
        total = low2 + sum(np.abs(w) ** 2 for _, w in windows)
        assert np.abs(total - 1).max() < 1e-10


class TestReconstruction:
    @pytest.mark.parametrize(
        "method", [denoise.curvelet_denoise, denoise.wavelet_denoise]
    )
    def test_nothing_thresholded_gives_the_image_back(self, method):
        image = np.random.default_rng(1).normal(size=(70, 83)).cumsum(0)
        back = method(image, strength=0.0, workers=1)
        assert back.shape == image.shape
        assert np.abs(back - image).max() < 1e-9 * np.abs(image).max()


class TestNoise:
    def test_white_noise_level_is_measured(self):
        noise = np.random.default_rng(2).normal(scale=3.0, size=(256, 256))
        assert denoise.white_noise_sd(noise) == pytest.approx(3.0, rel=0.03)

    def test_a_smooth_signal_does_not_raise_the_noise_level(self):
        clean, noisy = filaments(noise_sd=1.0)
        assert denoise.white_noise_sd(noisy) == pytest.approx(1.0, rel=0.05)


class TestDenoise:
    def test_filaments_come_closer_to_the_truth(self):
        clean, noisy = filaments(noise_sd=1.0)
        out, info = denoise.denoise_micrograph(noisy, workers=1)
        assert out.dtype == np.float32 and out.shape == noisy.shape
        assert rmse(out, clean) < 0.35 * rmse(noisy, clean)
        assert info["strength"] == denoise.DEFAULT_STRENGTH
        assert info["noise_from"] == "the outer frequency ring"
        assert info["noise_sd"] == pytest.approx(1.0, rel=0.05)

    def test_a_given_noise_level_is_used_and_reported(self):
        _, noisy = filaments(noise_sd=1.0)
        weak, info = denoise.denoise_micrograph(noisy, noise_sd=0.1, workers=1)
        strong, _ = denoise.denoise_micrograph(noisy, noise_sd=2.0, workers=1)
        assert info["noise_from"] == "given" and info["noise_sd"] == 0.1
        # a lower noise level removes less
        assert rmse(weak, noisy) < rmse(strong, noisy)

    def test_pure_noise_is_mostly_removed(self):
        noise = np.random.default_rng(3).normal(size=(128, 128))
        out, _ = denoise.denoise_micrograph(noise, workers=1)
        assert out.std() < 0.2
