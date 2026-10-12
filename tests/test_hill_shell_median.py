"""Scaling a power spectrum by the median of each Fourier shell (HILL)."""

import numpy as np
import pytest

from helicon.webApps.lib import hill_compute as hill

CUTOFF = (4.0, 6.0)  # (res_y, res_x): the pixels are not square in frequency


def _grid(ny=128, nx=64):
    sy = (np.arange(ny) - ny // 2) / (ny // 2) / CUTOFF[0]
    sx = (np.arange(nx) - nx // 2) / (nx // 2) / CUTOFF[1]
    return np.hypot(sy[:, None], sx[None, :])


def _spectrum(line_strength=20.0, seed=0, ny=128, nx=64):
    """A helical-looking spectrum with a known answer: noise under an envelope
    that falls by two orders of magnitude, and layer lines of a known strength
    (as a multiple of the noise) on every 8th row."""
    rng = np.random.default_rng(seed)
    r = _grid(ny, nx)
    envelope = np.exp(-r / r.max() * np.log(100.0))
    noise = rng.rayleigh(1.0, size=r.shape)
    lines = np.zeros_like(r)
    lines[::8, :] = line_strength
    return envelope * (noise + lines * (rng.random(r.shape) < 0.5)), lines > 0


class TestFourierShells:
    def test_the_origin_is_shell_zero(self):
        shells = hill.fourier_shells((128, 64), CUTOFF)
        assert shells[64, 32] == 0

    def test_a_shell_is_a_circle_of_frequency_not_of_pixels(self):
        shells = hill.fourier_shells((128, 64), CUTOFF)
        # a step of k pixels along y is k * dsy; along x it is k * dsx, and the
        # two pixel sizes differ: 1/(64*res_y) against 1/(32*res_x)
        ds_y, ds_x = 1 / (64 * CUTOFF[0]), 1 / (32 * CUTOFF[1])
        k_y = 20
        k_x = round(k_y * ds_y / ds_x)
        assert shells[64 + k_y, 32] == shells[64, 32 + k_x]
        assert shells[64 + k_y, 32] != shells[64, 32 + k_y]  # same pixels, other shell

    def test_it_is_symmetric_about_the_origin(self):
        shells = hill.fourier_shells((128, 64), CUTOFF)
        assert np.array_equal(shells[1:, 1:], shells[1:, 1:][::-1, ::-1])

    def test_it_is_cached_and_not_writable(self):
        a = hill.fourier_shells((128, 64), CUTOFF)
        assert hill.fourier_shells((128, 64), CUTOFF) is a
        assert not a.flags.writeable


class TestShellMedianRatio:
    def test_the_median_of_every_shell_comes_out_1(self):
        amp, _ = _spectrum()
        ratio = hill.shell_median_ratio(amp, CUTOFF)
        shells = hill.fourier_shells(amp.shape, CUTOFF)
        for s in np.unique(shells):
            # every shell, the few-pixel ones near the origin too, and whether it
            # holds an even number of pixels or an odd one
            assert np.median(ratio[shells == s]) == pytest.approx(1.0), s

    def test_an_even_shell_is_divided_by_the_mean_of_its_two_middle_values(self):
        shells = hill.fourier_shells((8, 8), (4.0, 4.0))
        amp = np.ones((8, 8))
        members = np.flatnonzero(shells.ravel() == 1)
        assert len(members) % 2 == 0, "a shell of an even size is needed"
        # values 1, 2, ..., n in the shell: the median is the mean of the middle two
        amp.ravel()[members] = np.arange(1, len(members) + 1)
        middle = (len(members) // 2, len(members) // 2 + 1)  # 1-based
        expected = (middle[0] + middle[1]) / 2
        ratio = hill.shell_median_ratio(amp, (4.0, 4.0))
        assert ratio.ravel()[members][-1] == pytest.approx(len(members) / expected)

    def test_it_agrees_with_numpys_median_of_each_shell(self):
        amp, _ = _spectrum(seed=3)
        shells = hill.fourier_shells(amp.shape, CUTOFF)
        ratio = hill.shell_median_ratio(amp, CUTOFF)
        for s in np.unique(shells)[::5]:
            assert np.allclose(
                ratio[shells == s], amp[shells == s] / np.median(amp[shells == s])
            )

    def test_any_scale_per_shell_divides_out(self):
        amp, _ = _spectrum()
        shells = hill.fourier_shells(amp.shape, CUTOFF)
        scale = 10.0 ** (-shells / shells.max() * 3)  # three orders of magnitude
        assert np.allclose(
            hill.shell_median_ratio(amp * scale, CUTOFF),
            hill.shell_median_ratio(amp, CUTOFF),
        )

    def test_a_layer_line_stands_out_by_its_strength_over_the_background(self):
        amp, lines = _spectrum(line_strength=20.0)
        ratio = hill.shell_median_ratio(amp, CUTOFF)
        assert np.median(ratio[~lines]) == pytest.approx(1.0, abs=0.1)
        # the lines, 1/8 of the rows, do not drag the median up
        assert np.median(ratio[lines]) > 5

    def test_the_sign_does_not_matter(self):
        # cubic resampling leaves small negative values
        amp, _ = _spectrum()
        assert np.array_equal(
            hill.shell_median_ratio(-amp, CUTOFF), hill.shell_median_ratio(amp, CUTOFF)
        )

    def test_an_empty_shell_comes_out_0_not_nan(self):
        amp, _ = _spectrum()
        shells = hill.fourier_shells(amp.shape, CUTOFF)
        amp = amp.copy()
        amp[shells >= shells.max() - 2] = 0
        ratio = hill.shell_median_ratio(amp, CUTOFF)
        assert np.isfinite(ratio).all()
        assert (ratio[shells >= shells.max() - 2] == 0).all()


class TestScalingMakesLayerLinesEquallyVisibleAtEveryResolution:
    """The purpose of the mode, on a spectrum whose answer is known."""

    def _contrast_by_band(self, display):
        amp, lines = _spectrum(line_strength=20.0)
        r = _grid(*amp.shape)
        out = []
        for lo, hi in ((0.1, 0.3), (0.7, 0.9)):  # near and far from the origin
            band = (r >= lo * r.max()) & (r < hi * r.max())
            out.append(display[band & lines].mean() - display[band & ~lines].mean())
        return out

    def test_the_median_scaled_contrast_hardly_depends_on_resolution(self):
        amp, _ = _spectrum()
        near, far = self._contrast_by_band(
            hill.normalize_shell_scaled(hill.shell_median_ratio(amp, CUTOFF))
        )
        assert 0.7 < far / near < 1.4

    def test_plain_log_loses_the_far_layer_lines(self):
        amp, _ = _spectrum()
        near, far = self._contrast_by_band(np.log1p(amp) / np.log1p(amp).max())
        assert far / near < 0.5  # what the mode is for


class TestDisplayScaling:
    def test_it_is_0_to_1_and_the_top_is_clipped(self):
        ratio = 1 + 0.1 * np.random.default_rng(0).random((64, 64))
        ratio[0, 0] = 1e6  # one pixel hundreds of times above its shell
        out = hill.normalize_shell_scaled(ratio)
        assert out.min() >= 0 and out.max() == 1
        # the one outlier does not set the scale for all the rest
        assert out.mean() > 0.3 and (out > 0.9).sum() > 5

    def test_a_flat_spectrum_is_zero(self):
        assert not hill.normalize_shell_scaled(np.ones((8, 8))).any()


class TestThroughThePowerSpectrumFunctions:
    def _image(self):
        rng = np.random.default_rng(1)
        img = rng.normal(size=(64, 64))
        img[:, 28:36] += 3.0  # a filament
        return img

    def test_the_single_image_path_ignores_the_log_and_the_filters(self):
        a, _ = hill.compute_power_spectra(
            self._image(),
            apix=2.0,
            cutoff_res=(4.0, 6.0),
            output_size=(64, 32),
            shell_median=True,
            log=True,
            low_pass_fraction=0.2,
            high_pass_fraction=0.1,
        )
        b, phase = hill.compute_power_spectra(
            self._image(),
            apix=2.0,
            cutoff_res=(4.0, 6.0),
            output_size=(64, 32),
            shell_median=True,
            log=False,
        )
        assert np.array_equal(a, b)
        assert a.dtype == np.float32 and phase.dtype == np.float32
        assert a.min() >= 0 and a.max() <= 1

    def test_without_it_nothing_changes(self):
        kwargs = dict(apix=2.0, cutoff_res=(4.0, 6.0), output_size=(64, 32), log=True)
        a, _ = hill.compute_power_spectra(self._image(), **kwargs)
        b, _ = hill.compute_power_spectra(self._image(), shell_median=False, **kwargs)
        assert np.array_equal(a, b)

    def test_the_resampling_path_divides_by_the_shell_median_too(self):
        rng = np.random.default_rng(2)
        data = np.abs(rng.normal(size=(64, 64))) + 0.1
        resize = hill.resize_rescale_power_spectra.__wrapped__
        kwargs = dict(
            nyquist_res=4.0, cutoff_res=(4.0, 6.0), output_size=(64, 32), norm=1
        )
        out = resize(
            data, shell_median=True, log=True, high_pass_fraction=0.3, **kwargs
        )
        same = resize(data, shell_median=True, log=False, **kwargs)
        assert np.array_equal(out, same)  # log and filters are not used
        assert out.min() >= 0 and out.max() <= 1
        plain = resize(data, shell_median=False, log=True, **kwargs)
        assert not np.allclose(out, plain)
