"""Layer lines of a helix and the filter that keeps only them."""

import numpy as np
import pytest

from helicon.lib import layerlines as L
from helicon.webApps.lib import hill_compute as H


def _helix(twist=29.4, rise=21.92, csym=6, radius=77.6, ball=8.0, apix=2.3438, n=192):
    clean = H.simulate_helix(twist, rise, csym, radius, ball, n, n, apix, az0=17.0)
    return clean - clean.mean()


def _error(out, ref):
    out, ref = out - out.mean(), ref - ref.mean()
    return float(((out - ref) ** 2).mean() / (ref**2).mean())


class TestBesselOnsets:
    def test_the_signal_of_a_higher_order_starts_further_out(self):
        on = L.bessel_onsets(30)
        assert on[0] == 0.0
        assert np.all(np.diff(on) > 0)

    def test_the_onset_is_before_the_first_maximum(self):
        from scipy.special import jnp_zeros

        on = L.bessel_onsets(12)
        for n in (1, 5, 12):
            assert 0 < on[n] < jnp_zeros(n, 1)[0]

    def test_a_higher_floor_starts_later(self):
        assert L.bessel_onsets(10, 0.5)[8] > L.bessel_onsets(10, 0.05)[8]


class TestHeights:
    def test_the_lines_are_m_over_rise_plus_n_over_pitch(self):
        h, n = L.layer_line_heights(30.0, 20.0, 1, y_max=0.1)
        p = L.pitch(30.0, 20.0)
        m = np.round((h - n / p) * 20.0)
        assert np.allclose(h, m / 20.0 + n / p)

    def test_the_orders_are_multiples_of_the_symmetry(self):
        _, n = L.layer_line_heights(29.4, 21.92, 6, y_max=0.1)
        assert np.all(n % 6 == 0)

    def test_the_equator_is_there_with_both_signs_of_the_orders(self):
        h, n = L.layer_line_heights(29.4, 21.92, 6, y_max=0.1)
        assert np.any((n == 0) & (h == 0))
        assert np.any(n > 0) and np.any(n < 0)

    def test_a_radius_limits_the_orders_to_those_with_signal(self):
        _, all_n = L.layer_line_heights(1.2, 4.75, 1, y_max=0.2)
        _, few_n = L.layer_line_heights(1.2, 4.75, 1, y_max=0.2, radius=30.0, x_max=0.2)
        assert np.abs(few_n).max() < np.abs(all_n).max()
        # J_n(2 pi R X) is negligible below its onset: the highest order is the last
        # whose onset is within 2 pi R X, and the next one's is beyond it
        z_max = 2 * np.pi * 30.0 * 0.2
        top = int(np.abs(few_n).max())
        on = L.bessel_onsets(top + 3)
        assert on[top] <= z_max < on[top + 1]

    def test_no_twist_is_one_line_every_rise(self):
        assert L.pitch(0.0, 4.75) == 4.75


class TestMask:
    def test_the_mask_is_in_the_layout_of_rfft2_between_0_and_1(self):
        m = L.layer_line_mask((64, 48), 2.0, 29.4, 21.92, 6, 77.6)
        assert m.shape == (64, 25)
        assert m.min() >= 0.0 and m.max() <= 1.0

    def test_it_needs_the_radius(self):
        with pytest.raises(ValueError, match="radius"):
            L.layer_line_mask((64, 64), 2.0, 29.4, 21.92, 6)

    def test_nothing_is_kept_beyond_the_resolution_limit(self):
        m = L.layer_line_mask((128, 128), 2.0, 29.4, 21.92, 6, 77.6, cutoff_res=10.0)
        fy = np.fft.fftfreq(128, 2.0)[:, None]
        fx = np.fft.rfftfreq(128, 2.0)[None, :]
        assert np.all(m[np.hypot(fy, fx) > 0.11] == 0.0)

    def test_the_equator_is_kept_at_every_distance_from_the_axis(self):
        m = L.layer_line_mask((128, 128), 2.0, 29.4, 21.92, 6, 77.6)
        assert np.all(m[0, : len(m[0]) // 2] > 0.99)  # Y = 0, to well within Nyquist

    def test_a_wider_line_keeps_more(self):
        narrow = L.layer_line_mask((128, 128), 2.0, 29.4, 21.92, 6, 77.6, width=1.0)
        wide = L.layer_line_mask((128, 128), 2.0, 29.4, 21.92, 6, 77.6, width=5.0)
        assert wide.sum() > narrow.sum()


class TestMaskFloor:
    def test_a_lower_floor_keeps_each_line_from_closer_to_the_meridian(self):
        args = ((128, 128), 2.3438, 29.4, 21.92, 6, 77.6)
        low = L.layer_line_mask(*args, cutoff_res=12.0, floor=0.02)
        high = L.layer_line_mask(*args, cutoff_res=12.0, floor=0.4)
        assert low.sum() > high.sum()
        # nothing the higher floor keeps is lost by the lower one
        assert np.all(low >= high - 1e-9)

    def test_the_default_is_the_bessel_floor(self):
        args = ((128, 128), 2.3438, 29.4, 21.92, 6, 77.6)
        default = L.layer_line_mask(*args, cutoff_res=12.0)
        explicit = L.layer_line_mask(*args, cutoff_res=12.0, floor=L.BESSEL_FLOOR)
        assert np.array_equal(default, explicit)


class TestFilter:
    def test_the_clean_helix_is_kept(self):
        clean = _helix()
        out, kept = L.layer_line_filter(
            clean, 2.3438, 29.4, 21.92, 6, 77.6, width=3.0, wiener=False
        )
        assert _error(out, clean) < 0.1
        assert 0.05 < kept < 0.6

    def test_noise_is_removed(self):
        clean = _helix()
        noisy = clean + np.random.default_rng(0).normal(0, 3 * clean.std(), clean.shape)
        out, _ = L.layer_line_filter(
            noisy, 2.3438, 29.4, 21.92, 6, 77.6, cutoff_res=12.0, width=2.0
        )
        assert _error(out, clean) < _error(noisy, clean) / 20

    def test_what_is_kept_is_the_variance_of_white_noise_that_remains(self):
        noise = np.random.default_rng(1).normal(0, 1.0, (256, 256))
        out, kept = L.layer_line_filter(noise, 2.0, 29.4, 21.92, 6, 77.6, wiener=False)
        assert out.var() == pytest.approx(kept, rel=0.1)

    def test_the_axis_can_be_the_columns(self):
        clean = _helix()
        a, ka = L.layer_line_filter(clean, 2.3438, 29.4, 21.92, 6, 77.6)
        b, kb = L.layer_line_filter(clean.T, 2.3438, 29.4, 21.92, 6, 77.6, axis=1)
        assert np.allclose(a, b.T) and ka == kb

    def test_a_wrong_axis_is_refused(self):
        with pytest.raises(ValueError, match="axis"):
            L.layer_line_filter(np.zeros((32, 32)), 2.0, 29.4, 21.92, 6, 77.6, axis=2)

    def test_it_beats_the_best_isotropic_low_pass_on_an_ordered_helix(self):
        from scipy import ndimage as ndi

        clean = _helix()
        noisy = clean + np.random.default_rng(2).normal(0, 3 * clean.std(), clean.shape)
        low = min(
            _error(ndi.gaussian_filter(noisy, s), clean)
            for s in np.arange(0.5, 8, 0.25)
        )
        best = min(
            _error(
                L.layer_line_filter(
                    noisy, 2.3438, 29.4, 21.92, 6, 77.6, cutoff_res=c, width=w
                )[0],
                clean,
            )
            for c in (8.0, 12.0, 20.0)
            for w in (2.0, 3.0)
        )
        assert best < low


class TestWienerGain:
    def test_the_noise_power_is_read_from_between_the_lines(self):
        shape = (256, 256)
        noise = np.random.default_rng(3).normal(0, 1.0, shape)
        mask = L.layer_line_mask(shape, 2.0, 29.4, 21.92, 6, 77.6, width=3.0)
        power = np.abs(np.fft.rfft2(noise)) ** 2
        est = L.noise_power(power, mask)
        # white noise of unit variance has the mean power ny * nx at every pixel
        assert np.median(est[mask < 0.02]) == pytest.approx(256 * 256, rel=0.1)
        assert est.shape == power.shape

    def test_signal_on_the_lines_does_not_change_the_estimate(self):
        clean = _helix(n=256)
        noise = np.random.default_rng(4).normal(0, 1.0, clean.shape)
        mask = L.layer_line_mask(clean.shape, 2.3438, 29.4, 21.92, 6, 77.6, width=3.0)
        only = L.noise_power(np.abs(np.fft.rfft2(noise)) ** 2, mask)
        both = L.noise_power(
            np.abs(np.fft.rfft2(noise + 5 * clean / clean.std())) ** 2, mask
        )
        assert np.median(both[mask < 0.02]) == pytest.approx(
            np.median(only[mask < 0.02]), rel=0.15
        )

    def test_it_never_amplifies(self):
        clean = _helix()
        noisy = clean + np.random.default_rng(5).normal(0, 2 * clean.std(), clean.shape)
        args = (2.3438, 29.4, 21.92, 6, 77.6)
        masked, _ = L.layer_line_filter(noisy, *args, wiener=False)
        gained, _ = L.layer_line_filter(noisy, *args, wiener=True)
        assert np.all(
            np.abs(np.fft.rfft2(gained))
            <= np.abs(np.fft.rfft2(masked)) * (1 + 1e-9) + 1e-9
        )

    def test_a_higher_strength_removes_more_of_pure_noise(self):
        noise = np.random.default_rng(6).normal(0, 1.0, (192, 192))
        var = [
            L.layer_line_filter(noise, 2.3438, 29.4, 21.92, 6, 77.6, strength=s)[
                0
            ].var()
            for s in (0.5, 1.0, 2.0, 4.0)
        ]
        assert var == sorted(var, reverse=True)

    def test_it_is_better_than_the_bare_mask_without_a_resolution_limit(self):
        clean = _helix()
        noisy = clean + np.random.default_rng(7).normal(0, 3 * clean.std(), clean.shape)
        args = (2.3438, 29.4, 21.92, 6, 77.6)
        bare, _ = L.layer_line_filter(noisy, *args, wiener=False)
        gained, _ = L.layer_line_filter(noisy, *args, wiener=True)
        assert _error(gained, clean) < _error(bare, clean) / 2

    def test_the_default_keeps_a_clean_helix(self):
        clean = _helix()
        out, _ = L.layer_line_filter(clean, 2.3438, 29.4, 21.92, 6, 77.6)
        assert _error(out, clean) < 0.15


class TestSignalPower:
    def test_the_highest_frequencies_do_not_leak_into_the_zero_frequency(self):
        power = np.zeros((16, 9))
        power[:, -1] = 1000.0  # the Nyquist column, all noise
        signal = L.signal_power(power, np.zeros_like(power))
        assert np.all(signal[:, :3] == 0)  # nothing wrapped round to column 0 to 2
        assert signal[:, -1].max() > 0

    def test_the_rows_are_periodic(self):
        power = np.zeros((16, 9))
        power[-1, 4] = 90.0  # the last row is next to the first
        signal = L.signal_power(power, np.zeros_like(power), smooth=(3, 1))
        assert signal[0, 4] == pytest.approx(30.0)
        assert signal[1, 4] == 0.0

    def test_the_noise_is_taken_out_and_nothing_goes_below_zero(self):
        power = np.full((8, 5), 10.0)
        signal = L.signal_power(power, np.full_like(power, 4.0))
        assert np.allclose(signal, 6.0)
        assert L.signal_power(power, np.full_like(power, 40.0)).max() == 0.0

    def test_the_filter_uses_it(self):
        import inspect

        assert "signal_power(" in inspect.getsource(L.layer_line_filter)
        assert 'mode="wrap"' not in inspect.getsource(L)
