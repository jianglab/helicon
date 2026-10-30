"""Automatic denoising of micrographs: monogenic curvelets plus wavelets.

Two transforms see different things in a micrograph: curvelets are directional
and capture lines such as filaments, undecimated wavelets capture spots and
blobs. Each is shrunk with BayesShrink thresholds, set for
every sub-band from the image's own white-noise level, and the two results are
averaged. Nothing has to be tuned by hand; ``strength`` scales all thresholds.

Measured on 9 EMPIAR-10640 micrographs with independent even/odd frame sums
(denoise the even sum, correlate with the odd one), the average beats the best
Gaussian low-pass on every micrograph, and each transform alone.

Both transforms are built in Fourier space and share one shrinkage loop.

Curvelet windows W(xi) = radial(|xi|) *
angular(theta) tile the plane with sum W^2 = 1 (plus a low-pass), so the image
comes back exactly when nothing is thresholded. Every window is even, so each
sub-band of a real image is real and its Riesz transforms are exact on the same
grid: the monogenic amplitude is sqrt(q0^2 + q1^2 + q2^2) of three real images,
and soft-thresholding that amplitude keeps the local phase (where the edges
are). Sub-bands are full size and handled one at a time.
"""

from __future__ import annotations

import numpy as np

DEFAULT_STRENGTH = 2.0
# the wavelet's thresholds relative to the curvelet's: the pairing measured best
WAVELET_FACTOR = 2.0


def white_noise_sd(image: np.ndarray) -> float:
    """Per-pixel standard deviation of white noise, from the outer frequency ring.

    Detector shot noise is white, and between 0.45 and 0.5 cycles/pixel there is
    little else; what signal or fixed detector pattern remains there makes this
    a slight overestimate (6-19% in variance on the test micrographs).

    Parameters
    ----------
    image : numpy.ndarray
        2D image.

    Returns
    -------
    float
        Noise standard deviation per pixel.
    """
    x = np.asarray(image, np.float64)
    f = np.abs(np.fft.rfft2(x - x.mean())) ** 2 / x.size
    ny, nx = x.shape
    r = np.hypot(np.fft.rfftfreq(nx)[None, :], np.fft.fftfreq(ny)[:, None])
    return float(np.sqrt(f[(r >= 0.45) & (r < 0.5)].mean()))


def _nu(t):
    t = np.clip(t, 0.0, 1.0)
    return t**4 * (35 - 84 * t + 70 * t**2 - 20 * t**3)


def _window(t):
    """Smooth window on [-1, 1] with w(t)^2 + w(t-1)^2 = 1 for t in [0, 1]."""
    return np.where(np.abs(t) < 1, np.cos(np.pi / 2 * _nu(np.abs(t))), 0.0)


def _n_angles(band: int, finest: int) -> int:
    """Angles of a band (0 = finest), halved every second band going coarser."""
    return max(4, finest >> (band // 2))


def curvelet_windows(shape, n_bands: int, finest_angles: int = 16):
    """The low-pass and the curvelet windows on the half plane of ``rfft2``.

    Parameters
    ----------
    shape : tuple of int
        Image shape.
    n_bands : int
        Number of curvelet bands; the finest is centred on 0.5 cycles/pixel and
        each next one an octave lower.
    finest_angles : int
        Angles of the finest band (over 180 degrees).

    Returns
    -------
    low2 : numpy.ndarray
        Squared low-pass window.
    windows : generator
        ``(band, window)`` for every sub-band.
    """
    ny, nx = shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(nx)[None, :]
    r = np.hypot(fx, fy)
    with np.errstate(divide="ignore"):
        s = np.log2(r / 0.5)
    radial = []
    for j in range(n_bands):
        b = _window(s + j)
        radial.append(np.where(s >= 0, 1.0, b) if j == 0 else b)  # 0: the corners too
    low2 = np.where(
        s < -(n_bands - 1), np.clip(1 - sum(b**2 for b in radial), 0, 1), 0.0
    )
    low2[0, 0] = 1.0
    theta = np.mod(np.arctan2(fy, fx), np.pi)

    def windows():
        for j, b in enumerate(radial):
            n = _n_angles(j, finest_angles)
            step = np.pi / n
            for k in range(n):
                d = (theta - k * step) / step
                yield j, b * _window((d + n / 2) % n - n / 2)

    return low2, windows()


# sym4 analysis filters (as PyWavelets defines them)
_SYM4_LO = np.array([-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161,
                     0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427])  # fmt: skip
_SYM4_HI = np.array([-0.0322231006040427, -0.012603967262037833, 0.09921954357684722, 0.29785779560527736,
                     -0.8037387518059161, 0.49761866763201545, 0.02963552764599851, -0.07576571478927333])  # fmt: skip


def wavelet_windows(shape, levels: int):
    """The undecimated sym4 wavelet as windows on the half plane of ``rfft2``.

    Level j filters with H(2^j w)/sqrt(2) and G(2^j w)/sqrt(2), so the squared
    windows of the three detail sub-bands of every level and the final
    approximation add up to one: the image comes back exactly.

    Parameters
    ----------
    shape : tuple of int
        Image shape.
    levels : int
        Number of levels.

    Returns
    -------
    low2 : numpy.ndarray
        Squared window of the approximation that is kept.
    windows : generator
        ``(level, window)`` for every detail sub-band (complex windows).
    """
    ny, nx = shape
    wy = 2 * np.pi * np.fft.fftfreq(ny)[:, None]
    wx = 2 * np.pi * np.fft.rfftfreq(nx)[None, :]

    def response(taps, w, j):
        n = np.arange(len(taps))
        return (
            taps[:, None, None] * np.exp(-1j * (2**j) * w[None] * n[:, None, None])
        ).sum(0) / np.sqrt(2)

    approx = np.ones((ny, nx // 2 + 1), complex)
    filters = []
    for j in range(levels):
        hy, gy = response(_SYM4_LO, wy, j), response(_SYM4_HI, wy, j)
        hx, gx = response(_SYM4_LO, wx, j), response(_SYM4_HI, wx, j)
        filters.append([approx * hy * gx, approx * gy * hx, approx * gy * gx])
        approx = approx * hy * hx

    def windows():
        for j, subs in enumerate(filters):
            for w in subs:
                yield j, w

    return np.abs(approx) ** 2, windows()


def _pad(x, margin):
    """Mirror-pad, so that the periodic FFT does not wrap one edge into the other."""
    return np.pad(x, margin, mode="reflect") if margin else x


def _shrink(x, tiling, strength, noise_sd, monogenic, workers):
    """Soft-threshold every sub-band of a Fourier tiling with BayesShrink thresholds.

    Each sub-band q0 = IFFT(F W) gets ``strength * s_n^2 / s_x``: s_n its noise
    (the white-noise level carried through W), s_x the signal left after the
    noise. With ``monogenic`` the threshold acts on the monogenic amplitude
    sqrt(q0^2 + q1^2 + q2^2) (q1, q2 its Riesz transforms), which keeps the
    local phase; otherwise on |q0|.
    """
    import scipy.fft as sf

    margin = min(64, min(x.shape) // 8)
    xp = _pad(x, margin)
    shape = xp.shape
    ny, nx = shape
    low2, windows = tiling(shape)
    F = sf.rfft2(xp, workers=workers)
    out = F * low2
    if monogenic:
        fy = np.fft.fftfreq(ny)[:, None]
        fx = np.fft.rfftfreq(nx)[None, :]
        r = np.hypot(fx, fy)
        r[0, 0] = 1.0
        rx, ry = -1j * fx / r, -1j * fy / r
    # |W|^2 summed over the full plane, from the half plane that rfft2 keeps
    cw = np.full(F.shape[1], 2.0)
    cw[0] = 1.0
    if nx % 2 == 0:
        cw[-1] = 1.0
    for _level, w in windows:
        Fw = F * w
        q0 = sf.irfft2(Fw, s=shape, workers=workers)
        if monogenic:
            q1 = sf.irfft2(Fw * rx, s=shape, workers=workers)
            q2 = sf.irfft2(Fw * ry, s=shape, workers=workers)
            amp = np.sqrt(q0**2 + q1**2 + q2**2)
        else:
            amp = np.abs(q0)
        s_n = noise_sd * np.sqrt(float((np.abs(w) ** 2 * cw).sum()) / (ny * nx))
        s_x = np.sqrt(max(q0.var() - s_n**2, 1e-12 * s_n**2))
        t = strength * s_n**2 / s_x
        att = np.divide(
            np.maximum(amp - t, 0), amp, out=np.zeros_like(amp), where=amp > 0
        )
        out += sf.rfft2(q0 * att, workers=workers) * np.conj(w)
    result = sf.irfft2(out, s=shape, workers=workers)
    return (
        result[margin : margin + x.shape[0], margin : margin + x.shape[1]]
        if margin
        else result
    )


def _default_levels(shape):
    """Down to periods of about 1/16 of the image."""
    return max(2, int(np.log2(min(shape) / 16)))


def curvelet_denoise(
    image: np.ndarray,
    strength: float = DEFAULT_STRENGTH,
    noise_sd: float | None = None,
    n_bands: int | None = None,
    finest_angles: int = 16,
    workers: int | None = None,
) -> np.ndarray:
    """Monogenic curvelet denoising with BayesShrink thresholds.

    Parameters
    ----------
    image : numpy.ndarray
        2D image.
    strength : float
        Multiplier on every threshold; 1 is the textbook BayesShrink.
    noise_sd : float, optional
        Per-pixel white-noise sd; by default :func:`white_noise_sd`.
    n_bands : int, optional
        Number of bands; by default down to periods of 1/16 of the image.
    finest_angles : int
        Angles of the finest band.
    workers : int, optional
        FFT threads; by default :func:`helicon.available_cpu`.

    Returns
    -------
    numpy.ndarray
        The denoised image.
    """
    import helicon

    x = np.asarray(image, np.float64)
    noise_sd = white_noise_sd(x) if noise_sd is None else noise_sd
    n_bands = n_bands or _default_levels(x.shape)
    tiling = lambda shape: curvelet_windows(shape, n_bands, finest_angles)  # noqa: E731
    return _shrink(
        x, tiling, strength, noise_sd, True, workers or helicon.available_cpu()
    )


def wavelet_denoise(
    image: np.ndarray,
    strength: float = DEFAULT_STRENGTH * WAVELET_FACTOR,
    noise_sd: float | None = None,
    levels: int | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """Undecimated sym4 wavelet denoising with BayesShrink thresholds.

    Parameters
    ----------
    image : numpy.ndarray
        2D image.
    strength : float
        Multiplier on every threshold; 1 is the textbook BayesShrink.
    noise_sd : float, optional
        Per-pixel white-noise sd; by default :func:`white_noise_sd`.
    levels : int, optional
        Number of levels; by default down to periods of 1/16 of the image.
    workers : int, optional
        FFT threads; by default :func:`helicon.available_cpu`.

    Returns
    -------
    numpy.ndarray
        The denoised image.
    """
    import helicon

    x = np.asarray(image, np.float64)
    noise_sd = white_noise_sd(x) if noise_sd is None else noise_sd
    levels = levels or min(6, _default_levels(x.shape))
    tiling = lambda shape: wavelet_windows(shape, levels)  # noqa: E731
    return _shrink(
        x, tiling, strength, noise_sd, False, workers or helicon.available_cpu()
    )


def denoise_micrograph(
    image: np.ndarray,
    strength: float = DEFAULT_STRENGTH,
    noise_sd: float | None = None,
    workers: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Denoise a micrograph: the average of the curvelet and wavelet results.

    Parameters
    ----------
    image : numpy.ndarray
        2D image.
    strength : float
        Multiplier on every threshold. 2 (the default) gives a clean view for
        looking at and picking filaments; 1 keeps more fine texture and is what
        best predicts an independent exposure of the same area.
    noise_sd : float, optional
        Per-pixel white-noise sd; by default :func:`white_noise_sd`.
    workers : int, optional
        FFT threads; by default :func:`helicon.available_cpu`.

    Returns
    -------
    denoised : numpy.ndarray
        The denoised image, float32.
    info : dict
        What was chosen and from what: ``noise_sd``, ``noise_from``, ``strength``.
    """
    x = np.asarray(image, np.float64)
    noise_from = "given"
    if noise_sd is None:
        noise_sd = white_noise_sd(x)
        noise_from = "the outer frequency ring"
    c = curvelet_denoise(x, strength=strength, noise_sd=noise_sd, workers=workers)
    s = wavelet_denoise(
        x, strength=strength * WAVELET_FACTOR, noise_sd=noise_sd, workers=workers
    )
    info = dict(noise_sd=noise_sd, noise_from=noise_from, strength=strength)
    return (0.5 * (c + s)).astype(np.float32), info
