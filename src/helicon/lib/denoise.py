"""Automatic denoising of micrographs and maps: monogenic curvelets plus wavelets.

Two transforms see different things: curvelets are directional and capture
lines and sheets such as filaments, undecimated wavelets capture spots and
blobs. Each is shrunk with BayesShrink thresholds, set for every sub-band from
the noise, and the two results are averaged. Nothing has to be tuned by hand;
``strength`` scales all thresholds.

Measured on 9 EMPIAR-10640 micrographs with independent even/odd frame sums
(denoise the even sum, correlate with the odd one), the average beats the best
Gaussian low-pass on every micrograph, and each transform alone. On the half
maps of EMDB entries 0259, 10650, 13708 and 10940 (denoise half 1, compare with
half 2) it leaves less noise than a filter by the true half-map FSC.

Both transforms are built in Fourier space and share one shrinkage loop.

Curvelet windows W(xi) = radial(|xi|) * angular(direction) tile the frequency
space with sum W^2 = 1 (plus a low-pass), so the image comes back exactly when
nothing is thresholded. Every window is even, so each sub-band of a real image
is real and its Riesz transforms are exact on the same grid: the monogenic
amplitude is sqrt(q0^2 + q1^2 + ...) of real images, and soft-thresholding that
amplitude keeps the local phase (where the edges are). Sub-bands are full size
and handled one at a time.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

DEFAULT_STRENGTH = 2.0
# best on the half maps of 5 amyloid entries (EMPIAR-10940/10943/10944); others
# wanted 2-6, which tune_strength finds when the half maps are at hand
MAP_STRENGTH = 1.5
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


def _freqs(shape):
    """Frequencies of every axis on the half space of ``rfftn``, ready to broadcast."""
    f = [np.fft.fftfreq(n) for n in shape[:-1]] + [np.fft.rfftfreq(shape[-1])]
    nd = len(shape)
    return [
        v.reshape([-1 if i == k else 1 for i in range(nd)]) for k, v in enumerate(f)
    ]


def _window(t):
    """Smooth window on [-1, 1] with w(t)^2 + w(t-1)^2 = 1 for t in [0, 1]."""
    # cos(pi/2 nu(|t|)), nu = t^4 (35 - 84t + 70t^2 - 20t^3), in place: 3x faster
    a = np.minimum(np.abs(t), 1.0)
    w = a * -20.0
    w += 70
    w *= a
    w -= 84
    w *= a
    w += 35
    a *= a
    a *= a
    w *= a
    w *= np.pi / 2
    np.cos(w, out=w)
    w[a >= 1] = 0.0
    return w


def _n_angles(band: int, finest: int) -> int:
    """Angles of a band (0 = finest), halved every second band going coarser."""
    return max(4, finest >> (band // 2))


def _n_directions(band: int, finest: int) -> int:
    """Directions of a 3D band (0 = finest), halved every second band going coarser."""
    return max(3, finest >> (band // 2))


def _hemisphere(n):
    """``n`` nearly uniform unit directions (z, y, x) on the half sphere z > 0."""
    i = np.arange(n) + 0.5
    z = 1 - i / n
    phi = np.pi * (1 + 5**0.5) * i
    rho = np.sqrt(1 - z * z)
    return np.stack([z, rho * np.sin(phi), rho * np.cos(phi)], 1)


def curvelet_windows(shape, n_bands: int, finest_angles: int | None = None):
    """The low-pass and the curvelet windows on the half space of ``rfftn``.

    Parameters
    ----------
    shape : tuple of int
        Image (2D) or map (3D) shape.
    n_bands : int
        Number of curvelet bands; the finest is centred on 0.5 cycles/pixel and
        each next one an octave lower.
    finest_angles : int, optional
        Directions of the finest band: angles over 180 degrees in 2D (default
        16), directions over the half sphere in 3D (default 12).

    Returns
    -------
    low2 : numpy.ndarray
        Squared low-pass window.
    windows : generator
        ``(band, window)`` for every sub-band.
    """
    fs = _freqs(shape)
    r = np.sqrt(sum(f**2 for f in fs))
    with np.errstate(divide="ignore"):
        s = np.log2(r / 0.5)

    def radial(j):
        b = _window(s + j)
        return np.where(s >= 0, 1.0, b) if j == 0 else b  # 0: the corners too

    # below the coarsest band's centre only that band overlaps the low-pass
    low2 = np.where(
        s < -(n_bands - 1), np.clip(1 - radial(n_bands - 1) ** 2, 0, 1), 0.0
    )
    low2[(0,) * len(shape)] = 1.0

    if len(shape) == 2:
        fy, fx = fs
        theta = np.mod(np.arctan2(fy, fx), np.pi)
        finest = finest_angles or 16

        def windows():
            for j in range(n_bands):
                b = radial(j)
                n = _n_angles(j, finest)
                step = np.pi / n
                for k in range(n):
                    d = (theta - k * step) / step
                    yield j, b * _window((d + n / 2) % n - n / 2)

        return low2, windows()

    # 3D: smooth bumps round directions on the half sphere, normalised so that
    # their squares add up to one; |cos| makes each even
    r[(0,) * len(shape)] = 1.0
    inv_r = 1 / r
    del r
    finest = finest_angles or 12

    def bumps(n):
        c0 = np.cos(min(1.6 * np.sqrt(2 * np.pi / n), np.pi / 2))  # > the spacing
        for d in _hemisphere(n):
            c = np.abs(sum(f * dk for f, dk in zip(fs, d))) * inv_r
            yield _window((1 - c) / (1 - c0))

    def windows():
        for j in range(n_bands):
            n = _n_directions(j, finest)
            norm2 = sum(g**2 for g in bumps(n))
            norm2[norm2 == 0] = 1.0
            scale = radial(j) / np.sqrt(norm2)
            del norm2
            for g in bumps(n):  # made again rather than all kept at once
                yield j, scale * g

    return low2, windows()


# sym4 analysis filters (as PyWavelets defines them)
_SYM4_LO = np.array([-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161,
                     0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427])  # fmt: skip
_SYM4_HI = np.array([-0.0322231006040427, -0.012603967262037833, 0.09921954357684722, 0.29785779560527736,
                     -0.8037387518059161, 0.49761866763201545, 0.02963552764599851, -0.07576571478927333])  # fmt: skip


def wavelet_windows(shape, levels: int):
    """The undecimated sym4 wavelet as windows on the half space of ``rfftn``.

    Level j filters every axis with H(2^j w)/sqrt(2) or G(2^j w)/sqrt(2), so the
    squared windows of the detail sub-bands of every level (3 in 2D, 7 in 3D)
    and the final approximation add up to one: the image comes back exactly.

    Parameters
    ----------
    shape : tuple of int
        Image or map shape.
    levels : int
        Number of levels.

    Returns
    -------
    low2 : numpy.ndarray
        Squared window of the approximation that is kept.
    windows : generator
        ``(level, window)`` for every detail sub-band (complex windows).
    """
    from itertools import product

    nd = len(shape)
    ws = [2 * np.pi * np.ravel(f) for f in _freqs(shape)]
    axes = [[-1 if i == k else 1 for i in range(nd)] for k in range(nd)]

    def response(taps, w, j):
        n = np.arange(len(taps))[:, None]
        return (taps[:, None] * np.exp(-1j * (2**j) * w[None] * n)).sum(0) / np.sqrt(2)

    # separable: every window is an outer product of 1D responses, made when needed
    approx = [np.ones(w.size, complex) for w in ws]
    levels_1d = []
    for j in range(levels):
        lo = [response(_SYM4_LO, w, j) for w in ws]
        hi = [response(_SYM4_HI, w, j) for w in ws]
        levels_1d.append(
            ([a * f for a, f in zip(approx, lo)], [a * f for a, f in zip(approx, hi)])
        )
        approx = [a * f for a, f in zip(approx, lo)]

    def outer(parts):
        w = 1.0
        for k, p in enumerate(parts):
            w = w * p.reshape(axes[k])
        return w

    def windows():
        for j, (lo, hi) in enumerate(levels_1d):
            for combo in product((0, 1), repeat=nd):
                if any(combo):
                    yield j, outer([hi[k] if c else lo[k] for k, c in enumerate(combo)])

    return np.abs(outer(approx)) ** 2, windows()


@contextmanager
def _reuse_freed_memory():
    """Keep freed memory in the process (glibc) while denoising.

    Every temporary array of a large image or map is a fresh allocation, which
    glibc maps from the kernel and gives back when freed; on a machine whose
    memory is filled by the file cache, faulting the pages in costs ~1 s per
    130 MB, 30 times the arithmetic. Allocating from the heap and not trimming it
    reuses the pages. The defaults are restored, and the memory given back, after.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-4, 0)  # M_MMAP_MAX
        libc.mallopt(-1, 2**30)  # M_TRIM_THRESHOLD
    except (OSError, AttributeError):  # not glibc
        libc = None
    try:
        yield
    finally:
        if libc is not None:
            libc.mallopt(-4, 65536)
            libc.mallopt(-1, 128 * 1024)
            libc.malloc_trim(0)


def _flat_threshold(x):
    """Local variance below which a voxel counts as flat.

    Maps are (nearly) constant outside the sphere: a soft-masked outside is many
    orders of magnitude below the noise, so a sample of the map is enough.
    """
    from scipy.ndimage import uniform_filter

    step = int(np.ceil((x.size / 2**24) ** (1 / x.ndim)))  # at most ~16M voxels
    sample = x[(slice(None, None, step),) * x.ndim]
    m = uniform_filter(sample, 3)
    v = uniform_filter(sample * sample, 3) - m * m
    return 1e-4 * float(np.percentile(v[(slice(None, None, 3),) * x.ndim], 90))


def _not_flat(x, threshold=None):
    """Voxels whose local variance is above ``threshold`` (by default, the map's own)."""
    from scipy.ndimage import uniform_filter

    if threshold is None:
        threshold = _flat_threshold(x)
    m = uniform_filter(x, 3)
    v = uniform_filter(x * x, 3) - m * m
    return v > threshold


def _odd_pad(x, margin):
    """Mirror-pad by ``margin``, then to odd sizes.

    The margin keeps the periodic FFT from wrapping one edge into the other; odd
    sizes leave no Nyquist plane to break the windows' symmetry.
    """
    pad = [(margin, margin + (n + 2 * margin + 1) % 2) for n in x.shape]
    return np.pad(x, pad, mode="reflect"), pad


def _filter(xp, tiling, monogenic, workers, shrink):
    """Run ``shrink(index, w, q0, amp)`` on every sub-band of ``xp``.

    q0 is the real sub-band IFFT(F W), amp its monogenic amplitude
    sqrt(q0^2 + q1^2 + ...) (q1... its Riesz transforms) with ``monogenic``,
    else |q0|. ``shrink`` returns the new sub-band, or None to only look at it.

    Returns
    -------
    numpy.ndarray or None
        ``xp`` rebuilt from the low-pass and the new sub-bands; None if
        ``shrink`` returned None for every sub-band.
    """
    import scipy.fft as sf

    shape = xp.shape
    real = xp.dtype
    low2, windows = tiling(shape)
    F = sf.rfftn(xp, workers=workers)
    out = None
    if monogenic:
        # Riesz kernels -i f_k / |f|: 1/|f| once, the f_k broadcast from 1D
        fs = [(-1j * f).astype(F.dtype) for f in _freqs(shape)]
        inv_r = np.sqrt(sum(f.imag**2 for f in fs))
        inv_r[(0,) * len(shape)] = 1.0
        inv_r = (1 / inv_r).astype(real)
    for i, (_level, w) in enumerate(windows):
        w = w.astype(F.dtype if np.iscomplexobj(w) else real)
        Fw = F * w
        q0 = sf.irfftn(Fw, s=shape, workers=workers)
        if monogenic:
            amp = q0**2
            Fw *= inv_r
            for f in fs:
                amp += sf.irfftn(Fw * f, s=shape, workers=workers) ** 2
            np.sqrt(amp, out=amp)
        else:
            amp = np.abs(q0)
        del Fw
        new = shrink(i, w, q0, amp)
        del q0, amp
        if new is not None:
            if out is None:
                out = F * low2.astype(real)
            out += sf.rfftn(new, workers=workers) * np.conj(w)
        del new
    del F
    return None if out is None else sf.irfftn(out, s=shape, workers=workers)


def _mad(sample):
    """Noise sd from the median absolute deviation of a sample."""
    return float(np.median(np.abs(sample - np.median(sample))) / 0.6745)


def _soft(q0, amp, t):
    """Soft-threshold the amplitude: q0 * max(amp - t, 0) / amp."""
    att = np.maximum(amp - t, 0)
    np.divide(att, amp, out=att, where=amp > 0)
    att *= q0
    return att


def _shrink(x, tiling, strength, noise_sd, monogenic, workers, margin):
    """Soft-threshold every sub-band of a Fourier tiling with BayesShrink thresholds.

    Each sub-band q0 gets ``strength * s_n^2 / s_x``: s_n its noise, s_x the
    signal left after the noise. ``noise_sd`` is the per-pixel sd of white
    noise, carried through W; or a map of the noise alone (from the half maps),
    s_n the spread of its sub-band; with None, s_n is the spread (MAD) of the
    sub-band itself, for coloured noise where the signal is sparse (a map). The
    statistics leave out flat voxels. With ``monogenic`` the threshold acts on
    the monogenic amplitude, which keeps the local phase; otherwise on |q0|.
    """
    xp, pad = _odd_pad(x, margin)
    noise_map = isinstance(noise_sd, np.ndarray)
    region = _not_flat(xp) if noise_sd is None or noise_map else None
    shape = xp.shape
    # |W|^2 summed over the whole space, from the half that rfftn keeps
    cw = np.full(shape[-1] // 2 + 1, 2.0)
    cw[0] = 1.0
    if shape[-1] % 2 == 0:
        cw[-1] = 1.0
    if noise_map:
        s_ns = {}

        def measure(i, _w, q0, _amp):
            s_ns[i] = float(q0[region].std(dtype=np.float64))

        noise = _odd_pad(noise_sd.astype(x.dtype, copy=False), margin)[0]
        _filter(noise, tiling, False, workers, measure)
        del noise

    def shrink(i, w, q0, amp):
        q = q0 if region is None else q0[region]
        if noise_map:
            s_n = s_ns[i]
        elif noise_sd is None:
            s_n = _mad(q.ravel()[::7])
        else:
            s_n = noise_sd * np.sqrt(float((np.abs(w) ** 2 * cw).sum()) / xp.size)
        s_x = np.sqrt(max(float(q.var()) - s_n**2, 1e-12 * s_n**2))
        return _soft(q0, amp, strength * s_n**2 / s_x)

    result = _filter(xp, tiling, monogenic, workers, shrink)
    return result[tuple(slice(lo, lo + n) for (lo, _), n in zip(pad, x.shape))]


def _tile(x, start, size):
    """The block of ``x`` from ``start`` of ``size`` voxels, mirrored where it is outside."""
    src, pad = [], []
    for s0, n, total in zip(start, size, x.shape):
        lo, hi = max(s0, 0), min(s0 + n, total)
        src.append(slice(lo, hi))
        pad.append((lo - s0, s0 + n - hi))
    return np.pad(x[tuple(src)], pad, mode="reflect")


def _tiled_shrink(x, transforms, workers, core, margin, noise=None):
    """Denoise a map tile by tile, with thresholds from the whole map.

    Tiles of ``core`` voxels a side are cut with ``margin`` more round them
    (mirrored past the map's edges, so that every tile has the same shape and
    the same windows) and only their cores kept. A first pass pools every
    sub-band's statistics over the cores of all tiles, so that a tile full of
    molecule does not take its signal for noise; a second pass thresholds.
    With a ``noise`` map, each sub-band's noise is its spread there instead.

    Parameters
    ----------
    x : numpy.ndarray
        3D map.
    transforms : list of (callable, bool, float)
        ``(tiling, monogenic, strength)`` of each transform; the results are
        averaged.
    workers : int
        FFT threads.
    core, margin : int
        Tile core and margin, in voxels.

    Returns
    -------
    numpy.ndarray
        The denoised map, in the precision of ``x``.
    """
    from itertools import product

    size = core + 2 * margin
    size += 1 - size % 2  # odd: no Nyquist plane
    starts = list(product(*[range(0, n, core) for n in x.shape]))
    threshold = _flat_threshold(x)
    # every sub-band's MAD from at most about a million samples
    n_region = x.size * float(
        _not_flat(x[(slice(None, None, 4),) * 3], threshold).mean()
    )
    stride = max(7, int(n_region / 1e6))

    def core_of(start):
        """The core in tile coordinates, cut at the map's edges."""
        return tuple(
            slice(margin, margin + min(core, n - s0)) for s0, n in zip(start, x.shape)
        )

    stats, noise_stats = {}, {}
    for start in starts:
        piece = _tile(x, [s0 - margin for s0 in start], [size] * 3)
        region = np.zeros(piece.shape, bool)
        c = core_of(start)
        region[c] = _not_flat(piece, threshold)[c]
        for t, (tiling, _monogenic, _strength) in enumerate(transforms):

            def collect(i, _w, q0, _amp, t=t):
                q = q0[region]
                acc = stats.setdefault((t, i), [0, 0.0, 0.0, []])
                acc[0] += q.size
                acc[1] += float(q.sum(dtype=np.float64))
                acc[2] += float(np.square(q, dtype=np.float64).sum())
                acc[3].append(q[::stride].copy())

            # the statistics need q0 only: no Riesz transforms
            _filter(piece, tiling, False, workers, collect)
            if noise is not None:

                def collect_noise(i, _w, q0, _amp, t=t):
                    q = q0[region]
                    acc = noise_stats.setdefault((t, i), [0.0, 0.0])
                    acc[0] += float(q.sum(dtype=np.float64))
                    acc[1] += float(np.square(q, dtype=np.float64).sum())

                noise_piece = _tile(noise, [s0 - margin for s0 in start], [size] * 3)
                _filter(noise_piece, tiling, False, workers, collect_noise)
    thresholds = {}
    for (t, i), (n, s1, s2, samples) in stats.items():
        if noise is not None and n:
            m1, m2 = (v / n for v in noise_stats[t, i])
            s_n = np.sqrt(max(m2 - m1**2, 0.0))
        else:
            s_n = _mad(np.concatenate(samples)) if n else 0.0
        var = s2 / n - (s1 / n) ** 2 if n else 0.0
        s_x = np.sqrt(max(var - s_n**2, 1e-12 * s_n**2))
        thresholds[t, i] = transforms[t][2] * s_n**2 / max(s_x, 1e-30)
    stats.clear()

    out = np.zeros(x.shape, x.dtype)
    for start in starts:
        piece = _tile(x, [s0 - margin for s0 in start], [size] * 3)
        c = core_of(start)
        dest = tuple(slice(s0, s0 + cc.stop - cc.start) for s0, cc in zip(start, c))
        for t, (tiling, monogenic, _strength) in enumerate(transforms):

            def shrink(i, _w, q0, amp, t=t):
                return _soft(q0, amp, thresholds[t, i])

            out[dest] += _filter(piece, tiling, monogenic, workers, shrink)[c]
    out /= len(transforms)
    return out


def _default_levels(shape):
    """Down to periods of about 1/16 of the image."""
    return max(2, int(np.log2(min(shape) / 16)))


def _margin(shape):
    # a map is flat, or a helix continues, at its edges: no margin
    return min(64, min(shape) // 8) if len(shape) == 2 else 0


def curvelet_denoise(
    image: np.ndarray,
    strength: float = DEFAULT_STRENGTH,
    noise_sd: float | None = None,
    n_bands: int | None = None,
    finest_angles: int | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """Monogenic curvelet denoising with BayesShrink thresholds.

    Parameters
    ----------
    image : numpy.ndarray
        2D image or 3D map; float32 is kept as float32 (half the memory).
    strength : float
        Multiplier on every threshold; 1 is the textbook BayesShrink.
    noise_sd : float, optional
        Per-pixel white-noise sd. By default :func:`white_noise_sd` for a 2D
        image, and the spread of each sub-band for a map.
    n_bands : int, optional
        Number of bands; by default down to periods of 1/16 of the image.
    finest_angles : int, optional
        Directions of the finest band (see :func:`curvelet_windows`).
    workers : int, optional
        FFT threads; by default :func:`helicon.available_cpu`.

    Returns
    -------
    numpy.ndarray
        The denoised image.
    """
    import helicon

    x = _as_float(image)
    if noise_sd is None and x.ndim == 2:
        noise_sd = white_noise_sd(x)
    n_bands = n_bands or _default_levels(x.shape)
    tiling = lambda shape: curvelet_windows(shape, n_bands, finest_angles)  # noqa: E731
    with _reuse_freed_memory():
        return _shrink(
            x,
            tiling,
            strength,
            noise_sd,
            True,
            workers or helicon.available_cpu(),
            _margin(x.shape),
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
        2D image or 3D map; float32 is kept as float32 (half the memory).
    strength : float
        Multiplier on every threshold; 1 is the textbook BayesShrink.
    noise_sd : float, optional
        Per-pixel white-noise sd. By default :func:`white_noise_sd` for a 2D
        image, and the spread of each sub-band for a map.
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

    x = _as_float(image)
    if noise_sd is None and x.ndim == 2:
        noise_sd = white_noise_sd(x)
    levels = levels or min(6, _default_levels(x.shape))
    tiling = lambda shape: wavelet_windows(shape, levels)  # noqa: E731
    with _reuse_freed_memory():
        return _shrink(
            x,
            tiling,
            strength,
            noise_sd,
            False,
            workers or helicon.available_cpu(),
            _margin(x.shape),
        )


def _as_float(image):
    x = np.asarray(image)
    return x if x.dtype == np.float32 else x.astype(np.float64)


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


def denoise_map(
    volume: np.ndarray,
    strength: float = MAP_STRENGTH,
    workers: int | None = None,
    precision: str = "single",
    tile_size: int | None = None,
    noise: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Denoise a 3D map: the average of the curvelet and wavelet results.

    The noise of a map is neither white nor the same everywhere, so every
    sub-band gets its own noise level, from its spread (MAD) over the voxels
    that are not flat: the molecule fills a small part of the box, the rest is
    noise. Measured on half maps, this did as well as the true noise level.

    Parameters
    ----------
    volume : numpy.ndarray
        3D map.
    strength : float
        Multiplier on every threshold; 1.5 (the default) was the best on amyloid
        half maps, other maps wanted up to 6 (see :func:`tune_strength`).
    workers : int, optional
        FFT threads; by default :func:`helicon.available_cpu`.
    precision : {"single", "double"}
        Arithmetic. Single (the default) gave the same results as double to 4
        digits on four maps, 2-3 times faster, in half the memory.
    tile_size : int, optional
        Denoise in tiles of this many voxels a side (plus a margin), with the
        thresholds still from the whole map; 0 for the whole map at once. By
        default tiles only when the whole map does not fit in memory, and
        smaller tiles if memory runs out. Tiles of 128 voxels changed the map by
        1-8% of what denoising changes (scores within 0.5%), and took 2-4 times
        longer.
    noise : numpy.ndarray, optional
        A map of the noise of ``volume`` alone, from the half maps (see
        :func:`half_map_noise`): every sub-band's noise is then its spread
        there, not guessed from the map.

    Returns
    -------
    denoised : numpy.ndarray
        The denoised map, float32.
    info : dict
        What was used: ``strength``, ``precision``, ``noise_from``,
        ``tile_size`` (None: the whole map) and ``tiles_why``.
    """
    import helicon

    x = np.asarray(volume)
    if x.ndim != 3:
        raise ValueError(f"denoise_map needs a 3D map, not {x.ndim}D")
    x = x.astype(np.float64 if precision == "double" else np.float32)
    if noise is not None:
        if np.shape(noise) != x.shape:
            raise ValueError(f"the noise map is {np.shape(noise)}, the map {x.shape}")
        noise = np.asarray(noise, x.dtype)
    workers = workers or helicon.available_cpu()
    core, tiles_why = _plan_tiles(x, tile_size)
    while True:
        try:
            if core is None:
                c = curvelet_denoise(
                    x, strength=strength, noise_sd=noise, workers=workers
                )
                s = wavelet_denoise(
                    x,
                    strength=strength * WAVELET_FACTOR,
                    noise_sd=noise,
                    workers=workers,
                )
                out = (0.5 * (c + s)).astype(np.float32)
            else:
                # the bands follow the tile, as for a map of that size
                transforms = [
                    (lambda s: curvelet_windows(s, _default_levels(s)), True, strength),
                    (
                        lambda s: wavelet_windows(s, min(6, _default_levels(s))),
                        False,
                        strength * WAVELET_FACTOR,
                    ),
                ]
                with _reuse_freed_memory():
                    out = _tiled_shrink(
                        x, transforms, workers, core, _TILE_MARGIN, noise
                    )
                out = out.astype(np.float32, copy=False)
            break
        except MemoryError:
            smaller = (core or max(x.shape)) // 2
            if smaller < 8:
                raise
            core = _even_core(x.shape, smaller)
            tiles_why = "halved after running out of memory"
    info = dict(
        strength=strength,
        precision=precision,
        noise_from=(
            "the half maps' difference"
            if noise is not None
            else "each sub-band's spread over the voxels that are not flat"
        ),
        tile_size=core,
        tiles_why=tiles_why,
    )
    return out, info


_TILE_MARGIN = 8  # measured: as close to the untiled map as 16 or 32, and faster
_MIN_CORE = 112  # tiles of 128+ voxels keep 3 bands; 2 bands lost 3% on a map
# measured peaks (tracemalloc), in copies of the map or tile in its precision
_COPIES = 21
_TILE_COPIES = 26


def _even_core(shape, core):
    """The smallest core that cuts every axis into as many tiles as ``core`` does."""
    return max(-(-n // -(-n // core)) for n in shape)


def _plan_tiles(x, tile_size=None):
    """Tile core for a map, or None for the whole map, and why.

    Parameters
    ----------
    x : numpy.ndarray
        The map, in the precision it is denoised in.
    tile_size : int, optional
        Tile core in voxels; 0 for the whole map. By default the whole map when
        it fits in 80% of :func:`helicon.available_memory`, otherwise the
        largest tiles that fit.

    Returns
    -------
    core : int or None
    why : str
    """
    import helicon

    if tile_size is not None:
        return (_even_core(x.shape, tile_size) if tile_size else None), "given"
    gb = 2**-30
    budget = 0.8 * helicon.available_memory()
    need = _COPIES * x.nbytes
    if need <= budget:
        return None, f"the whole map fits ({need * gb:.1f} of {budget * gb:.1f} GB)"
    # the output stays whole beside the input
    side = (max(budget - x.nbytes, 0) / (_TILE_COPIES * x.itemsize)) ** (1 / 3)
    core = _even_core(x.shape, max(_MIN_CORE, int(side) - 2 * _TILE_MARGIN))
    return core, f"the whole map needs {need * gb:.1f} GB, {budget * gb:.1f} GB is free"


# the other half of a half map: RELION, cryoSPARC, EMDB
_HALF_PAIRS = [("half1", "half2"), ("half_A", "half_B"), ("half_map_1", "half_map_2")]
# the half maps of a full map: (name pattern, half-map name with {h}, the two h)
_FULL_TO_HALVES = [
    (r"^(.*)_class(\d+)$", r"\1_half{h}_class\2_unfil", "12"),  # RELION Refine3D
    (r"^(.*_map)(_sharp)?$", r"\1_half_{h}", "AB"),  # cryoSPARC
    (r"^(emd_\d+)$", r"\1_half_map_{h}", "12"),  # EMDB
]


def find_half_maps(path):
    """The half maps of a map, found by the names RELION, cryoSPARC and EMDB give them.

    Parameters
    ----------
    path : str or pathlib.Path
        A half map, or a full map (``run_class001.mrc``,
        ``J12_005_volume_map[_sharp].mrc``, ``emd_1234.map``).

    Returns
    -------
    tuple of pathlib.Path or None
        ``(half, other half)``, the given map first if it is a half map; None
        when the other half (or either half) is not in the same folder.
    """
    import re
    from pathlib import Path

    path = Path(path)
    stem, ext = path.stem, path.suffix
    for one, two in _HALF_PAIRS:
        for a, b in ((one, two), (two, one)):
            if a in stem:
                other = path.with_name(stem.replace(a, b) + ext)
                return (path, other) if other.is_file() else None
    for pattern, template, hs in _FULL_TO_HALVES:
        if re.match(pattern, stem):
            halves = tuple(
                path.with_name(re.sub(pattern, template.replace("{h}", h), stem) + ext)
                for h in hs
            )
            if all(h.is_file() for h in halves):
                return halves
    return None


def half_map_noise(
    half1: np.ndarray, half2: np.ndarray, of: str = "half"
) -> np.ndarray:
    """A map with the noise of a half map, or of the average of the two.

    The halves share the signal and have independent noise of the same size,
    so (h1 - h2) / sqrt(2) is noise like a half map's, and (h1 - h2) / 2 like
    their average's: half the variance.

    Parameters
    ----------
    half1, half2 : numpy.ndarray
        The two half maps.
    of : {"half", "average"}
        Whose noise.

    Returns
    -------
    numpy.ndarray
        float32.
    """
    scale = {"half": np.sqrt(0.5), "average": 0.5}[of]
    d = np.subtract(half1, half2, dtype=np.float32)
    d *= scale
    return d


def tune_strength(
    half1: np.ndarray,
    half2: np.ndarray,
    workers: int | None = None,
    precision: str = "single",
    tile_size: int | None = None,
    noise_from_halves: bool = False,
) -> tuple[float, dict]:
    """The strength whose denoised half map 1 best predicts half map 2.

    The noise of the two halves is independent, so the mean squared difference
    from half map 2 is the error from the truth plus a constant. Strengths are
    tried a factor 2 apart from :data:`MAP_STRENGTH` until the middle one of
    three is the best (3-6 runs of :func:`denoise_map`), then refined by a
    parabola in log(strength), which is tried too (one more run).

    Parameters
    ----------
    half1, half2 : numpy.ndarray
        The two half maps.
    workers, precision, tile_size
        As for :func:`denoise_map`.
    noise_from_halves : bool
        Denoise half map 1 with its noise from :func:`half_map_noise`; the
        strength is then for maps denoised that way.

    Returns
    -------
    strength : float
    scores : dict
        Mean squared difference from half map 2 of every strength tried.
    """
    h2 = np.asarray(half2, np.float32)
    noise = half_map_noise(half1, h2) if noise_from_halves else None
    scores = {}

    def score(s):
        if s not in scores:
            d, _ = denoise_map(
                half1,
                s,
                workers=workers,
                precision=precision,
                tile_size=tile_size,
                noise=noise,
            )
            d -= h2
            scores[s] = float(np.square(d, dtype=np.float64).mean())
        return scores[s]

    s = MAP_STRENGTH
    while s > 0.2 and score(s / 2) < score(s):
        s /= 2
    while s < 20 and score(s * 2) < score(s):
        s *= 2
    a, b, c = score(s / 2), score(s), score(s * 2)
    curvature = a - 2 * b + c
    step = float(np.clip(0.5 * (a - c) / curvature, -1, 1)) if curvature > 0 else 0.0
    # the parabola can miss on a lopsided curve: keep it only if it is better
    score(round(s * 2**step, 2))
    return min(scores, key=scores.get), scores
