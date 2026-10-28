"""Half maps whose signal and noise are known, for the trueFSC tests.

The signal is a compact particle of Gaussian atoms with a weak domain attached
(easy to cut off with a tight mask), or a helical tube along z whose box ends are
noisier than its middle. Knowing the signal ``S`` and the noises ``n1``, ``n2``,
the FSC that the masked signal really has is its power over signal plus noise
power, shell by shell (:func:`true_fsc`).
"""

from __future__ import annotations

import numpy as np


def _atoms(centres, n, sigma=1.2, amp=1.0):
    from scipy.ndimage import gaussian_filter

    v = np.zeros((n, n, n))
    idx = np.clip(np.round(centres).astype(int), 0, n - 1)
    np.add.at(v, tuple(idx.T), amp)
    return gaussian_filter(v, sigma)


def _in_ellipsoid(rng, count, radii, centre):
    radii = np.asarray(radii, dtype=float)
    points = []
    while len(points) < count:
        p = rng.uniform(-radii, radii)
        if np.sum((p / radii) ** 2) < 1:
            points.append(p + centre)
    return np.array(points)


def particle(n=64, sigma=0.9, seed=0):
    """A particle with a weak domain on top, in white noise.

    Returns
    -------
    tuple
        ``S, n1, n2`` (signal of standard deviation 1, and the two noises).
    """
    rng = np.random.default_rng(seed)
    c = n / 2
    s = _atoms(_in_ellipsoid(rng, 1200, (15, 11, 10), c), n)
    weak = _in_ellipsoid(rng, 150, (6, 6, 6), c + np.array([15, 0, 0]))
    s = s + _atoms(weak, n, amp=0.35)
    s /= s.std()
    return s, sigma * rng.normal(size=s.shape), sigma * rng.normal(size=s.shape)


def helix(n=64, sigma=0.6, seed=0):
    """A tube along z, three times noisier in the outer 40% of the sections.

    Returns
    -------
    tuple
        ``S, n1, n2``.
    """
    rng = np.random.default_rng(seed)
    count = 1600
    z = rng.uniform(0, n, count)
    phi = rng.uniform(0, 2 * np.pi, count)
    r = rng.uniform(0.15 * n, 0.26 * n, count)
    centres = np.stack([z, n / 2 + r * np.sin(phi), n / 2 + r * np.cos(phi)], axis=1)
    s = _atoms(centres, n)
    s /= s.std()
    zc = np.arange(n) - n / 2
    scale = np.where(np.abs(zc) > 0.3 * n, 3.0, 1.0)[:, None, None]
    return (
        s,
        sigma * scale * rng.normal(size=s.shape),
        sigma * scale * rng.normal(size=s.shape),
    )


def _shell_power(v):
    n = v.shape[0]
    F = np.fft.rfftn(v)
    k2 = np.fft.fftfreq(n) ** 2
    kr2 = np.fft.rfftfreq(n) ** 2
    shell = np.round(
        np.sqrt(k2[:, None, None] + k2[None, :, None] + kr2[None, None, :]) * n
    ).astype(int)
    shell = np.clip(shell, 0, n // 2).ravel()
    return np.bincount(shell, np.abs(F.ravel()) ** 2, minlength=n // 2 + 1)


def true_fsc(S, n1, n2, mask):
    """The FSC the masked signal really has, shell by shell."""
    ps = _shell_power(S * mask)
    pn = (_shell_power(n1 * mask) + _shell_power(n2 * mask)) / 2
    return ps / (ps + pn)


def write(path, data, apix):
    import mrcfile

    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(np.asarray(data, dtype=np.float32))
        m.voxel_size = apix


def run(tmp_dir, maps, apix=1.5, **options):
    """trueFSC on ``S + n1`` and ``S + n2``, and how its answer compares with the
    truth.

    Returns
    -------
    dict
        ``result`` (of :func:`compute_truefsc`), ``mask``, ``measured`` and
        ``truth`` (the resolutions, as 1/area under the curve, of the reported
        and of the real FSC of the mask it chose, Angstrom) and ``signal_kept``
        (the fraction of the positive signal in the scored region that the mask
        keeps). The area rather than the 0.143 crossing: on a small box the
        crossing jumps by a whole shell (0.17 Angstrom here) with any small
        numerical difference.
    """
    import mrcfile

    from helicon.commands import trueFSC

    S, n1, n2 = maps
    n = S.shape[0]
    f1, f2 = tmp_dir / "half1.mrc", tmp_dir / "half2.mrc"
    write(f1, S + n1, apix)
    write(f2, S + n2, apix)
    result = trueFSC.compute_truefsc(f1, f2, tmp_dir / "fsc.pdf", **options)
    mask = mrcfile.read(result["volumes"]["mask1_file"]).astype(float)
    region = np.ones((n, 1, 1))
    z = trueFSC._z_pixels(
        n, apix, options.get("mask_z_len", -1), options.get("mask_z_fraction", 1)
    )
    if z > 0:
        region = trueFSC._central_z(S.shape, z)
    positive = np.maximum(S, 0) * region
    saxis = np.arange(n // 2 + 1) / (n * apix)
    return dict(
        result=result,
        mask=mask,
        measured=result["resolution_area"],
        truth=trueFSC._fsc_area(saxis, true_fsc(S, n1, n2, mask))[1],
        signal_kept=float(np.sum(positive * mask) / np.sum(positive)),
    )
