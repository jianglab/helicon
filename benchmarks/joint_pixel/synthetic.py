"""Synthetic helical data with a known repeat, for the joint ring tests.

A helical structure is a few Gaussians repeated by a screw; its side view over
the axial coordinate ``z`` repeats after the pitch ``P`` (C1). A class average
is a window of that side view, and a 2D classification puts each segment into
the class whose window is nearest the azimuth the segment is at. Everything is
known here, so the repeat and the phase of every class can be scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from helicon.webApps.lib.map_gauss_fit import MapGaussians


def helix(period, rise=4.75, n_gauss=14, radius=18.0, sigma=3.0, seed=0):
    """A random one-start helix whose side view repeats every ``period`` A.

    Returns the asymmetric unit and the twist (degrees) that gives that period.
    """
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n_gauss))
    a = 2 * np.pi * rng.random(n_gauss)
    centers = np.column_stack(
        [r * np.cos(a), r * np.sin(a), rng.uniform(-rise / 2, rise / 2, n_gauss)]
    )
    amp = rng.uniform(0.5, 1.5, n_gauss)
    unit = MapGaussians(amp.astype(np.float32), centers.astype(np.float32), sigma, 1.0)
    twist = 360.0 * rise / period
    return unit, twist, rise


def side_view(unit, twist, rise, z0, ny, nx, apix):
    """The side projection of a window ``nx`` pixels wide centred at ``z0`` (A)."""
    half = nx * apix / 2.0
    pad = 4 * unit.sigma
    centers, amp = unit.screw_expand(twist, rise, 1, z0 - half - pad, z0 + half + pad)
    y = (np.arange(ny) - ny / 2.0 + 0.5) * apix
    z = z0 + (np.arange(nx) - nx / 2.0 + 0.5) * apix
    s = unit.sigma
    gy = np.exp(-0.5 * ((y[:, None] - centers[None, :, 1]) / s) ** 2)
    gz = np.exp(-0.5 * ((z[:, None] - centers[None, :, 2]) / s) ** 2)
    img = (gy * (amp * np.sqrt(2 * np.pi) * s)[None, :]) @ gz.T
    return img.astype(np.float32)


def class_averages(
    period,
    n_classes,
    rise=4.75,
    ny=48,
    nx=96,
    apix=4.0,
    blur=5,
    noise=0.05,
    contamination=0.0,
    seed=0,
    structure_seed=0,
):
    """Class averages, one per sector of the repeat, and the true class phases.

    Each average is the mean of ``blur`` windows spread over its sector (a class
    holds segments from the whole sector, which blurs it), plus noise. Returns
    ``(images, phase_A)``: ``phase_A[k]`` is where the sector's centre lies
    along the repeat, in A.

    ``contamination`` is the share of each average made of segments of other
    classes (what a poor assignment does to an average): that share is the
    mean over the whole repeat instead of the sector's own windows.
    """
    unit, twist, rise = helix(period, rise, seed=structure_seed)
    rng = np.random.default_rng(seed)
    images, phases = [], []
    everywhere = None
    if contamination > 0:
        everywhere = np.mean(
            [
                side_view(unit, twist, rise, (j + 0.5) / 24 * period, ny, nx, apix)
                for j in range(24)
            ],
            axis=0,
        )
    for k in range(n_classes):
        zs = [(k + (j + 0.5) / blur) / n_classes * period for j in range(blur)]
        img = np.mean(
            [side_view(unit, twist, rise, z, ny, nx, apix) for z in zs], axis=0
        )
        if everywhere is not None:
            img = (1.0 - contamination) * img + contamination * everywhere
        images.append(img + noise * img.max() * rng.standard_normal(img.shape))
        phases.append((k + 0.5) / n_classes * period)
    return images, np.asarray(phases)


def class2d_table(
    period,
    n_classes,
    n_fil=200,
    length=800.0,
    step=20.0,
    mislabel=0.0,
    position_sd=0.0,
    period_sd=0.0,
    seed=0,
):
    """Class2D parameters for straight filaments of known period.

    A segment is labelled by the sector of the repeat it sits at; a fraction
    ``mislabel`` is given a random class instead (a poor assignment). Positions
    along the filament have noise ``position_sd`` (A).
    """
    rng = np.random.default_rng(seed)
    n_seg = max(2, int(length / step) + 1)
    rows = []
    for f in range(n_fil):
        p = period * (1.0 + period_sd * rng.standard_normal())
        track = np.arange(n_seg) * step
        phase = (track / p + rng.random()) % 1.0
        cls = np.minimum((phase * n_classes).astype(int), n_classes - 1)
        bad = rng.random(n_seg) < mislabel
        cls[bad] = rng.integers(0, n_classes, int(bad.sum()))
        pos = track + position_sd * rng.standard_normal(n_seg)
        for t, c in zip(pos, cls):
            rows.append(
                dict(
                    rlnMicrographName=f"mic{f // 10:03d}.mrc",
                    rlnHelicalTubeID=f % 10 + 1,
                    rlnHelicalTrackLengthAngst=t,
                    rlnClassNumber=int(c) + 1,
                    rlnAnglePsi=0.0,
                    rlnAnglePsiPrior=0.0,
                    rlnCoordinateX=1000.0 + t,
                    rlnCoordinateY=500.0,
                    rlnOriginXAngst=0.0,
                    rlnOriginYAngst=0.0,
                    rlnMicrographOriginalPixelSize=1.0,
                )
            )
    return pd.DataFrame(rows)


def phase_error(estimated_rad, true_A, period):
    """RMS error of the class phases, as a fraction of the repeat.

    The estimate is the eigenvector's angle (minus ``2 pi theta / P``, up to one
    common constant), so it is compared after removing the best constant, and
    with either sign of the direction, taking the better.
    """
    best = np.inf
    for s in (1.0, -1.0):
        d = s * np.asarray(estimated_rad) + 2 * np.pi * np.asarray(true_A) / period
        c = np.angle(np.mean(np.exp(1j * d)))
        r = np.angle(np.exp(1j * (d - c)))
        best = min(best, float(np.sqrt(np.mean(r**2)) / (2 * np.pi)))
    return best


def class2d_table_mixture(
    periods=(400.0, 440.0),
    fractions=(0.5, 0.5),
    n_classes=12,
    n_fil=200,
    length=1200.0,
    step=20.0,
    mislabel=0.0,
    seed=0,
):
    """Class2D parameters of two helical types that share their classes.

    Each filament is of one type, with that type's repeat; a segment is labelled
    by the sector of *its own* repeat it sits at, so both types fall into the same
    classes. Returns the table and the true type of every filament (by the order
    in which ``prepare_pairs`` numbers them, which is the table's order).
    """
    rng = np.random.default_rng(seed)
    n_seg = max(2, int(length / step) + 1)
    types = rng.choice(len(periods), size=n_fil, p=np.asarray(fractions))
    rows = []
    for f in range(n_fil):
        p = periods[types[f]]
        track = np.arange(n_seg) * step
        phase = (track / p + rng.random()) % 1.0
        cls = np.minimum((phase * n_classes).astype(int), n_classes - 1)
        bad = rng.random(n_seg) < mislabel
        cls[bad] = rng.integers(0, n_classes, int(bad.sum()))
        for t, c in zip(track, cls):
            rows.append(
                dict(
                    rlnMicrographName=f"mic{f // 10:03d}.mrc",
                    rlnHelicalTubeID=f % 10 + 1,
                    rlnHelicalTrackLengthAngst=t,
                    rlnClassNumber=int(c) + 1,
                    rlnAnglePsi=0.0,
                    rlnAnglePsiPrior=0.0,
                    rlnCoordinateX=1000.0 + t,
                    rlnCoordinateY=500.0,
                    rlnOriginXAngst=0.0,
                    rlnOriginYAngst=0.0,
                    rlnMicrographOriginalPixelSize=1.0,
                )
            )
    return pd.DataFrame(rows), types
