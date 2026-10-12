"""Smoothing along the local direction of filaments.

An isotropic blur averages noise over a disc, which at the width of a filament
(50 A) averages over little of it. Along a filament the image is nearly
constant for hundreds of A, so an elongated filter averages far more noise
without blurring the filament across. The filaments' direction is not known, so
a bank of such filters is applied, one for each direction, and each pixel takes
the filter that responds best to a ridge there (a steerable blend, not a hard
choice, so that curved filaments are followed).

All filters are Gaussians in the Fourier domain, so the bank costs one FFT of the
image and one inverse FFT per direction.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft


def _frequencies(shape):
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    return fy, fx


def bank(image, apix, across=20.0, along=300.0, n_angles=24, ridge_width=None):
    """The image smoothed along each of ``n_angles`` directions, and its ridge response.

    Parameters
    ----------
    image : numpy.ndarray
        The micrograph (filaments dark).
    apix : float
        Pixel size, A.
    across, along : float
        Standard deviations of the Gaussian across and along the filter's
        direction, in A.
    n_angles : int
        Directions in [0, 180).
    ridge_width : float, optional
        Standard deviation (A) across which the ridge response is taken;
        defaults to ``across``.

    Returns
    -------
    smooth : numpy.ndarray
        ``(n_angles, ny, nx)``: the image smoothed along each direction.
    ridge : numpy.ndarray
        The same shape: the curvature across the direction of the smoothed
        image (positive across a dark line), scaled by its width squared.
    angles : numpy.ndarray
        The directions, radians, of the line (not of its normal).
    """
    img = np.asarray(image, np.float32)
    img = img - img.mean()
    F = sfft.rfft2(img, workers=-1)
    fy, fx = (a.astype(np.float32) for a in _frequencies(img.shape))
    sa, sl = across / apix, along / apix
    sr = (ridge_width or across) / apix
    angles = np.arange(n_angles) * np.pi / n_angles
    smooth = np.empty((n_angles,) + img.shape, np.float32)
    ridge = np.empty_like(smooth)
    for k, t in enumerate(angles):
        u_par = fx * np.cos(t) + fy * np.sin(t)  # along the line
        u_perp = -fx * np.sin(t) + fy * np.cos(t)  # across it
        H = np.exp(-2 * np.pi**2 * (sl**2 * u_par**2 + sa**2 * u_perp**2))
        smooth[k] = sfft.irfft2(F * H, s=img.shape, workers=-1)
        Hr = np.exp(-2 * np.pi**2 * (sl**2 * u_par**2 + sr**2 * u_perp**2))
        ridge[k] = sfft.irfft2(
            F * Hr * -((2 * np.pi * u_perp * sr) ** 2), s=img.shape, workers=-1
        )
    return smooth, ridge, angles


def ridge_map(image, apix, across=25.0, along=300.0, n_angles=24):
    """The strongest ridge response over the directions, and the direction."""
    _, ridge, angles = bank(image, apix, across, along, n_angles)
    best = ridge.argmax(0)
    return np.maximum(ridge.max(0), 0.0), angles[best]


def enhance(image, apix, across=20.0, along=300.0, n_angles=24, sharp=4.0, mix=1.0):
    """The image smoothed along the local direction of its filaments.

    Each pixel is the blend of the directional smoothings, weighted by how well
    their direction fits the strongest ridge response there (``sharp`` sets how
    narrow); where no ridge stands out the blend is the plain isotropic
    smoothing of the same width across, so that empty background is not streaked.
    ``mix`` scales how much of the directional result is used (0: isotropic).
    """
    smooth, ridge, angles = bank(image, apix, across, along, n_angles)
    pos = np.maximum(ridge, 0.0)
    strongest = pos.max(0)
    mean = pos.mean(0)
    # how much stronger the best direction is than the average one: 0 for a
    # pixel that responds alike in every direction (background), towards 1 for a line
    coherence = np.where(strongest > 0, 1.0 - mean / np.maximum(strongest, 1e-12), 0.0)
    w = np.exp(sharp * (pos / np.maximum(strongest, 1e-12) - 1.0))
    w /= w.sum(0, keepdims=True)
    directional = (w * smooth).sum(0)
    iso = smooth.mean(0)
    c = np.clip(mix * coherence, 0.0, 1.0)
    return c * directional + (1.0 - c) * iso
