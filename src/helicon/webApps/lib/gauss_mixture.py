"""Analytic 2D Gaussian-mixture algebra for the helical Gaussian solver.

The solver never rasterises. A 3D Gaussian projects to a 2D Gaussian exactly,
so both the model (an asymmetric unit replicated by the screw symmetry) and the
target (a mixture fitted to the class average) are 2D Gaussian mixtures, and
every quantity the solver needs is a sum of pairwise overlap integrals.

For unnormalised components ``f(x) = a exp(-0.5 (x-mu)^T P (x-mu))`` with
``P = Sigma^-1``:

    <f_i, f_j> = a_i a_j * 2*pi / sqrt(det(P_i + P_j))
                 * exp(-0.5 d^T (S_i + S_j)^-1 d),    d = mu_i - mu_j

Note which matrix appears where: the prefactor uses ``det(P_i + P_j)`` while
the exponent uses ``(S_i + S_j)^-1``. Using ``S_i + S_j`` for both is a
seductive-looking error that costs a factor ``det(S_i) det(S_j)`` -- for
isotropic components ``sigma_i^2 sigma_j^2`` -- and silently drags every score
down without changing their ordering much, so it does not announce itself.
"""

from __future__ import annotations

import numpy as np

SQRT_2PI = float(np.sqrt(2 * np.pi))

__all__ = [
    "iso_overlap",
    "gen_overlap",
    "self_energy",
    "SQRT_2PI",
]


def iso_overlap(a1, m1, s1, a2, m2, s2):
    """``<f1, f2>`` for isotropic 2D Gaussians, broadcasting over leading axes.

    Parameters
    ----------
    a1, a2 : array_like
        Amplitudes.
    m1, m2 : array_like
        Centres, last axis of length 2.
    s1, s2 : array_like
        Isotropic sigmas.
    """
    ss = s1**2 + s2**2
    d2 = ((m1 - m2) ** 2).sum(-1)
    return a1 * a2 * 2 * np.pi * (s1**2) * (s2**2) / ss * np.exp(-0.5 * d2 / ss)


def gen_overlap(a1, m1, c1, a2, m2, c2):
    """``<f1, f2>`` for general 2D Gaussians given covariances ``c1``, ``c2``.

    Reduces to :func:`iso_overlap` for ``c = sigma**2 * I``; that equivalence is
    checked in the tests, because the prefactor is easy to get wrong.
    """
    ssum = c1 + c2
    sdet = ssum[..., 0, 0] * ssum[..., 1, 1] - ssum[..., 0, 1] * ssum[..., 1, 0]
    sinv = np.empty_like(ssum)
    sinv[..., 0, 0] = ssum[..., 1, 1]
    sinv[..., 1, 1] = ssum[..., 0, 0]
    sinv[..., 0, 1] = -ssum[..., 0, 1]
    sinv[..., 1, 0] = -ssum[..., 1, 0]
    sinv = sinv / sdet[..., None, None]

    d1 = c1[..., 0, 0] * c1[..., 1, 1] - c1[..., 0, 1] * c1[..., 1, 0]
    d2 = c2[..., 0, 0] * c2[..., 1, 1] - c2[..., 0, 1] * c2[..., 1, 0]
    pdet = sdet / (d1 * d2)  # det(P1 + P2) = det(S1+S2)/(detS1 detS2)

    d = m1 - m2
    quad = np.einsum("...i,...ij,...j->...", d, sinv, d)
    return a1 * a2 * 2 * np.pi / np.sqrt(pdet) * np.exp(-0.5 * quad)


def self_energy(mixture):
    """``<t, t>`` for a general mixture ``(amp, mu, cov)``."""
    a, m, c = mixture
    return float(
        gen_overlap(
            a[:, None],
            m[:, None, :],
            c[:, None, :, :],
            a[None, :],
            m[None, :, :],
            c[None, :, :, :],
        ).sum()
    )
