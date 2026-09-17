"""Tests for the analytic 2D Gaussian-mixture algebra."""

import numpy as np
import pytest

from helicon.webApps.lib import gauss_mixture as gm


def _render(amp, mu, cov, extent=40.0, step=0.2):
    """Evaluate a mixture on a fine grid, for numerical integration."""
    g = np.arange(-extent, extent, step)
    Y, X = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([Y.ravel(), X.ravel()], -1)
    P = np.linalg.inv(cov)
    out = np.zeros(len(pts))
    for a, m, p in zip(amp, mu, P):
        d = pts - m
        out += a * np.exp(-0.5 * np.einsum("ni,ij,nj->n", d, p, d))
    return out, step


def _iso_cov(sigmas):
    c = np.zeros((len(sigmas), 2, 2))
    c[:, 0, 0] = np.asarray(sigmas) ** 2
    c[:, 1, 1] = np.asarray(sigmas) ** 2
    return c


class TestOverlapIdentity:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_general_reduces_to_isotropic(self, seed):
        """The two code paths must agree where they overlap.

        This is the check that catches a wrong prefactor: using Sigma1+Sigma2
        instead of det(P1+P2) is off by sigma1^2 sigma2^2, which this exposes
        immediately while a fit-quality check would not.
        """
        rng = np.random.default_rng(seed)
        a1, a2 = rng.uniform(0.5, 2, 2)
        s1, s2 = rng.uniform(0.8, 3, 2)
        m1, m2 = rng.normal(0, 3, 2), rng.normal(0, 3, 2)
        iso = gm.iso_overlap(a1, m1, s1, a2, m2, s2)
        gen = gm.gen_overlap(a1, m1, np.eye(2) * s1**2, a2, m2, np.eye(2) * s2**2)
        assert gen == pytest.approx(iso, rel=1e-12)

    def test_prefactor_is_not_the_naive_one(self):
        """Guard the specific mistake: sigma1^2 sigma2^2 apart from the truth."""
        s1, s2 = 1.5, 2.5
        m = np.zeros(2)
        correct = gm.gen_overlap(1.0, m, np.eye(2) * s1**2, 1.0, m, np.eye(2) * s2**2)
        naive = 1.0 * 1.0 * 2 * np.pi / np.sqrt((s1**2 + s2**2) ** 2)
        assert correct == pytest.approx(naive * s1**2 * s2**2, rel=1e-12)
        assert correct != pytest.approx(naive, rel=1e-3)

    @pytest.mark.parametrize("seed", [0, 3])
    def test_matches_numerical_integration(self, seed):
        rng = np.random.default_rng(seed)
        n = 3
        amp = rng.uniform(0.5, 2, n)
        mu = rng.uniform(-4, 4, (n, 2))
        A = rng.normal(0, 1, (n, 2, 2))
        cov = np.einsum("nij,nkj->nik", A, A) + np.eye(2) * 2.0
        f, step = _render(amp, mu, cov)
        numeric = float((f * f).sum() * step * step)
        analytic = gm.self_energy((amp, mu, cov))
        assert analytic == pytest.approx(numeric, rel=1e-4)

    def test_cross_term_matches_numerical_integration(self):
        rng = np.random.default_rng(7)
        amp1, amp2 = rng.uniform(0.5, 2, 3), rng.uniform(0.5, 2, 4)
        mu1, mu2 = rng.uniform(-4, 4, (3, 2)), rng.uniform(-4, 4, (4, 2))
        c1, c2 = _iso_cov(rng.uniform(1.0, 2.5, 3)), _iso_cov(rng.uniform(1.0, 2.5, 4))
        f, step = _render(amp1, mu1, c1)
        g, _ = _render(amp2, mu2, c2)
        numeric = float((f * g).sum() * step * step)
        analytic = float(
            gm.gen_overlap(
                amp1[:, None],
                mu1[:, None, :],
                c1[:, None, :, :],
                amp2[None, :],
                mu2[None, :, :],
                c2[None, :, :, :],
            ).sum()
        )
        assert analytic == pytest.approx(numeric, rel=1e-4)


class TestProperties:
    def test_self_energy_is_positive(self):
        rng = np.random.default_rng(1)
        amp = rng.uniform(0.5, 2, 5)
        mu = rng.uniform(-5, 5, (5, 2))
        cov = _iso_cov(rng.uniform(1, 3, 5))
        assert gm.self_energy((amp, mu, cov)) > 0

    def test_cauchy_schwarz(self):
        """<f,g> <= ||f|| ||g||: a cosine built from these can never exceed 1."""
        rng = np.random.default_rng(2)
        for _ in range(5):
            a1, a2 = rng.uniform(0.5, 2, 4), rng.uniform(0.5, 2, 4)
            m1, m2 = rng.uniform(-5, 5, (4, 2)), rng.uniform(-5, 5, (4, 2))
            c1, c2 = _iso_cov(rng.uniform(1, 3, 4)), _iso_cov(rng.uniform(1, 3, 4))
            cross = float(
                gm.gen_overlap(
                    a1[:, None],
                    m1[:, None, :],
                    c1[:, None, :, :],
                    a2[None, :],
                    m2[None, :, :],
                    c2[None, :, :, :],
                ).sum()
            )
            n1 = gm.self_energy((a1, m1, c1))
            n2 = gm.self_energy((a2, m2, c2))
            assert cross <= np.sqrt(n1 * n2) * (1 + 1e-9)

    def test_translation_invariance(self):
        rng = np.random.default_rng(4)
        amp = rng.uniform(0.5, 2, 3)
        mu = rng.uniform(-3, 3, (3, 2))
        cov = _iso_cov(rng.uniform(1, 2, 3))
        shift = np.array([3.7, -2.1])
        assert gm.self_energy((amp, mu + shift, cov)) == pytest.approx(
            gm.self_energy((amp, mu, cov)), rel=1e-12
        )
