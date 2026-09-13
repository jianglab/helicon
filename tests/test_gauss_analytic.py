"""Tests for the grid-free analytic Gaussian helical solver."""

import numpy as np
import pytest

from helicon.webApps.lib import gauss_mixture as gm
from helicon.webApps.lib import solver_gauss_analytic as sga

RISE = 4.75 / 4.944  # px, EMPIAR-10940 geometry
NX = 128
NREP = int(NX / 2 / RISE) + 2


def _iso_target(mu, amp, sig, nx=NX):
    """Turn projected components into a target mixture, cropped to the image."""
    mu = mu.reshape(-1, 2)
    amp = amp.ravel()
    sig = sig.ravel()
    keep = np.abs(mu[:, 1]) < nx / 2
    cov = np.zeros((keep.sum(), 2, 2))
    cov[:, 0, 0] = sig[keep] ** 2
    cov[:, 1, 1] = sig[keep] ** 2
    return amp[keep], mu[keep], cov


def _truth(twist=1.2, radius=3.2, n=3, sigma=1.8):
    c = np.array(
        [
            [radius * np.cos(t), radius * np.sin(t), 0.0]
            for t in np.linspace(0, 2 * np.pi, n, endpoint=False)
        ]
    )
    return c, np.full(n, sigma)


class TestGeometry:
    def test_zero_twist_keeps_every_copy_at_one_azimuth(self):
        c, s = _truth()
        mu, amp, sig = sga.expand_project(c, s, 0.0, RISE, 1, 5)
        for g in range(len(c)):
            assert np.allclose(mu[g, :, 0], mu[g, 0, 0])

    def test_copies_are_spaced_by_the_rise(self):
        c, s = _truth()
        mu, _, _ = sga.expand_project(c, s, 1.2, RISE, 1, 5)
        dz = np.diff(mu[0, :, 1])
        assert np.allclose(dz, RISE)

    def test_projection_amplitude_factor(self):
        """Projecting an isotropic 3D Gaussian scales amplitude by sigma*sqrt(2pi)."""
        c = np.array([[0.0, 0.0, 0.0]])
        s = np.array([2.0])
        _, amp, sig = sga.expand_project(c, s, 0.0, RISE, 1, 0)
        assert amp[0, 0] == pytest.approx(2.0 * gm.SQRT_2PI)
        assert sig[0, 0] == pytest.approx(2.0)

    def test_csym_adds_copies(self):
        c, s = _truth()
        m1, _, _ = sga.expand_project(c, s, 1.2, RISE, 1, 3)
        m2, _, _ = sga.expand_project(c, s, 1.2, RISE, 3, 3)
        assert m2.shape[1] == 3 * m1.shape[1]


class TestScoring:
    def test_perfect_score_when_the_basis_is_the_truth(self):
        """The decisive check: the right structure at the right twist scores 1.

        Without this, a solver can look plausible while being wrong by a
        constant factor -- which is exactly what a mistaken overlap prefactor
        does.
        """
        c, s = _truth()
        mu, amp, sig = sga.expand_project(c, s, 1.2, RISE, 1, NREP)
        target = _iso_target(mu, amp, sig)
        tt = gm.self_energy(target)
        basis = sga.expand_project(c, s, 1.2, RISE, 1, NREP)
        score, _ = sga.best_score(basis, target, tt)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_wrong_twist_scores_lower(self):
        c, s = _truth()
        mu, amp, sig = sga.expand_project(c, s, 1.2, RISE, 1, NREP)
        target = _iso_target(mu, amp, sig)
        tt = gm.self_energy(target)
        right, _ = sga.best_score(
            sga.expand_project(c, s, 1.2, RISE, 1, NREP), target, tt
        )
        wrong, _ = sga.best_score(
            sga.expand_project(c, s, 1.9, RISE, 1, NREP), target, tt
        )
        assert right > wrong

    def test_score_never_exceeds_one(self):
        """A cosine above 1 means the Gram truncation is under-counting <f,f>."""
        c, s = _truth()
        mu, amp, sig = sga.expand_project(c, s, 1.2, RISE, 1, NREP)
        target = _iso_target(mu, amp, sig)
        tt = gm.self_energy(target)
        for twist in (0.6, 1.2, 1.8, 2.4):
            grid = sga.cylindrical_grid(7.0, 2.0)
            score, _ = sga.best_score(
                sga.expand_project(*grid, twist, RISE, 1, NREP), target, tt
            )
            assert score <= 1.0 + 1e-6, f"score {score} at twist {twist}"

    def test_recovers_a_known_twist_with_a_generic_basis(self):
        """End-to-end: a grid that does not contain the truth still finds it."""
        c, s = _truth(twist=1.2, sigma=1.8)
        mu, amp, sig = sga.expand_project(c, s, 1.2, RISE, 1, NREP)
        target = _iso_target(mu, amp, sig)
        tt = gm.self_energy(target)
        grid = sga.cylindrical_grid(7.0, 1.8)
        # The target was cropped to the image, so the model needs the matching
        # envelope; see test_envelope_is_required_for_recovery.
        env = lambda z: (np.abs(z) < NX / 2).astype(float)
        twists = np.arange(0.8, 1.81, 0.05)
        scores = [
            sga.best_score(
                sga.expand_project(*grid, t, RISE, 1, NREP, envelope=env), target, tt
            )[0]
            for t in twists
        ]
        peak = twists[int(np.argmax(scores))]
        assert abs(peak - 1.2) <= 0.051, f"peaked at {peak}"

    def test_envelope_is_required_for_recovery(self):
        """Without an envelope the same scan collapses to the scan edge.

        The model extends uniformly while any real or cropped target is finite,
        and that mismatch is absorbed by the twist. Recording it as a test so
        the envelope is not mistaken for a refinement.
        """
        c, s = _truth(twist=1.2, sigma=1.8)
        mu, amp, sig = sga.expand_project(c, s, 1.2, RISE, 1, NREP)
        target = _iso_target(mu, amp, sig)
        tt = gm.self_energy(target)
        grid = sga.cylindrical_grid(7.0, 1.8)
        twists = np.arange(0.8, 1.81, 0.05)
        scores = [
            sga.best_score(sga.expand_project(*grid, t, RISE, 1, NREP), target, tt)[0]
            for t in twists
        ]
        assert abs(twists[int(np.argmax(scores))] - 1.2) > 0.2


class TestGrid:
    def test_arc_spacing_tracks_sigma(self):
        """Outer shells must not be sampled more coarsely than sigma."""
        centres, _ = sga.cylindrical_grid(8.0, 1.0)
        r = np.hypot(centres[:, 0], centres[:, 1])
        outer = centres[r > 7.0]
        ang = np.sort(np.arctan2(outer[:, 1], outer[:, 0]))
        arc = np.diff(ang).mean() * r.max()
        assert arc < 1.5

    def test_finer_sigma_gives_more_centres(self):
        assert len(sga.cylindrical_grid(8.0, 1.0)[0]) > len(
            sga.cylindrical_grid(8.0, 2.0)[0]
        )


class TestLasso:
    def test_solution_is_non_negative(self):
        rng = np.random.default_rng(0)
        A = rng.normal(0, 1, (12, 12))
        M = A @ A.T + np.eye(12)
        u = rng.normal(0, 1, 12)
        a = sga.nn_lasso(M, u, 0.1)
        assert (a >= 0).all()

    def test_large_penalty_drives_everything_to_zero(self):
        rng = np.random.default_rng(1)
        A = rng.normal(0, 1, (8, 8))
        M = A @ A.T + np.eye(8)
        u = np.abs(rng.normal(0, 1, 8))
        assert np.allclose(sga.nn_lasso(M, u, 1e6), 0.0)


class TestFullFieldSupport:
    """The reconstruction support must cover the field, not just the filament.

    Capping the basis at the filament radius biased the recovered twist upward
    by 0.6-0.7 degrees on real data and made this whole approach look broken.
    Both directions are pinned here so the choice cannot silently regress.
    """

    NY, NX = 24, 64
    RISE = 0.95
    TRUE = 1.2

    def _image(self):
        """Synthetic helical filament: a few Gaussians swept by the screw."""
        radius = 4.0
        centres = np.array(
            [[radius * np.cos(t), radius * np.sin(t), 0.0] for t in (0.0, 2.1, 4.2)]
        )
        grid = (centres, np.full(len(centres), 1.2))
        n_rep = int(self.NX / 2 / self.RISE) + 2
        A = sga.design_matrix(grid, self.TRUE, self.RISE, 1, n_rep, self.NY, self.NX)
        return (A @ np.array([1.0, 0.8, 0.6])).reshape(self.NY, self.NX)

    def _peak(self, radius_px):
        img = self._image()
        grid = sga.full_field_grid(radius_px, spacing_px=2.0, sigma_px=1.2)
        twists = np.arange(0.6, 2.01, 0.1)
        scores = [sga.twist_score(img, grid, t, self.RISE) for t in twists]
        return twists[int(np.argmax(scores))], max(scores)

    def test_full_field_support_recovers_the_twist(self):
        peak, _ = self._peak(self.NY / 2.0)
        assert abs(peak - self.TRUE) <= 0.11, f"peaked at {peak}"

    @pytest.mark.parametrize("radius", [2.5, 3.0])
    def test_support_inside_the_structure_biases_the_twist_upward(self, radius):
        """The failure mode, and it is specifically an UPWARD bias.

        A basis function at radius r sweeps a transverse range proportional to
        r as the symmetry copies rotate, so reproducing a filament of a given
        apparent width needs either large enough radii or more twist per copy.
        A support cut inside the structure removes the first option and the fit
        compensates with the second. The synthetic structure here has radius 4.
        """
        peak, _ = self._peak(radius)
        assert peak > self.TRUE + 0.3, f"expected an upward bias, peaked at {peak}"

    def test_explained_variance_cannot_exceed_one(self):
        img = self._image()
        grid = sga.full_field_grid(self.NY / 2.0)
        for t in (0.8, 1.2, 1.8):
            assert sga.twist_score(img, grid, t, self.RISE) <= 1.0 + 1e-9


class TestElasticnetSolver:
    def test_l1_sparsifies_and_l2_shrinks(self):
        """A penalty that does nothing is the signature of a scale-invariant loss."""
        rng = np.random.default_rng(0)
        A = rng.normal(0, 1, (12, 12))
        M = A @ A.T + np.eye(12)
        u = np.abs(rng.normal(0, 1, 12))
        plain = sga.nn_elasticnet(M, u)
        l1 = sga.nn_elasticnet(M, u, lam1=2.0)
        l2 = sga.nn_elasticnet(M, u, lam2=5.0)
        assert (l1 > 1e-12).sum() < (plain > 1e-12).sum()
        assert np.linalg.norm(l2) < np.linalg.norm(plain)

    def test_solution_is_non_negative(self):
        rng = np.random.default_rng(1)
        A = rng.normal(0, 1, (10, 10))
        M = A @ A.T + np.eye(10)
        assert (sga.nn_elasticnet(M, rng.normal(0, 1, 10), 0.1, 0.1) >= 0).all()

    def test_design_matrix_shape(self):
        grid = sga.full_field_grid(8.0)
        A = sga.design_matrix(grid, 1.2, 0.95, 1, 10, 16, 32)
        assert A.shape == (16 * 32, len(grid[0]))
