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


class TestSolverEntryPoint:
    """The pipeline contract: ((rec3d, set1, set2), score)."""

    NY, NX, RISE = 24, 64, 0.95

    def _image(self):
        radius = 4.0
        centres = np.array(
            [[radius * np.cos(t), radius * np.sin(t), 0.0] for t in (0.0, 2.1, 4.2)]
        )
        grid = (centres, np.full(len(centres), 1.2))
        n_rep = int(self.NX / 2 / self.RISE) + 2
        A = sga.design_matrix(grid, 1.2, self.RISE, 1, n_rep, self.NY, self.NX)
        return (A @ np.array([1.0, 0.8, 0.6])).reshape(self.NY, self.NX)

    def _run(self, twist, **kw):
        return sga.gauss_analytic_reconstruct(
            self._image(),
            1.0,
            twist,
            self.RISE,
            reconstruct_length_3d_pixel=4,
            reconstruct_diameter_3d_pixel=self.NY,
            target_apix2d=5.0,
            **kw,
        )

    def test_returns_the_pipeline_contract(self):
        (rec3d, s1, s2), score = self._run(1.2)
        assert isinstance(score, float)
        assert s1 is None and s2 is None
        assert rec3d.shape == (4, self.NY, self.NY)
        assert rec3d.dtype == np.float32

    def test_score_is_a_cosine(self):
        for twist in (0.8, 1.2, 1.8):
            _, score = self._run(twist)
            assert 0.0 <= score <= 1.0 + 1e-9

    def test_prefers_the_true_twist(self):
        _, right = self._run(1.2)
        _, wrong = self._run(1.9)
        assert right > wrong

    def test_reconstruction_is_non_negative(self):
        (rec3d, _, _), _ = self._run(1.2)
        assert (rec3d >= 0).all()

    def test_tolerates_unknown_keyword_arguments(self):
        """The pipeline passes options this solver has no use for."""
        _, score = self._run(
            1.2,
            sym_oversample=300,
            interpolation="linear",
            fsc_test=0,
            thresh_fraction=-1,
            tilt_degree=0.0,
        )
        assert 0.0 <= score <= 1.0 + 1e-9

    @pytest.mark.parametrize("apix", [2.5, 5.0, 10.0])
    def test_basis_is_the_same_physical_grid_at_any_pixel_size(self, apix):
        """Defaults are in angstroms, so the basis tracks the structure, not the sampling.

        Radius and spacing both scale as 1/apix, so a fixed physical field
        yields the same number of basis functions however the image is sampled.
        That is the point of expressing the defaults in angstroms rather than
        pixels: they were tuned at 5 A/px and must not silently change meaning
        on data binned differently.
        """
        field_angstrom = 80.0
        grid = sga.full_field_grid(
            field_angstrom / apix,
            sga.DEFAULT_SPACING_ANGSTROM / apix,
            sga.DEFAULT_SIGMA_ANGSTROM / apix,
        )
        reference = sga.full_field_grid(
            field_angstrom / 5.0,
            sga.DEFAULT_SPACING_ANGSTROM / 5.0,
            sga.DEFAULT_SIGMA_ANGSTROM / 5.0,
        )
        assert len(grid[0]) == len(reference[0])

    def test_alpha_and_l1_ratio_are_honoured(self):
        """Those sliders exist in the tab; they must not be silently inert."""
        _, plain = self._run(1.2)
        _, heavy = self._run(1.2, algorithm={"alpha": 5.0, "l1_ratio": 1.0})
        assert heavy != plain


class TestDesignMatrixCulling:
    def test_culled_matches_a_dense_evaluation(self):
        """Footprint culling is an optimisation, not an approximation to notice."""
        ny, nx, rise, n_rep = 16, 32, 0.95, 8
        grid = sga.full_field_grid(8.0, spacing_px=2.0, sigma_px=1.2)
        A = sga.design_matrix(grid, 1.3, rise, 1, n_rep, ny, nx, cutoff_sigma=6.0)
        mu, amp, sig = sga.expand_project(grid[0], grid[1], 1.3, rise, 1, n_rep)
        yy, xx = np.meshgrid(
            np.arange(ny) - ny / 2.0, np.arange(nx) - nx / 2.0, indexing="ij"
        )
        pts = np.stack([yy.ravel(), xx.ravel()], -1)
        d2 = ((pts[None, :, None, :] - mu[:, None, :, :]) ** 2).sum(-1)
        dense = (amp[:, None, :] * np.exp(-0.5 * d2 / sig[:, None, :] ** 2)).sum(-1).T
        assert np.abs(A - dense).max() / dense.max() < 1e-5

    def test_columns_are_one_per_basis_function(self):
        grid = sga.full_field_grid(8.0)
        A = sga.design_matrix(grid, 1.2, 0.95, 1, 6, 16, 32)
        assert A.shape == (16 * 32, len(grid[0]))
        assert (A >= 0).all()


class TestPipelineDispatch:
    def test_gauss_analytic_is_routed(self):
        from helicon.webApps.lib import denovo3d_pipeline as pipeline

        assert pipeline.gauss_analytic_reconstruct is sga.gauss_analytic_reconstruct


class TestRobustDefaults:
    """Defaults must hold up on boxes and pixel sizes other than the tuned one."""

    def test_support_reaches_twice_the_filament_radius(self):
        """Less than that biases the twist upward; 2.0 was the measured threshold."""
        assert sga.support_radius(256, 17) >= 2.0 * 17 / 2.0

    def test_support_never_exceeds_the_image(self):
        """Basis functions outside the image explain nothing."""
        for ny, diam in ((32, 17), (32, 100), (64, 60)):
            assert sga.support_radius(ny, diam) <= ny / 2.0 + 1e-9

    def test_tight_box_is_unchanged_from_the_validated_setting(self):
        """The 32 px EMPIAR-10940 box must still use the validated radius of 16."""
        assert sga.support_radius(32, 17) == 16.0
        assert sga.support_radius(32, 32) == 16.0

    def test_unknown_diameter_falls_back_to_the_field(self):
        assert sga.support_radius(64, None) == 32.0
        assert sga.support_radius(64, 0) == 32.0

    def test_large_box_does_not_explode_the_basis(self):
        """The basis grows as radius^2; an uncapped 128 px radius is ~12800
        functions and a multi-gigabyte design matrix."""
        uncapped = len(sga.full_field_grid(128.0, 2.0, 1.2)[0])
        assert uncapped > 10000
        _, _, n = sga.fit_spacing_to_budget(128.0, 2.0, 1.2)
        assert n <= sga.MAX_BASIS

    def test_budget_leaves_a_small_basis_alone(self):
        spacing, sigma, n = sga.fit_spacing_to_budget(16.0, 2.0, 1.2)
        assert (spacing, sigma) == (2.0, 1.2)
        assert n == len(sga.full_field_grid(16.0, 2.0, 1.2)[0])

    def test_budget_keeps_sigma_proportional_to_spacing(self):
        """Coarsening must not leave a spiky basis that cannot represent density."""
        spacing, sigma, _ = sga.fit_spacing_to_budget(128.0, 2.0, 1.2)
        assert sigma / spacing == pytest.approx(1.2 / 2.0, rel=1e-6)

    def test_solver_runs_on_a_large_box_within_budget(self):
        rng = np.random.default_rng(0)
        img = rng.random((96, 128))
        (rec3d, _, _), score = sga.gauss_analytic_reconstruct(
            img,
            1.0,
            1.2,
            0.95,
            reconstruct_diameter_2d_pixel=0,
            reconstruct_length_3d_pixel=4,
            reconstruct_diameter_3d_pixel=96,
            target_apix2d=5.0,
        )
        assert 0.0 <= score <= 1.0 + 1e-9
        assert rec3d.shape == (4, 96, 96)
