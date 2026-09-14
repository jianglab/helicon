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
    """The pipeline contract: ((rec3d, set1, set2), score).

    NX is 128 rather than 64 because a 64 px image at 5 A/px is a 320 A
    filament, and at the twist under test that is less than a quarter of a
    pitch -- not one full crossover. The twist is barely determined by such an
    image for any method, and the narrow basis in particular gets it wrong; see
    TestShortFilamentsAreNotReliablyIndexable, which pins that limit rather than
    hiding it.
    """

    NY, NX, RISE = 24, 128, 0.95

    def _grid(self, truth_sigma_px=1.2):
        radius = 4.0
        centres = np.array(
            [[radius * np.cos(t), radius * np.sin(t), 0.0] for t in (0.0, 2.1, 4.2)]
        )
        return centres, np.full(len(centres), truth_sigma_px)

    def _n_rep(self):
        return int(self.NX / 2 / self.RISE) + 2

    def _image(self, truth_sigma_px=1.2):
        A = sga.design_matrix(
            self._grid(truth_sigma_px),
            1.2,
            self.RISE,
            1,
            self._n_rep(),
            self.NY,
            self.NX,
        )
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

    def test_score_is_a_correlation(self):
        for twist in (0.8, 1.2, 1.8):
            _, score = self._run(twist)
            assert -1.0 - 1e-9 <= score <= 1.0 + 1e-9

    def test_score_is_centred_not_a_plain_cosine(self):
        """The reported score is Pearson correlation, not raw cosine.

        Both vectors are non-negative -- the model by construction, the image
        because it is thresholded -- so their cosine cannot fall much below a
        large floor, and the twist only modulates the score on top of it. That
        floor is what made the search curve look flat.

        Checked on an imperfect model, since a perfect one scores 1 either way:
        the model built at the wrong twist must score lower once centred than it
        does on raw cosine, and the raw cosine must sit implausibly high for two
        vectors that disagree.
        """
        b = self._image().ravel()
        A = sga.design_matrix(
            self._grid(), 1.9, self.RISE, 1, self._n_rep(), self.NY, self.NX
        )
        pred = A @ np.array([1.0, 0.8, 0.6])
        raw = float(pred @ b / (np.linalg.norm(pred) * np.linalg.norm(b)))
        centred = sga.correlation_score(pred, b)
        assert centred < raw
        assert raw > 0.5  # the floor this is meant to remove

    def test_correlation_is_invariant_to_model_scale(self):
        rng = np.random.default_rng(0)
        p, t = rng.random(500), rng.random(500)
        assert sga.correlation_score(p, t) == pytest.approx(
            sga.correlation_score(3.7 * p, t)
        )

    def test_prefers_the_true_twist(self):
        _, right = self._run(1.2)
        for wrong_twist in (0.8, 1.6, 1.9):
            _, wrong = self._run(wrong_twist)
            assert right > wrong, f"twist {wrong_twist} outscored the truth"

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


class TestShortFilamentsAreNotReliablyIndexable:
    """A known limit, recorded rather than hidden.

    Twist discrimination needs crossovers to look at. At twist 1.2 and rise
    4.75 A the pitch is 1425 A, so half a pitch -- one crossover -- is 712 A. A
    64 px image at 5 A/px is 320 A, under a quarter of a pitch, and the twist is
    barely determined by it.

    The narrow basis is worse there than the broad one it replaced: on a
    synthetic structure of known twist it returns 1.4, 1.9 or 2.2 depending on
    the width the structure was drawn with, where the old sigma 5 A basis
    returned 1.2 for two of the three. That is the price of the sharper peak,
    and it is paid where the answer was weakly determined anyway -- at 128 px
    (640 A) the narrow basis is correct at every width tested.

    This test asserts the length threshold, not the wrong answer, so it keeps
    documenting the limit if the solver later improves.
    """

    RISE_PX, TRUE, APIX = 0.95, 1.2, 5.0

    def _image(self, ny, nx, truth_sigma_px=1.2):
        radius = 4.0
        centres = np.array(
            [[radius * np.cos(t), radius * np.sin(t), 0.0] for t in (0.0, 2.1, 4.2)]
        )
        grid = (centres, np.full(len(centres), truth_sigma_px))
        n_rep = int(nx / 2 / self.RISE_PX) + 2
        A = sga.design_matrix(grid, self.TRUE, self.RISE_PX, 1, n_rep, ny, nx)
        return (A @ np.array([1.0, 0.8, 0.6])).reshape(ny, nx)

    def _best_twist(self, ny, nx, truth_sigma_px):
        twists = [0.8, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2]
        image = self._image(ny, nx, truth_sigma_px)
        scores = [
            sga.gauss_analytic_reconstruct(
                image,
                1.0,
                t,
                self.RISE_PX,
                reconstruct_length_3d_pixel=4,
                reconstruct_diameter_3d_pixel=ny,
                target_apix2d=self.APIX,
            )[1]
            for t in twists
        ]
        return twists[int(np.argmax(scores))]

    @pytest.mark.parametrize("truth_sigma_px", [0.8, 1.2, 2.0])
    def test_recovers_the_twist_when_a_crossover_is_visible(self, truth_sigma_px):
        """128 px at 5 A is 640 A, most of a half pitch. This must work."""
        assert self._best_twist(24, 128, truth_sigma_px) == pytest.approx(self.TRUE)

    def test_an_image_shorter_than_a_quarter_pitch_is_not_required_to_work(self):
        """Pins the threshold: 320 A is below what the method can index.

        Asserted as a property of the geometry so this stays meaningful if the
        solver later handles the short case -- it does not assert failure.
        """
        half_pitch_angstrom = 360.0 / self.TRUE * (self.RISE_PX * self.APIX) / 2
        assert 64 * self.APIX < half_pitch_angstrom / 2
        assert 128 * self.APIX > half_pitch_angstrom * 0.8


class TestDefaultBasisIsNotTooFine:
    """The basis is narrow across the filament and wide along it, on purpose.

    Those two directions are governed by different requirements and used to be
    conflated, because the basis was isotropic.

    ACROSS, narrow is what keeps the joint answer. Transverse width lets the fit
    smear a wrong twist into a plausible image, so a broad basis scores well
    everywhere. Measured on the 32 good classes prepared as the tab prepares
    them, the old sigma 5 A basis puts the joint peak at 1.15 with 11/32 single
    images correct; sigma 3.5 A puts it on 1.20 with 19/32.

    ALONG, wide is what keeps the answer honest. Screw copies sit one rise
    apart, so a basis narrow in z leaves them as a comb rather than continuous
    density. Pulling sigma_z down to match sigma scores BETTER on the class
    averages and gets a synthetic structure of known twist wrong in 3 of 9
    cases, so the synthetic test decides it.

    Shipping sigma 4 A isotropic once moved the joint peak off the truth while
    improving per-image counts, which is why the per-image count is never
    sufficient on its own.
    """

    TYPICAL_RISE_ANGSTROM = 4.75

    def test_axial_sigma_spans_a_typical_rise(self):
        """Copies must overlap into continuous density, not sit as a comb.

        This is a constraint on the AXIAL width only: copies are spaced by the
        rise along z, so that is the direction in which they have to merge.
        """
        assert sga.DEFAULT_SIGMA_Z_ANGSTROM >= self.TYPICAL_RISE_ANGSTROM

    def test_transverse_sigma_is_narrow_enough_to_sharpen_the_peak(self):
        """Across the filament the basis samples rather than describes.

        The bound is half the spacing because that is what the default used to
        be, and widening back to it measurably flattens the peak.
        """
        assert sga.DEFAULT_SIGMA_ANGSTROM < sga.DEFAULT_SPACING_ANGSTROM / 2.0

    def test_basis_is_anisotropic(self):
        assert sga.DEFAULT_SIGMA_Z_ANGSTROM > sga.DEFAULT_SIGMA_ANGSTROM

    def test_rendering_is_wider_than_the_fit(self):
        """A narrow search basis renders a spiky map, so the map uses its own."""
        assert sga.DEFAULT_RENDER_SIGMA_ANGSTROM > sga.DEFAULT_SIGMA_ANGSTROM

    def test_default_spacing_is_coarser_than_sigma(self):
        assert sga.DEFAULT_SPACING_ANGSTROM > sga.DEFAULT_SIGMA_ANGSTROM

    def test_default_basis_stays_small(self):
        """197 functions at the validated geometry; a fine basis is ~5x that."""
        grid = sga.full_field_grid(
            16.0, sga.DEFAULT_SPACING_ANGSTROM / 5.0, sga.DEFAULT_SIGMA_ANGSTROM / 5.0
        )
        assert len(grid[0]) < 400

    def test_grid_is_a_single_z_plane(self):
        """Adding z planes was tried against the artefacts and did not help."""
        centres, _ = sga.full_field_grid(8.0, 2.0, 1.2)
        assert np.unique(centres[:, 2]).size == 1


class TestRenderedVolumeIsAxiallyContinuous:
    """Guards the banding fix.

    The basis lies in one z plane, so a slab rendered without the screw copies
    is a single Gaussian bump in z. helicon.apply_helical_symmetry then repeats
    that bump every rise, and with a sub-voxel rise (0.95 px here) the repetition
    beats against the pixel lattice -- measured at a 20 px period carrying 39% of
    the axial power, which is visible as banding along the reconstruction.
    """

    def test_slab_is_filled_along_z(self):
        img = np.zeros((32, 128))
        img[14:18, 20:108] = 1.0
        (rec3d, _, _), _ = sga.gauss_analytic_reconstruct(
            img,
            1.0,
            1.2,
            0.95,
            reconstruct_diameter_2d_pixel=17,
            reconstruct_length_3d_pixel=4,
            reconstruct_diameter_3d_pixel=32,
            target_apix2d=5.0,
        )
        z = rec3d.sum(axis=(1, 2))
        assert z.min() > 0.5 * z.max(), f"z profile not continuous: {z}"

    def test_without_copies_the_slab_is_a_single_bump(self):
        """The failure mode itself, so the mechanism stays documented."""
        centres = np.array([[0.0, 3.0, 0.0], [0.0, -3.0, 0.0]])
        sigmas = np.full(2, 1.2)
        amps = np.ones(2)
        bump = sga._render_volume(
            centres, sigmas, amps, 1.0, 8, 32, 32, twist_deg=0.0, rise_px=0.0
        )
        z = bump.sum(axis=(1, 2))
        assert z.min() < 0.1 * z.max()

    def test_copies_do_not_change_the_score(self):
        """Filling the slab is a rendering fix; the fit must be untouched."""
        rng = np.random.default_rng(0)
        img = rng.random((32, 128))
        _, a = sga.gauss_analytic_reconstruct(
            img,
            1.0,
            1.2,
            0.95,
            reconstruct_diameter_2d_pixel=17,
            reconstruct_length_3d_pixel=4,
            reconstruct_diameter_3d_pixel=32,
            target_apix2d=5.0,
        )
        _, b = sga.gauss_analytic_reconstruct(
            img,
            1.0,
            1.2,
            0.95,
            reconstruct_diameter_2d_pixel=17,
            reconstruct_length_3d_pixel=1,
            reconstruct_diameter_3d_pixel=32,
            target_apix2d=5.0,
        )
        assert a == pytest.approx(b, rel=1e-12)
