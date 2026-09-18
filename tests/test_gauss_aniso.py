"""Tests for the free anisotropic Gaussian helical solver."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helicon.webApps.lib import gauss_mixture as gm
from helicon.webApps.lib import solver_gauss_analytic as sga
from helicon.webApps.lib import solver_gauss_aniso as sa

RISE = 4.75 / 4.944


@pytest.fixture(autouse=True)
def _f64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _iso_raw(n, sigma):
    """raw_L whose softplus diagonal gives an isotropic Sigma = sigma^2 I."""
    raw = torch.zeros(n, 3, 3)
    inv_sp = float(np.log(np.exp(sigma - 1e-3) - 1))
    for i in range(3):
        raw[:, i, i] = inv_sp
    return raw


class TestReducesToIsotropic:
    """The anisotropic path must agree with the committed isotropic one."""

    def test_projection_matches_exactly(self):
        centres = np.array([[3.2, 0.0, 0.0], [1.0, 2.5, 0.0], [-2.0, 1.0, 0.0]])
        sigma = 1.8
        mu_i, amp_i, sig_i = sga.expand_project(
            centres, np.full(3, sigma), 1.2, RISE, 1, 12
        )
        mu_a, amp_a, cov_a = sa.expand_project(
            torch.ones(3),
            torch.tensor(centres),
            _iso_raw(3, sigma),
            torch.tensor(1.2),
            RISE,
            1,
            12,
        )
        assert np.abs(mu_a.numpy() - mu_i).max() < 1e-12
        assert np.abs(amp_a.numpy() - amp_i).max() / np.abs(amp_i).max() < 1e-12
        assert np.abs(cov_a[..., 0, 0].numpy() - sig_i**2).max() < 1e-12
        assert np.abs(cov_a[..., 0, 1].numpy()).max() < 1e-12

    def test_overlap_matches_the_numpy_implementation(self):
        rng = np.random.default_rng(0)
        a1, a2 = rng.uniform(0.5, 2, 2)
        m1, m2 = rng.normal(0, 3, 2), rng.normal(0, 3, 2)
        s1, s2 = rng.uniform(1.0, 3, 2)
        c1, c2 = np.eye(2) * s1**2, np.eye(2) * s2**2
        ref = gm.gen_overlap(a1, m1, c1, a2, m2, c2)
        got = sa._overlap(
            torch.tensor(a1),
            torch.tensor(m1),
            torch.tensor(c1),
            torch.tensor(a2),
            torch.tensor(m2),
            torch.tensor(c2),
        )
        assert float(got) == pytest.approx(float(ref), rel=1e-12)


class TestScoreIsBounded:
    """A cosine cannot exceed 1; the axial truncation must not let it."""

    def _target(self, centres, sigma, nrep):
        mu, amp, sig = sga.expand_project(
            centres, np.full(len(centres), sigma), 1.2, RISE, 1, nrep
        )
        m = mu.reshape(-1, 2)
        a = amp.ravel()
        s = sig.ravel()
        cov = np.zeros((len(s), 2, 2))
        cov[:, 0, 0] = s**2
        cov[:, 1, 1] = s**2
        return (torch.tensor(a), torch.tensor(m), torch.tensor(cov)), gm.self_energy(
            (a, m, cov)
        )

    @pytest.mark.parametrize("sigma_model", [1.0, 3.0, 6.0, 12.0])
    def test_elongated_model_cannot_exceed_one(self, sigma_model):
        """Growing a Gaussian along the axis must not inflate the score.

        With a fixed dk the far-separated self-overlaps of a long Gaussian fall
        outside the truncation, so <f,f> loses mass the cross term still
        counts and the cosine climbs above 1 -- a free reward for elongating
        that gradient descent reliably finds (scores of 1.10 to 1.18 were
        observed before axial_dk was derived from the fitted covariance).
        """
        nrep = 40
        centres = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        target, tt = self._target(centres, 1.8, nrep)
        model = sa.expand_project(
            torch.ones(2),
            torch.tensor(centres),
            _iso_raw(2, sigma_model),
            torch.tensor(1.2),
            RISE,
            1,
            nrep,
        )
        s = float(sa.score(model, target, torch.tensor(tt), RISE))
        assert s <= 1.0 + 1e-6, f"score {s} at sigma {sigma_model}"

    def test_axial_dk_grows_with_the_covariance(self):
        nrep = 30
        centres = np.array([[3.0, 0.0, 0.0]])
        narrow = sa.expand_project(
            torch.ones(1),
            torch.tensor(centres),
            _iso_raw(1, 1.0),
            torch.tensor(1.2),
            RISE,
            1,
            nrep,
        )
        wide = sa.expand_project(
            torch.ones(1),
            torch.tensor(centres),
            _iso_raw(1, 6.0),
            torch.tensor(1.2),
            RISE,
            1,
            nrep,
        )
        assert sa.axial_dk(wide, RISE) > sa.axial_dk(narrow, RISE)

    def test_axial_dk_is_capped_by_the_number_of_copies(self):
        nrep = 3
        centres = np.array([[1.0, 0.0, 0.0]])
        model = sa.expand_project(
            torch.ones(1),
            torch.tensor(centres),
            _iso_raw(1, 50.0),
            torch.tensor(1.2),
            RISE,
            1,
            nrep,
        )
        assert sa.axial_dk(model, RISE) <= model[1].shape[1] - 1


class TestGeometry:
    def test_covariance_determinant_is_rotation_invariant(self):
        """det(R Sigma R^T) = det(Sigma); the projection factor relies on it."""
        raw = torch.zeros(1, 3, 3)
        raw[0] = torch.tensor([[0.4, 0.0, 0.0], [0.7, 0.2, 0.0], [-0.3, 0.5, 0.1]])
        L = sa._chol(raw)
        Sigma = (L @ L.transpose(-1, -2))[0]
        th = 0.7
        R = torch.tensor(
            [
                [np.cos(th), -np.sin(th), 0.0],
                [np.sin(th), np.cos(th), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert float(torch.det(R @ Sigma @ R.T)) == pytest.approx(
            float(torch.det(Sigma)), rel=1e-12
        )

    def test_cholesky_is_positive_definite(self):
        rng = np.random.default_rng(2)
        raw = torch.tensor(rng.normal(0, 2, (5, 3, 3)))
        L = sa._chol(raw)
        Sigma = L @ L.transpose(-1, -2)
        for i in range(5):
            assert np.all(np.linalg.eigvalsh(Sigma[i].numpy()) > 0)

    def test_envelope_scales_the_amplitudes(self):
        centres = np.array([[3.0, 0.0, 0.0]])
        args = (
            torch.ones(1),
            torch.tensor(centres),
            _iso_raw(1, 2.0),
            torch.tensor(1.2),
            RISE,
            1,
            10,
        )
        plain = sa.expand_project(*args)
        damped = sa.expand_project(*args, envelope=lambda z: torch.zeros_like(z))
        assert float(plain[1].abs().sum()) > 0
        assert float(damped[1].abs().sum()) == pytest.approx(0.0)


class TestMemoryIsBounded:
    """Guards against the failure mode that once exhausted the machine's RAM.

    The self-energy sum broadcasts to (G, G, P, 2, 2) and autograd retains
    every intermediate, so an unbounded pair count is not a slow path but an
    out-of-memory kill.
    """

    def test_axial_dk_is_capped_regardless_of_the_fitted_width(self):
        """A grown Gaussian must not be able to widen the truncation forever."""
        nrep = 200
        centres = np.array([[1.0, 0.0, 0.0]])
        model = sa.expand_project(
            torch.ones(1),
            torch.tensor(centres),
            _iso_raw(1, 400.0),
            torch.tensor(1.2),
            RISE,
            1,
            nrep,
        )
        assert sa.axial_dk(model, RISE) <= sa.MAX_AXIAL_DK

    def test_cap_is_honoured_over_the_five_sigma_request(self):
        nrep = 200
        centres = np.array([[1.0, 0.0, 0.0]])
        model = sa.expand_project(
            torch.ones(1),
            torch.tensor(centres),
            _iso_raw(1, 80.0),
            torch.tensor(1.2),
            RISE,
            1,
            nrep,
        )
        assert sa.axial_dk(model, RISE, max_dk=4) == 4

    def test_row_chunking_engages_for_large_problems(self):
        """A configuration that would allocate ~1 GB must be split into chunks."""
        G, P = 40, 18769
        assert sa.self_energy_bytes(G, P) > 500 << 20
        chunk = sa._row_chunk(G, P, 8)
        assert chunk < G, "large problem must be chunked, not taken in one block"
        per_chunk = chunk * G * P * 4 * 8
        assert per_chunk <= sa.SELF_ENERGY_CHUNK_BYTES

    def test_small_problems_are_not_chunked(self):
        G, P = 6, 500
        assert sa._row_chunk(G, P, 8) == G

    def test_chunked_result_matches_unchunked(self):
        """Chunking must be exact, not approximate."""
        nrep = 12
        centres = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [-2.0, 1.0, 0.0]])
        model = sa.expand_project(
            torch.ones(3),
            torch.tensor(centres),
            _iso_raw(3, 1.8),
            torch.tensor(1.2),
            RISE,
            1,
            nrep,
        )
        ref = sa.SELF_ENERGY_CHUNK_BYTES
        try:
            full = float(sa.self_energy(model, 6))
            sa.SELF_ENERGY_CHUNK_BYTES = 1  # force one row per chunk
            chunked = float(sa.self_energy(model, 6))
        finally:
            sa.SELF_ENERGY_CHUNK_BYTES = ref
        assert chunked == pytest.approx(full, rel=1e-12)

    def test_gradients_survive_checkpointed_chunking(self):
        nrep = 10
        centres = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        amp = torch.ones(2, requires_grad=True)
        raw = _iso_raw(2, 1.8).requires_grad_(True)
        ref = sa.SELF_ENERGY_CHUNK_BYTES
        try:
            sa.SELF_ENERGY_CHUNK_BYTES = 1
            model = sa.expand_project(
                amp, torch.tensor(centres), raw, torch.tensor(1.2), RISE, 1, nrep
            )
            sa.self_energy(model, 4).backward()
        finally:
            sa.SELF_ENERGY_CHUNK_BYTES = ref
        assert amp.grad is not None and torch.isfinite(amp.grad).all()
        assert raw.grad is not None and torch.isfinite(raw.grad).all()
