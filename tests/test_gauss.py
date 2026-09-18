import numpy as np
import pytest

# torch is an optional dependency (no Intel-mac wheels for the supported Python
# versions), so skip rather than fail collection where it is absent -- an error
# here would show up in every test run on a machine that never wanted it.
torch = pytest.importorskip("torch")

import helicon


def test_AnisotropicGuassians():
    nz = np.random.randint(32, 64) // 2 * 2
    ny = np.random.randint(32, 64) // 2 * 2
    nx = np.random.randint(32, 64) // 2 * 2
    apix = np.random.uniform(0.5, 1.5)
    twist = np.random.uniform(-180, 180)
    rise = np.random.uniform(1, nz / 10) * apix
    csym = np.random.randint(1, 6 + 1)
    min_dist_sigma = 3

    n = np.random.randint(1, 10)

    gaussians_list = []
    for i in range(n):
        amplitude = np.random.uniform(0, 10)
        min_dim = np.min([nx, ny, nz])
        center = np.random.uniform(-0.5, 0.5, size=3) * min_dim * 0.7 * apix
        sigma = np.random.uniform(1, max(2, min_dim * 0.05)) * apix
        quaternion = np.random.uniform(-1, 1, size=4)
        quaternion /= np.linalg.norm(quaternion)
        g = helicon.AnisotropicGaussian(amplitude, center, sigma, quaternion)
        print(f"{i+1}: {g}")
        gaussians_list += [g]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussians = helicon.AnisotropicGaussianSet.from_list(gaussians_list, device=device)
    gaussians = gaussians.mask(
        nx=nx, ny=ny, nz=nz, apix=apix, min_dist_sigma=min_dist_sigma
    )
    gaussians = gaussians.apply_helical_symmetry(
        twist=twist,
        rise=rise,
        csym=csym,
        zmin=-nz // 2 * apix,
        zmax=nz // 2 * apix,
        min_dist_sigma=min_dist_sigma,
    )
    gaussians = gaussians.mask(
        nx=nx, ny=ny, nz=nz, apix=apix, min_dist_sigma=min_dist_sigma
    )
    proj1 = gaussians.sample_volume(nx, ny, nz, apix, batch_size=100).sum(axis=0)
    proj2 = gaussians.projection_z(nx, ny, apix)
    err = (proj1 - proj2).abs()
    err_power_ratio = torch.linalg.norm(err) / (
        (torch.linalg.norm(proj2) + torch.linalg.norm(proj2)) / 2
    )
    print(f"{err.min()=}\t{err.max()=}\t{err.median()=}\t{err_power_ratio=}")

    assert err_power_ratio < 1e-3


def test_IsotropicGuassians():
    nz = np.random.randint(32, 64) // 2 * 2
    ny = np.random.randint(32, 64) // 2 * 2
    nx = np.random.randint(32, 64) // 2 * 2
    apix = np.random.uniform(0.5, 1.5)
    twist = np.random.uniform(-180, 180)
    rise = np.random.uniform(1, nz / 10) * apix
    csym = np.random.randint(1, 6 + 1)
    min_dist_sigma = 3

    n = np.random.randint(1, 10)

    gaussians_list_iso = []
    gaussians_list_aniso = []
    for i in range(n):
        amplitude = np.random.uniform(0, 10)
        min_dim = np.min([nx, ny, nz])
        center = np.random.uniform(-0.5, 0.5, size=3) * min_dim * 0.5 * apix
        sigma = np.random.uniform(1, max(2, min_dim * 0.05)) * apix
        g = helicon.IsotropicGaussian(amplitude, center, sigma)
        print(f"{i+1}: {g}")
        gaussians_list_iso += [g]

        quaternion = np.random.uniform(-1, 1, size=4)
        quaternion /= np.linalg.norm(quaternion)
        g = helicon.AnisotropicGaussian(amplitude, center, [sigma] * 3, quaternion)
        print(f"{i+1}: {g}")
        gaussians_list_aniso += [g]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gaussians_iso = helicon.IsotropicGaussianSet.from_list(
        gaussians_list_iso, device=device
    )
    gaussians_aniso = helicon.AnisotropicGaussianSet.from_list(
        gaussians_list_aniso, device=device
    )

    projs = []
    for gaussians in [gaussians_iso, gaussians_aniso]:
        gaussians = gaussians.apply_helical_symmetry(
            twist=twist,
            rise=rise,
            csym=csym,
            zmin=-nz // 2 * apix,
            zmax=nz // 2 * apix,
            min_dist_sigma=min_dist_sigma,
        )
        gaussians = gaussians.mask(
            nx=nx, ny=ny, nz=nz, apix=apix, min_dist_sigma=min_dist_sigma
        )
        projs += [gaussians.projection_z(nx, ny, apix)]
        projs += [gaussians.sample_volume(nx, ny, nz, apix, batch_size=100).sum(axis=0)]

    projs_pairs = []
    for i in range(len(projs)):
        for j in range(i + 1, len(projs)):
            proj1, proj2 = projs[i], projs[j]
            err = (proj1 - proj2).abs()
            err_power_ratio = torch.linalg.norm(err) / (
                (torch.linalg.norm(proj2) + torch.linalg.norm(proj2)) / 2
            )
            proj_ratio = proj1.max() / proj2.max() if proj2.max() != 0 else float("inf")
            max_proj1 = proj1.max()
            max_proj2 = proj2.max()
            max_ratio = max_proj1 / max_proj2 if max_proj2 != 0 else float("inf")
            assert err_power_ratio < 1e-3


def _numerical_projection_z(amp, center, cov, nx, ny, apix, n=6001, span=100.0):
    """Brute-force line integral along z, as an independent reference."""
    inv = np.linalg.inv(cov)
    mu = np.asarray(center, dtype=np.float64)
    xs = (np.arange(nx) - nx // 2) * apix
    ys = (np.arange(ny) - ny // 2) * apix
    zs = np.linspace(-span, span, n)
    dz = zs[1] - zs[0]
    ref = np.zeros((ny, nx))
    for iy, yv in enumerate(ys):
        for ix, xv in enumerate(xs):
            d = np.stack(
                [
                    np.full_like(zs, xv - mu[0]),
                    np.full_like(zs, yv - mu[1]),
                    zs - mu[2],
                ],
                axis=1,
            )
            ref[iy, ix] = (
                amp * np.exp(-0.5 * np.einsum("ni,ij,nj->n", d, inv, d)).sum() * dz
            )
    return ref / apix


def _cov_of(gset):
    from helicon.lib.gauss import quaternion_to_rotation_matrix

    R = quaternion_to_rotation_matrix(gset.quaternions)
    return (
        (R @ torch.diag_embed(gset.sigmas**2) @ R.transpose(1, 2))[0].double().numpy()
    )


def test_anisotropic_projection_z_prefactor():
    """projection_z must use the Schur-complement prefactor.

    Integrating a 3D Gaussian along z gives a 2D Gaussian with prefactor
    sqrt(2*pi * det(Sigma)/det(Sigma_xy)).  The simpler sqrt(2*pi * Sigma_zz)
    is only valid without z cross-terms, so a *rotated* anisotropic Gaussian is
    exactly the case that catches the difference (~90% error).
    """
    amp, center, sigma = 2.0, [0.7, -1.3, 0.5], [8.0, 3.0, 2.0]
    q = np.array([0.804, 0.322, 0.483, 0.117])
    q /= np.linalg.norm(q)
    gset = helicon.AnisotropicGaussianSet.from_list(
        [helicon.AnisotropicGaussian(amp, center, sigma, list(q))], device="cpu"
    )
    cov = _cov_of(gset)
    assert abs(cov[0, 2]) > 1e-3, "test needs genuine z cross-terms"

    nx = ny = 24
    apix = 2.0
    got = gset.projection_z(nx=nx, ny=ny, apix=apix).numpy()
    ref = _numerical_projection_z(amp, center, cov, nx, ny, apix)
    assert np.abs(got - ref).max() / ref.max() < 1e-4


def test_isotropic_projection_z_unchanged_by_prefactor_fix():
    """For an isotropic Gaussian det(Sigma)/det(Sigma_xy) == Sigma_zz, so the
    fix must leave the isotropic result exactly as it was."""
    amp, center, sigma = 1.5, [1.0, -2.0, 0.5], 3.0
    gset = helicon.IsotropicGaussianSet.from_list(
        [helicon.IsotropicGaussian(amp, center, sigma)], device="cpu"
    )
    nx = ny = 24
    apix = 2.0
    got = gset.projection_z(nx=nx, ny=ny, apix=apix).numpy()
    cov = np.eye(3) * sigma**2
    ref = _numerical_projection_z(amp, center, cov, nx, ny, apix)
    assert np.abs(got - ref).max() / ref.max() < 1e-4


class TestProjectionFootprint:
    """Each gaussian is accumulated over its own box, not against every pixel.

    Evaluating the whole grid per gaussian is what made a symmetry-expanded set
    unusable: 60k gaussians onto 64x526 took 48 s that way and 0.28 s this way,
    for the same picture. The default cutoff is chosen so the difference stays
    at the level of float rounding.
    """

    def _iso(self, n=150, seed=0):
        torch.manual_seed(seed)
        return helicon.IsotropicGaussianSet(
            torch.rand(n) + 0.5, (torch.rand(n, 3) - 0.5) * 60, torch.rand(n) * 3 + 2
        )

    def _aniso(self, n=150, seed=0):
        torch.manual_seed(seed)
        q = torch.randn(n, 4)
        return helicon.AnisotropicGaussianSet(
            torch.rand(n) + 0.5,
            (torch.rand(n, 3) - 0.5) * 60,
            torch.rand(n, 3) * 3 + 2,
            q / q.norm(dim=-1, keepdim=True),
        )

    @pytest.mark.parametrize("kind", ["iso", "aniso"])
    def test_a_generous_cutoff_reproduces_the_exact_projection(self, kind):
        g = self._iso() if kind == "iso" else self._aniso()
        exact = g.projection_z(nx=96, ny=64, apix=1.0, cutoff_sigma=None)
        fast = g.projection_z(nx=96, ny=64, apix=1.0, cutoff_sigma=6.0)
        assert fast.shape == exact.shape
        assert float((fast - exact).abs().max() / exact.std()) < 1e-4

    @pytest.mark.parametrize("kind", ["iso", "aniso"])
    def test_the_default_cutoff_is_close_enough(self, kind):
        g = self._iso() if kind == "iso" else self._aniso()
        exact = g.projection_z(nx=96, ny=64, apix=1.0, cutoff_sigma=None)
        fast = g.projection_z(nx=96, ny=64, apix=1.0)
        assert float((fast - exact).abs().max() / exact.std()) < 1e-3

    def test_a_tight_cutoff_is_visibly_worse(self):
        """Guards the direction: if a tight cutoff were as good as a loose one,
        the footprint would not be doing what it claims."""
        g = self._iso()
        exact = g.projection_z(nx=96, ny=64, apix=1.0, cutoff_sigma=None)
        loose = g.projection_z(nx=96, ny=64, apix=1.0, cutoff_sigma=6.0)
        tight = g.projection_z(nx=96, ny=64, apix=1.0, cutoff_sigma=1.5)
        assert float((tight - exact).abs().max()) > float((loose - exact).abs().max())

    def test_gaussians_off_the_edge_do_not_wrap(self):
        """A footprint clipped by the image border must be dropped, not folded
        back in at the far side."""
        g = helicon.IsotropicGaussianSet(
            torch.tensor([1.0]), torch.tensor([[-60.0, 0.0, 0.0]]), torch.tensor([2.0])
        )
        p = g.projection_z(nx=32, ny=32, apix=1.0)
        assert float(p[:, -8:].abs().max()) < 1e-6

    def test_it_still_sums_contributions_from_every_gaussian(self):
        g = self._iso(n=40)
        total = float(g.projection_z(nx=96, ny=64, apix=1.0).sum())
        halves = sum(
            float(
                helicon.IsotropicGaussianSet(g.amplitudes[s], g.centers[s], g.sigmas[s])
                .projection_z(nx=96, ny=64, apix=1.0)
                .sum()
            )
            for s in (slice(0, 20), slice(20, 40))
        )
        assert total == pytest.approx(halves, rel=1e-5)
