"""Tests for the Gaussian-splatting helical solver primitives.

The riskiest pieces are (a) the analytic line integral of an *anisotropic*
Gaussian along the projection axis -- whose prefactor is a Schur complement,
not the naive sqrt(2*pi*Sigma_xx) -- and (b) the footprint culling, which must
not change the result.  Both are checked against independent references here.
"""

import math

import numpy as np
import pytest
import torch

from helicon.webApps.lib.solver_gaussian import _project_gaussians
from helicon.webApps.lib.solver_gaussian_splat import (
    build_cov3d,
    helical_expand,
    quat_to_rotation_matrix,
    splat_project,
)


def test_culled_splat_matches_dense_isotropic():
    """Culled splatting must reproduce the existing dense isotropic projector."""
    torch.manual_seed(0)
    ny, nz, apix, M = 32, 64, 5.0, 500
    amps = torch.rand(M).double() + 0.1
    centers = torch.stack(
        [
            (torch.rand(M) - 0.5) * 100,
            (torch.rand(M) - 0.5) * ny * apix,
            (torch.rand(M) - 0.5) * nz * apix,
        ],
        dim=1,
    ).double()
    sig = 3.0
    sigmas = torch.full((M,), sig).double()
    cov = torch.eye(3).double().expand(M, 3, 3) * sig**2

    dense = _project_gaussians(
        amps.float(), centers.float(), sigmas.float(), ny, nz, apix
    ).double()
    wide = splat_project(amps, centers, cov, ny, nz, apix, patch_radius=30)
    assert (wide - dense).abs().max() / dense.abs().max() < 1e-5


def test_culling_is_accurate():
    """Auto (SIGMA_CUTOFF, size-bucketed) culling must stay close to the exact sum,
    including when a few very large Gaussians are mixed in with small ones."""
    torch.manual_seed(0)
    ny, nz, apix, M = 32, 64, 5.0, 400
    amps = torch.rand(M).double() + 0.1
    centers = torch.stack(
        [
            (torch.rand(M) - 0.5) * 100,
            (torch.rand(M) - 0.5) * ny * apix,
            (torch.rand(M) - 0.5) * nz * apix,
        ],
        dim=1,
    ).double()
    sig = torch.full((M,), 2.5).double()
    sig[:10] = 25.0
    cov = torch.zeros(M, 3, 3).double()
    for i in range(3):
        cov[:, i, i] = sig**2

    exact = splat_project(amps, centers, cov, ny, nz, apix, patch_radius=40)
    culled = splat_project(amps, centers, cov, ny, nz, apix)
    # residual is inherent SIGMA_CUTOFF truncation of the Gaussian tails
    assert (culled - exact).abs().max() / exact.abs().max() < 5e-3


def test_anisotropic_projection_matches_numerical_line_integral():
    """Validate the anisotropic prefactor sqrt(2*pi)*sqrt(det(S)/det(S_yz)).

    Reference: brute-force numerical integration of the 3D Gaussian along x.
    Note sqrt(2*pi*Sigma_xx) (as used in helicon.lib.gauss.projection_z) is only
    correct when Sigma has no x cross-terms, so a *rotated* anisotropic Gaussian
    is exactly the case that distinguishes them.
    """
    torch.manual_seed(3)
    ny, nz, apix = 24, 24, 2.0
    # one strongly anisotropic Gaussian, rotated so it has x cross-terms
    log_scales = torch.log(torch.tensor([[8.0, 3.0, 2.0]], dtype=torch.float64))
    q = torch.tensor([[0.804, 0.322, 0.483, 0.117]], dtype=torch.float64)
    cov = build_cov3d(log_scales, q)
    assert abs(float(cov[0, 0, 1])) > 1e-3, "test needs genuine x cross-terms"

    amps = torch.tensor([2.0], dtype=torch.float64)
    centers = torch.tensor([[0.7, 1.3, -0.9]], dtype=torch.float64)
    got = splat_project(amps, centers, cov, ny, nz, apix, patch_radius=40)

    # brute-force: integrate exp(-0.5 (r-mu)^T S^-1 (r-mu)) along x
    inv = torch.inverse(cov)[0].numpy()
    mu = centers[0].numpy()
    xs = np.linspace(-60, 60, 4001)
    dx = xs[1] - xs[0]
    ys = (np.arange(ny) - ny // 2) * apix
    zs = (np.arange(nz) - nz // 2) * apix
    ref = np.zeros((ny, nz))
    for iy, yv in enumerate(ys):
        for iz, zv in enumerate(zs):
            d = np.stack(
                [
                    xs - mu[0],
                    np.full_like(xs, yv - mu[1]),
                    np.full_like(xs, zv - mu[2]),
                ],
                axis=1,
            )
            e = -0.5 * np.einsum("ni,ij,nj->n", d, inv, d)
            ref[iy, iz] = 2.0 * np.exp(e).sum() * dx
    ref /= apix  # splat_project divides the line integral by apix

    assert np.allclose(got.numpy(), ref, rtol=2e-3, atol=1e-6 * ref.max())


def test_gradients_reach_all_parameter_groups():
    torch.manual_seed(0)
    M = 20
    centers = (torch.rand(M, 3, dtype=torch.float64) - 0.5) * 20
    centers.requires_grad_(True)
    log_amp = torch.zeros(M, dtype=torch.float64, requires_grad=True)
    log_scales = torch.full(
        (M, 3), math.log(3.0), dtype=torch.float64, requires_grad=True
    )
    quats = torch.zeros(M, 4, dtype=torch.float64)
    quats[:, 0] = 1.0
    quats.requires_grad_(True)

    cov = build_cov3d(log_scales, quats)
    ea, ec, ecov = helical_expand(
        torch.exp(log_amp),
        centers,
        cov,
        torch.tensor(15.0, dtype=torch.float64),
        torch.tensor(5.0, dtype=torch.float64),
        1,
        3,
    )
    proj = splat_project(ea, ec, ecov, 24, 24, 2.0)
    proj.sum().backward()

    for name, p in [
        ("centers", centers),
        ("log_amp", log_amp),
        ("log_scales", log_scales),
        ("quats", quats),
    ]:
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient not finite"
    # quaternion gradient is zero at identity for a symmetric config only if the
    # Gaussians are isotropic; here scales are equal so just check the others move
    for name, p in [
        ("centers", centers),
        ("log_amp", log_amp),
        ("log_scales", log_scales),
    ]:
        assert p.grad.abs().sum() > 0, f"{name} gradient identically zero"


def test_helical_expand_rotates_covariance():
    """A helical copy must have its covariance rotated, not just its centre."""
    log_scales = torch.log(torch.tensor([[10.0, 2.0, 2.0]], dtype=torch.float64))
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )  # long axis along x
    cov = build_cov3d(log_scales, quats)
    centers = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float64)
    amps = torch.ones(1, dtype=torch.float64)

    _, ec, ecov = helical_expand(
        amps,
        centers,
        cov,
        torch.tensor(90.0, dtype=torch.float64),
        torch.tensor(4.0, dtype=torch.float64),
        1,
        1,
    )
    # copies are at n = -1, 0, +1; the n=+1 copy is rotated by +90 deg about z
    c_plus, cov_plus = ec[2], ecov[2]
    assert np.allclose(c_plus.numpy(), [0.0, 10.0, 4.0], atol=1e-9)
    # the long axis (was x) must now lie along y
    assert cov_plus[1, 1] > cov_plus[0, 0]
    assert np.isclose(float(cov_plus[1, 1]), 100.0, rtol=1e-9)


def _synthetic_target(
    tilt, psi, dy, seed, apix=5.0, rise=15.0, twist=25.0, ny=32, nz=64, M=40
):
    """Anisotropic Gaussians in one rise, helically duplicated, with a known
    tilt/psi/dy misalignment applied."""
    from helicon.webApps.lib.solver_gaussian_splat import _rigid_transform_t

    g = torch.Generator().manual_seed(seed)
    tube_r = ny * apix * 0.35
    r = torch.sqrt(torch.rand(M, generator=g)) * tube_r * 0.85
    th = torch.rand(M, generator=g) * 2 * math.pi
    z = (torch.rand(M, generator=g) - 0.5) * rise
    centers = torch.stack([r * torch.cos(th), r * torch.sin(th), z], -1)
    ls = torch.log(3.0 + 4.0 * torch.rand(M, 3, generator=g))
    q = torch.nn.functional.normalize(torch.rand(M, 4, generator=g) - 0.5, dim=-1)
    amps = 0.3 + torch.rand(M, generator=g)
    cov = build_cov3d(ls, q)
    n_rep = max(1, int(np.ceil(nz * apix / (2 * rise))) + 4)
    ea, ec, ecov = helical_expand(
        amps, centers, cov, torch.tensor(twist), torch.tensor(rise), 1, n_rep
    )
    ec, ecov = _rigid_transform_t(
        ec,
        ecov,
        torch.tensor(float(tilt)),
        torch.tensor(float(psi)),
        torch.tensor(float(dy)),
        apix,
    )
    return splat_project(ea, ec, ecov, ny, nz, apix).detach().numpy().astype(np.float32)


@pytest.mark.parametrize("tilt,psi,dy", [(0.0, 0.0, 0.0), (2.0, -1.5, 0.4)])
def test_splat_recovers_tilt_psi_dy(tilt, psi, dy):
    """Differentiable tilt/psi/dy refinement must recover a known misalignment.

    This is the capability that motivates the Gaussian solver at all: lsq can
    only refine these by finite-difference Gauss-Newton, whereas splat gets the
    gradients from the same backward pass.  Starting guess is always zero.
    """
    from helicon.webApps.lib.solver_gaussian_splat import splat_fit

    apix, rise, twist, ny = 5.0, 15.0, 25.0, 32
    target = _synthetic_target(tilt, psi, dy, seed=0)
    splat_fit(
        target,
        apix2d=apix,
        twist_degree=twist,
        rise_angstrom=rise,
        csym=1,
        tube_radius=ny * apix * 0.35,
        n_init=120,
        iters=500,
        max_gauss=300,
        densify_until=350,
        seed=0,
        refine_tilt_psi_dy=True,
        rigid_lr=0.1,
        device="cpu",
    )
    t, p, d = splat_fit.refined_tilt_psi_dy
    assert abs(t - tilt) < 0.3, f"tilt {t} vs {tilt}"
    assert abs(p - psi) < 0.3, f"psi {p} vs {psi}"
    assert abs(d - dy) < 0.3, f"dy {d} vs {dy}"


def test_fit_is_deterministic_regardless_of_global_rng():
    """Densification jitter must come from a seeded per-model generator.

    If it used the global torch RNG, identical arguments would give different
    answers run to run -- which also makes the result unsafe to cache.
    """
    from helicon.webApps.lib.solver_gaussian_splat import splat_fit

    rng = np.random.default_rng(0)
    img = rng.random((32, 64)).astype(np.float32)
    results = []
    for trial in range(3):
        torch.manual_seed(trial * 997)  # deliberately perturb the global RNG
        m, cos, _ = splat_fit(
            img,
            apix2d=5.0,
            twist_degree=20.0,
            rise_angstrom=15.0,
            csym=1,
            tube_radius=50.0,
            n_init=30,
            iters=120,
            max_gauss=80,
            densify_until=90,
            seed=0,
            device="cpu",
        )
        results.append((round(cos, 8), m.n))
    assert len(set(results)) == 1, f"non-deterministic across runs: {results}"


def test_cache_hit_preserves_refined_params(tmp_path):
    """A cache hit performs no side effects, so the wrapper must re-publish the
    refined tilt/psi/dy rather than leaving them stale from a previous call."""
    from helicon.webApps.lib import solver_gaussian_splat as m

    rng = np.random.default_rng(3)
    img = rng.random((24, 48)).astype(np.float32)
    kw = dict(
        projection_image=img,
        scale2d_to_3d=1.0,
        twist_degree=20.0,
        rise_pixel=3.0,
        csym=1,
        reconstruct_diameter_3d_pixel=24,
        reconstruct_length_3d_pixel=6,
        target_apix2d=5.0,
        algorithm=dict(
            model="gauss_splat",
            n_init=20,
            iters=60,
            max_gauss=40,
            densify_until=45,
            refine_psi=True,
            refine_dy=True,
            device="cpu",
        ),
    )
    (_, _, _), score_a = m.gauss_splat_reconstruct(**kw)
    refined_a = dict(m.gauss_splat_reconstruct._refined_params)

    m.gauss_splat_reconstruct._refined_params = None  # simulate a later call
    (_, _, _), score_b = m.gauss_splat_reconstruct(**kw)
    refined_b = m.gauss_splat_reconstruct._refined_params

    assert score_a == score_b, "cached score differs from the computed one"
    assert refined_b is not None, "refined params lost on cache hit"
    assert refined_a == refined_b


def _expanded_set(M=30, seed=0, aniso=False):
    torch.manual_seed(seed)
    if aniso:
        ls = torch.log(torch.tensor([[12.0, 3.0, 2.5]], dtype=torch.float64)).repeat(
            M, 1
        )
    else:
        ls = torch.log(3.0 + 3.0 * torch.rand(M, 3, dtype=torch.float64))
    q = torch.nn.functional.normalize(
        torch.rand(M, 4, dtype=torch.float64) - 0.5, dim=-1
    )
    c = torch.stack(
        [
            (torch.rand(M, dtype=torch.float64) - 0.5) * 60,
            (torch.rand(M, dtype=torch.float64) - 0.5) * 60,
            (torch.rand(M, dtype=torch.float64) - 0.5) * 15,
        ],
        -1,
    )
    a = 0.3 + torch.rand(M, dtype=torch.float64)
    cov = build_cov3d(ls, q)
    return helical_expand(
        a,
        c,
        cov,
        torch.tensor(15.0, dtype=torch.float64),
        torch.tensor(12.0, dtype=torch.float64),
        1,
        6,
    )


@pytest.mark.parametrize("psi", [5.0, -12.0, 20.0])
def test_psi_in_3d_equals_in_plane_rotation_of_the_projection(psi):
    """psi rotates about the projection axis, so applying it to the 3D model
    before projecting must equal rotating the projected image.

    This is the property that makes psi meaningful as an in-plane correction;
    if it failed, psi would be modelling something other than the horizontal
    angle of the filament.
    """
    ndi = pytest.importorskip("scipy.ndimage")
    from helicon.webApps.lib.solver_gaussian_splat import _rigid_transform_t

    ny, nz, apix = 48, 96, 4.0
    ea, ec, ecov = _expanded_set()
    Z = torch.tensor(0.0, dtype=torch.float64)
    base = splat_project(ea, ec, ecov, ny, nz, apix, patch_radius=40).numpy()

    c2, cov2 = _rigid_transform_t(
        ec, ecov, Z, torch.tensor(psi, dtype=torch.float64), Z, apix
    )
    got = splat_project(ea, c2, cov2, ny, nz, apix, patch_radius=40).numpy()
    ref = ndi.rotate(base, +psi, reshape=False, order=3, mode="constant")

    m = np.zeros((ny, nz), dtype=bool)
    m[8:-8, 8:-8] = True  # ignore the frame edge scipy pads with zeros
    assert np.corrcoef(got[m], ref[m])[0, 1] > 0.99


@pytest.mark.parametrize("dy_px", [2.0, -3.0])
def test_dy_in_3d_equals_shifting_the_projection(dy_px):
    ndi = pytest.importorskip("scipy.ndimage")
    from helicon.webApps.lib.solver_gaussian_splat import _rigid_transform_t

    ny, nz, apix = 48, 96, 4.0
    ea, ec, ecov = _expanded_set()
    Z = torch.tensor(0.0, dtype=torch.float64)
    base = splat_project(ea, ec, ecov, ny, nz, apix, patch_radius=40).numpy()

    c2, cov2 = _rigid_transform_t(
        ec, ecov, Z, Z, torch.tensor(dy_px, dtype=torch.float64), apix
    )
    got = splat_project(ea, c2, cov2, ny, nz, apix, patch_radius=40).numpy()
    ref = ndi.shift(base, (dy_px, 0), order=3, mode="constant")

    m = np.zeros((ny, nz), dtype=bool)
    m[8:-8, 8:-8] = True
    assert np.corrcoef(got[m], ref[m])[0, 1] > 0.99


def test_rigid_transform_rotates_the_covariance():
    """Anisotropic Gaussians must have their shape rotated, not just their
    centre: Sigma -> R Sigma R^T."""
    from helicon.webApps.lib.solver_gaussian_splat import _rigid_transform_t

    Z = torch.tensor(0.0, dtype=torch.float64)
    S = torch.diag(torch.tensor([144.0, 9.0, 6.25], dtype=torch.float64))
    _, cov_out = _rigid_transform_t(
        torch.zeros(1, 3, dtype=torch.float64),
        S.unsqueeze(0),
        Z,
        torch.tensor(45.0, dtype=torch.float64),
        Z,
        4.0,
    )
    th = math.radians(45.0)
    Rx = torch.tensor(
        [[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]],
        dtype=torch.float64,
    )
    assert torch.allclose(cov_out[0], Rx @ S @ Rx.T, atol=1e-10)
