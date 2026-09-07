"""Tests for the differentiable tilt/psi/dy rigid-transform helper used by
the gauss solver, ahead of wiring it into the Adam optimization loop.

These tests validate the two riskiest, easiest-to-get-silently-wrong pieces
in isolation:
  1. The rotation/shift convention exactly matches the reference used by the
     (working) linear-regression solver in denovo3d_solver.py, so refined
     tilt/psi/dy values will mean the same physical thing in both solvers.
  2. Gradients actually flow to tilt/psi/dy (and centers) through the
     transform, with values matching finite differences — i.e. the parameters
     are inside the computational graph, not silently detached.
"""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as R

from helicon.webApps.lib.solver_gaussian import _apply_tilt_psi_dy


def _reference_transform(points_xyz, tilt_degree, psi_degree, dy_pixel):
    """Forward transform matching denovo3d_solver.build_A_data_matrix's
    inverse usage: data_point = R_yx(tilt, psi) @ canonical_point + [0, dy, 0].
    """
    r = R.from_euler("yx", (tilt_degree, psi_degree), degrees=True)
    out = r.apply(points_xyz)
    out[:, 1] += dy_pixel
    return out


@pytest.mark.parametrize("seed", range(5))
def test_convention_matches_denovo3d_solver(seed):
    rng = np.random.default_rng(seed)
    points_np = rng.normal(size=(20, 3)).astype(np.float64)
    tilt = float(rng.uniform(-30, 30))
    psi = float(rng.uniform(-30, 30))
    dy = float(rng.uniform(-5, 5))

    expected = _reference_transform(points_np.copy(), tilt, psi, dy)

    centers = torch.tensor(points_np, dtype=torch.float64)
    got = _apply_tilt_psi_dy(
        centers,
        torch.tensor(tilt, dtype=torch.float64),
        torch.tensor(psi, dtype=torch.float64),
        torch.tensor(dy, dtype=torch.float64),
    ).numpy()

    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_zero_params_is_identity():
    points_np = np.random.default_rng(1).normal(size=(10, 3))
    centers = torch.tensor(points_np, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    got = _apply_tilt_psi_dy(centers, zero, zero, zero).numpy()
    np.testing.assert_allclose(got, points_np, atol=1e-12)


def test_gradients_flow_and_match_finite_differences():
    torch.manual_seed(0)
    centers = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    tilt = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)
    psi = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    dy = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda c, t, p, d: _apply_tilt_psi_dy(c, t, p, d),
        (centers, tilt, psi, dy),
        eps=1e-6,
        atol=1e-8,
    )


def test_dy_only_shifts_y_axis():
    centers = torch.zeros(3, 3, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    dy = torch.tensor(2.5, dtype=torch.float64)
    got = _apply_tilt_psi_dy(centers, zero, zero, dy).numpy()
    np.testing.assert_allclose(got, np.array([[0, 2.5, 0]] * 3), atol=1e-12)
