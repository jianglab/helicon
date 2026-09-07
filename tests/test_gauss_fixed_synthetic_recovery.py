"""Synthetic ground-truth recovery tests for gauss_fixed_reconstruct's
tilt/psi/dy refinement.

Unlike the unit tests in test_gauss_tilt_psi_dy.py (which check the rotation
transform in isolation), these tests exercise the *full* optimization loop:
a target image is generated from a known Gaussian-mixture density with a
known (tilt, psi, dy) misalignment, and the solver -- started from zero
misalignment -- must recover values close to ground truth.

This is the only way to know refinement actually *works*, not just runs:
real class averages have no known ground truth to check against, and the
previous gauss solver attempt failed silently for exactly that reason.

Run standalone for fast hyperparameter iteration:
    python tests/test_gauss_fixed_synthetic_recovery.py
Run under pytest for the locked-in pass/fail gate:
    pytest tests/test_gauss_fixed_synthetic_recovery.py -v
"""

import math

import numpy as np
import torch

from helicon.webApps.lib.solver_gaussian import (
    _apply_tilt_psi_dy,
    _fixed_helical_expand,
    _make_cylindrical_grid,
    _project_gaussians,
    gauss_fixed_reconstruct,
)

# Shared synthetic geometry -- deliberately small/fast so many seeds can run.
GRID_NR = 4
GRID_NA = 10
APIX = 5.0  # apix2d == apix3d (scale2d_to_3d = 1.0)
TWIST_DEGREE = 25.0
RISE_ANGSTROM = 15.0
CSYM = 1
PROJ_NY = 40
PROJ_NZ = 40
TUBE_RADIUS_3D = PROJ_NY * APIX * 0.35


def _n_repeats(reconstruct_length_3d_pixel, apix3d, rise_angstrom):
    Lz = reconstruct_length_3d_pixel * apix3d
    if rise_angstrom > 0.01:
        return max(1, int(np.ceil(Lz / (2 * rise_angstrom))) + 4)
    return 4


def make_synthetic_target(tilt_degree, psi_degree, dy_pixel, seed, grid_sigma):
    """Build a target 2D projection from a random-amplitude Gaussian mixture
    on the same fixed cylindrical grid the solver uses, with a known rigid
    (tilt, psi, dy) misalignment applied after helical expansion.
    """
    rng = np.random.default_rng(seed)
    centers = _make_cylindrical_grid(GRID_NR, GRID_NA, TUBE_RADIUS_3D, "cpu")
    n_gauss = centers.shape[0]
    # Random (non-rotationally-symmetric) amplitudes -- a uniform/symmetric
    # density would make psi undetectable by construction, which would be
    # testing the wrong thing.
    amps_np = rng.uniform(0.3, 1.0, size=n_gauss).astype(np.float32)
    amps = torch.from_numpy(amps_np)
    sigmas = torch.full((n_gauss,), grid_sigma)

    n_repeats = _n_repeats(PROJ_NZ, APIX, RISE_ANGSTROM)
    exp_amps, exp_centers, exp_sigmas = _fixed_helical_expand(
        amps,
        centers,
        sigmas,
        torch.tensor(TWIST_DEGREE),
        torch.tensor(RISE_ANGSTROM),
        CSYM,
        n_repeats,
    )
    exp_centers = _apply_tilt_psi_dy(
        exp_centers,
        torch.tensor(float(tilt_degree)),
        torch.tensor(float(psi_degree)),
        torch.tensor(float(dy_pixel)),
    )
    proj = _project_gaussians(exp_amps, exp_centers, exp_sigmas, PROJ_NY, PROJ_NZ, APIX)
    return proj.numpy().astype(np.float32)


def recover(target_img, grid_sigma, algorithm_overrides=None):
    algorithm = dict(
        model="gauss_fixed",
        grid_n_radial=GRID_NR,
        grid_n_angular=GRID_NA,
        grid_sigma=grid_sigma,
        refine_tilt_psi_dy=True,
    )
    if algorithm_overrides:
        algorithm.update(algorithm_overrides)

    (rec3d, _, _), score = gauss_fixed_reconstruct(
        projection_image=target_img,
        scale2d_to_3d=1.0,
        twist_degree=TWIST_DEGREE,
        rise_pixel=RISE_ANGSTROM / APIX,
        csym=CSYM,
        tilt_degree=0.0,
        psi_degree=0.0,
        dy_pixel=0.0,
        reconstruct_diameter_3d_pixel=int(round(2 * TUBE_RADIUS_3D / APIX)),
        reconstruct_length_3d_pixel=PROJ_NZ,
        target_apix2d=APIX,
        algorithm=algorithm,
    )
    rp = gauss_fixed_reconstruct._refined_params
    return rp["tilt"], rp["psi"], rp["dy"], score


def _run_trial(tilt_gt, psi_gt, dy_gt, seed, grid_sigma=APIX / 2.0, overrides=None):
    target = make_synthetic_target(tilt_gt, psi_gt, dy_gt, seed, grid_sigma)
    tilt_hat, psi_hat, dy_hat, score = recover(target, grid_sigma, overrides)
    return dict(
        tilt_gt=tilt_gt,
        psi_gt=psi_gt,
        dy_gt=dy_gt,
        tilt_hat=tilt_hat,
        psi_hat=psi_hat,
        dy_hat=dy_hat,
        tilt_err=tilt_hat - tilt_gt,
        psi_err=psi_hat - psi_gt,
        dy_err=dy_hat - dy_gt,
        score=score,
        seed=seed,
    )


def test_null_control_no_drift():
    """With zero ground-truth misalignment, refinement should not wander."""
    for seed in range(4):
        r = _run_trial(tilt_gt=0.0, psi_gt=0.0, dy_gt=0.0, seed=seed + 100)
        assert abs(r["tilt_err"]) < 0.2, r
        assert abs(r["psi_err"]) < 0.2, r
        assert abs(r["dy_err"]) < 0.2, r
        assert r["score"] > 0.9, r


def test_recovers_small_misalignment():
    """Residual misalignment in the range auto-horizontalization would
    typically leave behind."""
    for seed in range(4):
        r = _run_trial(tilt_gt=1.5, psi_gt=-1.0, dy_gt=0.3, seed=seed)
        assert abs(r["tilt_err"]) < 0.5, r
        assert abs(r["psi_err"]) < 0.3, r
        assert abs(r["dy_err"]) < 0.3, r
        assert r["score"] > 0.9, r


def test_recovers_medium_misalignment():
    for seed in range(4):
        r = _run_trial(tilt_gt=8.0, psi_gt=-6.0, dy_gt=1.5, seed=seed)
        assert abs(r["tilt_err"]) < 1.0, r
        assert abs(r["psi_err"]) < 0.3, r
        assert abs(r["dy_err"]) < 0.3, r
        assert r["score"] > 0.9, r


def test_large_misalignment_is_a_known_limitation(caplog):
    """Refinement is a *local* correction (gradient descent from the given
    starting value), not a blind global search. Starting from zero with a
    large (20 deg tilt / 15 deg psi) true misalignment is expected to fail
    to converge to the true value -- this test documents that boundary and
    asserts the failure is at least visible (a bound-hit warning and/or a
    clearly degraded score), not silently confident.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="helicon.webApps.lib.solver_gaussian"):
        r = _run_trial(tilt_gt=20.0, psi_gt=15.0, dy_gt=4.0, seed=0)
    hit_bound_warning = any("hit its bound" in rec.message for rec in caplog.records)
    assert hit_bound_warning or r["score"] < 0.8, r


if __name__ == "__main__":
    torch.manual_seed(0)
    print(
        f"{'seed':>4} {'tilt_gt':>8} {'psi_gt':>8} {'dy_gt':>7} | "
        f"{'tilt_hat':>9} {'psi_hat':>8} {'dy_hat':>7} | "
        f"{'d_tilt':>7} {'d_psi':>7} {'d_dy':>6} | {'score':>7}"
    )
    for seed in range(6):
        r = _run_trial(tilt_gt=8.0, psi_gt=-6.0, dy_gt=1.5, seed=seed)
        print(
            f"{r['seed']:4d} {r['tilt_gt']:8.2f} {r['psi_gt']:8.2f} {r['dy_gt']:7.2f} | "
            f"{r['tilt_hat']:9.3f} {r['psi_hat']:8.3f} {r['dy_hat']:7.3f} | "
            f"{r['tilt_err']:7.3f} {r['psi_err']:7.3f} {r['dy_err']:6.3f} | {r['score']:7.4f}"
        )

    print("\nNull control (zero ground-truth misalignment):")
    for seed in range(4):
        r = _run_trial(tilt_gt=0.0, psi_gt=0.0, dy_gt=0.0, seed=seed + 100)
        print(
            f"{r['seed']:4d} {r['tilt_gt']:8.2f} {r['psi_gt']:8.2f} {r['dy_gt']:7.2f} | "
            f"{r['tilt_hat']:9.3f} {r['psi_hat']:8.3f} {r['dy_hat']:7.3f} | "
            f"{r['tilt_err']:7.3f} {r['psi_err']:7.3f} {r['dy_err']:6.3f} | {r['score']:7.4f}"
        )
