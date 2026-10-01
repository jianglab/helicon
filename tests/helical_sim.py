"""Simulated Class2D output of a helix with known Euler angles and centres.

The metadata follow what RELION writes for helical segments: the psi prior is
the direction each filament was picked in, a class average is a view of one
polarity (polar axis to the right, or its 180-degree in-plane rotation), and a
segment is aligned to a class either as picked or turned by 180 degrees, so
``rlnAnglePsi - rlnAnglePsiPrior`` is 0 or 180 plus a small deviation. What is
known for validation: the true azimuth (rot), the true psi and the true segment
centre, and a point-scatterer model of the structure to render views from.
"""

import numpy as np
import pandas as pd

APIX = 1.0


class Helix:
    """An asymmetric unit of three points repeated with the helical symmetry."""

    def __init__(self, pitch=1400.0, fold=2, rise=4.75):
        self.pitch, self.fold, self.rise = pitch, fold, rise
        self.twist = 360.0 * rise / pitch
        # (radius, axial offset, azimuth offset) of the three points of a subunit
        self.unit = np.array([[30.0, 0.0, 0.0], [45.0, 1.5, 25.0], [20.0, -1.0, -40.0]])

    def points(self, zeta_center, half_length=150.0):
        n = np.arange(
            int((zeta_center - half_length) / self.rise) - 1,
            int((zeta_center + half_length) / self.rise) + 2,
        )
        zeta, rho, theta = [], [], []
        for j in range(self.fold):
            for r, dz, dth in self.unit:
                zeta.append(n * self.rise + dz)
                rho.append(np.full(len(n), r))
                theta.append(n * self.twist + dth + 360.0 * j / self.fold)
        return (
            np.concatenate(zeta),
            np.concatenate(rho),
            np.deg2rad(np.concatenate(theta)),
        )

    def render(self, zeta_center, rot, psi, size=64, pixel=5.0, sigma=3.0):
        """Side view at azimuth ``rot`` (deg), polar axis pointing at ``psi`` (deg)."""
        zeta, rho, theta = self.points(zeta_center)
        a = zeta - zeta_center
        t = rho * np.sin(theta - np.deg2rad(rot))
        ps = np.deg2rad(psi)
        x = a * np.cos(ps) - t * np.sin(ps)
        y = a * np.sin(ps) + t * np.cos(ps)
        g = (np.arange(size) - size / 2 + 0.5) * pixel
        gx = np.exp(-0.5 * ((g[None, :] - x[:, None]) / sigma) ** 2)
        gy = np.exp(-0.5 * ((g[None, :] - y[:, None]) / sigma) ** 2)
        return gy.T @ gx


def _wrap(a, period=360.0):
    return (a + period / 2) % period - period / 2


def simulate(
    helix,
    n_fil=200,
    n_seg=60,
    step=40.0,
    n_views=12,
    flip_rate=0.2,
    pitch_sd=0.0,
    misclass=0.2,
    shift_sd=(12.0, 8.0),
    shift_err=3.0,
    psi_sd=3.0,
    bend=0.0,
    class_dy_sd=0.0,
    turned_offset_sd=0.0,
    ydir=1.0,
    seed=0,
):
    """Class2D parameters plus the truth, per segment.

    A segment is put in a neighbouring view with probability ``misclass``.
    Classes ``1..n_views`` are views with the polar axis to the right,
    ``n_views+1..2*n_views`` the same views turned by 180 degrees in plane, each
    class at an azimuth of its own (the turned ones displaced by up to
    ``turned_offset_sd`` degrees from the plain ones, as two classifications of
    the same views centre differently). A segment is aligned to the class
    nearest its true azimuth: the alignment moves its centre along the axis to
    where the helix has the class's azimuth (so refined centres are quantized
    like that), and across it by ``class_dy_sd`` per class, the error of a
    class average that is not centred on its filament.

    ``ydir`` is the sign of the micrograph y axis in the psi convention (-1 for
    RELION, whose y runs down while psi runs counterclockwise).

    ``bend`` is the total change of direction along a filament, in degrees.
    Returns ``(params, repeat, class_info)``, ``class_info`` giving the azimuth
    and turn of every class for rendering class averages.
    """
    rng = np.random.default_rng(seed)
    per = 360.0 / helix.fold
    P = helix.pitch / helix.fold
    az_plain = (np.arange(n_views) + 0.5) * per / n_views
    az_turned = az_plain + rng.normal(0, turned_offset_sd, n_views)
    class_az = np.concatenate([az_plain, az_turned])
    class_turned = np.arange(2 * n_views) >= n_views
    class_dy = rng.normal(0, class_dy_sd, 2 * n_views)
    rows = []
    for f in range(n_fil):
        s = rng.choice([-1.0, 1.0])
        alpha0 = rng.uniform(-180, 180)
        bend_f = bend * rng.uniform(-1, 1)
        m0 = np.array([3000.0, 3000.0]) + rng.uniform(-500, 500, 2)
        pitch = helix.pitch * (1 + pitch_sd * rng.standard_normal())
        theta0 = rng.uniform(0, 360)
        length = n_seg * step
        tt = np.arange(0.0, length + 1.0, 2.0)
        ang = alpha0 + bend_f * tt / length
        xy = m0 + np.cumsum(
            np.stack([np.cos(np.deg2rad(ang)), ydir * np.sin(np.deg2rad(ang))], 1)
            * 2.0,
            0,
        )
        for k in range(n_seg):
            t = k * step
            i = int(t / 2.0)
            alpha = ang[i]
            u = np.array([np.cos(np.deg2rad(alpha)), ydir * np.sin(np.deg2rad(alpha))])
            v = np.array([-u[1], u[0]])
            true_center = xy[i]
            a, b = rng.normal(0, shift_sd[0]), rng.normal(0, shift_sd[1])
            coord = true_center + a * u + b * v  # where the picker put it
            rot = (theta0 + 360.0 * s * t / pitch) % 360.0
            polar_right = s > 0
            flip = rng.random() < flip_rate
            if flip:
                polar_right = not polar_right
            pool = np.flatnonzero(class_turned == (not polar_right))
            c = pool[np.argmin(np.abs(_wrap(class_az[pool] - rot, per)))]
            if rng.random() < misclass:
                # a neighbouring view, in the same or the other polarity set
                same = pool[np.argsort(np.abs(_wrap(class_az[pool] - rot, per)))[:4]]
                c = int(rng.choice(same))
            dz = _wrap(class_az[c] - rot, per) * helix.pitch / 360.0
            est = (
                true_center + s * dz * u + class_dy[c] * v + rng.normal(0, shift_err, 2)
            )
            rot_est = rot + 360.0 * dz / helix.pitch
            origin = coord - est
            psi_true = alpha + (180.0 if s < 0 else 0.0)
            psi_2d = alpha + (180.0 if flip else 0.0) + rng.normal(0, psi_sd)
            rows.append(
                dict(
                    rlnMicrographName=f"mic{f // 10:03d}.mrc",
                    rlnHelicalTubeID=f % 10 + 1,
                    rlnHelicalTrackLengthAngst=t,
                    rlnClassNumber=int(c) + 1,
                    rlnAnglePsi=(psi_2d + 180) % 360 - 180,
                    rlnAnglePsiPrior=(alpha + rng.normal(0, 1.0) + 180) % 360 - 180,
                    rlnCoordinateX=coord[0] / APIX,
                    rlnCoordinateY=coord[1] / APIX,
                    rlnOriginXAngst=origin[0],
                    rlnOriginYAngst=origin[1],
                    rlnMicrographOriginalPixelSize=APIX,
                    true_rot=rot,
                    true_rot_est=rot_est % 360.0,
                    true_psi=(psi_true + 180) % 360 - 180,
                    true_center_x=true_center[0],
                    true_center_y=true_center[1],
                    est_center_x=est[0],
                    est_center_y=est[1],
                    _sign=s,
                    _flip=flip,
                    _turned=bool(class_turned[c]),
                )
            )
    info = dict(azimuth=class_az, turned=class_turned, dy=class_dy)
    return pd.DataFrame(rows), P, info
