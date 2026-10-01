"""Validate the AbInitio3D angle assignment by reconstructing with RELION.

Simulates helical segments of a known structure (see ``helical_sim.py``),
projects them with ``relion_project``, and reconstructs with
``relion_reconstruct`` from the true angles, from the angles and origins the
AbInitio3D tab assigns, and from a naive assignment (class angle, Class2D psi
and origin). The reconstructions are compared with the structure by FSC and
correlation inside a cylinder.

Run with RELION on the PATH (or ``RELION_BIN`` naming its bin directory)::

    RELION_BIN=/path/to/relion/bin python tests/validation/relion_reconstruction.py WORKDIR

Not collected by pytest: it needs RELION and takes several minutes.
"""

import os, pathlib, subprocess, sys
import numpy as np, pandas as pd, mrcfile, starfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from helical_sim import Helix, simulate

RELION = os.environ.get("RELION_BIN", "")
ENV = dict(os.environ, PATH=(RELION + ":" if RELION else "") + os.environ["PATH"])
APIX, BOX = 3.0, 128


def run(cmd):
    r = subprocess.run(cmd, shell=True, env=ENV, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(cmd + "\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def volume(helix, path, box=BOX, apix=APIX, sigma=4.0):
    zeta, rho, theta = helix.points(0.0, half_length=box * apix / 2 + 30.0)
    g = (np.arange(box) - box / 2) * apix
    gx = np.exp(-0.5 * ((g[None] - (rho * np.cos(theta))[:, None]) / sigma) ** 2)
    gy = np.exp(-0.5 * ((g[None] - (rho * np.sin(theta))[:, None]) / sigma) ** 2)
    gz = np.exp(-0.5 * ((g[None] - zeta[:, None]) / sigma) ** 2)
    vol = np.einsum("ix,iy,iz->zyx", gx, gy, gz).astype(np.float32)
    vol /= vol.max()
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(vol)
        m.voxel_size = apix
    return vol


def particles_star(df, path, image_stack=None, apix=APIX, box=BOX):
    """A particle star from a data frame with rlnAngle*, rlnOrigin*Angst columns."""
    n = len(df)
    p = pd.DataFrame(
        dict(
            rlnImageName=[
                f"{i+1:06d}@{image_stack}" if image_stack else f"{i+1:06d}@x.mrcs"
                for i in range(n)
            ],
            rlnAngleRot=df.rlnAngleRot.values,
            rlnAngleTilt=df.rlnAngleTilt.values,
            rlnAnglePsi=df.rlnAnglePsi.values,
            rlnOriginXAngst=df.rlnOriginXAngst.values,
            rlnOriginYAngst=df.rlnOriginYAngst.values,
            rlnOpticsGroup=1,
        )
    )
    o = pd.DataFrame(
        dict(
            rlnOpticsGroupName=["o1"],
            rlnOpticsGroup=[1],
            rlnImagePixelSize=[apix],
            rlnImageSize=[box],
            rlnImageDimensionality=[2],
            rlnVoltage=[300.0],
            rlnSphericalAberration=[2.7],
            rlnAmplitudeContrast=[0.1],
        )
    )
    starfile.write(dict(optics=o, particles=p), path, overwrite=True)


import sys, time, pathlib
from helicon.webApps.lib import helical_pitch_phase as ph

W = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "relion_validation_work")
W.mkdir(exist_ok=True)
FOLD = 2
BIG = 192
ABLATE = True


def wrap(a, period=360.0):
    return (a + period / 2) % period - period / 2


def fsc(a, b, apix=APIX, radius=70.0):
    n = a.shape[0]
    g = (np.arange(n) - n / 2) * apix
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")
    mask = ((np.hypot(xx, yy) < radius) & (np.abs(zz) < 100.0)).astype(np.float32)
    A, B = np.fft.fftn(a * mask), np.fft.fftn(b * mask)
    f = np.fft.fftfreq(n, apix)
    fz, fy, fx = np.meshgrid(f, f, f, indexing="ij")
    r = np.sqrt(fx**2 + fy**2 + fz**2)
    bins = np.linspace(0, 0.5 / apix, n // 2 + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    num = np.bincount(idx, (A * B.conj()).real.ravel(), minlength=len(bins) + 1)[:-1]
    d = (
        np.bincount(idx, (np.abs(A) ** 2).ravel(), minlength=len(bins) + 1)[:-1]
        * np.bincount(idx, (np.abs(B) ** 2).ravel(), minlength=len(bins) + 1)[:-1]
    )
    c = num / np.sqrt(d + 1e-20)
    return (bins[:-1] + bins[1:]) / 2, c[: len(bins) - 1]


def masked_cc(a, b, apix=APIX, radius=70.0, half=100.0):
    n = a.shape[0]
    g = (np.arange(n) - n / 2) * apix
    zz, yy, xx = np.meshgrid(g, g, g, indexing="ij")
    m = (np.hypot(xx, yy) < radius) & (np.abs(zz) < half)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def res_at(freq, c, level):
    below = np.flatnonzero(c < level)
    below = below[freq[below] > 0]
    return 1.0 / freq[below[0]] if len(below) else 2 * APIX


def crop_stack(src, dst):
    a = mrcfile.read(str(W / src))
    c0 = (BIG - BOX) // 2
    with mrcfile.new(str(W / dst), overwrite=True) as m:
        m.set_data(
            np.ascontiguousarray(a[:, c0 : c0 + BOX, c0 : c0 + BOX]).astype(np.float32)
        )
        m.voxel_size = APIX


def reconstruct(name, ang, images):
    df = ang.copy()
    particles_star(df, str(W / f"{name}.star"), image_stack=images)
    run(
        f"cd {W} && relion_reconstruct --i {name}.star --o {name}.mrc --angpix {APIX} --j 20 2>&1 | tail -1"
    )
    return mrcfile.read(str(W / f"{name}.mrc")).astype(np.float32)


def scenario(name, n_fil=120, n_seg=50, noise=0.6, **kw):
    hx = Helix(pitch=700.0 * FOLD, fold=FOLD)
    big = volume(hx, str(W / "ref_big.mrc"), box=BIG)
    c0 = (BIG - BOX) // 2
    ref = big[c0 : c0 + BOX, c0 : c0 + BOX, c0 : c0 + BOX].copy()
    df, P, info = simulate(hx, n_fil=n_fil, n_seg=n_seg, ydir=-1.0, **kw)
    df["rlnAngleTilt"] = 90.0
    truth = pd.DataFrame(
        dict(
            rlnAngleRot=df.true_rot.values,
            rlnAngleTilt=90.0,
            rlnAnglePsi=df.true_psi.values,
            rlnOriginXAngst=(df.rlnCoordinateX.values * 1.0 - df.true_center_x.values),
            rlnOriginYAngst=(df.rlnCoordinateY.values * 1.0 - df.true_center_y.values),
        )
    )
    particles_star(
        truth, str(W / "truth_in.star"), image_stack="segs_big.mrcs", box=BIG
    )
    run(
        f"cd {W} && relion_project --i ref_big.mrc --ang truth_in.star --o segs_big --angpix {APIX} --add_noise --white_noise {noise} 2>&1 | tail -1"
    )
    crop_stack("segs_big.mrcs", "segs.mrcs")
    # class averages by projection
    cls = pd.DataFrame(
        dict(
            rlnAngleRot=info["azimuth"],
            rlnAngleTilt=90.0,
            rlnAnglePsi=np.where(info["turned"], 180.0, 0.0),
            rlnOriginXAngst=0.0,
            rlnOriginYAngst=0.0,
        )
    )
    particles_star(cls, str(W / "cls_in.star"), image_stack="cls_big.mrcs", box=BIG)
    run(
        f"cd {W} && relion_project --i ref_big.mrc --ang cls_in.star --o cls_big --angpix {APIX} 2>&1 | tail -1"
    )
    crop_stack("cls_big.mrcs", "cls.mrcs")
    cimgs = list(mrcfile.read(str(W / "cls.mrcs")))
    t0 = time.time()
    cp = ph.pair_counterparts(cimgs, range(len(cimgs)), min_corr=0.8)
    cp = [dict(a=d["a"] + 1, b=d["b"] + 1, corr=d["corr"], dx=d["dx"]) for d in cp]
    print(f"[{name}] {len(cp)} counterpart pairs ({time.time()-t0:.0f}s)")
    r = ph.analyze(df, n_boot=3, counterparts=cp or None, image_apix=APIX)
    print(f"[{name}] period {r.period:.1f} (true {P:.0f})")
    ang = ph.segment_angles(df, r, fold=FOLD)
    ok = ang.rlnAngleRot.notna().values
    # global gauge from the truth
    best = None
    for mu in (1, -1):
        for dpsi in (0.0, 180.0):
            d = mu * ang.rlnAngleRot.values[ok] - df.true_rot_est.values[ok]
            g = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(d * FOLD)).mean())) / FOLD
            e = (
                np.abs(wrap(d - g, 360.0 / FOLD)).mean()
                + np.abs(
                    wrap(ang.rlnAnglePsi.values[ok] + dpsi - df.true_psi.values[ok])
                ).mean()
            )
            if best is None or e < best[0]:
                best = (e, mu, dpsi, g)
    _, mu, dpsi, g = best
    used = r.params_used
    variants = {}
    variants["truth angles"] = truth
    b = pd.DataFrame(
        dict(
            rlnAngleRot=mu * ang.rlnAngleRot.values - g,
            rlnAngleTilt=90.0,
            rlnAnglePsi=ang.rlnAnglePsi.values + dpsi,
            rlnOriginXAngst=ang.rlnOriginXAngst.values,
            rlnOriginYAngst=ang.rlnOriginYAngst.values,
        )
    )
    variants["assigned (this work)"] = b
    # naive: class angle of the ring, Class2D psi and origins as found (after merge)
    az = dict(zip(r.class_ids, np.rad2deg(r.phases)))
    cnum = used.rlnClassNumber.astype(int).values
    nrot = np.array([az.get(c, np.nan) for c in cnum]) / FOLD
    d = mu * nrot[ok] - df.true_rot_est.values[ok]
    g2 = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(d * FOLD)).mean())) / FOLD
    nv = pd.DataFrame(
        dict(
            rlnAngleRot=mu * nrot - g2,
            rlnAngleTilt=90.0,
            rlnAnglePsi=used.rlnAnglePsi.values + dpsi,
            rlnOriginXAngst=used.rlnOriginXAngst.values,
            rlnOriginYAngst=used.rlnOriginYAngst.values,
        )
    )
    variants["naive (class angle, Class2D psi/origin)"] = nv
    if ABLATE:
        ax = np.deg2rad(df.true_psi.values)
        u = np.stack([np.cos(ax), -np.sin(ax)], 1)
        tc = df[["true_center_x", "true_center_y"]].values
        ec = df[["est_center_x", "est_center_y"]].values
        axial = ((ec - tc) * u).sum(1)
        ideal = tc + axial[:, None] * u
        cx = df[["rlnCoordinateX", "rlnCoordinateY"]].values * 1.0
        ideal_origin = cx - ideal
        t = pd.DataFrame(
            dict(
                rlnAngleRot=df.true_rot_est.values,
                rlnAngleTilt=90.0,
                rlnAnglePsi=df.true_psi.values,
                rlnOriginXAngst=ideal_origin[:, 0],
                rlnOriginYAngst=ideal_origin[:, 1],
            )
        )
        variants["  V1 true rot(est centre), psi; ideal origin"] = t
        x = t.copy()
        x["rlnAngleRot"] = b.rlnAngleRot.values
        variants["  V2 assigned rot; true psi, ideal origin"] = x
        y = t.copy()
        y["rlnAnglePsi"] = b.rlnAnglePsi.values
        variants["  V3 assigned psi; true rot, ideal origin"] = y
        z = t.copy()
        z["rlnOriginXAngst"] = b.rlnOriginXAngst.values
        z["rlnOriginYAngst"] = b.rlnOriginYAngst.values
        variants["  V4 synchronized origin; true rot, psi"] = z
        z2 = t.copy()
        z2["rlnOriginXAngst"] = used.rlnOriginXAngst.values
        z2["rlnOriginYAngst"] = used.rlnOriginYAngst.values
        variants["  V5 Class2D origin; true rot, psi"] = z2
    rnd = b.copy()
    rnd["rlnAngleRot"] = np.random.default_rng(0).uniform(0, 360, len(rnd))
    variants["random rot"] = rnd
    print(f"[{name}] gauge: mirror {mu:+d}, psi+{dpsi:.0f}, rot offset {g:.1f}")
    print(f"[{name}]   {'':40s} FSC0.5   FSC0.143   masked CC")
    for k, a in variants.items():
        m = a.notna().all(axis=1).values
        vol = (
            reconstruct("v", a[m].reset_index(drop=True), "segs.mrcs")
            if m.all()
            else None
        )
        if vol is None:
            # keep image numbering: unanalyzed segments cannot be used, write with their own index
            sub = a[m]
            particles_star(sub, str(W / "v.star"), image_stack="segs.mrcs")
            st = starfile.read(str(W / "v.star"))
            st["particles"]["rlnImageName"] = [
                f"{i+1:06d}@segs.mrcs" for i in np.flatnonzero(m)
            ]
            starfile.write(st, str(W / "v.star"), overwrite=True)
            run(
                f"cd {W} && relion_reconstruct --i v.star --o v.mrc --angpix {APIX} --j 20 2>&1 | tail -1"
            )
            vol = mrcfile.read(str(W / "v.mrc")).astype(np.float32)
        f, c = fsc(vol, ref)
        cc = masked_cc(vol, ref)
        print(
            f"[{name}]   {k:40s} {res_at(f, c, 0.5):6.1f}  {res_at(f, c, 0.143):8.1f}   {cc:6.3f}"
        )


if __name__ == "__main__":
    scenario("classes linked", flip_rate=0.3, misclass=0.1, n_views=24)
    scenario(
        "EMPIAR-10940-like (unlinked, bent, off-centre classes)",
        flip_rate=0.0,
        turned_offset_sd=10.0,
        bend=30.0,
        class_dy_sd=6.0,
        misclass=0.1,
        n_views=24,
    )
