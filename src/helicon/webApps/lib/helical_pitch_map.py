"""A 3D map from the class averages, at the azimuthal angles of the ring.

The AbInitio3D tab places every selected class on a ring, one azimuthal angle
per class, and reads the repeat distance from it. Given the cyclic symmetry that
turns the repeat into a pitch, and the handedness, that is everything needed to
assemble the class averages into one helically symmetric map -- the same
joint reconstruction the denovo3D tab's projection-matching mode does, only with
the azimuths given instead of searched for.

Conventions
-----------
* The repeat distance ``P`` is the axial period of the side projection, and the
  pitch is ``csym * P``. The twist per rise is ``360 * rise / (csym * P)``,
  negative for a left-handed helix, as in RELION (amyloid filaments are
  left-handed, with negative twists).
* Every class average is straightened, and turned by 180 degrees where the
  ring's axial direction points to the left, so that the axial coordinate
  increases to the right in all of them.
* A class at ring angle ``phi`` (360 degrees per repeat) is a view at the
  azimuth ``sign(twist) * phi / csym``, in the convention of the denovo3D
  tab.
* The hand cannot be told from the projections: the mirror image of a helix
  gives the same side views at mirrored azimuths. It is an input.
"""

from __future__ import annotations

import logging

import numpy as np

import helicon

from . import helix_transform as HT

logger = logging.getLogger(__name__)


def _wrap(a, period=360.0):
    return (a + period / 2.0) % period - period / 2.0


def _measure(image):
    """Tilt (degrees) and row offset (pixels) of the filament in an average.

    From the second moments of what stands above the background: the principal
    axis of a long thin filament is along it, and its centroid row is where it
    sits across the image. Only used to tell which way a correction acts.
    """
    x = np.asarray(image, dtype=float)
    border = np.concatenate([x[:2].ravel(), x[-2:].ravel()])
    x = x - np.median(border)
    x[x < 0.3 * x.max()] = 0.0
    total = x.sum()
    if total <= 0:
        return 0.0, 0.0
    yy, xx = np.mgrid[: x.shape[0], : x.shape[1]]
    cy = (x * yy).sum() / total
    cx = (x * xx).sum() / total
    vyy = (x * (yy - cy) ** 2).sum() / total
    vxx = (x * (xx - cx) ** 2).sum() / total
    vxy = (x * (yy - cy) * (xx - cx)).sum() / total
    tilt = 0.5 * np.rad2deg(np.arctan2(2.0 * vxy, vxx - vyy))
    return float(tilt), float(cy - (x.shape[0] - 1) / 2.0)


def _metadata_signs(raw, tilt, dy):
    """Which way the metadata corrections act, from the averages themselves.

    Returns ``(k_rot, k_dy)``, each +1 or -1: the sign under which applying the
    correction leaves the filaments closer to horizontal and to the middle. A
    correction too small to see leaves the sign at +1.
    """
    signs = []
    for values, index in ((tilt, 0), (dy, 1)):
        big = [i for i, v in enumerate(values) if np.isfinite(v) and abs(v) > 0.5]
        best = (np.inf, 1.0)
        for k in (1.0, -1.0):
            resid = []
            for i in big:
                if index == 0:
                    x = HT.apply_transform(raw[i], rotation=k * values[i])
                else:
                    x = HT.apply_transform(raw[i], shift_y=k * values[i])
                resid.append(abs(_measure(x)[index]))
            if resid and np.median(resid) < best[0]:
                best = (np.median(resid), k)
        signs.append(best[1])
    return signs[0], signs[1]


def straighten_classes(
    images,
    zdir,
    target_apix,
    apix,
    max_height=240.0,
    tilt=None,
    shift_y=None,
    counts=None,
    min_count=50,
):
    """Class averages with the filament horizontal and the axial direction to the right.

    By default the rotation and the centring of each average are estimated from
    the image. They can instead be given per class, from the segments that make
    the average up: the picks have random errors in direction and position, so
    the median over a class of how its segments' in-plane angles differ from
    their picking angles (``tilt``), and of how far their refined centres lie
    across the filament (``shift_y``), should measure how far the average is
    from horizontal and from centred.

    Measured on EMPIAR-10940 that did *not* beat the image estimate, and it is
    not the default. The segment medians put every average within 0.4 degrees of
    horizontal and 1.6 A of the middle, but RELION's psi is restrained to the
    prior there (``--sigma_psi 2``), so that is nearly true by construction; the
    image estimate rotates the averages by up to 19 degrees and gave the better
    map (joint fit 0.68 against 0.55, correlation with the deposited map 0.68
    against 0.64). Centring made no difference (the image estimate moves them by
    1.2 pixels, sd). The medians may be the better guide for classifications
    that do not restrain psi; that is untested.

    Parameters
    ----------
    images : sequence of 2D arrays
        The class averages.
    zdir : sequence of float
        Per class, the direction in which the ring's axial coordinate increases
        in the average's own frame, in degrees (``PhasePitchResult.class_zdir``).
    target_apix : float
        Pixel size to work at, in A. Not smaller than ``apix``.
    apix : float
        Pixel size of the averages, in A.
    max_height : float, optional
        Rows across the filament beyond this, in A, are cropped away: they hold
        only background and cost solving time. Defaults to 240.
    tilt : sequence of float, optional
        Per class, the median deviation of the segments' psi from their priors,
        in degrees (``PhasePitchResult.class_tilt``).
    shift_y : sequence of float, optional
        Per class, the median offset of the segments' centres across the
        filament, in pixels of ``images``.
    counts : sequence of int, optional
        Segments per class. A class with fewer than ``min_count`` uses the
        image estimate.

    Returns
    -------
    list of np.ndarray
        Images of one shape, background at zero, scaled to a maximum of 1.
    """
    raw = [np.asarray(im, dtype=np.float32) for im in images]
    at = HT.auto_transform(raw)
    per_image = [list(t) for t in at.per_image]
    if tilt is not None:
        tilt = np.asarray(tilt, dtype=float)
        shift_y = np.zeros(len(raw)) if shift_y is None else np.asarray(shift_y, float)
        enough = (
            np.ones(len(raw), bool)
            if counts is None
            else np.asarray(counts) >= min_count
        )
        good = enough & np.isfinite(tilt) & np.isfinite(shift_y)
        k_rot, k_dy = _metadata_signs(
            raw,
            np.where(good, tilt, np.nan),
            np.where(good, shift_y, np.nan),
        )
        for i in np.flatnonzero(good):
            per_image[i] = [k_rot * tilt[i], k_dy * shift_y[i]]
    zdir = np.asarray(zdir, dtype=float)
    rotations = np.array([r for r, _ in per_image], dtype=float)
    # which way the estimated rotation turns the direction is a convention of the
    # in-plane angles; take the one that leaves the directions along the axis
    best = None
    for kappa in (1.0, -1.0):
        after = _wrap(zdir - kappa * rotations)
        resid = np.minimum(np.abs(_wrap(after)), np.abs(_wrap(after - 180.0)))
        if best is None or np.median(resid) < best[0]:
            best = (np.median(resid), kappa, after)
    _, _, after = best
    out = []
    target_apix = max(float(target_apix), float(apix))
    for im, (rot, sy), a in zip(raw, per_image, after):
        x = HT.apply_transform(im, rotation=rot, shift_y=sy, crop_size=at.crop_size)
        if abs(_wrap(a)) > 90.0:
            x = x[::-1, ::-1]
        if target_apix > apix:
            x = helicon.down_scale(x, target_apix, apix)
        out.append(np.ascontiguousarray(x, dtype=np.float32))
    ny = min(x.shape[0] for x in out)
    ny = min(ny, max(16, 2 * int(round(max_height / target_apix / 2))))
    nx = min(x.shape[1] for x in out)
    out = [helicon.crop_center(x, shape=(ny, nx)) for x in out]
    prepared = []
    for x in out:
        border = np.concatenate([x[:2].ravel(), x[-2:].ravel()])
        x = x - np.median(border)
        x = helicon.threshold_data(x, thresh_fraction=0.05)
        m = float(x.max())
        prepared.append(x / m if m > 0 else x)
    return prepared


def _tile_canvas(images, centres, length):
    """The images averaged into one canvas, each centred on its column."""
    ny, nx = images[0].shape
    acc = np.zeros((ny, length))
    weight = np.zeros(length)
    ramp = np.minimum(1.0, np.minimum(np.arange(nx) + 1, nx - np.arange(nx)) / 8.0)
    for image, c in zip(images, centres):
        left = int(round(c - nx / 2.0))
        lo, hi = max(0, left), min(length, left + nx)
        if hi <= lo:
            continue
        w = ramp[lo - left : hi - left]
        acc[:, lo:hi] += image[:, lo - left : hi - left] * w
        weight[lo:hi] += w
    out = np.zeros_like(acc)
    covered = weight > 0
    out[:, covered] = acc[:, covered] / weight[covered]
    return out


def _align_all(images, volume, twist, rise, apix3d, apix2d, csym, cpu=1):
    """Place every image on the side projection of the map, as denovo3D does.

    Returns the alignments (``dx``, ``phi``, ``psi`` and ``corr`` each) against a
    projection of one repeat plus an image width.
    """
    from .denovo3d_align import (
        align_to_model,
        long_side_projection,
        model_length_pixel,
    )

    ny, nx = images[0].shape
    length = model_length_pixel(twist, rise, apix2d, csym, nx, False)
    length += length % 2
    model = long_side_projection(
        volume, apix3d, twist, rise, csym, apix2d, ny, length, cpu
    )
    return [align_to_model(im, model, twist, rise, apix2d) for im in images]


def _turn(image, psi):
    """Only the half turn of the alignment, when a class average sits reversed."""
    return image[::-1, ::-1] if abs((psi + 90.0) % 360.0 - 90.0) > 90.0 else image


def _wrap_dx(dx, period):
    return (dx + period / 2.0) % period - period / 2.0


def _fbp_slice(job):
    sino, angles, size = job
    from skimage.transform import iradon

    return iradon(sino, theta=angles, filter_name="ramp", output_size=size, circle=True)


def backproject(images, azimuths, twist, rise, csym, apix, helical_sym_order=1, cpu=1):
    """Filtered back-projection of side views, each used ``helical_sym_order`` times.

    The class averages are side views (tilt 90): row ``v`` of an image across the
    filament, column ``z`` along it. Each ``z`` slice of the map is then a 2D
    tomographic reconstruction, from the column ``z`` of every image at the
    azimuth of that image. A helical structure gives more: column ``z - n * rise``
    of an image is a view of the same slice at the azimuth turned by ``n * twist``,
    so each image can be used once per ``n`` -- ``helical_sym_order`` times,
    symmetrically about zero -- which fills in the azimuths a few class
    averages leave empty. With 1, an image contributes only its own column, at
    its own azimuth, and nothing about the twist and rise is imposed.

    Parameters
    ----------
    images : sequence of 2D arrays
        Class averages with the filament horizontal and ``z`` to the right.
    azimuths : sequence of float
        Azimuth of each, in degrees, in the convention of the joint solver
        (``sign(twist) * ring_angle / csym``).
    twist, rise : float
        Helical twist (degrees) and rise (A).
    csym : int
        Cyclic symmetry to impose; each view is used ``csym`` times, turned by
        ``360 / csym``. 1 imposes none.
    apix : float
        Pixel size of the images, in A.
    helical_sym_order : int, optional
        Number of times each image is used, along ``z``. Defaults to 1.

    Returns
    -------
    np.ndarray
        The map ``(nz, ny, ny)`` on the images' pixel grid, ``nz`` the image
        length.
    """
    from concurrent.futures import ThreadPoolExecutor

    ny, nx = images[0].shape
    nz = nx
    order = int(max(1, helical_sym_order))
    reps = np.arange(order) - (order - 1) // 2
    jobs = []
    for iz in range(nz):
        z = (iz - nz / 2 + 0.5) * apix
        profiles, angles = [], []
        for image, az in zip(images, azimuths):
            for n in reps:
                colf = (z - n * rise) / apix + nx / 2 - 0.5
                if colf < 0 or colf > nx - 1:
                    continue
                i0 = int(np.floor(colf))
                f = colf - i0
                i1 = min(i0 + 1, nx - 1)
                column = (1 - f) * image[:, i0] + f * image[:, i1]
                for c in range(max(1, int(csym))):
                    profiles.append(column)
                    angles.append(-(az + c * 360.0 / max(1, int(csym)) + n * twist))
        jobs.append((np.stack(profiles, 1), np.array(angles), ny) if profiles else None)
    volume = np.zeros((nz, ny, ny), dtype=np.float32)
    todo = [(i, j) for i, j in enumerate(jobs) if j is not None]
    with ThreadPoolExecutor(max(1, int(cpu))) as pool:
        for (i, _), out in zip(todo, pool.map(_fbp_slice, [j for _, j in todo])):
            volume[i] = out
    return volume


def _sym_long(volume, twist, rise, csym, apix, length_pixel, cpu=1):
    """The central third of ``volume`` symmetrized to a longer volume."""
    nz, ny, nx = volume.shape
    third = volume[nz // 3 : nz - nz // 3]
    return helicon.apply_helical_symmetry(
        np.ascontiguousarray(third, dtype=np.float32),
        apix,
        twist,
        rise,
        int(max(csym, 1)),
        new_size=(int(length_pixel), ny, nx),
        new_apix=apix,
        cpu=cpu,
    )


def _side_projection(long_volume):
    """Side projection at azimuth 0 of a volume: rows across, columns along."""
    return long_volume.sum(axis=1).T.copy()


def _align_to(images, model, twist, rise, apix):
    from .denovo3d_align import align_to_model

    return [align_to_model(im, model, twist, rise, apix) for im in images]


def reconstruct_map(
    images,
    ring_angles,
    repeat,
    csym,
    rise,
    left_handed=True,
    apix=1.0,
    helical_sym_order=1,
    impose_csym=False,
    method="joint",
    length_in_rises=3,
    algorithm=None,
    rounds=1,
    refine_psi=True,
    output_box=None,
    output_apix=None,
    cpu=1,
):
    """Assemble straightened class averages into one map.

    Parameters
    ----------
    images : sequence of 2D arrays
        From :func:`straighten_classes`, at ``apix``.
    ring_angles : sequence of float
        The class angles on the ring, in degrees, 360 per repeat
        (``PhasePitchResult.phases`` in degrees).
    repeat : float
        Repeat distance, in A.
    csym : int
        Cyclic symmetry: the pitch is ``csym`` times the repeat. Always used to
        turn the repeat into the twist and the ring angles into azimuths.
    rise : float
        Helical rise, in A.
    left_handed : bool, optional
        Handedness of the helix; the twist is negative when it is. Defaults to
        True.
    apix : float
        Pixel size of ``images``, in A.
    helical_sym_order : int, optional
        Back-projection only; the joint fit imposes the full helical symmetry.
        1 imposes no helical symmetry: every image is used
        once. Larger, every image is used that many times along the filament
        according to the twist, rise and cyclic symmetry, and the map is then
        symmetrized again from the central third of its z sections. Defaults to
        1.
    impose_csym : bool, optional
        Whether the cyclic symmetry is also imposed in the reconstruction. Left
        off, the apparent symmetry of the z section is an independent check on
        the reconstruction. Defaults to False.
    method : {"joint", "backprojection"}, optional
        ``"joint"`` is the denovo3D joint fit (elastic net unless ``algorithm``
        says otherwise), which imposes the full helical symmetry (and the cyclic
        symmetry, if ``impose_csym``) and takes ``algorithm`` and
        ``length_in_rises`` into account; it agreed better with
        the deposited map of EMPIAR-10940 (0.73 against 0.67 at a helical sym
        order of 81). ``"backprojection"`` uses each average
        ``helical_sym_order`` times. Defaults to the joint fit.
    rounds : int, optional
        With 2, each average is aligned to the side projection of the first
        map -- azimuth and in-plane rotation, by the sliding correlation the
        denovo3D and helicalProjection tabs use -- and the map is rebuilt from
        the aligned averages. Defaults to 1: measured against the deposited map
        of EMPIAR-10940 the second round raised the fit to the averages and
        lowered the agreement with the map, the reference bias documented in
        ``denovo3d_joint``.
    refine_psi : bool, optional
        Whether the second round also takes the alignments' small in-plane
        rotations, not only the azimuths and any half turn. Defaults to True.
    output_box, output_apix : optional
        Size (voxels, cubic) and pixel size (A) of the exported map,
        ``volume_out``, so that it can serve as a reference in RELION.

    Returns
    -------
    dict
        ``volume`` (z, y, x), ``volume_out`` (or None) with its pixel size
        ``apix_out``, ``apix``, ``twist`` (degrees), ``score`` (the joint fit for
        the joint method, else the mean correlation of the images with the
        projection of the map), ``projection`` (the side projection over about
        1.2 pitches), ``tiles`` (the averages at their alignments on the same
        canvas), ``z_view`` (the middle z section), ``phis``, ``alignments``.
    """
    csym = int(max(csym, 1))
    sign = -1.0 if left_handed else 1.0
    twist = sign * 360.0 * rise / (csym * float(repeat))
    phis = [sign * float(a) / csym for a in ring_angles]
    if method == "joint":
        return _reconstruct_joint(
            images,
            phis,
            twist,
            repeat,
            csym if impose_csym else 1,
            rise,
            apix,
            length_in_rises,
            algorithm,
            rounds,
            refine_psi,
            output_box,
            output_apix,
            cpu,
        )
    ny, nx = images[0].shape
    apix = float(apix)
    pitch = csym * float(repeat)
    length = int(np.ceil(1.2 * pitch / apix))
    length += length % 2
    c_used = csym if impose_csym else 1
    period = float(repeat) / apix
    order = int(max(1, helical_sym_order))

    def build(imgs, azimuths):
        volume = backproject(imgs, azimuths, twist, rise, c_used, apix, order, cpu)
        long_volume = _sym_long(volume, twist, rise, c_used, apix, length, cpu)
        return volume, long_volume

    volume, long_volume = build(images, phis)
    used = list(images)
    model = _side_projection(long_volume)
    alignments = _align_to(used, model, twist, rise, apix)
    if rounds >= 2:
        # the alignments give each image's offset along the projection; that is
        # its azimuth up to the sign of the convention, which the offsets
        # already found fix
        from .denovo3d_align import _rotate

        dxs = np.array([a["dx"] for a in alignments])
        predicted = np.array([p * rise / (twist * apix) for p in phis])
        best = min(
            (1.0, -1.0),
            key=lambda c: np.abs(_wrap_dx(dxs - c * predicted, period)).sum(),
        )
        phis = [best * float(dx) * twist * apix / rise for dx in dxs]
        used = [
            _rotate(im, a["psi"]) if refine_psi else _turn(im, a["psi"])
            for im, a in zip(images, alignments)
        ]
        volume, long_volume = build(used, phis)
        model = _side_projection(long_volume)
        alignments = _align_to(used, model, twist, rise, apix)
    centres = [length / 2.0 + _wrap_dx(a["dx"], period) for a in alignments]
    tiles = _tile_canvas(used, centres, length)
    projection = model
    if projection.shape[0] != ny:
        projection = helicon.crop_center(projection, shape=(ny, projection.shape[1]))

    final = volume
    if order > 1:
        final = _sym_long(volume, twist, rise, c_used, apix, volume.shape[0], cpu)
    volume_out = None
    if output_box:
        volume_out = _to_box(final, apix, int(output_box), float(output_apix or apix))
    mid = final.shape[0] // 2
    corr = float(np.mean([a["corr"] for a in alignments]))
    return dict(
        volume=final,
        volume_out=volume_out,
        apix=apix,
        apix_out=float(output_apix or apix),
        twist=twist,
        score=corr,
        per_image=[float(a["corr"]) for a in alignments],
        projection=projection,
        tiles=tiles,
        z_view=final[mid],
        phis=phis,
        alignments=alignments,
    )


def _to_box(volume, apix, box, out_apix):
    """A volume resampled to a cubic box of ``box`` voxels at ``out_apix``."""
    from scipy import ndimage

    v = ndimage.zoom(volume, apix / out_apix, order=1) if out_apix != apix else volume
    out = np.zeros((box, box, box), dtype=np.float32)
    dst = tuple(
        slice(max(0, (box - s) // 2), max(0, (box - s) // 2) + min(box, s))
        for s in v.shape
    )
    src = tuple(
        slice(max(0, (s - box) // 2), max(0, (s - box) // 2) + min(box, s))
        for s in v.shape
    )
    out[dst] = v[src]
    return out


def _reconstruct_joint(
    images,
    phis,
    twist,
    repeat,
    csym,
    rise,
    apix,
    length_in_rises,
    algorithm,
    rounds,
    refine_psi,
    output_box,
    output_apix,
    cpu,
):
    """The denovo3D joint fit: every image explained by one helically symmetric map.

    The solver defaults to elastic net, as the denovo3D tab's reconstruction does;
    the Gaussian basis (``dict(model="gauss")``) is faster and smoother.
    """
    from .denovo3d_align import _rotate, long_side_projection
    from .denovo3d_jointsolve import joint_reconstruct

    algorithm = algorithm or dict(model="elasticnet", l1_ratio=0.5)
    ny, nx = images[0].shape
    apix2d = float(apix)
    apix3d = apix2d
    d3 = int(round(ny * apix2d / apix3d))
    d3 += d3 % 2
    nz = max(int(np.ceil(rise / apix3d)), int(np.ceil(length_in_rises * rise / apix3d)))
    nz += nz % 2
    geometry = dict(
        scale2d_to_3d=apix2d / apix3d,
        reconstruct_diameter_2d_pixel=ny + ny % 2,
        reconstruct_diameter_3d_pixel=d3,
        reconstruct_length_2d_pixel=nx + nx % 2,
        reconstruct_length_3d_pixel=nz,
        reconstruct_diameter_3d_inner_pixel=0,
        sym_oversample=-1,
        interpolation="linear",
        positive_constraint=-1,
    )

    def solve(imgs, azimuths):
        return joint_reconstruct(
            imgs,
            azimuths,
            twist_degree=twist,
            rise_pixel=rise / apix3d,
            csym=csym,
            algorithm=algorithm,
            target_apix2d=apix2d,
            cpu=cpu,
            **geometry,
        )

    volume, info = solve(images, phis)
    used = list(images)
    alignments = _align_all(used, volume, twist, rise, apix3d, apix2d, csym, cpu)
    if rounds >= 2:
        used = [
            _rotate(im, a["psi"]) if refine_psi else _turn(im, a["psi"])
            for im, a in zip(images, alignments)
        ]
        phis = [float(a["phi"]) for a in alignments]
        volume, info = solve(used, phis)
        alignments = _align_all(used, volume, twist, rise, apix3d, apix2d, csym, cpu)
    pitch = abs(360.0 * rise / twist)
    length = int(np.ceil(1.2 * pitch / apix2d))
    length += length % 2
    projection = long_side_projection(
        volume, apix3d, twist, rise, csym, apix2d, ny, length, cpu
    )
    period = pitch / max(csym, 1) / apix2d
    centres = [length / 2.0 + _wrap_dx(a["dx"], period) for a in alignments]
    tiles = _tile_canvas(used, centres, length)
    nz_per_rise = max(1, int(np.ceil(rise / max(apix3d, 1e-6))))
    z0 = max(0, volume.shape[0] // 2 - nz_per_rise // 2)
    z_view = np.sum(volume[z0 : z0 + nz_per_rise], axis=0)
    volume_out = None
    if output_box:
        volume_out = helicon.apply_helical_symmetry(
            np.ascontiguousarray(volume, dtype=np.float32),
            apix3d,
            twist,
            rise,
            csym,
            new_size=(int(output_box),) * 3,
            new_apix=float(output_apix or apix3d),
            cpu=cpu,
        )
    return dict(
        volume=volume,
        volume_out=volume_out,
        apix=apix3d,
        apix_out=float(output_apix or apix3d),
        twist=twist,
        score=info["score"],
        per_image=info["per_image"],
        projection=projection,
        tiles=tiles,
        z_view=z_view,
        phis=phis,
        alignments=alignments,
    )
