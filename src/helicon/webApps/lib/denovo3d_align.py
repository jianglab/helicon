"""Align a 2D image against a side projection of the current 3D model.

Where the pairwise registration in ``denovo3d_register`` relates images to each
other, this relates an image to the reconstruction. The difference is not
cosmetic: a model projection can be made as long as the search needs and is
complete everywhere, so every offset is scored on a full overlap. The minimum
overlap floor, the prominence filter and the flip-graph synchronisation that
pairwise registration needs all exist because a pair of real images overlap
only partially, and none of them is needed here.

Axis convention
---------------
The helical axis runs along **x in 2D** and along **z in 3D**, as everywhere
else in helicon. So an image has the filament horizontal -- rows across it,
columns along it -- and a volume is ``(nz, ny, nx)`` with the axis at index 0.
A side projection is therefore ``np.sum(volume, axis=2).T``, which is how the
pipeline builds its own ``rec3d_x_proj``; anything that matches a model
projection against an input image has to agree with that or the comparison is
silently transposed. Rotation about the helical axis acts in the ``(ny, nx)``
plane, which for coordinates stacked as ``(X, Y, Z)`` is a rotation about
``"z"``.

Rotation is translation
-----------------------
For a volume that has been symmetrised at the candidate twist and rise, sliding
it along its axis and rotating it about that axis are the same operation:
shifting by ``dz`` equals rotating by ``twist * dz / rise``. Measured on a
synthetic helix by simulating the object at azimuth ``phi`` and correlating it
against a window of the object at azimuth 0, the best offset is
``-phi * (rise/twist) / apix`` modulo the pitch and the correlation there is
1.0000 at every azimuth tried (30, 90, 180, 270 and 359 degrees).

Three things follow, and they are the reason this module is small:

* an image needs exactly ONE alignment parameter, not an axial position and an
  azimuth. Where an image sits along the filament and which way it faces are
  the same unknown;
* the projection repeats after one period -- a whole turn, or ``1/csym`` of one
  -- so the search range is one period and a longer model re-tests azimuths it
  has already tested;
* no mirror branch is needed at all. Projecting along the viewing axis, a 180
  degree rotation about the helical axis sends (x, y, z) to (-x, -y, z), and
  the line integral is blind to the sign of x, so the projection is simply
  mirrored in y. Combined with the equivalence above that rotation is also half
  a period of axial shift, which gives, for ANY helical structure,

      M(-y, z) = M(y, z + period/2)

  -- a y-mirror IS a half-period translation, already inside the search range.
  Polarity is an x-mirror, and an x-mirror is an in-plane 180 degree rotation
  composed with a y-mirror, so once the y-mirror is absorbed by the shift what
  remains of polarity is a rotation near 180 degrees. Hence the search is over
  in-plane rotation about 0 and about 180, plus the offset, and nothing else.

  A structure with C2 about the helical axis is invariant under that rotation,
  so M(-y, z) = M(y, z): its side projection is mirror-symmetric across the
  filament and its period is half a pitch, which is what csym=2 means to
  ``period_pixel``. Searching such a structure at csym=1 leaves the azimuth
  determined only modulo 180 degrees, and the placements are then free to flip
  by half a turn between iterations.

What is checked, and against what
---------------------------------
Everything below was measured against a synthetic helix built independently of
this code, not against this code's own output:

* the long projection assembled from several slabs equals the one assembled
  from a single slab, cc 1.0000. Slabs after the first come from rotating the
  model rather than from symmetrising a taller volume, which keeps memory flat
  in the covered length -- and is only correct because of the equivalence
  above, so this doubles as a test of its sign;
* a window cut from the model at a known offset is placed back at that offset
  to 0.00 px, with correlation 1.0000;
* the azimuth reported for a placement is the rotation which, applied to the
  volume itself, reproduces the window: cc 1.0000, against 0.62-0.72 for the
  opposite sign. Both signs agree at zero offset and nowhere else, so this is
  what pins the convention.

Prominence is the wrong statistic here
--------------------------------------
``denovo3d_register`` screens pairs on peak prominence, the peak measured
against the whole correlation profile. That works there because a profile over
many offsets is mostly flat. Over exactly one period the profile is a single
broad hump -- measured width at half height 60-70 px for a 128 px image on a
288 px period -- so prominence reads about 2.0 even for a placement that is
exactly right, which is also ``denovo3d_register``'s rejection threshold. What
threatens a placement is a *second* hump, so the rival peak outside the first
one's shoulders is reported instead, and it is the number to judge a placement
by. On noiseless synthetic data the rival sits at 0.87 against a peak of 1.00:
a 128 px image covers only 44% of the period, and placement within a period is
correspondingly soft even when nothing is wrong.

In-plane rotation and the perpendicular shift are not searched
--------------------------------------------------------------
Only the axial offset. Aligning psi and dy to a shared reference makes the
images agree with each other while letting them drift together away from
horizontal; measured in ``denovo3d_joint``, that pushed the absolute tilt of
auto-transformed images from 0.07 to 0.72 degrees while the match correlation
improved. Absolute orientation stays where the tab's auto-transform put it, and
the match correlation must never be used as a convergence criterion.
"""

from __future__ import annotations

import logging

import numpy as np

import helicon

logger = logging.getLogger(__name__)

# Voxel budget for one symmetrised slab. Above this the long projection is
# assembled from several shorter slabs instead of one tall one.
MAX_SLAB_VOXELS = 24_000_000

# Shortest slab worth symmetrising, in pixels.
MIN_CHUNK = 16

# In-plane rotation searched about 0 and about 180 degrees. The two branches
# settle polarity; the range about each covers a class average that is not
# quite horizontal. No mirror is tried -- see align_to_model for why one is not
# needed against a reference spanning a whole period.
PSI_RANGE = 6.0
PSI_STEP = 1.0
PSI_STEP_FINE = 0.25


# ──────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────


def period_pixel(twist, rise, apix, csym=1, two_fold=False):
    """Axial period of the side projection, in pixels.

    Projecting the symmetrised volume at azimuth ``phi`` and at ``phi + twist``
    differ by one rise, so the projection repeats after a full turn -- or after
    ``1/csym`` of one, since ``csym`` makes azimuths ``360/csym`` apart
    identical. Searching a longer range than this re-tests azimuths already
    tested.

    ``two_fold`` halves it again, for a structure with a two-fold about the
    helical axis. This covers a 2-1 screw as well as a true C2, and it is a
    statement about the *projection*, not about the density:

    * for any helical structure, a 180 degree rotation about the axis mirrors
      the projection in y and is also half a period of axial shift, so
      ``M(-y, z) = M(y, z + period/2)``;
    * a 2-1 screw is the operation ``u = (180 + twist/2, rise/2)``, which
      satisfies ``u * u = (twist, rise)``. In projection its extra half-twist
      rotation is a shift of ``-rise/2`` and its translation is ``+rise/2``, so
      the two cancel and ``u`` acts exactly as the bare 180 degree rotation.
      Invariance under ``u`` therefore means ``M(-y, z) = M(y, z)``.

    Put together those give ``M(y, z) = M(y, z + period/2)``: the projection
    repeats twice as often. Nothing is lost by searching only half, because the
    two placements it merges are the same placement -- for such a structure
    azimuth ``phi`` and ``phi + 180`` describe an identical object.

    This is emphatically NOT the same as passing ``csym=2``. That imposes a
    rotational symmetry on the reconstruction, which a 2-1 screw does not have;
    measured on EMPIAR-10940 it moved the recovered twist from 1.20 to 1.05.
    The symmetry is also deliberately left unimposed so that recovering it
    validates the reconstruction. ``two_fold`` changes only where the placement
    search looks.
    """
    csym = max(1, int(csym))
    period = abs(rise) * 360.0 / (abs(twist) * csym) / apix
    return period / 2.0 if two_fold else period


def dx_to_phi(dx_pixel, twist, rise, apix):
    """Azimuth (degrees) of an image sitting ``dx_pixel`` along the model.

    For a helically symmetric object an axial shift and an azimuthal rotation
    are the same operation, which is why one number per image is enough.
    """
    return -float(dx_pixel) * apix * twist / rise


def phi_to_dx(phi_degree, twist, rise, apix):
    """Inverse of :func:`dx_to_phi`."""
    return -float(phi_degree) * rise / twist / apix


# ──────────────────────────────────────────────────────────────────────────
# Model projection
# ──────────────────────────────────────────────────────────────────────────


def long_side_projection(
    rec3d,
    apix3d,
    twist,
    rise,
    csym,
    out_apix,
    out_ny,
    length_pixel,
    cpu=1,
):
    """Side projection of the symmetrised model, ``length_pixel`` columns wide.

    Rows are across the filament and columns along it -- the helical axis is
    x in 2D -- and column ``M // 2`` is the model's own axial origin, the same
    convention the tab's ``x`` projection uses.

    The projection is assembled from one or more symmetrised slabs. A slab
    covers as many columns as the voxel budget allows; when that is the whole
    length there is a single slab and nothing is joined. Further slabs are
    produced by rotating the model about its axis rather than by symmetrising a
    taller volume, which keeps the memory flat in the covered length -- and is
    exact, because that rotation is what an axial shift is.
    """
    length_pixel = int(np.ceil(length_pixel))
    length_pixel += length_pixel % 2
    out_ny = int(out_ny)
    out_ny += out_ny % 2

    per_column = max(1, out_ny * out_ny)
    chunk = int(min(length_pixel, max(MIN_CHUNK, MAX_SLAB_VOXELS // per_column)))
    chunk += chunk % 2
    n_chunks = int(np.ceil(length_pixel / chunk))

    chunks = []
    for c in range(n_chunks):
        zc = (c - (n_chunks - 1) / 2.0) * chunk
        phi_c = dx_to_phi(zc, twist, rise, out_apix) if zc else 0.0
        src = helicon.transform_map(rec3d, rot=phi_c) if phi_c else rec3d
        slab = helicon.apply_helical_symmetry(
            data=np.ascontiguousarray(src, dtype=np.float32),
            apix=apix3d,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=max(1, int(csym)),
            new_size=(chunk, out_ny, out_ny),
            new_apix=out_apix,
            cpu=cpu,
        )
        chunks.append(np.sum(slab, axis=2).T)

    proj = np.concatenate(chunks, axis=1)
    m = proj.shape[1]
    if m > length_pixel:
        lo = m // 2 - length_pixel // 2
        proj = proj[:, lo : lo + length_pixel]
    return np.ascontiguousarray(proj, dtype=np.float32)


def mirror_shift_pixel(twist, rise, apix, csym=1):
    """Axial shift that accompanies the y-mirror, in pixels.

    Zero for an even ``csym``, where the mirror is a symmetry of the projection
    on its own. For an odd ``csym`` it is half the projection period,
    ``pitch/(2*csym)``, because ``pitch/2`` is then not a whole number of
    periods and the remainder has to be taken up by a shift along the axis.

    ``csym=1`` is odd, so this returns half a pitch -- which is the honest
    answer and also says why a plain helix shows no self-mirror symmetry: no
    window is half a pitch long, so there is no shift within an image that
    could match it to its own mirror.
    """
    period = abs(rise) * 360.0 / (abs(twist) * max(1, int(csym))) / apix
    return 0.0 if int(csym) % 2 == 0 else period / 2.0


def model_length_pixel(twist, rise, apix, csym, image_width, two_fold=False):
    """How wide the model projection has to be to test every azimuth once.

    One period, so that every distinct azimuth appears, plus the image width,
    so that every one of them can be tested with the image lying entirely
    inside the model and no offset is scored on a partial overlap.
    """
    period = period_pixel(twist, rise, apix, csym, two_fold)
    return int(np.ceil(period)) + int(image_width)


# ──────────────────────────────────────────────────────────────────────────
# Alignment
# ──────────────────────────────────────────────────────────────────────────


def _match_height(b, ny):
    """Centre ``b`` into a canvas ``ny`` tall."""
    h = b.shape[0]
    if h == ny:
        return b
    if h > ny:
        top = (h - ny) // 2
        return b[top : top + ny]
    out = np.zeros((ny, b.shape[1]), dtype=b.dtype)
    top = (ny - h) // 2
    out[top : top + h] = b
    return out


def sliding_ncc(model, image):
    """Normalised correlation of ``image`` against every full-overlap window.

    Returns ``(offsets, corr)``; offset ``k`` means the image matches
    ``model[:, k:k+L]``. Unlike the pairwise registration this module's
    counterpart does, every offset here has complete overlap -- the model
    projection is synthetic and covers the whole period -- so there is no
    minimum-overlap floor to set and no offset that scores well by having
    little to disagree about.
    """
    model = np.asarray(model, dtype=np.float64)
    image = _match_height(np.asarray(image, dtype=np.float64), model.shape[0])
    ny, m = model.shape
    l = image.shape[1]
    if l > m:
        raise ValueError(f"image ({l} px) is wider than the model ({m} px)")

    v = image - image.mean()
    vn = float(np.sqrt((v * v).sum()))
    n = m + l
    num = np.fft.irfft(
        np.fft.rfft(model, n=n, axis=1) * np.conj(np.fft.rfft(v, n=n, axis=1)),
        n=n,
        axis=1,
    ).sum(axis=0)[: m - l + 1]

    col1 = np.concatenate([[0.0], np.cumsum(model.sum(axis=0))])
    col2 = np.concatenate([[0.0], np.cumsum((model**2).sum(axis=0))])
    k = np.arange(m - l + 1)
    s1 = col1[k + l] - col1[k]
    s2 = col2[k + l] - col2[k]
    var = np.maximum(s2 - s1 * s1 / (ny * l), 0.0)
    denom = np.sqrt(var) * vn
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, num / denom, -2.0)
    return k, corr


def _peak(offsets, corr):
    """Sub-pixel peak position, its height, prominence, width and best rival."""
    i = int(np.argmax(corr))
    x = float(offsets[i])
    if 0 < i < len(corr) - 1:
        cl, cm, cr = corr[i - 1], corr[i], corr[i + 1]
        den = cl - 2.0 * cm + cr
        if den < -1e-12:
            d = 0.5 * (cl - cr) / den
            if abs(d) <= 0.5:
                x += float(d)
    valid = corr[corr > -2.0]
    if valid.size > 8 and valid.std() > 1e-9:
        prom = float((corr[i] - valid.mean()) / valid.std())
    else:
        prom = 0.0
    # Width at half the peak's height above the profile's own floor. A class
    # average that pools a range of azimuths matches a range of offsets, so a
    # broad peak is the signature of exactly the smearing that a single azimuth
    # per image cannot represent.
    floor = float(valid.min()) if valid.size else 0.0
    half = floor + 0.5 * (float(corr[i]) - floor)
    lo = i
    while lo > 0 and corr[lo] > half:
        lo -= 1
    hi = i
    while hi < len(corr) - 1 and corr[hi] > half:
        hi += 1
    width = float(offsets[hi] - offsets[lo])
    # The best rival: the highest correlation outside the peak's own shoulders.
    # Prominence measures the peak against the whole profile, which over one
    # period is a single smooth hump, so it reads low even for an unambiguous
    # placement. What actually threatens the placement is a *second* hump, and
    # only this sees it.
    outside = np.concatenate([corr[:lo], corr[hi + 1 :]])
    outside = outside[outside > -2.0]
    rival = float(outside.max()) if outside.size else -2.0
    return x, float(corr[i]), prom, width, rival


def _rotate(image, psi):
    """Rotate in the image plane by ``psi`` degrees about the image centre."""
    if not psi:
        return np.asarray(image, dtype=np.float32)
    return helicon.transform_image(
        image=np.asarray(image, dtype=np.float32), rotation=float(psi)
    )


def align_to_model(
    image,
    model,
    twist,
    rise,
    apix,
    psi_range=PSI_RANGE,
    psi_step=PSI_STEP,
    refine=True,
):
    """Place ``image`` along ``model`` and report the azimuth that implies.

    Against a reference spanning a whole period, an in-plane rotation and a
    shift are the only transforms needed -- no mirror branch. The reason is the
    identity above: a y-mirror of a helical side projection is a half-period
    translation, which the offset search already covers. Polarity, the other
    flip, is an x-mirror, and an x-mirror is an in-plane 180 degree rotation
    composed with a y-mirror -- so once the y-mirror is absorbed by the shift,
    polarity is just a rotation near 180 degrees.

    Hence ``psi`` is searched in a small range about 0 and about 180 rather
    than a mirror being tried: the two branches settle polarity, and the range
    about each absorbs the couple of degrees by which a class average is not
    quite horizontal.

    The perpendicular shift is deliberately not searched. Aligning dy to a
    shared reference makes the images agree with each other while letting them
    drift together off-axis, the reference bias measured in ``denovo3d_joint``
    (absolute tilt 0.07 -> 0.72 degrees while the match correlation improved).
    Centring stays with the tab's auto-transform, and the match correlation
    must never be used as a convergence criterion.

    Returns ``{dx, phi, psi, reversed, corr, prominence, width, rival}``, where
    ``dx`` is the offset of the image centre from the model centre in pixels
    and ``reversed`` says whether the winning branch was the one near 180
    degrees, i.e. the opposite polarity.
    """
    image = np.asarray(image, dtype=np.float32)
    model = np.asarray(model, dtype=np.float32)
    m = model.shape[1]
    l = image.shape[1]

    def scan(centre, half, step):
        best = None
        for psi in np.arange(centre - half, centre + half + step / 2, step):
            offsets, corr = sliding_ncc(model, _rotate(image, psi))
            k, c, prom, width, rival = _peak(offsets, corr)
            if best is None or c > best[1]:
                best = (k, c, prom, width, rival, float(psi))
        return best

    best = None
    for centre in (0.0, 180.0):
        cand = scan(centre, psi_range, psi_step)
        if best is None or cand[1] > best[1]:
            best = cand

    if refine and psi_step > PSI_STEP_FINE:
        best = min(
            [best, scan(best[5], psi_step, PSI_STEP_FINE)],
            key=lambda b: -b[1],
        )

    k, c, prom, width, rival, psi = best
    dx = k + l / 2.0 - m // 2
    return dict(
        dx=float(dx),
        phi=float(dx_to_phi(dx, twist, rise, apix) % 360.0),
        psi=float(psi),
        reversed=bool(abs((psi + 90.0) % 360.0 - 90.0) > 90.0),
        corr=float(c),
        prominence=float(prom),
        width=float(width),
        rival=float(rival),
    )
