"""Automatic registration of several 2D class averages along a filament.

Manual stitching asks the user to drag each image into place. This does the
same job from the images: register every usable pair, reconcile the pairwise
measurements into one transform per image, and composite them into a single
longer image.

Why this is worth doing, and what it cannot do
----------------------------------------------
Class averages of one filament occupy different parts of the helical pitch, so
a longer composite spans more of a turn than any single image and gives the
twist search a longer lever arm. Measured on EMPIAR-10940 (twist 1.2 deg,
pitch 288 px at 4.944 A/pixel, images 128 px wide): a single image covers 44%
of the pitch, and 18 registrable classes together span 202 px, or 70% -- a
1.58x gain. Useful, but far short of the 5x that five images might suggest,
because the classes cluster in a few axial registers instead of tiling the
pitch. The diagnostics report that span so a dataset can be judged before any
of this is trusted.

Registration is deliberately *twist-independent*: pairs are registered by
correlation alone, and the offsets are then used as axial placements.
Converting an axial offset into an azimuth with phi = twist * dz / rise would
make the placement depend on the twist being tested, and the constraint would
then hold identically for every candidate -- self-fulfilling, and useless for
determining twist. Keep the placement axial.

Four ways two images can be related
-----------------------------------
Besides in-plane rotation and the two shifts, a pair can differ by a flip:

* mirrored in x -- the filament was picked with the opposite polarity. This is
  a 180 degree rotation of the structure about an in-plane axis perpendicular
  to the filament, so it is a proper rotation and a physically real relation.
* mirrored in y -- the structure viewed from the other side, a 180 degree
  rotation about the filament axis itself. Also proper.
* both -- a 180 degree rotation about the viewing axis, i.e. in-plane.

None of these changes the hand of the structure; they are alternative
orientations of the same object, and a pair registered under the wrong one
will simply correlate badly. All four are tried, and the choices are then made
globally consistent (a sign-synchronisation over the graph) rather than pair by
pair, because independent pairwise choices can easily disagree around a loop.

A genuine handedness flip is a mirror, not a rotation, and cannot be resolved
from projections at all -- every image in one dataset shares the specimen's
hand, so it is not something registration can or should decide.

On mutual versus absolute alignment
-----------------------------------
Aligning images to a common reference makes them mutually consistent while
letting them drift together away from horizontal; that reference bias is why
psi and dy are not refined that way (see ``denovo3d_joint``). Here mutual
consistency is exactly what is wanted -- the composite only needs its pieces to
agree with each other -- and absolute orientation still comes from the tab's
auto-transform, which runs before any of this.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)

# A pair registered across less than this fraction of the image width is not
# trusted: the correlation is computed over the overlap, so a sliver of overlap
# can score highly on noise alone.
MIN_OVERLAP = 0.35

# Pairs below this normalised correlation are dropped from the graph.
MIN_CORR = 0.5

# ...and so are pairs whose correlation profile has no distinct peak, however
# high that peak scores. A pair with no real overlap still yields a best
# offset, and on synthetic data its correlation (0.95-0.98) is not obviously
# worse than a genuine pair's (0.997) -- but its profile is flat, and
# prominence separated the two cleanly (1.94-2.20 against 2.42-2.59).
#
# This is only a first filter, deliberately loose. Some pairs simply cannot be
# registered -- at low twist the images may share little or no overlap -- so
# the global solve is robust to the ones that slip through, and connectivity is
# reported rather than assumed.
#
# Otsu on the prominence values was tried for choosing this cutoff per dataset
# and rejected: it always returns a split, so on a well-overlapping set where
# every pair is good it discarded half of them, and its own separability score
# stayed high (0.63) there, so that cannot guard it either. A loose floor plus
# robust consensus classified all three synthetic regimes exactly, including
# one where 6 of 10 pairs were unregistrable.
MIN_PROMINENCE = 2.0

# (flip_x, flip_y) branches tried for every pair.
FLIPS = ((False, False), (True, False), (False, True), (True, True))


def _flip(img, fx=False, fy=False):
    out = img
    if fx:
        out = out[:, ::-1]
    if fy:
        out = out[::-1, :]
    return np.ascontiguousarray(out)


def _apply(img, psi=0.0, dy=0.0):
    import helicon

    out = np.asarray(img, dtype=np.float32)
    if psi or dy:
        out = helicon.transform_image(
            image=out, rotation=float(psi), post_translation=(float(dy), 0)
        )
    return out


# ──────────────────────────────────────────────────────────────────────────
# Pairwise registration
# ──────────────────────────────────────────────────────────────────────────


def _match_height(b, ny):
    """Centre ``b`` into a canvas ``ny`` tall (the composite may be taller)."""
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


def _dx_profile(a, b, min_overlap=MIN_OVERLAP):
    """Best axial offset of ``b`` relative to ``a``, and its correlation.

    Normalised over the overlapping columns only, so an offset that leaves the
    images barely touching is not rewarded for having little to disagree about.
    Non-circular: the images are zero-padded, unlike a ``np.roll`` search,
    which is periodic in the image width and so cannot express an offset larger
    than one image -- exactly the offsets stitching needs.

    ``a`` and ``b`` may differ in size; the refinement pass registers each
    image against the whole composite, which is several images wide.

    Returns (dx, corr); shifting ``b`` by ``dx`` aligns it with ``a``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = _match_height(np.asarray(b, dtype=np.float64), a.shape[0])
    na, nb = a.shape[1], b.shape[1]
    av = a - a.mean()
    bv = b - b.mean()

    n = na + nb
    num = np.fft.irfft(
        np.fft.rfft(av, n=n, axis=1) * np.conj(np.fft.rfft(bv, n=n, axis=1)),
        n=n,
        axis=1,
    ).sum(axis=0)

    # Column energies, so each offset is normalised by only the part of each
    # image that actually overlaps.
    ea = np.concatenate([[0.0], np.cumsum((av**2).sum(axis=0))])
    eb = np.concatenate([[0.0], np.cumsum((bv**2).sum(axis=0))])

    min_cols = max(4, int(round(min_overlap * min(na, nb))))
    k = np.arange(-(nb - min_cols), na - min_cols + 1)
    if k.size == 0:
        return 0, -2.0
    lo_a = np.maximum(0, k)
    hi_a = np.minimum(na, nb + k)
    lo_b = np.maximum(0, -k)
    hi_b = np.minimum(nb, na - k)
    cols = hi_a - lo_a
    sa = ea[hi_a] - ea[lo_a]
    sb = eb[hi_b] - eb[lo_b]
    denom = np.sqrt(np.maximum(sa * sb, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, num[k % n] / denom, -2.0)
    corr[cols < min_cols] = -2.0
    best = int(np.argmax(corr))
    valid = corr[corr > -2.0]
    # Peak prominence. A pair with little or no real overlap still produces a
    # best-scoring offset, and its correlation there can look respectable; what
    # gives it away is that the whole profile is equally good, i.e. no distinct
    # peak. Measured on synthetic data, this is what separates the 4 genuinely
    # overlapping pairs from the 10 that registration otherwise accepts.
    if valid.size > 8 and valid.std() > 1e-9:
        prom = float((corr[best] - valid.mean()) / valid.std())
    else:
        prom = 0.0
    return int(k[best]), float(corr[best]), prom


def register_pair(
    a,
    b,
    rot_range=6.0,
    dy_range=5.0,
    coarse_step=1.0,
    refine=True,
    min_overlap=MIN_OVERLAP,
    try_flips=True,
    flips=None,
):
    """Register ``b`` onto ``a`` over flip, in-plane rotation, dy and dx.

    All of them matter: two class averages of one filament generally differ in
    how the filament tilts in the plane and in how well each was centred, not
    only in where along the axis they sit, and either may have been picked with
    the opposite polarity. Searching dx alone leaves those differences to be
    absorbed as a worse correlation, which both degrades the offset and makes
    the pair look less trustworthy than it is.

    Returns ``{flip_x, flip_y, psi, dy, dx, corr}``: flip ``b``, then apply psi
    and dy, then offset by dx, to bring it into ``a``'s frame.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    candidates = (
        flips if flips is not None else (FLIPS if try_flips else [(False, False)])
    )

    def scan(src, psi0, dy0, half_psi, half_dy, step):
        best = (-2.0, psi0, dy0, 0, 0.0)
        for psi in np.arange(psi0 - half_psi, psi0 + half_psi + step / 2, step):
            for dy in np.arange(dy0 - half_dy, dy0 + half_dy + step / 2, step):
                dx, c, prom = _dx_profile(a, _apply(src, psi, dy), min_overlap)
                if c > best[0]:
                    best = (c, float(psi), float(dy), int(dx), prom)
        return best

    overall = None
    for fx, fy in candidates:
        src = _flip(b, fx, fy)
        best = scan(src, 0.0, 0.0, rot_range, dy_range, coarse_step)
        if overall is None or best[0] > overall[0][0]:
            overall = (best, fx, fy)

    best, fx, fy = overall
    if refine:
        src = _flip(b, fx, fy)
        for step in (0.25, 0.0625):
            best = scan(src, best[1], best[2], 2 * step, 2 * step, step)
    return dict(
        corr=best[0],
        psi=best[1],
        dy=best[2],
        dx=best[3],
        prominence=best[4],
        flip_x=fx,
        flip_y=fy,
    )


def n_registration_jobs(n_images, refine_against_composite=2):
    """How many pair registrations ``auto_stitch`` will run, for a progress bar.

    Two passes over every pair -- one to settle polarity, one to measure the
    rotation and shifts with polarity fixed -- plus one registration per image
    against the composite for each refinement round.
    """
    pairs = n_images * (n_images - 1) // 2
    return 2 * pairs + max(0, refine_against_composite) * n_images


def register_graph(
    images, min_corr=MIN_CORR, min_prominence=MIN_PROMINENCE, progress=None, **kw
):
    """Register every pair. Returns the accepted pairs.

    All pairs rather than a spanning chain: the redundancy is what makes the
    global solve robust to a few bad registrations, and it is what lets the
    closure error act as a check that the answer means anything.
    """
    n = len(images)
    pairs = []
    for i, j in itertools.combinations(range(n), 2):
        if progress is not None:
            progress(f"registering images {i + 1} and {j + 1}")
        m = register_pair(images[i], images[j], **kw)
        m.update(i=i, j=j)
        if m["corr"] >= min_corr and m["prominence"] >= min_prominence:
            pairs.append(m)
        else:
            logger.debug(
                "pair (%d,%d) rejected: corr %.3f (min %.2f), prominence %.1f (min %.1f)",
                i,
                j,
                m["corr"],
                min_corr,
                m["prominence"],
                min_prominence,
            )
    return pairs


# ──────────────────────────────────────────────────────────────────────────
# Global reconciliation
# ──────────────────────────────────────────────────────────────────────────


def _adjacency(n, pairs):
    adj = {i: [] for i in range(n)}
    for m in pairs:
        adj[m["i"]].append(m)
        adj[m["j"]].append(m)
    return adj


def _connected_to(n, pairs, gauge=0):
    adj = _adjacency(n, pairs)
    seen = {gauge}
    stack = [gauge]
    while stack:
        v = stack.pop()
        for m in adj[v]:
            u = m["j"] if m["i"] == v else m["i"]
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return seen


def sync_flips(n, pairs, gauge=0):
    """Make the pairwise flip choices globally consistent.

    Each pair says whether j is mirrored relative to i; that is a relative sign,
    so the absolute flips follow by propagating from the gauge. Propagation
    follows the most confident edges first (a maximum-correlation spanning
    tree), so a weak pair cannot flip a whole branch of the graph. Edges left
    over are checked against the result and the disagreements are returned --
    a high count means the flips are not reliably determined and the composite
    should not be trusted.

    Returns (flips, n_conflicts) with flips[i] = (flip_x, flip_y).
    """
    flips = [(False, False)] * n
    if not pairs:
        return flips, 0
    order = sorted(pairs, key=lambda m: -m["corr"])
    assigned = {gauge}
    changed = True
    while changed:
        changed = False
        for m in order:
            i, j = m["i"], m["j"]
            if i in assigned and j not in assigned:
                fx, fy = flips[i]
                flips[j] = (fx != m["flip_x"], fy != m["flip_y"])
                assigned.add(j)
                changed = True
            elif j in assigned and i not in assigned:
                fx, fy = flips[j]
                # The relation is its own inverse: mirroring is an involution.
                flips[i] = (fx != m["flip_x"], fy != m["flip_y"])
                assigned.add(i)
                changed = True

    conflicts = 0
    for m in pairs:
        i, j = m["i"], m["j"]
        if i not in assigned or j not in assigned:
            continue
        want_x = flips[i][0] != flips[j][0]
        want_y = flips[i][1] != flips[j][1]
        if want_x != m["flip_x"] or want_y != m["flip_y"]:
            conflicts += 1
    return flips, conflicts


def _solve_one(n, pairs, key, gauge=0, robust_iterations=4):
    """Weighted least squares for one parameter over the pair graph.

    Each pair contributes ``q_j - q_i = delta_ij`` weighted by its correlation,
    plus a gauge row pinning image ``gauge`` to zero -- standard
    graph-Laplacian averaging.

    Then it is re-solved a few times with pairs that disagree with the
    consensus downweighted (IRLS against a MAD-derived scale). Pairwise
    registration cannot be made reliable enough to skip this: a pair with
    little or no overlap still produces a confident-looking answer, and with no
    redundancy in the graph such a pair would simply be believed. Redundancy
    plus robust weighting is what lets a wrong pair be outvoted.

    Returns (values, weights) so callers can see which pairs were discounted.
    """
    if not pairs:
        return np.zeros(n), np.zeros(0)
    rows = np.zeros((len(pairs) + 1, n))
    rhs = np.zeros(len(pairs) + 1)
    base = np.zeros(len(pairs) + 1)
    for e, m in enumerate(pairs):
        rows[e, m["j"]] = 1.0
        rows[e, m["i"]] = -1.0
        rhs[e] = float(m[key])
        base[e] = float(max(m["corr"], 0.0))
    rows[-1, gauge] = 1.0
    rhs[-1] = 0.0
    base[-1] = 10.0 * max(base[:-1].max(), 1e-6)

    w = base.copy()
    q = np.zeros(n)
    for _ in range(max(1, robust_iterations)):
        A = rows * w[:, None]
        q, *_ = np.linalg.lstsq(A, rhs * w, rcond=None)
        resid = np.abs(rows @ q - rhs)
        mad = float(np.median(resid[:-1])) if len(pairs) else 0.0
        scale = max(1.4826 * mad, 1e-3)
        w = base / (1.0 + (resid / (3.0 * scale)) ** 2)
        w[-1] = base[-1]
    return q, w[:-1]


# Registration is a sub-pixel measurement, so a pair that disagrees with the
# consensus by more than this is not noisy -- it is wrong, and no amount of
# down-weighting should keep it. Absolute rather than derived from the spread:
# a MAD-based scale inflates when many pairs are bad, which is exactly when the
# rejection is needed. On the case that motivated this, half the accepted pairs
# were wrong and IRLS alone rejected none of them.
TOLERANCE = {"psi": 0.75, "dy": 2.0, "dx": 3.0}


def solve_global(
    n,
    pairs,
    image_width,
    gauge=0,
    tolerance=None,
    max_rejects=None,
    min_overlap=MIN_OVERLAP,
):
    """Reconcile pairwise measurements into one transform per image.

    psi, dy and dx are each averaged over the graph. Solving them independently
    linearises the fact that a rotation and a translation do not commute; with
    psi of a few degrees and dy of a few pixels the coupling is second order,
    and ``auto_stitch`` refines against the composite afterwards.

    Pairs that cannot be registered -- at low twist two images may share no
    overlap at all -- still produce a confident-looking answer, so after each
    solve the pair furthest outside ``tolerance`` is dropped and the rest
    re-solved. Dropping one at a time rather than all at once matters: the
    residuals are computed against a consensus that a bad pair is currently
    distorting, so the second-worst pair is often innocent.

    Images left unreachable get a zero transform and are listed in
    ``diagnostics['unconnected']`` -- they must be excluded from the composite
    rather than stacked at the origin.
    """
    if n == 0:
        return [], dict(unconnected=[], n_pairs=0)
    tol = dict(TOLERANCE if tolerance is None else tolerance)
    kept = list(pairs)
    rejected = []
    # Never strip the graph below a spanning tree; past that point the answer
    # is not improving, it is only losing images.
    budget = len(kept) if max_rejects is None else max_rejects

    for _ in range(budget):
        if len(kept) <= max(0, n - 1):
            break
        solved = {k: _solve_one(n, kept, k, gauge)[0] for k in tol}
        worst_at, worst_score = None, 0.0
        for e, m in enumerate(kept):
            score = max(
                abs((solved[k][m["j"]] - solved[k][m["i"]]) - float(m[k])) / tol[k]
                for k in tol
            )
            # Geometry, not just consistency: if the consensus puts these two
            # images further apart than they could overlap, the pair was never
            # registrable and its measurement is meaningless however
            # self-consistent it looks. This needs no new threshold -- it is the
            # same min_overlap the correlation search already respects.
            gap = abs(solved["dx"][m["j"]] - solved["dx"][m["i"]])
            if image_width and gap > image_width * (1.0 - min_overlap):
                score = max(score, 1.0 + gap / max(image_width, 1))
            if score > worst_score:
                worst_at, worst_score = e, score
        if worst_at is None or worst_score <= 1.0:
            break
        rejected.append(kept.pop(worst_at))

    connected = _connected_to(n, kept, gauge)
    solved, weights = {}, {}
    for k in ("psi", "dy", "dx"):
        solved[k], weights[k] = _solve_one(n, kept, k, gauge)
    for m, wt in zip(kept, weights["dx"]):
        m["weight"] = float(wt)

    transforms = []
    for i in range(n):
        ok = i in connected
        transforms.append(
            dict(
                psi=float(solved["psi"][i]) if ok else 0.0,
                dy=float(solved["dy"][i]) if ok else 0.0,
                dx=float(solved["dx"][i]) if ok else 0.0,
                connected=ok,
            )
        )

    # Closure: how well the surviving transforms reproduce their measurements.
    # This is the honest check on the whole scheme -- and note it only means
    # something where the graph has loops. A bare chain fits exactly by
    # construction, so a clean closure over a tree is not evidence of anything.
    resid = {k: [] for k in ("psi", "dy", "dx")}
    for m in kept:
        for k in resid:
            resid[k].append(
                abs((transforms[m["j"]][k] - transforms[m["i"]][k]) - float(m[k]))
            )

    dxs = [t["dx"] for t in transforms if t["connected"]]
    span = (max(dxs) - min(dxs) + image_width) if dxs else image_width
    diagnostics = dict(
        n_pairs=len(kept),
        n_rejected=len(rejected),
        redundancy=len(kept) - max(0, len(connected) - 1),
        n_connected=len(connected),
        unconnected=sorted(set(range(n)) - connected),
        span_px=float(span),
        span_gain=float(span / image_width) if image_width else 1.0,
        closure={k: (float(np.median(v)) if v else 0.0) for k, v in resid.items()},
        mean_corr=float(np.mean([m["corr"] for m in kept])) if kept else 0.0,
        n_downweighted=int(
            sum(1 for m in kept if m.get("weight", 1.0) < 0.25 * max(m["corr"], 1e-6))
        ),
    )
    # Closure over a graph with no loops fits exactly by construction, so a
    # clean number there is not evidence of anything. Say so rather than
    # letting it read as a pass.
    diagnostics["closure_meaningful"] = diagnostics["redundancy"] > 0
    diagnostics["trustworthy"] = bool(
        diagnostics["redundancy"] > 0
        and not diagnostics["unconnected"]
        and all(diagnostics["closure"][k] <= tol[k] for k in tol)
    )
    return transforms, diagnostics


# ──────────────────────────────────────────────────────────────────────────
# Compositing
# ──────────────────────────────────────────────────────────────────────────


def _core_stats(image, margin_frac=0.25):
    """Mean and std of an image's central columns, excluding its outer ends.

    Class averages taper towards their axial ends -- fewer particles
    contribute coherently there, so the ends are genuinely weaker than the
    middle in every image, not just the mismatched ones. Measuring brightness
    from the whole image lets that expected taper dilute the estimate, and
    measuring it from just the overlap (which *is* the tapered end) makes the
    estimate noisy and confuses real weakness with a contrast mismatch. The
    central region is the one part of the image that reliably represents its
    true brightness/contrast level.
    """
    h, w = image.shape
    margin = int(round(w * margin_frac))
    core = image[:, margin : w - margin] if w - 2 * margin >= max(8, w // 4) else image
    return float(core.mean()), float(core.std())


def _normalize_intensity(image, flatten_sigma_frac=1 / 6, core_margin_frac=0.25):
    """Put an image on a common brightness/contrast footing before blending.

    Two passes, for two different problems. First, a wide Gaussian low-pass
    along the filament axis is subtracted off (a light high-pass): this
    removes the smooth axial brightness gradient towards each image's ends --
    weaker signal there comes from the class average itself, not from a
    per-image contrast difference, so it should not be asked to average
    cleanly against a neighbour's flat interior. The kernel is wide enough
    (a sizeable fraction of the image width) that it tracks only that broad
    trend and leaves ordinary structural spatial frequencies alone. Second,
    the flattened image is rescaled to zero mean / unit std using its central
    columns as the reference (see ``_core_stats``), since even after
    flattening the ends carry less signal and would bias a whole-image
    estimate.
    """
    from scipy.ndimage import gaussian_filter1d

    h, w = image.shape
    sigma = max(4.0, w * flatten_sigma_frac)
    baseline = gaussian_filter1d(image, sigma=sigma, axis=1, mode="nearest")
    flattened = image - baseline

    mean, std = _core_stats(flattened, core_margin_frac)
    if std > 1e-6:
        flattened = (flattened - mean) / std
    return flattened


def composite(images, transforms, feather=8, match_intensity=True):
    """Average the registered images onto one canvas.

    Averaging the overlaps rather than picking one image or blending with a
    seam: an overlap is genuinely the same object seen twice, so averaging is
    what raises the signal there. ``feather`` tapers each image's weight
    towards its ends so an image edge does not print a step into the composite.

    ``match_intensity`` puts each image on a common brightness/contrast
    footing before blending (see ``_normalize_intensity``), so overlaps
    between images of differing contrast -- e.g. class averages built from
    different numbers of particles, or with an axial brightness gradient --
    average together instead of one side dominating.

    Returns (image, coverage) where coverage counts contributions per column.
    """
    used = [(im, t) for im, t in zip(images, transforms) if t.get("connected", True)]
    if not used:
        return None, None
    ny = max(np.shape(im)[0] for im, _ in used)
    nx = max(np.shape(im)[1] for im, _ in used)
    x0 = int(np.floor(min(t["dx"] for _, t in used)))
    total = int(np.ceil(max(t["dx"] for _, t in used) - x0)) + nx

    acc = np.zeros((ny, total), dtype=np.float64)
    wsum = np.zeros((ny, total), dtype=np.float64)
    for im, t in used:
        work = _apply(
            _flip(im, t.get("flip_x", False), t.get("flip_y", False)),
            t["psi"],
            t["dy"],
        ).astype(np.float64)
        if match_intensity:
            work = _normalize_intensity(work)
        h, w = work.shape
        ramp = np.ones(w)
        if feather and w > 2 * feather:
            ramp[:feather] = np.linspace(0, 1, feather, endpoint=False)
            ramp[-feather:] = np.linspace(1, 0, feather, endpoint=False)
        weight = np.broadcast_to(ramp, (h, w)).copy()

        top = (ny - h) // 2
        left = int(round(t["dx"] - x0))
        acc[top : top + h, left : left + w] += work * weight
        wsum[top : top + h, left : left + w] += weight

    out = np.zeros(acc.shape, dtype=np.float32)
    hit = wsum > 1e-9
    out[hit] = (acc[hit] / wsum[hit]).astype(np.float32)
    return out, (wsum > 1e-9).sum(axis=0)


def auto_stitch(
    images, refine_against_composite=2, image_width=None, progress=None, **kw
):
    """Register the images, reconcile, and composite them into one image.

    Flips are settled first and applied, so the rotation and shift solve never
    has to carry a sign convention: after this stage every image is in the same
    polarity and the remaining relation is a small rotation and a shift.

    ``refine_against_composite`` re-registers each image against the current
    composite a couple of times, which mops up the error from solving psi, dy
    and dx independently. Safe here for the reason in the module docstring: the
    composite only needs internal consistency.

    Returns (stitched, transforms, diagnostics).
    """
    images = [np.asarray(im, dtype=np.float32) for im in images]
    if not images:
        return None, [], dict(n_pairs=0)
    nx = image_width or images[0].shape[1]
    if len(images) == 1:
        return (
            images[0],
            [dict(psi=0.0, dy=0.0, dx=0.0, flip_x=False, flip_y=False, connected=True)],
            dict(
                n_pairs=0,
                n_connected=1,
                unconnected=[],
                span_px=float(nx),
                span_gain=1.0,
                closure={"psi": 0.0, "dy": 0.0, "dx": 0.0},
                mean_corr=1.0,
                flip_conflicts=0,
            ),
        )

    # Stage 1: which images are mirrored relative to each other.
    pairs = register_graph(images, progress=progress, **kw)
    flips, conflicts = sync_flips(len(images), pairs)
    oriented = [_flip(im, fx, fy) for im, (fx, fy) in zip(images, flips)]

    # Stage 2: with polarity settled, re-register without the flip search and
    # reconcile the rotations and shifts.
    kw2 = dict(kw)
    kw2["try_flips"] = False
    pairs = register_graph(oriented, progress=progress, **kw2)
    transforms, diagnostics = solve_global(len(images), pairs, nx)
    for t, (fx, fy) in zip(transforms, flips):
        t["flip_x"], t["flip_y"] = fx, fy
    diagnostics["flip_conflicts"] = conflicts
    diagnostics["n_flipped"] = sum(1 for fx, fy in flips if fx or fy)

    # Registered against a raw (un-normalised) composite throughout: matching
    # each image's brightness/contrast changes its low-frequency content, and
    # register_pair's correlation relies on that content lining up with the
    # unmodified image being tested against it. Intensity matching is a
    # cosmetic step for the composite a caller displays, not for this loop.
    stitched, coverage = composite(images, transforms, match_intensity=False)
    for _ in range(max(0, refine_against_composite)):
        if stitched is None:
            break
        updated = []
        for k, (im, t) in enumerate(zip(images, transforms)):
            if not t["connected"]:
                updated.append(t)
                continue
            if progress is not None:
                progress(f"refining image {k + 1} against the composite")
            m = register_pair(stitched, _flip(im, t["flip_x"], t["flip_y"]), **kw2)
            updated.append(
                dict(
                    psi=m["psi"],
                    dy=m["dy"],
                    dx=float(m["dx"]),
                    flip_x=t["flip_x"],
                    flip_y=t["flip_y"],
                    connected=True,
                    corr=m["corr"],
                )
            )
        transforms = updated
        stitched, coverage = composite(images, transforms, match_intensity=False)

    if coverage is not None:
        diagnostics["max_coverage"] = int(coverage.max())
        diagnostics["mean_coverage"] = float(coverage.mean())
    dxs = [t["dx"] for t in transforms if t.get("connected")]
    if dxs:
        diagnostics["span_px"] = float(max(dxs) - min(dxs) + nx)
        diagnostics["span_gain"] = float(diagnostics["span_px"] / nx)

    # Transforms are now settled; rebuild the returned image with intensity
    # matching for display.
    stitched, _coverage = composite(images, transforms)
    return stitched, transforms, diagnostics


# ──────────────────────────────────────────────────────────────────────────
# Composing with the tab's auto-transform
# ──────────────────────────────────────────────────────────────────────────


def normalize_flip(flip_x, flip_y):
    """Reduce the four flip branches to one mirror plus a 180 degree rotation.

    An x-flip followed by a y-flip is exactly a 180 degree in-plane rotation,
    so every branch is {no mirror, x-mirror} x {0, 180}. Only one mirror is
    ever needed, and the other degree of freedom folds into the rotation --
    which is what makes it possible to collapse the whole chain into a single
    interpolation below.
    """
    if flip_x and flip_y:
        return False, 180.0
    if flip_y:
        return True, 180.0
    if flip_x:
        return True, 0.0
    return False, 0.0


def compose_transforms(auto_rotation, auto_shift_y, reg):
    """Fold the auto-transform and the registration into one transform.

    The tab's auto-transform already rotated and centred the image, and
    registration then measured a small residual on top of it. Applying both in
    turn interpolates the image twice, which needlessly blurs it; composing
    them resamples the original once.

    The mirror and the 180 degree rotation stay as array operations rather than
    being folded into the rotation angle. They are exact that way, and an array
    reversal is not the same as ``transform_image(rotation=180)`` on an
    even-sized image -- they differ by half a pixel, which measured as a drop
    from 0.9999 to 0.95 correlation when the 180 was folded in.

    Returns ``{mirror_x, rot180, rotation, shift_y, extra_dx}`` to apply to the
    *original*: mirror, then rot180, then rotate, then translate. ``extra_dx``
    is the axial component the composition generates -- the auto-transform's
    vertical shift acquires one once a rotation is applied after it -- and
    belongs with the placement offset, not the transform.
    """
    mirror, extra = normalize_flip(reg.get("flip_x", False), reg.get("flip_y", False))
    rot180 = bool(extra)
    psi = float(reg.get("psi", 0.0))
    dy = float(reg.get("dy", 0.0))
    # A mirror reverses the sense of any rotation applied before it; a 180
    # degree rotation reverses the sign of any translation applied before it.
    r_eff = -float(auto_rotation) if mirror else float(auto_rotation)
    s_eff = -float(auto_shift_y) if rot180 else float(auto_shift_y)
    phi = np.radians(psi)
    return dict(
        mirror_x=mirror,
        rot180=rot180,
        rotation=float(psi + r_eff),
        shift_y=float(dy + s_eff * np.cos(phi)),
        extra_dx=float(-s_eff * np.sin(phi)),
    )


def apply_composed(img, t):
    """Apply a ``compose_transforms`` result to an original image, once.

    The mirror and 180 degree rotation are exact array operations; only the
    residual rotation and shift go through an interpolation.
    """
    import helicon

    out = np.asarray(img, dtype=np.float32)
    if t.get("mirror_x"):
        out = out[:, ::-1]
    if t.get("rot180"):
        out = np.rot90(out, 2)
    out = np.ascontiguousarray(out)
    if t.get("rotation") or t.get("shift_y"):
        out = helicon.transform_image(
            image=out,
            rotation=float(t.get("rotation", 0.0)),
            post_translation=(float(t.get("shift_y", 0.0)), 0),
        )
    return out
