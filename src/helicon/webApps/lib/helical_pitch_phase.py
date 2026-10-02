"""Helical pitch from the phases of every class pair along the filaments.

The pair-distance histogram of the HelicalPitch tab uses only pairs of
segments assigned to the *same* class: their separations cluster at multiples
of the distance over which the projection repeats. This module, behind the
AbInitio3D tab, uses every pair among the classes it is given.
A class average is a view of the filament at some azimuth, so each class k has a
phase theta_k on a circle whose circumference P is that repeat distance, and a
pair of segments of classes a and b, a distance d apart along one filament,
satisfies ``d = theta_b - theta_a (mod P)``. For each candidate P the matrix

    H_ab(P) = sum over pairs of w * exp(2 pi i d / P)

is, at the right P, close to rank one, and its top eigenvector gives every
class's phase at once -- angular synchronisation. The top eigenvalue scores P.

What is measured is the repeat of the *projection*, which is the pitch divided
by the number of copies the projection cannot tell apart: the cyclic symmetry,
times two for a two-fold perpendicular to the axis or a 2-1 screw. It is the
same quantity the pair-distance histogram measures, and the rise is not needed
to measure it. Converting it to a twist needs the rise and that factor.

Measured on EMPIAR-10940 (98,287 segments, 3,700 filaments, 42 classes), the
period comes out at 700 A. The second and third peaks sit at P/2 and P/3,
as they must, since phases consistent at P are also consistent at its
submultiples. On random subsets of filaments it is steadier than the
first-peak estimate: at 37 filaments the spread of ten draws was 38 A against
273 A, at 185 it was right in all ten against eight.

Three corrections, each measured rather than assumed:

* Positions are path lengths along the smoothed *refined* segment centres (box
  coordinate minus alignment origin), not the track length, which follows the
  straight line the filament was picked along. On EMPIAR-10940 the difference
  is 0.07-0.35% -- those filaments are straight at the scale of a period --
  but a curved filament picked with a straight line is under-measured by its
  arc/chord ratio, which biases the period low.
* The peak is biased low when filaments are short compared with the period,
  as most of EMPIAR-10940's are (median 271 A against 700 A): with perfect
  synthetic labels on its real segment positions the bias was -0.5% to -2.3%
  depending on the distance weighting. :func:`calibrate_period` removes it by
  generating labels at known periods on the dataset's own segment geometry.
* Segments with a large shift perpendicular to the axis are down-weighted;
  dropping the worst 20% on EMPIAR-10940 raised the peak score 0.698 -> 0.741.

One ring means one helical type. Classes of two types sit at different
azimuths of different rings, so on one ring their constraints conflict and the
period comes out as a blend -- EMPIAR-10230's two tau polymorphs, sharing most
of cryoSPARC's classes, gave 765 A pooled against 762 and 840 A apart. Typing
filaments blindly from their class usage (as CHEP does) was tried and found not
reliable enough to depend on: it also splits a single type by picking
direction or by other per-filament differences. So the classes are the user's
choice -- those of one type, junk left out -- and :func:`class_fit`,
:func:`largest_phase_gap`, the ``double_ratio`` of :func:`estimate_period` and
:func:`suggest_counterparts` exist to help make and check that choice.

Direction is handled per segment and then per filament. :func:`class_polarity`
reads each segment's direction from its in-plane angle relative to its class's
axis; :func:`sync_filament_directions` then chooses each whole filament's
direction to agree with the ring, which repairs data whose angles say little
about direction (half of EMPIAR-10230's filaments were flipped, raising the
score from 0.36 to 0.55). Pairs are always ordered: discarding which class
comes first would make the method blind to direction only by destroying it.

The pooled period is an average: filaments differ. :func:`filament_heterogeneity`
measures by how much, per filament, from a split-half test, and on EMPIAR-10940
finds a real spread of about 6-7% of the period between filaments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Below this fraction of pair weight beyond the estimated period, the estimate
# cannot be told from half the true period (see estimate_period).
MIN_SUPPORT = 0.01

# At or above this ratio of the score at twice the period to the score at the
# period, the estimate may be half the true repeat (see estimate_period).
DOUBLE_RATIO_WARN = 0.8

# RELION columns that a refined path needs; without them the track length is
# used. The origins are optional: without them the path runs through the box
# centres, which is what a cryoSPARC file gives (see helical_pitch_compute).
_REFINED_COLUMNS = ("rlnCoordinateX", "rlnCoordinateY")


@dataclass
class SegmentPairs:
    """Segments and the same-polarity pairs between them, ready to score.

    Attributes
    ----------
    class_ids : np.ndarray
        The class numbers, one per row of every class-indexed array.
    seg_class : np.ndarray
        Row in ``class_ids`` of each segment.
    seg_filament : np.ndarray
        Filament index of each segment.
    seg_pos : np.ndarray
        Position of each segment along its filament, in A.
    seg_pol : np.ndarray
        Psi polarity of each segment, +1 or -1.
    I, J : np.ndarray
        Segment indices of each pair, ``I < J`` within one filament.
    D : np.ndarray
        Signed offset from segment I to segment J along the filament, in A.
    W : np.ndarray
        Pair weight, the product of the two segments' weights.
    filament_span : np.ndarray
        Length covered by each filament's segments, in A.
    refined_path : bool
        Whether positions came from the refined path or the track length.
    max_sep : float
        Largest separation kept, in A.
    filament_keys : list of tuple
        ``(rlnMicrographName, rlnHelicalTubeID)`` of each filament.
    seg_row : np.ndarray
        Row of each segment in the parameter table it was collected from.
    seg_small : np.ndarray
        Each segment's in-plane deviation from its filament direction, in
        degrees within (-90, 90]: psi minus the psi prior, minus its class's
        axis (which absorbs the 0/180 flips and any rotation of the class).
    seg_ref : np.ndarray
        The psi prior of each segment (the filament direction), in degrees.
    class_tilt, class_across, class_count : np.ndarray
        Per class, the median in-plane deviation of its segments' psi from their
        priors (degrees, after removing the class axis) and the median offset of
        their refined centres across the smoothed filament path (A), which
        measure how far the class average is from horizontal and centred, and
        the number of segments. Picking errors are random, so the medians are
        robust.
    class_axis, class_sign : np.ndarray
        Per class, the axis angle of its segments' in-plane deviation (degrees,
        modulo 180) and the sign that makes the polarity agree within filaments.
    seg_weight : np.ndarray
        Weight of each segment: its alignment confidence, down-weighted when
        it lies far off its filament's path.
    """

    class_ids: np.ndarray
    seg_class: np.ndarray
    seg_filament: np.ndarray
    seg_pos: np.ndarray
    seg_pol: np.ndarray
    I: np.ndarray
    J: np.ndarray
    D: np.ndarray
    W: np.ndarray
    filament_span: np.ndarray
    refined_path: bool
    max_sep: float = 1500.0
    filament_keys: list = field(default_factory=list)
    seg_row: np.ndarray = None
    seg_small: np.ndarray = None
    seg_ref: np.ndarray = None
    seg_weight: np.ndarray = None
    class_axis: np.ndarray = None
    class_sign: np.ndarray = None
    class_tilt: np.ndarray = None
    class_across: np.ndarray = None
    class_count: np.ndarray = None

    @property
    def n_classes(self) -> int:
        return len(self.class_ids)

    @property
    def n_filaments(self) -> int:
        return len(self.filament_span)

    @property
    def pair_filament(self) -> np.ndarray:
        return self.seg_filament[self.I]


def _micrograph_apix(params):
    optics = params.attrs.get("optics") if hasattr(params, "attrs") else None
    for col in ("rlnMicrographOriginalPixelSize", "rlnMicrographPixelSize"):
        if col in params:
            return float(params[col].astype(float).iloc[0])
        if optics is not None and col in optics:
            return float(optics[col].astype(float).iloc[0])
    return None


def _class_medians(values, seg_class, n_classes):
    """Median of ``values`` within each class; NaN for a class without segments."""
    out = np.full(n_classes, np.nan)
    order = np.argsort(seg_class, kind="stable")
    sorted_class = seg_class[order]
    bounds = np.flatnonzero(np.diff(sorted_class)) + 1
    for group in np.split(order, bounds):
        if len(group):
            out[seg_class[group[0]]] = np.median(values[group])
    return out


def _class_axis(delta, seg_class, n_classes):
    """Axis angle (radians, modulo 180 degrees) of each class from its ``delta``."""
    delta = np.deg2rad(np.asarray(delta, dtype=float))
    z = np.bincount(seg_class, weights=np.cos(2 * delta), minlength=n_classes) + 1j * (
        np.bincount(seg_class, weights=np.sin(2 * delta), minlength=n_classes)
    )
    return 0.5 * np.angle(z)


def class_polarity(delta, seg_class, seg_filament, n_classes, details=False):
    """Which way along its class average each segment's filament runs: +1 or -1.

    ``delta`` is each segment's in-plane angle relative to its filament's
    direction (psi minus the psi prior), in degrees. In RELION with psi held to
    its prior it is 0 or 180, and its sign is the answer. cryoSPARC does not
    keep class averages horizontal: on EMPIAR-10230 only a third of segments
    were within 30 degrees of their prior or its opposite, because every class
    carries a rotation of its own. So each class's axis angle (modulo 180) is
    estimated first, as the doubled-angle mean of its segments' ``delta``, and
    polarity is measured from it.

    That leaves each class's choice of which end of its axis is "+" arbitrary.
    All segments of one filament run the same way, so the classes are made to
    agree -- a sign per class chosen so polarity is consistent within
    filaments, the top eigenvector of the class co-polarity matrix. On data
    whose classes already agree this changes nothing but, at most, the sign of
    every polarity at once.
    """
    axis = _class_axis(delta, seg_class, n_classes)
    delta = np.deg2rad(np.asarray(delta, dtype=float))
    pol = np.where(np.cos(delta - axis[seg_class]) >= 0, 1.0, -1.0)
    # per filament and class, the summed polarity; M = sum over filaments s s^T
    n_fil = int(seg_filament.max()) + 1 if len(seg_filament) else 0
    s = np.zeros((n_fil, n_classes))
    np.add.at(s, (seg_filament, seg_class), pol)
    M = s.T @ s
    np.fill_diagonal(M, 0.0)
    sign_out = np.ones(n_classes)
    if np.any(M):
        vals, vecs = np.linalg.eigh(M)
        sign = np.sign(vecs[:, -1])
        sign[sign == 0] = 1.0
        if np.sum(sign * np.abs(vecs[:, -1])) < 0:  # keep "no flip" the default
            sign = -sign
        pol = pol * sign[seg_class]
        sign_out = sign
    if details:
        return pol, np.rad2deg(axis), sign_out
    return pol


def refined_path_geometry(track, x, y):
    """The smoothed path through the refined segment centres, and each segment on it.

    Returns
    -------
    dict
        ``position`` (arc length along the path plus the segment's own offset
        along it, A), ``across`` (offset from the path across it, A), ``along``
        (that offset along it), ``x`` and ``y`` (the path point of each
        segment) and ``tx``, ``ty`` (the unit tangent there). For a filament
        too short to fit a path the track length is the position and the path
        is the centres themselves.
    """
    track = np.asarray(track, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(track)
    span = track[-1] - track[0] if n else 0.0
    if n < 4 or span <= 0:
        return dict(
            position=track.copy(),
            across=np.zeros(n),
            along=np.zeros(n),
            x=x.copy(),
            y=y.copy(),
            tx=np.ones(n),
            ty=np.zeros(n),
        )
    degree = 1 if span < 500 else (2 if span < 2000 else 3)
    degree = min(degree, n - 2)
    t0 = track[0]
    t = (track - t0) / span  # well-conditioned fit
    px = np.polyfit(t, x, degree)
    py = np.polyfit(t, y, degree)
    grid = np.linspace(0.0, 1.0, max(50, int(span / 5.0)))
    gx, gy = np.polyval(px, grid), np.polyval(py, grid)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(gx), np.diff(gy)))])
    s = np.interp(t, grid, arc)
    dx, dy = np.polyval(np.polyder(px), t), np.polyval(np.polyder(py), t)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    tx, ty = dx / norm, dy / norm
    on_x, on_y = np.polyval(px, t), np.polyval(py, t)
    rx, ry = x - on_x, y - on_y
    along = rx * tx + ry * ty
    across = -rx * ty + ry * tx
    return dict(
        position=s + along,
        across=across,
        along=along,
        x=on_x,
        y=on_y,
        tx=tx,
        ty=ty,
    )


def refined_path_positions(track, x, y):
    """Positions along the smoothed path through the refined segment centres.

    Parameters
    ----------
    track : np.ndarray
        Track length of each segment, in A, sorted ascending.
    x, y : np.ndarray
        Refined centre of each segment in the micrograph, in A.

    Returns
    -------
    positions : np.ndarray
        Arc length along a low-order polynomial fit of the centres, evaluated
        at each segment, plus the segment's own offset along the path.
    perpendicular : np.ndarray
        Each segment's offset from the fitted path, across it, in A.

    Notes
    -----
    The fit is smooth on purpose: segment centres scatter by about the size of
    the alignment shifts (sd 13 A on EMPIAR-10940), and the length of a path
    drawn through them point to point would be dominated by that scatter.
    """
    g = refined_path_geometry(track, x, y)
    return g["position"], g["across"]


def _filament_slices(df):
    """The rows of each filament, in the order ``groupby(sort=False)`` gives.

    Returns
    -------
    order : np.ndarray
        Row positions sorted by filament (first appearance) and, within one,
        by track length.
    starts, ends : np.ndarray
        Each filament's slice of ``order``.
    keys : list of tuple
        ``(rlnMicrographName, rlnHelicalTubeID)`` of each filament.
    """
    key = ["rlnMicrographName", "rlnHelicalTubeID"]
    fil = df.groupby(key, sort=False).ngroup().to_numpy()
    track = df["rlnHelicalTrackLengthAngst"].to_numpy().astype(float)
    order = np.lexsort((track, fil))
    fil = fil[order]
    starts = np.flatnonzero(np.r_[True, np.diff(fil) != 0])
    ends = np.r_[starts[1:], len(fil)]
    first = order[starts]
    keys = list(
        zip(
            df["rlnMicrographName"].to_numpy()[first].tolist(),
            df["rlnHelicalTubeID"].to_numpy()[first].tolist(),
        )
    )
    return order, starts, ends, keys


def prepare_pairs(
    params,
    class_ids=None,
    use_refined_path=True,
    max_sep=1500.0,
    weight_perpendicular=True,
    weight_confidence=True,
    max_classes=200,
):
    """Collect same-polarity segment pairs within every filament.

    Parameters
    ----------
    params : pandas.DataFrame
        Class2D parameters as ``helical_pitch_compute`` loads them: at least
        ``rlnMicrographName``, ``rlnHelicalTubeID``,
        ``rlnHelicalTrackLengthAngst``, ``rlnClassNumber`` and ``rlnAnglePsi``.
    class_ids : sequence of int, optional
        Class numbers to use. Defaults to the ``max_classes`` most populated.
    use_refined_path : bool, optional
        Measure positions along the refined path when the coordinate and
        origin columns are present. Defaults to True.
    max_sep : float, optional
        Largest separation kept, in A. Defaults to 1500.
    weight_perpendicular, weight_confidence : bool, optional
        Down-weight segments far off the fitted path, and segments with a low
        ``rlnMaxValueProbDistribution``. Both default to True.
    max_classes : int, optional
        When ``class_ids`` is not given, use only this many of the most
        populated classes; segments of the rest are left out. The cost grows as
        the square of the class count -- the class-pair histogram alone would be
        about 5 GB for the 1000 classes of a typical cryoSPARC run -- while the
        phase circle is sampled finely enough by far fewer. Defaults to 200.

    Returns
    -------
    SegmentPairs
    """
    df = params
    classes = df["rlnClassNumber"].astype(int).values
    if class_ids is None:
        uniq, counts = np.unique(classes, return_counts=True)
        class_ids = uniq[np.argsort(counts)[::-1][:max_classes]]
    class_ids = np.asarray(sorted(int(c) for c in class_ids))
    keep = np.isin(classes, class_ids)
    df = df.loc[keep]
    if len(df) == 0:
        raise ValueError("no segments of the selected classes")
    df = df.assign(_row=np.flatnonzero(keep))
    row_of = {c: i for i, c in enumerate(class_ids)}

    apix = _micrograph_apix(params)
    refined = (
        use_refined_path and apix is not None and all(c in df for c in _REFINED_COLUMNS)
    )
    has_prior = "rlnAnglePsiPrior" in df
    has_conf = weight_confidence and "rlnMaxValueProbDistribution" in df

    seg_class, seg_fil, seg_pos, seg_pol, seg_perp, seg_conf = [], [], [], [], [], []
    seg_row, seg_ref, spans = [], [], []
    # The columns are taken out once and the filaments walked as slices of
    # them: selecting columns of each filament's sub-frame cost 18 of 25 s on
    # EMPIAR-10940 (10,800 filaments), most of it pandas copying the frame's
    # attrs on every access.
    order, starts, ends, keys = _filament_slices(df)
    col = lambda name: df[name].to_numpy()[order]  # noqa: E731
    track_all = col("rlnHelicalTrackLengthAngst").astype(float)
    psi_all = col("rlnAnglePsi").astype(float)
    prior_all = col("rlnAnglePsiPrior").astype(float) if has_prior else None
    class_all = np.array([row_of[int(c)] for c in col("rlnClassNumber")], np.int32)
    row_all = col("_row")
    conf_all = (
        col("rlnMaxValueProbDistribution").astype(float)
        if has_conf
        else np.ones(len(order))
    )
    if refined:
        ox = col("rlnOriginXAngst").astype(float) if "rlnOriginXAngst" in df else 0.0
        oy = col("rlnOriginYAngst").astype(float) if "rlnOriginYAngst" in df else 0.0
        x_all = col("rlnCoordinateX").astype(float) * apix - ox
        y_all = col("rlnCoordinateY").astype(float) * apix - oy
    for f, (a, b) in enumerate(zip(starts, ends)):
        track = track_all[a:b]
        psi = psi_all[a:b]
        if has_prior:
            ref = prior_all[a:b]
        else:
            r = np.deg2rad(psi)
            ref = np.full(
                len(psi), np.rad2deg(np.arctan2(np.sin(r).sum(), np.cos(r).sum()))
            )
        delta = (psi - ref + 180.0) % 360.0 - 180.0
        seg_ref.append(ref)
        if refined:
            pos, perp = refined_path_positions(track, x_all[a:b], y_all[a:b])
        else:
            pos, perp = track, np.zeros(len(track))
        seg_class.append(class_all[a:b])
        seg_fil.append(np.full(b - a, f))
        seg_row.append(row_all[a:b])
        seg_pos.append(pos)
        seg_pol.append(delta)
        seg_perp.append(perp)
        seg_conf.append(conf_all[a:b])
        spans.append(pos.max() - pos.min() if len(pos) else 0.0)

    seg_class = np.concatenate(seg_class).astype(np.int32)
    seg_fil = np.concatenate(seg_fil).astype(np.int32)
    seg_pos = np.concatenate(seg_pos)
    delta_all = np.concatenate(seg_pol)
    axis = np.rad2deg(_class_axis(delta_all, seg_class, len(class_ids)))
    seg_small = (delta_all - axis[seg_class] + 90.0) % 180.0 - 90.0
    seg_pol, class_axis, class_sign = class_polarity(
        delta_all, seg_class, seg_fil, len(class_ids), details=True
    )
    perp = np.concatenate(seg_perp)
    w = np.concatenate(seg_conf)
    if refined and weight_perpendicular:
        scale = 1.4826 * np.median(np.abs(perp - np.median(perp)))
        if scale > 0:
            w = w / (1.0 + (perp / (2.0 * scale)) ** 2)

    # pairs, filament by filament (segments of one filament are contiguous)
    I, J = [], []
    starts = np.flatnonzero(np.r_[True, np.diff(seg_fil) != 0])
    ends = np.r_[starts[1:], len(seg_fil)]
    for s, e in zip(starts, ends):
        n = e - s
        if n < 2:
            continue
        i, j = np.triu_indices(n, k=1)
        i, j = i + s, j + s
        same = seg_pol[i] == seg_pol[j]
        d = np.abs(seg_pos[j] - seg_pos[i])
        ok = same & (d > 0) & (d <= max_sep)
        I.append(i[ok])
        J.append(j[ok])
    I = np.concatenate(I).astype(np.int64) if I else np.zeros(0, np.int64)
    J = np.concatenate(J).astype(np.int64) if J else np.zeros(0, np.int64)
    D = (seg_pos[J] - seg_pos[I]) * seg_pol[I]
    W = w[I] * w[J]
    return SegmentPairs(
        class_ids=class_ids,
        seg_class=seg_class,
        seg_filament=seg_fil,
        seg_pos=seg_pos,
        seg_pol=seg_pol,
        I=I,
        J=J,
        D=D,
        W=W,
        filament_span=np.asarray(spans),
        filament_keys=keys,
        refined_path=bool(refined),
        max_sep=float(max_sep),
        seg_row=np.concatenate(seg_row).astype(np.int64),
        seg_small=seg_small,
        seg_ref=np.concatenate(seg_ref),
        seg_weight=w,
        class_axis=class_axis,
        class_sign=class_sign,
        class_tilt=_class_medians(
            ((delta_all - axis[seg_class] + 90.0) % 180.0 - 90.0),
            seg_class,
            len(class_ids),
        ),
        class_across=_class_medians(perp, seg_class, len(class_ids)),
        class_count=np.bincount(seg_class, minlength=len(class_ids)),
    )


def class_pair_histogram(pairs, labels=None, pair_weights=None, bin_width=5.0):
    """Offsets binned per ordered class pair: ``(K, K, n_bins)`` and bin centres.

    ``labels`` replaces the segments' class rows (synthetic labels use this);
    ``pair_weights`` multiplies the pair weights (bootstrap resampling uses it).
    """
    labels = pairs.seg_class if labels is None else labels
    K = int(max(pairs.n_classes, labels.max() + 1 if len(labels) else 0))
    nb = int(2 * pairs.max_sep / bin_width) + 1
    idx = np.clip(
        np.round((pairs.D + pairs.max_sep) / bin_width).astype(np.int64), 0, nb - 1
    )
    w = pairs.W if pair_weights is None else pairs.W * pair_weights
    flat = (labels[pairs.I].astype(np.int64) * K + labels[pairs.J]) * nb + idx
    hist = np.bincount(flat, weights=w, minlength=K * K * nb).reshape(K, K, nb)
    centers = np.arange(nb) * bin_width - pairs.max_sep
    return hist, centers


def phase_scan(hist, centers, periods, length_scale=800.0, chunk=64):
    """Score every candidate period and return the class phases at each.

    Parameters
    ----------
    hist, centers : np.ndarray
        From :func:`class_pair_histogram`.
    periods : np.ndarray
        Candidate periods, in A.
    length_scale : float, optional
        Pairs are weighted by ``exp(-d^2 / 2 L^2)``: track lengths accumulate
        error with distance, so distant pairs are less reliable. Defaults to
        800 A.

    Returns
    -------
    scores : np.ndarray
        Top eigenvalue of the normalised, envelope-corrected H at each period.
    phases : np.ndarray
        ``(n_periods, K)`` class phases in radians, from the top eigenvector.

    Notes
    -----
    The class-agnostic part of H -- the periodicity every pair would show if
    class labels carried no phase, which is large at long candidate periods
    where every offset looks like zero -- is subtracted, so only phase
    structure that depends on the class labels scores.
    """
    periods = np.asarray(periods, dtype=float)
    w = np.exp(-0.5 * (centers / length_scale) ** 2)
    hw = hist * w
    n_ab = hw.sum(2)
    n_ab = n_ab + n_ab.T
    deg = n_ab.sum(1)
    norm = np.sqrt(np.outer(deg, deg)) + 1e-12
    total = hw.sum()
    marg = hw.sum((0, 1))
    scores = np.zeros(len(periods))
    phases = np.zeros((len(periods), hist.shape[0]))
    K = hist.shape[0]
    hw_flat = hw.reshape(K * K, -1).astype(complex)
    for s in range(0, len(periods), chunk):
        p = periods[s : s + chunk]
        E = np.exp(2j * np.pi * centers[None, :] / p[:, None])
        # one (K*K, n_bins) x (n_bins, n_periods) product: BLAS, where the
        # equivalent einsum ran as a plain loop and dominated the run time
        H = (hw_flat @ E.T).T.reshape(len(p), K, K)
        H = H + np.conj(np.transpose(H, (0, 2, 1)))
        ebar = (E @ marg) / max(total, 1e-12)
        H = H - n_ab[None] * ebar.real[:, None, None]
        vals, vecs = np.linalg.eigh(H / norm)
        scores[s : s + chunk] = vals[:, -1]
        phases[s : s + chunk] = np.angle(vecs[:, :, -1])
    return scores, phases


def _peak(scores, periods):
    k = int(np.argmax(scores))
    if 0 < k < len(scores) - 1:  # parabolic interpolation between grid points
        y0, y1, y2 = scores[k - 1 : k + 2]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        return float(periods[k] + off * (periods[k + 1] - periods[k])), k
    return float(periods[k]), k


def estimate_period(
    pairs,
    labels=None,
    pair_weights=None,
    periods=None,
    length_scale=800.0,
    refine_fraction=0.03,
):
    """Coarse scan over all periods, then a fine scan about the best one.

    Returns a dict with ``period`` (A), ``score``, the coarse ``periods`` and
    ``scores``, ``phases`` (radians, per class) at the best period, and
    ``support`` -- the fraction of pair weight at separations beyond the period.

    Phases consistent with a period P are also consistent with P/2 (doubled),
    and only a pair of segments more than P apart can tell the two apart. When
    no filament is that long, the half period is at least as good a fit and
    often the higher peak, while the true period's own peak is weak and biased
    -- with 290 A filaments of period 600 A the peaks were 300 A (0.584) and
    about 500 A (0.53). The estimate is then really "this, or twice this", which
    ``support`` below :data:`MIN_SUPPORT` flags; nothing in these data can
    resolve it, and the pair-distance histogram shares the blind spot.

    The same ambiguity arises from the other side when the classes cover only
    part of the ring. Six classes covering half the views of a 300 A repeat are
    placed evenly around a 150 A ring instead, and that fits better than the
    true one. ``double_ratio`` -- the score at twice the period over the score
    at the period -- was 0.92 there, against 0.02-0.56 for every correct
    estimate measured (synthetic full and 9-of-12 selections, EMPIAR-10940 with
    all or its ten largest classes, EMPIAR-10230), so at or above
    :data:`DOUBLE_RATIO_WARN` the estimate may be half the repeat.
    """
    hist, centers = class_pair_histogram(pairs, labels, pair_weights)
    if periods is None:
        periods = np.arange(100.0, pairs.max_sep + 0.1, 2.0)
    scores, _ = phase_scan(hist, centers, periods, length_scale)
    p0, _ = _peak(scores, periods)
    fine = np.arange(p0 * (1 - refine_fraction), p0 * (1 + refine_fraction), 0.25)
    fs, fph = phase_scan(hist, centers, fine, length_scale)
    p, k = _peak(fs, fine)
    w = pairs.W if pair_weights is None else pairs.W * pair_weights
    support = float(w[np.abs(pairs.D) > p].sum() / max(w.sum(), 1e-12))
    s2, _ = phase_scan(hist, centers, np.array([2.0 * p]), length_scale)
    return dict(
        period=p,
        score=float(fs[k]),
        periods=np.asarray(periods),
        scores=scores,
        phases=fph[k],
        support=support,
        double_ratio=float(s2[0] / fs[k]) if fs[k] > 0 else 0.0,
    )


def sync_filament_directions(pairs, est, length_scale=800.0, n_iter=10):
    """Choose each filament's direction to agree with the ring, then refit it.

    Reversing a filament negates every offset in it at once, which turns its
    contribution to the ring into the complex conjugate -- the same ring,
    mirrored. So the direction is one unknown per filament, not per pair, and
    it can be chosen, filament by filament, as whichever of the two agrees
    better with the current ring; the ring is then refitted and the choice
    repeated. Ignoring the order of each pair instead would make the method
    blind to direction only by discarding it: on EMPIAR-10940 that gave
    958 A for a 702 A repeat.

    The per-segment directions from :func:`class_polarity` are the starting
    point; this corrects them where they are wrong. Measured:

    * EMPIAR-10940 with half its filaments reversed at random: the plain ring
      gives 957 A (score 0.55), this gives 701.8 A (score 0.83, the original's)
      and recovers 78% of the reversals, the misses mostly short filaments.
    * Unmodified EMPIAR-10940: 15% of filaments flipped, period and score
      unchanged (701.7 -> 701.8 A, 0.832 -> 0.834) -- harmless.
    * EMPIAR-10230 (cryoSPARC): 49% of filaments flipped and the score rose
      from 0.36 to 0.55 at an unchanged 762 A, so the directions read from the
      in-plane angles there were only weakly informative.

    ``pairs.D`` is changed in place. Returns ``(est, flipped)``: the refitted
    estimate and a boolean per filament.
    """
    pf = pairs.pair_filament
    a, b = pairs.seg_class[pairs.I], pairs.seg_class[pairs.J]
    flipped = np.zeros(pairs.n_filaments, bool)
    for _ in range(n_iter):
        ang = 2 * np.pi * pairs.D / est["period"]
        dphi = est["phases"][a] - est["phases"][b]
        keep = np.bincount(
            pf, weights=pairs.W * np.cos(ang - dphi), minlength=pairs.n_filaments
        )
        swap = np.bincount(
            pf, weights=pairs.W * np.cos(-ang - dphi), minlength=pairs.n_filaments
        )
        flip = swap > keep
        if not flip.any():
            break
        pairs.D = np.where(flip[pf], -pairs.D, pairs.D)
        flipped ^= flip
        # the period moves little once the first scan has found it; searching
        # near it, not the whole range again, is most of this function's speed
        p = est["period"]
        est = estimate_period(
            pairs,
            periods=np.arange(0.9 * p, 1.1 * p, 1.0),
            length_scale=length_scale,
            refine_fraction=0.01,
        )
    return est, flipped


def synthetic_labels(pairs, period, n_classes, noise=0.3, rng=None):
    """Class labels a perfect helix of this period would produce on these segments.

    Each filament gets a random phase; each segment's label is its phase binned
    into ``n_classes`` sectors, and a fraction ``noise`` of labels is replaced at
    random. The segment positions, filament lengths and pair weights are the
    dataset's own, which is the point: the estimator's bias depends on them.
    """
    rng = np.random.default_rng(rng)
    offset = rng.random(pairs.n_filaments)
    phase = (pairs.seg_pos * pairs.seg_pol / period + offset[pairs.seg_filament]) % 1.0
    labels = np.minimum((phase * n_classes).astype(np.int32), n_classes - 1)
    flip = rng.random(len(labels)) < noise
    labels[flip] = rng.integers(0, n_classes, int(flip.sum()))
    return labels


# The pairs handed to the processes of _estimate_periods; they inherit it when
# forked rather than receiving a copy.
_POOL_PAIRS = []


def _estimate_one(task):
    """:func:`estimate_period` of ``_POOL_PAIRS[0]``, one BLAS thread."""
    from threadpoolctl import threadpool_limits

    pairs = _POOL_PAIRS[0]
    task = dict(task)
    counts = task.pop("filament_counts", None)
    if counts is not None:
        task["pair_weights"] = counts[pairs.pair_filament].astype(float)
    with threadpool_limits(limits=1, user_api="blas"):
        return estimate_period(pairs, **task)["period"]


def _pool_workers(pairs, n_tasks):
    """Processes for ``n_tasks`` period fits: the free CPUs, as memory allows.

    Each fit bins every pair on its own, which takes about 40 bytes a pair.
    """
    try:
        import helicon

        gb = max(0.1, 40.0 * len(pairs.D) / 1024**3)
        cpu = min(int(helicon.available_cpu()), int(helicon.available_cpu(gb)))
    except Exception:
        cpu = 1
    return max(1, min(cpu, int(n_tasks)))


def _estimate_periods(pairs, tasks):
    """The period of each of ``tasks`` (keyword arguments of estimate_period).

    ``filament_counts`` in a task, one count per filament, stands for the pair
    weights of a resampling. The fits are independent and run in parallel;
    the random draws are made by the caller, so the result does not depend on
    the number of processes.
    """
    workers = _pool_workers(pairs, len(tasks))
    if workers == 1:
        _POOL_PAIRS[:] = [pairs]
        try:
            return [_estimate_one(t) for t in tasks]
        finally:
            _POOL_PAIRS.clear()
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    _POOL_PAIRS[:] = [pairs]
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
            return list(pool.map(_estimate_one, tasks))
    finally:
        _POOL_PAIRS.clear()


def calibrate_period(
    pairs,
    period_raw,
    length_scale=800.0,
    factors=(0.95, 1.0, 1.05),
    seeds=2,
    noise=0.3,
    rng=0,
):
    """Map a raw period estimate to the period that would produce it here.

    Synthetic labels are generated at a few known periods about the raw
    estimate, on this dataset's own segment geometry, and each is estimated the
    same way. A straight line through (true, estimated) is inverted at the raw
    estimate.

    Returns
    -------
    dict
        ``period`` (calibrated), ``slope`` and ``intercept`` of the fitted line,
        and ``bias`` -- the raw estimate's relative error at the calibrated
        period, e.g. -0.008 for 0.8% low.
    """
    rng = np.random.default_rng(rng)
    grid = np.arange(period_raw * 0.85, period_raw * 1.15, 1.0)
    truth, tasks = [], []
    for f in factors:
        for _ in range(seeds):
            p_true = period_raw * f
            lab = synthetic_labels(pairs, p_true, pairs.n_classes, noise, rng)
            truth.append(p_true)
            tasks.append(dict(labels=lab, periods=grid, length_scale=length_scale))
    est = _estimate_periods(pairs, tasks)
    slope, intercept = np.polyfit(truth, est, 1)
    period = (period_raw - intercept) / slope if slope > 0 else period_raw
    return dict(
        period=float(period),
        slope=float(slope),
        intercept=float(intercept),
        bias=float(period_raw / period - 1.0),
    )


def bootstrap_period(pairs, period_raw, n_boot=20, length_scale=800.0, rng=0):
    """Raw period estimates from filaments resampled with replacement."""
    rng = np.random.default_rng(rng)
    grid = np.arange(period_raw * 0.9, period_raw * 1.1, 1.0)
    tasks = []
    for _ in range(n_boot):
        counts = rng.multinomial(
            pairs.n_filaments, np.full(pairs.n_filaments, 1.0 / pairs.n_filaments)
        )
        tasks.append(
            dict(
                filament_counts=counts,
                periods=grid,
                length_scale=length_scale,
                refine_fraction=0.01,
            )
        )
    return np.asarray(_estimate_periods(pairs, tasks))


def filament_periods(
    pairs,
    phases,
    period,
    labels=None,
    min_span=None,
    rel_range=0.25,
    step=1.0,
    min_pairs=50,
    pair_mask=None,
):
    """The period that best fits each long filament, given the class phases.

    A filament of period P_f places class k at the same fraction of its period
    as every other filament does, so with the phases fixed each filament's
    pairs score a curve ``sum w cos(2 pi d / P_f - (phi_a - phi_b))`` over P_f
    alone.

    Parameters
    ----------
    pairs : SegmentPairs
    phases : np.ndarray
        Class phases in radians, from :func:`estimate_period`.
    period : float
        The pooled period, in A; the search is ``period * (1 +/- rel_range)``.
    labels : np.ndarray, optional
        Segment labels to use in place of the real ones.
    min_span : float, optional
        Filaments shorter than this are skipped: a filament that covers less
        than a period constrains its own period poorly. Defaults to ``period``.
    min_pairs : int, optional
        Filaments with fewer pairs than this are skipped. Defaults to 50.
    pair_mask : np.ndarray of bool, optional
        Use only these pairs (the split-half test uses this).

    Returns
    -------
    dict
        ``filament`` indices, their ``period`` estimates, ``span`` and
        ``n_pairs``. A fit whose best period is at either end of the search
        range has not found a peak, and its period is NaN.
    """
    labels = pairs.seg_class if labels is None else labels
    min_span = period if min_span is None else min_span
    pf = pairs.pair_filament
    use = np.ones(len(pf), bool) if pair_mask is None else np.asarray(pair_mask)
    long_fil = np.flatnonzero(pairs.filament_span >= min_span)
    use = use & np.isin(pf, long_fil)
    n_pairs = np.bincount(pf[use], minlength=pairs.n_filaments)
    long_fil = long_fil[n_pairs[long_fil] >= min_pairs]
    sel = use & np.isin(pf, long_fil)
    d, w, f = pairs.D[sel], pairs.W[sel], pf[sel]
    dphi = phases[labels[pairs.I[sel]]] - phases[labels[pairs.J[sel]]]
    grid = np.arange(
        period * (1 - rel_range), period * (1 + rel_range) + step / 2, step
    )
    remap = np.full(pairs.n_filaments, -1)
    remap[long_fil] = np.arange(len(long_fil))
    fi = remap[f]
    curves = _filament_curves(fi, len(long_fil), d, w, dphi, grid)
    best = np.full(len(long_fil), np.nan)
    for i in range(len(long_fil)):
        k = int(np.argmax(curves[:, i]))
        if 0 < k < len(grid) - 1:
            best[i] = _peak(curves[:, i], grid)[0]
    return dict(
        filament=long_fil,
        period=best,
        span=pairs.filament_span[long_fil],
        n_pairs=n_pairs[long_fil],
    )


def _filament_curves(fi, n_fil, d, w, dphi, grid, bin_width=0.5):
    """``sum w cos(2 pi d / p - dphi)`` over each filament's pairs, at each ``p``.

    The pairs are first binned by separation, each carrying ``w exp(-i dphi)``,
    so that the sum over pairs becomes one matrix product per batch of
    filaments: ``(filaments, bins) x (bins, periods)``. Binning to
    ``bin_width`` moves a pair by at most a quarter of an Angstrom, a phase
    error of 2 pi 0.25 / P (0.002 rad at a 700 A repeat). Evaluating every pair
    at every period took 24 of the 83 s of a fit of EMPIAR-10940.
    """
    curves = np.zeros((len(grid), n_fil))
    if not n_fil or not len(d):
        return curves
    lo = float(d.min())
    b = np.round((d - lo) / bin_width).astype(np.int64)
    # as many bins as the rounded positions reach: with int() a pair at the
    # longest separation could round one bin past the end, into the next
    # filament's first bin (or past the array, for the last filament)
    nb = int(b.max()) + 1
    c = w * np.exp(-1j * dphi)
    centres = lo + np.arange(nb) * bin_width
    E = np.exp(2j * np.pi * centres[:, None] / np.asarray(grid)[None, :])
    # A filament's pairs fill only the bins its own length reaches, and most
    # filaments are far shorter than the longest: with every filament over all
    # the bins, most of the product multiplied zeros, more so the longer the
    # repeat searched. The filaments are taken shortest first, in batches that
    # each use only the rows of E their pairs reach (the same bins and sums).
    first = np.full(n_fil, nb, np.int64)
    last = np.full(n_fil, -1, np.int64)
    np.minimum.at(first, fi, b)
    np.maximum.at(last, fi, b)
    zero = int(np.round(-lo / bin_width))  # the bin of separation 0
    reach = np.maximum(zero - first, last - zero).clip(min=0)
    by_reach = np.argsort(reach, kind="stable")
    rank = np.empty(n_fil, np.int64)
    rank[by_reach] = np.arange(n_fil)
    order = np.argsort(rank[fi], kind="stable")
    fr, b, c = rank[fi][order], b[order], c[order]
    edges = np.searchsorted(fr, np.arange(n_fil + 1))
    s0 = 0
    while s0 < n_fil:
        s1 = s0 + 1
        # grow the batch while the binned array stays small
        while s1 < n_fil and (s1 + 1 - s0) * (2 * reach[by_reach[s1]] + 1) <= 2e7:
            s1 += 1
        r = int(reach[by_reach[s1 - 1]])
        i0, i1 = max(0, zero - r), min(nb, zero + r + 1)
        width = i1 - i0
        lo_p, hi_p = edges[s0], edges[s1]
        flat = (fr[lo_p:hi_p] - s0) * width + (b[lo_p:hi_p] - i0)
        size = (s1 - s0) * width
        h = np.bincount(flat, weights=c[lo_p:hi_p].real, minlength=size) + 1j * (
            np.bincount(flat, weights=c[lo_p:hi_p].imag, minlength=size)
        )
        curves[:, by_reach[s0:s1]] = (h.reshape(s1 - s0, width) @ E[i0:i1]).real.T
        s0 = s1
    return curves


def _smooth_angle(t, angle, span=None):
    """A low-order polynomial fit of an angle (degrees) along a filament.

    The angle is unwrapped first and the fit is robust to outliers: a filament
    that bends smoothly keeps its bend, and the segment-to-segment scatter of
    the alignment is averaged out.
    """
    t = np.asarray(t, dtype=float)
    a = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(angle, dtype=float))))
    n = len(t)
    span = float(t.max() - t.min()) if span is None else span
    if n < 3 or span <= 0:
        return np.full(n, a.mean()) if n else a
    degree = min(1 if span < 500 else (2 if span < 2000 else 3), n - 2)
    u = (t - t.min()) / span
    keep = np.ones(n, bool)
    for _ in range(3):
        c = np.polyfit(u[keep], a[keep], degree)
        r = a - np.polyval(c, u)
        scale = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) < max(3.0 * scale, 1.0)
        if new.sum() <= degree + 1 or (new == keep).all():
            break
        keep = new
    return np.polyval(c, u)


def segment_angles(params, result, fold=1):
    """Euler angles and origins for every segment, side views at tilt 90.

    * ``rlnAngleRot``: the segment's own azimuthal angle, placed along its
      filament's fit, ``phi(z) = phi0 - 2 pi e z / P`` with the filament's repeat
      ``P``, its direction ``e`` (+1 or -1) and ``phi0`` the circular mean of
      ``phi + 2 pi e z / P`` over its analyzed segments. It agrees with the
      class angles but is finer than a class angle and averages the
      classification noise of the filament's segments (a segment put in the
      wrong class does not spoil its own angle).
    * ``rlnAnglePsi``: the psi prior (the direction the filament was picked in)
      plus the segment's Class2D in-plane deviation from it, turned by 180
      degrees for a filament that points against the reference direction (the
      direction is a property of the whole filament: its majority over its
      segments), then smoothed along the filament so that a bending filament
      keeps its bend and the per-segment scatter is averaged out.
      ``rlnAnglePsiPrior`` is the prior turned the same way.
    * ``rlnAngleTilt`` and ``rlnAngleTiltPrior``: 90.
    * ``rlnOriginXAngst``, ``rlnOriginYAngst``: the refined segment centre
      (coordinate minus Class2D origin, including any shift the counterpart
      merge of :func:`analyze` applied) moved onto the smoothed filament path
      across it. A class average that is not centred on its filament, and the
      alignment noise, scatter the centres across the path; the filament as a
      whole is straight to a low-order curve, so this synchronizes the centres
      of the segments. The position along the filament is kept, since the
      azimuthal angle refers to it.

    Parameters
    ----------
    params : pandas.DataFrame
        The Class2D parameters :func:`analyze` ran on.
    result : PhasePitchResult
    fold : int, optional
        The pitch is ``fold`` times the repeat distance (the point-group
        symmetry about the axis, times 2 for a 2_1 axis); the azimuthal angle
        advances 360/fold degrees per repeat, so the ring angle is divided by
        it. Defaults to 1.

    Returns
    -------
    pandas.DataFrame
        One row per row of ``params``. NaN in the angle columns for segments of
        filaments that were not analyzed; the origin columns are those of
        ``params`` where no path could be fitted. The origin of the azimuthal
        angle, the sign of its direction and the choice among the ``fold``
        equivalent angles are arbitrary: the most populated class is at 0.
    """
    seg = result.segments
    if seg is None:
        raise ValueError("the result has no per-segment directions")
    used = result.params_used if result.params_used is not None else params
    n = len(params)
    azimuth = np.full(n, np.nan)
    psi = np.full(n, np.nan)
    psi_prior = np.full(n, np.nan)
    fil = result.filaments
    pitch_of = {
        (m, t): p
        for m, t, p in zip(
            fil["rlnMicrographName"], fil["rlnHelicalTubeID"], fil["pitch"]
        )
        if np.isfinite(p)
    }
    index_of = {k: i for i, k in enumerate(result.filament_keys)}
    analyzed = seg.groupby("filament").indices
    rows_of = used.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False)
    rows_of = rows_of.indices
    apix = _micrograph_apix(used)
    refined = bool(
        result.refined_path
        and apix is not None
        and all(c in used for c in _REFINED_COLUMNS)
    )
    origin_x = (
        used["rlnOriginXAngst"].values.astype(float).copy()
        if "rlnOriginXAngst" in used
        else np.zeros(n)
    )
    origin_y = (
        used["rlnOriginYAngst"].values.astype(float).copy()
        if "rlnOriginYAngst" in used
        else np.zeros(n)
    )
    track_all = used["rlnHelicalTrackLengthAngst"].values.astype(float)
    prior_all = (
        used["rlnAnglePsiPrior"].values.astype(float)
        if "rlnAnglePsiPrior" in used
        else None
    )
    for key, period in pitch_of.items():
        if key not in rows_of or index_of.get(key) not in analyzed:
            continue
        rows = np.sort(rows_of[key])
        order = np.argsort(track_all[rows], kind="stable")
        h = used.iloc[rows[order]]
        track_sorted = track_all[rows[order]]
        if refined:
            ox = h["rlnOriginXAngst"].values if "rlnOriginXAngst" in h else 0.0
            oy = h["rlnOriginYAngst"].values if "rlnOriginYAngst" in h else 0.0
            cx = h["rlnCoordinateX"].values.astype(float) * apix
            cy = h["rlnCoordinateY"].values.astype(float) * apix
            geo = refined_path_geometry(
                track_sorted, cx - np.asarray(ox, float), cy - np.asarray(oy, float)
            )
            z_sorted = geo["position"]
            # the centre on the smoothed path, keeping the position along it
            new_x = geo["x"] + geo["along"] * geo["tx"]
            new_y = geo["y"] + geo["along"] * geo["ty"]
            origin_x[rows[order]] = cx - new_x
            origin_y[rows[order]] = cy - new_y
        else:
            z_sorted = track_sorted
        z = np.empty(len(rows))
        z[order] = z_sorted
        k = analyzed[index_of[key]]
        at = np.searchsorted(rows, seg["row"].values[k])
        e = seg["direction"].values[k]
        phi = seg["azimuth"].values[k]
        e_f = 1.0 if e.sum() >= 0 else -1.0
        use = e == e_f
        z_seg = z[at]
        off = np.exp(1j * (phi + 2 * np.pi * e_f * z_seg / period))
        vote = seg["weight"].values[k]
        phi0 = np.angle((off * vote)[use].sum())
        for _ in range(3):
            # segments put in the wrong class do not vote on the filament's angle
            near = use & (np.cos(np.angle(off) - phi0) > 0.0)
            if near.sum() < 2:
                break
            phi0 = np.angle((off * vote)[near].sum())
        az = phi0 - 2 * np.pi * e_f * z / period
        small = np.full(len(rows), seg["small"].values[k].mean())
        small[at] = seg["small"].values[k]
        if prior_all is not None:
            ref = prior_all[rows]
        else:
            ref = np.full(len(rows), seg["ref"].values[k][0])
        turn = 180.0 if e_f < 0 else 0.0
        raw = ref + small + turn
        smooth = np.empty(len(rows))
        smooth[order] = _smooth_angle(z[order], raw[order])
        azimuth[rows] = np.rad2deg(np.angle(np.exp(1j * az))) / fold
        psi[rows] = (smooth + 180.0) % 360.0 - 180.0
        psi_prior[rows] = (ref + turn + 180.0) % 360.0 - 180.0
    out = dict(
        rlnAngleRot=azimuth,
        rlnAnglePsi=psi,
        rlnAnglePsiPrior=psi_prior,
        rlnAngleTilt=90.0,
        rlnAngleTiltPrior=90.0,
    )
    if refined:
        out["rlnOriginXAngst"] = origin_x
        out["rlnOriginYAngst"] = origin_y
        for col, values in (("rlnOriginX", origin_x), ("rlnOriginY", origin_y)):
            if col in params:
                out[col] = values / apix
    return pd.DataFrame(out)


def select_segments(
    params,
    filaments,
    low=None,
    high=None,
    min_length=None,
    max_length=None,
    angles=None,
    class_numbers=None,
):
    """Segments of the filaments whose pitch and length are in the given ranges.

    Every returned segment carries its filament's estimated pitch and length in
    the columns ``heliconFilamentPitch`` and ``heliconFilamentLength`` (A), and
    how well its direction is determined in ``heliconFilamentDirectionScore``
    (0 for a coin flip, 1 for a clear direction; short filaments score low), so
    that the star file can be filtered further downstream. Filaments without a
    pitch estimate are never returned.

    Parameters
    ----------
    params : pandas.DataFrame
        The Class2D parameters the analysis ran on.
    filaments : pandas.DataFrame
        ``PhasePitchResult.filaments``.
    low, high : float, optional
        Pitch band, in A. A bound left as None is not applied.
    min_length, max_length : float, optional
        Range of the filament length (``span`` of the filament table: the
        extent of its segments in the analyzed classes), in A. A bound left as
        None is not applied.
    class_numbers : sequence of int, optional
        Keep only the segments in these classes: the selection an analysis was
        run on. The filaments' other segments are left out of the export.
    angles : pandas.DataFrame, optional
        From :func:`segment_angles`, one row per row of ``params``. When given,
        its columns are written to the selected segments.

    Returns
    -------
    pandas.DataFrame
        The rows of ``params`` belonging to those filaments, with the
        ``attrs`` (optics) carried over so it can be written as a star file.
    """
    keep = filaments["pitch"].notna()
    if low is not None:
        keep &= filaments["pitch"] >= low
    if high is not None:
        keep &= filaments["pitch"] <= high
    if min_length is not None:
        keep &= filaments["span"] >= min_length
    if max_length is not None:
        keep &= filaments["span"] <= max_length
    band = filaments[keep]
    score = band["direction_score"] if "direction_score" in band else band["span"] * 0
    info = {
        (m, t): (p, length, sc)
        for m, t, p, length, sc in zip(
            band["rlnMicrographName"],
            band["rlnHelicalTubeID"],
            band["pitch"],
            band["span"],
            score,
        )
    }
    keys = list(zip(params["rlnMicrographName"], params["rlnHelicalTubeID"]))
    mask = np.array([k in info for k in keys], dtype=bool)
    if class_numbers is not None:
        mask &= params["rlnClassNumber"].astype(int).isin(list(class_numbers)).values
    out = params.loc[mask].copy()
    found = [info[k] for k, m in zip(keys, mask) if m]
    out["heliconFilamentPitch"] = [f[0] for f in found]
    out["heliconFilamentLength"] = [f[1] for f in found]
    if "direction_score" in band:
        out["heliconFilamentDirectionScore"] = [f[2] for f in found]
    if angles is not None:
        for col in angles:
            out[col] = angles[col].values[mask]
    out.attrs = dict(params.attrs)
    return out


def class_fit(pairs, phases, period, min_separation=0.25, prior=0.05):
    """How well each class sits on the ring, as evidence that it is a real view.

    For every pair that involves class k, the agreement is
    ``cos(2 pi d / P - (phi_a - phi_b))`` -- 1 when the pair is exactly where
    the ring puts it, 0 on average for a class whose segments have nothing to
    do with the ring. Two choices keep a junk class from looking good:

    * Only pairs at least ``min_separation`` of a period apart count. Close
      neighbours are at nearly the same azimuth whatever their classes, so
      they agree with any phase a class is given: on EMPIAR-10940 an ice blob
      (class 35) scored 0.53 with them and -0.16 without.
    * The mean is shrunk towards 0 in proportion to how little evidence the
      class has: by ``n / (n + n0)``, ``n`` its number of such pairs and ``n0``
      ``prior`` times the median class's (:func:`class_evidence`). A class of 1
      or 3 particles is not taken to fit; a large class is barely affected. The
      count, not the pair weights, measures the evidence: the weights only say
      how much each pair counts within the class's mean, and on EMPIAR-10940
      the tilted views have low weights (their centres sit off the filament
      path) with as many pairs as any.

    Returns
    -------
    np.ndarray
        Shrunk mean agreement per class; NaN for a class with no such pairs.
    """
    a, b = pairs.seg_class[pairs.I], pairs.seg_class[pairs.J]
    far = np.abs(pairs.D) >= min_separation * period
    w = pairs.W * far
    r = w * np.cos(2 * np.pi * pairs.D / period - (phases[a] - phases[b]))
    K = pairs.n_classes
    num = np.bincount(a, weights=r, minlength=K) + np.bincount(
        b, weights=r, minlength=K
    )
    den = np.bincount(a, weights=w, minlength=K) + np.bincount(
        b, weights=w, minlength=K
    )
    n = class_evidence(pairs, period, min_separation)
    n0 = prior * float(np.median(n[n > 0])) if np.any(n > 0) else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den * (n / (n + n0)), np.nan)


def class_evidence(pairs, period, min_separation=0.25):
    """Each class's number of pairs ``min_separation`` periods or more apart.

    The evidence :func:`class_fit` rests on: the informative pairs a class
    takes part in, with another segment of the selection on the same filament.
    """
    a, b = pairs.seg_class[pairs.I], pairs.seg_class[pairs.J]
    far = np.abs(pairs.D) >= min_separation * period
    K = pairs.n_classes
    return (np.bincount(a[far], minlength=K) + np.bincount(b[far], minlength=K)).astype(
        float
    )


TOO_FEW = "too few informative segment pairs to judge"
AT_CHANCE = "fits the ring no better than chance"


def diagnose_classes(result, prior=0.05):
    """The classes to take out of a fit, each with the reason.

    * ``TOO_FEW``: a class with fewer informative pairs (:func:`class_evidence`)
      than ``prior`` times the median class's -- too little to tell either
      way, as for a class of a handful of segments, or one whose segments are
      on filaments with few others of the selection;
    * ``AT_CHANCE``: a class flagged by :func:`poorly_fitting` -- its segments do
      not sit at one azimuth on the ring: junk, or another type;
    * ``"180-degree copy of class N"``: a class merged into a flagged class as
      its turned counterpart, which goes with it.

    Only the segments are consulted, not the look of the averages: distinct
    helical types can differ in morphology too subtly for that.

    Parameters
    ----------
    result : PhasePitchResult

    Returns
    -------
    dict
        Class number -> reason, the merged copies last.
    """
    ids = [int(c) for c in result.class_ids]
    out = {}
    evidence = result.class_evidence
    if evidence is not None and len(evidence):
        ev = np.asarray(evidence, dtype=float)
        typical = float(np.median(ev[ev > 0])) if np.any(ev > 0) else 0.0
        for c, e in zip(ids, ev):
            if e < prior * typical:
                out[c] = TOO_FEW
    for c in poorly_fitting(ids, result.class_fit):
        out.setdefault(c, AT_CHANCE)
    for gone, kept in (result.merged_into or {}).items():
        if int(kept) in out and int(gone) not in out:
            out[int(gone)] = f"180\u00b0 copy of class {int(kept)}"
    return out


def poorly_fitting(class_ids, fit, level=0.1, relative=0.25):
    """The classes whose fit is close to chance: junk, or another type.

    A class is flagged when its :func:`class_fit` is below ``level``, or below
    ``relative`` times the median class's when the whole selection fits less
    well (a poorer dataset). Needs three classes with a fit.

    Returns
    -------
    list of int
        Class numbers, as in ``class_ids``.
    """
    fit = np.asarray(fit, dtype=float)
    ok = np.isfinite(fit)
    if ok.sum() < 3:
        return []
    cut = min(level, relative * float(np.median(fit[ok])))
    return [int(c) for c, f in zip(class_ids, fit) if not np.isfinite(f) or f < cut]


def largest_phase_gap(phases, weight=None):
    """The widest stretch of the ring, in degrees, that no selected class covers.

    Classes cover the azimuths they were picked for; a selection that leaves a
    large part of the ring empty constrains the phases, and so the period,
    only through the views it has.
    """
    phases = np.asarray(phases, dtype=float)
    if weight is not None:
        phases = phases[np.asarray(weight) > 0]
    if len(phases) < 2:
        return 360.0
    t = np.sort(np.rad2deg(phases) % 360.0)
    gaps = np.diff(np.concatenate([t, [t[0] + 360.0]]))
    return float(gaps.max())


def _robust_sd(x):
    x = np.asarray(x, dtype=float)
    return (
        float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")
    )


def filament_heterogeneity(
    pairs, period, phases, min_span_factor=1.2, min_pairs=30, n_boot=1000, rng=0, **kw
):
    """Do filaments differ in period? A split-half test that needs no noise model.

    Each filament at least ``min_span_factor`` periods long is cut at its
    midpoint and each half is fitted on its own pairs. Estimation noise is
    independent between the halves; a real difference between filaments is
    shared by both. So the covariance of the two halves' periods across
    filaments estimates the variance of the real per-filament period, with no
    assumption about how noisy each fit is.

    A simulated null -- the same fit on synthetic labels at one fixed period --
    was the first version of this test and is not used: synthetic labels are
    cleaner than real classes, so it under-states the noise and reads it as
    heterogeneity. Measured on EMPIAR-10940 the halves of 178 filaments
    correlate at 0.48, a real per-filament sd of 44 A (95% CI 35-51), about 6%
    of the period; on the 93 filaments made almost entirely of one polarity of
    class the correlation is stronger, 0.66 (sd 49 A), so it is not an
    artefact of mixing mirrored classes, which if anything add noise.

    Returns
    -------
    dict
        ``filament``, the two halves' periods ``first`` and ``second``, the
        whole filament's ``period``, ``corr``, ``sd_between`` (A) and its 95%
        bootstrap interval ``sd_between_ci``, and ``n`` filaments used.
    """
    rng = np.random.default_rng(rng)
    pf = pairs.pair_filament
    long_fil = np.flatnonzero(pairs.filament_span >= min_span_factor * period)
    lo_pos = np.full(pairs.n_filaments, np.inf)
    hi_pos = np.full(pairs.n_filaments, -np.inf)
    np.minimum.at(lo_pos, pairs.seg_filament, pairs.seg_pos)
    np.maximum.at(hi_pos, pairs.seg_filament, pairs.seg_pos)
    mid = 0.5 * (lo_pos + hi_pos)
    a = pairs.seg_pos[pairs.I] < mid[pf]
    b = pairs.seg_pos[pairs.J] < mid[pf]
    fits = []
    for mask in (a & b, ~a & ~b):
        r = filament_periods(
            pairs,
            phases,
            period,
            min_span=min_span_factor * period,
            min_pairs=min_pairs,
            pair_mask=mask,
            **kw,
        )
        fits.append(dict(zip(r["filament"], r["period"])))
    whole = filament_periods(
        pairs, phases, period, min_span=min_span_factor * period, **kw
    )
    whole = dict(zip(whole["filament"], whole["period"]))
    common = np.array(sorted(set(fits[0]) & set(fits[1]) & set(long_fil)), dtype=int)
    x = np.array([fits[0][f] for f in common])
    y = np.array([fits[1][f] for f in common])
    ok = np.isfinite(x) & np.isfinite(y)
    common, x, y = common[ok], x[ok], y[ok]
    out = dict(
        filament=common,
        first=x,
        second=y,
        period=np.array([whole.get(f, np.nan) for f in common]),
        n=len(common),
        corr=float("nan"),
        sd_between=float("nan"),
        sd_between_ci=(float("nan"), float("nan")),
    )
    if len(common) < 10:
        return out
    cov = np.cov(x, y)[0, 1]
    boot = []
    for _ in range(n_boot):
        k = rng.integers(0, len(x), len(x))
        boot.append(np.cov(x[k], y[k])[0, 1])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out.update(
        corr=float(np.corrcoef(x, y)[0, 1]),
        sd_between=float(np.sqrt(max(cov, 0.0))),
        sd_between_ci=(float(np.sqrt(max(lo, 0.0))), float(np.sqrt(max(hi, 0.0)))),
    )
    return out


def class_groups(pairs, threshold=0.9, min_weight=0.005):
    """Split classes into groups that share filaments with each other.

    Two helical types classified into separate classes appear together only in
    filaments of their own type, so the class co-occurrence matrix is block
    diagonal. The number of groups is the number of eigenvalues of the
    normalised co-occurrence above ``threshold``; with one group every class is
    returned together.

    Same-class pairs are left out of the co-occurrence -- neighbouring segments
    mostly share a class, and that diagonal alone pushes several eigenvalues
    towards one -- and so are classes holding less than ``min_weight`` of the
    pairs, each of which would otherwise be a block of its own. With both, the
    second eigenvalue is 0.76 on EMPIAR-10940 (one type) and 0.96 on the same
    data made into two types with distinct classes, which the default
    threshold separates. A class too small to group joins the group it shares
    the most pairs with.

    Types that *share* classes are not separated this way;
    :func:`filament_heterogeneity` is what shows their spread.

    Returns
    -------
    list of np.ndarray
        Rows of ``pairs.class_ids`` in each group, largest first.
    """
    hist, _ = class_pair_histogram(pairs)
    co = hist.sum(2)
    co = co + co.T
    np.fill_diagonal(co, 0.0)
    deg = co.sum(1)
    ok = deg > min_weight * max(deg.sum(), 1e-12)
    rows = np.flatnonzero(ok)
    small = np.flatnonzero(~ok)
    if len(rows) < 2:
        return [np.arange(pairs.n_classes)]
    A = co[np.ix_(ok, ok)] / np.sqrt(np.outer(deg[ok], deg[ok]))
    vals, vecs = np.linalg.eigh(A)
    k = int(np.sum(vals > threshold))
    if k <= 1:
        return [np.arange(pairs.n_classes)]
    from scipy.cluster.vq import kmeans2

    emb = vecs[:, -k:]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    _, lab = kmeans2(emb, k, seed=0, minit="++")
    groups = [list(rows[lab == g]) for g in range(k) if np.any(lab == g)]
    for c in small:  # to the group it shares the most pairs with
        share = [co[c, g].sum() for g in groups]
        groups[int(np.argmax(share))].append(c)
    return sorted(
        [np.sort(np.array(g, dtype=int)) for g in groups], key=len, reverse=True
    )


def _bin_to_width(image, max_width=128):
    """Block-average an image so it is at most ``max_width`` pixels wide."""
    image = np.asarray(image, dtype=np.float32)
    f = int(np.ceil(image.shape[1] / max_width))
    if f <= 1:
        return image
    ny, nx = (image.shape[0] // f) * f, (image.shape[1] // f) * f
    return image[:ny, :nx].reshape(ny // f, f, nx // f, f).mean((1, 3))


def _register_prepared(images, indices, max_width):
    """Class averages with their filament horizontal, ready to register."""
    from . import helix_transform as HT

    raw = [_bin_to_width(images[i], max_width) for i in indices]
    factor = [
        max(1, int(np.ceil(np.asarray(images[i]).shape[1] / max_width)))
        for i in indices
    ]
    at = HT.auto_transform(raw)
    prepared = {
        i: HT.apply_transform(im, rotation=r, shift_y=sy, crop_size=at.crop_size)
        for i, im, (r, sy) in zip(indices, raw, at.per_image)
    }
    return prepared, dict(zip(indices, factor))


_POOL_IMAGES = {}


def _register_chunk(args):
    from . import denovo3d_register as R

    pairs, turn, refine = args
    return [
        R.register_pair(_POOL_IMAGES[i], _POOL_IMAGES[j], flips=turn, refine=refine)
        for i, j in pairs
    ]


def _register_all(prepared, todo, turn, progress=None, refine=False):
    """Register every pair in ``todo``, on all available CPUs when there are many.

    The results are in the order of ``todo``.
    """
    from . import denovo3d_register as R

    workers = 1
    try:
        import helicon

        workers = max(1, min(int(helicon.available_cpu()), 32))
    except Exception:
        pass
    if workers == 1 or len(todo) < 40:
        out = []
        for k, (i, j) in enumerate(todo):
            if progress is not None:
                progress(k, len(todo))
            out.append(
                R.register_pair(prepared[i], prepared[j], flips=turn, refine=refine)
            )
        return out
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    _POOL_IMAGES.clear()
    _POOL_IMAGES.update(prepared)
    size = max(1, len(todo) // (workers * 4))
    chunks = [(todo[k : k + size], turn, refine) for k in range(0, len(todo), size)]
    out = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
        for k, part in enumerate(pool.map(_register_chunk, chunks)):
            out.extend(part)
            if progress is not None:
                progress(min(len(todo), (k + 1) * size), len(todo))
    return out


def pair_counterparts(images, indices, min_corr=0.85, max_width=128, progress=None):
    """Pair up classes that are each other's 180-degree rotation.

    A class of backward-picked filaments is its forward class turned by 180
    degrees in the plane and shifted along the axis by the difference of how
    the two averages were centred. The pair is registered on the class
    averages, so that the two can be treated as one class afterwards.

    Parameters
    ----------
    images : sequence of 2D arrays
        Class averages.
    indices : sequence of int
        Positions in ``images`` of the classes to pair among.
    min_corr : float, optional
        Smallest correlation of a pair. Defaults to 0.85.
    max_width : int, optional
        Images are block-averaged to at most this width first. Defaults to 128.

    Returns
    -------
    list of dict
        ``a``, ``b`` (positions in ``images``), ``corr`` and ``dx`` (pixels of
        the original images: how far ``b`` turned by 180 degrees lies along the
        axis from ``a``). Each class is in at most one pair, the best ones
        first; a class that looks the same both ways pairs with nothing.
    """
    from . import denovo3d_register as R

    indices = [int(i) for i in indices]
    if len(indices) < 2:
        return []
    prepared, factor = _register_prepared(images, indices, max_width)
    turn = [(True, True)]
    self_corr = {
        i: R.register_pair(prepared[i], prepared[i], flips=turn, refine=False)["corr"]
        for i in indices
    }
    todo = [(i, j) for x, i in enumerate(indices) for j in indices[x + 1 :]]
    results = _register_all(prepared, todo, turn, progress)
    found = []
    for (i, j), res in zip(todo, results):
        if res["corr"] >= min_corr and res["corr"] > max(self_corr[i], self_corr[j]):
            found.append(
                dict(a=i, b=j, corr=float(res["corr"]), dx=float(res["dx"]) * factor[i])
            )
    found.sort(key=lambda d: d["corr"], reverse=True)
    used, out = set(), []
    for d in found:
        if d["a"] not in used and d["b"] not in used:
            used |= {d["a"], d["b"]}
            out.append(d)
    return out


def _direction_convention(params, min_span=300.0):
    """+1 or -1: the sign of the micrograph y axis in the psi angle's convention.

    The direction of a filament in the coordinates is ``atan2(sign * dy, dx)``
    for the sign that agrees with the psi prior.
    """
    devs = {1.0: [], -1.0: []}
    order, starts, ends, _ = _filament_slices(params)
    track = params["rlnHelicalTrackLengthAngst"].to_numpy().astype(float)[order]
    x = params["rlnCoordinateX"].to_numpy().astype(float)[order]
    y = params["rlnCoordinateY"].to_numpy().astype(float)[order]
    prior_all = params["rlnAnglePsiPrior"].to_numpy().astype(float)[order]
    for a, b in zip(starts, ends):
        if b - a < 4 or track[b - 1] - track[a] < min_span:
            continue
        dx = float(x[b - 1] - x[a])
        dy = float(y[b - 1] - y[a])
        prior = np.deg2rad(float(np.nanmedian(prior_all[a:b])))
        for sgn in devs:
            angle = np.arctan2(sgn * dy, dx)
            devs[sgn].append(abs(np.angle(np.exp(2j * (angle - prior)))))
    if not devs[1.0]:
        return 1.0
    return 1.0 if np.median(devs[1.0]) <= np.median(devs[-1.0]) else -1.0


def merge_counterparts(params, pairs, image_apix, sign=1.0, class_of=None):
    """Treat each turned class as its counterpart, turned back and shifted.

    Segments of the smaller class of each pair take the class number of the
    larger, their psi is turned by 180 degrees and their refined centre is moved
    along the axis by the registered offset (``sign`` is the direction of that
    move; the two signs are the two conventions the registration could use, and
    the caller keeps the one that gives the better ring).

    Parameters
    ----------
    params : pandas.DataFrame
    pairs : list of dict
        ``a``, ``b`` class numbers and ``dx`` in class-average pixels.
    image_apix : float
        Pixel size of the class averages, in A.
    sign : float, optional
        +1 or -1.

    Returns
    -------
    pandas.DataFrame
        A copy of ``params`` with the merged segments changed, and the class
        numbers of the merged-away classes.
    """
    out = params.copy()
    cls = out["rlnClassNumber"].astype(int).values
    counts = pd.Series(cls).value_counts()
    conv = _direction_convention(params) if "rlnCoordinateX" in params else 1.0
    gone = []
    for d in pairs:
        a, b = int(d["a"]), int(d["b"])
        keep, drop = (a, b) if counts.get(a, 0) >= counts.get(b, 0) else (b, a)
        rows = cls == drop
        if not rows.any():
            continue
        psi = (out["rlnAnglePsi"].values.astype(float) + 180.0 + 180.0) % 360.0 - 180.0
        out.loc[rows, "rlnAnglePsi"] = psi[rows]
        shift = sign * d["dx"] * image_apix
        ang = np.deg2rad(psi[rows])
        for col, comp in (
            ("rlnOriginXAngst", np.cos(ang)),
            ("rlnOriginYAngst", conv * np.sin(ang)),
        ):
            if col in out:
                out.loc[rows, col] = (
                    out.loc[rows, col].values.astype(float) + shift * comp
                )
        out.loc[rows, "rlnClassNumber"] = keep
        gone.append(drop)
    return out, gone


def _merged_into(counterparts, merged):
    """Merged-away class -> the class it was merged into."""
    gone = {int(c) for c in merged}
    out = {}
    for d in counterparts or []:
        a, b = int(d["a"]), int(d["b"])
        if a in gone:
            out[a] = b
        elif b in gone:
            out[b] = a
    return out


def suggest_counterparts(
    images, selected, candidates, min_corr=0.85, max_width=128, progress=None
):
    """Unselected classes that look like a selected class turned 180 degrees.

    When a 2D classification keeps the two picking directions of a filament in
    separate classes -- as EMPIAR-10940's did, never flipping a segment's psi
    although RELION allowed it -- a type's backward-picked filaments sit in
    classes that are its forward classes rotated by 180 degrees. Leaving those
    out of a selection drops those filaments, and with them half the views,
    which puts the ring at half its size (see ``estimate_period``). On
    EMPIAR-10940, this registration paired 16 classes, 94% of them a reversed
    with a non-reversed class by an independent pixel check, at a median
    correlation of 0.98. With the 9 non-reversed of its 20 largest classes
    selected, this function suggested 15 of the other 33 classes, all 15
    reversed ones (of 20).

    Parameters
    ----------
    images : sequence of 2D arrays
        Class averages, indexed like ``selected`` and ``candidates``.
    selected, candidates : sequence of int
        Indices of the selected classes and of the classes to consider.
    min_corr : float, optional
        Smallest correlation reported. Defaults to 0.85.
    max_width : int, optional
        Images are block-averaged to at most this width first; the match is on
        the filament's shape, which needs no more. Defaults to 128.

    Returns
    -------
    list of dict
        ``selected``, ``candidate`` and ``corr``, best first, one per candidate
        -- the selected class it matches best -- for candidates whose rotated
        match beats ``min_corr`` and beats the selected class's match with its
        own rotated self (a class that looks the same both ways is its own
        counterpart and needs no partner).
    """
    from . import helix_transform as HT

    selected = [int(i) for i in selected]
    candidates = [int(j) for j in candidates if int(j) not in set(selected)]
    if not selected or not candidates:
        return []
    idx = sorted(set(selected) | set(candidates))
    raw = [_bin_to_width(images[i], max_width) for i in idx]
    at = HT.auto_transform(raw)
    prepared = {
        i: HT.apply_transform(im, rotation=r, shift_y=sy, crop_size=at.crop_size)
        for i, im, (r, sy) in zip(idx, raw, at.per_image)
    }
    rot = [(True, True)]  # flip x and y: a 180-degree rotation
    # every registration at once, on all available CPUs (one at a time, the
    # 360 of 12 selected classes and 30 candidates took 53 s); the results come
    # back in order, so the best match of each candidate is chosen as before
    selves = [(i, i) for i in selected]
    todo = [(i, j) for i in selected for j in candidates]
    corr = [
        r["corr"]
        for r in _register_all(prepared, selves + todo, rot, progress, refine=True)
    ]
    self_corr = dict(zip(selected, corr[: len(selves)]))
    best = {}
    for (i, j), c in zip(todo, corr[len(selves) :]):
        if c >= min_corr and c > self_corr[i] and c > best.get(j, (None, -1))[1]:
            best[j] = (i, c)
    out = [dict(selected=i, candidate=j, corr=float(c)) for j, (i, c) in best.items()]
    return sorted(out, key=lambda d: d["corr"], reverse=True)


def suggest_expansion(
    params,
    selected,
    candidates=None,
    min_segments=5,
    seed_share=0.5,
    min_filaments=5,
    min_z=3.0,
):
    """Unselected classes that the filaments of the selected classes also use.

    A filament is taken to be of the selected type when at least ``seed_share``
    of its segments are in selected classes. An unselected class is suggested
    when the filaments that contain it are of the selected type much more often
    than filaments are in general (binomial z-score at the filament level). The
    selected classes only define which filaments count; nothing is re-fitted, so
    the suggestions cannot drift away from the user's selection, and the user
    decides which to accept.

    Parameters
    ----------
    params : pandas.DataFrame
        Class2D parameters with ``rlnMicrographName``, ``rlnHelicalTubeID`` and
        ``rlnClassNumber``.
    selected : sequence of int
        Selected class numbers.
    candidates : sequence of int, optional
        Class numbers to consider. Defaults to all unselected classes.
    min_segments : int, optional
        Filaments with fewer segments are ignored. Defaults to 5.
    seed_share : float, optional
        Share of a filament's segments that must be in selected classes for it
        to count as of the selected type. Defaults to 0.5.
    min_filaments : int, optional
        Fewest filaments a suggested class must occur on. Defaults to 5.
    min_z : float, optional
        Smallest z-score suggested. Defaults to 3.

    Returns
    -------
    list of dict
        ``candidate`` (class number), ``filaments``, ``share`` (of those
        filaments that are of the selected type), ``baseline`` (the same share
        over all filaments with an unselected segment) and ``z``, best first.
    """
    selected = {int(c) for c in selected}
    cls = params["rlnClassNumber"].astype(int).values
    fil = params.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False).ngroup()
    fil = fil.values
    nf = fil.max() + 1
    is_sel = np.isin(cls, list(selected))
    n_seg = np.bincount(fil, minlength=nf)
    n_sel = np.bincount(fil, weights=is_sel, minlength=nf)
    of_type = (n_seg >= min_segments) & (n_sel >= seed_share * n_seg)
    eligible = (n_seg >= min_segments) & (n_sel < n_seg)
    if eligible.sum() < 2 or of_type[eligible].sum() == 0:
        return []
    baseline = float(of_type[eligible].mean())
    if not 0.0 < baseline < 1.0:
        return []
    if candidates is None:
        candidates = np.unique(cls)
    candidates = {int(c) for c in candidates} - selected
    keep = ~is_sel & eligible[fil] & np.isin(cls, list(candidates))
    # one count per (filament, class): the unit of evidence is the filament
    pairs = np.unique(np.stack([fil[keep], cls[keep]], 1), axis=0)
    out = []
    for c in np.unique(pairs[:, 1]):
        f = pairs[pairs[:, 1] == c, 0]
        n = len(f)
        if n < min_filaments:
            continue
        k = int(of_type[f].sum())
        z = (k - n * baseline) / np.sqrt(n * baseline * (1.0 - baseline))
        if z >= min_z and k / n > baseline:
            out.append(
                dict(
                    candidate=int(c),
                    filaments=n,
                    share=k / n,
                    baseline=baseline,
                    z=float(z),
                )
            )
    return sorted(out, key=lambda d: d["z"], reverse=True)


@dataclass
class PhasePitchResult:
    """Everything :func:`analyze` measures, for display.

    ``filaments`` is a DataFrame of every filament long enough to be fitted on
    its own -- micrograph, tube ID, span, pair count and its own ``pitch`` (the pooled
    repeat where the fit found no peak, marked by ``fitted`` False) -- for selecting the filaments of one narrow
    pitch band.
    """

    period_raw: float
    period: float
    period_sd: float
    calibration: dict
    scan_periods: np.ndarray
    scan_scores: np.ndarray
    phases: np.ndarray
    class_ids: np.ndarray
    class_weight: np.ndarray
    heterogeneity: dict
    filaments: object = None
    support: float = 1.0
    double_ratio: float = 0.0
    class_fit: np.ndarray = None
    phase_gap: float = 0.0
    groups: list = field(default_factory=list)
    group_periods: list = field(default_factory=list)
    refined_path: bool = False
    n_filaments: int = 0
    n_pairs: int = 0
    segments: object = None
    filament_keys: list = field(default_factory=list)
    params_used: object = None
    merged_classes: list = field(default_factory=list)
    merged_into: dict = field(default_factory=dict)
    class_evidence: np.ndarray = None
    class_zdir: np.ndarray = None
    class_tilt: np.ndarray = None
    class_across: np.ndarray = None
    class_count: np.ndarray = None


def analyze(
    params,
    class_ids=None,
    n_boot=20,
    length_scale=800.0,
    progress=None,
    counterparts=None,
    image_apix=None,
    **kw,
):
    """Pitch, its uncertainty, class phases and per-filament spread in one call.

    ``progress``, if given, is called with a short message before each stage.

    ``counterparts`` (from :func:`pair_counterparts`, with class numbers in
    ``a`` and ``b``) and ``image_apix`` (the class averages' pixel size, A)
    merge each pair of classes that are each other's 180-degree rotation into
    one before the ring is fitted: the two directions in which filaments were
    picked then share their classes, which is what ties the azimuthal angles of
    the two sets together.
    """

    def step(msg):
        if progress is not None:
            progress(msg)

    used = params
    merged = []
    pairs = None
    if counterparts:
        step("merging classes that are each other's 180-degree rotation")

        def trial_of(sign):
            trial, gone = merge_counterparts(params, counterparts, image_apix, sign)
            keep = None
            if class_ids is not None:
                keep = [c for c in class_ids if int(c) not in set(gone)]
            try:
                tp = prepare_pairs(trial, class_ids=keep, **kw)
                score = float(
                    np.max(estimate_period(tp, length_scale=length_scale)["scores"])
                )
            except ValueError:
                return None
            return score, trial, keep, gone, tp

        # the two signs at once: the scans are numpy work that runs in parallel
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(2) as pool:
            trials = [t for t in pool.map(trial_of, (1.0, -1.0)) if t is not None]
        best = None
        for t in trials:  # in sign order, so a tie keeps +1 as before
            if best is None or t[0] > best[0]:
                best = t
        if best is not None:
            _, used, class_ids, merged, pairs = best
    if pairs is None:
        step("collecting segment pairs")
        # the winning merge trial already collected its pairs
        pairs = prepare_pairs(used, class_ids=class_ids, **kw)
    if len(pairs.D) == 0:
        raise ValueError("no same-polarity segment pairs in the selected filaments")
    step(f"scanning repeats up to {pairs.max_sep:.0f} \u00c5")
    est = estimate_period(pairs, length_scale=length_scale)
    step("syncing filament directions")
    est, flipped = sync_filament_directions(pairs, est, length_scale=length_scale)
    if flipped.any():
        # the sync refits near the period only; the full scan, harmonics and
        # all, is what the display and the half-repeat check should see
        est = estimate_period(pairs, length_scale=length_scale)
    step("calibrating on this dataset's segment geometry")
    cal = calibrate_period(pairs, est["period"], length_scale=length_scale)
    step("resampling filaments")
    boot = bootstrap_period(
        pairs, est["period"], n_boot=n_boot, length_scale=length_scale
    )
    sd = (
        float(np.std(boot) / max(cal["slope"], 1e-6)) if len(boot) > 1 else float("nan")
    )
    step("fitting individual filaments")
    het = filament_heterogeneity(pairs, cal["period"], est["phases"])
    # every filament with a pair is fitted, however short: a filament whose fit
    # finds no peak (typically one much shorter than a repeat, or with a very
    # small twist) has no information of its own and takes the pooled repeat
    fit = filament_periods(
        pairs, est["phases"], cal["period"], min_span=0.0, min_pairs=1
    )
    own = np.full(pairs.n_filaments, np.nan)
    own[fit["filament"]] = fit["period"]
    n_pairs = np.bincount(pairs.pair_filament, minlength=pairs.n_filaments)
    fitted = np.isfinite(own)
    # how much better the filament's direction fits than the opposite one:
    # near 0 for a filament whose direction is a coin flip (a few segments)
    pf = pairs.pair_filament
    ang = 2 * np.pi * pairs.D / est["period"]
    dphi = (
        est["phases"][pairs.seg_class[pairs.I]]
        - est["phases"][pairs.seg_class[pairs.J]]
    )
    keep_fit = np.bincount(
        pf, weights=pairs.W * np.cos(ang - dphi), minlength=pairs.n_filaments
    )
    swap_fit = np.bincount(
        pf, weights=pairs.W * np.cos(-ang - dphi), minlength=pairs.n_filaments
    )
    direction_score = (keep_fit - swap_fit) / (
        np.abs(keep_fit) + np.abs(swap_fit) + 1e-12
    )
    filaments = pd.DataFrame(
        dict(
            rlnMicrographName=[k[0] for k in pairs.filament_keys],
            rlnHelicalTubeID=[k[1] for k in pairs.filament_keys],
            span=pairs.filament_span,
            n_pairs=n_pairs,
            pitch=np.where(fitted, own, cal["period"]),
            fitted=fitted,
            direction_score=direction_score,
        )
    )
    step("grouping classes by the filaments they share")
    groups = class_groups(pairs)
    group_periods = []
    if len(groups) > 1:
        for g in groups:
            mask = np.isin(pairs.seg_class[pairs.I], g) & np.isin(
                pairs.seg_class[pairs.J], g
            )
            r = estimate_period(
                pairs, pair_weights=mask.astype(float), length_scale=length_scale
            )
            group_periods.append(r["period"])
    hist, _ = class_pair_histogram(pairs)
    weight = hist.sum((1, 2)) + hist.sum((0, 2))
    # the origin of the phases is arbitrary: put the most populated class at 0
    phases = np.angle(np.exp(1j * (est["phases"] - est["phases"][np.argmax(weight)])))
    return PhasePitchResult(
        period_raw=est["period"],
        period=cal["period"],
        period_sd=sd,
        calibration=cal,
        scan_periods=est["periods"],
        scan_scores=est["scores"],
        phases=phases,
        class_ids=pairs.class_ids,
        class_weight=weight,
        class_fit=class_fit(pairs, est["phases"], est["period"]),
        class_evidence=class_evidence(pairs, est["period"]),
        phase_gap=largest_phase_gap(est["phases"], weight),
        heterogeneity=het,
        filaments=filaments,
        support=est["support"],
        double_ratio=est["double_ratio"],
        groups=[pairs.class_ids[g] for g in groups],
        group_periods=group_periods,
        refined_path=pairs.refined_path,
        n_filaments=pairs.n_filaments,
        n_pairs=len(pairs.D),
        segments=pd.DataFrame(
            dict(
                row=pairs.seg_row,
                filament=pairs.seg_filament,
                direction=pairs.seg_pol * np.where(flipped[pairs.seg_filament], -1, 1),
                azimuth=phases[pairs.seg_class],
                small=pairs.seg_small,
                ref=pairs.seg_ref,
                weight=pairs.seg_weight,
            )
        ),
        filament_keys=pairs.filament_keys,
        params_used=used,
        merged_classes=[int(c) for c in merged],
        merged_into=_merged_into(counterparts, merged),
        class_zdir=(-pairs.class_axis + np.where(pairs.class_sign < 0, 180.0, 0.0)),
        class_tilt=pairs.class_tilt,
        class_across=pairs.class_across,
        class_count=pairs.class_count,
    )
