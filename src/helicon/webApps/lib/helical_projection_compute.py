from __future__ import annotations

"""Compute functions for HelicalProjection tab."""


import os
import math
import pathlib
import re
import numpy as np

import helicon
from helicon import align_images


def extract_emdb_id(url: str):
    import re

    pattern = r"EMD-(\d+)"
    match = re.search(pattern, url)
    if match:
        return f"EMD-{match.group(1)}"
    return None


class MapInfo:
    def __init__(
        self,
        data=None,
        filename=None,
        url=None,
        emd_id=None,
        label="",
        apix=None,
        twist=None,
        rise=None,
        csym=1,
    ):
        non_nones = [p for p in [data, filename, url, emd_id] if p is not None]
        if len(non_nones) > 1:
            raise ValueError(
                "MapInfo(): only one of these parameters can be set: data, filename, url, emd_id"
            )
        elif len(non_nones) < 1:
            raise ValueError(
                "MapInfo(): one of these parameters must be set: data, filename, url, emd_id"
            )
        self.data = data
        self.filename = filename
        self.url = url
        self.emd_id = emd_id
        self.label = label
        self.apix = apix
        self.twist = twist
        self.rise = rise
        self.csym = csym

    def __repr__(self):
        return (
            f"MapInfo(label={self.label}, emd_id={self.emd_id}, "
            f"twist={self.twist}, rise={self.rise}, csym={self.csym}, "
            f"apix={self.apix})"
        )

    def get_data(self):
        """The map's voxels, loaded on demand and NOT kept afterwards.

        A loaded volume used to be stored back on the instance, so every map the
        session ever touched stayed in memory for as long as the list of maps
        did. Measured on six 384-cubed maps that was 1359 MB pinned after the
        work had finished, and it grows with every map examined -- which is what
        made this tab expensive to leave open rather than merely expensive to
        run.

        Reloading instead is cheap: a URL or EMDB entry comes back through the
        joblib cache, and a local file is a plain read. Data handed in at
        construction is a different matter and is kept, since there is nowhere
        to reload it from.
        """
        if self.data is not None:
            return self.data, self.apix
        if (
            isinstance(self.filename, str)
            and len(self.filename)
            and pathlib.Path(self.filename).exists()
        ):
            return get_map_from_file(self.filename)
        if isinstance(self.url, str) and len(self.url):
            return get_map_from_url(self.url)
        if isinstance(self.emd_id, str) and len(self.emd_id):
            emdb = helicon.dataset.EMDB()
            return emdb(self.emd_id)
        raise ValueError("MapInfo.get_data(): failed to obtain data")


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "helical_lab"), expires_after=7, verbose=0
)
def get_images_from_url(url: str):
    url_final = helicon.get_direct_url(url)
    fileobj = helicon.download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    data, apix = get_images_from_file(fileobj.name)
    return data, apix


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "helical_lab"), expires_after=7, verbose=0
)
def get_map_from_url(url: str):
    """Download a 3D map and read it with :func:`get_map_from_file`.

    Parameters
    ----------
    url : str
        Direct (or cloud-drive) download link of an MRC map.

    Returns
    -------
    tuple
        ``(data, apix)`` with ``data`` in (z, y, x) order.
    """
    url_final = helicon.get_direct_url(url)
    fileobj = helicon.download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    return get_map_from_file(fileobj.name)


def get_map_from_file(mapFile: str):
    """Read a 3D map as stored, in (z, y, x) order with its original values.

    Unlike :func:`get_images_from_file`, which is meant for stacks of 2D images
    and transposes and inverts each image when it is taller than wide, a map is
    returned unchanged, so a non-cubic map keeps its orientation and contrast.

    Parameters
    ----------
    mapFile : str
        Path to an MRC map.

    Returns
    -------
    tuple
        ``(data, apix)``.
    """
    import mrcfile

    with mrcfile.open(mapFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = np.array(mrc.data)
    return data, round(apix, 4)


def get_images_from_file(imageFile: str):
    import mrcfile

    with mrcfile.open(imageFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data

    if isinstance(data, np.ndarray):
        if len(data.shape) < 3:
            ny, nx = np.shape(data)
            data = np.expand_dims(data, axis=0)
        else:
            nz, ny, nx = np.shape(data)
        if nx < ny:
            data = np.array([np.max(img) - np.transpose(img) for img in data])

    return data, round(apix, 4)


def projection_workers(n_maps, per_map_bytes=1_000_000_000):
    """How many maps to symmetrise at once without running the machine out.

    Each map in flight holds its own volume, a filtered copy and the symmetrised
    result, which measured about 760 MB of peak for a 384-cubed input -- so a
    pool sized purely by CPU count asks for that many gigabytes at once, and on
    a 14-core machine that is more memory than most have to spare.

    The cap is therefore whichever is smaller: one worker per CPU, or as many as
    fit in half of what is free. Never fewer than one, and never more than there
    are maps to do. ``per_map_bytes`` is the measured figure above rounded up,
    not a guess at the particular map, which is not known until it is loaded.
    """
    workers = helicon.available_cpu()
    try:
        import psutil

        budget = psutil.virtual_memory().available * 0.5
        workers = min(workers, int(budget // per_map_bytes))
    except Exception:
        pass
    return max(1, min(workers, max(1, int(n_maps))))


def get_amyloid_n_sub_1_symmetry(twist: float, rise: float, max_n: int = 10) -> int:
    ret = 1
    for n in range(max_n, 1, -1):
        if not (4.5 < rise * n < 5):
            continue
        if abs(360 - abs(twist * n)) > 90:
            continue
        ret = n
        break
    return ret


def as_number(value, default=0.0) -> float:
    """A helical parameter as a number, whatever the table holds.

    The EMDB table is assembled from a deposited table and a curated one, and
    what survives a merge is not always a number: an empty cell, the string
    ``"nan"``, a value the curators left blank. Anything that will not convert
    -- and a NaN, which converts but compares false against everything -- comes
    back as the default.

    Parameters
    ----------
    value : object
        The cell's contents.
    default : float, optional
        What to return when there is no usable number. Defaults to 0.

    Returns
    -------
    float
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def as_csym(value, default=1) -> int:
    """A cyclic symmetry as an integer, from ``"C2"``, ``2``, ``"2"`` or junk.

    The column is usually ``"C<n>"``, but a merge that found no curated value
    writes ``"Cnan"``, and ``int("nan")`` raises -- which in a web app ends the
    session rather than the parse. Anything without digits in it is taken as
    unknown and returns the default.

    Parameters
    ----------
    value : object
        The cell's contents.
    default : int, optional
        What to return when there is no usable symmetry. Defaults to 1.

    Returns
    -------
    int
    """
    digits = re.sub(r"[^0-9]", "", str(value))
    if not digits:
        return default
    try:
        number = int(digits)
    except ValueError:
        return default
    return number if number > 0 else default


def axial_stretch_factors(range_percent=0.0, step_percent=5.0):
    """The stretch factors a twist search tries, evenly spaced in log.

    At low resolution, where the 4.75 A stacking of an amyloid is not resolved,
    a filament with the same packing but a different twist is the same
    filament stretched along its axis: with the density written as
    ``F(r, phi - omega z)``, multiplying the twist per Angstrom ``omega`` by
    ``s`` is scaling ``z`` by ``1 / s``. Its side projection is the side
    projection stretched by ``1 / s`` -- exactly, for an untilted view -- so a
    coarse twist search needs one projection per map and a stretch per factor.

    A factor ``k`` multiplies the crossover distance by ``k`` and divides the
    twist by ``k``. The factors run from ``1 / (1 + r)`` to ``1 + r``, so a
    range of 20% reaches a twist 20% larger and one 20% smaller in the same
    number of steps, and are evenly spaced in ``log k``: what a step of the
    stretch moves at the end of a projection is proportional to ``k``, so the
    same relative step costs the same everywhere. A step of ``d / L`` keeps
    that movement under the resolution ``d`` over a segment ``L`` long, e.g.
    5% for 20 A over 400 A.

    Parameters
    ----------
    range_percent : float, optional
        How far the crossover distance may move, in percent. 0 (the default)
        searches no stretch at all.
    step_percent : float, optional
        The relative step between factors, in percent. Defaults to 5.

    Returns
    -------
    tuple of float
        The factors in increasing order, always including 1.0 and as many on
        either side of it.
    """
    r = float(range_percent or 0.0) / 100.0
    s = float(step_percent or 0.0) / 100.0
    if r <= 0 or s <= 0:
        return (1.0,)
    half = math.log1p(r)
    n = max(1, int(math.ceil(half / math.log1p(s) - 1e-9)))
    return tuple(float(k) for k in np.exp(np.linspace(-half, half, 2 * n + 1)))


def stretched_twist(twist, rise, factor):
    """The twist of the filament stretched by ``factor``, at the same rise.

    Stretching along the axis divides the twist per Angstrom by ``factor``.
    For an amyloid that is a change of twist and not of rise, which stays at
    4.75 A; but the twist to change is the small one between neighbouring
    rungs, not the stored value. A pseudo-2_1 filament stores 179.4 degrees
    at 2.38 A, and it is the 1.2 degrees per 4.75 A rung -- the deviation from
    two units per turn -- that the stretch divides. The same holds for any
    ``n``-fold pseudo screw, found as the rest of helicon finds it
    (:func:`get_amyloid_n_sub_1_symmetry`).

    For a helix whose subunits are resolved at the search resolution the
    stretch is better read as the rise multiplied by ``factor`` at the same
    twist; the twist returned here is then the continuous-helix equivalent.

    Parameters
    ----------
    twist, rise : float
        The map's helical parameters, in degrees and Angstroms.
    factor : float
        The stretch along the axis; above 1 the crossover is longer.

    Returns
    -------
    float
        The twist in degrees, in the same convention as ``twist``.
    """
    n = get_amyloid_n_sub_1_symmetry(twist=twist, rise=rise)
    deviation = helicon.set_to_periodic_range(n * float(twist), min=-180, max=180)
    return float(twist) + deviation * (1.0 / float(factor) - 1.0) / n


def stretch_along_axis(image, factor, length):
    """A side projection stretched along the helical axis.

    The axis runs along x, as in every side projection here. The image is
    resampled about its centre, so ``out[:, j]`` is the input at
    ``(j - length // 2) / factor + nx // 2``, and the result is ``length``
    pixels wide; the part of it the input does not reach is zero. A
    compression is blurred along the axis first, by as much as the step
    between the samples grows, so the finer detail does not alias.

    Parameters
    ----------
    image : np.ndarray
        ``(ny, nx)`` projection, helical axis along x.
    factor : float
        The stretch: 2 doubles every distance along the axis.
    length : int
        Width of the output, in pixels.

    Returns
    -------
    np.ndarray
        ``(ny, length)`` float32.
    """
    from scipy.ndimage import gaussian_filter1d, map_coordinates

    image = np.asarray(image, dtype=np.float32)
    ny, nx = image.shape
    factor = float(factor)
    length = int(length)
    if factor == 1.0 and length == nx:
        return image.copy()
    if factor < 1.0:
        sigma = 0.5 * math.sqrt(1.0 / factor**2 - 1.0)
        if sigma > 0.1:
            image = gaussian_filter1d(image, sigma=sigma, axis=1, mode="constant")
    x = (np.arange(length, dtype=np.float64) - length // 2) / factor + nx // 2
    y = np.arange(ny, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    out = map_coordinates(image, [yy, xx], order=1, mode="constant", cval=0.0)
    return out.astype(np.float32)


def _stretch_mixture(mixture, factor, length, apix):
    """A projection mixture stretched along the axis and cut to a window.

    The centres move, the width does not: a stretched filament with the same
    packing has subunits of the same size further apart, which is also the
    assumption the image stretch makes at the resolution it is used at.
    """
    from dataclasses import replace

    centers = np.array(mixture.centers, dtype=np.float64)
    centers[:, 1] *= float(factor)
    reach = 0.5 * length * apix + 4.0 * float(mixture.sigma)
    inside = np.abs(centers[:, 1]) <= reach
    return replace(
        mixture,
        amplitudes=np.asarray(mixture.amplitudes)[inside],
        centers=centers[inside].astype(np.float32),
    )


def has_twist(map_info) -> bool:
    """Whether a map can be searched at all.

    A map with no twist, or a twist of zero, is not a helix: there is nothing
    to symmetrize along and no side projection to make. EMDB entries often
    carry no helical parameters -- and a filtered table can hand over hundreds
    of maps at once -- so callers use this to pass them by rather than fail.

    ``float(None)`` raises and ``float("nan")`` compares false, which is why
    this is a function rather than an inline comparison: in a web app an
    exception here ends the session.

    Parameters
    ----------
    map_info : MapInfo
        The map to test.

    Returns
    -------
    bool
        True when the twist is a number distinguishable from zero.
    """
    try:
        return abs(float(map_info.twist)) > 1e-3
    except (TypeError, ValueError):
        return False


@helicon.cache(expires_after=7, cache_dir=helicon.cache_dir / "helical_lab", verbose=0)
def get_one_map_xyz_projects(map_info, length_z, map_projection_xyz_choices):
    label = map_info.label
    try:
        data, apix = map_info.get_data()
    except Exception as e:
        # what went wrong, not which map: the caller knows which map it asked
        # about and says so, and repeating it reads as "EMD-38069: Failed to
        # download the map from EMDB for EMD-38069". A URL or a file name is
        # not the label, so those stay.
        if map_info.filename:
            msg = f"Failed to read the uploaded map {map_info.filename}"
        elif map_info.url:
            msg = f"Failed to download the map from {map_info.url}"
        else:
            msg = "Failed to download the map from EMDB"
        raise ValueError(msg) from e

    images = []
    image_labels = []
    if "z" in map_projection_xyz_choices:
        rise = map_info.rise
        if rise > 0:
            rise *= get_amyloid_n_sub_1_symmetry(
                twist=map_info.twist, rise=map_info.rise
            )
            images += [
                helicon.crop_center_z(
                    data, n=max(1, int(0.5 + length_z * rise / apix))
                ).sum(axis=0)
            ]
        else:
            images += [data.sum(axis=0)]
        image_labels += [label + ":Z"]
    if "y" in map_projection_xyz_choices:
        images += [data.sum(axis=1)]
        image_labels += [label + ":Y"]
    if "x" in map_projection_xyz_choices:
        images += [data.sum(axis=2)]
        image_labels += [label + ":X"]

    return images, image_labels


@helicon.cache(expires_after=7, cache_dir=helicon.cache_dir / "helical_lab", verbose=0)
def symmetrize_project_align_one_map(
    map_info,
    image_query,
    image_query_label,
    image_query_apix,
    rescale_apix,
    length_xy_factor,
    match_sf,
    angle_range,
    scale_range,
    projection_method="volume",
    query_fits=None,
    stretch_factors=(1.0,),
):
    """Project one map and match the query images to it.

    Parameters
    ----------
    map_info : MapInfo
        The map and its helical parameters.
    image_query : np.ndarray or list of np.ndarray
        One query image, or several to be scored jointly.
    image_query_label : str or list of str
        Their labels.
    image_query_apix : float
        Pixel size of the queries.
    rescale_apix : bool
        Make the projection at the queries' pixel size, a pitch long.
    length_xy_factor : float
        Extra projection length, in query widths.
    match_sf : bool
        Match the projection's structure factor to the placed queries.
    angle_range, scale_range : float
        Passed to ``align_images``.
    projection_method : {"volume", "gaussian"}, optional
        Symmetrise the voxels, or render the map's gaussian fit.
    query_fits : list of ImageGaussians, optional
        The queries as gaussians, to match by overlap in gaussian mode.
    stretch_factors : sequence of float, optional
        Stretches of the projection along the helical axis to try, as from
        :func:`axial_stretch_factors`: a coarse search for a filament of the
        same packing with another twist. The best is kept. Defaults to none.

    Returns
    -------
    tuple
        ``(map_info, result)``, where ``result`` is None when the map could
        not be searched, and otherwise ``(flip, scale, rotation, shift,
        score, placed_queries, query_label, projection, map_label,
        stretch_factor, stretched_twist)``.
    """
    if abs(map_info.twist) < 1e-3:
        return map_info, None

    # One query image or several. Several are scored jointly: each is aligned
    # against the same projection and the scores are averaged, which is what
    # asking "do these class averages come from this structure" means when the
    # averages are different views of one filament.
    if isinstance(image_query, np.ndarray) and image_query.ndim == 2:
        queries = [image_query]
    else:
        queries = list(image_query)
    if isinstance(image_query_label, str):
        query_labels = [image_query_label] * len(queries)
    else:
        query_labels = list(image_query_label)
        if len(query_labels) < len(queries):
            query_labels += [""] * (len(queries) - len(query_labels))

    twist = map_info.twist
    rise = map_info.rise
    csym = map_info.csym
    label = map_info.label
    factors = tuple(float(k) for k in (stretch_factors or (1.0,)) if float(k) > 0)
    factors = factors or (1.0,)

    # The gaussian route never reads the volume: the fit is cached, and the
    # projection is rendered from it. Loading and filtering a map only to
    # discard it is most of what this function used to cost, so the load is
    # deferred rather than done up front. Without rescaling the output box is
    # the map's own box, which is one thing only the map can tell us.
    needs_volume = projection_method != "gaussian" or not rescale_apix
    data = None
    if needs_volume:
        try:
            data, apix = map_info.get_data()
        except Exception:
            return map_info, None
        nz, ny, nx = data.shape

    if rescale_apix:
        image_ny = max(q.shape[0] for q in queries)
        image_nx = max(q.shape[1] for q in queries)
        new_apix = image_query_apix
        twist_work = helicon.set_to_periodic_range(twist, min=-180, max=180)
        if abs(twist_work) < 90:
            pitch = 360 / abs(twist_work) * rise
        elif abs(twist_work) < 180:
            pitch = 360 / (180 - abs(twist_work)) * rise
        else:
            pitch = image_nx * new_apix
        # A pitch and a query long, so a query can slide over every part of a
        # period; stretched by k the pitch is k times as long, and so is the
        # projection it is compared with.
        lengths = [
            int(pitch * k / new_apix + image_nx * length_xy_factor) // 2 * 2
            for k in factors
        ]
    else:
        new_apix = apix
        lengths = [nz] * len(factors)
    # The projection every stretch is cut from: long enough that the most
    # compressed one still fills its window. With no stretch it is the length
    # the search has always used.
    base_length = max(int(math.ceil(n / k - 1e-6)) for n, k in zip(lengths, factors))
    if rescale_apix:
        new_size = (base_length, image_ny, image_ny)
    else:
        new_size = (base_length, ny, nx)

    if projection_method == "gaussian":
        # The map as a few hundred gaussians, fitted once and cached, so the
        # screw operation acts on parameters instead of on voxels and no volume
        # is ever built. The picture it renders is close to the one the volume
        # path produces -- ncc 0.98, 0.97 and 0.80 on EMD-46496, 12268 and
        # 71835 -- and everything downstream of `proj` is untouched.
        #
        # How well it ranks depends on what the query is, and the answer
        # moved once that was measured honestly. Against queries cut from the
        # volume route's own projections it looked clearly worse -- 60% top-1
        # against 73% over 60 maps -- but that benchmark rewards whatever
        # resembles a volume projection, and a fit is not one. Against REAL
        # class averages, all 42 of EMPIAR-10940 searched over 61 maps with
        # EMD-14046 as the truth, it ranks the true map first 35 times
        # against the volume route's 33, and far better on average -- mean
        # rank 4.4 against 7.2. That is why this route is the default.

        from . import map_gauss_fit

        mixture = None
        try:
            fit = map_gauss_fit.gaussians_for_map_info(map_info)
        except Exception:
            return map_info, None
        proj = map_gauss_fit.side_projection(
            fit,
            twist=twist,
            rise=rise,
            csym=csym,
            length=new_size[0],
            ny=new_size[1],
            apix=new_apix,
        )
        if query_fits is not None:
            # The projection as gaussians rather than as pixels, so the match
            # can be an integral instead of a cross-correlation. Projecting an
            # isotropic gaussian gives an isotropic gaussian, so this costs a
            # screw expansion and a binning pass and no rendering at all.
            try:
                mixture = map_gauss_fit.projection_mixture(
                    fit,
                    twist=twist,
                    rise=rise,
                    csym=csym,
                    length=new_size[0],
                    ny=new_size[1],
                    apix=new_apix,
                    sigma=query_fits[0].sigma,
                    # Coarse, because the match does not need the components
                    # to be finer than the width they are compared at, and
                    # every one of them costs in the overlap. Measured over 42
                    # real class averages against 61 maps, binning at twice
                    # the width ranks the true map first 34 times against 35
                    # at a quarter of it, with a better mean rank (4.2 against
                    # 4.3) and a thousand components instead of five and a
                    # half thousand -- 2.8 s a query against 10.2. Each bin
                    # keeps its amplitude-weighted centre rather than the grid
                    # point, which is why coarse bins cost so little.
                    bin_scale=2.0,
                )
            except Exception:
                mixture = None
    else:
        mixture = None
        # Referenced to its solvent first, as the gaussian fit is: the
        # symmetrisation finds the structure along z from slice sums, which
        # only means something when zero is solvent. On EMD-19855, whose
        # solvent sits below zero, this route scored 0.000.
        data = (data - helicon.helical_background(data).mean).astype(np.float32)
        if rescale_apix:
            data_work = helicon.low_high_pass_filter(
                data, low_pass_fraction=apix / new_apix
            )
        else:
            data_work = data

        fraction = 5 * rise / (nz * apix)

        data_sym = helicon.apply_helical_symmetry(
            data=data_work,
            apix=apix,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=csym,
            fraction=fraction,
            new_size=new_size,
            new_apix=new_apix,
            # One thread here, because the caller runs these maps in a pool and
            # the parallelism belongs at one level or the other, not both.
            # Measured, it costs nothing: a single map takes 2.5 s on one
            # thread and 2.6 s on fourteen, the kernel being bound by memory
            # traffic rather than by arithmetic. What asking for more did cost
            # was correctness -- apply_helical_symmetry sets numba's thread
            # count GLOBALLY, so every worker was rewriting a setting the
            # others were using.
            cpu=1,
        )
        proj = data_sym.sum(axis=2).T

    # One projection, stretched along the axis once per factor: the same
    # packing at another twist (see axial_stretch_factors). Every stretch is
    # matched the way an unstretched projection is, and the one the queries
    # agree with best -- by the mean score, as across maps -- is the result.
    chosen = None
    for factor, length in zip(factors, lengths):
        if factor == 1.0 and length == proj.shape[1]:
            proj_k = proj
        else:
            proj_k = stretch_along_axis(proj, factor, length)
        if mixture is not None:
            mixture_k = mixture
            if factor != 1.0 or length != new_size[0]:
                mixture_k = _stretch_mixture(mixture, factor, length, new_apix)
            scores, placed, best = _gaussian_matches(
                queries, query_fits, mixture_k, proj_k, new_apix
            )
        else:
            scores, placed, best = _pixel_matches(
                queries, proj_k, scale_range, angle_range
            )
        mean_score = float(np.mean(scores))
        if chosen is None or mean_score > chosen[0]:
            chosen = (mean_score, factor, scores, placed, best, proj_k)

    _, factor, scores, placed, best, proj_k = chosen
    result = _compose_result(
        scores, placed, best, query_labels, proj_k, new_apix, match_sf, label
    )
    return map_info, result + (factor, stretched_twist(twist, rise, factor))


def _pixel_matches(queries, proj, scale_range, angle_range):
    """Align every query to a projection image with ``align_images``.

    Returns
    -------
    tuple
        ``(scores, placed, best)``: a score and a placed image per query, and
        ``(score, flip, scale, rotation, shift)`` of the best of them.
    """
    scores = []
    placed = []
    best = None
    for one_query in queries:
        (
            one_flip,
            one_scale,
            one_rotation,
            one_shift,
            one_score,
            one_aligned,
        ) = align_images(
            image_moving=one_query,
            image_ref=proj,
            scale_range=scale_range,
            angle_range=angle_range,
            check_polarity=True,
            check_flip=True,
            return_aligned_moving_image=True,
        )
        scores.append(one_score)
        placed.append(one_aligned)
        if best is None or one_score > best[0]:
            best = (one_score, one_flip, one_scale, one_rotation, one_shift)
    return scores, placed, best


def _compose_result(
    scores, placed, best, query_labels, proj, new_apix, match_sf, label
):
    """One result from however many queries were matched against one map.

    Each query comes back already placed in the projection's own container --
    both aligners return it that way -- so the composite is those placements
    laid over one another.

    At each pixel the contribution of largest magnitude wins. Not the sum,
    which reads as brighter density where two averages overlap; and not the
    maximum, which drops density wherever one image is negative and another
    contributes the zero of its own padding -- on a map whose projection is
    negative throughout, and EMD-1427 is one, that erases whole placements.
    """
    similarity_score = float(np.mean(scores))
    if len(placed) == 1:
        aligned_image_moving = placed[0]
    else:
        stack = np.stack(placed)
        strongest = np.argmax(np.abs(stack), axis=0)
        aligned_image_moving = np.take_along_axis(stack, strongest[np.newaxis], axis=0)[
            0
        ]
    _, flip, scale, rotation_angle, shift_cartesian = best
    image_query_label = (
        query_labels[0] if len(scores) == 1 else "%d images" % len(scores)
    )

    if match_sf:
        mask = aligned_image_moving > 0
        proj = helicon.match_structural_factors(
            data=proj,
            apix=new_apix,
            data_target=aligned_image_moving,
            apix_target=new_apix,
            mask=mask,
        )

    return (
        flip,
        scale,
        rotation_angle,
        shift_cartesian,
        similarity_score,
        aligned_image_moving,
        image_query_label,
        proj,
        label,
    )


def _gaussian_matches(queries, query_fits, mixture, proj, new_apix):
    """Match every query to one map by the overlap of their gaussians.

    The pixel route pads each query into the projection's box and correlates
    the two images over shift, polarity and flip. This does the same search
    with no image in it: the overlap of two mixtures has a closed form, and as
    a function of the shift it is itself a sum of gaussians, so the whole
    landscape comes from one scatter and one blur. Over 61 maps and 16 real
    class averages it ranks the true map first as often as the pixel route
    does -- 12 of 16, agreeing query by query -- in a sixth of the time.

    The picture is still made of pixels: the transform the mixtures agree on is
    applied to the original image, so the display keeps the detail the fit
    discarded.

    Returns
    -------
    tuple
        ``(scores, placed, best)`` as :func:`_pixel_matches` returns them.
    """
    from . import gauss_align

    # The query is centred across the filament by the auto transform, so the
    # shift across it is small; along it a short average slides anywhere on a
    # projection a pitch long.
    half_y = max(4.0 * mixture.sigma, 0.25 * proj.shape[0] * new_apix)
    half_x = 0.5 * proj.shape[1] * new_apix

    scores = []
    placed = []
    best = None
    for one_query, one_fit in zip(queries, query_fits):
        alignment = gauss_align.align_mixtures(
            one_fit.amplitudes,
            one_fit.centers,
            one_fit.sigma,
            mixture.amplitudes,
            mixture.centers,
            mixture.sigma,
            half_y=half_y,
            half_x=half_x,
            step=max(0.5 * mixture.sigma, new_apix),
        )
        scores.append(alignment.score)
        placed.append(
            gauss_align.place_query(one_query, alignment, proj.shape, new_apix)
        )
        entry = (
            alignment.score,
            alignment.flip < 0,
            alignment.scale,
            180.0 if alignment.polarity < 0 else 0.0,
            (alignment.shift[0] / new_apix, alignment.shift[1] / new_apix),
        )
        if best is None or entry[0] > best[0]:
            best = entry
    return scores, placed, best


def placement_agreement(placed, proj) -> float:
    """How well a placed query sits on a projection: NCC over its footprint.

    The measure a user applies by eye -- does the class average line up with
    the projection under it -- and, importantly, one that neither aligner
    optimises, so it can arbitrate between them. ``align_images`` scores its
    own placements higher than the gaussian route's, which is how the pixel
    aligner came to be trusted for the display in the first place; judged by
    this instead, it is sometimes far worse.

    Parameters
    ----------
    placed : np.ndarray
        The query already placed in the projection's frame, zero outside it.
    proj : np.ndarray
        The projection, of the same shape.

    Returns
    -------
    float
        Normalised cross-correlation over the non-zero pixels of ``placed``,
        or -1.0 when there is nothing to compare.
    """
    mask = placed != 0
    if mask.sum() < 2:
        return -1.0
    a = placed[mask].astype(np.float64)
    b = proj[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else -1.0


def refine_placement_for_display(result, queries, scale_range, angle_range=0.0):
    """Offer the pixel aligner's placement for the picture, and keep the better.

    The gaussian route ranks as well as the pixel route and far faster, but it
    never varies scale, and on the default query against EMD-14046 the pixel
    aligner's placement fits visibly better: NCC 0.778 over the query's
    footprint against 0.676. So the matches that get looked at are re-placed
    by the same ``align_images`` the volume route uses.

    But not unconditionally. On a class average against EMD-60539 the pixel
    aligner lands on a poor optimum -- NCC 0.740 where the gaussian placement
    has 0.907 -- and the display showed the query visibly off the crossover
    it belongs on. Neither placement is reliably better, so both are judged by
    :func:`placement_agreement`, which neither aligner optimises, and the
    better one is shown. The comparison is of the composite when several
    images were searched together.

    The score is left alone either way: it came from the search, and a
    handful of maps rescored by a different measure could not be compared
    with the rest.

    Parameters
    ----------
    result : tuple
        One entry as returned by :func:`symmetrize_project_align_one_map`.
    queries : list of np.ndarray
        The query images, in the order they were searched with.
    scale_range, angle_range : float
        Passed to ``align_images``.

    Returns
    -------
    tuple
        The entry with the better placement, and the flip, scale and rotation
        that go with it.
    """
    _, _, _, _, score, current, query_label, proj, label = result[:9]
    scores, placed, best = _pixel_matches(queries, proj, scale_range, angle_range)

    refined = _compose_result(
        scores, placed, best, [query_label], proj, 1.0, False, label
    )
    if placement_agreement(refined[5], proj) <= placement_agreement(current, proj):
        return result
    # the stretch the search chose, if any, belongs to the projection, which
    # is unchanged
    return (
        refined[:4]
        + (score,)
        + refined[5:6]
        + (query_label, proj, label)
        + tuple(result[9:])
    )


def anisotropic_low_high_pass_filter(
    data: np.ndarray,
    low_pass_fraction_x: float = 0,
    high_pass_fraction_x: float = 0,
    ratio: float = 1,
):
    if data.ndim not in [2]:
        raise ValueError("Input data must be a 2D array.")

    fft = np.fft.fft2(data)
    ny, nx = fft.shape
    Y, X = np.meshgrid(
        np.arange(ny, dtype=np.float32) - ny // 2,
        np.arange(nx, dtype=np.float32) - nx // 2,
        indexing="ij",
    )
    Y /= ny // 2
    X /= nx // 2
    R2 = X**2 + (Y * ratio) ** 2

    if 0 < low_pass_fraction_x < 1:
        f2 = np.log(2) / (low_pass_fraction_x**2)
        filter_lp = np.exp(-f2 * R2)
        fft *= np.fft.fftshift(filter_lp)
    if 0 < high_pass_fraction_x < 1:
        f2 = np.log(2) / (high_pass_fraction_x**2)
        filter_hp = 1.0 - np.exp(-f2 * R2)
        fft *= np.fft.fftshift(filter_hp)
    ret = np.real(np.fft.ifftn(fft))
    return ret
