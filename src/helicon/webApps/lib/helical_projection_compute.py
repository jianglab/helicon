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
            return get_images_from_file(self.filename)
        if isinstance(self.url, str) and len(self.url):
            return get_images_from_url(self.url)
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
):
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
        length = int(pitch / new_apix + image_nx * length_xy_factor) // 2 * 2
        new_size = (length, image_ny, image_ny)
    else:
        new_apix = apix
        new_size = (nz, ny, nx)

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

    if mixture is not None:
        return map_info, _align_in_gaussian_space(
            queries,
            query_labels,
            query_fits,
            mixture,
            proj,
            new_apix,
            new_size,
            match_sf,
            label,
        )

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

    return map_info, _compose_result(
        scores, placed, best, query_labels, proj, new_apix, match_sf, label
    )


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


def _align_in_gaussian_space(
    queries,
    query_labels,
    query_fits,
    mixture,
    proj,
    new_apix,
    new_size,
    match_sf,
    label,
):
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
    """
    from . import gauss_align

    # The query is centred across the filament by the auto transform, so the
    # shift across it is small; along it a short average slides anywhere on a
    # projection a pitch long.
    half_y = max(4.0 * mixture.sigma, 0.25 * new_size[1] * new_apix)
    half_x = 0.5 * new_size[0] * new_apix

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

    return _compose_result(
        scores, placed, best, query_labels, proj, new_apix, match_sf, label
    )


def refine_placement_for_display(result, queries, scale_range, angle_range=0.0):
    """Re-place the queries on one map with the pixel aligner, for the picture.

    The gaussian route ranks as well as the pixel route and far faster, but the
    placement it settles on is the best one for mixtures rather than for
    pixels, and a user looking at a top match is looking at pixels. So the
    matches that get looked at are re-placed here, by the same
    ``align_images`` the volume route uses -- which also recovers the scale the
    gaussian search does not vary. The score is left alone: it came from the
    search, and a handful of maps rescored by a different measure could not be
    compared with the rest.

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
        The same entry with its placement, flip, scale and rotation replaced.
    """
    _, _, _, _, score, _, query_label, proj, label = result
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

    refined = _compose_result(
        scores, placed, best, [query_label], proj, 1.0, False, label
    )
    return refined[:4] + (score,) + refined[5:6] + (query_label, proj, label)


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
