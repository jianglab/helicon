from __future__ import annotations

"""Compute functions for HelicalProjection tab."""


import os
import pathlib
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


@helicon.cache(expires_after=7, cache_dir=helicon.cache_dir / "helical_lab", verbose=0)
def get_one_map_xyz_projects(map_info, length_z, map_projection_xyz_choices):
    label = map_info.label
    try:
        data, apix = map_info.get_data()
    except Exception as e:
        if map_info.filename:
            msg = f"Failed to obtain uploaded map {label}"
        elif map_info.url:
            msg = f"Failed to download the map from {map_info.url}"
        elif map_info.emd_id:
            msg = f"Failed to download the map from EMDB for {map_info.emd_id}"
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
        # is ever built. The picture it renders is the one the volume path
        # produces -- ncc 0.98, 0.97 and 0.80 on EMD-46496, 12268 and 71835 --
        # and everything downstream of `proj` is untouched.
        from . import map_gauss_fit

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
    else:
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

    # align_images pads the moving image into the reference frame, so every
    # query comes back already placed in the projection's own container; the
    # composite is those placements laid over one another.
    #
    # At each pixel the contribution of largest magnitude wins. Not the sum,
    # which reads as brighter density where two averages overlap; and not the
    # maximum, which drops density wherever one image is negative and another
    # contributes the zero of its own padding -- on a map whose projection is
    # negative throughout, and EMD-1427 is one, that erases whole placements.
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
        query_labels[0] if len(queries) == 1 else "%d images" % len(queries)
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

    return map_info, (
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
