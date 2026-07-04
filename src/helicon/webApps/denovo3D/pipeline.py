import logging
from pathlib import Path
import numpy as np
import helicon

logger = logging.getLogger(__name__)


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "denovo3D"), expires_after=7, verbose=0
)  # 7 days
def get_images_from_url(url):
    url_final = helicon.get_direct_url(
        url
    )  # convert cloud drive indirect url to direct url
    fileobj = helicon.download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    data, apix = get_images_from_file(fileobj.name)
    return data, apix


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "denovo3D"), expires_after=7, verbose=0
)  # 7 days
def get_images_from_emdb(emdb_id):
    emdb = helicon.dataset.EMDB()
    data, apix = emdb(emdb_id)
    if data is None:
        raise IOError(f"ERROR: failed to download {emdb_id} from EMDB")

    return data, round(apix, 4)


def get_images_from_file(imageFile):
    import mrcfile

    with mrcfile.open(imageFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, round(apix, 4)


from .solver_linear_regression import lsq_reconstruct
from .utils import (
    generate_xyz_projections,
    symmetrize_transform_map,
    is_vertical,
    auto_horizontalize,
    tilt_psi_dy_str,
)


def process_one_task(
    ti,
    ntasks,
    data,
    imageFile,
    imageIndex,
    twist,
    rise,
    rise_range,
    csym,
    tilt,
    tilt_range,
    psi,
    psi_range,
    dy,
    dy_range,
    apix2d_orig,
    denoise,
    low_pass,
    transpose,
    horizontalize,
    target_apix3d,
    target_apix2d,
    thresh_fraction,
    positive_constraint,
    tube_length,
    tube_diameter,
    tube_diameter_inner,
    reconstruct_length,
    sym_oversample,
    interpolation,
    fsc_test,
    return_3d,
    score_metric,
    algorithm,
    verbose,
    n_cpu=1,
):
    """Process a single (image, helical parameters) combination.

    Prepares the input image, sets up reconstruction geometry, runs the
    least-squares reconstruction, applies helical symmetry, and computes
    projection scores.

    Parameters
    ----------
    ti : int
        Task index.
    ntasks : int
        Total number of tasks.
    data : numpy.ndarray
        Input image data.
    imageFile : str
        Image file name.
    imageIndex : int
        Image index.
    twist : float
        Helical twist in degrees (unused for LSQ).
    rise : float
        Helical rise in Angstroms.
    rise_range : tuple
        Range of rise values.
    csym : int
        Cyclic symmetry.
    tilt : float
        Tilt in degrees (unused for LSQ).
    tilt_range : tuple
        Range of tilt values.
    psi : float
        In-plane rotation in degrees.
    dy : float
        Perpendicular shift in Angstroms.
    apix2d_orig : float
        Original pixel size in Angstroms.
    denoise : str
        Denoising method (empty string = disabled).
    low_pass : float
        Low-pass filter resolution in Angstroms.
    transpose : int
        Transpose mode (-1=auto, 0=no, 1=yes).
    horizontalize : int
        Whether to auto-horizontalize.
    target_apix3d : float
        Target 3D pixel size.
    target_apix2d : float
        Target 2D pixel size.
    thresh_fraction : float
        Threshold fraction.

    Returns
    -------
    tuple
        (score, return_data, metadata)
    """

    def prepare_data(
        data, imageFile, imageIndex, denoise, low_pass, transpose, horizontalize, apix
    ):
        if low_pass > 2 * apix:
            data = helicon.low_high_pass_filter(
                data,
                low_pass_fraction=2 * apix / low_pass,
                high_pass_fraction=2.0 / np.max(data.shape),
            )
        if denoise:
            if denoise == "nl_mean":
                from skimage.restoration import denoise_nl_means

                data = denoise_nl_means(data)
            elif denoise == "tv":
                from skimage.restoration import denoise_tv_chambolle

                data = denoise_tv_chambolle(data)
            elif denoise == "wavelet":
                from skimage.restoration import denoise_wavelet

                data = denoise_wavelet(data)
        if transpose > 0 or (transpose < 0 and is_vertical(data)):
            data = data.T
        if horizontalize:
            data, theta_best, shift_best = auto_horizontalize(data, refine=True)
            logger.debug(
                f"Image {imageFile}-{imageIndex}: rotation={round(theta_best, 2)}° shift={round(shift_best*apix, 1)}Å"
            )
        return data

    if data is None:
        data = helicon.read_image_2d(imageFile, imageIndex - 1)

    if not np.std(data):  # images with const (0) pixel values
        logger.warning(
            f"WARNING: the input image {imageFile}:{imageIndex} is a blank image"
        )
        return None

    data = prepare_data(
        data,
        imageFile,
        imageIndex,
        denoise,
        low_pass,
        transpose,
        horizontalize,
        apix2d_orig,
    )
    ny, nx = data.shape
    ny_orig, nx_orig = ny, nx

    if tube_diameter < 0:
        rotation, shift_y, diameter = helicon.estimate_helix_rotation_center_diameter(
            data
        )
        tube_diameter = int(min(ny, diameter) * apix2d_orig * 2.5)
        logger.debug(
            f"Image {imageFile}-{imageIndex}: estimated tube diameter={tube_diameter}Å"
        )

    if tube_length < 0:
        if tube_diameter > ny * apix2d_orig / 2:
            tube_length = int(nx * apix2d_orig)
        else:
            tube_length = round(
                np.sqrt((nx * apix2d_orig) ** 2 / 4 - tube_diameter**2 / 4) * 2
            )
        logger.debug(
            f"Image {imageFile}-{imageIndex}: estimated tube length={tube_length}Å"
        )

    reconstruct_diameter = (
        tube_diameter if 0 < tube_diameter < ny * apix2d_orig else ny * apix2d_orig
    )
    reconstruct_diameter_inner = (
        tube_diameter_inner if 0 < tube_diameter_inner < reconstruct_diameter else 0
    )
    if reconstruct_length < rise:
        reconstruct_length = max(
            min(3 * np.max(rise_range), tube_length),
            round(np.tan(np.deg2rad(np.max(np.abs(tilt_range)))) * tube_diameter * 3),
        )
        logger.debug(
            f"Image {imageFile}-{imageIndex}: reconstruct_length set to {reconstruct_length}Å"
        )

    if target_apix2d < apix2d_orig:
        target_apix2d = apix2d_orig
    logger.debug(
        f"Image {imageFile}-{imageIndex}: target_apix2d set to {target_apix2d}"
    )

    data = helicon.down_scale(data, target_apix2d, apix2d_orig)
    ny, nx = data.shape

    if thresh_fraction >= 0:
        data_orig = data
        nr = min(
            ny // 2 - 1, int(np.ceil(reconstruct_diameter / 2 / target_apix2d) + 1)
        )
        data -= np.median(data[(ny // 2 - nr, ny // 2 + nr), :])  # set background to 0
        data = helicon.threshold_data(data, thresh_fraction=thresh_fraction)
        data /= np.max(data)
    else:
        data_orig = data

    if target_apix3d < 0:
        vol = (
            reconstruct_length
            * (reconstruct_diameter**2 - reconstruct_diameter_inner**2)
            / 4
            * np.pi
        )
        target_apix3d = max(
            target_apix2d, round(np.power(vol / (nx * ny), 1 / 3) + 0.5)
        )
    elif target_apix3d == 0:
        target_apix3d = target_apix2d
    logger.debug(
        f"Image {imageFile}-{imageIndex}: target_apix3d set to {target_apix3d}"
    )

    csym_to_enforce = csym
    thresh_fraction = thresh_fraction
    reconstruct_diameter_3d_pixel = int(round(reconstruct_diameter / target_apix3d))
    reconstruct_diameter_3d_pixel += reconstruct_diameter_3d_pixel % 2
    reconstruct_diameter_3d_inner_pixel = int(
        round(tube_diameter_inner / target_apix3d)
    )
    reconstruct_diameter_2d_pixel = int(round(reconstruct_diameter / target_apix2d))
    reconstruct_diameter_2d_pixel += reconstruct_diameter_2d_pixel % 2
    reconstruct_length_2d = (
        tube_length if 0 < tube_length < nx * target_apix2d else nx * target_apix2d
    )
    reconstruct_length_2d_pixel = int(reconstruct_length_2d / target_apix2d)
    reconstruct_length_2d_pixel += reconstruct_length_2d_pixel % 2
    pitch = round(rise * 360 / abs(twist), 1)
    if reconstruct_length > 0:
        reconstruct_length_3d_pixel = max(
            int(np.ceil(rise / target_apix3d)),
            int(np.ceil(reconstruct_length / target_apix3d)),
        )
        reconstruct_length_3d_pixel += reconstruct_length_3d_pixel % 2
    else:
        reconstruct_length_3d_pixel = int(
            reconstruct_length_2d_pixel * target_apix2d / target_apix3d + 0.5
        )
        reconstruct_length_3d_pixel += reconstruct_length_3d_pixel % 2
        logger.debug(
            f"Image {imageFile}-{imageIndex}: reconstruct_length set to {reconstruct_length_3d_pixel}pixels ({round(reconstruct_length_3d_pixel*target_apix3d)}Å)"
        )

    if sym_oversample <= 0:
        n_voxels = reconstruct_length_3d_pixel * (
            reconstruct_diameter_3d_pixel**2 - reconstruct_diameter_3d_inner_pixel**2
        )
        ratio = 2**20 / n_voxels
        if ratio < 10:
            sym_oversample = max(1, int(round(ratio)))
        elif ratio < 100:
            sym_oversample = max(1, int(round(ratio / 10)) * 10)
        else:
            sym_oversample = max(1, int(round(ratio / 100)) * 100)
        if return_3d:
            sym_oversample *= 2
        logger.debug(
            f"Image {imageFile}-{imageIndex}: sym_oversample set to {sym_oversample}"
        )

    solve_fn = lsq_reconstruct
    label = "lsq_reconstruct"
    with helicon.Timer(
        f"{label}: {round(pitch, 1)}Å/twist={round(twist, 3)}° rise={round(rise, 3)}Å",
        verbose=verbose > 10,
    ):
        # Build refinement range dict for LSQ solver
        refine_range = None
        if algorithm.get("model", "lsq") in ("lsq", "elasticnet", "lasso", "ridge"):
            r_dict = {}
            if tilt_range[1] > tilt_range[0]:
                r_dict["tilt"] = max(abs(tilt_range[0]), abs(tilt_range[1]))
            if psi_range > 0:
                r_dict["psi"] = psi_range
            if dy_range > 0:
                r_dict["dy"] = dy_range
            if r_dict:
                refine_range = r_dict

        solve_kwargs = dict(
            projection_image=data,
            scale2d_to_3d=target_apix2d / target_apix3d,
            twist_degree=twist,
            rise_pixel=rise / target_apix3d,
            csym=csym_to_enforce,
            tilt_degree=tilt,
            psi_degree=psi,
            dy_pixel=dy / target_apix2d,
            thresh_fraction=thresh_fraction,
            positive_constraint=positive_constraint,
            reconstruct_diameter_3d_inner_pixel=reconstruct_diameter_3d_inner_pixel,
            reconstruct_diameter_2d_pixel=reconstruct_diameter_2d_pixel,
            reconstruct_diameter_3d_pixel=reconstruct_diameter_3d_pixel,
            reconstruct_length_2d_pixel=reconstruct_length_2d_pixel,
            reconstruct_length_3d_pixel=reconstruct_length_3d_pixel,
            sym_oversample=sym_oversample,
            interpolation=interpolation,
            fsc_test=fsc_test,
            score_metric=score_metric,
            target_apix2d=target_apix2d,
            verbose=verbose,
            algorithm=algorithm,
            refine_tilt_psi_dy_range=refine_range,
        )
        if algorithm.get("model", "lsq") in (
            "lsq",
            "elasticnet",
            "lasso",
            "ridge",
            "ard",
            "lreg",
        ):
            solve_kwargs["cpu"] = n_cpu
        (rec3d, rec3d_set_1, rec3d_set_2), score = solve_fn(**solve_kwargs)
    with helicon.Timer("apply_helical_symmetry", verbose=verbose > 10):
        twist_degree = twist if abs(twist) < 90 else 180 - abs(twist)
        if abs(twist_degree) > 1e-2:
            pitch_pixel = int(360 / abs(twist_degree) * rise / apix2d_orig + 0.5)
        else:
            pitch_pixel = int(np.ceil(2 * rise / apix2d_orig))
        new_length = max(nx_orig, int(pitch_pixel * 1.2))
        cpu = helicon.available_cpu()
        rec3d_xform = helicon.apply_helical_symmetry(
            data=rec3d,
            apix=target_apix3d,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=csym,
            new_size=(
                new_length,
                ny_orig,
                ny_orig,
            ),
            new_apix=apix2d_orig,
            cpu=cpu,
        )
    # Use refined tilt/psi/dy if available from local refinement
    tilt_viz = tilt
    psi_viz = psi
    dy_viz = dy
    if hasattr(lsq_reconstruct, "_refined_params") and lsq_reconstruct._refined_params:
        rp = lsq_reconstruct._refined_params
        tilt_viz = rp.get("tilt", tilt)
        psi_viz = rp.get("psi", psi)
        dy_viz = rp.get("dy", dy)
        lsq_reconstruct._refined_params = {}  # consume once

    rec3d_xform_2 = helicon.transform_map(
        rec3d_xform, scale=1.0, tilt=tilt_viz, psi=psi_viz, dy=dy_viz / apix2d_orig
    )
    rec3d_x_proj = np.sum(rec3d_xform_2, axis=2).T
    rec3d_y_proj = np.sum(rec3d_xform_2, axis=1).T
    rec3d_y_proj_max = rec3d_y_proj.max()
    if rec3d_y_proj_max > 0:
        rec3d_y_proj *= rec3d_x_proj.max() / rec3d_y_proj_max

    nz_per_rise = max(1, int(np.ceil(rise / apix2d_orig)))
    z0 = rec3d_xform.shape[0] // 2 - nz_per_rise // 2
    z1 = z0 + nz_per_rise
    rec3d_z_sections = np.sum(rec3d_xform[z0:z1, :, :], axis=0)
    vmin = rec3d_z_sections.min()
    vmax = rec3d_z_sections.max()
    if vmax > vmin:
        vmin_target = rec3d_x_proj.min()
        vmax_target = rec3d_x_proj.max()
        rec3d_z_sections = (rec3d_z_sections - vmin) * (vmax_target - vmin_target) / (
            vmax - vmin
        ) + vmin_target

    nz, ny, nx = rec3d.shape
    if target_apix2d != apix2d_orig:
        apix_tag = f"apix={apix2d_orig}->{target_apix2d}->{apix2d_orig}Å"
    else:
        apix_tag = f"apix={target_apix2d}Å"
    logger.info(
        f"Task {ti+1}/{ntasks}: {imageFile}-{imageIndex}:\t{apix_tag}\t{data.shape[-1]}x{data.shape[0]}pixels\tpitch={round(pitch, 1)}Å/twist={round(twist, 3)}° rise={round(rise, 3)}Å {csym=}{tilt_psi_dy_str(tilt, psi, dy)} => reconstruction size={nx}x{ny}x{nz}voxels voxelsize={round(target_apix3d, 3)}Å length={round(nz*target_apix3d, 1)}Å/{round(nz*target_apix3d/pitch, 2)}pitch/{round(nz*target_apix3d/rise, 2)}rise\t=>\tscore={round(score, 6)}"
    )

    return_data = (
        rec3d_x_proj,
        rec3d_y_proj,
        rec3d_z_sections,
        (rec3d, rec3d_set_1, rec3d_set_2) if return_3d else None,
        reconstruct_diameter_2d_pixel,
        reconstruct_diameter_3d_pixel,
        reconstruct_length_2d_pixel,
        reconstruct_length_3d_pixel,
    )
    result = (
        score,
        return_data,
        (
            data_orig,
            imageFile,
            imageIndex,
            target_apix3d,
            target_apix2d,
            twist,
            rise,
            csym,
            tilt,
            psi,
            dy,
        ),
    )

    return result


def itk_stitch(temp_dir):
    # ==========================================================================
    #
    #   Copyright NumFOCUS
    #
    #   Licensed under the Apache License, Version 2.0 (the "License");
    #   you may not use this file except in compliance with the License.
    #   You may obtain a copy of the License at
    #
    #          https://www.apache.org/licenses/LICENSE-2.0.txt
    #
    #   Unless required by applicable law or agreed to in writing, software
    #   distributed under the License is distributed on an "AS IS" BASIS,
    #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    #   See the License for the specific language governing permissions and
    #   limitations under the License.
    #
    # ==========================================================================*/
    import sys
    import os
    import itk
    import numpy as np
    import tempfile
    from pathlib import Path

    input_path = Path(temp_dir)
    output_path = Path(temp_dir)
    out_file = Path(temp_dir + "/itk_stitched.mrc")
    if not out_file.is_absolute():
        out_file = (output_path / out_file).resolve()

    dimension = 2

    stage_tiles = itk.TileConfiguration[dimension]()
    stage_tiles.Parse(str(input_path / "TileConfiguration.txt"))

    color_images = []  # for mosaic creation
    grayscale_images = []  # for registration
    for t in range(stage_tiles.LinearSize()):
        origin = stage_tiles.GetTile(t).GetPosition()
        filename = str(input_path / stage_tiles.GetTile(t).GetFileName())
        image = itk.imread(filename)
        spacing = image.GetSpacing()

        # tile configurations are in pixel (index) coordinates
        # so we convert them into physical ones
        for d in range(dimension):
            origin[d] *= spacing[d]

        image.SetOrigin(origin)
        color_images.append(image)

        image = itk.imread(filename, itk.F)  # read as grayscale
        image.SetOrigin(origin)
        grayscale_images.append(image)

    # only float is wrapped as coordinate representation type in TileMontage
    montage = itk.TileMontage[type(grayscale_images[0]), itk.F].New()
    montage.SetMontageSize(stage_tiles.GetAxisSizes())
    for t in range(stage_tiles.LinearSize()):
        montage.SetInputTile(t, grayscale_images[t])

    logger.info("Computing tile registration transforms")
    montage.Update()

    logger.info("Writing tile transforms")
    actual_tiles = stage_tiles  # we will update it later
    for t in range(stage_tiles.LinearSize()):
        index = stage_tiles.LinearIndexToNDIndex(t)
        regTr = montage.GetOutputTransform(index)
        tile = stage_tiles.GetTile(t)
        itk.transformwrite([regTr], str(output_path / (tile.GetFileName() + ".tfm")))

        # calculate updated positions - transform physical into index shift
        pos = tile.GetPosition()
        for d in range(dimension):
            pos[d] -= regTr.GetOffset()[d] / spacing[d]
        tile.SetPosition(pos)
        actual_tiles.SetTile(t, tile)
    actual_tiles.Write(str(output_path / "TileConfiguration.registered.txt"))

    logger.info("Producing the mosaic")

    input_pixel_type = itk.template(color_images[0])[1][0]
    try:
        input_rgb_type = itk.template(input_pixel_type)[0]
        accum_type = input_rgb_type[itk.F]  # RGB or RGBA input/output images
    except KeyError:
        accum_type = itk.D  # scalar input / output images

    resampleF = itk.TileMergeImageFilter[type(color_images[0]), accum_type].New()
    resampleF.SetMontageSize(stage_tiles.GetAxisSizes())
    for t in range(stage_tiles.LinearSize()):
        resampleF.SetInputTile(t, color_images[t])
        index = stage_tiles.LinearIndexToNDIndex(t)
        resampleF.SetTileTransform(index, montage.GetOutputTransform(index))
    resampleF.Update()
    # itk.imwrite(resampleF.GetOutput(), str(out_file))
    logger.info("Resampling complete")
    return np.array(resampleF.GetOutput())
