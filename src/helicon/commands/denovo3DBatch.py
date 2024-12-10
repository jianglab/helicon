#!/usr/bin/env python

"""A command line tool for de novo helical indexing and 3D reconstruction from a single 2D image"""

import itertools, os, sys, pathlib, datetime, joblib
import numpy as np

import helicon

from scipy.sparse import csr_matrix

try:
    from numba import jit, set_num_threads, prange
except ImportError:
    helicon.color_print(
        f"WARNING: failed to load numba. The program will run correctly but will be much slower. Run 'pip install numba' to install numba and speed up the program"
    )

    def jit(*args, **kwargs):
        return lambda f: f

    def set_num_threads(n: int):
        return

    prange = range

cache_dir = helicon.cache_dir / "denovo3DBatch"


def main(args):
    helicon.log_command_line()

    logger = helicon.get_logger(
        logfile="helicon.denovo3DBatch.log",
        verbose=args.verbose,
    )

    output_path = pathlib.Path(args.output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.force:
        import shutil

        shutil.rmtree(cache_dir)

    logger.info(" ".join(sys.argv))

    start_time = datetime.datetime.now()
    logger.info(f"Started at {start_time}")

    if args.cpu <= 0:
        args.cpu = helicon.available_cpu()
        logger.info(f"Will use {args.cpu} CPUs")

    input_file = pathlib.Path(args.input_file)

    if input_file.suffix in [".star", ".mrcs", ".mrc"]:
        if not input_file.exists():
            logger.error(f"ERROR: cannot find the input image file {str(input_file)}")
            sys.exit(-1)
        if input_file.suffix in [".star"]:
            df = star_to_dataframe(str(input_file), logger=logger)
            if args.i:
                bad_i = [i for i in args.i if i < 1 or i > len(df)]
                if len(bad_i):
                    logger.error(
                        f"ERROR: image {' '.join(map(str, bad_i))} out of the valid range [1, {nz}]"
                    )
                    sys.exit(-1)
                indices = [i - 1 for i in args.i]
            else:
                indices = range(len(df))
            images = []
            for i in indices:
                imageIndex, imageFile = df.loc[i, "pid"], df.loc[i, "filename"]
                imageFile = pathlib.Path(imageFile)
                if imageFile.suffix in [".mrc"]:
                    imageFile, imageFileOrig = (
                        pathlib.Path(imageFile.with_suffix(".mrcs").name),
                        imageFile,
                    )
                    if imageFile.exists():
                        imageFile.symlink_to(imageFileOrig)

                import mrcfile

                with mrcfile.open(imageFile, header_only=True) as mrc:
                    nx, ny, nz = (
                        int(mrc.header.nx),
                        int(mrc.header.ny),
                        int(mrc.header.nz),
                    )
                    apix = round(mrc.voxel_size.x * 1.0, 3)
                    images.append((None, str(imageFile), imageIndex + 1))
        elif input_file.suffix in [".mrcs", ".mrc"]:  # mrc/mrcs file
            if input_file.suffix in [".mrc"]:
                input_file, input_fileOrig = (
                    pathlib.Path(input_file.with_suffix(".mrcs").name),
                    input_file,
                )
                if not input_file.exists():
                    input_file.symlink_to(input_fileOrig)

            import mrcfile

            with mrcfile.open(input_file, header_only=True) as mrc:
                nx, ny, nz = int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)
                apix = round(mrc.voxel_size.x * 1.0, 3)
            if args.i:
                bad_i = [i for i in args.i if i < 1 or i > nz]
                if len(bad_i):
                    logger.error(
                        f"ERROR: image {' '.join(map(str, bad_i))} out of the valid range [1, {nz}]"
                    )
                    sys.exit(-1)
                images = [(None, str(input_file), i) for i in args.i]
            else:
                images = [(None, str(input_file), i + 1) for i in range(nz)]
        if args.apix > 0:
            apix = args.apix
        info = (
            f"size={nx}x{ny}pixels apix={apix}Å/pixel length={round(nx*apix)}Å (X-axis)"
        )
        logger.info(
            f"Input: {len(images)} image{'s' if len(images)>1 else ''} in {str(input_file)} with {info}"
        )
    else:  # simulation
        name, params = helicon.parse_param_str(args.input_file)
        if name is None:
            name = "Simulation"
        args.transpose = 0
        args.horizontalize = 0
        rise = params.get("rise", np.median(args.rises))
        if "twist" in params:
            twist = params.get("twist")
        elif "pitch" in params:
            pitch = params.get("pitch")
            twist = 360 / (pitch / rise)
        elif len(args.twists):
            twist = np.median(args.twists)
        else:
            pitch = np.median(args.pitches)
            twist = 360 / (pitch / rise)

        polymer = params.get("polymer", 0)
        n = params.get("n", 50 if polymer else 1)
        planarity = params.get("planarity", 0.9)
        apix = params.get("apix", args.apix if args.apix > 0 else 2)
        nx = params.get("nx", 80)
        ny = params.get("ny", 64)
        args.tube_diameter = ny * apix
        csym = params.get("csym", 1)
        noise = params.get("noise", 0.0)
        helical_diameter = params.get("helical_diameter", apix * ny * 0.5)
        ball_radius = params.get("ball_radius", max(apix, 1.9))
        tilt = params.get("tilt", 0)  # out-of-plane tilt
        rot = params.get("rot", 0)  # rotation around the helical axis
        psi = params.get("psi", 0)  # inplane rotation
        dy = params.get("dy", 0)  # shift (Å) perpendicular to the helical axis

        image = simulate_helical_projection(
            n=n,
            twist=twist,
            rise=rise,
            csym=csym,
            helical_diameter=helical_diameter,
            ball_radius=ball_radius,
            polymer=polymer,
            planarity=planarity,
            ny=ny,
            nx=nx,
            apix=apix,
            tilt=tilt,
            rot=rot,
            psi=psi,
            dy=dy,
        )
        if noise > 0:
            sigma = np.std(image[image > 1e-3])  # ignore background pixels
            image += np.random.normal(scale=sigma * noise, size=image.shape)
        images = [(image, name, 0)]
        info = f"pitch={round(rise*360/abs(twist), 1)}Å/twist={round(twist, 3)}° rise={round(rise, 3)}Å {csym=}{tilt_psi_dy_str(tilt, psi, dy)} size={nx}x{ny}pixels apix={apix}Å/pixel length={nx*apix}Å/{round(nx*apix/(rise*360/abs(twist)), 2)}pitch (X-axis)"
        logger.info(f"Input: {len(images)} simulated image with {info}")

    if len(args.twists):
        tr_pairs = list(itertools.product(args.twists, args.rises))
    else:
        pr_pairs = list(itertools.product(args.pitches, args.rises))
        tr_pairs = []
        for pitch, rise in pr_pairs:
            if pitch <= rise:
                continue
            twist = helicon.set_to_periodic_range(
                360 / (pitch / rise), min=-180, max=180
            )
            tr_pairs.append((twist, rise))

    itrtpy_list = list(
        itertools.product(images, tr_pairs, args.tilts, args.psis, args.dys)
    )
    return_3d = not (
        len(tr_pairs) * len(args.tilts) * len(args.psis) * len(args.dys) > 1
    )

    tasks = []
    for ti, t in enumerate(itrtpy_list):
        (data, imageFile, imageIndex), (twist, rise), tilt, psi, dy = t
        twist = np.round(helicon.set_to_periodic_range(twist, min=-180, max=180), 6)
        if abs(twist) < 0.01:
            logger.warning(
                f"WARNING: (twist={round(twist, 3)}, rise={round(rise, 3)}) will be ignored due to very small twist value (twist={round(twist, 3)}°)"
            )
            continue
        if abs(rise) < 0.01:
            logger.warning(
                f"WARNING: (twist={round(twist, 3)}, rise={round(rise, 3)}) will be ignored due to very small rise value (rise={round(rise, 3)}Å)"
            )
            continue
        if abs(rise) >= (
            args.tube_length / 2 if args.tube_length > 0 else nx * apix / 2
        ):
            logger.warning(
                f"WARNING: (twist={round(twist, 3)}, rise={round(rise, 3)}) will be ignored due to very large rise value (rise={round(rise, 3)}Å)"
            )
            continue
        if abs(tilt) > 45:
            logger.warning(f"WARNING: tilt={round(tilt, 2)} will be ignored")
            continue
        if abs(psi) > 45:
            logger.warning(f"WARNING: psi={round(psi, 2)} will be ignored")
            continue
        if abs(dy) > (ny * apix) / 4:
            logger.warning(f"WARNING: dy={round(dy, 2)} will be ignored")
            continue
        reconstruct_length = -1
        if args.reconstruct_length > 0:
            reconstruct_length = max(rise, args.reconstruct_length)
        elif args.reconstruct_length_pitch > 0:
            reconstruct_length = max(
                rise, args.reconstruct_length_pitch * rise * 360 / abs(twist)
            )
        elif args.reconstruct_length_rise > 0:
            reconstruct_length = max(rise, args.reconstruct_length_rise * rise)
        tasks.append(
            (
                ti,
                len(itrtpy_list),
                data,
                imageFile,
                imageIndex,
                twist,
                rise,
                (np.min(args.rises), np.max(args.rises)),
                args.csym,
                tilt,
                (np.min(args.tilts), np.max(args.tilts)),
                psi,
                dy,
                apix,
                args.denoise,
                args.low_pass,
                args.transpose,
                args.horizontalize,
                args.target_apix3d,
                args.target_apix2d,
                args.thresh_fraction,
                args.positive_constraint,
                args.tube_length,
                args.tube_diameter,
                args.tube_diameter_inner,
                reconstruct_length,
                args.sym_oversample,
                args.interpolation,
                args.fsc_test,
                return_3d,
                args.algorithm,
                max(0, args.verbose - 2),
                logger,
            )
        )

    if len(tasks) < 1:
        logger.warning("Nothing to do. I will quit")
        return

    if args.reconstruct_length_pitch > 0:
        np.random.shuffle(
            tasks
        )  # spread jobs using a large amount of memory (e.g. long pitches)

    from tqdm import tqdm
    from joblib import Parallel, delayed

    results = list(
        tqdm(
            Parallel(return_as="generator", n_jobs=args.cpu if len(tasks) > 1 else 1)(
                delayed(process_one_task)(*task) for task in tasks
            ),
            unit="job",
            total=len(tasks),
            disable=len(tasks) < 2
            or (args.cpu > 1 and args.verbose < 2)
            or (args.cpu == 1 and args.verbose != 2),
        )
    )

    end_time = datetime.datetime.now()
    logger.info(
        f"{len(tasks)} reconstruction jobs finished at {end_time}. Runtime = {end_time-start_time}"
    )

    results_none = [res for res in results if res is None]
    if len(results_none):
        logger.info(
            f"{len(results_none)}/{len(results)} results are None and thus discarded"
        )
        results = [res for res in results if res is not None]

    results.sort(key=lambda x: x[2][1:])

    if return_3d:
        mapFiles = saveMaps(
            results,
            target_size=(nx, ny, ny),
            target_apix=apix,
            mapFilePrefix=args.output_prefix,
            verbose=args.verbose,
            logger=logger,
        )
        if len(mapFiles) > 1:
            s = "\n" + "\n".join(mapFiles)
            logger.info(f"Reconstructed maps saved to {s}")
        else:
            logger.info(f"Reconstructed map saved to {' '.join(mapFiles)}")

    if args.save_projections:
        lstFiles = writeLstFile(
            results,
            top_k=args.top_k,
            apix2d_orig=apix,
            lstFilePrefix=args.output_prefix,
        )
        if len(lstFiles) > 1:
            s = "\n" + "\n".join(lstFiles) + "\n"
            logger.info(
                f"Reconstructed map projections saved to {s}Use 'e2display.py' to display the images"
            )
        else:
            logger.info(
                f"Reconstructed map projections saved to {' '.join(lstFiles)}. Use 'e2display.py {' '.join(lstFiles)}' to display the images"
            )

    pdfFiles = writePdf(
        results,
        pdfFilePrefix=args.output_prefix,
        top_k=args.top_k,
        use_pitch=len(args.pitches) > 0,
        image_info=info,
        cmap=args.color_map,
    )
    if len(pdfFiles) > 1:
        s = "\n" + "\n".join(pdfFiles)
        logger.info(f"Reconstructed map projections/scores saved to {s}")
    else:
        logger.info(
            f"Reconstructed map projections/scores saved to {' '.join(pdfFiles)}"
        )

    if len(images) > 1:
        pdfFile = plotAllScores(
            results, pdfFilePrefix=args.output_prefix, use_pitch=len(args.pitches) > 0
        )
        if pdfFile:
            pdfFiles.append(pdfFile)
            logger.info(f"Reconstruction scores of all input images saved to {pdfFile}")

    end_time = datetime.datetime.now()
    logger.info(f"Completed at {end_time}. Total run time = {end_time-start_time}")

    if args.verbose > 2:
        import webbrowser

        browser = webbrowser.get()
        for i, pdfFile in enumerate(pdfFiles):
            url = pathlib.Path(pdfFile).resolve().as_uri()
            browser.open(url, new=1 if i == 0 else 2)


np.set_printoptions(
    precision=6
)  # to speed up joblib memory cache for input numpy arrays and avoid the "UserWarning: Persisting input arguments took 0.65s to run"


@helicon.cache(
    cache_dir=cache_dir,
    ignore=["ti", "ntasks", "verbose", "logger"],
    expires_after=7,
    verbose=0,
)  # 7 days
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
    dy,
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
    algorithm,
    verbose,
    logger,
):

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

    if target_apix2d <= 0:
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
            apix2d_orig, round(np.power(vol / (nx * ny), 1 / 3) + 0.5)
        )  # make the number of 3D voxels < number of data points (i.e. number of 2D pixels of down-scaled image)
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
    )  # direction of helical axis
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

    with helicon.Timer(
        f"lsq_reconstruct: {round(pitch, 1)}Å/twist={round(twist, 3)}° rise={round(rise, 3)}Å",
        verbose=verbose > 10,
    ):
        (rec3d, rec3d_set_1, rec3d_set_2), score = lsq_reconstruct(
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
            verbose=verbose,
            algorithm=algorithm,
        )

    with helicon.Timer("apply_helical_symmetry", verbose=verbose > 10):
        twist_degree = twist if abs(twist) < 90 else 180 - abs(twist)
        if abs(twist_degree) > 1e-2:
            pitch_pixel = int(360 / abs(twist_degree) * rise / target_apix2d + 0.5)
        else:
            pitch_pixel = int(np.ceil(2 * rise / target_apix2d))
        new_length = max(reconstruct_length_2d_pixel, int(pitch_pixel * 1.2))
        cpu = helicon.available_cpu()
        rec3d_xform = helicon.apply_helical_symmetry(
            data=rec3d,
            apix=target_apix3d,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=csym,
            new_size=(
                new_length,
                reconstruct_diameter_2d_pixel,
                reconstruct_diameter_2d_pixel,
            ),
            new_apix=target_apix2d,
            cpu=cpu,
        )
    rec3d_xform_2 = helicon.transform_map(
        rec3d_xform, scale=1.0, tilt=tilt, psi=psi, dy_pixel=dy / target_apix2d
    )
    rec3d_x_proj = np.sum(rec3d_xform_2, axis=2).T
    rec3d_y_proj = np.sum(rec3d_xform_2, axis=1).T
    rec3d_y_proj_max = rec3d_y_proj.max()
    if rec3d_y_proj_max > 0:
        rec3d_y_proj *= rec3d_x_proj.max() / rec3d_y_proj_max

    nz_per_rise = max(1, int(np.ceil(rise / target_apix2d)))
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

    # shape_target = (min(data.shape[0], rec3d_x_proj.shape[0]), min(data.shape[1], rec3d_x_proj.shape[1]))
    # score *= helicon.cosine_similarity(helicon.crop_center(data, shape=shape_target), helicon.crop_center(rec3d_x_proj, shape=shape_target))

    nz, ny, nx = rec3d.shape
    if target_apix2d != apix2d_orig:
        apix_tag = f"apix={apix2d_orig}->{target_apix2d}Å"
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


def lsq_reconstruct(
    projection_image,
    scale2d_to_3d,
    twist_degree,
    rise_pixel,
    csym=1,
    tilt_degree=0,
    psi_degree=0,
    dy_pixel=0,
    thresh_fraction=-1,
    positive_constraint=-1,
    reconstruct_diameter_3d_inner_pixel=0,
    reconstruct_diameter_2d_pixel=-1,
    reconstruct_diameter_3d_pixel=-1,
    reconstruct_length_2d_pixel=-1,
    reconstruct_length_3d_pixel=-1,
    sym_oversample=1,
    interpolation="nn",
    fsc_test=0,
    verbose=0,
    algorithm=dict(model="lsq"),
):

    rmin = reconstruct_diameter_3d_inner_pixel / 2
    rmax = reconstruct_diameter_3d_pixel // 2 - 1

    mask = helicon.get_cylindrical_mask(
        nz=reconstruct_length_3d_pixel,
        ny=reconstruct_diameter_3d_pixel,
        nx=reconstruct_diameter_3d_pixel,
        rmin=rmin,
        rmax=rmax,
    )
    mz, my, mx = mask.shape
    assert mz == reconstruct_length_3d_pixel

    n_3d_voxels = np.count_nonzero(mask)
    n_2d_pixels = reconstruct_diameter_2d_pixel * reconstruct_length_2d_pixel
    max_equations = 2**26  # 64 million

    with helicon.Timer(f"build_A_data_matrix - {interpolation}", verbose=verbose > 10):
        A_data, b_data, b_data_pid = build_A_data_matrix(
            image=projection_image,
            scale2d_to_3d=scale2d_to_3d,
            twist_degree=twist_degree,
            rise_pixel=rise_pixel,
            csym=csym,
            tilt_degree=tilt_degree,
            psi_degree=psi_degree,
            dy_pixel=dy_pixel,
            reconstruct_diameter_2d_pixel=reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel=reconstruct_length_2d_pixel,
            reconstruct_diameter_3d_pixel=reconstruct_diameter_3d_pixel,
            reconstruct_diameter_3d_inner_pixel=reconstruct_diameter_3d_inner_pixel,
            reconstruct_length_3d_pixel=reconstruct_length_3d_pixel,
            min_projection_lines=min(
                max_equations, int(max(n_2d_pixels, n_3d_voxels) * sym_oversample)
            ),
            interpolation=interpolation,
            verbose=verbose,
        )

    with helicon.Timer(
        f"build_A_helical_sym_matrix - {interpolation}", verbose=verbose > 10
    ):
        A_hsym, b_hsym = build_A_helical_sym_matrix(
            nz=mz,
            ny=my,
            nx=mx,
            twist_degree=twist_degree,
            rise_pixel=rise_pixel,
            csym=csym,
            rmin=rmin,
            rmax=rmax,
            min_sym_pairs=min(
                max_equations, int(max(n_2d_pixels, n_3d_voxels) * sym_oversample)
            ),
            interpolation=interpolation,
            verbose=verbose,
        )

    def split_A_b(A, b, b_id, mode):
        if mode <= 0:  # no split, return the same data for both sets
            return (A, b), (A, b)

        if b_id is None:
            b_id_unique = np.arrange(len(b))
        else:
            b_id_unique = sorted(set(b_id))
        n = len(b_id_unique)
        if mode == 1:  # random split of input image pixels
            b_id_unique = list(set(b_id))
            np.random.shuffle(b_id_unique)
            b_id_unique_set_1 = b_id_unique[: n // 2]
        elif mode == 2:  # even/odd pixels
            b_id_unique_set_1 = b_id_unique[::2]
        elif mode == 3:  # left/right halves
            b_id_unique_set_1 = b_id_unique[: n // 2]
        else:  # left 1/3 + right 1/3 vs center 1/3
            b_id_unique_set_1 = b_id_unique[: n // 3] + b_id_unique[n * 2 // 3 :]

        is_set_1 = np.isin(b_id, b_id_unique_set_1)
        A_set_1 = A[is_set_1]
        b_set_1 = b[is_set_1]
        A_set_2 = A[~is_set_1]
        b_set_2 = b[~is_set_1]
        assert len(b_set_1) + len(b_set_2) == len(
            b
        ), f"ERROR: {len(b_set_1)=:,}\t{len(b_set_2)=:,}\t{len(b)=:,}"
        return (A_set_1, b_set_1), (A_set_2, b_set_2)

    def solve_equations(
        A_data,
        b_data,
        A_hsym,
        b_hsym,
        positive=False,
        algorithm="elasticnet",
        train_fraction=1.0,
        verbose=0,
    ):
        if not (A_hsym is None or b_hsym is None):
            from scipy.sparse import vstack

            A = vstack((A_data, A_hsym))
            b = np.concatenate((b_data, b_hsym))
        else:
            A = A_data
            b = b_data
        if 0 < train_fraction < 1:
            shuffled_indices = np.arange(A.shape[0])
            np.random.shuffle(shuffled_indices)
            n = int(len(shuffled_indices) * train_fraction + 0.5)
            A_train = A[shuffled_indices[:n], :]
            b_train = b[shuffled_indices[:n]]
            A_test = A[shuffled_indices[n:], :]
            b_test = b[shuffled_indices[n:]]
        else:
            A_train = A
            b_train = b
            A_test = None
            b_test = None

        tol = 1e-2
        max_iter = 200

        # TODO: use the GPU accelerated Nvidia cuml library: https://docs.rapids.ai/api/cuml/stable/api/#linear-regression

        if (
            algorithm["model"] == "lsq"
        ):  # ordinary linear least square without regularization
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
            if positive:
                # positive constraint - very important when pitch>2*tube_length
                lb = 0.0
                ub = np.max(b_data)
                logger.debug(
                    f"Imposing constraint for the reconstruction: lb={round(lb, 6)} ub={round(ub, 6)}"
                )
            else:
                lb = -np.inf
                ub = np.inf

            from scipy.optimize import lsq_linear

            res = lsq_linear(
                A,
                b,
                bounds=(lb, ub),
                tol=tol,
                max_iter=max_iter,
                lsmr_maxiter=1000,
                lsmr_tol="auto",
                verbose=verbose,
            )
            return res.x.astype(np.float32), None

        if (
            algorithm["model"] == "lreg"
        ):  # ordinary linear least square without regularization
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
            from sklearn.linear_model import LinearRegression

            if positive:
                A_train = A_train.toarray()
                logger.warn(
                    "WARNING: --algorithm=lreq with positive contraints uses very large amount memory!"
                )
            model = LinearRegression(fit_intercept=True, positive=positive)
        elif algorithm["model"] == "lasso":  # linear least square + L1 regularization
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
            from sklearn.linear_model import Lasso

            model = Lasso(
                alpha=algorithm.get("alpha", 1e-4),
                fit_intercept=True,
                positive=positive,
                selection="random",
                tol=tol,
                max_iter=max_iter,
            )
        elif (
            algorithm["model"] == "elasticnet"
        ):  # linear least square + L1/L2 regularization
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
            from sklearn.linear_model import ElasticNet

            model = ElasticNet(
                alpha=algorithm.get("alpha", 1e-4),
                l1_ratio=algorithm.get("l1_ratio", 0.5),
                fit_intercept=True,
                positive=positive,
                selection="random",
                tol=tol,
                max_iter=max_iter,
            )
        elif algorithm["model"] == "ridge":  # linear least square + L2 regularization
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
            from sklearn.linear_model import Ridge

            model = Ridge(
                alpha=algorithm.get("alpha", 1),
                fit_intercept=True,
                positive=positive,
                tol=tol,
                max_iter=max_iter,
            )
        elif algorithm["model"] == "ard":  # very slow, not practical
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html
            from sklearn.linear_model import ARDRegression

            A_train = A_train.toarray()
            model = ARDRegression(
                alpha_1=algorithm.get("alpha", 1e-6),
                alpha_2=algorithm.get("alpha", 1e-6),
                fit_intercept=True,
                tol=tol,
                max_iter=max_iter,
            )

        model.fit(X=A_train, y=b_train)
        res = model.coef_.astype(np.float32)
        if not np.any(res):
            if algorithm["model"] in ["lreg"]:  # this should not happen
                res[len(res) // 2] = 1
            else:
                while not np.any(res):
                    model.alpha *= 0.1
                    model.fit(X=A_train, y=b_train)
                    res = model.coef_.astype(np.float32)
        if A_test is not None and b_test is not None:
            score = helicon.cosine_similarity(A_test.dot(res), b_test)
            # score = model.score(X=A_test, y=b_test)
        else:
            score = None
        return res, score

    n_eqns = A_data.shape[0]
    n_unknowns = A_data.shape[1]
    n_nonzeros = A_data.count_nonzero()
    if A_hsym is not None:
        n_eqns += A_hsym.shape[0]
        n_nonzeros += A_hsym.count_nonzero()
    sparsity = 1 - n_nonzeros / (n_eqns * n_unknowns)

    pitch_pixel = round(rise_pixel * 360 / abs(twist_degree))
    positive = positive_constraint > 0 or (
        positive_constraint < 0 and pitch_pixel > round(reconstruct_length_3d_pixel * 2)
    )
    train_fraction = 1.0

    with helicon.Timer(
        f"solve_equations{' Full' if fsc_test>0 else ''}: {n_eqns:,} equations, {n_unknowns:,} unknowns, {sparsity*100:f}% sparsity",
        verbose=verbose > 10,
    ):
        x, score = solve_equations(
            A_data,
            b_data,
            A_hsym,
            b_hsym,
            positive=positive,
            algorithm=algorithm,
            train_fraction=train_fraction,
            verbose=2 if verbose > 10 else 0,
        )

    Abx_data_triplets = [(A_data, b_data, x)]

    xs = [x]
    scores = [score]

    if fsc_test >= 1:
        (A_data_set_1, b_data_set_1), (A_data_set_2, b_data_set_2) = split_A_b(
            A_data, b_data, b_data_pid, mode=fsc_test
        )

        Ab_pairs = [
            (A_data_set_1, A_hsym, b_data_set_1, b_hsym, "Half-1"),
            (A_data_set_2, A_hsym, b_data_set_2, b_hsym, "Half-2"),
        ]

        for pair_A_data, pair_A_hsym, pair_b_data, pair_b_hsym, tag in Ab_pairs:
            n_eqns = pair_A_data.shape[0]
            n_unknowns = pair_A_data.shape[1]
            n_nonzeros = pair_A_data.count_nonzero()
            if pair_A_hsym is not None and pair_b_hsym is not None:
                n_eqns += pair_A_hsym.shape[0]
                n_nonzeros += pair_A_hsym.count_nonzero()
            sparsity = 1 - n_nonzeros / (n_eqns * n_unknowns)
            with helicon.Timer(
                f"solve_equations {tag}: {n_eqns:,} equations, {n_unknowns:,} unknowns, {sparsity*100:f}% sparsity",
                verbose=verbose > 10,
            ):
                x, score = solve_equations(
                    pair_A_data,
                    pair_b_data,
                    pair_A_hsym,
                    pair_b_hsym,
                    positive=positive,
                    algorithm=algorithm,
                    train_fraction=train_fraction,
                    verbose=2 if verbose > 10 else 0,
                )
                xs.append(x)
                scores.append(score)

        Abx_data_triplets += [
            (A_data_set_1, b_data_set_1, xs[1]),
            (A_data_set_2, b_data_set_2, xs[2]),
        ]

    if np.any([score is None for score in scores]):
        scores = []
        for tmp_A, tmp_b, tmp_x in Abx_data_triplets:
            if thresh_fraction < 0:
                scores.append(helicon.cosine_similarity(tmp_A.dot(tmp_x), tmp_b))
            else:
                scores.append(
                    helicon.cosine_similarity(tmp_A.dot(np.clip(tmp_x, 0, None)), tmp_b)
                )

    if len(scores) == 3:
        score = scores[0] / 2 + (scores[1] + scores[2]) / 4
    else:
        score = scores[0]

    shape = (
        reconstruct_length_3d_pixel,
        reconstruct_diameter_3d_pixel,
        reconstruct_diameter_3d_pixel,
    )
    rec3d = np.zeros(shape, dtype=np.float32)
    rec3d[mask] = xs[0]

    if len(xs) == 1:
        return (rec3d, None, None), score
    else:
        rec3d_set_1 = np.zeros(shape, dtype=np.float32)
        rec3d_set_2 = np.zeros(shape, dtype=np.float32)
        rec3d_set_1[mask] = xs[1]
        rec3d_set_2[mask] = xs[2]
        return (rec3d, rec3d_set_1, rec3d_set_2), score


@helicon.cache(
    cache_dir=cache_dir, ignore=["verbose"], expires_after=7, verbose=0
)  # 7 days
def build_A_helical_sym_matrix(
    nz: int,
    ny: int,
    nx: int,
    twist_degree: float,
    rise_pixel: float,
    csym: int,
    rmin: float,
    rmax: float,
    min_sym_pairs: int,
    interpolation: str,
    verbose: int = 0,
):

    hcsym_pairs = sorted_hsym_csym_pairs(twist_degree, rise_pixel, csym, nz)

    mask, (Z, Y, X) = helicon.get_cylindrical_mask(
        nz, ny, nx, rmin=rmin, rmax=rmax, return_xyz=True
    )
    n_x = np.count_nonzero(mask)
    mask_nonzero_indices_zyx_tuple = np.nonzero(mask)
    mask_nonzero_indices_matrix = np.zeros(mask.shape, dtype=int) - 1
    mask_nonzero_indices_matrix[mask_nonzero_indices_zyx_tuple] = np.arange(n_x)
    xyz = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).transpose()

    # sparse A matrix
    csr_A = []

    from scipy.spatial.transform import Rotation as R

    if interpolation in ["linear", "linear01", "linear11"]:

        @jit(nopython=True, cache=True, nogil=True)
        def loop_kji(
            Zi,
            Yi,
            Xi,
            Zj,
            Yj,
            Xj,
            mask,
            mask_nonzero_indices_zyx_tuple,
            mask_nonzero_indices_matrix,
            pair_ids,
        ):
            mask_indices_Z, mask_indices_Y, mask_indices_X = (
                mask_nonzero_indices_zyx_tuple
            )
            mz, my, mx = mask.shape
            n_indices = len(mask_indices_Z)
            csr_row_tmp = np.zeros(n_indices * 16, dtype=np.float32)
            csr_col_tmp = np.zeros(n_indices * 16, dtype=np.float32)
            csr_data_tmp = np.zeros(n_indices * 16, dtype=np.float32)
            csr_rc_tmp_count = 0
            row_count_tmp = 0
            for mi in range(n_indices):
                k = mask_indices_Z[mi]
                j = mask_indices_Y[mi]
                i = mask_indices_X[mi]
                zi = int(Zi[k, j, i])
                yi = int(Yi[k, j, i])
                xi = int(Xi[k, j, i])
                zj = int(Zj[k, j, i])
                yj = int(Yj[k, j, i])
                xj = int(Xj[k, j, i])
                if zi < 0 or zi > mz - 1:
                    continue
                if zj < 0 or zj > mz - 1:
                    continue
                if yi < 0 or yi > my - 1:
                    continue
                if yj < 0 or yj > my - 1:
                    continue
                if xi < 0 or xi > mx - 1:
                    continue
                if xj < 0 or xj > mx - 1:
                    continue
                if zi + 1 > mz - 1 or yi + 1 > my - 1 or xi + 1 > mx - 1:
                    continue
                if zj + 1 > mz - 1 or yj + 1 > my - 1 or xj + 1 > mx - 1:
                    continue
                if not mask[zi, yi, xi]:
                    continue
                if not mask[zi + 1, yi, xi]:
                    continue
                if not mask[zi, yi + 1, xi]:
                    continue
                if not mask[zi + 1, yi + 1, xi]:
                    continue
                if not mask[zi, yi, xi + 1]:
                    continue
                if not mask[zi + 1, yi, xi + 1]:
                    continue
                if not mask[zi, yi + 1, xi + 1]:
                    continue
                if not mask[zi + 1, yi + 1, xi + 1]:
                    continue
                if not mask[zj, yj, xj]:
                    continue
                if not mask[zj + 1, yj, xj]:
                    continue
                if not mask[zj, yj + 1, xj]:
                    continue
                if not mask[zj + 1, yj + 1, xj]:
                    continue
                if not mask[zj, yj, xj + 1]:
                    continue
                if not mask[zj + 1, yj, xj + 1]:
                    continue
                if not mask[zj, yj + 1, xj + 1]:
                    continue
                if not mask[zj + 1, yj + 1, xj + 1]:
                    continue

                i_000 = mask_nonzero_indices_matrix[zi, yi, xi]
                if i_000 < 0 or i_000 > n_x - 1:
                    continue
                i_001 = mask_nonzero_indices_matrix[zi, yi, xi + 1]
                if i_001 < 0 or i_001 > n_x - 1:
                    continue
                i_010 = mask_nonzero_indices_matrix[zi, yi + 1, xi]
                if i_010 < 0 or i_010 > n_x - 1:
                    continue
                i_011 = mask_nonzero_indices_matrix[zi, yi + 1, xi + 1]
                if i_011 < 0 or i_011 > n_x - 1:
                    continue
                i_100 = mask_nonzero_indices_matrix[zi + 1, yi, xi]
                if i_100 < 0 or i_100 > n_x - 1:
                    continue
                i_101 = mask_nonzero_indices_matrix[zi + 1, yi, xi + 1]
                if i_101 < 0 or i_101 > n_x - 1:
                    continue
                i_110 = mask_nonzero_indices_matrix[zi + 1, yi + 1, xi]
                if i_110 < 0 or i_110 > n_x - 1:
                    continue
                i_111 = mask_nonzero_indices_matrix[zi + 1, yi + 1, xi + 1]
                if i_111 < 0 or i_111 > n_x - 1:
                    continue

                j_000 = mask_nonzero_indices_matrix[zj, yj, xj]
                if j_000 < 0 or j_000 > n_x - 1:
                    continue
                j_001 = mask_nonzero_indices_matrix[zj, yj, xj + 1]
                if j_001 < 0 or j_001 > n_x - 1:
                    continue
                j_010 = mask_nonzero_indices_matrix[zj, yj + 1, xj]
                if j_010 < 0 or j_010 > n_x - 1:
                    continue
                j_011 = mask_nonzero_indices_matrix[zj, yj + 1, xj + 1]
                if j_011 < 0 or j_011 > n_x - 1:
                    continue
                j_100 = mask_nonzero_indices_matrix[zj + 1, yj, xj]
                if j_100 < 0 or j_100 > n_x - 1:
                    continue
                j_101 = mask_nonzero_indices_matrix[zj + 1, yj, xj + 1]
                if j_101 < 0 or j_101 > n_x - 1:
                    continue
                j_110 = mask_nonzero_indices_matrix[zj + 1, yj + 1, xj]
                if j_110 < 0 or j_110 > n_x - 1:
                    continue
                j_111 = mask_nonzero_indices_matrix[zj + 1, yj + 1, xj + 1]
                if j_111 < 0 or j_111 > n_x - 1:
                    continue

                if abs(zi - zj) < 3 or abs(yi - yj) < 3 or abs(xi - xj) < 3:
                    continue

                zir = round(Zi[k, j, i])
                yir = round(Yi[k, j, i])
                xir = round(Xi[k, j, i])
                zjr = round(Zj[k, j, i])
                yjr = round(Yj[k, j, i])
                xjr = round(Xj[k, j, i])
                ir = mask_nonzero_indices_matrix[zir, yir, xir]
                jr = mask_nonzero_indices_matrix[zjr, yjr, xjr]
                pair_id = ir * n_indices + jr
                if pair_id in pair_ids:
                    continue
                pair_id2 = jr * n_indices + ir
                pair_ids.add(pair_id)
                pair_ids.add(pair_id2)

                zf = Zi[k, j, i] - zi
                yf = Yi[k, j, i] - yi
                xf = Xi[k, j, i] - xi
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_000
                csr_data_tmp[csr_rc_tmp_count] = (1 - zf) * (1 - yf) * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_001
                csr_data_tmp[csr_rc_tmp_count] = (1 - zf) * (1 - yf) * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_010
                csr_data_tmp[csr_rc_tmp_count] = (1 - zf) * yf * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_011
                csr_data_tmp[csr_rc_tmp_count] = (1 - zf) * yf * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_100
                csr_data_tmp[csr_rc_tmp_count] = zf * (1 - yf) * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_101
                csr_data_tmp[csr_rc_tmp_count] = zf * (1 - yf) * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_110
                csr_data_tmp[csr_rc_tmp_count] = xf * yf * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = i_111
                csr_data_tmp[csr_rc_tmp_count] = xf * yf * zf
                csr_rc_tmp_count += 1

                zf = Zj[k, j, i] - zj
                yf = Yj[k, j, i] - yj
                xf = Xj[k, j, i] - xj
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_000
                csr_data_tmp[csr_rc_tmp_count] = -(1 - zf) * (1 - yf) * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_001
                csr_data_tmp[csr_rc_tmp_count] = -(1 - zf) * (1 - yf) * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_010
                csr_data_tmp[csr_rc_tmp_count] = -(1 - zf) * yf * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_011
                csr_data_tmp[csr_rc_tmp_count] = -(1 - zf) * yf * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_100
                csr_data_tmp[csr_rc_tmp_count] = -zf * (1 - yf) * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_101
                csr_data_tmp[csr_rc_tmp_count] = -zf * (1 - yf) * xf
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_110
                csr_data_tmp[csr_rc_tmp_count] = -xf * yf * (1 - xf)
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = j_111
                csr_data_tmp[csr_rc_tmp_count] = -xf * yf * zf
                csr_rc_tmp_count += 1

                row_count_tmp += 1
            return (
                csr_row_tmp[:csr_rc_tmp_count],
                csr_col_tmp[:csr_rc_tmp_count],
                csr_data_tmp[:csr_rc_tmp_count],
                row_count_tmp,
            )

    else:  # nearest neighbor

        @jit(nopython=True, cache=True, nogil=True)
        def loop_kji(
            Zi,
            Yi,
            Xi,
            Zj,
            Yj,
            Xj,
            mask,
            mask_nonzero_indices_zyx_tuple,
            mask_nonzero_indices_matrix,
            pair_ids,
        ):
            mask_indices_Z, mask_indices_Y, mask_indices_X = (
                mask_nonzero_indices_zyx_tuple
            )
            mz, my, mx = mask.shape
            n_indices = len(mask_indices_Z)
            csr_row_tmp = np.zeros(n_indices * 2, dtype=np.float32)
            csr_col_tmp = np.zeros(n_indices * 2, dtype=np.float32)
            csr_data_tmp = np.zeros(n_indices * 2, dtype=np.float32)
            csr_rc_tmp_count = 0
            row_count_tmp = 0
            for mi in range(n_indices):
                k = mask_indices_Z[mi]
                j = mask_indices_Y[mi]
                i = mask_indices_X[mi]
                zi = round(Zi[k, j, i])
                yi = round(Yi[k, j, i])
                xi = round(Xi[k, j, i])
                zj = round(Zj[k, j, i])
                yj = round(Yj[k, j, i])
                xj = round(Xj[k, j, i])
                if zi < 0 or zi > mz - 1:
                    continue
                if zj < 0 or zj > mz - 1:
                    continue
                if yi < 0 or yi > my - 1:
                    continue
                if yj < 0 or yj > my - 1:
                    continue
                if xi < 0 or xi > mx - 1:
                    continue
                if xj < 0 or xj > mx - 1:
                    continue
                if not mask[zi, yi, xi]:
                    continue
                if not mask[zj, yj, xj]:
                    continue
                index_i = mask_nonzero_indices_matrix[zi, yi, xi]
                if index_i < 0 or index_i > n_indices - 1:
                    continue
                index_j = mask_nonzero_indices_matrix[zj, yj, xj]
                if index_j < 0 or index_j > n_indices - 1:
                    continue
                pair_id = index_i * n_indices + index_j
                if pair_id in pair_ids:
                    continue
                pair_ids.add(pair_id)
                pair_id2 = index_j * n_indices + index_i
                pair_ids.add(pair_id2)

                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = index_i
                csr_data_tmp[csr_rc_tmp_count] = 1
                csr_rc_tmp_count += 1
                csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                csr_col_tmp[csr_rc_tmp_count] = index_j
                csr_data_tmp[csr_rc_tmp_count] = -1
                csr_rc_tmp_count += 1
                row_count_tmp += 1
            return (
                csr_row_tmp[:csr_rc_tmp_count],
                csr_col_tmp[:csr_rc_tmp_count],
                csr_data_tmp[:csr_rc_tmp_count],
                row_count_tmp,
            )

    with helicon.Timer(f"\tbuild csr matrix - {interpolation}", verbose=verbose > 10):
        pair_ids = set([-1])
        row_count = 0
        for pi, p in enumerate(hcsym_pairs):
            angle, ((hsym_i, csym_i), (hsym_j, csym_j)) = p[0], p[-1]
            ri = R.from_euler(
                "z", twist_degree * hsym_i + csym_i * 360 / csym, degrees=True
            )
            tmp_xyz = ri.apply(xyz, inverse=False)
            Xi = tmp_xyz[:, 0].reshape((nz, ny, nx)) + nx // 2  # axes order: z, y, x
            Yi = tmp_xyz[:, 1].reshape((nz, ny, nx)) + ny // 2  # axes order: z, y, x
            Zi = (
                tmp_xyz[:, 2].reshape((nz, ny, nx)) + nz // 2 + rise_pixel * hsym_i
            )  # axes order: z, y, x

            rj = R.from_euler(
                "z", twist_degree * hsym_j + csym_j * 360 / csym, degrees=True
            )
            tmp_xyz = rj.apply(xyz, inverse=False)
            Xj = tmp_xyz[:, 0].reshape((nz, ny, nx)) + nx // 2  # axes order: z, y, x
            Yj = tmp_xyz[:, 1].reshape((nz, ny, nx)) + ny // 2  # axes order: z, y, x
            Zj = (
                tmp_xyz[:, 2].reshape((nz, ny, nx)) + nz // 2 + rise_pixel * hsym_j
            )  # axes order: z, y, x

            try:
                from numba.core.errors import NumbaPendingDeprecationWarning
                import warnings

                warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
            except ImportError:
                pass

            csr_row_tmp, csr_col_tmp, csr_data_tmp, row_count_tmp = loop_kji(
                Zi,
                Yi,
                Xi,
                Zj,
                Yj,
                Xj,
                mask,
                mask_nonzero_indices_zyx_tuple,
                mask_nonzero_indices_matrix,
                pair_ids,
            )
            row_count += row_count_tmp
            if verbose > 20:
                print(
                    f"{pi+1}/{len(hcsym_pairs)}:",
                    round(angle, 2),
                    (hsym_i, csym_i),
                    (hsym_j, csym_j),
                    f"{n_x:,}",
                    f"+{row_count_tmp:,}",
                    f"{row_count:,}",
                    f"target={min_sym_pairs:,}",
                )
            if row_count_tmp:
                csr_A_tmp = csr_matrix(
                    (csr_data_tmp, (csr_row_tmp, csr_col_tmp)),
                    shape=(row_count_tmp, n_x),
                    dtype=np.float32,
                )
                csr_A.append(csr_A_tmp)
            if row_count >= min_sym_pairs:
                break

        if len(csr_A):
            from scipy.sparse import vstack

            A = vstack(csr_A)
            b = np.zeros(row_count, dtype=np.float32)
            assert A.shape[0] == row_count
        else:
            A = None
            b = None
        return A, b


@helicon.cache(
    cache_dir=cache_dir, ignore=["verbose"], expires_after=7, verbose=0
)  # 7 days
def build_A_data_matrix(
    image,
    scale2d_to_3d,
    twist_degree,
    rise_pixel,
    csym,
    tilt_degree,
    psi_degree,
    dy_pixel,
    reconstruct_diameter_2d_pixel,
    reconstruct_length_2d_pixel,
    reconstruct_diameter_3d_pixel,
    reconstruct_diameter_3d_inner_pixel,
    reconstruct_length_3d_pixel,
    min_projection_lines,
    interpolation,
    verbose=0,
):

    with helicon.Timer("back_project_2d_coords_to_3d_coords", verbose=verbose > 10):
        coords_3d, pixel_vals = back_project_2d_coords_to_3d_coords(
            image=image,
            scale2d_to_3d=scale2d_to_3d,
            reconstruct_diameter_2d_pixel=reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel=reconstruct_length_2d_pixel,
        )

    X0, Y0, Z0 = coords_3d  # axes order:  z, y, x. helical axis along z
    assert X0[:, :, 0].shape[::-1] == pixel_vals.shape

    rmin = reconstruct_diameter_3d_inner_pixel / 2
    rmax = reconstruct_diameter_3d_pixel // 2 - 1

    nz, ny, nx = X0.shape  # axes order: z, y, x
    if reconstruct_length_3d_pixel <= 0:
        reconstruct_length_3d_pixel = nz

    mask = helicon.get_cylindrical_mask(
        nz=reconstruct_length_3d_pixel, ny=ny, nx=nx, rmin=rmin, rmax=rmax
    )
    n_x = np.count_nonzero(mask)
    mask_nonzero_indices_matrix = np.zeros(mask.shape, dtype=int) - 1
    mask_nonzero_indices_matrix[np.nonzero(mask)] = np.arange(n_x)

    coords0 = np.vstack((X0.ravel(), Y0.ravel(), Z0.ravel())).transpose()
    from scipy.spatial.transform import Rotation as R

    coords0[:, 1] -= dy_pixel
    r = R.from_euler("yx", (tilt_degree, psi_degree), degrees=True)
    coords0 = r.apply(coords0, inverse=True)

    # sparse A matrix
    csr_A = []
    csr_b = []  # one value for each pixel in pixel_vals
    b_pid = []
    n_b = 0
    if interpolation in ["linear", "linear10", "linear11"]:

        @jit(nopython=True, cache=True, nogil=True)
        def loop_kji(Z, Y, X, mask, mask_nonzero_indices_matrix, n_x, pixel_vals):
            nz, ny, nx = Z.shape
            mz, my, mx = mask.shape
            csr_row_tmp = np.zeros(nz * ny * nx * 8, dtype=np.float32)
            csr_col_tmp = np.zeros(nz * ny * nx * 8, dtype=np.float32)
            csr_data_tmp = np.zeros(nz * ny * nx * 8, dtype=np.float32)
            b_tmp = np.zeros(nz * ny, dtype=np.float32)
            b_pid_tmp = np.zeros(nz * ny, dtype=np.int32)
            csr_rc_tmp_count = 0
            row_count_tmp = 0
            for k in range(nz):  # old x axis before back projection
                for j in range(ny):  # same y axis before back projection
                    row_tmp = {}
                    has_projection_data = False
                    for i in range(nx):  # old z axis before back projection
                        zi = int(Z[k, j, i])
                        yi = int(Y[k, j, i])
                        xi = int(X[k, j, i])
                        if zi < 0 or zi > mz - 1:
                            continue
                        if yi < 0 or yi > my - 1:
                            continue
                        if xi < 0 or xi > mx - 1:
                            continue
                        if zi + 1 > mz - 1 or yi + 1 > my - 1 or xi + 1 > mx - 1:
                            continue
                        if not mask[zi, yi, xi]:
                            continue
                        if not mask[zi + 1, yi, xi]:
                            continue
                        if not mask[zi, yi + 1, xi]:
                            continue
                        if not mask[zi + 1, yi + 1, xi]:
                            continue
                        if not mask[zi, yi, xi + 1]:
                            continue
                        if not mask[zi + 1, yi, xi + 1]:
                            continue
                        if not mask[zi, yi + 1, xi + 1]:
                            continue
                        if not mask[zi + 1, yi + 1, xi + 1]:
                            continue
                        index_000 = mask_nonzero_indices_matrix[zi, yi, xi]
                        if index_000 < 0 or index_000 > n_x - 1:
                            continue
                        index_001 = mask_nonzero_indices_matrix[zi, yi, xi + 1]
                        if index_001 < 0 or index_001 > n_x - 1:
                            continue
                        index_010 = mask_nonzero_indices_matrix[zi, yi + 1, xi]
                        if index_010 < 0 or index_010 > n_x - 1:
                            continue
                        index_011 = mask_nonzero_indices_matrix[zi, yi + 1, xi + 1]
                        if index_011 < 0 or index_011 > n_x - 1:
                            continue
                        index_100 = mask_nonzero_indices_matrix[zi + 1, yi, xi]
                        if index_100 < 0 or index_100 > n_x - 1:
                            continue
                        index_101 = mask_nonzero_indices_matrix[zi + 1, yi, xi + 1]
                        if index_101 < 0 or index_101 > n_x - 1:
                            continue
                        index_110 = mask_nonzero_indices_matrix[zi + 1, yi + 1, xi]
                        if index_110 < 0 or index_110 > n_x - 1:
                            continue
                        index_111 = mask_nonzero_indices_matrix[zi + 1, yi + 1, xi + 1]
                        if index_111 < 0 or index_111 > n_x - 1:
                            continue
                        has_projection_data = True

                        zf = Z[k, j, i] - zi
                        yf = Y[k, j, i] - yi
                        xf = X[k, j, i] - xi
                        for index in (
                            index_000,
                            index_001,
                            index_010,
                            index_011,
                            index_100,
                            index_101,
                            index_110,
                            index_111,
                        ):
                            if index not in row_tmp:
                                row_tmp[index] = 0.0
                        row_tmp[index_000] += (1 - zf) * (1 - yf) * (1 - xf)
                        row_tmp[index_001] += (1 - zf) * (1 - yf) * (xf)
                        row_tmp[index_010] += (1 - zf) * (yf) * (1 - xf)
                        row_tmp[index_011] += (1 - zf) * (yf) * (xf)
                        row_tmp[index_100] += (zf) * (1 - yf) * (1 - xf)
                        row_tmp[index_101] += (zf) * (1 - yf) * (xf)
                        row_tmp[index_110] += (zf) * (yf) * (1 - xf)
                        row_tmp[index_111] += (zf) * (yf) * (xf)
                    if has_projection_data:
                        for index in row_tmp:
                            csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                            csr_col_tmp[csr_rc_tmp_count] = index
                            csr_data_tmp[csr_rc_tmp_count] = row_tmp[index]
                            csr_rc_tmp_count += 1
                        b_tmp[row_count_tmp] = pixel_vals[j, k]
                        b_pid_tmp[row_count_tmp] = k * ny + j
                        row_count_tmp += 1
            return (
                csr_row_tmp[:csr_rc_tmp_count],
                csr_col_tmp[:csr_rc_tmp_count],
                csr_data_tmp[:csr_rc_tmp_count],
                b_tmp[:row_count_tmp],
                b_pid_tmp[:row_count_tmp],
            )

    else:  # nearest neighbor

        @jit(nopython=True, cache=True, nogil=True)
        def loop_kji(Z, Y, X, mask, mask_nonzero_indices_matrix, n_x, pixel_vals):
            nz, ny, nx = Z.shape
            mz, my, mx = mask.shape
            csr_row_tmp = np.zeros(nz * ny * nx, dtype=np.float32)
            csr_col_tmp = np.zeros(nz * ny * nx, dtype=np.float32)
            csr_data_tmp = np.ones(nz * ny * nx, dtype=np.float32)
            b_tmp = np.zeros(nz * ny, dtype=np.float32)
            b_pid_tmp = np.zeros(nz * ny, dtype=np.int32)
            csr_rc_tmp_count = 0
            row_count_tmp = 0
            for k in range(nz):  # old x axis before back projection
                for j in range(ny):  # same y axis before back projection
                    has_projection_data = False
                    for i in range(nx):  # old z axis before back projection
                        zi = round(Z[k, j, i])
                        yi = round(Y[k, j, i])
                        xi = round(X[k, j, i])
                        if zi < 0 or zi > mz - 1:
                            continue
                        if yi < 0 or yi > my - 1:
                            continue
                        if xi < 0 or xi > mx - 1:
                            continue
                        if not mask[zi, yi, xi]:
                            continue
                        index = mask_nonzero_indices_matrix[zi, yi, xi]
                        if index < 0 or index > n_x - 1:
                            continue
                        has_projection_data = True
                        csr_row_tmp[csr_rc_tmp_count] = row_count_tmp
                        csr_col_tmp[csr_rc_tmp_count] = index
                        csr_rc_tmp_count += 1
                    if has_projection_data:
                        b_tmp[row_count_tmp] = pixel_vals[j, k]
                        b_pid_tmp[row_count_tmp] = k * ny + j
                        row_count_tmp += 1
            return (
                csr_row_tmp[:csr_rc_tmp_count],
                csr_col_tmp[:csr_rc_tmp_count],
                csr_data_tmp[:csr_rc_tmp_count],
                b_tmp[:row_count_tmp],
                b_pid_tmp[:row_count_tmp],
            )

    hsym_max = max(1, int(np.ceil(reconstruct_length_3d_pixel + nz) / 2 / rise_pixel))
    hsyms = range(-hsym_max, hsym_max + 1)
    csyms = range(csym)
    from itertools import product, combinations

    hcsyms = list(product(hsyms, csyms))
    hcsyms.sort(key=lambda x: (abs(x[0]), x[1]))
    from scipy.stats import qmc

    qmc_method = qmc.Halton(d=1, scramble=False)
    n = len(hcsyms)
    indices = qmc_method.integers(l_bounds=0, u_bounds=n, n=n)
    hcsyms = [hcsyms[int(i[0])] for i in indices]
    for hci, (hi, ci) in enumerate(hcsyms):
        angle = twist_degree * hi + 360 * ci / csym
        r = R.from_euler("z", angle, degrees=True)
        coords = r.apply(coords0, inverse=True)
        coords[:, 2] -= hi * rise_pixel
        X = coords[:, 0].reshape((nz, ny, nx)) + nx // 2  # axes order: z, y, x
        Y = coords[:, 1].reshape((nz, ny, nx)) + ny // 2  # axes order: z, y, x
        Z = (
            coords[:, 2].reshape((nz, ny, nx)) + reconstruct_length_3d_pixel // 2
        )  # axes order: z, y, x

        csr_row_tmp, csr_col_tmp, csr_data_tmp, b_tmp, b_pid_tmp = loop_kji(
            Z, Y, X, mask, mask_nonzero_indices_matrix, n_x, pixel_vals
        )
        n_b += len(b_tmp)
        if verbose > 20:
            print(
                f"{hci+1}/{len(hcsyms)}: {hi=} {ci=} +{len(b_tmp):,} {n_b:,} target_lines={min_projection_lines:,}"
            )
        if len(b_tmp):
            # assert max(csr_row_tmp)+1 == len(b_tmp)
            # assert min(csr_col_tmp) >= 0
            # assert max(csr_col_tmp) < n_x
            b_tmp = np.array(b_tmp, dtype=np.float32)
            csr_A_tmp = csr_matrix(
                (csr_data_tmp, (csr_row_tmp, csr_col_tmp)),
                shape=(len(b_tmp), n_x),
                dtype=np.float32,
            )
            csr_A.append(csr_A_tmp)
            csr_b.append(b_tmp)
            b_pid += [b_pid_tmp]
        if min_projection_lines > 0 and n_b > min_projection_lines:
            break
    from scipy.sparse import vstack

    A = vstack(csr_A)
    b = np.concatenate(csr_b, dtype=np.float32)
    b_pid = np.concatenate(b_pid)
    return A, b, b_pid


def back_project_2d_coords_to_3d_coords(
    image,
    scale2d_to_3d,
    reconstruct_diameter_2d_pixel=-1,
    reconstruct_length_2d_pixel=-1,
):
    ny, nx = image.shape
    if reconstruct_diameter_2d_pixel <= 0:
        reconstruct_diameter_2d_pixel = ny
    if reconstruct_length_2d_pixel <= 0:
        reconstruct_length_2d_pixel = nx  # direction of helical axis

    reconstruct_diameter_2d_pixel = int(np.rint(reconstruct_diameter_2d_pixel))
    reconstruct_length_2d_pixel = int(np.rint(reconstruct_length_2d_pixel))

    k = (
        np.arange(0, reconstruct_diameter_2d_pixel, dtype=np.int32)
        - reconstruct_diameter_2d_pixel // 2
    )
    j = (
        np.arange(0, reconstruct_diameter_2d_pixel, dtype=np.int32)
        - reconstruct_diameter_2d_pixel // 2
    )
    i = (
        np.arange(0, reconstruct_length_2d_pixel, dtype=np.int32)
        - reconstruct_length_2d_pixel // 2
    )
    region_pixel_vals = image[
        np.ix_(j + ny // 2, i + nx // 2)
    ]  # pixel values to be used for lsq solution

    from scipy.spatial.transform import Rotation as R

    r = R.from_euler("y", 90, degrees=True)
    Z, Y, X = np.meshgrid(
        k.astype(np.float32), j.astype(np.float32), i.astype(np.float32), indexing="ij"
    )
    coords = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).transpose()
    coords = r.apply(coords, inverse=True)
    if scale2d_to_3d != 1.0:
        coords *= scale2d_to_3d
    X2 = coords[:, 0].reshape(
        (
            reconstruct_diameter_2d_pixel,
            reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel,
        )
    )  # axes order: z, y, x
    Y2 = coords[:, 1].reshape(
        (
            reconstruct_diameter_2d_pixel,
            reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel,
        )
    )  # axes order: z, y, x
    Z2 = coords[:, 2].reshape(
        (
            reconstruct_diameter_2d_pixel,
            reconstruct_diameter_2d_pixel,
            reconstruct_length_2d_pixel,
        )
    )  # axes order: z, y, x
    # axes change after 90 degree rotation around +y axis: x -> z, y -> y, z -> x
    X2 = np.swapaxes(X2, 0, 2)  # new axes order: z', y, x'
    Y2 = np.swapaxes(Y2, 0, 2)  # new axes order: z', y, x'
    Z2 = np.swapaxes(Z2, 0, 2)  # new axes order: z', y, x'
    assert X2[:, :, 0].shape[::-1] == region_pixel_vals.shape
    return (X2, Y2, Z2), region_pixel_vals


def simulate_helical_projection(
    n,
    twist,
    rise,
    csym,
    helical_diameter,
    ball_radius,
    polymer,
    planarity,
    ny,
    nx,
    apix,
    tilt=0,
    rot=0,
    psi=0,
    dy=0,
):
    assert helical_diameter + ball_radius < ny * apix * 0.99
    import numpy as np

    def simulate_projection(centers, sigma, ny, nx, apix):
        sigma2 = sigma * sigma / np.log(2)
        d = np.zeros((ny, nx))
        Y, X = np.meshgrid(
            np.arange(0, ny, dtype=np.float32) - ny // 2,
            np.arange(0, nx, dtype=np.float32) - nx // 2,
            indexing="ij",
        )
        X *= apix
        Y *= apix
        for ci in range(len(centers)):
            yc, xc = centers[ci]
            x = X - xc
            y = Y - yc
            d += np.exp(-(x * x + y * y) / sigma2)
        return d

    def helical_unit_positions(
        n,
        twist,
        rise,
        csym,
        diameter,
        height,
        polymer=0,
        planarity=1.0,
        tilt=0,
        rot=0,
        psi=0,
        dy=0,
    ):
        assert n >= 1
        from scipy.spatial.transform import Rotation as R

        if polymer:
            centers_0 = random_polymer(
                n_atoms=n,
                rmin=0,
                rmax=helical_diameter / 2,
                csym=csym,
                planarity=planarity,
            )
            rot = R.from_euler("y", 90, degrees=True)
            centers_0 = rot.apply(centers_0)
            centers_0 = centers_0[:, [2, 1, 0]]  # axes order: x,y,z -> z,y,x
            n = len(centers_0)  # in case that a polymer with fewer atoms is returned
        else:
            centers_0 = np.zeros((n, 3), dtype=np.float32)
            if n > 1:
                r = np.sqrt(np.random.uniform(0, diameter**2 / 4, n))
                angle = np.random.uniform(-np.pi, np.pi, n) + np.deg2rad(rot)
                centers_0[:, 0] = r * np.cos(angle)
                centers_0[:, 1] = r * np.sin(angle)
                centers_0[:, 2] = np.random.uniform(-rise / 2, rise / 2, n)
            else:
                angle = np.deg2rad(rot)  # start from +z axis
                z = np.cos(angle) * diameter / 2
                y = np.sin(angle) * diameter / 2
                centers_0[0, 0] = z
                centers_0[0, 1] = y
                centers_0[0, 2] = 0

        imax = int(np.ceil(height / rise))
        i0 = -imax
        i1 = imax
        centers = np.zeros(((2 * imax + 1) * csym * n, 3), dtype=np.float32)

        index = 0
        for i in range(i0, i1 + 1):
            for si in range(csym):
                angle = twist * i + si * 360.0 / csym
                rot = R.from_euler("z", angle, degrees=True)
                centers[index : index + n, :] = rot.apply(centers_0)
                centers[index : index + n, 2] += i * rise
                index += n
        if tilt or psi:
            rot = R.from_euler("yx", (tilt, -psi), degrees=True)
            centers = rot.apply(centers)
        if dy:
            centers[:, 1] += dy
        centers_2d = centers[:, [1, 2]]  # project along z
        return centers_2d

    centers = helical_unit_positions(
        n,
        twist,
        rise,
        csym,
        helical_diameter,
        height=nx * apix,
        polymer=polymer,
        planarity=planarity,
        tilt=tilt,
        rot=rot,
        psi=psi,
        dy=dy,
    )
    projection = simulate_projection(centers, ball_radius, ny, nx, apix)
    return projection


def random_polymer(n_atoms=100, rmin=0, rmax=100, csym=1, planarity=0.9):
    import numpy as np

    def symmetrize(p, csym=1):
        if csym <= 1:
            return np.expand_dims(p, axis=0)
        from scipy.spatial.transform import Rotation as R

        ret = [p]
        for si in range(1, csym):
            rot = R.from_euler("z", si * 360 / csym, degrees=True)
            ps = rot.apply(p)
            ret.append(ps)
        ret = np.vstack(ret)
        return ret

    def are_positions_good(new_points, existing_points, min_dist):
        def pairwise_distances(points_a, points_b):
            differences = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
            squared_distances = np.sum(differences**2, axis=-1)
            distances = np.sqrt(squared_distances)
            return distances

        if len(new_points) > 1:
            dist = pairwise_distances(new_points, new_points)
            dist[np.diag_indices_from(dist)] = 1e10
            if np.any(dist < min_dist):
                return False
        dist = pairwise_distances(new_points, existing_points)
        if new_points.shape == existing_points.shape and np.allclose(
            new_points, existing_points
        ):
            dist[np.diag_indices_from(dist)] = 1e10
        if np.any(dist < min_dist):
            return False
        return True

    def next_point(step_length, csym, rmin, rmax, planarity, existing_points):
        n_trials = 1
        while True:
            angle_out_plane_max = 90 * (1 - planarity)  # planarity should be in [0, 1]
            sigma_z = np.abs(np.random.normal(0, angle_out_plane_max / 3))
            sigma_xy = 180 / 3
            if len(existing_points) < 2:
                d0 = existing_points[-1, :] * 0
            else:
                d0 = existing_points[-1, :] - existing_points[-2, :]
                d0 /= np.linalg.norm(d0)
                d0 /= n_trials
                r = np.linalg.norm(existing_points[-1, :])
                d0 *= (rmax - r) / rmax
            d = np.random.normal(0, (sigma_xy, sigma_xy, sigma_z))
            d /= np.linalg.norm(d)
            d = (d0 + d) / np.linalg.norm(d0 + d)
            p = existing_points[-1, :] + step_length * d
            r = np.linalg.norm(p)
            if rmin <= r <= rmax or n_trials > 10:
                break
            n_trials += 1  # avoid dead loop
        p = symmetrize(p, csym)
        return p

    ca_dist = 3.8  # Angstrom

    n_good_points = 0

    max_trials = 10
    n_trials = 0
    while n_trials < max_trials:
        xyz = np.zeros([csym * n_atoms, 3], dtype=float)

        good_start_point_added = False
        ns_trials = 0
        while ns_trials < max_trials:
            r = np.sqrt(np.random.uniform(rmin**2, rmax**2))
            angle = np.random.uniform(-np.pi, np.pi)
            xyz[0, 0] = r * np.sin(angle)
            xyz[0, 1] = r * np.cos(angle)
            xyz[0, 2] = 0
            xyz[0:csym, :] = symmetrize(xyz[0, :], csym=csym)
            if are_positions_good(
                xyz[0:csym, :], xyz[0:csym, :], min_dist=ca_dist * 0.8
            ):
                good_start_point_added = True
                n_good_points = 1
                break
            ns_trials += 1

        if not good_start_point_added:
            n_trials += 1
            break

        for i in range(1, n_atoms):
            good_point_added = False
            ni_trials = 0
            while ni_trials < max_trials:
                existing_points = xyz[: i * csym, :]
                p = next_point(
                    step_length=ca_dist,
                    csym=csym,
                    rmin=rmin,
                    rmax=rmax,
                    planarity=planarity,
                    existing_points=existing_points,
                )
                if are_positions_good(p, existing_points, min_dist=ca_dist * 0.8):
                    xyz[i * csym : (i + 1) * csym, :] = p
                    good_point_added = True
                    n_good_points = i + 1
                    break
                ni_trials += 1
            if not good_point_added:
                break

        if n_good_points == n_atoms:
            break

        n_trials += 1

    return xyz[: n_good_points * csym, :]


def auto_horizontalize(data, refine=False):
    from skimage.transform import radon
    from scipy.interpolate import interp1d
    from scipy.signal import correlate

    data_work = np.clip(data, 0, None)

    theta, shift_y, diameter = helicon.estimate_helix_rotation_center_diameter(data)

    if refine:  # refine to sub-degree, sub-pixel level

        def score_rotation_shift(x):
            theta, shift_y = x
            data_tmp = helicon.rotate_shift_image(
                data_work, angle=theta, post_shift=(shift_y, 0)
            )
            # data_tmp /= np.linalg.norm(data_tmp, axis=0)
            y = np.sum(data_tmp, axis=1)[1:]
            y += y[::-1]
            score = -np.std(y)
            return score

        from scipy.optimize import fmin

        res = fmin(score_rotation_shift, x0=(theta, shift_y), xtol=1e-2, disp=0)
        theta, shift_y = res

    rotated_shifted_data = helicon.rotate_shift_image(
        data, angle=theta, post_shift=(shift_y, 0), order=3
    )
    return rotated_shifted_data, theta, shift_y


def is_vertical(data):
    py_max = np.max(np.sum(data, axis=0))
    px_max = np.max(np.sum(data, axis=1))
    if py_max > px_max:
        return True
    else:
        return False


def sorted_hsym_csym_pairs(twist, rise, csym, nz):
    hsym_max = max(1, int(np.ceil(nz / (2 * rise))))
    hsyms = range(-hsym_max, hsym_max + 1)
    csyms = range(csym)
    from itertools import product, combinations

    hcsyms = product(hsyms, csyms)
    hcsym_pairs = combinations(hcsyms, r=2)
    hcsym_pair_angles = []
    for p in hcsym_pairs:
        (hsym1, csym1), (hsym2, csym2) = p
        angle1 = twist * hsym1 + csym1 * 360 / csym
        angle2 = twist * hsym2 + csym2 * 360 / csym
        angle = round(abs((angle2 - angle1 + 180) % 360 - 180), 2)  # range: [0, 180]
        hcsym_pair_angles.append(
            (angle, abs(hsym1 + hsym2), abs(hsym1 - hsym2), abs(hsym1), abs(hsym2), p)
        )
    hcsym_pair_angles.sort(key=lambda x: x[:-1])
    from scipy.stats import qmc

    qmc_method = qmc.Halton(d=1, scramble=False)
    n = len(hcsym_pair_angles)
    indices = qmc_method.integers(l_bounds=0, u_bounds=n, n=n)
    ret = [hcsym_pair_angles[int(i[0])] for i in indices]
    return ret


def saveMaps(results, target_size, target_apix, mapFilePrefix, verbose=0, logger=None):
    logger = logger
    mapFiles = []
    result_groups = itertools.groupby(
        results, key=lambda x: x[2][1:3]
    )  # group by input (imageFile, imageIndex)
    for group_key, result_group in result_groups:
        result_group = sorted(result_group, key=lambda x: x[0], reverse=True)
        for ri, result in enumerate(result_group):
            (
                score,
                (
                    rec3d_x_proj,
                    rec3d_y_proj,
                    rec3d_z_sections,
                    rec3d,
                    reconstruct_diameter_2d_pixel,
                    reconstruct_diameter_3d_pixel,
                    reconstruct_length_2d_pixel,
                    reconstruct_length_3d_pixel,
                ),
                (
                    data,
                    imageFile,
                    imageIndex,
                    apix3d,
                    apix2d,
                    twist,
                    rise,
                    csym,
                    tilt,
                    psi,
                    dy,
                ),
            ) = result
            mapFilePrefix_tmp = (
                mapFilePrefix
                + f".{pathlib.Path(imageFile).stem}-{imageIndex}.pitch-{round(rise*360/abs(twist),1)}_twist-{round(twist,3)}_rise-{rise}_csym-{csym}{tilt_psi_dy_str(tilt, psi, dy, sep='_', sep2='-', unit=False)}"
            )
            if rec3d is not None:
                fsc_test = rec3d[1] is not None and rec3d[2] is not None
                for rec3d_i, rec3d_map in enumerate(rec3d):
                    if rec3d_map is None:
                        continue
                    if len(list(result_group)) > 1:
                        tag = f".top-{ri+1}"
                    else:
                        tag = ""
                    if fsc_test:
                        set_tag = ["full", "half_1", "half_2"]
                        tag += f".{set_tag[rec3d_i]}"
                    mapFile = mapFilePrefix_tmp + f"{tag}.mrc"
                    mapFiles.append(mapFile)
                    with helicon.Timer(
                        "Applying helical symmetry ...", verbose=verbose > 10
                    ):
                        cur_size = rec3d_map.shape[0]
                        logger.info(
                            f"Applying helical symmetry to extend the map from the reconstructed length {cur_size*apix3d} Å (={cur_size} pixels * {apix3d} Å/pixel) to {target_size[0] * target_apix} Å (={target_size[0]} pixels * {target_apix} Å/pixel)"
                        )
                        cpu = helicon.available_cpu()
                        rec3d_map = helicon.apply_helical_symmetry(
                            data=rec3d_map,
                            apix=apix3d,
                            twist_degree=twist,
                            rise_angstrom=rise,
                            csym=csym,
                            fraction=1.0,
                            new_size=target_size,
                            new_apix=target_apix,
                            cpu=cpu,
                        )
                    import mrcfile

                    with mrcfile.new(str(mapFile), overwrite=True) as mrc:
                        mrc.set_data(rec3d_map.astype(np.float32))
                        mrc.voxel_size = target_apix
    return mapFiles


def writePdf(
    results, pdfFilePrefix, top_k=10, use_pitch=None, image_info="", cmap=None
):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({"figure.max_open_warning": 0})
    result_groups = itertools.groupby(
        results, key=lambda x: x[2][1:3]
    )  # group by input (imageFile, imageIndex)
    pdfFiles = []
    fscs = []
    for (imageFile, imageIndex), group_results in result_groups:
        pdfFile = pdfFilePrefix + f".{pathlib.Path(imageFile).stem}-{imageIndex}.pdf"
        pdfFiles.append(pdfFile)
        pdf = PdfPages(pdfFile)

        group_results = sorted(group_results, key=lambda x: x[0], reverse=True)

        if len(group_results) > 1:
            figs = plotOneGroupScores(group_results, image_info, use_pitch)
            for fig in figs:
                pdf.attach_note(fig._suptitle.get_text())
                pdf.savefig(fig)
                plt.close(fig)

        for ri, result in enumerate(group_results[: top_k if top_k > 0 else None]):
            (
                score,
                (
                    rec3d_x_proj,
                    rec3d_y_proj,
                    rec3d_z_sections,
                    rec3d,
                    reconstruct_diameter_2d_pixel,
                    reconstruct_diameter_3d_pixel,
                    reconstruct_length_2d_pixel,
                    reconstruct_length_3d_pixel,
                ),
                (
                    data,
                    imageFile,
                    imageIndex,
                    apix3d,
                    apix2d,
                    twist,
                    rise,
                    csym,
                    tilt,
                    psi,
                    dy,
                ),
            ) = result
            fig = plotOneResult(
                ri if len(group_results) > 1 else -1,
                score,
                data,
                rec3d_y_proj,
                rec3d_x_proj,
                rec3d_z_sections,
                reconstruct_diameter_2d_pixel,
                reconstruct_diameter_3d_pixel,
                reconstruct_length_2d_pixel,
                reconstruct_length_3d_pixel,
                imageFile,
                imageIndex,
                apix3d,
                apix2d,
                twist,
                rise,
                csym,
                tilt,
                psi,
                dy,
                image_info,
                cmap=cmap,
            )
            pdf.attach_note(fig._suptitle.get_text())
            pdf.savefig(fig)
            plt.close(fig)

            if rec3d is not None:
                fsc_test = (rec3d[1] is not None) and (rec3d[2] is not None)
                if fsc_test:
                    map1, map2 = rec3d[1:]
                    shape = [max(map1.shape) + (1 if max(map1.shape) % 2 else 0)] * 3
                    fsc = helicon.calc_fsc(
                        helicon.pad_to_size(map1, shape),
                        helicon.pad_to_size(map2, shape),
                        apix3d,
                    )
                    if len(group_results) > 1:
                        tag = f".top-{ri+1}"
                    else:
                        tag = ""
                    fscFilePrefix = (
                        pdfFilePrefix
                        + f".{pathlib.Path(imageFile).stem}-{imageIndex}{tag}.pitch-{round(rise*360/abs(twist),1)}_twist-{round(twist,3)}_rise-{rise}"
                    )
                    fscFile = f"{fscFilePrefix}.fsc.txt"
                    np.savetxt(fscFile, fsc)

                    fscs.append((fsc, fscFilePrefix))

                    import matplotlib.pyplot as plt

                    figsize = 8
                    fig = plt.figure(figsize=(figsize, figsize))
                    fig.suptitle(
                        fscFilePrefix,
                        fontsize=1.5 * round(figsize * 72 / len(fscFilePrefix)),
                    )
                    plt.title(f"Resolution={round(helicon.get_resolution(fsc), 2)}Å")
                    plt.plot(fsc[:, 0], fsc[:, 1])
                    plt.axhline(y=0.143, color="r", linestyle="dashed")
                    plt.xlim(0, 1 / (2 * apix3d))
                    plt.gca().set_ylim(bottom=min(0, plt.gca().get_ylim()[0]))
                    plt.gca().set_ylim(top=1)
                    plt.xlabel("Resolution (1/Å)")
                    plt.ylabel("Fourier Shell Correlation")
                    plt.grid(linestyle="dashed")
                    pdf.attach_note(fig._suptitle.get_text())
                    pdf.savefig(fig)
                    plt.close(fig)
        pdf.close()
    if len(fscs) > 1:
        import matplotlib.pyplot as plt

        figsize = 8
        fig = plt.figure(figsize=(figsize, figsize))
        for fsc, label in fscs:
            plt.plot(fsc[:, 0], fsc[:, 1], label=label)
        plt.xlim(0, 1 / (2 * apix3d))
        plt.gca().set_ylim(bottom=min(0, plt.gca().get_ylim()[0]))
        plt.gca().set_ylim(top=1)
        plt.xlabel("Spatial Frequency (1/Å)")
        plt.ylabel("Fourier Shell Correlation")
        plt.grid(linestyle="dashed")
        fontsize = 7
        ncol = int(round(figsize * 72 / (fontsize * len(label))))
        nrow = len(fig.gca().lines) // ncol + (1 if len(fig.gca().lines) % ncol else 0)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02 + 0.025 * nrow),
            ncol=ncol,
            fontsize=fontsize,
        )
        pdfFile = f"{pdfFilePrefix}.fsc.pdf"
        pdfFiles.append(pdfFile)
        plt.savefig(pdfFile, format="pdf")
    return pdfFiles


def plotOneResult(
    rank,
    score,
    data,
    rec3d_y_proj,
    rec3d_x_proj,
    rec3d_z_sections,
    reconstruct_diameter_2d_pixel,
    reconstruct_diameter_3d_pixel,
    reconstruct_length_2d_pixel,
    reconstruct_length_3d_pixel,
    imageFile,
    imageIndex,
    apix3d,
    apix2d,
    twist,
    rise,
    csym,
    tilt,
    psi,
    dy,
    image_info,
    cmap=None,
):

    ny, nx = data.shape
    nx_pad = max(reconstruct_length_2d_pixel, nx)
    images = [
        helicon.pad_to_size(
            helicon.crop_center(
                data, shape=(reconstruct_diameter_2d_pixel, reconstruct_length_2d_pixel)
            ),
            shape=(reconstruct_diameter_2d_pixel, nx_pad),
        )
    ]
    images += [
        helicon.crop_center(
            helicon.pad_to_size(image, shape=(reconstruct_diameter_2d_pixel, nx_pad)),
            shape=(reconstruct_diameter_2d_pixel, nx_pad),
        )
        for image in (rec3d_y_proj, rec3d_x_proj, rec3d_z_sections)
    ]

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    title = f"{pathlib.Path(imageFile).stem}-{imageIndex}: {image_info}\nreconstruction: {reconstruct_diameter_3d_pixel}x{reconstruct_diameter_3d_pixel}x{reconstruct_length_3d_pixel}voxels@{round(apix3d, 3)}Å/voxel\npitch={round(rise*360/abs(twist),1)}Å/twist={round(twist, 3)}° rise={round(rise, 3)}Å {csym=}{tilt_psi_dy_str(tilt, psi, dy)} ⇒ {score=:.6f}"
    if rank >= 0:
        title += f" rank={rank+1}"
    plot_extra_row = rec3d_x_proj.shape[1] > nx
    fig = plt.figure(figsize=(8, 9))
    fig.suptitle(title, wrap=True)
    gs = gridspec.GridSpec(3 if plot_extra_row else 2, 2)
    for i in range(4):
        r = i // 2
        c = i % 2
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(images[i], cmap=cmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.axes[0].set_title(f"{pathlib.Path(imageFile).stem}-{imageIndex}")
    fig.axes[1].set_title("Projection-Y")
    fig.axes[2].set_title(
        "Projection-X", y=-0.1 * nx_pad / reconstruct_diameter_2d_pixel
    )
    fig.axes[3].set_title(
        "Z central section", y=-0.1 * nx_pad / reconstruct_diameter_2d_pixel
    )
    if plot_extra_row:
        ax = fig.add_subplot(gs[2, :])
        ax.imshow(rec3d_x_proj, cmap=cmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(
            "Projection-X (length = 1.2 pitch)",
            y=-0.04 * rec3d_x_proj.shape[1] / reconstruct_diameter_2d_pixel,
        )
    plt.tight_layout()
    return fig


def plotOneGroupScores(results, image_info, use_pitch=None):
    scores = []
    for result in results:
        (
            score,
            _,
            (
                _,
                imageFile,
                imageIndex,
                apix3d,
                apix2d,
                twist,
                rise,
                csym,
                tilt,
                psi,
                dy,
            ),
        ) = result
        scores.append([twist, rise, tilt, psi, dy, score])
    if use_pitch:
        scores = [
            [np.round(360 / np.abs(twist) * rise, 1), rise, tilt, psi, dy, score]
            for twist, rise, tilt, psi, dy, score in scores
        ]
    scores.sort()
    scores = np.array(scores)

    rises = scores[:, 1]
    if use_pitch:
        pitches = scores[:, 0]
        twists = np.round(360 / (pitches / rises), 3)
    else:
        twists = scores[:, 0]
        pitches = np.round(360 / np.abs(twists) * rises, 1)
    tilts = scores[:, 2]
    psis = scores[:, 3]
    dys = scores[:, 4]
    scores = scores[:, 5]

    best_index = np.argmax(scores)
    title = f"{pathlib.Path(imageFile).stem}-{imageIndex}: {image_info}\nbest pitch={pitches[best_index]}Å/twist={round(twists[best_index], 3)}° rise={round(rises[best_index], 3)}Å {csym=}{tilt_psi_dy_str(tilts[best_index], psis[best_index], dys[best_index])} ⇒ score={round(scores[best_index], 6)}"

    n_tilts = len(np.unique(tilts))
    n_psis = len(np.unique(psis))
    n_dys = len(np.unique(dys))
    n_rises = len(np.unique(rises))
    if use_pitch is None:
        sigma_pitch_steps = np.std(np.ediff1d(np.sort(np.unique(pitches))))
        sigma_twist_steps = np.std(np.ediff1d(np.sort(np.unique(twists))))
        use_pitch = np.isnan(sigma_pitch_steps) or sigma_pitch_steps < sigma_twist_steps
    if use_pitch:
        pts = pitches
        pts_label = "pitch (Å)"
        n_pts = len(np.unique(pitches))
    else:
        pts = twists
        pts_label = "twist (°)"
        n_pts = len(np.unique(twists))
    vars = [
        (n_pts, pts, pts_label),
        (n_rises, rises, "rise (Å)"),
        (n_tilts, tilts, "tilt (°)"),
        (n_psis, psis, "psi (°)"),
        (n_dys, dys, "dy (Å)"),
    ]
    vars_2 = [var for var in vars if var[0] > 1]
    figs = []
    if len(vars_2) > 1:
        from itertools import combinations

        pairs = list(combinations(vars_2, r=2))
        vmin = np.min(scores)
        vmax = np.max(scores)
        for pi, pair in enumerate(pairs):
            (_, x, xlabel), (_, y, ylabel) = pair
            x_unique = np.unique(x)
            y_unique = np.unique(y)
            Y, X = np.meshgrid(y_unique, x_unique, indexing="ij")
            Z = np.zeros_like(X) + vmin
            for j in range(Z.shape[0]):
                for i in range(Z.shape[1]):
                    vals = []
                    for si in range(len(x)):
                        if y[si] == Y[j, i] and x[si] == X[j, i]:
                            vals.append(scores[si])
                    if len(vals):
                        Z[j, i] = np.max(vals)
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(8, 8))
            fig.suptitle(title, wrap=True)
            ax = fig.add_subplot(111)
            pcm = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, shading="gouraud")
            fig.colorbar(pcm)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            figs.append(fig)
    for var in vars:
        _, x, xlabel = var
        unique_x = np.sort(np.unique(x))
        if len(unique_x) <= 1:
            continue
        y = [np.max(scores[x == tmp_x]) for tmp_x in unique_x]
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 8))
        fig.suptitle(title, wrap=True)
        ax = fig.add_subplot(111)
        ax.plot(unique_x, y, marker=".")
        ax.minorticks_on()
        ax.grid(True, which="both", linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("score")
        figs.append(fig)
    return figs


def plotAllScores(results, pdfFilePrefix, use_pitch=None):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdfFile = pdfFilePrefix + f".scores.pdf"
    pdf = PdfPages(pdfFile)
    figsize = 8  # inch
    fig = plt.figure(figsize=(figsize, figsize))

    result_groups = itertools.groupby(
        results, key=lambda x: x[2][1:3]
    )  # group by input (imageFile, imageIndex)
    for (imageFile, imageIndex), group_results in result_groups:
        scores = []
        for result in group_results:
            (
                score,
                _,
                (
                    _,
                    imageFile,
                    imageIndex,
                    apix3d,
                    apix2d,
                    twist,
                    rise,
                    csym,
                    tilt,
                    psi,
                    dy,
                ),
            ) = result
            if use_pitch:
                pitch = np.round(360 / np.abs(twist) * rise, 1)
                scores.append([pitch, rise, tilt, psi, dy, score])
            else:
                scores.append([twist, rise, tilt, psi, dy, score])
        scores.sort()
        scores = np.array(scores)

        pts = scores[:, 0]
        rises = scores[:, 1]
        tilts = scores[:, 2]
        psis = scores[:, 3]
        dys = scores[:, 4]
        scores = scores[:, 5]

        n_pts = len(np.unique(pts))
        n_rises = len(np.unique(rises))
        n_tilts = len(np.unique(tilts))
        n_psis = len(np.unique(psis))
        n_dys = len(np.unique(dys))
        ns = np.array([n_pts, n_rises, n_tilts, n_psis, n_dys])
        if np.count_nonzero(ns > 1) > 1 or np.max(ns) <= 1:
            return None

        label = f"{pathlib.Path(imageFile).stem}-{imageIndex}"
        if n_pts > 1:
            plt.plot(pts, scores, marker=".", label=label)
            if use_pitch:
                xlabel = "pitch (Å)"
            else:
                xlabel = "twist (°)"
        elif n_rises > 1:
            plt.plot(rises, scores, marker=".", label=label)
            xlabel = "rise (Å)"
        elif n_tilts > 1:
            plt.plot(tilts, scores, marker=".", label=label)
            xlabel = "tilt (°)"
        elif n_psis > 1:
            plt.plot(psis, scores, marker=".", label=label)
            xlabel = "psi (°)"
        else:
            plt.plot(dys, scores, marker=".", label=label)
            xlabel = "dy (Å)"
    plt.minorticks_on()
    plt.grid(True, which="both", linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel("score")
    fontsize = 7
    ncol = int(round(figsize * 72 / (fontsize * len(label))))
    nrow = len(fig.gca().lines) // ncol + (1 if len(fig.gca().lines) % ncol else 0)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02 + 0.025 * nrow),
        ncol=ncol,
        fontsize=fontsize,
    )
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)
    return pdfFile


def writeLstFile(results, top_k, apix2d_orig, lstFilePrefix):
    import os

    sep = "\t"
    lstFiles = []
    lstFileFolder = pathlib.Path(lstFilePrefix).absolute().parent
    result_groups = itertools.groupby(
        results, key=lambda x: x[2][1:3]
    )  # group by input (imageFile, imageIndex)
    for (imageFile, imageIndex), group_results in result_groups:
        lstFile = lstFilePrefix + f".{pathlib.Path(imageFile).stem}-{imageIndex}.lst"
        lstFiles.append(lstFile)
        group_results = sorted(group_results, key=lambda x: x[0], reverse=True)
        with open(lstFile, "w") as fp:
            fp.write("#LST\n")
            for ri, result in enumerate(group_results):
                (
                    score,
                    (rec3d_x_proj, rec3d_y_proj, rec3d_z_sections, _, _, _, _, _),
                    (
                        data,
                        imageFile,
                        imageIndex,
                        apix3d,
                        apix2d,
                        twist,
                        rise,
                        csym,
                        tilt,
                        psi,
                        dy,
                    ),
                ) = result
                imageFilePath = pathlib.Path(imageFile)
                if imageFilePath.is_absolute():
                    imageFilePath_final = imageFilePath
                else:
                    imageFilePath_final = pathlib.Path(
                        os.path.relpath(
                            str(imageFilePath.absolute()), str(lstFileFolder)
                        )
                    )
                n = top_k if top_k > 0 else len(group_results)
                if ri < n:
                    if imageFilePath.exists():
                        fp.write(
                            f"{imageIndex-1}\t{str(imageFilePath_final)}\tapix={round(apix2d_orig,3)}\n"
                        )
                    images = [
                        helicon.pad_to_size(rec3d_x_proj, data.shape),
                        helicon.pad_to_size(rec3d_y_proj, data.shape),
                        helicon.pad_to_size(
                            rec3d_z_sections, (data.shape[0], data.shape[0])
                        ),
                    ]
                    for ii, image in enumerate(images):
                        tag = ("projection-x", "projection-y", "z-central-section")[ii]
                        fname = (
                            lstFilePrefix
                            + f".{pathlib.Path(imageFile).stem}-{imageIndex}_pitch-{round(rise*360/abs(twist),1)}_twist-{round(twist,3)}_rise-{rise}_csym-{csym}{tilt_psi_dy_str(tilt, psi, dy, sep='_', sep2='-', unit=False)}.{tag}.mrcs"
                        )
                        with mrcfile.new(str(fname), overwrite=True) as mrc:
                            mrc.set_data(image.astype(np.float32))
                            mrc.voxel_size = apix3d
                        fname = str(
                            pathlib.Path(fname)
                            .resolve()
                            .relative_to(pathlib.Path(lstFileFolder).resolve())
                        )
                        fp.write(
                            f"0\t{fname}\ttag={tag}\tsrc_image={str(imageFilePath_final)}\tsrc_image_index={imageIndex-1}\tapix={round(apix2d,3)}\tpitch={round(rise*360/abs(twist),1)}\ttwist={round(twist, 3)}\trise={round(rise, 3)}\t{csym=}{tilt_psi_dy_str(tilt, psi, dy, sep=sep, unit=False)}\tscore={round(score, 6)}\n"
                        )
                else:
                    fp.write(
                        f"0\t{str(imageFilePath_final)}\tsrc_image_index={imageIndex-1}\tpitch={round(rise*360/abs(twist),1)}\ttwist={round(twist, 3)}\trise={round(rise, 3)}\t{csym=}{tilt_psi_dy_str(tilt, psi, dy, sep=sep, unit=False)}\tscore={round(score, 6)}\n"
                    )
    return lstFiles


def tilt_psi_dy_str(tilt, psi, dy, sep=" ", sep2="=", unit=True):
    tpy_str = ""
    if tilt:
        tpy_str += f"{sep}tilt{sep2}{round(tilt, 2)}" + ("°" if unit else "")
    if psi:
        tpy_str += f"{sep}psi{sep2}{round(psi, 2)}" + ("°" if unit else "")
    if dy:
        tpy_str += f"{sep}dy{sep2}{round(dy, 2)}" + ("Å" if unit else "")
    return tpy_str


def star_to_dataframe(starFile, logger):
    df = helicon.star2dataframe(starFile=starFile)

    fileNameCol = ""
    for col in ["rlnImageName", "rlnReferenceImage"]:
        if col in df:
            fileNameCol = col
            break
    if not fileNameCol:
        logger.error(
            f"ERROR: cannot find 'rlnImageName' or 'rlnReferenceImage' in the input {starFile}"
        )
        sys.exit(-1)

    tmp = df[fileNameCol].str.split("@", expand=True)
    indices, filenames = tmp.iloc[:, 0], tmp.iloc[:, -1]
    indices = indices.astype(int) - 1
    df["pid"] = indices
    df["filename"] = filenames

    dir0 = pathlib.Path(starFile).parent
    mapping = {}
    for f in df["filename"].unique():
        if f in mapping:
            continue
        fp = pathlib.Path(f)
        name = fp.name
        choices = [
            fp,
            dir0 / fp,
            dir0 / ".." / fp,
            dir0 / "../.." / fp,
            dir0 / name,
            dir0 / ".." / name,
            dir0 / "../.." / name,
        ]
        for choice in choices:
            if choice.exists():
                mapping[f] = choice.resolve().as_posix()
                break
        if f in mapping:
            fp2 = pathlib.Path(mapping[f])
            for fo in fp2.parent.glob("*" + fp.suffix):
                ftmp = (fp.parent / fo.name).as_posix()
                mapping[ftmp] = fo.as_posix()
    for f in df["filename"].unique():
        if f not in mapping:
            logger.warning(f"WARNING: {f} is not accessible")
            mapping[f] = f
    df.loc[:, "filename"] = df.loc[:, "filename"].map(mapping)

    return df


helicon.import_with_auto_install(
    "numpy scipy matplotlib mrcfile skimage:scikit-image sklearn:scikit-learn joblib psutil tqdm".split()
)


def add_args(parser):
    parser.add_argument(
        "input_file",
        metavar="<filename>",
        type=str,
        nargs="?",
        help="Input STAR or mrc/mrcs file containing the input 2D class average image(s). default to %(default)s",
        default="Simulation:n=1:nx=128:ny=96:apix=5:twist=0.5:rise=4.75:csym=2:poylmer=0:rot=30:tilt=0:psi=0:noise=0",
    )
    parser.add_argument(
        "--output_prefix",
        metavar="<string>",
        type=str,
        nargs="?",
        help="Output rootname. default to %(default)s",
        default=f"HELICON/denovo3DBatch",
    )
    i_parser = parser.add_mutually_exclusive_group(required=False)
    i_parser.add_argument(
        "--i0",
        type=int,
        metavar="<n>",
        nargs="+",
        help="Which image to process. 0 (not 1) for the first image. EMAN/EMAN2 convention. default to all images",
        default=[],
    )
    i_parser.add_argument(
        "--i1",
        type=int,
        metavar="<n>",
        nargs="+",
        help="Which image to process. 1 (not 0) for the first image. RELION convention. default to all images",
        default=[],
    )

    pt_parser = parser.add_mutually_exclusive_group(required=True)
    pt_parser.add_argument(
        "--pitch",
        dest="pitches",
        metavar="<Å>",
        type=float,
        nargs="+",
        help="Use this pitch value (unit: Å)",
        default=[],
    )
    pt_parser.add_argument(
        "--pitch_range",
        metavar=("<min>", "<max>", "<step>"),
        type=float,
        nargs=3,
        action="append",
        help="Test pitch values in this range (unit: Å). default to %(default)s",
        default=[],
    )
    pt_parser.add_argument(
        "--twist",
        dest="twists",
        metavar="<°>",
        type=float,
        nargs="+",
        help="Use this twist value (unit: °)",
        default=[],
    )
    pt_parser.add_argument(
        "--twist_range",
        metavar=("<min>", "<max>", "<step>"),
        type=float,
        nargs=3,
        action="append",
        help="Test twist values in this range (unit: °). default to %(default)s",
        default=[],
    )

    rise_parser = parser.add_mutually_exclusive_group(required=True)
    rise_parser.add_argument(
        "--rise",
        dest="rises",
        metavar="<Å>",
        type=float,
        nargs="+",
        help="Use this rise value (unit: Å). default to %(default)s",
        default=[],
    )
    rise_parser.add_argument(
        "--rise_range",
        metavar=("<min>", "<max>", "<step>"),
        type=float,
        nargs=3,
        action="append",
        help="Test rise values in this range (unit: Å). default to %(default)s",
        default=[],
    )

    tilt_parser = parser.add_mutually_exclusive_group(required=False)
    tilt_parser.add_argument(
        "--tilt",
        dest="tilts",
        metavar="<°>",
        type=float,
        nargs="+",
        help="Use this tilt value (unit: °). default to %(default)s",
        default=[],
    )
    tilt_parser.add_argument(
        "--tilt_range",
        metavar=("<min>", "<max>", "<step>"),
        type=float,
        nargs=3,
        action="append",
        help="Test tilt values in this range (unit: °). default to %(default)s",
        default=[],
    )

    psi_parser = parser.add_mutually_exclusive_group(required=False)
    psi_parser.add_argument(
        "--psi",
        dest="psis",
        metavar="<°>",
        type=float,
        nargs="+",
        help="Use this in-plane rotation value (unit: °). default to %(default)s",
        default=[],
    )
    psi_parser.add_argument(
        "--psi_range",
        metavar=("<min>", "<max>", "<step>"),
        type=float,
        nargs=3,
        action="append",
        help="Test in-plane rotation values in this range (unit: °). default to %(default)s",
        default=[],
    )

    dy_parser = parser.add_mutually_exclusive_group(required=False)
    dy_parser.add_argument(
        "--dy",
        dest="dys",
        metavar="<Å>",
        type=float,
        nargs="+",
        help="Use this shift value (unit: Å). default to %(default)s",
        default=[],
    )
    dy_parser.add_argument(
        "--dy_range",
        metavar=("<min>", "<max>", "<step>"),
        type=float,
        nargs=3,
        action="append",
        help="Test shift values in this range (unit: Å). default to %(default)s",
        default=[],
    )

    parser.add_argument(
        "--csym",
        type=int,
        metavar="<n>",
        help="Rotational symmetry around the helical axis. default: %(default)s",
        default=1,
    )
    parser.add_argument(
        "--apix",
        type=float,
        metavar="<Å>",
        help="Use this pixel size. A positive value will overwrite the pixel size read from the image file. default to %(default)s",
        default=-1,
    )

    parser.add_argument(
        "--target_apix2d",
        type=float,
        metavar="<Å>",
        help="Down-scale images to have this pixel size before 3D reconstruction. <=0 -> no down-scaling. default to %(default)s",
        default=0,
    )
    parser.add_argument(
        "--target_apix3d",
        type=float,
        metavar="<Å>",
        help="Pixel size of 3D reconstruction. 0 -> Set to target_apix2d. <0 -> auto-decision. default to %(default)s",
        default=-1,
    )

    parser.add_argument(
        "--fsc",
        dest="fsc_test",
        type=int,
        metavar="<0|1>",
        help="Preform Fourier Shell Correlation test. 0-no test, 1-random split of input image pixels (recommended), 2-eve/odd split of input image pixels, 3-first/second half split of input image pixels. default: %(default)s",
        default=0,
    )
    parser.add_argument(
        "--tube_diameter",
        type=float,
        metavar="<Å>",
        help="The outer diameter (Å) of a tube mask to be applied to the 2D image/3D reconstruction. Automatically estimated if <0. default to %(default)s",
        default=-1,
    )
    parser.add_argument(
        "--tube_diameter_inner",
        type=float,
        metavar="<Å>",
        help="The inner diameter (Å) of a tube mask to be applied to the 3D reconstruction. default to  %(default)s",
        default=0,
    )
    parser.add_argument(
        "--tube_length",
        type=float,
        metavar="<Å>",
        help="The length (Å) of a tubular mask to be applied to the 2D image. Automatically estimated if <0. default to %(default)s",
        default=-1,
    )

    rec3d_length_parser = parser.add_mutually_exclusive_group(required=False)
    rec3d_length_parser.add_argument(
        "--reconstruct_length",
        type=float,
        metavar="<Å>",
        help="The length of the 3D reconstruction will be max(this_value, rise) when >0 or 4*rise when <=0. default: %(default)s",
        default=-1,
    )
    rec3d_length_parser.add_argument(
        "--reconstruct_length_pitch",
        type=float,
        metavar="<val>",
        help="The length of the 3D reconstruction will be max(this_value * pitch, rise) when >0 or 4*rise when <=0. default: %(default)s",
        default=-1,
    )
    rec3d_length_parser.add_argument(
        "--reconstruct_length_rise",
        type=float,
        metavar="<val>",
        help="The length of the 3D reconstruction will be max(this_value * rise, rise) when >0 or 4*rise when <=0. default: %(default)s",
        default=-1,
    )

    choices = ["tv", "nl_mean", "wavelet"]
    parser.add_argument(
        "--denoise",
        type=str,
        metavar=f"<{'|'.join(choices)}>",
        choices=choices,
        help="denoise the input images(s). disabled by default",
        default="",
    )
    parser.add_argument(
        "--low_pass",
        type=float,
        metavar="<Å>",
        help="Low pass filter of the input image(s) before further analysis. disabled by default",
        default=-1,
    )
    parser.add_argument(
        "--transpose",
        type=int,
        metavar="<-1|0|1>",
        help="Transpose the image to change the helical structure from vertical direction (cryoSPARC) to horizontal direction (RELION). Use a negative value to enable auto-detection. default to %(default)s",
        default=-1,
    )
    parser.add_argument(
        "--horizontalize",
        type=int,
        metavar="<0|1>",
        help="Automatically rotate the helical structure to horizontal direction (e.g. the helical axis along X-axis). default to %(default)s",
        default=1,
    )
    parser.add_argument(
        "--thresh_fraction",
        type=float,
        metavar="[0, 1)",
        help="Threshold the image (threshold=image_max*thresh_fraction) before analysis. Use a negative value to disable thresholding. default to %(default)s",
        default=0,
    )
    parser.add_argument(
        "--positive_constraint",
        type=int,
        metavar="<-1|0|1>",
        help="Enforce positive constraint for the reconstruction. Use a negative value for automatic decision. default to %(default)s",
        default=0,
    )
    parser.add_argument(
        "--sym_oversample",
        type=int,
        metavar="<n>",
        help="Helical sym and csym oversampling factor that controls the number of equations when setting up the A matrix of the least square solution. larger values (e.g. 100) -> slower but better quality. A negative value means auto-decision. default: %(default)s",
        default=-1,
    )
    choices = "linear linear11 linear01 linear10 nn".split()  # linear = linear11
    parser.add_argument(
        "--interpolation",
        type=str,
        choices=choices,
        metavar=f"<{'|'.join(choices)}>",
        help="Choose the interpolation method. default: %(default)s",
        default="linear",
    )
    choices = "elasticnet lasso ridge lreg lsq".split()
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=choices,
        metavar=f"<{'|'.join(choices)}>",
        help="Choose the algorithm that will be used to solve the equations. default: %(default)s",
        default="elasticnet",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        metavar="<val>",
        help="Weight of regularization. Only used for elasticnet, lasso and ridge algorithms. default: 1e-4 for elasticnet/lasso and 1 for ridge",
        default=-1,
    )
    parser.add_argument(
        "--l1_ratio",
        type=float,
        metavar="[0,1]",
        help="The ratio (0 to 1) of L1 regularization in the L1/L2 combined regularization. Only used for the elasticnet algorithms. default: %(default)s",
        default=0.5,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        metavar="<n>",
        help="Output the largest k solutions. Use a negative value to output all solutions. default: %(default)s",
        default=10,
    )
    parser.add_argument(
        "--color_map",
        type=str,
        metavar="<viridis|gray|etc>",
        help="Use this color map for image plots. Valid choices are available at https://matplotlib.org/stable/users/explain/colors/colormaps.html. default: %(default)s",
        default="gray",
    )
    parser.add_argument(
        "--save_projections",
        type=int,
        metavar="<0|1>",
        help="Save x/y/z projection images of the reconstructed 3D map. default: %(default)s",
        default=0,
    )
    parser.add_argument(
        "--cpu",
        type=int,
        metavar="<n>",
        help="Use this number of cpus. A negative value means all currently unused cpus. default: %(default)s",
        default=-1,
    )
    parser.add_argument(
        "--force",
        type=int,
        metavar="<0|1>",
        help="Clear the cached results and re-compute the reconstruction(s). default: %(default)s",
        default=0,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        metavar="<n>",
        help="Verbose level. default: %(default)s",
        default=2,
    )

    return parser


def check_args(args, parser):
    if len(args.pitches):
        args.pitches = np.array(args.pitches)
        args.twists = []
    elif len(args.pitch_range):
        tmp = [
            np.arange(start, end + step / 2, step)
            for start, end, step in args.pitch_range
        ]
        args.pitches = np.sort(np.unique(np.concatenate(tmp)))
        args.twists = []
    elif len(args.twists) > 0:
        args.twists = np.array(args.twists)
        args.pitches = []
    elif len(args.twist_range):
        tmp = [
            np.arange(start, end + step / 2, step)
            for start, end, step in args.twist_range
        ]
        args.twists = np.sort(np.unique(np.concatenate(tmp)))
        args.pitches = []
    else:
        print(
            f"ERROR: --pitch, --pitch_range, --twist, or --twist_range must be specified"
        )
        sys.exit(-1)

    if len(args.rises):
        args.rises = np.array(args.rises)
    elif len(args.rise_range):
        tmp = [
            np.arange(start, end + step / 2, step)
            for start, end, step in args.rise_range
        ]
        args.rises = np.sort(np.unique(np.concatenate(tmp)))
    else:
        print(f"ERROR: --rise, --rise_range must be specified")
        sys.exit(-1)

    if len(args.tilts):
        args.tilts = np.array(args.tilts)
    elif len(args.tilt_range):
        tmp = [
            np.arange(start, end + step / 2, step)
            for start, end, step in args.tilt_range
        ]
        args.tilts = np.sort(np.unique(np.concatenate(tmp)))
    else:
        args.tilts = np.array([0.0])

    if len(args.psis):
        args.psis = np.array(args.psis)
    elif len(args.psi_range):
        tmp = [
            np.arange(start, end + step / 2, step)
            for start, end, step in args.psi_range
        ]
        args.psis = np.sort(np.unique(np.concatenate(tmp)))
    else:
        args.psis = np.array([0.0])

    if len(args.dys):
        args.dys = np.array(args.dys)
    elif len(args.dy_range):
        tmp = [
            np.arange(start, end + step / 2, step) for start, end, step in args.dy_range
        ]
        args.dys = np.sort(np.unique(np.concatenate(tmp)))
    else:
        args.dys = np.array([0.0])

    args.i = []
    if len(args.i1):
        args.i = sorted(list(set(args.i1)))
    if len(args.i0):
        args.i = sorted([i + 1 for i in set(args.i0)])

    args.l1_ratio = max(0, min(1, args.l1_ratio))
    args.algorithm = dict(model=args.algorithm, l1_ratio=args.l1_ratio)
    if args.alpha >= 0:
        args.algorithm["alpha"] = args.alpha

    if args.output_prefix.find("/") == -1 or args.output_prefix[-1] == "/":
        args.output_prefix = (
            str(pathlib.Path(__file__).stem.upper())
            + "/"
            + args.output_prefix.rstrip("/")
        )

    if args.csym > parser.get_default("csym"):
        args.output_prefix += f".csym-{args.csym}"
    if args.tube_diameter > 0:
        args.output_prefix += f".diameter-{args.tube_diameter}"
    if args.tube_length > 0:
        args.output_prefix += f".length2d-{args.tube_length}"
    if args.reconstruct_length > 0:
        args.output_prefix += f".length3d-{args.reconstruct_length}A"
    if args.reconstruct_length_pitch > 0:
        args.output_prefix += f".length3d-{args.reconstruct_length_pitch}pitch"
    if args.reconstruct_length_rise > 0:
        args.output_prefix += f".length3d-{args.reconstruct_length_rise}rise"
    if args.denoise != parser.get_default("denoise"):
        args.output_prefix += f".denoise-{args.denoise}"
    if args.low_pass != parser.get_default("low_pass"):
        args.output_prefix += f".low_pass-{args.low_pass}"
    if args.horizontalize != parser.get_default("horizontalize"):
        args.output_prefix += f".horizontalize-{args.horizontalize}"
    if args.positive_constraint != parser.get_default("positive_constraint"):
        args.output_prefix += f".positive-{args.positive_constraint}"
    if args.target_apix3d != parser.get_default("target_apix3d"):
        args.output_prefix += f".apix3d-{args.target_apix3d}"
    if args.target_apix2d != parser.get_default("target_apix2d"):
        args.output_prefix += f".apix2d-{args.target_apix2d}"
    if args.thresh_fraction > 0:
        args.output_prefix += f".thresh-{args.thresh_fraction}"
    if args.sym_oversample != parser.get_default("sym_oversample"):
        args.output_prefix += f".sym_oversample-{args.sym_oversample}"
    if args.interpolation != parser.get_default("interpolation"):
        args.output_prefix += f".interpolation-{args.interpolation}"
    if args.algorithm["model"] != parser.get_default("algorithm"):
        args.output_prefix += f".algorithm-{args.algorithm['model']}"
    if args.alpha != parser.get_default("alpha"):
        args.output_prefix += f".alpha-{args.alpha}"
    if args.l1_ratio != parser.get_default("l1_ratio"):
        args.output_prefix += f".l1_ratio-{args.l1_ratio}"

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
