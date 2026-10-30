"""Handler for the denoiseCurvelet option: automatic curvelet + wavelet denoising."""

from __future__ import annotations

import logging
from pathlib import Path

import helicon

logger = logging.getLogger("helicon.commands.images2star")

option_name = "denoiseCurvelet"


def add_args(parser):
    parser.add_argument(
        "--denoiseCurvelet",
        nargs="?",
        const="",
        metavar="[strength=<2>][:outdir=<./denoised/>][:noiseSd=<auto>]",
        action="append",
        help="denoise particle images, or micrographs when rlnImageName is absent, "
        "automatically: monogenic curvelets (lines such as filaments) and wavelets "
        "(spots) with thresholds set for every sub-band from the image's own noise, "
        "averaged. strength scales all thresholds: 2 (default) gives a clean view "
        "for looking at and picking filaments, 1 keeps more fine texture. noiseSd "
        "overrides the per-pixel noise level estimated from each image. Images are "
        "written to outdir. CPU count from --cpu. disabled by default",
        default=[],
    )


def handle(data, args, index_d, param):
    if param is not None:
        _, param_dict = helicon.parse_param_str(param) if param else ({}, {})
        strength = float(param_dict.get("strength", 2.0))
        outdir = param_dict.get("outdir", None) or "./denoised/"
        noise_sd = param_dict.get("noiseSd", None)
        noise_sd = float(noise_sd) if noise_sd not in (None, "", "auto") else None

        has_image_col = "rlnImageName" in data
        has_micrograph_col = "rlnMicrographName" in data
        if not has_image_col and not has_micrograph_col:
            if args.verbose:
                logger.info(
                    "\tdenoiseCurvelet: no rlnImageName or rlnMicrographName column in data, skipping"
                )
            index_d[option_name] += 1
            return data, index_d

        n_jobs = getattr(args, "cpu", 0)
        if n_jobs < 1:
            n_jobs = helicon.available_cpu()
        outdir = Path(outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        if has_image_col:
            _denoise_particles(data, args, strength, noise_sd, outdir, n_jobs)
        else:
            _denoise_micrographs(data, args, strength, noise_sd, outdir, n_jobs)

        index_d[option_name] += 1

    return data, index_d


def _report(info, what):
    logger.info(
        "\tdenoiseCurvelet %s: strength %g, noise sd %.4g (from %s)",
        what,
        info["strength"],
        info["noise_sd"],
        info["noise_from"],
    )


def _one(job):
    from helicon.lib.denoise import denoise_micrograph

    image, strength, noise_sd = job
    return denoise_micrograph(image, strength=strength, noise_sd=noise_sd, workers=1)


def _pool(n_jobs, n_images):
    """Worker processes for the particles, or None to denoise them in this process."""
    if n_jobs < 2 or n_images < 2:
        return None
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # a fork server: forking this process (threads, or the GUI) can hang the children
    try:
        context = mp.get_context("forkserver")
        context.set_forkserver_preload(["helicon.lib.denoise", "scipy.fft"])
    except ValueError:
        context = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=min(n_jobs, n_images), mp_context=context)


def _denoise_particles(data, args, strength, noise_sd, outdir, n_jobs):
    from contextlib import nullcontext

    import mrcfile
    import numpy as np

    image_col = "rlnImageName"
    tmp_col = helicon.unique_attr_name(data, attr_prefix=image_col)
    data[tmp_col] = data[image_col].str.split("@", expand=True).iloc[:, -1]
    n_denoised = 0
    pool = _pool(n_jobs, len(data))
    with pool or nullcontext():
        for stack_path in data[tmp_col].unique():
            # one stack at a time: holding every particle of every stack at once
            # would need tens of GB
            rows = data[data[tmp_col] == stack_path]
            with mrcfile.open(stack_path, permissive=True) as mrc:
                stack = mrc.data
            if stack.ndim == 2:
                images = [np.asarray(stack, np.float64)] * len(rows)
            else:
                idx = [int(s.split("@")[0]) - 1 for s in rows[image_col]]
                images = [np.asarray(stack[i], np.float64) for i in idx]
            # small images: one per worker, each FFT single-threaded
            jobs = [(img, strength, noise_sd) for img in images]
            if pool is not None and len(jobs) > 1:
                chunk = max(1, len(jobs) // (4 * n_jobs))
                done = list(pool.map(_one, jobs, chunksize=chunk))
            else:
                done = [_one(j) for j in jobs]
            if args.verbose and done:
                _report(done[0][1], f"{len(done)} particles of {stack_path}, first")
            # always rebuild the output stack: appending to an existing one would
            # leave the rewritten particle numbers pointing at the old images
            out_stack = str(outdir / Path(stack_path).name)
            with mrcfile.new(out_stack, overwrite=True) as mrc:
                mrc.set_data(np.stack([d for d, _ in done]).astype(np.float32))
            data.loc[rows.index, image_col] = [
                f"{k + 1}@{out_stack}" for k in range(len(rows))
            ]
            n_denoised += len(done)
    data.drop(tmp_col, inplace=True, axis=1)
    if args.verbose > 1:
        logger.info("\tdenoised %d particles written to %s", n_denoised, outdir)


def _denoise_micrographs(data, args, strength, noise_sd, outdir, n_jobs):
    import mrcfile
    import numpy as np

    from helicon.lib.denoise import denoise_micrograph
    from helicon.lib.exceptions import HeliconError

    image_col = "rlnMicrographName"
    new_names = {}
    for path in data[image_col].unique():
        with mrcfile.open(path, permissive=True) as mrc:
            image = np.asarray(mrc.data, np.float64)
            voxel_size = mrc.voxel_size
        if image.ndim != 2:
            raise HeliconError(
                "\tERROR: --denoiseCurvelet expects 2D MRC files in rlnMicrographName. "
                f"{path} has {image.ndim} dimensions."
            )
        # one large image at a time, its FFTs using all the CPUs
        denoised, info = denoise_micrograph(
            image, strength=strength, noise_sd=noise_sd, workers=n_jobs
        )
        if args.verbose:
            _report(info, f"{Path(path).name}")
        out = str(outdir / Path(path).name)
        with mrcfile.new(out, overwrite=True) as mrc:
            mrc.set_data(denoised)
            mrc.voxel_size = voxel_size
        new_names[path] = out
    data[image_col] = data[image_col].map(new_names)
    if args.verbose > 1:
        logger.info("\tdenoised %d micrographs written to %s", len(new_names), outdir)
