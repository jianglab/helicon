"""Handler for the denoiseCurvelet option (supports FDCT, UDCT, and MCT backends)."""

from __future__ import annotations
import logging
from pathlib import Path
import helicon
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)

option_name = "denoiseCurvelet"


def add_args(parser):
    parser.add_argument(
        "--denoiseCurvelet",
        nargs="?",
        const="",
        metavar="[transform=<mct|udct|fdct>[:sigma=<float>][:numScales=<auto>][:wedgesPerDir=<3>][:gpu=<true|false>][:tileSize=<N>][:overlap=<N>][:outdir=<path>]]",
        action="append",
        help="apply curvelet-based denoising to particle images in parallel, "
        "or micrographs when rlnImageName is absent. "
        "Transform defaults to MCT. "
        "Use tileSize=N to enable tiled processing (parallel within each large image). "
        "Defaults: transform=mct, sigma=3, numScales=auto, wedgesPerDir=3, gpu=false, "
        "outdir=./denoised/, overlap=32. "
        "CPU count from --cpu flag. Requires curvepy-fdct (fdct) or curvelets (udct/mct). "
        "disabled by default",
        default=[],
    )


def handle(data, args, index_d, param):
    if param is not None:
        _, param_dict = helicon.parse_param_str(param) if param else ({}, {})
        transform = param_dict.get("transform", "mct")
        sigma = param_dict.get("sigma", None)
        if sigma is not None:
            sigma = float(sigma)
        else:
            sigma = 3.0
        num_scales = param_dict.get("numScales", None)
        if num_scales is not None:
            num_scales = int(num_scales)
        wedges_per_dir = int(param_dict.get("wedgesPerDir", 3))
        use_gpu = param_dict.get("gpu", False) in (True, 1, "true", "1", "yes")
        tile_size = param_dict.get("tileSize", None)
        if tile_size is not None:
            tile_size = int(tile_size)
        overlap = int(param_dict.get("overlap", 32))
        outdir = param_dict.get("outdir", None) or "./denoised/"

        has_image_col = "rlnImageName" in data
        has_micrograph_col = "rlnMicrographName" in data
        if not has_image_col and not has_micrograph_col:
            if args.verbose:
                logger.info(
                    "\tdenoiseCurvelet: no rlnImageName or rlnMicrographName column in data, skipping"
                )
            index_d[option_name] += 1
            return data, index_d

        _check_transform_dependencies(transform, use_gpu)

        if num_scales is not None and num_scales < 2:
            num_scales = None  # values < 2 trigger auto-decide

        n_jobs = getattr(args, "cpu", 0)
        if n_jobs < 1:
            n_jobs = helicon.available_cpu()

        if has_image_col:
            import mrcfile
            import numpy as np

            outdir = str(Path(outdir).resolve())
            Path(outdir).mkdir(parents=True, exist_ok=True)

            image_col = "rlnImageName"
            tmp_col = helicon.unique_attr_name(data, attr_prefix=image_col)
            data[tmp_col] = data[image_col].str.split("@", expand=True).iloc[:, -1]

            unique_stacks = data[tmp_col].unique()
            images = []
            for stack_path in unique_stacks:
                with mrcfile.open(stack_path, permissive=True) as mrc:
                    stack_data = mrc.data
                single_image = stack_data.ndim == 2
                for i, row in data[data[tmp_col] == stack_path].iterrows():
                    if single_image:
                        img = np.asarray(stack_data, dtype=np.float64)
                    else:
                        particle_idx = int(row[image_col].split("@")[0]) - 1
                        img = np.asarray(stack_data[particle_idx], dtype=np.float64)
                    images.append(img)

            if args.verbose > 1:
                device_tag = "GPU" if use_gpu else "CPU"
                logger.info(
                    "\tdenoising %d particles with %d workers (%s on %s) ...",
                    len(images),
                    n_jobs,
                    transform.upper(),
                    device_tag,
                )

            if tile_size is not None:
                denoised = [
                    _denoise_tiled(
                        img,
                        transform,
                        sigma,
                        num_scales,
                        wedges_per_dir,
                        tile_size,
                        overlap,
                        n_jobs,
                        use_gpu,
                    )
                    for img in images
                ]
            else:
                denoised = _denoise_batch(
                    images,
                    transform,
                    sigma=sigma,
                    num_scales=num_scales,
                    wedges_per_dir=wedges_per_dir,
                    n_jobs=n_jobs,
                    use_gpu=use_gpu,
                )

            new_stacks = {}
            for stack_path in unique_stacks:
                out_stack = str(Path(outdir) / Path(stack_path).name)
                new_stacks[stack_path] = out_stack

            idx = 0
            for stack_path in unique_stacks:
                n_in_stack = len(data[data[tmp_col] == stack_path])
                out_stack = new_stacks[stack_path]
                if Path(out_stack).exists():
                    with mrcfile.open(out_stack, permissive=True) as mrc:
                        existing = mrc.data
                    stacked = np.concatenate(
                        [existing, np.stack(denoised[idx : idx + n_in_stack])]
                    )
                    with mrcfile.new(out_stack, overwrite=True) as mrc:
                        mrc.set_data(stacked.astype(np.float32))
                else:
                    with mrcfile.new(out_stack, overwrite=True) as mrc:
                        mrc.set_data(
                            np.stack(denoised[idx : idx + n_in_stack]).astype(
                                np.float32
                            )
                        )
                idx += n_in_stack

            for i, row in data.iterrows():
                orig_path = row[tmp_col]
                particle_num = row[image_col].split("@")[0]
                data.at[i, image_col] = f"{particle_num}@{new_stacks[orig_path]}"

            data.drop(tmp_col, inplace=True, axis=1)

            if args.verbose > 1:
                logger.info(
                    "\tdenoised %d particles written to %s",
                    len(denoised),
                    outdir,
                )

        elif has_micrograph_col:
            import mrcfile
            import numpy as np

            outdir = str(Path(outdir).resolve())
            Path(outdir).mkdir(parents=True, exist_ok=True)

            image_col = "rlnMicrographName"
            unique_micrographs = data[image_col].unique()
            images = []
            voxel_sizes = {}
            for micrograph_path in unique_micrographs:
                with mrcfile.open(micrograph_path, permissive=True) as mrc:
                    micrograph_data = mrc.data
                    if micrograph_data.ndim != 2:
                        raise HeliconError(
                            "\tERROR: --denoiseCurvelet micrograph fallback expects "
                            f"2D MRC files in rlnMicrographName. {micrograph_path} "
                            f"has {micrograph_data.ndim} dimensions."
                        )
                    images.append(np.asarray(micrograph_data, dtype=np.float64).copy())
                    voxel_sizes[micrograph_path] = (
                        float(mrc.voxel_size.x),
                        float(mrc.voxel_size.y),
                        float(mrc.voxel_size.z),
                    )

            if args.verbose > 1:
                device_tag = "GPU" if use_gpu else "CPU"
                logger.info(
                    "\tdenoising %d micrographs with %d workers (%s on %s) ...",
                    len(images),
                    n_jobs,
                    transform.upper(),
                    device_tag,
                )

            if tile_size is not None:
                denoised = [
                    _denoise_tiled(
                        img,
                        transform,
                        sigma,
                        num_scales,
                        wedges_per_dir,
                        tile_size,
                        overlap,
                        n_jobs,
                        use_gpu,
                    )
                    for img in images
                ]
            else:
                denoised = _denoise_batch(
                    images,
                    transform,
                    sigma,
                    num_scales,
                    wedges_per_dir,
                    n_jobs,
                    use_gpu,
                )

            new_micrographs = {}
            for micrograph_path, denoised_micrograph in zip(
                unique_micrographs, denoised
            ):
                out_micrograph = str(Path(outdir) / Path(micrograph_path).name)
                new_micrographs[micrograph_path] = out_micrograph
                with mrcfile.new(out_micrograph, overwrite=True) as mrc:
                    mrc.set_data(denoised_micrograph.astype(np.float32))
                    mrc.voxel_size = voxel_sizes[micrograph_path]

            data[image_col] = data[image_col].map(new_micrographs)

            if args.verbose > 1:
                logger.info(
                    "\tdenoised %d micrographs written to %s",
                    len(denoised),
                    outdir,
                )

        index_d[option_name] += 1

    return data, index_d


def _check_transform_dependencies(transform, use_gpu):
    if transform == "fdct":
        if not helicon.has_curvelet_fdct():
            raise HeliconError(
                "\tERROR: curvepy-fdct is required for --denoiseCurvelet transform=fdct. "
                "Install with: pip install curvepy-fdct"
            )
        if use_gpu:
            raise HeliconError(
                "\tERROR: FDCT does not support GPU. Use transform=udct for GPU."
            )
    elif transform == "udct":
        if not helicon.has_curvelet_udct():
            raise HeliconError(
                "\tERROR: curvelets package is required for --denoiseCurvelet transform=udct. "
                "Install with: pip install curvelets"
            )
        if use_gpu and not helicon.has_curvelet_udct_gpu():
            raise HeliconError(
                "\tERROR: UDCT GPU support requires torch. Install with: pip install torch"
            )
    elif transform == "mct":
        if not helicon.has_curvelet_udct():
            raise HeliconError(
                "\tERROR: curvelets package is required for --denoiseCurvelet transform=mct. "
                "Install with: pip install curvelets"
            )
        if use_gpu:
            raise HeliconError(
                "\tERROR: MCT does not support GPU. Use transform=udct for GPU."
            )
    else:
        raise HeliconError(
            f"\tERROR: unknown transform '{transform}' for --denoiseCurvelet. "
            "Use 'fdct', 'udct', or 'mct'."
        )


def _denoise_tiled(
    img,
    transform,
    sigma,
    num_scales,
    wedges_per_dir,
    tile_size,
    overlap,
    n_jobs,
    use_gpu,
):
    if transform == "fdct":
        return helicon.curvelet_denoise_fdct_tiled(
            img,
            sigma=sigma,
            num_scales=num_scales,
            tile_size=tile_size,
            overlap=overlap,
            n_jobs=n_jobs,
        )
    if transform == "mct":
        return helicon.curvelet_denoise_mct_tiled(
            img,
            sigma=sigma,
            num_scales=num_scales,
            wedges_per_dir=wedges_per_dir,
            tile_size=tile_size,
            overlap=overlap,
            n_jobs=n_jobs,
        )
    return helicon.curvelet_denoise_udct_tiled(
        img,
        sigma=sigma,
        num_scales=num_scales,
        wedges_per_dir=wedges_per_dir,
        tile_size=tile_size,
        overlap=overlap,
        n_jobs=n_jobs,
        use_gpu=use_gpu,
    )


def _denoise_batch(
    images,
    transform,
    sigma,
    num_scales,
    wedges_per_dir,
    n_jobs,
    use_gpu,
):
    if transform == "fdct":
        return helicon.curvelet_denoise_batch_fdct(
            images,
            sigma=sigma,
            num_scales=num_scales,
            n_jobs=n_jobs,
        )
    if transform == "mct":
        return helicon.curvelet_denoise_batch_mct(
            images,
            sigma=sigma,
            num_scales=num_scales,
            wedges_per_dir=wedges_per_dir,
            n_jobs=n_jobs,
        )
    return helicon.curvelet_denoise_batch_udct(
        images,
        sigma=sigma,
        num_scales=num_scales,
        wedges_per_dir=wedges_per_dir,
        n_jobs=n_jobs,
        use_gpu=use_gpu,
    )
