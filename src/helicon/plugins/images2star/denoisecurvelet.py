"""Handler for the denoiseCurvelet option (supports FDCT and UDCT backends)."""

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
        metavar="[transform=<udct|fdct>[:sigma=<float>][:numScales=<4>][:wedgesPerDir=<3>][:gpu=<true|false>][:outdir=<path>]]",
        action="append",
        help="apply curvelet-based denoising to particle images in parallel. "
        "Transform defaults to UDCT. "
        "Defaults: transform=udct, elbow mode, numScales=4, wedgesPerDir=3, gpu=false, outdir=./denoised/. "
        "CPU count from --cpu flag. Requires curvepy-fdct (fdct) or curvelets (udct). "
        "disabled by default",
        default=[],
    )


def handle(data, args, index_d, param):
    if param is not None:
        _, param_dict = helicon.parse_param_str(param) if param else ({}, {})
        transform = param_dict.get("transform", "udct")
        sigma = param_dict.get("sigma", None)
        if sigma is not None:
            sigma = float(sigma)
        num_scales = int(param_dict.get("numScales", 4))
        wedges_per_dir = int(param_dict.get("wedgesPerDir", 3))
        use_gpu = param_dict.get("gpu", False) in (True, 1, "true", "1", "yes")
        outdir = param_dict.get("outdir", None) or "./denoised/"

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
                    "\tERROR: UDCT GPU support requires torch. "
                    "Install with: pip install torch"
                )
        else:
            raise HeliconError(
                f"\tERROR: unknown transform '{transform}' for --denoiseCurvelet. "
                "Use 'fdct' or 'udct'."
            )

        if num_scales < 2:
            raise HeliconError("\tERROR: numScales must be >= 2 for --denoiseCurvelet")

        n_jobs = getattr(args, "cpu", 0)
        if n_jobs < 1:
            n_jobs = helicon.available_cpu()

        if "rlnImageName" in data:
            import mrcfile
            import numpy as np
            from joblib import Parallel, delayed

            outdir = str(Path(outdir).resolve())
            Path(outdir).mkdir(parents=True, exist_ok=True)

            image_col = "rlnImageName"
            tmp_col = helicon.unique_attr_name(data, attr_prefix=image_col)
            data[tmp_col] = data[image_col].str.split("@", expand=True).iloc[:, -1]

            unique_stacks = data[tmp_col].unique()
            new_paths = []
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

            if transform == "fdct":
                denoised = helicon.curvelet_denoise_batch_fdct(
                    images,
                    sigma=sigma,
                    num_scales=num_scales,
                    n_jobs=n_jobs,
                )
            else:
                denoised = helicon.curvelet_denoise_batch_udct(
                    images,
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

        elif args.verbose:
            logger.info("\tdenoiseCurvelet: no rlnImageName column in data, skipping")

        index_d[option_name] += 1

    return data, index_d
