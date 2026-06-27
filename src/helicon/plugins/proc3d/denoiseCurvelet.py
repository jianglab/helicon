"""Handler for the denoiseCurvelet option (UDCT volume denoising)."""

from __future__ import annotations
import argparse
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError

logger = helicon.get_logger(__name__)

option_name = "denoiseCurvelet"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the denoiseCurvelet option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--denoiseCurvelet",
        type=str,
        metavar="<sigma=<float>[:numScales=<3>][:wedgesPerDir=<3>][:gpu=<true|false>][:]tileSize=<int>[:overlap=<32>]>",
        help="apply UDCT-based denoising to the 3D volume. "
        "Defaults: elbow mode, numScales=3, wedgesPerDir=3, gpu=false. "
        "tileSize=0 auto-detects from available memory. "
        "Requires curvelets package. disabled by default",
        default=None,
    )


def handle(
    data: np.ndarray,
    args: argparse.Namespace,
    index_d: dict,
    param: object,
    apix: float,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, float, int, int, int]:
    """Handle the denoiseCurvelet option.

    Parameters
    ----------
    data : np.ndarray
        3D volume data.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter string for this option.
    apix : float
        Current pixel size.
    nx : int
        Current X dimension.
    ny : int
        Current Y dimension.
    nz : int
        Current Z dimension.

    Returns
    -------
    tuple[np.ndarray, float, int, int, int]
        (data, apix, nx, ny, nz) after processing.
    """
    if param is not None:
        if not helicon.has_curvelet_udct():
            raise HeliconError(
                "\tERROR: curvelets package is required for --denoiseCurvelet. "
                "Install with: pip install curvelets"
            )

        _, param_dict = helicon.parse_param_str(param) if param else ({}, {})
        sigma = param_dict.get("sigma", None)
        if sigma is not None:
            sigma = float(sigma)
        num_scales = int(param_dict.get("numScales", 3))
        wedges_per_dir = int(param_dict.get("wedgesPerDir", 3))
        use_gpu = param_dict.get("gpu", False) in (True, 1, "true", "1", "yes")
        tile_size_str = param_dict.get("tileSize", None)
        overlap = int(param_dict.get("overlap", 32))

        if tile_size_str is not None:
            tile_size = int(tile_size_str)
            tile_size = (tile_size, tile_size, tile_size) if tile_size > 0 else None
        else:
            tile_size = None

        if use_gpu and not helicon.has_curvelet_udct_gpu():
            raise HeliconError(
                "\tERROR: UDCT GPU support requires torch. "
                "Install with: pip install torch"
            )

        if num_scales < 2:
            raise HeliconError("\tERROR: numScales must be >= 2 for --denoiseCurvelet")

        import psutil

        estimated_peak = data.nbytes * 300
        use_tiled = (
            tile_size is not None
            or estimated_peak > psutil.virtual_memory().available * 0.8
        )

        if tile_size is None and use_tiled:
            safe_memory = psutil.virtual_memory().available * 0.8
            max_tile_bytes = safe_memory / 300
            total_voxels = int(np.floor(max_tile_bytes / 8))
            target_edge = int(np.floor(total_voxels ** (1 / 3)))
            tile_size = tuple(min(target_edge, d) for d in (nx, ny, nz))

        if args.verbose:
            mode = "elbow" if (sigma is None or sigma <= 0) else "MAD"
            device_tag = "GPU" if use_gpu else "CPU"
            ts_tag = (
                f"tileSize=({tile_size[0]},{tile_size[1]},{tile_size[2]})"
                if tile_size
                else "auto-tile"
            )
            use_tiled_tag = "tiled" if use_tiled else "untiled"
            logger.info(
                "\tdenoising 3D volume (%d x %d x %d) with UDCT (%s, %s, numScales=%d, "
                "wedgesPerDir=%d, sigma=%s, %s, %s, overlap=%d) ...",
                nz,
                ny,
                nx,
                mode,
                device_tag,
                num_scales,
                wedges_per_dir,
                str(sigma),
                ts_tag,
                use_tiled_tag,
                overlap,
            )

        if use_tiled:
            n_jobs = helicon.available_cpu(mem_gb_per_cpu=4)
            data = helicon.curvelet_denoise_3d_udct_tiled(
                data,
                sigma=sigma,
                num_scales=num_scales,
                wedges_per_dir=wedges_per_dir,
                tile_size=tile_size,
                overlap=overlap,
                use_gpu=use_gpu,
                n_jobs=n_jobs,
            )
        else:
            data = helicon.curvelet_denoise_3d_udct(
                data,
                sigma=sigma,
                num_scales=num_scales,
                wedges_per_dir=wedges_per_dir,
                use_gpu=use_gpu,
            )

    index_d[option_name] += 1
    return data, apix, nx, ny, nz
