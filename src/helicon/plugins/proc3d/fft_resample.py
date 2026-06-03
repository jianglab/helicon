"""Handler for the fft_resample option."""

from __future__ import annotations
import argparse
import logging
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


option_name = "fft_resample"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the fft_resample option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--fft_resample",
        type=str,
        metavar="<param>=<val>:...",
        help="fft_resample parameter",
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
    """Handle the fft_resample option.

    Parameters
    ----------
    data : np.ndarray
        3D volume data.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter value for this option.
    apix : float
        Current pixel size in Angstroms.
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
    if param:
        param_dict_default = dict(
            new_nx=nx,
            new_ny=ny,
            new_nz=nz,
        )
        _, param_dict = helicon.parse_param_str(param)
        param_dict, param_changed, param_unsuppported = helicon.validate_param_dict(
            param=param_dict, param_ref=param_dict_default
        )
        if len(param_unsuppported):
            logger.warning("ignoring unknown parameters: %s", param_unsuppported)
        if args.verbose > 2:
            logger.info(f"\tCustom parameters: {param_changed}")
        new_nx = int(param_dict["new_nx"])
        new_ny = int(param_dict["new_ny"])
        new_nz = int(param_dict["new_nz"])
        if new_nx < 1:
            raise HeliconError("\\tnew_nx must be >0")
        if new_ny <= 0:
            raise HeliconError("\\tnew_ny must be >0")
        if new_nz <= 0:
            raise HeliconError("\\tnew_nz must be >0")

        if len(set([new_nx / nx, new_ny / ny, new_nz / nz])) > 1:
            msg = f"nx,ny,nz={nx},{ny},{nz} -> {new_nx},{new_ny},{new_nz} FFT-resampling will result in nonuniform pixel size in x/y/z dimensions"
            logger.warning("%s", msg)

        fft = helicon.fft_rescale(
            data,
            apix=apix,
            cutoff_res=(
                2 * apix * nz / new_nz,
                2 * apix * ny / new_ny,
                2 * apix * nx / new_nx,
            ),
            output_size=(new_nz, new_ny, new_nx),
        )
        data = np.abs(np.fft.ifftn(fft)).astype(np.float32)
        data *= new_nx * new_ny * new_nz / (nx * ny * nz)
        apix = round(apix * nx / new_nx, 4)
        nx = new_nx
        ny = new_ny
        nz = new_nz
    return data, apix, nx, ny, nz
