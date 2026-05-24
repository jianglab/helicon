"""Handler for the z_moving_average option."""

from __future__ import annotations
import argparse
import logging
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


option_name = "z_moving_average"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the z_moving_average option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--z_moving_average",
        type=str,
        metavar="<param>=<val>:...",
        help="apply a moving average filter along the z-axis",
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
    """Handle the z_moving_average option.

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
            length=0.0,
            n_pixel=0,
        )
        _, param_dict = helicon.parse_param_str(param)
        param_dict, param_changed, param_unsuppported = helicon.validate_param_dict(
            param=param_dict, param_ref=param_dict_default
        )
        if len(param_unsuppported):
            logger.warning("ignoring unknown parameters: %s", param_unsuppported)
        if args.verbose > 2:
            logger.info(f"\tCustom parameters: {param_changed}")
        length = float(param_dict["length"])
        n_pixel = float(param_dict["n_pixel"])
        if length <= 0 and n_pixel <= 0:
            raise HeliconError("length (>0) or n_pixel (>0) should be specified")
        if length > 0 and n_pixel > 0:
            raise HeliconError(
                "either length (>0) or n_pixel (>0) but not both should be specified"
            )

        if length > 0:
            n_pixel = int(np.round(length / apix))

        tmp = np.cumsum(data, axis=0, dtype=float)
        data = data.copy()
        data[n_pixel // 2 : -n_pixel // 2] = (tmp[n_pixel:] - tmp[:-n_pixel]) / n_pixel

        index_d[option_name] += 1

    return data, apix, nx, ny, nz
