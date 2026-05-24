"""Handler for the clip option."""

from __future__ import annotations
import argparse
import logging
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


option_name = "clip"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the clip option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--clip",
        type=str,
        metavar="<param>=<val>:...",
        help="clip parameter",
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
    """Handle the clip option.

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
            center_x=nx // 2,
            center_y=ny // 2,
            center_z=nz // 2,
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
        center_x = int(param_dict["center_x"])
        center_y = int(param_dict["center_y"])
        center_z = int(param_dict["center_z"])
        if new_nx < 1:
            raise HeliconError("\\tERROR: new_nx must be >0")
        if new_ny < 1:
            raise HeliconError("\\tERROR: new_ny must be >0")
        if new_nz < 1:
            raise HeliconError("\\tERROR: new_nz must be >0")

        data = helicon.get_clip3d(
            data,
            z0=center_z - new_nz // 2,
            y0=center_y - new_ny // 2,
            x0=center_x - new_nx // 2,
            nz=new_nz,
            ny=new_ny,
            nx=new_nx,
        )
        nx = new_nx
        ny = new_ny
        nz = new_nz
    return data, apix, nx, ny, nz
