"""Handler for the flip_hand option."""

from __future__ import annotations
import argparse
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError


option_name = "flip_hand"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the flip_hand option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--flip_hand",
        type=str,
        metavar="<param>=<val>:...",
        help="flip_hand parameter",
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
    """Handle the flip_hand option.

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
        axis = param.lower()
        if axis not in ["x", "y", "z"]:
            raise HeliconError("\\tERROR: invalid axis: {axis}")
        data = helicon.flip_hand(data, axis=axis)
    return data, apix, nx, ny, nz


import logging

logger = logging.getLogger(__name__)
