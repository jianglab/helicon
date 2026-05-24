"""Handler for the apix option."""

from __future__ import annotations
import argparse
import helicon
import numpy as np


option_name = "apix"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the apix option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--apix",
        type=str,
        metavar="<param>=<val>:...",
        help="apix parameter",
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
    """Handle the apix option.

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
    apix = float(param)
    index_d[option_name] += 1
    return data, apix, nx, ny, nz


import logging

logger = logging.getLogger(__name__)
