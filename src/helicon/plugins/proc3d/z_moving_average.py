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
            n_pixel = length / apix
        n_pixel = max(1, int(np.round(n_pixel)))
        if n_pixel > nz:
            raise HeliconError(
                f"the moving average window ({n_pixel} slices) is longer than the map ({nz} slices)"
            )

        data = z_moving_average(data, n_pixel)

        index_d[option_name] += 1

    return data, apix, nx, ny, nz


def z_moving_average(data: np.ndarray, n_pixel: int) -> np.ndarray:
    """Average each z-slice with its neighbours in a window of ``n_pixel`` slices.

    The window is centred on the slice (for an even ``n_pixel`` it extends one
    slice further towards higher z than lower z). The slices near the ends,
    whose window would extend beyond the map, are left unchanged.

    Parameters
    ----------
    data : np.ndarray
        3D map (nz, ny, nx).
    n_pixel : int
        Window length in slices (>= 1).

    Returns
    -------
    np.ndarray
        The averaged map, the same shape and dtype as ``data``.
    """
    nz = data.shape[0]
    csum = np.zeros((nz + 1,) + data.shape[1:], dtype=float)
    np.cumsum(data, axis=0, dtype=float, out=csum[1:])
    ret = data.copy()
    # window [a, a+n_pixel-1] -> its centre slice a + (n_pixel-1)//2
    first = (n_pixel - 1) // 2
    ret[first : first + nz - n_pixel + 1] = (
        csum[n_pixel:] - csum[: nz - n_pixel + 1]
    ) / n_pixel
    return ret
