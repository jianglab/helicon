"""Handler for the helical_sym option."""

from __future__ import annotations
import argparse
import logging
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


option_name = "helical_sym"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the helical_sym option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--helical_sym",
        type=str,
        metavar="<param>=<val>:...",
        help="helical_sym parameter",
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
    """Handle the helical_sym option.

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
            twist=0.0,  # °
            rise=0.0,  # Å
            csym=1,
            center_len=0.0,  # Å
            center_n_rise=0.0,
            center_fraction=0.0,
            new_apix=apix,
            new_nz=nz,
            new_nxy=nx,
        )
        _, param_dict = helicon.parse_param_str(param)
        param_dict, param_changed, param_unsuppported = helicon.validate_param_dict(
            param=param_dict, param_ref=param_dict_default
        )
        if len(param_unsuppported):
            logger.warning("ignoring unknown parameters: %s", param_unsuppported)
        if args.verbose > 2:
            logger.info(f"\tCustom parameters: {param_changed}")
        twist = float(param_dict["twist"])
        rise = float(param_dict["rise"])
        csym = int(param_dict.get("csym", 1))
        if rise <= 0:
            raise HeliconError("\\tERROR: rise (>0) must be specified")
        if csym < 1:
            raise HeliconError("\\tERROR: csym (>0) must be specified")
        new_apix = float(param_dict.get("new_apix", apix))
        new_nz = int(param_dict["new_nz"])
        new_nxy = int(param_dict["new_nxy"])
        center_len = float(param_dict["center_len"])
        center_n_rise = float(param_dict["center_n_rise"])
        center_fraction = float(param_dict["center_fraction"])
        tmp = int(center_len > 0) + int(center_n_rise > 0) + int(center_fraction > 0)
        if tmp != 1:
            if tmp <= 0:
                msg = "\tERROR: center_len or center_n_rise or center_fraction must be specified"
            else:
                msg = "\tERROR: only one of the these three options (center_len, center_n_rise, center_fraction) should be specified"
            raise HeliconError(msg)
        if center_len > 0:
            if center_len < rise:
                raise HeliconError(
                    f"\tERROR: center_len must be larger than rise (={rise} Å)"
                )
            center_fraction = center_len / (nz * apix)
        elif center_n_rise > 0:
            center_fraction = center_n_rise * rise / (nz * apix)
        center_fraction = max(rise / (nz * apix), min(1.0, center_fraction))
        data = helicon.apply_helical_symmetry(
            data=data,
            apix=apix,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=csym,
            fraction=center_fraction,
            new_size=(new_nz, new_nxy, new_nxy),
            new_apix=new_apix,
            cpu=args.cpu,
        )
        apix = new_apix
        nz, ny, nx = data.shape
    return data, apix, nx, ny, nz
