"""Handler for the changePixelSize option."""

from __future__ import annotations
import argparse
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError
import logging

logger = logging.getLogger(__name__)


option_name = "changePixelSize"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the changePixelSize option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--changePixelSize",
        type=float,
        metavar="<Angstrom>",
        help="change the pixel size to this value. Adjust defocus and Cs accordingly. disabled by default",
        default=0,
    )


def handle(
    data,
    args: argparse.Namespace,
    index_d: dict,
    param: object,
    output_title: str,
    output_slots: set,
    exp_group_id_name: str,
    micrograph_name: str,
    original_exp_group_ids: list,
):
    """Handle the changePixelSize option.

    Parameters
    ----------
    data : Dataset
        The cryosparc Dataset.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter value for this option.
    output_title : str
        Title for output filename construction.
    output_slots : set
        Output slot names.
    exp_group_id_name : str
        Name of the exposure group ID column.
    micrograph_name : str
        Name of the micrograph name column.
    original_exp_group_ids : list
        Original exposure group IDs.

    Returns
    -------
    tuple
        (data, output_title, output_slots, index_d) after processing.
    """
    if param > 0:
        col_apix = "blob/psize_A"
        if col_apix not in data:
            raise HeliconError(f"required parameter {col_apix} is not available")
        apix_orig = data[col_apix][0]
        apix_new = param
        data[col_apix] = apix_new

        for col in ["ctf/df1_A", "ctf/df2_A"]:
            if col in data:
                data[col] *= (apix_new / apix_orig) ** 2
        for col in ["ctf/cs_mm"]:
            if col in data:
                data[col] *= (apix_new / apix_orig) ** 4

        if args.verbose > 1:
            logger.info(f"\tPixel size: {apix_orig:.4f} -> {apix_new} Angstrom/pixel")
    return data, output_title, output_slots, index_d
