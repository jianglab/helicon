"""Handler for the splitByMicrograph option."""

from __future__ import annotations
import argparse
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError
import logging

logger = logging.getLogger(__name__)


option_name = "splitByMicrograph"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the splitByMicrograph option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--splitByMicrograph",
        type=int,
        metavar="<0|1>",
        help="split the dataset by micrograph. disabled by default",
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
    """Handle the splitByMicrograph option.

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
    if param:
        col_mid = "location/micrograph_uid"
        mids = np.unique(data[col_mid])
        masks = [data[col_mid] == mid for mid in mids]
        counts = [np.sum(m) for m in masks]
        group1, group2 = helicon.split_array(counts)

        col_split = "alignments3D/split"
        if col_split not in data:
            data.add_fields([col_split], ["u4"])

        for gi, g in enumerate([group1, group2]):
            for mid_index in g:
                data[col_split][masks[mid_index]] = gi

        output_slots.add("alignments3D")
        output_title += f"->per-micrograph split"

        if args.verbose > 1:
            logger.info(
                f"\twhole  dataset: {len(mids)} micrographs, {len(data)} particles"
            )
            logger.info(
                f"\thalf dataset 1: {len(group1)} micrographs, {np.sum(data[col_split]==0)} particles"
            )
            logger.info(
                f"\thalf dataset 2: {len(group2)} micrographs, {np.sum(data[col_split]==1)} particles"
            )
    return data, output_title, output_slots, index_d
