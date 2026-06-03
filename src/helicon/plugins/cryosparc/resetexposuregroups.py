"""Handler for the resetExposureGroups option."""

from __future__ import annotations
import argparse
import helicon
import numpy as np
import logging

logger = logging.getLogger(__name__)


option_name = "resetExposureGroups"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the resetExposureGroups option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--resetExposureGroups",
        type=bool,
        metavar="<0|1>",
        help="reset all exposure groups to a single group. disabled by default",
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
    """Handle the resetExposureGroups option.

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
        source_group_ids = np.sort(np.unique(data[exp_group_id_name]))
        data[exp_group_id_name] = 1

        helicon.sync_group_columns(data, exp_group_id_name)

        group_ids = np.sort(np.unique(data[exp_group_id_name]))
        output_slots.add(exp_group_id_name.split("/")[0])
        output_title += f"->{len(group_ids)} group"
        if args.verbose > 1:
            logger.info(
                f"\t{len(source_group_ids)} -> {len(group_ids)} exposure groups"
            )
    return data, output_title, output_slots, index_d
