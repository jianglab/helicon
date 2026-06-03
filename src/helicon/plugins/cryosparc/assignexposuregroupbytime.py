"""Handler for the assignExposureGroupByTime option."""

from __future__ import annotations
import argparse
import logging
import helicon
import numpy as np

logger = logging.getLogger(__name__)


option_name = "assignExposureGroupByTime"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the assignExposureGroupByTime option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--assignExposureGroupByTime",
        type=int,
        metavar="<n>",
        help="assign images to exposure groups according to data collection time, n movies per group. disabled by default",
        default=-1,
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
    """Handle the assignExposureGroupByTime option.

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
    if abs(param) > 0:
        time_group_size = param
        source_group_ids = np.sort(np.unique(data[exp_group_id_name]))

        # Negative param: merge existing groups into one before splitting by time
        if time_group_size < 0 and len(source_group_ids) > 1:
            if args.verbose > 1:
                logger.info(
                    f"\tCombining {len(source_group_ids)} exposure groups into 1 group"
                )
            data[exp_group_id_name] = 1
            source_group_ids = np.sort(np.unique(data[exp_group_id_name]))
            time_group_size = abs(time_group_size)

        micrographs = np.asarray(data[micrograph_name])
        new_group_ids, _, _ = helicon.assign_time_groups(
            micrographs=micrographs,
            source_group_ids=source_group_ids,
            group_id_lookup=data[exp_group_id_name],
            time_group_size=time_group_size,
            verbose=args.verbose,
            use_mtime_fallback=None,
        )
        data[exp_group_id_name] = new_group_ids

        helicon.sync_group_columns(data, exp_group_id_name)
        helicon.propagate_ctf_median(data, exp_group_id_name)

        group_ids = np.sort(np.unique(data[exp_group_id_name]))

        output_slots.add(exp_group_id_name.split("/")[0])
        output_title += f"->{len(group_ids)} time groups"

        if args.verbose > 1:
            logger.info(
                f"\t{len(source_group_ids)} -> {len(group_ids)} exposure groups"
            )
    return data, output_title, output_slots, index_d
