"""Handler for the assignExposureGroupByBeamShiftLabel option."""

from __future__ import annotations
import argparse
import logging
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


option_name = "assignExposureGroupByBeamShiftLabel"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the assignExposureGroupByBeamShiftLabel option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--assignExposureGroupByBeamShiftLabel",
        type=str,
        metavar="0|1",
        help="assign images to exposure groups by beam shift label from filenames. One group per distinct beam shift value. disabled by default",
        default=None,
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
    """Handle the assignExposureGroupByBeamShiftLabel option.

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
    if param is not None and param != "0":
        source_group_ids = np.sort(np.unique(data[exp_group_id_name]))

        software = helicon.guess_data_collection_software(data[micrograph_name][0])
        if software is None:
            logger.warning(
                "cannot detect the data collection software using %s: %s\n\tI only know the filenames by %s",
                micrograph_name,
                data[micrograph_name][0],
                ", ".join(sorted(helicon.movie_filename_patterns().keys())),
            )
            raise HeliconError("cannot detect data collection software")

        micrographs = np.sort(np.unique(data[micrograph_name]))

        if software in ["EPU", "serialEM_pncc", "serialEM_embl_heidelberg"]:
            micrograph_to_beamshift_clusters = helicon.assign_beamshift_groups(
                micrographs, software
            )
        else:
            logger.warning(
                "software %s does not have a beam shift label in its filenames. Try --assignExposureGroupByTime instead.",
                software,
            )
            raise HeliconError(
                f"software {software} does not have a beam shift label in its filenames"
            )

        exposure_groups = [
            micrograph_to_beamshift_clusters[row[micrograph_name]]
            for row in data.rows()
        ]
        data[exp_group_id_name] = helicon.combine_groups(
            data[exp_group_id_name], np.array(exposure_groups)
        )

        helicon.sync_group_columns(data, exp_group_id_name)
        helicon.propagate_ctf_median(data, exp_group_id_name)

        group_ids = np.sort(np.unique(data[exp_group_id_name]))

        output_slots.add(exp_group_id_name.split("/")[0])
        output_title += (
            f" {len(source_group_ids)}->{len(group_ids)} beamshift label groups"
        )

        if args.verbose > 1:
            logger.info(
                f"\t{len(source_group_ids)} -> {len(group_ids)} exposure groups stored in {' '.join(helicon.all_matched_attrs(data, query_str='exp_group_id'))}"
            )
    return data, output_title, output_slots, index_d
