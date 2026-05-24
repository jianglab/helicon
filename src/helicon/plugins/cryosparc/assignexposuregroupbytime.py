"""Handler for the assignExposureGroupByTime option."""

from __future__ import annotations
import argparse
import logging
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError

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

        if (
            time_group_size < 0 and len(source_group_ids) > 1
        ):  # combine previous groups (if there are) into a single group first
            if args.verbose > 1:
                logger.info(
                    f"\tCombining {len(source_group_ids)} exposure groups into 1 group"
                )
            data[exp_group_id_name] = 1
            source_group_ids = np.sort(np.unique(data[exp_group_id_name]))
            time_group_size = abs(time_group_size)

        software = helicon.guess_data_collection_software(data[micrograph_name][0])
        if software is None:
            logger.warning(
                "cannot detect the data collection software using %s: %s\n\tI only know the filenames by %s",
                micrograph_name,
                data[micrograph_name][0],
                ", ".join(sorted(helicon.movie_filename_patterns().keys())),
            )
            raise HeliconError("cannot detect data collection software")

        micrographs = np.unique(data[micrograph_name])
        micrograph_path_2_time = helicon.extract_timestamps(micrographs, software)
        last_group_id = 0
        new_particle_group_ids = np.zeros(len(data))
        for gi in source_group_ids:
            mask = np.where(data[exp_group_id_name] == gi)
            group_micrographs = np.unique(data[micrograph_name][mask])
            group_micrograph_time = [
                micrograph_path_2_time[m] for m in group_micrographs
            ]
            group_time_2_subgroup = helicon.assign_to_groups(
                group_micrograph_time, time_group_size
            )
            group_particle_2_subgroup = [
                group_time_2_subgroup[micrograph_path_2_time[m]]
                for m in data[micrograph_name][mask]
            ]
            new_particle_group_ids[mask] = (
                np.array(group_particle_2_subgroup) + last_group_id
            )
            last_group_id = np.max(new_particle_group_ids)
        data[exp_group_id_name] = new_particle_group_ids
        if len(exp_group_id_names_all) > 1:
            for attr in exp_group_id_names_all:
                if attr != exp_group_id_name:
                    data[attr] = data[exp_group_id_name]

        group_ids = np.sort(np.unique(data[exp_group_id_name]))
        for gi in group_ids:
            mask = np.where(data[exp_group_id_name] == gi)
            for (
                col
            ) in "ctf/cs_mm ctf/phase_shift_rad ctf/shift_A ctf/tilt_A ctf/trefoil_A ctf/tetra_A ctf/anisomag".split():
                if col in data:
                    data[col][mask] = np.median(data[col][mask])

        output_slots.add(exp_group_id_name.split("/")[0])
        output_title += f"->{len(group_ids)} time groups"

        if args.verbose > 1:
            logger.info(
                f"\t{len(source_group_ids)} -> {len(group_ids)} exposure groups"
            )
    return data, output_title, output_slots, index_d
