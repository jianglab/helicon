"""Handler for the copyExposureGroupAssignments option."""

from __future__ import annotations
import argparse
import logging
import helicon
import numpy as np
from pathlib import Path
from tqdm import tqdm
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


option_name = "copyExposureGroupAssignments"


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for the copyExposureGroupAssignments option.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to attach arguments to.
    """
    parser.add_argument(
        "--copyExposureGroupAssignments",
        type=str,
        metavar="<star file>",
        help="copy the optics group assignments from this star file. rlnMicrographMovieName and rlnOpticsGroup must be in this star file. disabled by default",
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
    """Handle the copyExposureGroupAssignments option.

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

        dataFrom = helicon.images2dataframe(
            inputFiles=param,
            ignore_bad_particle_path=True,
            ignore_bad_micrograph_path=True,
            warn_missing_ctf=0,
            target_convention="relion",
        )

        helicon.check_required_columns(
            dataFrom, required_cols=["rlnMicrographMovieName", "rlnOpticsGroup"]
        )
        dataFrom["rlnOpticsGroup"] = dataFrom["rlnOpticsGroup"].astype(int)
        dataFrom["rlnOpticsGroup"] = (
            dataFrom["rlnOpticsGroup"].astype(int)
            - np.min(dataFrom["rlnOpticsGroup"])
            + 1
        )
        mapping = {}
        for i, row in dataFrom.iterrows():
            mapping[Path(row["rlnMicrographMovieName"]).stem.split(".")[0]] = row[
                "rlnOpticsGroup"
            ]

        micrographs = np.unique(data[micrograph_name])
        for mi, m in tqdm(
            enumerate(micrographs),
            total=len(micrographs),
            desc="\tProcessing micrographs",
            unit="micrograph",
        ):
            group = 0
            for k, v in mapping.items():
                if m.find(k) != -1:
                    group = v
                    break
            mask = np.where(data[micrograph_name] == m)
            data[exp_group_id_name][mask] = group
            if group == 0:
                logger.warning(
                    "cannot find matching optics group info in %s for %s. Assign it to exposure group 0",
                    param,
                    m,
                )

        helicon.sync_group_columns(data, exp_group_id_name)

        group_ids = np.sort(np.unique(data[exp_group_id_name]))

        output_slots.add(exp_group_id_name.split("/")[0])
        output_title += (
            f"->{len(group_ids)} exposure groups copied from {Path(param).name}"
        )

        if args.verbose > 1:
            logger.info(
                f"\t{len(source_group_ids)} -> {len(group_ids)} exposure groups"
            )
    return data, output_title, output_slots, index_d
