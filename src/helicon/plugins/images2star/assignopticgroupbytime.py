"""Handler for the assignOpticGroupByTime option."""

from __future__ import annotations

import logging
import helicon
from helicon.lib.exceptions import HeliconError
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


option_name = "assignOpticGroupByTime"


def add_args(parser):
    parser.add_argument(
        "--assignOpticGroupByTime",
        type=int,
        metavar="<n>",
        help="assign images to optic groups according to data collection time, n movies per group. disabled by default",
        default=-1,
    )


def handle(data, args, index_d, param):
    """Handle the assignOpticGroupByTime option.

    Parameters
    ----------
    data : pd.DataFrame
        The particle data DataFrame.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter value for this option.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (data, index_d) after processing.
    """
    if param > 0:
        try:
            optics_orig = data.attrs["optics"]
        except Exception:
            optics_orig = None
        if optics_orig is None:
            raise HeliconError("\tdata_optics block must be available")

        image_name = helicon.first_matched_attr(
            data,
            attrs="rlnMicrographMovieName rlnMicrographName rlnImageName".split(),
        )
        if image_name is None:
            raise HeliconError(
                "\tERROR: rlnMicrographMovieName, rlnMicrographName or rlnImageName must be available"
            )

        software = helicon.guess_data_collection_software(
            filename=data[image_name].iloc[0]
        )
        if software in ["EPU", "EPU_old"]:
            required_cols = "rlnOpticsGroup".split()
            if args.verbose > 2:
                logger.info(
                    f"\tIt appears that you used EPU to collect the movies. Data collection time will be extracted from the file names specified in the {image_name} column"
                )
        else:
            required_cols = "rlnOpticsGroup".split()
            if "rlnMicrographMovieName" in data:
                image_name = "rlnMicrographMovieName"
                if args.verbose > 2:
                    logger.info(
                        "Data collection time will use the file modification time of the movie files specified in the rlnMicrographMovieName column. Make sure that the file modification times are indeed the movie collection times"
                    )
            elif args.verbose > 2:
                logger.info(
                    f"\tData collection time will be extracted from the {image_name} column (rlnMicrographMovieName not available)"
                )

        missing_cols = [c for c in required_cols if c not in data]
        if missing_cols:
            missing_str = " ".join(missing_cols)
            raise HeliconError(
                f"\tERROR: required attrs {missing_str} must be available. "
                "This task (by time) can use movie, micrograph or particle filenames. "
                "The actual time in the filename is the first choice; if not, "
                "the serial number in filename can be used as the proxy; "
                "if neither time nor serial number is available, use file modification time."
            )

        movies = data[image_name].values
        source_group_ids = np.sort(data["rlnOpticsGroup"].unique())
        group_id_lookup = data["rlnOpticsGroup"].values

        _, moive2time, moive2time_str = helicon.assign_time_groups(
            micrographs=movies,
            source_group_ids=source_group_ids,
            group_id_lookup=group_id_lookup,
            time_group_size=param,
            verbose=args.verbose,
        )

        optics = optics_orig.copy().iloc[0:0]

        ogs = data.groupby("rlnOpticsGroup", sort=False)
        og_count = 0
        for ogName, ogData in ogs:
            optics_row_index = optics_orig[
                optics_orig["rlnOpticsGroup"].astype(str) == str(ogName)
            ].last_valid_index()
            times = [moive2time[m] for m in ogData[image_name].unique()]
            time2group = helicon.assign_to_groups(times, param)
            movie2group = {
                m: time2group[moive2time[m]] + og_count
                for m in ogData[image_name].unique()
            }
            if args.verbose > 10:
                logger.info(
                    f"{movie2group=} {len(movie2group)=} {len(set(movie2group.values()))=}"
                )
            data.loc[ogData.index, "rlnOpticsGroup"] = ogData[image_name].map(
                movie2group
            )
            data.loc[ogData.index, "rlnMovieCollectionTime"] = ogData[image_name].map(
                moive2time_str
            )
            n = len(data.loc[ogData.index, "rlnOpticsGroup"].unique())
            new_rows = pd.concat(
                [optics_orig.iloc[[optics_row_index]]] * n, ignore_index=True
            )
            new_rows["rlnOpticsGroup"] = np.arange(
                og_count + 1, og_count + 1 + n, dtype=int
            )
            new_rows["rlnOpticsGroupName"] = "opticsGroup" + new_rows[
                "rlnOpticsGroup"
            ].astype(str)
            optics = pd.concat([optics, new_rows], ignore_index=True)
            og_count += n
        data.attrs["optics"] = optics
        if args.verbose > 1:
            logger.info(f"\t{len(ogs)} optics groups -> {len(optics)} optic groups")
    return data, index_d
