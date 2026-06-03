"""Handler for the assignOpticGroupByBeamShiftLabel option."""

from __future__ import annotations
import helicon
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


option_name = "assignOpticGroupByBeamShiftLabel"


def add_args(parser):
    choices = "no auto EPU serialEM_pncc".split()
    parser.add_argument(
        "--assignOpticGroupByBeamShiftLabel",
        choices=choices,
        metavar=f"<{'|'.join(choices)}>",
        help="assign images to optic groups according to beam shift labels in filenames, one group per beam shift position. default to no",
        default="no",
    )


def handle(data, args, index_d, param):
    """Handle the assignOpticGroupByBeamShiftLabel option.

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
    if param != "no":
        # choices = "no auto EPU serialEM_pncc".split()
        try:
            optics_orig = data.attrs["optics"]
        except:
            optics_orig = None
        if optics_orig is None:
            raise HeliconError("\\tdata_optics block must be available")

        image_name = helicon.first_matched_attr(
            data,
            attrs="rlnMicrographMovieName rlnMicrographName rlnImageName".split(),
        )
        if image_name is None:
            raise HeliconError(
                "\\tERROR: rlnMicrographMovieName, rlnMicrographName or rlnImageName must be available"
            )

        required_cols = "rlnOpticsGroup".split()
        missing_cols = [c for c in required_cols if c not in data]
        if missing_cols:
            raise HeliconError(
                "\\tERROR: required attrs {' '.join(missing_cols)} must be available"
            )

        if param == "auto":
            format = helicon.guess_data_collection_software(
                filename=data[image_name].iloc[0]
            )
            if format is None:
                raise HeliconError(
                    "\\tERROR: cannot detect the format of filename {image_name}: {data[image_name].iloc[0]}"
                )
            else:
                if args.verbose > 1:
                    logger.info(
                        f"\tAuto-detect the format as {format} based on {image_name}: {data[image_name].iloc[0]}"
                    )
        else:
            format = param
            if (
                helicon.verify_data_collection_software(
                    data[image_name].iloc[0], format
                )
                is None
            ):
                raise HeliconError(
                    "\\tERROR: the specified format {format} is inconsistent with filename {image_name}: {data[image_name].iloc[0]}. If you are not sure, specify auto as the format and let me guess for you"
                )

        if format == "EPU_old":
            raise HeliconError(
                "\\tERROR: the old EPU data are not supported as the associated xml files are required to obtain the beam shifts."
            )

        optics = optics_orig.copy().iloc[0:0]

        tmp_col = "TEMP_beam_shift_pos"
        ogs = data.groupby("rlnOpticsGroup", sort=False)
        og_count = 0
        pattern = helicon.movie_filename_patterns()[format]
        for ogName, ogData in ogs:
            optics_row_index = optics_orig[
                optics_orig["rlnOpticsGroup"].astype(str) == str(ogName)
            ].last_valid_index()
            extracted = ogData.loc[:, image_name].str.extract(pattern)
            ogData[tmp_col] = extracted.get("beamshift", extracted.iloc[:, 0])
            if format in ["EPU", "serialEM_embl_heidelberg", "serialEM_cuhksz"]:
                ogData[tmp_col] = ogData[tmp_col].astype(int)
            else:
                ogData[tmp_col] = ogData[tmp_col].astype(str)
            unique_beam_shift_pos = sorted(ogData[tmp_col].unique())
            n = len(unique_beam_shift_pos)
            mapping = {
                p: pi + 1 + og_count for pi, p in enumerate(unique_beam_shift_pos)
            }
            if args.verbose > 10:
                logger.info(f"{mapping=}")
            data.loc[ogData.index, "rlnOpticsGroup"] = ogData[tmp_col].map(mapping)
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
