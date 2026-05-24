"""Handler for the assignOpticGroupPerMicrograph option."""

from __future__ import annotations
import helicon
import pandas as pd
import logging

logger = logging.getLogger(__name__)


option_name = "assignOpticGroupPerMicrograph"


def add_args(parser):
    parser.add_argument(
        "--assignOpticGroupPerMicrograph",
        type=bool,
        metavar="<0|1>",
        help="assign images to optic groups, one group per micrograph. default to 0",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the assignOpticGroupPerMicrograph option.

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
    if param:
        try:
            optics_orig = data.attrs["optics"]
        except:
            optics_orig = None
        if optics_orig is None:
            raise HeliconError("\\tERROR: data_optics block must be available")

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

        tmp_col = "TEMP_image_name"
        data[tmp_col] = data[image_name].str.split("@", expand=True).iloc[:, -1]
        mgraphs = data.groupby(tmp_col, sort=False)

        optics = pd.concat([optics_orig.iloc[[0]]] * len(mgraphs), ignore_index=True)
        for gi, (mgraphName, mgraphData) in enumerate(mgraphs):
            data.loc[mgraphData.index, "rlnOpticsGroup"] = gi + 1
            new_row = optics_orig.copy().iloc[0]
            optics.loc[gi, "rlnOpticsGroup"] = gi + 1
            optics.loc[gi, "rlnOpticsGroupName"] = f"opticsGroup{gi+1}"
        data.attrs["optics"] = optics
        data.drop(tmp_col, axis=1, inplace=True)
        if args.verbose > 1:
            logger.info(
                f"\t{len(mgraphs)} micrographs -> {len(data.attrs['optics'])} optic groups"
            )
    return data, index_d
