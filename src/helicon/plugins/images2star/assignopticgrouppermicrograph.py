"""Handler for the assignOpticGroupPerMicrograph option."""

from __future__ import annotations
import helicon
import pandas as pd
from helicon.lib.exceptions import HeliconError
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

        required_cols = "rlnOpticsGroup".split()
        missing_cols = [c for c in required_cols if c not in data]
        if missing_cols:
            raise HeliconError(
                f"\tERROR: required attrs {' '.join(missing_cols)} must be available"
            )

        micrograph_names = data[image_name].str.split("@", expand=True).iloc[:, -1]
        unique_names = micrograph_names.unique()
        mapping = helicon.per_micrograph_mapping(unique_names)
        data["rlnOpticsGroup"] = micrograph_names.map(mapping)

        optics = pd.concat(
            [optics_orig.iloc[[0]]] * len(unique_names), ignore_index=True
        )
        for gi, name in enumerate(unique_names):
            optics.loc[gi, "rlnOpticsGroup"] = gi + 1
            optics.loc[gi, "rlnOpticsGroupName"] = f"opticsGroup{gi+1}"
        data.attrs["optics"] = optics
        if args.verbose > 1:
            logger.info(
                f"\t{len(unique_names)} micrographs -> {len(data.attrs['optics'])} optic groups"
            )
    return data, index_d
