"""Handler for the resetOpticGroup option."""

from __future__ import annotations
import helicon
import pandas as pd
from helicon.lib.exceptions import HeliconError
import logging

logger = logging.getLogger(__name__)


option_name = "resetOpticGroup"


def add_args(parser):
    parser.add_argument(
        "--resetOpticGroup",
        type=bool,
        metavar="<0|1>",
        help="reset all optics groups to a single group. disabled by default",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the resetOpticGroup option.

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

        n_orig = data["rlnOpticsGroup"].nunique()
        data["rlnOpticsGroup"] = 1

        optics = optics_orig.copy().iloc[0:0]
        new_row = optics_orig.iloc[[0]].copy()
        new_row["rlnOpticsGroup"] = 1
        new_row["rlnOpticsGroupName"] = "opticsGroup1"
        optics = pd.concat([optics, new_row], ignore_index=True)
        data.attrs["optics"] = optics

        if args.verbose > 1:
            logger.info(
                f"\t{n_orig} -> {data['rlnOpticsGroup'].nunique()} optics groups"
            )
    return data, index_d
