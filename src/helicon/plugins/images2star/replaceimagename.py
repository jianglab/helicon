"""Handler for the replaceImageName option."""

from __future__ import annotations
import helicon
import pandas as pd
from pathlib import Path


option_name = "replaceImageName"


def add_args(parser):
    parser.add_argument(
        "--replaceImageName",
        metavar="<new mrcs file>",
        type=str,
        help="replace rlnImageName column by the provided mrcs file that has the same number of particles",
        default="",
    )


def handle(data, args, index_d, param):
    """Handle the replaceImageName option.

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
        replaceImageName = param

        if not Path(replaceImageName).exists():
            raise HeliconError("\\tERROR: %s does not exist")

        nImage = helicon.EMUtil.get_image_count(replaceImageName)
        if nImage != len(data):
            raise HeliconError(
                "\\tERROR: {replaceImageName} contains {len(nImage)} particles, different from the expected {len(data)} particles"
            )

        data["rlnImageName"] = (
            pd.Series(list(range(1, nImage + 1))).map("{:06d}".format)
            + "@"
            + replaceImageName
        )
        index_d[option_name] += 1
    return data, index_d


import logging

logger = logging.getLogger(__name__)
