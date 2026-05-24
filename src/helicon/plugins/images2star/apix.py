"""Handler for the apix option."""

from __future__ import annotations
import helicon
from helicon.lib.io import getPixelSize, setPixelSize


option_name = "apix"


def add_args(parser):
    parser.add_argument(
        "--apix",
        type=float,
        metavar="<A/pixel>",
        help="set mag to have this sampling",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the apix option.

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
        setPixelSize(data, apix_new=param)
        index_d[option_name] += 1
    return data, index_d


import logging

logger = logging.getLogger(__name__)
