"""Handler for the keepParm option."""

from __future__ import annotations
import helicon


option_name = "keepParm"


def add_args(parser):
    parser.add_argument(
        "--keepParm",
        metavar="<var>",
        type=str,
        nargs="+",
        help="keep parameter var for each image, remove other parameters",
        action="append",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the keepParm option.

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
    if len(param):
        dropParms = [c for c in data if c not in param]
        data = data.drop(dropParms, inplace=False, axis=1)
        index_d[option_name] += 1
    return data, index_d


import logging

logger = logging.getLogger(__name__)
