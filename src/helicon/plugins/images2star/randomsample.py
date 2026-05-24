"""Handler for the randomSample option."""

from __future__ import annotations
import helicon


option_name = "randomSample"


def add_args(parser):
    parser.add_argument(
        "--randomSample",
        metavar="<n>",
        type=int,
        help="take random n images subset. disabled by default",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the randomSample option.

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
    if 0 < param < len(data):
        data = data.sample(args.randomSample)
        data.reset_index(drop=True, inplace=True)
        index_d[option_name] += 1
    return data, index_d


import logging

logger = logging.getLogger(__name__)
