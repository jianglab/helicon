"""Handler for the rsortby option."""

from __future__ import annotations
import logging
from .sortby import _sort_dataframe

logger = logging.getLogger(__name__)

option_name = "rsortby"


def add_args(parser):
    parser.add_argument(
        "--rsortby",
        type=str,
        action="append",
        metavar="<parameter>",
        nargs="+",
        help="reverse sort (large to small) by the specified parameter(s). disabled by default",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the rsortby option.

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
        data = _sort_dataframe(data, param, ascending=False)
        index_d[option_name] += 1
    return data, index_d
