"""Handler for the replaceStr option."""

from __future__ import annotations
import logging
import helicon

logger = logging.getLogger(__name__)


option_name = "replaceStr"


def add_args(parser):
    parser.add_argument(
        "--replaceStr",
        metavar=("<var>", "<original str>", "<new str>"),
        type=str,
        nargs=3,
        help="replace substr in the variable with new str",
        action="append",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the replaceStr option.

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
    if len(param) == 3:
        var, oldStr, newStr = param
        if var in data:
            data[var] = data[var].str.replace(oldStr, newStr)
        else:
            logger.warning("variable %s does not exist. Skipped", var)
        index_d[option_name] += 1
    return data, index_d
