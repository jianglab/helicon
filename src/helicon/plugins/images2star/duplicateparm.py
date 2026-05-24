"""Handler for the duplicateParm option."""

from __future__ import annotations
import logging
import helicon

logger = logging.getLogger(__name__)


option_name = "duplicateParm"


def add_args(parser):
    parser.add_argument(
        "--duplicateParm",
        metavar="<from> <to>",
        type=str,
        nargs=2,
        help="duplicate parameter",
        action="append",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the duplicateParm option.

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
        for var_from, var_to in param:
            if var_from not in data:
                logger.warning(
                    "%s does not exist. Cannot duplicate %s to %s",
                    var_from,
                    var_from,
                    var_to,
                )
                continue
            if var_to in data:
                logger.warning(
                    "%s already exists. Will not duplicating %s to %s",
                    var_to,
                    var_from,
                    var_to,
                )
                continue
            data[var_to] = data[var_from]
        index_d[option_name] += 1
    return data, index_d
