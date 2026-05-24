"""Handler for the multParm option."""

from __future__ import annotations
import logging
import helicon

logger = logging.getLogger(__name__)


option_name = "multParm"


def add_args(parser):
    parser.add_argument(
        "--multParm",
        metavar="<var> <val>",
        type=str,
        nargs=2,
        help="modify parameter: var*=val",
        action="append",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the multParm option.

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
    if len(param) == 2:
        var, val = param
        if var not in data:
            logger.error(
                "parameter %s does not exist. Cannot multiply it by another value", var
            )
        data[var] *= float(val)
        index_d[option_name] += 1
    return data, index_d
