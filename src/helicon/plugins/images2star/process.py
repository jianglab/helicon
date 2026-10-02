"""Handler for the process option."""

from __future__ import annotations
import logging

from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


option_name = "process"


def add_args(parser):
    parser.add_argument(
        "--process",
        metavar="processor_name:param1=value1:param2=value2",
        type=str,
        nargs="+",
        action="append",
        help="no longer supported: it needed EMAN2's image processors.",
    )


def handle(data, args, index_d, param):
    """Handle the process option.

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
        # The processors were EMAN2's (EMData.process); helicon no longer
        # uses EMAN2, so the option stops with a clear message instead of
        # failing on a missing name halfway through the particles.
        raise HeliconError(
            "--process needs EMAN2's image processors, which helicon no longer "
            "supports. Use proc3d or EMAN2's e2proc2d.py instead."
        )
    return data, index_d
