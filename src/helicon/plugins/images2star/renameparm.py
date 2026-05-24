"""Handler for the renameParm option."""

from __future__ import annotations
import logging
import helicon

logger = logging.getLogger(__name__)


option_name = "renameParm"


def add_args(parser):
    parser.add_argument(
        "--renameParm",
        metavar="<old> <new>",
        type=str,
        nargs=2,
        help="rename parameter",
        action="append",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the renameParm option.

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
        cols = {}
        for parm in zip(*[iter(param)] * 2):
            var_old, var_new = parm
            if var_old not in data:
                logger.warning(
                    "%s does not exist. Cannot rename %s to %s",
                    var_old,
                    var_old,
                    var_new,
                )
                continue
            if var_new in data:
                logger.warning(
                    "%s already exists. Cannot duplicate %s to %s",
                    var_new,
                    var_old,
                    var_new,
                )
                continue
            cols[var_old] = var_new
        data.rename(columns=cols, inplace=True)
        index_d[option_name] += 1
    return data, index_d
