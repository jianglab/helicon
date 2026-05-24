"""Handler for the delParm option."""

from __future__ import annotations
import logging
import helicon

logger = logging.getLogger(__name__)


option_name = "delParm"


def add_args(parser):
    parser.add_argument(
        "--delParm",
        metavar="<var>",
        type=str,
        nargs="+",
        action="append",
        help="remove parameter var for each image",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the delParm option.

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
        invalidParms = []
        dropParms = []
        for p in param:
            p = p.strip("_")
            if p in data:
                dropParms.append(p)
            else:
                invalidParms.append(p)
        if invalidParms:
            logger.warning("%s do not exist", invalidParms)
        if dropParms:
            data = data.drop(dropParms, inplace=False, axis=1)
        index_d[option_name] += 1

    # copy parameters in the other star file
    return data, index_d
