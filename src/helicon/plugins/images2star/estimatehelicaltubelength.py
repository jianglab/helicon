"""Handler for the estimateHelicalTubeLength option."""

from __future__ import annotations
import helicon
from helicon.lib.analysis import estimate_helicalTube_length


option_name = "estimateHelicalTubeLength"


def add_args(parser):
    parser.add_argument(
        "--estimateHelicalTubeLength",
        metavar="<0|1>",
        type=int,
        help="estimate the length of each helical filament/tube",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the estimateHelicalTubeLength option.

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
        badParms = [
            v
            for v in "rlnImageName rlnHelicalTubeID rlnCoordinateX rlnCoordinateY".split()
            if v not in data
        ]
        if badParms:
            s = "s" if len(badParms) > 1 else ""
            raise HeliconError(
                "\\tERROR: parameter{s} {' '.join(badParms)} do not exist"
            )
        data = estimate_helicalTube_length(data, verbose=args.verbose)
    return data, index_d


import logging

logger = logging.getLogger(__name__)
