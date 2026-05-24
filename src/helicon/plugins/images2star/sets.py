"""Handler for the sets option."""

from __future__ import annotations
import helicon
import logging

logger = logging.getLogger(__name__)


option_name = "sets"


def handle(data, args, index_d, param):
    """Handle the sets option.

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
    if param > 1:
        sets = param
        data = data[args.subset :: sets]
        if args.verbose > 1:
            logger.info("\t%d/%d: %d images selected" % (args.subset, sets, len(data)))
        index_d[option_name] += 1
    return data, index_d
