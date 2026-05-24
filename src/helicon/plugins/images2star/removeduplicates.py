"""Handler for the removeDuplicates option."""

from __future__ import annotations
import logging
import helicon

logger = logging.getLogger(__name__)


option_name = "removeDuplicates"


def add_args(parser):
    parser.add_argument(
        "--removeDuplicates",
        metavar="<var>",
        nargs="+",
        type=str,
        help="remove images with duplicate parameters. disabled by default",
        default="",
    )


def handle(data, args, index_d, param):
    """Handle the removeDuplicates option.

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
        vars = [v for v in param if v not in data]
        if vars:
            logger.warning("%s are not valid parameters", vars)
        vars = [v for v in param if v in data]
        if len(vars) < 1:
            logger.info(f"\tnothing to do when no valid parameter is provided")
        else:
            data2 = data.drop_duplicates(vars)
            if args.verbose:
                logger.info(
                    f"{len(data2)} image retained after removing {len(data) - len(data2)} images with duplicate {vars}"
                )
            data = data2.reset_index(drop=True)  # important to do this
    return data, index_d
