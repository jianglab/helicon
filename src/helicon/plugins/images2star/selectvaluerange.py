"""Handler for the selectValueRange option."""

from __future__ import annotations
import logging
import helicon
import pandas as pd

logger = logging.getLogger(__name__)


option_name = "selectValueRange"


def add_args(parser):
    parser.add_argument(
        "--selectValueRange",
        type=str,
        metavar=("<var>", "<valmin>", "<valmax>"),
        nargs=3,
        help="select images with the variable value in the specified range. disabled by default",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the selectValueRange option.

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
        var, val1, val2 = param
        if var in data:
            vmin, vmax = data[var].min(), data[var].max()
            if pd.api.types.is_integer_dtype(data[var]):
                val1 = int(val1)
                val2 = int(val2)
            elif pd.api.types.is_float_dtype(data[var]):
                val1 = float(val1)
                val2 = float(val2)
            data = data.loc[(data[var] > val1) & (data[var] < val2)]
            if len(data) < 1:
                raise HeliconError(
                    "WARNING: this selection has excluded all images. Hint: the actual data range is [{vmin}, {vmax}]"
                )
        else:
            if args.verbose:
                logger.warning(
                    'the variable "%s" specified by option "--selectValueRange %s %s %s" does NOT exist',
                    var,
                    var,
                    val1,
                    val2,
                )
        index_d[option_name] += 1
    return data, index_d
