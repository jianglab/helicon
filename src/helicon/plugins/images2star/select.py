"""Handler for the select option."""

from __future__ import annotations
import logging
import helicon
import pandas as pd

logger = logging.getLogger(__name__)


option_name = "select"


def add_args(parser):
    parser.add_argument(
        "--select",
        type=str,
        metavar=("<var>", "<val1<,val2>...>"),
        nargs=2,
        help="select images with exact matching of the specified variable value(s). disabled by default",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the select option.

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
        if var in data:
            vmin, vmax = data[var].min(), data[var].max()
            vals = val.split(",")
            if pd.api.types.is_integer_dtype(data[var]):
                vals = list(map(int, vals))
            elif pd.api.types.is_float_dtype(data[var]):
                vals = list(map(float, vals))
            data = data[data[var].isin(vals)]
            if len(data) < 1:
                raise HeliconError(
                    "WARNING: this selection has excluded all images. Hint: the actual data range is [{vmin}, {vmax}]"
                )
        else:
            if args.verbose:
                logger.warning(
                    'the variable "%s" specified by option "--select %s %s" does NOT exist',
                    var,
                    var,
                    val,
                )
        index_d[option_name] += 1
    return data, index_d
