"""Handler for the selectRatioRange option."""

from __future__ import annotations
import helicon
import pandas as pd
from helicon.lib.exceptions import HeliconError
import logging

logger = logging.getLogger(__name__)


option_name = "selectRatioRange"


def add_args(parser):
    parser.add_argument(
        "--selectRatioRange",
        type=str,
        metavar=("<var>", "<ratio min>", "<ratio max>"),
        nargs=3,
        help="select images with the variable value in the specified ratio range. disabled by default",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the selectRatioRange option.

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
            else:
                if args.verbose:
                    raise HeliconError(
                        'ERROR: the variable "%s" specified by option "--selectRatioRange %s %s %s" is NOT a number type'
                        % (var, var, val1, val2)
                    )
            data[var] = data[var].astype(float)
            val1 = float(val1)
            val2 = float(val2)
            if val1 == 0:
                valmin = data[var].min()
            else:
                valmin = data[var].nsmallest(int(len(data) * val1)).iloc[-1]
            if val2 == 1:
                valmax = data[var].max() + 0.1
            else:
                valmax = data[var].nsmallest(int(len(data) * val2) + 1).iloc[-1]
            data = data.loc[(data[var] >= valmin) & (data[var] < valmax)]
            if len(data) < 1:
                raise HeliconError(
                    f"WARNING: this selection has excluded all images. Hint: the actual data range is [{vmin}, {vmax}]"
                )
        elif var.lower() == "index":
            val1 = int(round(float(val1) * len(data)))
            val2 = int(round(float(val2) * len(data)))
            if val1 < 0:
                val1 = 0
            if val2 < 0:
                val2 = len(data)
            data = data.iloc[val1:val2]
        else:
            if args.verbose:
                raise HeliconError(
                    '\tERROR: the variable "%s" specified by option "--selectRatioRange %s %s %s" does NOT exist'
                    % (var, var, val1, val2)
                )
        if args.verbose > 1:
            logger.info("\t%d images selected" % (len(data)))
        if not len(data):
            raise HeliconError(
                "Nothing to do when there is no particle image left. I will quit"
            )
        index_d[option_name] += 1
    return data, index_d
