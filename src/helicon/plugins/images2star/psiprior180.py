"""Handler for the psiPrior180 option."""

from __future__ import annotations
import logging
import helicon
import pandas as pd

logger = logging.getLogger(__name__)


option_name = "psiPrior180"


def add_args(parser):
    parser.add_argument(
        "--psiPrior180",
        metavar="<0|1>",
        type=int,
        help="duplicate data by adding 180 degrees to rlnAnglePsiPrior",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the psiPrior180 option.

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
        var = "rlnAnglePsiPrior"
        if var not in data:
            logger.error("parameter %s does not exist. Cannot add a value to it", var)
            sys.exist(-1)
        data1 = data
        data2 = data1.copy()
        data2.loc[:, var] += 180.0
        var = "rlnHelicalTubeID"
        if var in data2:
            idMax = data2[var].astype(int).max()
            idMax = helicon.ceil_power_of_10(idMax)
            data2.loc[:, var] += idMax
        data = pd.concat((data1, data2), axis=0)
        try:
            data.attrs = data1.attrs
        except:
            pass
        index_d[option_name] += 1
    return data, index_d
