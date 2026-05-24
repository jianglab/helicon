"""Handler for the setParm option."""

from __future__ import annotations
import helicon


option_name = "setParm"


def add_args(parser):
    parser.add_argument(
        "--setParm",
        metavar=("<var> <val>"),
        type=str,
        nargs="+",
        help="set parameter var val pair for each image",
        action="append",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the setParm option.

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
        if len(param) % 2:
            raise HeliconError(
                "ERROR: you specified odd number of --setParm arguments. Only even number of arguments are allowed for var val pairs"
            )
        for i in range(len(param) // 2):
            var, val = param[2 * i : 2 * (i + 1)]
            if var in helicon.Relion_OpticsGroup_Parameters:
                try:
                    data.attrs["optics"].loc[:, var] = helicon.guess_data_type(val)(val)
                except:
                    data.loc[:, var] = helicon.guess_data_type(val)(val)
            else:
                data.loc[:, var] = helicon.guess_data_type(val)(val)
        index_d[option_name] += 1
    return data, index_d


import logging

logger = logging.getLogger(__name__)
