"""Handler for the randomSample option."""

from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)

option_name = "randomSample"


def add_args(parser):
    parser.add_argument(
        "--randomSample",
        metavar="<n>",
        type=int,
        help="take random n images per rlnRandomSubset. disabled by default",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the randomSample option.

    Samples *param* particles per ``rlnRandomSubset`` group.

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
    if 0 < param < len(data):
        idx = []
        for _, g in data.groupby("rlnRandomSubset", sort=False):
            n = min(param, len(g))
            idx.extend(g.sample(n=n).index.tolist())
        data = data.loc[idx].reset_index(drop=True)
        n_total = len(data)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"\trandom {param} images per rlnRandomSubset: {n_total} total")
        index_d[option_name] += 1
    return data, index_d
