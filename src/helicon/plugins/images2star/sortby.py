"""Handler for the sortby option."""

from __future__ import annotations
import logging
import helicon
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)

option_name = "sortby"


def add_args(parser):
    parser.add_argument(
        "--sortby",
        type=str,
        action="append",
        metavar="<parameter>",
        nargs="+",
        help="sort (small to large) by the specified parameter(s). disabled by default",
        default=[],
    )


def _sort_dataframe(data, sortby, ascending=True):
    """Sort dataframe by the given columns.

    Parameters
    ----------
    data : pd.DataFrame
    sortby : list of str
        Column names to sort by.
    ascending : bool
        Sort ascending (True) or descending (False).

    Returns
    -------
    pd.DataFrame
        Sorted dataframe.
    """
    badParms = [v for v in sortby if v not in data.columns]
    if badParms:
        if len(badParms) > 1:
            raise HeliconError(
                "\tERROR: parameters %s do not exist" % (" ".join(badParms))
            )
        else:
            raise HeliconError("\tERROR: parameter %s does not exist" % (badParms[0]))

    tmpCol = "tmp_sort_rlnImageName"
    if "rlnImageName" in sortby:
        tmp = data["rlnImageName"].str.split("@", expand=True)
        data[tmpCol] = tmp.iloc[:, -1] + "@" + tmp.iloc[:, 0]
        sortby = [tmpCol if v == "rlnImageName" else v for v in sortby]

    data_sorted = data.sort_values(sortby, ascending=ascending)
    if tmpCol in data_sorted.columns:
        data = data_sorted.drop(tmpCol, axis=1)
    else:
        data = data_sorted
    data.reset_index(drop=True, inplace=True)
    return data


def handle(data, args, index_d, param):
    """Handle the sortby option.

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
        data = _sort_dataframe(data, param, ascending=True)
        index_d[option_name] += 1
    return data, index_d
