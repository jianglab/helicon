"""Handler for the showTime option."""

from __future__ import annotations
import helicon
import os
import os
import logging

logger = logging.getLogger(__name__)


option_name = "showTime"


def add_args(parser):
    parser.add_argument(
        "--showTime",
        metavar="<attr>",
        type=str,
        help="include file create time of the attr in the output star file. disabled by default",
        default=None,
    )


def handle(data, args, index_d, param):
    """Handle the showTime option.

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
        if param in data:
            fileAttr = param
        else:
            fileAttr = helicon.first_matched_attr(
                data,
                attrs="rlnMicrographMovieName rlnMicrographName rlnImageName".split(),
            )
        tmpCol = helicon.unique_attr_name(data, attr_prefix=fileAttr)
        data.loc[:, tmpCol] = data[fileAttr].str.split("@", expand=True).iloc[:, -1]
        timeCol = f"{fileAttr}CreateTime"
        files = data.groupby(tmpCol, sort=False)
        for fileName, fileParticles in files:
            data.loc[fileParticles.index, timeCol] = os.path.getctime(fileName)
        data.drop(tmpCol, inplace=True, axis=1)
        if args.verbose > 1:
            logger.info(
                f"\tThe create time of {len(files):,} {fileAttr} files added to a new column {timeCol}"
            )
    return data, index_d
