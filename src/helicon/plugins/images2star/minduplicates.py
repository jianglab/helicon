"""Handler for the minDuplicates option."""

from __future__ import annotations
import helicon
import logging

logger = logging.getLogger(__name__)


option_name = "minDuplicates"


def add_args(parser):
    parser.add_argument(
        "--minDuplicates",
        metavar="<n>",
        type=int,
        help="only keep images >=n duplicates. disabled by default",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the minDuplicates option.

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
    if param > 0:
        minN = param
        attr = None
        for a in "rlnImageName rlnMicrographName".split():
            if a in data:
                attr = a
                break
        if attr is None:
            raise HeliconError(
                "\\tERROR: required parameter (rlnImageName or rlnMicrographName) is not available"
            )

        from helicon import convert_dataframe_file_path

        tmp = convert_dataframe_file_path(data, attr, to="abs")
        retained = tmp.map(tmp.value_counts() >= minN)
        data2 = data[retained]
        if len(data2) < 1:
            raise HeliconError("\\tWarning: no image is retained")
        data2 = data2.drop_duplicates([attr])
        if args.verbose > 1:
            logger.info(f"\t{len(data2)} images retained")
        data = data2.reset_index(drop=True)  # important to do this
    return data, index_d
