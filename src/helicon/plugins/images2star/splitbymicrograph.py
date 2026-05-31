"""Handler for the splitByMicrograph option."""

from __future__ import annotations
import helicon
from pathlib import Path
from helicon.lib.exceptions import HeliconExit
import logging

logger = logging.getLogger(__name__)


option_name = "splitByMicrograph"


def add_args(parser):
    parser.add_argument(
        "--splitByMicrograph",
        type=bool,
        metavar="<0|1>",
        help="split the output into separate star files, one per micrograph. default to 0",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the splitByMicrograph option.

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
        if "rlnMicrographName" in data:
            micrographNames = data["rlnMicrographName"]
        else:
            micrographNames = (
                data["rlnImageName"].str.split("@", expand=True).iloc[:, -1]
            )
        mgraphs = micrographNames.groupby(micrographNames, sort=False)

        count = 0

        prefix = Path(args.output_starFile).stem
        for mgraphName, mgraphParticles in mgraphs:
            tmpStarFile = "%s.%s.star" % (
                prefix,
                Path(mgraphName).stem,
            )
            tmpdata = data.loc[mgraphParticles.index]
            helicon.dataframe2file(tmpdata, tmpStarFile)
            count += 1
            if args.verbose > 1:
                logger.info(
                    "\t%d/%d: %d images saved to %s"
                    % (count, len(mgraphs), len(mgraphParticles), tmpStarFile)
                )
        raise HeliconExit()
    return data, index_d
