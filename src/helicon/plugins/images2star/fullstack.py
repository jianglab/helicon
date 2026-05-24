"""Handler for the fullStack option."""

from __future__ import annotations
import helicon
import pandas as pd
import logging

logger = logging.getLogger(__name__)


option_name = "fullStack"


def add_args(parser):
    parser.add_argument(
        "--fullStack",
        type=int,
        metavar="<0|1>",
        help="generate a star stack including all particles in the referenced image files. default to 0",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the fullStack option.

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
        valid_cols = set(
            "rlnVoltage rlnDefocusU rlnDefocusV rlnDefocusAngle rlnSphericalAberration rlnDetectorPixelSize rlnMagnification rlnAmplitudeContrast rlnMicrographName rlnGroupName rlnGroupNumber".split()
        )
        cols_to_keep = [c for c in data if c in valid_cols]

        micrographNames = data["rlnImageName"].str.split("@", expand=True).iloc[:, -1]

        mgraphs = micrographNames.groupby(micrographNames, sort=False)

        dataframes = []
        count = 0
        for mgraphName, mgraphParticles in mgraphs:
            n = helicon.EMUtil.get_image_count(mgraphName)
            rlnImageName = (
                pd.Series(list(range(1, n + 1))).map("{:06d}".format) + "@" + mgraphName
            )
            df = pd.DataFrame()
            df["rlnImageName"] = rlnImageName
            tmpdf = data.loc[mgraphParticles.index]
            for ci, c in enumerate(cols_to_keep):
                df[c] = tmpdf[c].values[0]
            dataframes.append(df)
            count += 1
            if args.verbose:
                logger.info(
                    "\t%d/%d: %s:\t%d -> %d images"
                    % (count, len(mgraphs), mgraphName, len(mgraphParticles), n)
                )
        data = pd.concat(dataframes)
        data = data.reset_index(drop=True)  # important to do this
        index_d[option_name] += 1
    return data, index_d
