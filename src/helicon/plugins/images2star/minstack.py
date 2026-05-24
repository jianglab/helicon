"""Handler for the minStack option."""

from __future__ import annotations
import helicon
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


option_name = "minStack"


def add_args(parser):
    parser.add_argument(
        "--minStack",
        type=int,
        metavar="<0|1>",
        help="generate a new set of mrcs files including only the subset of particles in this stack. default to 0",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the minStack option.

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
        tmp = data["rlnImageName"].str.split("@", expand=True)
        indices, micrographNames = tmp.iloc[:, 0], tmp.iloc[:, -1]

        mgraphs = micrographNames.groupby(micrographNames, sort=False)

        count = 0

        subdir = os.path.splitext(args.output_starFile)[0]
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        for mgraphName, mgraphParticles in mgraphs:
            mgraphName2 = os.path.join(subdir, os.path.basename(mgraphName))
            n = len(mgraphParticles)
            if not (
                os.path.exists(mgraphName2)
                and helicon.EMUtil.get_image_count(mgraphName2) == n
            ):
                particles_indices = sorted(
                    list(indices.iloc[mgraphParticles.index].astype(int))
                )
                for i in range(n):
                    i2 = particles_indices[i] - 1
                    d.read_image(mgraphName, i2)
                    d.write_image(mgraphName2, i)
            rlnImageName = (
                pd.Series(list(range(1, n + 1))).map("{:06d}".format)
                + "@"
                + mgraphName2
            )
            data.loc[mgraphParticles.index, "rlnImageName"] = (
                rlnImageName.values
            )  # critical to use values, otherwise the index-aligned assignment will mess up the values
            count += 1
            if args.verbose:
                logger.info(
                    "\t%d/%d: %d/%d images in %s saved to %s"
                    % (
                        count,
                        len(mgraphs),
                        len(mgraphParticles),
                        helicon.EMUtil.get_image_count(mgraphName),
                        mgraphName,
                        mgraphName2,
                    )
                )
        index_d[option_name] += 1
    return data, index_d
