"""Handler for the keepOneParticlePerHelicalTube option."""

from __future__ import annotations
import helicon
import logging

logger = logging.getLogger(__name__)


option_name = "keepOneParticlePerHelicalTube"


def add_args(parser):
    parser.add_argument(
        "--keepOneParticlePerHelicalTube",
        metavar="<0|1>",
        type=int,
        help="keep only one segment of each helical filament/tube",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the keepOneParticlePerHelicalTube option.

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
        var = ""
        for v in "rlnMicrographName rlnImageName".split():
            if v in data:
                var = v
                break
        if not var:
            raise HeliconError(
                "\\tERROR: rlnMicrographName or rlnImageName must be available"
            )
        if "rlnHelicalTubeID" not in data:
            raise HeliconError("\\trlnHelicalTubeID is not available")

        if "@" in data[var].iloc[0]:
            tmp = data.loc[:, var].str.split("@", expand=True)
            var = "filename"
            data.loc[:, var] = tmp.iloc[:, 1]

        data = data.groupby(
            [var, "rlnHelicalTubeID"], as_index=False, sort=False
        ).first()
        if var == "filename":
            data.drop(["filename"], inplace=True, axis=1)
        if args.verbose > 1:
            logger.info(f"\t{len(data)} helices found")
    return data, index_d
