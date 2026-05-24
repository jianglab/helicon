"""Handler for the copyParm option."""

from __future__ import annotations
import logging
import helicon
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


option_name = "copyParm"


def add_args(parser):
    parser.add_argument(
        "--copyParm",
        metavar="<starfile< var ~var ...>>",
        type=str,
        nargs="+",
        help="copy the specified parameters or all parameters if no var is specified. ~var will skip copying var",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the copyParm option.

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
    if len(param) >= 1:
        starFile = param[0]
        try:
            vars = param[1:]
        except:
            vars = []  # copy all parameters if a list is not specified

        data.convention = "relion"
        data = data.drop_duplicates(subset=["rlnImageName"], keep="last", inplace=False)

        data2 = helicon.images2dataframe(
            starFile,
            alternative_folders=args.folder,
            ignore_bad_particle_path=args.ignoreBadParticlePath,
            ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
            warn_missing_ctf=0,
            target_convention="relion",
        )
        data2 = data2.drop_duplicates(
            subset=["rlnImageName"], keep="last", inplace=False
        )
        if args.verbose > 1:
            logger.info(("\tRead in %d particles from %s" % (len(data2), starFile)))

        if len(data) > len(data2):
            raise HeliconError(
                "\\tERROR: --copyParm option requires that %s (%d) has the same number or more particles (>=%d needed)"
            )

        if not (set(data[["rlnImageName"]]).issubset(set(data2[["rlnImageName"]]))):
            raise HeliconError(
                "\\tERROR: --copyParm option requires that %s contains identical set or a superset of particles"
            )

        if len(vars):
            copyVars = [v for v in vars if v[0] != "~"]
            skipVars = [v[1:] for v in vars if v[0] == "~"]
            if skipVars and args.verbose > 1:
                logger.info(("\tSkipping parameters: %s" % (" ".join(skipVars))))
            if len(copyVars):
                invalidParms = [v for v in copyVars if v not in data2]
                if len(invalidParms):
                    logger.warning(
                        'parameters "%s" not in the list of parameters %s from file %s. ignored',
                        " ".join(invalidParms),
                        list(data2.columns),
                        starFile,
                    )
                validParms = [v for v in copyVars if v in data2]
            else:
                validParms = [v for v in data2 if v not in skipVars + ["rlnImageName"]]
        else:
            validParms = [v for v in data2 if v not in ["rlnImageName"]]

        if args.verbose > 1:
            logger.info(("\tCopying parameters: %s" % (" ".join(validParms))))

        for v in validParms:
            if v not in data:
                data.loc[:, v] = np.nan

        from helicon import convert_dataframe_file_path

        data.loc[:, "rlnImageName_abs"] = convert_dataframe_file_path(
            data, "rlnImageName", to="abs"
        )
        data2.loc[:, "rlnImageName_abs"] = convert_dataframe_file_path(
            data2, "rlnImageName", to="abs"
        )
        data.set_index(["rlnImageName_abs"], inplace=True)
        data2.set_index(["rlnImageName_abs"], inplace=True)
        data[validParms] = data2.loc[data.index, validParms]
        data.reset_index(drop=True, inplace=True)
        index_d[option_name] += 1
    return data, index_d
