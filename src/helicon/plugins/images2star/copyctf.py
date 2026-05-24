"""Handler for the copyCtf option."""

from __future__ import annotations
import helicon
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


option_name = "copyCtf"


def add_args(parser):
    parser.add_argument(
        "--copyCtf",
        metavar="<starfile>",
        type=str,
        help="Star file to copy CTF parameters from. Should be a CTF refinement output file.",
    )


def handle(data, args, index_d, param):
    """Handle the copyCtf option.

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
        targetStarFile = param
        logger.info(f"Copying CTF parameters from {targetStarFile}")

        data.convention = "relion"
        data = data.drop_duplicates(subset=["rlnImageName"], keep="last", inplace=False)

        data2 = helicon.images2dataframe(
            targetStarFile,
            alternative_folders=args.folder,
            ignore_bad_particle_path=args.ignoreBadParticlePath,
            ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
            warn_missing_ctf=1,
            target_convention="relion",
        )
        data2 = data2.drop_duplicates(
            subset=["rlnImageName"], keep="last", inplace=False
        )

        if args.verbose > 1:
            logger.info(
                ("\tRead in %d particles from %s" % (len(data2), targetStarFile))
            )

        common_optics_groups = set(optics["rlnOpticsGroup"].values) & set(
            data2.attrs["optics"]["rlnOpticsGroup"].values
        )
        if common_optics_groups:
            # copy 'rlnBeamTiltX', 'rlnBeamTiltY', 'rlnOddZernike', 'rlnEvenZernike' from the same optics group
            ctf_parms_candidate = [
                "rlnBeamTiltX",
                "rlnBeamTiltY",
                "rlnOddZernike",
                "rlnEvenZernike",
            ]
            ctf_parms = []
            for key in ctf_parms_candidate:
                if key in data2.attrs["optics"]:
                    ctf_parms.append(key)
                    if key not in optics:
                        optics.loc[:, key] = 0
            if ctf_parms:
                # for optics_group in data2.attrs["optics"].loc[:,'rlnOpticsGroup']:
                #    if optics_group in optics['rlnOpticsGroup'].values:
                #        optics.loc[optics['rlnOpticsGroup']==optics_group,ctf_parms]=data2.attrs["optics"].loc[data.attrs["optics"]['rlnOpticsGroup']==optics_group,ctf_parms].values
                for optics_group in common_optics_groups:
                    optics.loc[optics["rlnOpticsGroup"] == optics_group, ctf_parms] = (
                        data2.attrs["optics"]
                        .loc[
                            data.attrs["optics"]["rlnOpticsGroup"] == optics_group,
                            ctf_parms,
                        ]
                        .values
                    )
                data.attrs["optics"] = optics

        # copy from the same micrograph (average for particles in the same micrograph)
        ctf_parms = [
            "rlnDefocusU",
            "rlnDefocusV",
            "rlnDefocusAngle",
            "rlnCtfBfactor",
            "rlnCtfScalefactor",
            "rlnPhaseShift",
        ]
        for v in ctf_parms:
            if v not in data:
                data.loc[:, v] = np.nan

        data2["mean_defocus"] = (data2["rlnDefocusU"] + data2["rlnDefocusV"]) / 2
        data2["delta_defocus"] = (data2["rlnDefocusU"] - data2["rlnDefocusV"]) / 2
        data2["astig_x"] = data2["delta_defocus"] * np.cos(
            np.deg2rad(data2["rlnDefocusAngle"])
        )
        data2["astig_y"] = data2["delta_defocus"] * np.sin(
            np.deg2rad(data2["rlnDefocusAngle"])
        )
        data2 = data2.groupby("rlnMicrographName").mean()
        data2["mean_astig"] = np.sqrt(data2["astig_x"] ** 2 + data2["astig_y"] ** 2)
        data2["mean_astig_angle"] = (
            np.arctan2(data2["astig_y"], data2["astig_x"]) * 180 / np.pi
        )
        for micrograph in data2.index:
            if micrograph in data["rlnMicrographName"].values:
                micrograph_rows = data["rlnMicrographName"] == micrograph
                data.loc[micrograph_rows, "rlnDefocusU"] = (
                    data2.loc[micrograph, "mean_defocus"]
                    + data2.loc[micrograph, "mean_astig"]
                )
                data.loc[micrograph_rows, "rlnDefocusV"] = (
                    data2.loc[micrograph, "mean_defocus"]
                    - data2.loc[micrograph, "mean_astig"]
                )
                data.loc[
                    micrograph_rows,
                    [
                        "rlnDefocusAngle",
                        "rlnCtfBfactor",
                        "rlnCtfScalefactor",
                        "rlnPhaseShift",
                    ],
                ] = data2.loc[
                    micrograph,
                    [
                        "mean_astig_angle",
                        "rlnCtfBfactor",
                        "rlnCtfScalefactor",
                        "rlnPhaseShift",
                    ],
                ].values
    return data, index_d
