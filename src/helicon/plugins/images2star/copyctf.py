"""Handler for the copyCtf option."""

from __future__ import annotations
import helicon
import numpy as np
import pandas as pd
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

        optics = data.attrs.get("optics")
        optics2 = data2.attrs.get("optics")
        if optics is not None and optics2 is not None:
            common_optics_groups = set(optics["rlnOpticsGroup"].values) & set(
                optics2["rlnOpticsGroup"].values
            )
        else:
            common_optics_groups = set()
        if common_optics_groups:
            # copy 'rlnBeamTiltX', 'rlnBeamTiltY', 'rlnOddZernike', 'rlnEvenZernike' from the same optics group
            ctf_parms_candidate = [
                "rlnBeamTiltX",
                "rlnBeamTiltY",
                "rlnOddZernike",
                "rlnEvenZernike",
            ]
            optics = optics.copy()
            in_common = optics["rlnOpticsGroup"].isin(common_optics_groups)
            ctf_parms = [key for key in ctf_parms_candidate if key in optics2]
            for key in ctf_parms:
                copied = optics["rlnOpticsGroup"].map(
                    dict(zip(optics2["rlnOpticsGroup"], optics2[key]))
                )
                old = optics[key] if key in optics else pd.Series(0, index=optics.index)
                optics[key] = copied.where(in_common, old)
            if ctf_parms:
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
                data[v] = np.nan

        data2 = average_ctf_per_micrograph(data2)
        other_parms = [
            v
            for v in ["rlnCtfBfactor", "rlnCtfScalefactor", "rlnPhaseShift"]
            if v in data2
        ]
        micrographs = data2.index[data2.index.isin(data["rlnMicrographName"].values)]
        for micrograph in micrographs:
            micrograph_rows = data["rlnMicrographName"] == micrograph
            data.loc[micrograph_rows, "rlnDefocusU"] = (
                data2.loc[micrograph, "mean_defocus"]
                + data2.loc[micrograph, "mean_astig"]
            )
            data.loc[micrograph_rows, "rlnDefocusV"] = (
                data2.loc[micrograph, "mean_defocus"]
                - data2.loc[micrograph, "mean_astig"]
            )
            data.loc[micrograph_rows, "rlnDefocusAngle"] = data2.loc[
                micrograph, "mean_astig_angle"
            ]
            for v in other_parms:
                data.loc[micrograph_rows, v] = data2.loc[micrograph, v]
    return data, index_d


def average_ctf_per_micrograph(data: pd.DataFrame) -> pd.DataFrame:
    """Average the CTF parameters of the particles in each micrograph.

    The astigmatism is averaged as a vector at twice the astigmatism angle,
    because the angle has a period of 180°.

    Parameters
    ----------
    data : pd.DataFrame
        Particles with ``rlnMicrographName``, ``rlnDefocusU``, ``rlnDefocusV``
        and ``rlnDefocusAngle`` (degrees). ``rlnCtfBfactor``,
        ``rlnCtfScalefactor`` and ``rlnPhaseShift`` are averaged if available.

    Returns
    -------
    pd.DataFrame
        One row per micrograph (index ``rlnMicrographName``) with columns
        ``mean_defocus``, ``mean_astig`` (half of DefocusU-DefocusV),
        ``mean_astig_angle`` (degrees, in (-90, 90]) and the other averaged
        parameters.
    """
    delta_defocus = (data["rlnDefocusU"] - data["rlnDefocusV"]) / 2
    angle2 = np.deg2rad(data["rlnDefocusAngle"]) * 2
    tmp = pd.DataFrame(
        {
            "rlnMicrographName": data["rlnMicrographName"].values,
            "mean_defocus": ((data["rlnDefocusU"] + data["rlnDefocusV"]) / 2).values,
            "astig_x": (delta_defocus * np.cos(angle2)).values,
            "astig_y": (delta_defocus * np.sin(angle2)).values,
        }
    )
    for v in ["rlnCtfBfactor", "rlnCtfScalefactor", "rlnPhaseShift"]:
        if v in data:
            tmp[v] = pd.to_numeric(data[v], errors="coerce").values
    ret = tmp.groupby("rlnMicrographName", sort=False).mean()
    ret["mean_astig"] = np.hypot(ret["astig_x"], ret["astig_y"])
    ret["mean_astig_angle"] = np.rad2deg(np.arctan2(ret["astig_y"], ret["astig_x"])) / 2
    return ret
