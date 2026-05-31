"""Handler for the setCTF option."""

from __future__ import annotations
import helicon
from pathlib import Path


option_name = "setCTF"


def add_args(parser):
    parser.add_argument(
        "--setCTF",
        metavar="<filename>",
        type=str,
        help="set ctf parameters stored in this file (EMAN1 ctfparm.txt) to the output star file",
        default="",
    )


def handle(data, args, index_d, param):
    """Handle the setCTF option.

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
        setCTF = param
        data["rlnVoltage"] = 0
        data["rlnSphericalAberration"] = 0
        data["rlnAmplitudeContrast"] = 0
        if "rlnDetectorPixelSize" not in data:
            data["rlnDetectorPixelSize"] = 5
        data["rlnMagnification"] = 0
        data["rlnDefocusU"] = 0
        data["rlnDefocusV"] = 0
        data["rlnDefocusAngle"] = 0

        ctfparms = readCtfparmFile(setCTF)

        micrographNames = data["rlnImageName"].str.split("@", expand=True).iloc[:, -1]
        mgraphs = micrographNames.groupby(micrographNames, sort=False)

        def setMicrographCTF(mgraphName, mgraphParticles, data, ctfparms):

            mid = Path(mgraphName).stem
            mid2 = mid.split(".")[0]

            d = None
            if mid in ctfparms:
                d = ctfparms[mid]
            elif mid2 in ctfparms:
                d = ctfparms[mid2]
            else:
                raise HeliconError(
                    "\\tERROR: cannot find ctf parmeters for micrograph %s"
                )

            data.loc[mgraphParticles.index, "rlnVoltage"] = d["voltage"]
            data.loc[mgraphParticles.index, "rlnSphericalAberration"] = d["cs"]
            data.loc[mgraphParticles.index, "rlnAmplitudeContrast"] = (
                d["ampcont"] / 100.0
            )  # [0, 100] -> [0, 1]
            data.loc[mgraphParticles.index, "rlnMagnification"] = (
                data.loc[mgraphParticles.index, "rlnDetectorPixelSize"]
                * 1e4
                / d["apix"]
            )

            rlnDefocusU, rlnDefocusV, rlnDefocusAngle = (
                helicon.eman_astigmatism_to_relion(
                    d["defocus"], d["dfdiff"], d["dfang"]
                )
            )
            data.loc[mgraphParticles.index, "rlnDefocusU"] = rlnDefocusU
            data.loc[mgraphParticles.index, "rlnDefocusV"] = rlnDefocusV
            data.loc[mgraphParticles.index, "rlnDefocusAngle"] = rlnDefocusAngle

        for mgraphName, mgraphParticles in mgraphs:
            setMicrographCTF(mgraphName, mgraphParticles, data, ctfparms)
        index_d[option_name] += 1
    return data, index_d


import logging

logger = logging.getLogger(__name__)
