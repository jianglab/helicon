"""Handler for the setCTF option."""

from __future__ import annotations
import helicon
from helicon.lib.exceptions import HeliconError
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


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
        data["rlnVoltage"] = 0.0
        data["rlnSphericalAberration"] = 0.0
        data["rlnAmplitudeContrast"] = 0.0
        if "rlnDetectorPixelSize" not in data:
            data["rlnDetectorPixelSize"] = 5.0
        data["rlnMagnification"] = 0.0
        data["rlnDefocusU"] = 0.0
        data["rlnDefocusV"] = 0.0
        data["rlnDefocusAngle"] = 0.0

        ctfparms = read_ctfparm_file(setCTF)

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
                    f"cannot find ctf parameters for micrograph {mgraphName}"
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


def read_ctfparm_file(filename: str) -> dict[str, dict[str, float]]:
    """Read the CTF parameters of the micrographs in an EMAN1 ctfparm.txt file.

    Each line is ``<micrograph> <comma-separated CTF parameters>``, with 12
    values (``defocus,bfactor,amplitude,ampcont,noise1-4,voltage,cs,apix,...``)
    or, with astigmatism, 14 values
    (``defocus,dfdiff,dfang,bfactor,amplitude,ampcont,noise1-4,voltage,cs,apix,...``).

    Parameters
    ----------
    filename : str
        The ctfparm.txt file.

    Returns
    -------
    dict
        For both the micrograph name and its part before the first ".", a dict
        with ``defocus`` (µm, positive for underfocus), ``dfdiff`` (µm),
        ``dfang`` (degrees), ``ampcont`` (percent), ``voltage`` (kV), ``cs``
        (mm) and ``apix`` (Å/pixel).

    Raises
    ------
    HeliconError
        If a line is not in either format.
    """
    ret = {}
    for line in Path(filename).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise HeliconError(f"wrong format for line in {filename}:\n{line}")
        mid, rest = fields
        params = rest.split(",")
        if len(params) == 14:
            defocus, dfdiff, dfang, _, _, ampcont, _, _, _, _, voltage, cs, apix = map(
                float, params[:13]
            )
        elif len(params) == 12:
            defocus, _, _, ampcont, _, _, _, _, voltage, cs, apix = map(
                float, params[:11]
            )
            dfdiff = 0.0
            dfang = 0.0
        else:
            raise HeliconError(f"wrong format for line in {filename}:\n{line}")
        d = dict(
            defocus=abs(defocus),
            dfdiff=dfdiff,
            dfang=dfang,
            ampcont=ampcont * 100,
            voltage=voltage,
            cs=cs,
            apix=apix,
        )
        ret[mid] = d
        ret[mid.split(".")[0]] = d  # relax name matching
    return ret
