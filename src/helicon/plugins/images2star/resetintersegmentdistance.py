"""Handler for the resetInterSegmentDistance option."""

from __future__ import annotations
import logging
import helicon
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)

option_name = "resetInterSegmentDistance"


def add_args(parser):
    parser.add_argument(
        "--resetInterSegmentDistance",
        metavar="<\u00c5>",
        type=float,
        help="reset inter-segment distance by adding/removing 'particles' with updated 'rlnCoordinateX rlnCoordinateY rlnHelicalTrackLengthAngst' parameters. Warning: the output star file is meaningful only for particle extraction and it should NOT be used for 2d/3d classification or 3d refinement",
        default=0,
    )


def handle(data, args, index_d, param):
    """Handle the resetInterSegmentDistance option.

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
        badParms = [
            v
            for v in "rlnImageName rlnHelicalTubeID rlnCoordinateX rlnCoordinateY".split()
            if v not in data
        ]
        if badParms:
            s = "s" if len(badParms) > 1 else ""
            raise HeliconError(
                "\tERROR: parameter%s %s do not exist" % (s, " ".join(badParms))
            )

        apix_micrograph = 0
        try:
            optics = data.attrs["optics"]
            for attr in [
                "rlnMicrographPixelSize",
                "rlnMicrographOriginalPixelSize",
            ]:
                if attr in optics:
                    apix_micrograph = optics[attr].iloc[0]
                    break
        finally:
            if apix_micrograph <= 0:
                raise HeliconError(
                    "\tERROR: neither rlnMicrographPixelSize nor rlnMicrographOriginalPixelSize is available"
                )

        data = helicon.reset_inter_segment_distance(
            data,
            new_inter_segment_distance=param,
            apix_micrograph=apix_micrograph,
            verbose=args.verbose,
        )
    return data, index_d
