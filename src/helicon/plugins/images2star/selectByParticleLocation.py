"""Handler for the selectByParticleLocation option."""

from __future__ import annotations
import logging
from pathlib import Path
import helicon
import numpy as np
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)

option_name = "selectByParticleLocation"


def add_args(parser):
    parser.add_argument(
        "--selectByParticleLocation",
        type=str,
        metavar="starFile:maxDist=<pixel>",
        action="append",
        help="select particles that are at the same locations in the micrograph (example: x.star:maxDist=10). disabled by default",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the selectByParticleLocation option.

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
    if len(param) > 0:
        required_attrs = ["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY"]
        missing_attrs = [p for p in required_attrs if p not in data]
        if missing_attrs:
            raise HeliconError(
                "\tERROR: required parameters %s are not available"
                % " ".join(missing_attrs)
            )

        sf, param_dict = helicon.parse_param_str(param)
        maxDist = param_dict.get("maxDist", 5)
        if sf is None or not Path(sf).exists():
            raise HeliconError("\tERROR: %s does not exist" % sf)
        if not (maxDist >= 0):
            raise HeliconError("\tERROR: maxDist must be >= 0")

        data_sf = helicon.images2dataframe(
            sf,
            alternative_folders=args.folder,
            ignore_bad_particle_path=args.ignoreBadParticlePath,
            ignore_bad_micrograph_path=args.ignoreBadMicrographPath,
            warn_missing_ctf=0,
            target_convention="relion",
        )
        if args.verbose > 1:
            logger.info("\t%d images found in %s", len(data_sf), sf)

        missing_attrs = [p for p in required_attrs if p not in data_sf]
        if missing_attrs:
            raise HeliconError(
                "\tERROR: required parameters %s are not available in %s"
                % (" ".join(missing_attrs), sf)
            )

        from helicon import convert_dataframe_file_path

        data["sbpl_rlnMicrographName"] = convert_dataframe_file_path(
            data, "rlnMicrographName", to="abs"
        )
        data_sf["sbpl_rlnMicrographName"] = convert_dataframe_file_path(
            data_sf, "rlnMicrographName", to="abs"
        )

        from scipy.spatial import distance

        group2 = {gname: g for gname, g in data_sf.groupby("sbpl_rlnMicrographName")}
        matched_indices = []
        for gname, g in data.groupby("sbpl_rlnMicrographName"):
            if gname not in group2:
                continue
            cx = g["rlnCoordinateX"].values
            cy = g["rlnCoordinateY"].values
            cx2 = group2[gname]["rlnCoordinateX"].values
            cy2 = group2[gname]["rlnCoordinateY"].values

            loc = np.vstack((cx, cy)).T
            loc2 = np.vstack((cx2, cy2)).T
            dist_matrix = distance.cdist(loc, loc2, "euclidean")
            row_indices = np.where(np.min(dist_matrix, axis=1) <= maxDist)[0]
            matched_indices += list(g.index[row_indices])

        data2 = data.loc[matched_indices, :]
        data2.reset_index(drop=True, inplace=True)
        if args.verbose > 1:
            logger.info("\t%d/%d images retained", len(data2), len(data))
        if len(data2) <= 0:
            raise HeliconError("\tWARNING: no particle left. I will quit")
        data = data2
        data.drop(["sbpl_rlnMicrographName"], axis=1, inplace=True)
        index_d[option_name] += 1
    return data, index_d
