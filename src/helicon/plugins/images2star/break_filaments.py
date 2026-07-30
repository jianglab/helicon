"""Handler for the breakFilaments option."""

from __future__ import annotations
import logging
import helicon
from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)

option_name = "breakFilaments"


def add_args(parser):
    parser.add_argument(
        "--breakFilaments",
        type=str,
        metavar="maxSegments=<50>",
        action="append",
        help="break long filaments into shorter filaments with at most maxSegments segments per filament. Backs up original rlnHelicalTubeID to rlnHelicalTubeIDOriginal",
        default=[],
    )


def handle(data, args, index_d, param):
    if len(param):
        _, param_dict = helicon.parse_param_str(param)
        max_segments = int(param_dict.get("maxSegments", 50))
        if max_segments < 1:
            raise HeliconError("\tERROR: maxSegments must be >= 1")

        if "rlnHelicalTubeID" not in data:
            raise HeliconError(
                "\tERROR: rlnHelicalTubeID is required for --breakFilaments"
            )

        var = ""
        for v in "rlnMicrographName rlnImageName".split():
            if v in data:
                var = v
                break
        if not var:
            raise HeliconError(
                "\tERROR: rlnMicrographName or rlnImageName must be available"
            )

        if "@" in data[var].iloc[0]:
            tmp = data.loc[:, var].str.split("@", expand=True)
            group_var = "filename"
            data.loc[:, group_var] = tmp.iloc[:, 1]
        else:
            group_var = var

        # Backup original rlnHelicalTubeID if not already backed up
        if "rlnHelicalTubeIDOriginal" not in data:
            data["rlnHelicalTubeIDOriginal"] = data["rlnHelicalTubeID"]

        groups = data.groupby([group_var, "rlnHelicalTubeID"], sort=False)

        new_filaments = []
        for _, group in groups:
            n = len(group)
            if n <= max_segments:
                new_filaments.append(list(group.index))
            else:
                indices = list(group.index)
                if "rlnHelicalTrackLengthAngst" in data:
                    subset = data.loc[indices].sort_values("rlnHelicalTrackLengthAngst")
                    indices = subset.index.tolist()
                for i in range(0, n, max_segments):
                    new_filaments.append(indices[i : i + max_segments])

        for new_id, idx in enumerate(new_filaments):
            data.loc[idx, "rlnHelicalTubeID"] = new_id

        if group_var != var:
            data.drop(group_var, inplace=True, axis=1)

        if args.verbose > 1:
            logger.info(
                "\t%d filaments broken into %d filaments (max %d segments/filament)",
                groups.ngroups,
                len(new_filaments),
                max_segments,
            )

        index_d[option_name] += 1

    return data, index_d
