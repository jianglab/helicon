"""Handler for the normEulerDist option."""

from __future__ import annotations
import helicon
import logging

logger = logging.getLogger(__name__)


option_name = "normEulerDist"


def add_args(parser):
    parser.add_argument(
        "--normEulerDist",
        type=float,
        metavar=("<angle bin size>", "<nkeep>"),
        nargs=2,
        help="reduce Euler (view) preference by removing the worst particles in over-populated angular bins",
        default=[],
    )


def handle(data, args, index_d, param):
    """Handle the normEulerDist option.

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
    if len(param) == 2:
        bin, nkeep = param
        nkeep = int(nkeep)

        def assignEulerBins(rottilt):
            import math

            rot, tilt = rottilt
            tilt = int(tilt / bin + 0.5) * bin
            if tilt == 0 or tilt == 180:
                rot = 0
            else:
                angBinSizeTmp = bin / math.sin(tilt * math.pi / 180)
                rot = int(rot / angBinSizeTmp + 0.5) * angBinSizeTmp
            return (tilt, rot)

        binAngles = data[["rlnAngleRot", "rlnAngleTilt"]].apply(assignEulerBins, axis=1)
        binAssignments = binAngles.groupby(binAngles, sort=False)

        counts = binAssignments.size().sort_values(ascending=True)
        elbow = counts[helicon.findElbowPoint(counts)]
        if nkeep < 1:
            nkeep = elbow

        if args.verbose > 1:
            logger.info(
                "\tNumber of particles in %d Euler groups: Mean=%.1f\tSigma=%.1f\tMedian=%d\tElbow=%d"
                % (
                    len(binAssignments),
                    counts.mean(),
                    counts.std(),
                    counts.median(),
                    elbow,
                )
            )
            logger.info(
                "\tUpto %d best particles in each angular bin will be retained"
                % (nkeep)
            )

        binAssignments = dict(list(binAssignments))
        indices = []
        binEulers = sorted(binAssignments.keys())
        for bi, be in enumerate(binEulers):
            bm = binAssignments[be]
            binPtcls = data.iloc[bm.index, :]
            if "rlnLogLikeliContribution" in binPtcls:
                binPtcls = binPtcls.sort_values(
                    "rlnLogLikeliContribution", ascending=1
                )  # the larger rlnLogLikeliContribution the better
                binPtcls2 = binPtcls.tail(n=nkeep)
            else:
                if len(binPtcls) > nkeep:
                    binPtcls2 = binPtcls.sample(n=nkeep)
                else:
                    binPtcls2 = binPtcls
            if args.verbose > 2:
                logger.info(
                    "\t%3d/%3d:\talt=%.3g\taz=%.3g\t%d\t->\t%d"
                    % (
                        bi + 1,
                        len(binEulers),
                        be[0],
                        be[1],
                        len(binPtcls),
                        len(binPtcls2),
                    )
                )
            indices.extend(binPtcls2.index)
        indices.sort()
        data = data.iloc[indices, :]
        index_d[option_name] += 1
    return data, index_d
