"""Handler for the setBeamTiltClass option."""

from __future__ import annotations
import helicon
from helicon.lib.exceptions import HeliconError


option_name = "setBeamTiltClass"


def add_args(parser):
    parser.add_argument(
        "--setBeamTiltClass",
        metavar="<0|1>",
        type=int,
        help="set rlnBeamTiltClass column, one group per micrograph",
        default=0,
    )


def handle(data, args, index_d, param):
    micrographNames = data["rlnImageName"].str.split("@", expand=True).iloc[:, -1]
    mgraphs = micrographNames.groupby(micrographNames, sort=False)

    for mi, mgraph in enumerate(mgraphs):
        mgraphName, mgraphParticles = mgraph
        data.loc[mgraphParticles.index, "rlnBeamTiltClass"] = mi + 1
    index_d[option_name] += 1

    return data, index_d


import logging

logger = logging.getLogger(__name__)
