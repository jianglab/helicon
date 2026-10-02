#!/usr/bin/env python

"""A Helicon Web app with eight analytical tools as tabs: whereIsMyClass, helicalProjection, HILL, helicalPitch, denovo3D, abinitio3D, helicalLattice, and HI3D"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main(args):
    """Launch the Helicon Lab consolidated web app."""
    # imported here: loading shiny at import would slow every helicon command
    from helicon.lib.shiny import launch_shiny_app

    try:
        launch_shiny_app("helicon.webApps.app:app", block=True, reload=True)
    except Exception as e:
        homepage = "https://jianglab.science.psu.edu/helicon"
        logger.error("Please visit %s for more information", homepage)
        raise e


def add_args(parser):
    """No additional CLI arguments for this web app launcher."""
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
