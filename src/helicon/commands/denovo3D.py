#!/usr/bin/env python

"""A Web app that performs de novo helical indexing and 3D reconstruction from a single 2D image"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main(args):
    """Launch the de novo 3D reconstruction Shiny web app."""
    try:
        from pathlib import Path

        from helicon.lib.shiny import launch_shiny_app

        app_file = Path(__file__).parent.parent / "webApps" / "denovo3D" / "app.py"
        launch_shiny_app(app_file)
    except Exception:
        homephage = "https://jianglab.science.psu.edu/helicon"
        logger.error("Please visit %s for more information", homephage)


def add_args(parser):
    """No additional CLI arguments for this web app launcher."""
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
