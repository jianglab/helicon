#!/usr/bin/env python

"""A Web app that maps 2D classes to helical tube/filament images"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main(args):
    """Launch the whereIsMyClass Shiny web app."""
    try:
        from pathlib import Path

        from helicon.lib.shiny import launch_shiny_app

        folder = Path(__file__).parent.parent / "webApps" / "whereIsMyClass"

        launch_shiny_app(f"{folder}/app.py")
    except Exception:
        homephage = "https://jianglab.science.psu.edu/helicon"
        logger.error("Please visit %s for more information", homephage)


def add_args(parser):
    """No additional CLI arguments for this web app launcher."""
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
