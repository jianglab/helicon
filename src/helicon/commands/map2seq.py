#!/usr/bin/env python

"""A Web app that identifies the best protein sequence explaining a 3D density map"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main(args):
    """Open the map2seq web app in a browser."""
    try:
        url = "https://map2seq.streamlit.app/"
        import webbrowser

        webbrowser.open(url)
    except Exception:
        homephage = "https://jianglab.science.psu.edu/map2seq"
        logger.error(
            "ERROR in accessing the map2seq web app. Please visit %s for more information",
            homephage,
        )


def add_args(parser):
    """No additional CLI arguments for this web app launcher."""
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
