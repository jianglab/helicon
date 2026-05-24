#!/usr/bin/env python

"""A Web app for helical indexing using the cylindrical projection of a 3D map"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main(args):
    """Launch the HI3D helical indexing web app via Streamlit."""
    try:
        import subprocess

        cmd = f"streamlit run https://raw.githubusercontent.com/jianglab/HI3D/main/hi3d.py --server.maxUploadSize 2048 --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"
        subprocess.call(cmd, shell=True)
    except Exception:
        homephage = "https://jianglab.science.psu.edu/hi3d"
        logger.error(
            "ERROR in running a local instance of HI3D. Please visit %s to use the Web app instances",
            homephage,
        )


def add_args(parser):
    """No additional CLI arguments for this web app launcher."""
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
