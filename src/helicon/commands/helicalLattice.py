#!/usr/bin/env python

"""A Web app that illustrates the interconversion of 2D Lattice ⇔ Helical Lattice"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main(args):
    """Launch the helical lattice web app via Streamlit."""
    try:
        url = "https://raw.githubusercontent.com/jianglab/HelicalLattice/main/helical_lattice.py"
        cmd = f"streamlit run {url} --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"
        import subprocess

        subprocess.call(cmd, shell=True)
    except Exception:
        homephage = "https://jianglab.science.psu.edu/HelicalLattice"
        logger.error(
            "ERROR in running a local instance of helicalLattice. Please visit %s to use the Web app instances",
            homephage,
        )


def add_args(parser):
    """No additional CLI arguments for this web app launcher."""
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
