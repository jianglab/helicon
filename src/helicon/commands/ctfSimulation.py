#!/usr/bin/env python

"""A Web app that simulates 1D/2D TEM contrast transfer function (CTF)"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main(args):
    """Launch the CTF simulation web app via Streamlit."""
    try:
        url = "https://raw.githubusercontent.com/jianglab/ctfsimulation/refs/heads/master/ctf_simulation.py"
        cmd = f"streamlit run {url} --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"
        import subprocess

        subprocess.call(cmd, shell=True)
    except Exception:
        homephage = "https://jianglab.science.psu.edu/ctfsimulation"
        logger.error(
            "ERROR in running a local instance of ctfSimulation. Please visit %s to use the Web app instances",
            homephage,
        )


def add_args(parser):
    """No additional CLI arguments for this web app launcher."""
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
