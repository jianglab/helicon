#!/usr/bin/env python

"""A Web app that simulates 1D/2D TEM contrast transfer function (CTF)"""

import argparse
import logging
import sys

from helicon.lib.exceptions import HeliconError

logger = logging.getLogger(__name__)


def main(args):
    """Launch the CTF simulation web app via Streamlit."""
    import subprocess

    url = "https://raw.githubusercontent.com/jianglab/ctfsimulation/refs/heads/master/ctf_simulation.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        url,
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    homepage = "https://jianglab.science.psu.edu/ctfsimulation"
    try:
        returncode = subprocess.call(cmd)
    except OSError as e:
        returncode = None
        reason = str(e)
    else:
        reason = f"streamlit exited with code {returncode}"
    if returncode != 0:
        raise HeliconError(
            f"failed to run a local instance of ctfSimulation ({reason}). Make sure that streamlit is installed (pip install streamlit), or visit {homepage} to use the Web app instances"
        )


def add_args(parser):
    """No additional CLI arguments for this web app launcher."""
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
