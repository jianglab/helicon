#!/usr/bin/env python

"""A Web app that simulates 1D/2D TEM contrast transfer function (CTF)"""

import argparse


def main(args):
    try:
        url = "https://raw.githubusercontent.com/jianglab/ctfsimulation/refs/heads/master/ctf_simulation.py"
        cmd = f"streamlit run {url} --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"
        import subprocess

        subprocess.call(cmd, shell=True)
    except:
        homephage = "https://jianglab.science.psu.edu/ctfsimulation"
        print(
            f"ERROR in running a local instance of ctfSimulation. Please visit {homephage} to use the Web app instances"
        )


def add_args(parser):
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
