#!/usr/bin/env python

"""A Web app that illustrates the interconversion of 2D Lattice ⇔ Helical Lattice"""

import argparse


def main(args):
    try:
        url = "https://raw.githubusercontent.com/jianglab/HelicalLattice/main/helical_lattice.py"
        cmd = f"streamlit run {url} --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"
        import subprocess

        subprocess.call(cmd, shell=True)
    except:
        homephage = "https://jiang.bio.purdue.edu/HelicalLattice"
        print(
            f"ERROR in running a local instance of helicalLattice. Please visit {homephage} to use the Web app instances"
        )


def add_args(parser):
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
