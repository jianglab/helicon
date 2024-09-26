#!/usr/bin/env python

"""A Web app that plots cartoon illustration of the residue properties of amyloid atomic models"""

import argparse


def main(args):
    try:
        url = "https://raw.githubusercontent.com/jianglab/procart/main/procart.py"
        cmd = f"streamlit run {url} --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"
        import subprocess

        subprocess.call(cmd, shell=True)
    except:
        homephage = "https://jiang.bio.purdue.edu/procart"
        print(
            f"ERROR in running a local instance of ProCart. Please visit {homephage} to use the Web app instances"
        )


def add_args(parser):
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
