#!/usr/bin/env python

"""A Web app that performs de novo helical indexing and 3D reconstruction from a single 2D image"""

import argparse


def main(args):
    try:
        from pathlib import Path

        folder = Path(__file__).parent.parent / "webApps" / "image_stitching"

        cmd = f"shiny run --reload --port 0 {folder}/app.py"
        import subprocess

        subprocess.call(cmd, shell=True)
    except:
        homephage = "https://jianglab.science.psu.edu/helicon"
        print(f"Please visit {homephage} for more information")


def add_args(parser):
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
