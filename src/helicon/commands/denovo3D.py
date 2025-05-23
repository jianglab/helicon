#!/usr/bin/env python

"""A Web app that performs de novo helical indexing and 3D reconstruction from a single 2D image"""

import argparse


def main(args):
    try:
        from pathlib import Path

        app_file = Path(__file__).parent.parent / "webApps" / "denovo3D" / "app.py"

        cmd = f'shiny run --launch-browser --no-dev-mode --port 0 "{app_file}"'
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
