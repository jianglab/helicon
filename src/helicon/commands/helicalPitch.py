#!/usr/bin/env python

"""A Web app that helps you determine helical pitch/twist using 2D Classification info"""

import argparse


def main(args):
    try:
        import pathlib

        app_file = pathlib.Path(__file__).parent / "../web_apps/helicalPitch/app.py"
        cmd = f"shiny run --launch-browser --no-dev-mode --port 0 {app_file}"
        import subprocess

        subprocess.call(cmd, shell=True)
    except:
        homephage = "https://jiang.bio.purdue.edu/HelicalPitch"
        print(
            f"ERROR in running a local instance of HelicalPitch. Please visit {homephage} to use the Web app instances"
        )


def add_args(parser):
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
