#!/usr/bin/env python

"""A Web app that maps 2D classes to helical tube/filament images"""

import argparse


def main(args):
    try:
        from pathlib import Path

        folder = Path(__file__).parent.parent / "webApps" / "whereIsMyClass"

        cmd = f"shiny run --launch-browser --no-dev-mode --port 0 {folder}/app.py"
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
