#!/usr/bin/env python

"""A Web app for helical indexing using the cylindrical projection of a 3D map"""

import argparse


def main(args):
    try:
        import subprocess

        cmd = f"streamlit run https://raw.githubusercontent.com/jianglab/HI3D/main/hi3d.py --server.maxUploadSize 2048 --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"
        subprocess.call(cmd, shell=True)
    except:
        homephage = "https://jiang.bio.purdue.edu/hi3d"
        print(
            f"ERROR in running a local instance of HI3D. Please visit {homephage} to use the Web app instances"
        )


def add_args(parser):
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
