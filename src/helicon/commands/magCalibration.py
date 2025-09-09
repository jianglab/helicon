#!/usr/bin/env python

"""A Web app that performs TEM mag calibration using graphene/graphene oxide/gold diffractogram"""


import argparse


def main(args):
    try:
        urls = [
            "https://raw.githubusercontent.com/jianglab/magCalApp/refs/heads/main/app.py",
            "https://raw.githubusercontent.com/jianglab/magCalApp/refs/heads/main/compute.py",
        ]
        folder = download_files(urls)

        cmd = f"shiny run --launch-browser --no-dev-mode --port 0 {folder}/app.py"
        import subprocess

        subprocess.call(cmd, shell=True)
    except:
        homephage = "https://jianglab.science.psu.edu/magCalibration"
        print(
            f"ERROR in running a local instance of magCalibration. Please visit {homephage} to use the Web app instances"
        )


def add_args(parser):
    return parser


def download_files(urls=[]):
    import tempfile
    import shutil
    import os

    temp_folder = tempfile.mkdtemp()

    for url in urls:
        import urllib.request
        import tarfile
        from contextlib import closing

        filename = url.split("/")[-1]
        local_filename = os.path.join(temp_folder, filename)

        with closing(urllib.request.urlopen(url)) as r:
            with open(local_filename, "wb") as f:
                shutil.copyfileobj(r, f)

    return temp_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
