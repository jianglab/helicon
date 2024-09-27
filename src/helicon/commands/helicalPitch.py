#!/usr/bin/env python

"""A Web app that helps you determine helical pitch/twist using 2D Classification info"""

import argparse


def main(args):
    try:
        urls = [
            "https://raw.githubusercontent.com/jianglab/HelicalPitch/refs/heads/master/app.py",
            "https://raw.githubusercontent.com/jianglab/HelicalPitch/refs/heads/master/compute.py",
        ]
        folder = download_files(urls)

        cmd = f"shiny run --launch-browser --no-dev-mode --port 0 {folder}/app.py"
        import subprocess

        subprocess.call(cmd, shell=True)
    except:
        homephage = "https://jiang.bio.purdue.edu/HelicalPitch"
        print(
            f"ERROR in running a local instance of HelicalPitch. Please visit {homephage} to use the Web app instances"
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
