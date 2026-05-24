#!/usr/bin/env python

"""A Web app that helps you compare 2D images with helical structure projections"""

import argparse
import logging

logger = logging.getLogger(__name__)


def main(args):
    """Launch the helical projection Shiny web app."""
    try:
        urls = [
            "https://raw.githubusercontent.com/jianglab/HelicalProjection/refs/heads/main/app.py",
            "https://raw.githubusercontent.com/jianglab/HelicalProjection/refs/heads/main/compute.py",
        ]
        folder = download_files(urls)

        cmd = f"shiny run --launch-browser --no-dev-mode --host 0.0.0.0 --port 0 {folder}/app.py"
        import subprocess

        subprocess.call(cmd, shell=True)
    except Exception:
        homephage = "https://jianglab.science.psu.edu/HelicalProjection"
        logger.error(
            "ERROR in running a local instance of HelicalProjection. Please visit %s to use the Web app instances",
            homephage,
        )


def add_args(parser):
    """No additional CLI arguments for this web app launcher."""
    return parser


def download_files(urls=[]):
    """Download files from URLs to a temporary directory.

    Parameters
    ----------
    urls : list of str
        URLs to download.

    Returns
    -------
    str
        Path to the temporary directory.
    """
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
