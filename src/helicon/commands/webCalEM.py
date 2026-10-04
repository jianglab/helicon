"""Open WebCalEM, the web app to calibrate the magnification (pixel size) of a TEM"""

import logging
import webbrowser

logger = logging.getLogger(__name__)

URL = "https://jianglab.github.io/WebCalEM/"


def add_args(parser):
    """Add CLI arguments for the webCalEM command.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to attach arguments to.
    """
    parser.add_argument(
        "--printUrl",
        action="store_true",
        help="only print the web address of WebCalEM, do not open a browser",
        default=False,
    )


def main(args):
    """Open WebCalEM in the default web browser.

    WebCalEM runs entirely in the browser, so there is nothing to start here:
    the page is opened, and its address printed in case no browser can be.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    print(URL)
    if args.printUrl:
        return
    if not webbrowser.open(URL):
        logger.warning("could not open a web browser; open %s yourself", URL)
