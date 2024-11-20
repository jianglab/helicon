#!/usr/bin/env python

"""A Web app that identifies the best protein sequence explaining a 3D density map"""

import argparse


def main(args):
    try:
        url = "https://map2seq.streamlit.app/"
        import webbrowser

        webbrowser.open(url)
    except:
        homephage = "https://jiang.bio.purdue.edu/map2seq"
        print(
            f"ERROR in accessing the map2seq web app. Please visit {homephage} for more information"
        )


def add_args(parser):
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
