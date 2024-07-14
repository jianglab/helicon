#!/usr/bin/env python

'''A Web app that helps you determine helical pitch/twist using 2D Classification info'''

import argparse

def main(args):
    try:
        url = "https://raw.githubusercontent.com/jianglab/HelicalPitch/main/helicalPitch.py"
        cmd = f"streamlit run {url} --server.maxUploadSize 2048 --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false"
        import subprocess
        subprocess.call(cmd, shell=True)
    except:
        homephage = "https://jiang.bio.purdue.edu/HelicalPitch"
        print(f"ERROR in running a local instance of HILL. Please visit {homephage} to use the Web app instances")

def add_args(parser):
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())


