#!/usr/bin/python

"""
Runs coffea processors on the UW AF via condor or dask

Author(s): Ameya Thete
"""

import argparse
import os
import pickle
from pathlib import Path
import cowtools

from coffea import nanoevents, processor

from ttdm import run_utils


def main(args):
    processor = run_utils.get_processor(args.processor)

    if len(args.files):
        fileset = {f"{args.files_label}": args.files}
    else:
        fileset = None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    run_utils.parse_common_args(parser)
    parser.add_argument("--start", default=0, help="start index of files", type=int)
    parser.add_argument("--end", default=1, type=int, help="end index of files", type=int)
    parser.add_argument("--files", default=[], help="set of files to run on", nargs="*")
    parser.add_argument(
        "--files-label",
        type=str,
        default="files",
        help="label for files being run on, if --files option used",
    )
    args = parser.parse_args()

    main(args)
