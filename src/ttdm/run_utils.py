import json
import os
import subprocess
import sys
from pathlib import Path


def get_processor(processor: str):
    if processor == "trigger":
        pass
    elif processor == "skimmer":
        pass


def parse_common_args(parser):
    parser.add_argument(
        "--processor",
        required=True,
        help="which processor to run",
        type=str,
        choices=["trigger", "skimmer"],
    )

    parser.add_argument("--maxchunks", default=0, help="max chunks", type=int)

    parser.add_argument("--chunksize", default=10_000, help="chunk size", type=int)
    parser.add_argument("--label", help="label", type=str)
    parser.add_argument("--yaml", help="YAML file", type=str)
