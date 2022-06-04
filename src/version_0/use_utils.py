import argparse
from pathlib import Path
from typing import List, Dict, Union, Optional
import dataset_utils

def main(args, cls):
    util = cls(args)
    util()
    
if __name__ == "__main__":
    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument(
        "--util_name",
        help="Select util's name",
        type=str,
        required=True
    )
    initial_args, unrecognized = initial_parser.parse_known_args()
    cls = getattr(dataset_utils, initial_args.util_name)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--util_name",
        help="Select util's name",
        type=str,
        required=True
    )
    cls.add_args(parser)
    args = parser.parse_args()
    main(args, cls)
