#!/usr/bin/env python

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys

from download import download_file
from ae.utils import get_tqdm


def bed_to_bigbed(
    datasets: list,
    settings: dict,
    dataset_idx: int = None,
    base: str = ".",
    clear: bool = False,
    limit: int = math.inf,
    verbose: bool = False,
    silent: bool = False,
    check: bool = False,
):
    tqdm = get_tqdm()

    # Create data directory
    pathlib.Path("data").mkdir(parents=True, exist_ok=True)

    file_types = list(settings["file_types"].keys())
    targets = settings["targets"]

    num_conversions = 0

    if dataset_idx is not None:
        datasets_iter = [datasets[dataset_idx]]
        datasets_iter = datasets_iter if silent else tqdm(datasets_iter, desc="Dataset")

    else:
        datasets_iter = datasets if silent else tqdm(datasets, desc="Dataset")

    for e_id in datasets_iter:
        if num_conversions >= limit:
            break

        targets_iter = targets if silent else tqdm(targets, desc="Targets", leave=False)

        for target in targets_iter:
            for file_type in file_types:
                if file_type[-5:] == "peaks":
                    wurst_type = (
                        "narrowPeak" if file_type[:-6] == "narrow" else "broadPeak"
                    )
                    input_file = os.path.join(
                        base, "data", "{}-{}.{}.gz".format(e_id, target, wurst_type)
                    )
                    tmp_file = os.path.join(
                        base,
                        "data",
                        "{}-{}.{}.sorted.bed".format(e_id, target, wurst_type),
                    )
                    output_file = os.path.join(
                        base,
                        "data",
                        "{}-{}.peaks.{}.bigBed".format(e_id, target, file_type[:-6]),
                    )
                    chromsizes = os.path.join(base, "data", "hg19-chromsizes.tsv")

                    dtype = "bed6+4" if file_type[:-6] == "narrow" else "bed6+3"

                    if pathlib.Path(input_file).is_file():
                        print("Convert {} to {}".format(input_file, output_file))
                        subprocess.call(
                            "gunzip -c {} | sort -k1,1 -k2,2n - > {}".format(
                                input_file, tmp_file
                            ),
                            shell=True,
                        )
                        subprocess.call(
                            "bedToBigBed -type={} {} {} {}".format(
                                dtype, tmp_file, chromsizes, output_file
                            ),
                            shell=True,
                        )
                        # subprocess.call("rm {}".format(tmp_file), shell=True)
                    else:
                        print("Bed file not found: {}".format(input_file))

        num_conversions += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax .bed to .bigBed Converter")
    parser.add_argument(
        "-d", "--datasets", help="path to the datasets file", default="datasets.json"
    )
    parser.add_argument(
        "-x", "--dataset-idx", help="index of the dataset to be downloaded", type=int
    )
    parser.add_argument(
        "-s", "--settings", help="path to the settings file", default="settings.json"
    )
    parser.add_argument(
        "-c", "--clear", action="store_true", help="clears previously downloadeds"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="turn on verbose logging"
    )
    parser.add_argument(
        "-z", "--silent", action="store_true", help="if true hide all logs"
    )

    args = parser.parse_args()

    try:
        with open(args.datasets, "r") as f:
            datasets = json.load(f)
    except FileNotFoundError:
        print("Please provide a datasets file via `--datasets`")
        sys.exit(2)

    try:
        with open(args.settings, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Please provide a settings file via `--settings`")
        sys.exit(2)

    bed_to_bigbed(
        datasets,
        settings,
        dataset_idx=args.dataset_idx,
        base=args.base,
        clear=args.clear,
        limit=args.limit,
        verbose=args.verbose,
        silent=args.silent,
    )
