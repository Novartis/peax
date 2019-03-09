#!/usr/bin/env python

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys

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

    # Check if chrom sizes are available
    if not pathlib.Path(
        os.path.join(base, "data", "{}.chrom.sizes".format(settings["coord_system"]))
    ).is_file():
        subprocess.call(
            "fetchChromSizes {} > data/{}.chrom.sizes".format(
                settings["coord_system"], settings["coord_system"]
            ),
            shell=True,
        )

    for e_id in datasets_iter:
        if num_conversions >= limit:
            break

        targets_iter = targets if silent else tqdm(targets, desc="Targets", leave=False)

        for target in targets_iter:
            for file_type in file_types:
                if file_type[-5:] == "peaks":
                    peak_type = (
                        "narrowPeak" if file_type[:-6] == "narrow" else "broadPeak"
                    )
                    input_file = os.path.join(
                        base, "data", "{}-{}.{}.gz".format(e_id, target, peak_type)
                    )
                    tmp_file = os.path.join(
                        base,
                        "data",
                        "{}-{}.{}.sorted.bed".format(e_id, target, peak_type),
                    )
                    output_file = os.path.join(
                        base,
                        "data",
                        "{}-{}.peaks.{}.bigBed".format(e_id, target, file_type[:-6]),
                    )
                    chromsizes = os.path.join(
                        base, "data", "{}.chrom.sizes".format(settings["coord_system"])
                    )

                    # Usually narrowPeak are in bed6+4 format but the scores in
                    # the narrowPeak (and broadPeak) files from Roadmap Epigenomics are
                    # above 1000 which is not allowed and breaks bedToBigBed.
                    # Fortunately, we don't care about the scores so we just ignore them
                    # by using bed3+x formats
                    dtype = "bed3+7" if file_type[:-6] == "narrow" else "bed3+6"

                    if pathlib.Path(output_file).is_file() and not clear:
                        print("Already converted: {}".format(output_file))
                        continue

                    if pathlib.Path(input_file).is_file():
                        if verbose and not silent:
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
                        subprocess.call("rm {}".format(tmp_file), shell=True)
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
        sys.stderr.write("Please provide a datasets file via `--datasets`\n")
        sys.exit(2)

    try:
        with open(args.settings, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        sys.stderr.write("Please provide a settings file via `--settings`\n")
        sys.exit(2)

    bed_to_bigbed(
        datasets,
        settings,
        dataset_idx=args.dataset_idx,
        clear=args.clear,
        verbose=args.verbose,
        silent=args.silent,
    )
