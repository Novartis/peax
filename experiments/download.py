#!/usr/bin/env python

import argparse
import json
import math
import os
import pathlib
import requests
import sys

from ae.utils import get_tqdm


def download_encode_file(
    filename: str, base: str = ".", dir: str = "data", overwrite: bool = False
):
    """Method for downloading ENCODE datasets

    Arguments:
        filename {str} -- File access of the ENCODE data file

    Keyword Arguments:
        base {str} -- Base directory (default: {"."})
        dir {str} -- Download directory (default: {"data"})
        overwrite {bool} -- If {True} existing files with be overwritten (default: {False})

    Returns:
        {str} -- Returns a pointer to `filename`.
    """
    filepath = os.path.join(base, dir, filename)

    if pathlib.Path(filepath).is_file() and not overwrite:
        print("File already exist. To overwrite pass `overwrite=True`")
        return

    tqdm = get_tqdm()
    chunkSize = 1024
    name, _ = os.path.splitext(filename)
    url = "https://www.encodeproject.org/files/{}/@@download/{}".format(name, filename)
    r = requests.get(url, stream=True)

    with open(filepath, "wb") as f:
        pbar = tqdm(unit="B", unit_scale=True, total=int(r.headers["Content-Length"]))
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)

    return filename


def download(
    datasets: dict,
    settings: dict,
    base: str = ".",
    clear: bool = False,
    limit: int = math.inf,
    verbose: bool = False,
):
    tqdm = get_tqdm()

    # Create data directory
    pathlib.Path("data").mkdir(parents=True, exist_ok=True)

    file_types = settings["file_types"]
    data_types = list(settings["data_types"].keys())

    num_downloads = 0
    for dataset_name in tqdm(datasets, desc="Dataset"):
        samples = datasets[dataset_name]

        if num_downloads >= limit:
            break

        for sample_id in tqdm(samples, desc="Sample", leave=False):
            dataset = samples[sample_id]
            has_all_data_types = set(data_types).issubset(dataset.keys())

            assert has_all_data_types, "Dataset should contain all data types"

            for data_type in tqdm(data_types, desc="Data type", leave=False):
                fileext = file_types[data_type]
                filename = "{}.{}".format(os.path.basename(dataset[data_type]), fileext)

                download_encode_file(filename, base, overwrite=clear)

        num_downloads += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Downloader")
    parser.add_argument(
        "-d", "--datasets", help="path to the datasets file", default="datasets.json"
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
        "-l",
        "--limit",
        type=int,
        help="limit the number of datasets to be downloaded",
        default=math.inf,
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

    download(
        datasets, settings, clear=args.clear, limit=args.limit, verbose=args.verbose
    )
