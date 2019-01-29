#!/usr/bin/env python

import argparse
import json
import os
import pathlib
import requests
import tqdm
import sys


def download_file(filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    name, _ = os.path.splitext(filename)
    url = "https://www.encodeproject.org/files/{}/@@download/{}".format(name, filename)
    r = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        pbar = tqdm.tqdm(
            unit="B", unit_scale=True, total=int(r.headers["Content-Length"])
        )
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)
    return filename


parser = argparse.ArgumentParser(description="Peax Downloader")
parser.add_argument(
    "-s", "--settings", help="path to the settings file", default="settings.json"
)
parser.add_argument(
    "-c", "--clear", action="store_true", help="clears previously downloadeds"
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="turn on verbose logging"
)

args = parser.parse_args()

try:
    with open(args.settings, "r") as f:
        datasets = json.load(f)["datasets"]
except FileNotFoundError:
    print("Please provide a settings file via `--settings`")
    sys.exit(2)

# Create data directory
pathlib.Path("data").mkdir(parents=True, exist_ok=True)

supported_data_types = ["signal", "narrow_peaks", "broad_peaks"]

for dataset_name in datasets:
    dataset = datasets[dataset_name]
    has_all_data_types = set(supported_data_types).issubset(dataset.keys())

    assert has_all_data_types, "Dataset should contain all data types"

    print("Downloading dataset: {}".format(dataset_name))

    for data_type in supported_data_types:
        filename = os.path.basename(dataset[data_type])
        filepath = os.path.join("data", filename)

        if not pathlib.Path(filepath).is_file() or args.clear:
            print("Downloading {}".format(filename))
            download_file(filepath)
        else:
            print("Already downloaded {}".format(filename))
