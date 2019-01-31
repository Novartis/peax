#!/usr/bin/env python

import argparse
import json
import math
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
    with open(os.path.join("data", filename), "wb") as f:
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

# Create data directory
pathlib.Path("data").mkdir(parents=True, exist_ok=True)

file_types = settings["file_types"]
data_types = list(settings["data_types"].keys())

print(args.limit)

num_downloads = 0
for dataset_name in datasets:
    samples = datasets[dataset_name]

    if num_downloads >= args.limit:
        break

    for sample_id in samples:
        dataset = samples[sample_id]
        has_all_data_types = set(data_types).issubset(dataset.keys())

        assert has_all_data_types, "Dataset should contain all data types"

        print("Downloading dataset: {}".format(dataset_name))

        for data_type in data_types:
            fileext = file_types[data_type]
            filename = "{}.{}".format(os.path.basename(dataset[data_type]), fileext)

            if not pathlib.Path(os.path.join("data", filename)).is_file() or args.clear:
                print("Downloading {}".format(filename))
                download_file(filename)
            else:
                print("Already downloaded {}".format(filename))

    num_downloads += 1
