#!/usr/bin/env python

import argparse
import json
import os
import pathlib
import requests
import tqdm


def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
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
    "--settings", help="path to your settings file", default="settings.json"
)
parser.add_argument(
    "--clear", action="store_true", help="clears previously downloadeds"
)
parser.add_argument("--verbose", action="store_true", help="turn on verbose logging")

args = parser.parse_args()

try:
    with open(args.settings, "r") as f:
        data_files = json.load(f)["data"]
except FileNotFoundError:
    print("You need to provide a settings file via `--settings`")
    raise

# Create data directory
pathlib.Path("data").mkdir(parents=True, exist_ok=True)

for data_file in data_files:
    filename = os.path.basename(data_file)
    filepath = os.path.join("data", filename)

    if not pathlib.Path(filepath).is_file() or args.clear:
        print("Downloading {}".format(filename))
        download_file(data_file, filepath)
    else:
        print("Already downloaded {}".format(filename))
