#!/usr/bin/env python

import argparse
import json
import math
import os
import pathlib
import requests
import sys

from ae.utils import get_tqdm


def download_file(
    url: str,
    filename: str,
    base: str = ".",
    dir: str = "data",
    overwrite: bool = False,
    silent: bool = False,
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
        print(f"{filepath} already exist. To overwrite pass `overwrite=True`")
        return

    tqdm = get_tqdm()
    chunkSize = 1024
    name, _ = os.path.splitext(filename)
    r = requests.get(url, stream=True)

    with open(filepath, "wb") as f:
        if not silent:
            pbar = tqdm(
                unit="B", unit_scale=True, total=int(r.headers["Content-Length"])
            )
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk:  # filter out keep-alive new chunks
                if not silent:
                    pbar.update(len(chunk))
                f.write(chunk)

    return filename


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
    name, _ = os.path.splitext(filename)
    url = "https://www.encodeproject.org/files/{}/@@download/{}".format(name, filename)

    return download_file(url, filename, base=base, dir=dir, overwrite=overwrite)


def download_roadmap_epigenomics_file(
    e_id: str,
    data_type: str,
    target: str,
    base: str = ".",
    dir: str = "data",
    overwrite: bool = False,
    silent: bool = False,
    check: bool = False,
):
    """Method for downloading Roadmap Epigenomics datasets

    Arguments:
        e_id {str} -- Experiment ID, e.g., e116
        data_type {str} -- Data type, e.g., fc_signal
        target {str} -- Histone modification, e.g., H3K27ac

    Keyword Arguments:
        base {str} -- Base directory (default: {"."})
        dir {str} -- Download directory (default: {"data"})
        overwrite {bool} -- If {True} existing files with be overwritten (default: {False})

    Returns:
        {str} -- Returns a pointer to `filename`.
    """
    base_url = ""
    filename = ""
    out_filename = ""

    if data_type == "fc_signal":
        base_url = "https://egg2.wustl.edu/roadmap/data/byFileType/signal/consolidated/macs2signal/foldChange/"
        filename = "{}-{}.fc.signal.bigwig".format(e_id, target)
        out_filename = "{}-{}.fc.signal.bigWig".format(e_id, target)

    elif data_type == "narrow_peaks":
        base_url = "https://egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/narrowPeak/"
        filename = "{}-{}.narrowPeak.gz".format(e_id, target)
        out_filename = "{}-{}.narrowPeak.gz".format(e_id, target)

    elif data_type == "broad_peaks":
        base_url = "https://egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/broadPeak/"
        filename = "{}-{}.broadPeak.gz".format(e_id, target)
        out_filename = "{}-{}.broadPeak.gz".format(e_id, target)

    else:
        print("Unknown data type: {}".format(data_type))

    url = base_url + filename

    if check:
        if not pathlib.Path(os.path.join(base, dir, out_filename)).is_file():
            print("{}/{}/{} not found".format(base, dir, out_filename))
    else:
        return download_file(
            url, out_filename, base=base, dir=dir, overwrite=overwrite, silent=silent
        )


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


def download_roadmap_epigenomics(
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

    data_types = list(settings["data_types"].keys())
    targets = settings["targets"]

    num_downloads = 0

    if dataset_idx is not None:
        datasets_iter = [datasets[dataset_idx]]
        datasets_iter = datasets_iter if silent else tqdm(datasets_iter, desc="Dataset")

    else:
        datasets_iter = datasets if silent else tqdm(datasets, desc="Dataset")

    for e_id in datasets_iter:
        if num_downloads >= limit:
            break

        targets_iter = targets if silent else tqdm(targets, desc="Targets", leave=False)

        for target in targets_iter:
            for data_type in data_types:
                download_roadmap_epigenomics_file(
                    e_id,
                    data_type,
                    target,
                    base=base,
                    overwrite=clear,
                    silent=silent,
                    check=check,
                )

        num_downloads += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Downloader")
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
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="limit the number of datasets to be downloaded",
        default=math.inf,
    )
    parser.add_argument(
        "-r", "--roadmap", action="store_true", help="if true download roadmap data"
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

    if args.roadmap:
        download_roadmap_epigenomics(
            datasets,
            settings,
            dataset_idx=args.dataset_idx,
            clear=args.clear,
            limit=args.limit,
            verbose=args.verbose,
            silent=args.silent,
        )
    else:
        download(
            datasets, settings, clear=args.clear, limit=args.limit, verbose=args.verbose
        )
