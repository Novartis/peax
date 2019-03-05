#!/usr/bin/env python

import argparse
import h5py
import json
import numpy as np
import os
import pathlib
import sys

from ae.utils import create_hdf5_dset


ds_types = [
    "data_train",
    "data_dev",
    "data_test",
    "peaks_train",
    "peaks_dev",
    "peaks_test",
]


def merge(
    datasets: dict,
    settings: dict,
    base: str = ".",
    name: str = "merged",
    dtype: str = None,
    clear: bool = False,
    verbose: bool = False,
    silent: bool = False,
    shuffling: bool = False,
):
    data = {}
    with h5py.File(os.path.join(base, "data", "{}.h5".format(name)), "w") as m:
        for dataset in datasets:
            filepath = os.path.join(base, "data", "{}.h5".format(dataset))

            if not pathlib.Path(filepath).is_file():
                sys.stderr.write("Dataset not found: {}\n".format(filepath))
                sys.exit(2)

            with h5py.File(filepath, "r") as f:
                for ds_type in ds_types:
                    if ds_type in data:
                        data[ds_type] = np.concatenate(
                            (data[ds_type], f[ds_type][:]), axis=0
                        )
                    else:
                        data[ds_type] = f[ds_type][:]

        for ds_type in ds_types:
            if shuffling:
                num_windows = data[ds_type].shape[0]
                new_shuffling = np.arange(num_windows)

                # Shuffle window ids and use the shuffled ids to shuffle the window data and window peaks
                np.random.seed(settings["rnd_seed"])
                np.random.shuffle(new_shuffling)

                data[ds_type] = data[ds_type][new_shuffling]

            create_hdf5_dset(m, ds_type, data[ds_type], dtype=dtype)
            create_hdf5_dset(
                m, "{}_shuffling".format(ds_type), new_shuffling, dtype=dtype
            )

        no_peak_ratio = (m["data_train"].shape[0] - np.sum(m["peaks_train"])) / np.sum(
            m["peaks_train"]
        )
        # There are `no_peak_ratio` times more no peak samples. To equalize the
        # importance of samples we give samples that contain a peak more weight
        # but we never downweight peak windows!
        sample_weight = (m["peaks_train"] * np.max((0, no_peak_ratio - 1))) + 1
        create_hdf5_dset(m, "sample_weight", sample_weight, dtype=dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Preparer")
    parser.add_argument(
        "-d", "--datasets", help="path to the datasets file", default="datasets.json"
    )
    parser.add_argument(
        "-s", "--settings", help="path to the settings file", default="settings.json"
    )
    parser.add_argument(
        "-c", "--clear", action="store_true", help="clears previously prepared data"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="turn on verbose logging"
    )
    parser.add_argument(
        "-z", "--silent", action="store_true", help="disable all but error logs"
    )

    args = parser.parse_args()

    try:
        with open(args.datasets, "r") as f:
            datasets = json.load(f)
    except FileNotFoundError:
        sys.stderr.write("You need to provide a datasets file via `--datasets`\n")
        sys.exit(2)

    try:
        with open(args.settings, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        sys.stderr.write("You need to provide a settings file via `--settings`\n")
        sys.exit(2)

    merge(
        datasets, settings, clear=args.clear, verbose=args.verbose, silent=args.silent
    )
