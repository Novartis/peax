#!/usr/bin/env python

import argparse
import h5py
import json
import os
import pathlib
import re
import sys


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
):
    mode = "w" if clear else "w-"
    shapes = {}
    try:
        with h5py.File(os.path.join(base, "data", "{}.h5".format(name)), mode) as m:
            # 1. Get the global shape
            for dataset in datasets:
                filepath = os.path.join(base, "data", "{}.h5".format(dataset))

                if not pathlib.Path(filepath).is_file():
                    sys.stderr.write("Dataset not found: {}\n".format(filepath))
                    sys.exit(2)

                with h5py.File(filepath, "r") as f:
                    for ds_type in ds_types:
                        if ds_type in shapes:
                            shapes[ds_type][0] += f[ds_type].shape[0]
                        else:
                            shapes[ds_type] = list(f[ds_type].shape)

            # 2. Create the dataset
            for ds_type in ds_types:
                m.create_dataset(ds_type, tuple(shapes[ds_type]), dtype=dtype)

            # 3. Fill up the dataset
            pos = {}
            for dataset in datasets:
                filepath = os.path.join(base, "data", "{}.h5".format(dataset))

                with h5py.File(filepath, "r") as f:
                    for ds_type in ds_types:
                        if ds_type not in pos:
                            pos[ds_type] = 0

                        num = f[ds_type].shape[0]

                        m[ds_type][pos[ds_type] : pos[ds_type] + num] = f[ds_type][:]
                        pos[ds_type] += num
    except OSError as error:
        # When `clear` is `False` and the data is already prepared then we expect to
        # see error number 17 as we opened the file in `w-` mode.
        if not clear:
            # Stupid h5py doesn't populate `error.errno` so we have to parse it out
            # manually
            matches = re.search(r"errno = (\d+)", str(error))
            if matches and int(matches.group(1)) == 17:
                sys.stderr.write("Merged dataset already exist.\n")
                sys.exit(2)
            else:
                raise
        else:
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Preparer")
    parser.add_argument(
        "-d", "--datasets", help="path to the datasets file", default="datasets.json"
    )
    parser.add_argument(
        "-s", "--settings", help="path to the settings file", default="settings.json"
    )
    parser.add_argument("-n", "--name", help="name for the merged dataset file")
    parser.add_argument("-t", "--dtype", help="data type")
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
        datasets,
        settings,
        name=args.name,
        dtype=args.dtype,
        clear=args.clear,
        verbose=args.verbose,
        silent=args.silent,
    )
