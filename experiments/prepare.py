#!/usr/bin/env python

import argparse
import h5py
import json
import numpy as np
import os
import pathlib

from ae import bigwig
from sklearn.preprocessing import MinMaxScaler


parser = argparse.ArgumentParser(description="Peax Preparer")
parser.add_argument(
    "--settings", help="path to your settings file", default="settings.json"
)
parser.add_argument(
    "--clear", action="store_true", help="clears previously prepared data"
)
parser.add_argument("--verbose", action="store_true", help="turn on verbose logging")

args = parser.parse_args()

try:
    with open(args.settings, "r") as f:
        settings = json.load(f)
except FileNotFoundError:
    print("You need to provide a settings file via `--settings`")
    raise

# Create data directory
pathlib.Path("data").mkdir(parents=True, exist_ok=True)

step_size = settings["window_size"] // settings["step_frequency"]
bins_per_window = settings["window_size"] // settings["resolution"]
all_chroms = settings["training"] + settings["testing"]

if args.verbose:
    print("Window size: {} base pairs".format(settings["window_size"]))
    print("Resolution: {} base pairs".format(settings["resolution"]))
    print("Bins per window: {}".format(bins_per_window))
    print("Step frequency per window: {}".format(settings["step_frequency"]))
    print("Step size: {} base pairs".format(step_size))
    print("Training: {}".format(", ".join(settings["training"])))
    print("Testing: {}".format(", ".join(settings["testing"])))

prep_data_filename = "prepared-data.h5"
prep_data_filepath = os.path.join("data", prep_data_filename)


def print_progress():
    print(".", end="", flush=True)


with h5py.File(prep_data_filepath, "a") as f:
    for data_file in settings["data"]:
        raw_data_filename = os.path.basename(data_file)
        raw_data_filepath = os.path.join("data", raw_data_filename)

        if pathlib.Path(raw_data_filepath).is_file():
            if raw_data_filename in f:
                if args.clear:
                    del f[raw_data_filename]
                else:
                    print("Already prepared {}".format(raw_data_filename))
                    continue

            g = f.require_group(raw_data_filename)

            # 1. Extract the windows per chromosome
            print("Prepare {}: chunking".format(raw_data_filename), end="", flush=True)
            data = bigwig.chunk(
                raw_data_filepath,
                settings["window_size"],
                step_size,
                settings["resolution"],
                settings["training"] + settings["testing"],
                verbose=args.verbose,
                print_per_chrom=print_progress,
            )

            # 2. Concat training and test data (separately) into two arrays
            print(" merging... ", end="", flush=True)
            num_training = len(settings["training"])
            train_num = 0
            test_num = 0

            for i in range(num_training):
                train_num += data[i].shape[0]

            for i in range(num_training, len(data)):
                test_num += data[i].shape[0]

            data_train = np.zeros((train_num, bins_per_window))
            data_test = np.zeros((test_num, bins_per_window))

            k = 0
            for i in range(num_training):
                l = k + data[i].shape[0]
                data_train[k:l,] = data[i]
                k = l

            k = 0
            for i in range(num_training, len(data)):
                l = k + data[i].shape[0]
                data_test[k:l,] = data[i]
                k = l

            # 3. Normalize data
            print("normalizing... ", end="", flush=True)
            cutoff = np.percentile(data_train, tuple(settings["percentile_cutoff"]))
            data_train[np.where(data_train < cutoff[0])] = cutoff[0]
            data_train[np.where(data_train > cutoff[1])] = cutoff[1]

            cutoff = np.percentile(data_test, tuple(settings["percentile_cutoff"]))
            data_test[np.where(data_test < cutoff[0])] = cutoff[0]
            data_test[np.where(data_test > cutoff[1])] = cutoff[1]

            if args.verbose:
                print(
                    "Max: train {} | test {}".format(
                        np.max(data_train), np.max(data_test)
                    )
                )

            data_train = MinMaxScaler().fit_transform(data_train)
            data_test = MinMaxScaler().fit_transform(data_test)

            # 4. Pickle data
            print("saving...")
            g.create_dataset("training", data=data_train)
            g.create_dataset("testing", data=data_test)
        else:
            print("Already downloaded {}".format(raw_data_filename))
