#!/usr/bin/env python

import argparse
import h5py
import json
import os
import pathlib
import sys

from ae import utils
from server import bigwig


parser = argparse.ArgumentParser(description="Peax Preparer")
parser.add_argument(
    "-s", "--settings", help="path to the settings file", default="settings.json"
)
parser.add_argument(
    "-c", "--clear", action="store_true", help="clears previously prepared data"
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="turn on verbose logging"
)

args = parser.parse_args()

try:
    with open(args.settings, "r") as f:
        settings = json.load(f)
except FileNotFoundError:
    print("You need to provide a settings file via `--settings`")
    sys.exit(2)

# Create data directory
pathlib.Path("data").mkdir(parents=True, exist_ok=True)

window_size = settings["window_size"]
step_size = window_size // settings["step_frequency"]
bins_per_window = window_size // settings["resolution"]

if args.verbose:
    print("Window size: {}bp".format(settings["window_size"]))
    print("Resolution: {}bp".format(settings["resolution"]))
    print("Bins per window: {}".format(bins_per_window))
    print("Step frequency per window: {}".format(settings["step_frequency"]))
    print("Step size: {}bp".format(step_size))
    print("Chromosomes: {}".format(", ".join(settings["chromosomes"])))
    print("Dev set size: {}%".format(settings["dev_set_size"] * 100))
    print("Test set size: {}%".format(settings["test_set_size"] * 100))
    print(
        "Percentile cut off: [{}]".format(
            ", ".join([str(x) for x in settings["percentile_cutoff"]])
        )
    )

chromosomes = settings["chromosomes"]
datasets = settings["datasets"]
data_types = ["signal", "narrow_peaks", "broad_peaks"]
stored_data = [
    "data_train",
    "data_dev",
    "data_test",
    "peaks_train",
    "peaks_dev",
    "peaks_test",
    "shuffling",
    "settings",
]


def print_progress():
    print(".", end="", flush=True)


for dataset_name in datasets:
    dataset = datasets[dataset_name]
    has_all_data_types = set(data_types).issubset(dataset.keys())

    assert has_all_data_types, "Dataset should contain all data types"

    print("\nPrepare dataset {}".format(dataset_name))

    prep_data_filename = "{}.h5".format(dataset_name)
    prep_data_filepath = os.path.join("data", prep_data_filename)

    with h5py.File(prep_data_filepath, "a") as f:
        if dataset_name in f and args.clear:
            del f[dataset_name]

        ds = f.require_group(dataset_name)

        filename_signal = dataset["signal"]
        filepath_signal = os.path.join("data", filename_signal)
        filename_narrow_peaks = dataset["narrow_peaks"]
        filepath_narrow_peaks = os.path.join("data", filename_narrow_peaks)
        filename_broad_peaks = dataset["broad_peaks"]
        filepath_broad_peaks = os.path.join("data", filename_broad_peaks)

        files_are_available = [
            pathlib.Path(filepath_signal).is_file(),
            pathlib.Path(filepath_narrow_peaks).is_file(),
            pathlib.Path(filepath_broad_peaks).is_file(),
        ]

        assert all(files_are_available), "All files of the data should be available"

        [x in ds for x in stored_data]

        # If all datasets e
        if all([x in ds for x in stored_data]):
            print("Already prepared {}. Skipping".format(dataset_name))
            continue

        # Since we don't know if the settings have change we need to remove all existing
        # datasets
        if len(ds) > 0:
            print("Remove incomplete data to avoid inconsistencies")
            for data in ds:
                del ds[data]

        # 0. Sanity check
        chrom_sizes_signal = bigwig.get_chromsizes(filepath_signal)
        chrom_sizes_narrow_peaks = bigwig.get_chromsizes(filepath_narrow_peaks)
        chrom_sizes_broad_peaks = bigwig.get_chromsizes(filepath_broad_peaks)

        signal_has_all_chroms = [chrom in chrom_sizes_signal for chrom in chromosomes]
        narrow_peaks_has_all_chroms = [
            chrom in chrom_sizes_narrow_peaks for chrom in chromosomes
        ]
        broad_peaks_has_all_chroms = [
            chrom in chrom_sizes_broad_peaks for chrom in chromosomes
        ]

        assert all(signal_has_all_chroms), "Signal should have all chromosomes"
        assert all(
            narrow_peaks_has_all_chroms
        ), "Narrow peaks should have all chromosomes"
        assert all(
            broad_peaks_has_all_chroms
        ), "Broad peaks should have all chromosomes"

        # 1. Extract the windows, narrow peaks, and broad peaks per chromosome
        print("Extract windows from {}".format(filename_signal), end="", flush=True)
        data = bigwig.chunk(
            filepath_signal,
            window_size,
            settings["resolution"],
            step_size,
            chromosomes,
            verbose=args.verbose,
            print_per_chrom=print_progress,
        )
        print(
            "\nExtract narrow peaks from {}".format(filename_narrow_peaks),
            end="",
            flush=True,
        )
        narrow_peaks = utils.chunk_beds_binary(
            filepath_narrow_peaks,
            window_size,
            step_size,
            chromosomes,
            verbose=args.verbose,
            print_per_chrom=print_progress,
        )
        print(
            "\nExtract broad peaks from {}".format(filename_broad_peaks),
            end="",
            flush=True,
        )
        broad_peaks = utils.chunk_beds_binary(
            filepath_broad_peaks,
            window_size,
            step_size,
            chromosomes,
            verbose=args.verbose,
            print_per_chrom=print_progress,
        )

        # 4. Under-sampling: remove the majority of empty windows
        print("\nSelect windows to balance peaky and non-peaky ratio")
        selected_windows = utils.filter_windows_by_peaks(
            data,
            narrow_peaks,
            broad_peaks,
            incl_pctl_total_signal=settings["incl_pctl_total_signal"],
            incl_pct_no_signal=settings["incl_pct_no_signal"],
            verbose=args.verbose,
        )
        data_filtered = data[selected_windows]
        peaks_filtered = ((narrow_peaks + broad_peaks).flatten() > 0)[selected_windows]

        # 5. Shuffle and split data into a train, dev, and test set
        (
            data_train,
            data_dev,
            data_test,
            peaks_train,
            peaks_dev,
            peaks_test,
            shuffling,
        ) = utils.split_train_dev_test(
            data_filtered,
            peaks_filtered,
            settings["dev_set_size"],
            settings["test_set_size"],
            settings["rnd_seed"],
            verbose=True,
        )

        # 6. Pickle data
        print("Saving... ", end="", flush=True)
        ds.create_dataset("data_train", data=data_train)
        ds.create_dataset("data_dev", data=data_dev)
        ds.create_dataset("data_test", data=data_test)
        ds.create_dataset("peaks_train", data=peaks_train)
        ds.create_dataset("peaks_dev", data=peaks_dev)
        ds.create_dataset("peaks_test", data=peaks_test)
        ds.create_dataset("shuffling", data=shuffling)
        ds.create_dataset("settings", data=json.dumps(settings))

        print("done!")
