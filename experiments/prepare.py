#!/usr/bin/env python

import argparse
import h5py
import json
import numpy as np
import os
import pathlib
import sys

from itertools import product
from string import Template

from ae import utils
from server import bigwig


slurm_header = """#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p cox
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=12000
#SBATCH --array=0-$num_datasets
#SBATCH -t 2-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lekschas@g.harvard.edu
#SBATCH -o /n/pfister_lab/lekschas/peax/experiments/logs/prepare-out-%A-%a.txt
#SBATCH -e /n/pfister_lab/lekschas/peax/experiments/logs/prepare-err-%A-%a.txt

# add additional commands needed for Lmod and module loads here
source new-modules.sh
module load Anaconda/5.0.1-fasrc01
"""

slurm_body = Template(
    """
# add commands for analyses here
cd /n/pfister_lab/haehn/Projects/peax/experiments/
source activate /n/pfister_lab/haehn/ENVS/peax
python prepare.py \
  --type $dtype \
  --datasets $datasets \
  --settings $settings \
  --single-dataset-idx $single_dataset_idx \
  --silent \
  $roadmap \
  $clear

# end of program
exit 0;
"""
)


def prepare_chip(
    f,
    dataset,
    dataset_name,
    settings,
    data_dir,
    stored_data,
    chromosomes,
    window_size,
    step_size,
    clear: bool = False,
    print_progress: callable = None,
    verbose: bool = False,
    silent: bool = False,
):
    tqdm = utils.get_tqdm()

    targets_iter = (
        tqdm(settings["targets"], desc="targets", unit="target")
        if not silent
        else settings["targets"]
    )

    total_signal = None
    total_peaks = None

    for target in targets_iter:
        filename_signal = dataset["{}_fc_signal".format(target)]
        filepath_signal = os.path.join(data_dir, filename_signal)
        filename_narrow_peaks = dataset["{}_narrow_peaks".format(target)]
        filepath_narrow_peaks = os.path.join(data_dir, filename_narrow_peaks)
        filename_broad_peaks = dataset["{}_broad_peaks".format(target)]
        filepath_broad_peaks = os.path.join(data_dir, filename_broad_peaks)

        files_are_available = [
            pathlib.Path(filepath_signal).is_file(),
            pathlib.Path(filepath_narrow_peaks).is_file(),
            pathlib.Path(filepath_broad_peaks).is_file(),
        ]

        assert all(files_are_available), "Not all data files are available"

        # If all datasets are available
        if all([x in f for x in stored_data]) and not clear:
            print("Already prepared {}. Skipping".format(dataset_name))
            return None

        # Since we don't know if the settings have change we need to remove all existing
        # datasets
        if len(f) > 0:
            if verbose:
                print("Remove incomplete data to avoid inconsistencies")
            for data in f:
                del f[data]

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

        print_per_chrom = None
        pbar = None

        # 1. Extract the windows, narrow peaks, and broad peaks per chromosome
        if verbose:
            print("Extract windows from {}".format(filename_signal), end="", flush=True)
            print_per_chrom = print_progress
        elif not silent:
            pbar = tqdm(
                total=len(chromosomes) * 3,
                leave=False,
                desc="Chromosomes",
                unit="chromosome",
            )

            def update_pbar():
                pbar.update(1)

            print_per_chrom = update_pbar

        data = bigwig.chunk(
            filepath_signal,
            window_size,
            settings["resolution"],
            step_size,
            chromosomes,
            verbose=verbose,
            print_per_chrom=print_per_chrom,
        )

        if verbose:
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
            verbose=verbose,
            print_per_chrom=print_per_chrom,
        )

        if verbose:
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
            verbose=verbose,
            print_per_chrom=print_per_chrom,
        )

        if pbar is not None:
            pbar.close()

        # 4. Under-sampling: remove the majority of empty windows
        if verbose:
            print("\nSelect windows to balance peaky and non-peaky ratio")

        selected_windows = utils.filter_windows_by_peaks(
            data,
            narrow_peaks,
            broad_peaks,
            incl_pctl_total_signal=settings["incl_pctl_total_signal"],
            incl_pct_no_signal=settings["incl_pct_no_signal"],
            peak_ratio=settings["peak_ratio"],
            verbose=verbose,
        )

        data_filtered = data[selected_windows]
        peaks_filtered = ((narrow_peaks + broad_peaks).flatten() > 0)[selected_windows]

        if total_signal is None:
            total_signal = data_filtered
        else:
            total_signal = np.concatenate((total_signal, data_filtered), axis=0)

        if total_peaks is None:
            total_peaks = peaks_filtered
        else:
            total_peaks = np.concatenate((total_peaks, peaks_filtered), axis=0)

    # 5. Shuffle and split data into a train, dev, and test set
    (
        data_train,
        peaks_train,
        data_dev,
        peaks_dev,
        data_test,
        peaks_test,
        shuffling,
    ) = utils.split_train_dev_test(
        data_filtered,
        peaks_filtered,
        settings["dev_set_size"],
        settings["test_set_size"],
        settings["rnd_seed"],
        verbose=verbose,
    )

    # 6. Reshape to be directly usable by Keras
    data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], 1)
    data_dev = data_dev.reshape(data_dev.shape[0], data_dev.shape[1], 1)
    data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], 1)

    return (
        data_train,
        peaks_train,
        data_dev,
        peaks_dev,
        data_test,
        peaks_test,
        shuffling,
    )


def prepare_dnase(
    f,
    dataset,
    dataset_name,
    settings,
    data_dir,
    stored_data,
    chromosomes,
    window_size,
    step_size,
    clear: bool = False,
    print_progress: callable = None,
    verbose: bool = False,
    silent: bool = False,
):
    tqdm = utils.get_tqdm()

    filename_signal = "{}.bigWig".format(dataset["rdn_signal"])
    filepath_signal = os.path.join(data_dir, filename_signal)
    filename_narrow_peaks = "{}.bigBed".format(dataset["narrow_peaks"])
    filepath_narrow_peaks = os.path.join(data_dir, filename_narrow_peaks)
    filename_broad_peaks = "{}.bigBed".format(dataset["broad_peaks"])
    filepath_broad_peaks = os.path.join(data_dir, filename_broad_peaks)

    files_are_available = [
        pathlib.Path(filepath_signal).is_file(),
        pathlib.Path(filepath_narrow_peaks).is_file(),
        pathlib.Path(filepath_broad_peaks).is_file(),
    ]

    assert all(files_are_available), "Not all data files are available"

    # If all datasets are available
    if all([x in f for x in stored_data]) and not clear:
        print("Already prepared {}. Skipping".format(dataset_name))
        return None

    # Since we don't know if the settings have change we need to remove all existing
    # datasets
    if len(f) > 0:
        if verbose:
            print("Remove incomplete data to avoid inconsistencies")
        for data in f:
            del f[data]

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
    assert all(narrow_peaks_has_all_chroms), "Narrow peaks should have all chromosomes"
    assert all(broad_peaks_has_all_chroms), "Broad peaks should have all chromosomes"

    print_per_chrom = None
    pbar = None

    # 1. Extract the windows, narrow peaks, and broad peaks per chromosome
    if verbose:
        print("Extract windows from {}".format(filename_signal), end="", flush=True)
        print_per_chrom = print_progress
    elif not silent:
        pbar = tqdm(
            total=len(chromosomes) * 3,
            leave=False,
            desc="Chromosomes",
            unit="chromosome",
        )

        def update_pbar():
            pbar.update(1)

        print_per_chrom = update_pbar

    data = bigwig.chunk(
        filepath_signal,
        window_size,
        settings["resolution"],
        step_size,
        chromosomes,
        verbose=verbose,
        print_per_chrom=print_per_chrom,
    )

    if verbose:
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
        verbose=verbose,
        print_per_chrom=print_per_chrom,
    )

    if verbose:
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
        verbose=verbose,
        print_per_chrom=print_per_chrom,
    )

    if pbar is not None:
        pbar.close()

    # 4. Under-sampling: remove the majority of empty windows
    if verbose:
        print("\nSelect windows to balance peaky and non-peaky ratio")

    selected_windows = utils.filter_windows_by_peaks(
        data,
        narrow_peaks,
        broad_peaks,
        incl_pctl_total_signal=settings["incl_pctl_total_signal"],
        incl_pct_no_signal=settings["incl_pct_no_signal"],
        peak_ratio=settings["peak_ratio"],
        verbose=verbose,
    )
    data_filtered = data[selected_windows]
    peaks_filtered = ((narrow_peaks + broad_peaks).flatten() > 0)[selected_windows]

    # 5. Shuffle and split data into a train, dev, and test set
    (
        data_train,
        peaks_train,
        data_dev,
        peaks_dev,
        data_test,
        peaks_test,
        shuffling,
    ) = utils.split_train_dev_test(
        data_filtered,
        peaks_filtered,
        settings["dev_set_size"],
        settings["test_set_size"],
        settings["rnd_seed"],
        verbose=verbose,
    )

    # 6. Reshape to be directly usable by Keras
    data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], 1)
    data_dev = data_dev.reshape(data_dev.shape[0], data_dev.shape[1], 1)
    data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], 1)

    return (
        data_train,
        peaks_train,
        data_dev,
        peaks_dev,
        data_test,
        peaks_test,
        shuffling,
    )


def prepare(
    dtype: str,
    datasets: dict,
    settings: dict,
    single_dataset: str = None,
    single_dataset_idx: int = -1,
    base: str = ".",
    roadmap: bool = False,
    clear: bool = False,
    verbose: bool = False,
    silent: bool = False,
):
    dtype = dtype.lower()
    supported_dtypes = ["dnase", "chip"]

    if dtype.lower() not in supported_dtypes:
        print("Unknown data type: {}".format(dtype))
        print("Use one of {}".format(", ".join(supported_dtypes)))

    tqdm = utils.get_tqdm()

    # Create data directory
    data_dir = os.path.join(base, "data")
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    window_size = settings["window_size"]
    step_size = window_size // settings["step_frequency"]
    bins_per_window = window_size // settings["resolution"]

    if verbose:
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

    if roadmap:
        data_types = [
            "_".join(x)
            for x in product(settings["targets"], settings["data_types"].keys())
        ]
    else:
        data_types = list(settings["data_types"].keys())

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

    if single_dataset is not None:
        try:
            datasets = {single_dataset: datasets[single_dataset]}
        except KeyError:
            sys.stderr.write("Dataset not found: {}\n".format(single_dataset))
            return

    if single_dataset_idx >= 0:
        try:
            # Make sure the dataset names are sorted to avoid any inconsistencies
            datasets_idx = list(datasets.keys())
            datasets_idx.sort()
            single_dataset = datasets_idx[single_dataset_idx]
            datasets = {single_dataset: datasets[single_dataset]}
        except IndexError:
            sys.stderr.write("Dataset not found: #{}\n".format(single_dataset_idx))
            return

    datasets_iter = (
        datasets
        if verbose or silent
        else tqdm(datasets, desc="Datasets", unit="dataset")
    )

    for dataset_name in datasets_iter:
        samples = datasets[dataset_name]

        samples_iter = (
            samples
            if verbose or silent
            else tqdm(samples, desc="Samples", leave=False, unit="sample")
        )

        for sample_id in samples_iter:
            dataset = samples[sample_id]
            has_all_data_types = set(data_types).issubset(dataset.keys())

            if not has_all_data_types:
                sys.stderr.write(
                    "Dataset definition should specify all data types: {}\n".format(
                        ", ".join(data_types)
                    )
                )
                continue

            if verbose:
                print("\nPrepare dataset {}".format(dataset_name))

            prep_data_filename = "{}_w-{}_f-{}_r-{}.h5".format(
                dataset_name,
                settings["window_size"],
                settings["step_frequency"],
                settings["resolution"],
            )
            prep_data_filepath = os.path.join(data_dir, prep_data_filename)

            with h5py.File(prep_data_filepath, "a") as f:
                prepared_data = None
                if dtype == "dnase":
                    prepared_data = prepare_dnase(
                        f,
                        dataset,
                        dataset_name,
                        settings,
                        data_dir,
                        stored_data,
                        chromosomes,
                        window_size,
                        step_size,
                        clear=clear,
                        print_progress=print_progress,
                        verbose=verbose,
                        silent=silent,
                    )

                elif dtype == "chip":
                    prepared_data = prepare_chip(
                        f,
                        dataset,
                        dataset_name,
                        settings,
                        data_dir,
                        stored_data,
                        chromosomes,
                        window_size,
                        step_size,
                        clear=clear,
                        print_progress=print_progress,
                        verbose=verbose,
                        silent=silent,
                    )

                else:
                    print("Unsupported data type: {}".format(dtype))
                    sys.exit(2)

                if prepared_data is not None:
                    (
                        data_train,
                        peaks_train,
                        data_dev,
                        peaks_dev,
                        data_test,
                        peaks_test,
                        shuffling,
                    ) = prepared_data
                else:
                    continue

                # 7. Pickle data
                if verbose:
                    print("Saving... ", end="", flush=True)

                utils.create_hdf5_dset(f, "data_train", data_train, extendable=True)
                utils.create_hdf5_dset(f, "data_dev", data_dev, extendable=True)
                utils.create_hdf5_dset(f, "data_test", data_test, extendable=True)
                utils.create_hdf5_dset(f, "peaks_train", peaks_train, extendable=True)
                utils.create_hdf5_dset(f, "peaks_dev", peaks_dev, extendable=True)
                utils.create_hdf5_dset(f, "peaks_test", peaks_test, extendable=True)
                utils.create_hdf5_dset(f, "shuffling", shuffling, extendable=True)
                utils.create_hdf5_dset(f, "settings", json.dumps(settings))

                if verbose:
                    print("done!")


def prepare_jobs(
    dtype: str,
    datasets: str,
    settings: str,
    base: str = ".",
    roadmap: bool = False,
    clear: bool = False,
    verbose: bool = False,
):
    try:
        with open(os.path.join(base, datasets), "r") as f:
            datasets_dict = json.load(f)
    except FileNotFoundError:
        sys.stderr.write(
            "Could not find datasets file: {}\n".format(os.path.join(base, datasets))
        )
        return

    num_datasets = len(datasets_dict)

    new_slurm_body = slurm_body.substitute(
        dtype=dtype,
        datasets=datasets,
        settings=settings,
        single_dataset_idx="$SLURM_ARRAY_TASK_ID",
        roadmap="--roadmap" if roadmap else "",
        clear="--clear" if clear else "",
    )
    slurm = (
        slurm_header.replace("$num_datasets", str(num_datasets - 1)) + new_slurm_body
    )

    slurm_file = os.path.join(base, "prepare.slurm")

    with open(slurm_file, "w") as f:
        f.write(slurm)

    print("Created a slurm file for preparing {} datasets".format(num_datasets))


def get_roadmap_datasets(datasets: list, data_type: str):
    if data_type == "chip":
        targets = [
            "H3K4me1",
            "H3K4me3",
            "H3K27ac",
            "H3K27me3",
            "H3K9ac",
            "H3K9me3",
            "H3K36me3",
        ]
        out = {}

        for e_id in datasets:
            out[e_id] = {"1": {}}

            for target in targets:
                out[e_id]["1"][
                    "{}_fc_signal".format(target)
                ] = "{}-{}.fc.signal.bigWig".format(e_id, target)
                out[e_id]["1"][
                    "{}_narrow_peaks".format(target)
                ] = "{}-{}.peaks.narrow.bigBed".format(e_id, target)
                out[e_id]["1"][
                    "{}_broad_peaks".format(target)
                ] = "{}-{}.peaks.broad.bigBed".format(e_id, target)

        return out

    else:
        sys.stderr.write("Unsupported datatype: {}\n".format(data_type))
        sys.exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Preparer")
    parser.add_argument(
        "-t",
        "--type",
        help="specify the data type",
        choices=["dnase", "chip"],
        type=str.lower,
    )
    parser.add_argument(
        "-d", "--datasets", help="path to the datasets file", default="datasets.json"
    )
    parser.add_argument(
        "-s", "--settings", help="path to the settings file", default="settings.json"
    )
    parser.add_argument(
        "-i", "--single-dataset", help="name of a specific dataset to prepare"
    )
    parser.add_argument(
        "-x",
        "--single-dataset-idx",
        help="index of a specific dataset to prepare",
        type=int,
        default=-1,
    )
    parser.add_argument("-j", "--jobs", action="store_true", help="create jobs files")
    parser.add_argument(
        "-c", "--clear", action="store_true", help="clears previously prepared data"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="turn on verbose logging"
    )
    parser.add_argument(
        "-z", "--silent", action="store_true", help="disable all but error logs"
    )
    parser.add_argument(
        "-r", "--roadmap", action="store_true", help="prepare roadmap data"
    )

    args = parser.parse_args()

    if args.type is None:
        sys.stderr.write("You need to provide the data type via `--type`\n")
        sys.exit(2)

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

    if args.single_dataset is not None and args.single_dataset_idx >= 0:
        sys.stderr.write("Either provide a dataset name or index but not both\n")
        sys.exit(2)

    if args.jobs:
        prepare_jobs(
            args.type,
            args.datasets,
            args.settings,
            roadmap=args.roadmap,
            clear=args.clear,
            verbose=args.verbose,
        )
    else:
        if args.roadmap:
            datasets = get_roadmap_datasets(datasets, args.type)

        prepare(
            args.type,
            datasets,
            settings,
            single_dataset=args.single_dataset,
            single_dataset_idx=args.single_dataset_idx,
            roadmap=args.roadmap,
            clear=args.clear,
            verbose=args.verbose,
            silent=args.silent,
        )
