#!/usr/bin/env python

import argparse
import h5py
import json
import os
import pathlib
import sys
from string import Template

from ae import utils
from server import bigwig


slurm_header = """#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p holyseasgpu
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --mem=24000
#SBATCH -t 7-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu
#SBATCH -o /n/pfister_lab/lekschas/peax/experiments/logs/prepare-out-$name.txt
#SBATCH -e /n/pfister_lab/lekschas/peax/experiments/logs/prepare-err-$name.txt

# add additional commands needed for Lmod and module loads here
source new-modules.sh
module load Anaconda/5.0.1-fasrc01
"""

slurm_body = Template(
    """
# add commands for analyses here
cd /n/pfister_lab/lekschas/peax/experiments/
source activate /n/pfister_lab/haehn/ENVS/peax
python prepare.py --type $dtype --datasets $datasets --settings $settings --single-dataset $single_dataset

# end of program
exit 0;
"""
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

    # 1. Extract the windows, narrow peaks, and broad peaks per chromosome
    if verbose:
        print("Extract windows from {}".format(filename_signal), end="", flush=True)
        print_per_chrom = print_progress
    else:
        pbar = tqdm(total=len(chromosomes) * 3, leave=False, unit="chromosome")

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
    base: str = ".",
    clear: bool = False,
    verbose: bool = False,
):
    dtype = dtype.lower()
    supported_dtypes = ["DNase"]

    if dtype != "dnase":
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

    def create_dataset(f, name, data, extendable: bool = False):
        if name in f.keys():
            if extendable:
                f[name].resize((f[name].shape[0] + data.shape[0]), axis=0)
                f[name][-data.shape[0] :] = data
            else:
                # Overwrite existing dataset
                del f[name]
                f.create_dataset(name, data=data)
        else:
            if extendable:
                maxshape = (None, *data.shape[1:])
                f.create_dataset(name, data=data, maxshape=maxshape)
            else:
                f.create_dataset(name, data=data)

    if single_dataset is not None:
        try:
            datasets = {single_dataset: datasets[single_dataset]}
        except KeyError:
            print("Dataset not found: {}".format())

    datasets_iter = (
        datasets if verbose else tqdm(datasets, desc="Datasets", unit="dataset")
    )

    for dataset_name in datasets_iter:
        samples = datasets[dataset_name]

        samples_iter = (
            samples
            if verbose
            else tqdm(samples, desc="Samples", leave=False, unit="sample")
        )

        for sample_id in samples_iter:
            dataset = samples[sample_id]
            has_all_data_types = set(data_types).issubset(dataset.keys())

            if not has_all_data_types:
                print(
                    "Dataset definition should specify all data types: {}".format(
                        ", ".join(data_types)
                    )
                )
                continue

            if verbose:
                print("\nPrepare dataset {}".format(dataset_name))

            prep_data_filename = "{}.h5".format(dataset_name)
            prep_data_filepath = os.path.join(data_dir, prep_data_filename)

            with h5py.File(prep_data_filepath, "a") as f:
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
                    )

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
                else:
                    print("Unsupported data type: {}".format(dtype))
                    sys.exit(2)

                # 7. Pickle data
                if verbose:
                    print("Saving... ", end="", flush=True)

                create_dataset(f, "data_train", data_train, extendable=True)
                create_dataset(f, "data_dev", data_dev, extendable=True)
                create_dataset(f, "data_test", data_test, extendable=True)
                create_dataset(f, "peaks_train", peaks_train, extendable=True)
                create_dataset(f, "peaks_dev", peaks_dev, extendable=True)
                create_dataset(f, "peaks_test", peaks_test, extendable=True)
                create_dataset(f, "shuffling", shuffling, extendable=True)
                create_dataset(f, "settings", json.dumps(settings))

                if verbose:
                    print("done!")


def prepare_jobs(
    dtype: str,
    datasets: str,
    settings: str,
    base: str = ".",
    clear: bool = False,
    verbose: bool = False,
):
    tqdm = utils.get_tqdm()

    # Create slurm directory
    pathlib.Path("prepare").mkdir(parents=True, exist_ok=True)

    try:
        with open(os.path.join(base, datasets), "r") as f:
            datasets_dict = json.load(f)
    except FileNotFoundError:
        print("You need to provide a datasets file via `--datasets`")
        sys.exit(2)

    datasets_iter = (
        datasets_dict
        if verbose
        else tqdm(datasets_dict, desc="Datasets", unit="dataset")
    )

    for dataset_name in datasets_iter:
        new_slurm_body = slurm_body.substitute(
            dtype=dtype,
            datasets=datasets,
            settings=settings,
            single_dataset=dataset_name,
        )
        slurm = slurm_header.replace("$name", dataset_name) + new_slurm_body

        slurm_file = os.path.join(base, "prepare", "{}.slurm".format(dataset_name))

        if not pathlib.Path(slurm_file).is_file() or clear:
            with open(slurm_file, "w") as f:
                f.write(slurm)
        else:
            print("Job file already exists. Use `--clear` to overwrite it.")

    print("Created slurm files for preparing {} datasets".format(len(datasets_dict)))


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
    parser.add_argument("-j", "--jobs", action="store_true", help="create jobs files")
    parser.add_argument(
        "-c", "--clear", action="store_true", help="clears previously prepared data"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="turn on verbose logging"
    )

    args = parser.parse_args()

    if args.type is None:
        print("You need to provide the data type via `--type`")
        sys.exit(2)

    try:
        with open(args.datasets, "r") as f:
            datasets = json.load(f)
    except FileNotFoundError:
        print("You need to provide a datasets file via `--datasets`")
        sys.exit(2)

    try:
        with open(args.settings, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("You need to provide a settings file via `--settings`")
        sys.exit(2)

    if args.jobs:
        prepare_jobs(
            args.type,
            args.datasets,
            args.settings,
            clear=args.clear,
            verbose=args.verbose,
        )
    else:
        prepare(
            args.type,
            datasets,
            settings,
            single_dataset=args.single_dataset,
            clear=args.clear,
            verbose=args.verbose,
        )
