#!/usr/bin/env python

# LAUNCH THE JOBS WITH:
# for f in *.slurm; do sbatch $f; done

import argparse
import itertools as it
import json
import os
import pathlib
import sys

from string import Template
from ae.utils import namify

from ae.utils import get_tqdm


slurm_header = """#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p $cluster
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --mem=24000
#SBATCH --array=0-$num_definitions
#SBATCH -t $time
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lekschas@g.harvard.edu
#SBATCH -o /n/pfister_lab/lekschas/peax/experiments/logs/cae-out-%A-%a.txt
#SBATCH -e /n/pfister_lab/lekschas/peax/experiments/logs/cae-err-%A-%a.txt

# add additional commands needed for Lmod and module loads here
source new-modules.sh
module load Anaconda/5.0.1-fasrc01
"""

slurm_body = Template(
    """
# add commands for analyses here
cd /n/pfister_lab/haehn/Projects/peax/experiments/
source activate /n/pfister_lab/haehn/ENVS/peax
python train.py \\
  --definitions $definitions \\
  --definition-idx $definition_idx \\
  $datasets \\
  --settings $settings \\
  --epochs $epochs \\
  --batch_size $batch_size \\
  --peak-weight $peak_weight \\
  --signal-weighting $signal_weighting \\
  --signal-weighting-zero-point-percentage $signal_weighting_zero_point_percentage \\
  --silent

# end of program
exit 0;
"""
)


def jobs(
    search_filename: str,
    settings: str,
    datasets: str = None,
    dataset: str = None,
    name: str = None,
    epochs: int = None,
    batch_size: int = None,
    peak_weight: float = None,
    signal_weighting: str = None,
    signal_weighting_zero_point_percentage: float = None,
    cluster: str = "cox",
    base: str = ".",
    clear: bool = False,
    verbose: bool = False,
    repeat: int = 1,
):
    search_name = os.path.splitext(search_filename)[0]
    try:
        with open(os.path.join(base, search_filename), "r") as f:
            search = json.load(f)
    except FileNotFoundError:
        sys.stderr.write("Please provide a neural network search file\n")
        sys.exit(2)

    tqdm = get_tqdm()

    # Create models and slurm directory
    pathlib.Path(os.path.join(base, "models")).mkdir(parents=True, exist_ok=True)

    varying = search["hyperparameters"]["varying"]
    fixed = search["hyperparameters"]["fixed"]
    epochs = epochs if epochs is not None else search["epochs"]
    batch_size = batch_size if batch_size is not None else search["batch_size"]

    try:
        peak_weight = peak_weight if peak_weight is not None else search["peak_weight"]
    except KeyError:
        peak_weight = 1

    try:
        signal_weighting = (
            signal_weighting
            if signal_weighting is not None
            else search["signal_weighting"]
        )
    except KeyError:
        signal_weighting = "none"

    try:
        signal_weighting_zero_point_percentage = (
            signal_weighting_zero_point_percentage
            if signal_weighting_zero_point_percentage is not None
            else search["signal_weighting_zero_point_percentage"]
        )
    except KeyError:
        signal_weighting_zero_point_percentage = 0

    base_def = dict({}, **fixed)

    # Get the product of all possible combinations
    combinations = []
    for values in varying["values"]:
        combinations += list(it.product(*values))

    def finalize_def(prelim_def: dict) -> dict:
        conv_layers = prelim_def["conv_layers"]
        conv_filter_size = prelim_def["conv_filter_size"]
        conv_filter_size_reverse_order = prelim_def["conv_filter_size_reverse_order"]
        conv_kernel_size = prelim_def["conv_kernel_size"]
        conv_kernel_size_reverse_order = prelim_def["conv_kernel_size_reverse_order"]

        if isinstance(conv_filter_size, int) or isinstance(conv_filter_size, float):
            conv_filter_size = int(conv_filter_size)
            conv_filters = [conv_filter_size] * conv_layers
        elif (
            isinstance(conv_filter_size, list) and len(conv_filter_size) == conv_layers
        ):
            if conv_filter_size_reverse_order:
                conv_filters = list(reversed(conv_filter_size))
            else:
                conv_filters = conv_filter_size
        else:
            print("ERROR: conv_filter_size needs to be an int or a list of ints")
            sys.exit(1)

        if isinstance(conv_kernel_size, int) or isinstance(conv_kernel_size, float):
            conv_kernel_size = int(conv_kernel_size)
            conv_kernels = [conv_kernel_size] * conv_layers
        elif (
            isinstance(conv_kernel_size, list) and len(conv_kernel_size) == conv_layers
        ):
            if conv_kernel_size_reverse_order:
                conv_kernels = list(reversed(conv_kernel_size))
            else:
                conv_kernels = conv_kernel_size
        else:
            print("ERROR: conv_kernel_size needs to be an int or a list of ints")
            sys.exit(1)

        dense_units = prelim_def["dense_units"]

        if "dropouts" not in prelim_def and "dropout" in prelim_def:
            dropouts = [prelim_def["dropout"]] * (conv_layers + len(dense_units))
        else:
            dropouts = prelim_def["dropouts"]

        if "batch_norms" not in prelim_def and "batch_norm" in prelim_def:
            batch_norm = [prelim_def["batch_norm"]] * (conv_layers + len(dense_units))
        elif "batch_norms" not in prelim_def and "batch_norm" not in prelim_def:
            batch_norm = [False] * (conv_layers + len(dense_units))
        else:
            batch_norm = prelim_def["batch_norms"]

        if "batch_norm_input" in prelim_def:
            batch_norm_input = prelim_def["batch_norm_input"]
        else:
            batch_norm_input = False

        optimizer = prelim_def["optimizer"]
        if optimizer == "adam":
            if "adam_b1" in prelim_def:
                optimizer += "-" + str(prelim_def["adam_b1"])

            if "adam_b2" in prelim_def:
                optimizer += "-" + str(prelim_def["adam_b2"])

        return {
            "conv_filters": conv_filters,
            "conv_kernels": conv_kernels,
            "dense_units": prelim_def["dense_units"],
            "dropouts": dropouts,
            "embedding": prelim_def["embedding"],
            "reg_lambda": prelim_def["reg_lambda"],
            "optimizer": optimizer,
            "learning_rate": prelim_def["learning_rate"],
            "learning_rate_decay": prelim_def["learning_rate_decay"],
            "loss": prelim_def["loss"],
            "metrics": prelim_def["metrics"],
            "batch_norm": batch_norm,
            "batch_norm_input": batch_norm_input,
            "peak_weight": prelim_def["peak_weight"],
            "signal_weighting": prelim_def["signal_weighting"],
            "signal_weighting_zero_point_percentage": prelim_def[
                "signal_weighting_zero_point_percentage"
            ],
        }

    model_names = []

    if len(combinations) == 0:
        combinations = [[]] * repeat

    skipped = 0
    for i, combination in enumerate(tqdm(combinations, desc="Jobs", unit="job")):
        combined_def = dict({}, **base_def)

        for j, value in enumerate(combination):
            combined_def[varying["params"][i]] = value

        final_def = finalize_def(combined_def)
        model_name = namify(final_def)
        if repeat > 1:
            model_name = "{}__{}".format(model_name, i)
        def_file = os.path.join(base, "models", "{}.json".format(model_name))

        if not pathlib.Path(def_file).is_file() or clear:
            with open(def_file, "w") as f:
                json.dump(final_def, f, indent=2)
        else:
            skipped += 1

        model_names.append(model_name)

    if skipped > 0:
        print(
            "Skipped creating {} definition files as they already exists".format(
                skipped
            )
        )

    definitions_file = os.path.join(base, "definitions-{}.json".format(search_name))
    with open(definitions_file, "w") as f:
        json.dump(model_names, f, indent=2)

    datasets_arg = ""
    if datasets is not None:
        datasets_arg = "--datasets {}".format(datasets)
    elif dataset is not None:
        datasets_arg = "--dataset {}".format(dataset)
    else:
        sys.stderr.write(
            "Provide either a path to multiple datasets or to a single dataset\n"
        )
        sys.exit(2)

    if cluster == "cox":
        max_time = "7-12:00"
    elif cluster == "holyseas":
        cluster = "holyseasgpu"
        max_time = "7-00:00"
    elif cluster == "seasdgx1":
        cluster = "seas_dgx1"
        max_time = "3-00:00"
    else:
        sys.stderr.write("Unknown cluster: {}\n".format(cluster))
        sys.exit(2)

    new_slurm_body = slurm_body.substitute(
        datasets=datasets_arg,
        settings=settings,
        definitions="definitions-{}.json".format(search_name),
        definition_idx="$SLURM_ARRAY_TASK_ID",
        epochs=epochs,
        batch_size=batch_size,
        peak_weight=peak_weight,
        signal_weighting=signal_weighting,
        signal_weighting_zero_point_percentage=signal_weighting_zero_point_percentage,
    )
    slurm = (
        slurm_header.replace("$num_definitions", str(len(model_names) - 1))
        .replace("$cluster", cluster)
        .replace("$time", max_time)
        + new_slurm_body
    )

    slurm_name = "train-{}.slurm".format(name) if name is not None else "train.slurm"
    slurm_file = os.path.join(base, slurm_name)

    with open(slurm_file, "w") as f:
        f.write(slurm)

    print(
        "Created slurm file for training {} neural networks".format(len(combinations))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Job Creator")
    parser.add_argument(
        "-n", "--neuralnets", help="path to the neural network search file", default=""
    )
    parser.add_argument(
        "-d", "--datasets", help="path to the datasets file", default="datasets.json"
    )
    parser.add_argument(
        "-s", "--settings", help="path to the settings file", default="settings.json"
    )
    parser.add_argument(
        "-x", "--cluster", help="cluster", default="cox", choices=["cox", "seas"]
    )
    parser.add_argument(
        "-c", "--clear", action="store_true", help="clears previously downloads"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="turn on verbose logging"
    )
    parser.add_argument(
        "-i", "--ignore-warns", action="store_true", help="ignore Keras warnings"
    )

    args = parser.parse_args()

    try:
        with open(args.datasets, "r") as f:
            json.load(f)
    except FileNotFoundError:
        sys.stderr.write("Please provide a datasets file via `--datasets`\n")
        sys.exit(2)

    try:
        with open(args.settings, "r") as f:
            json.load(f)
    except FileNotFoundError:
        sys.stderr.write("Please provide a settings file via `--settings`\n")
        sys.exit(2)

    if args.ignore_warns:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    jobs(
        args.neuralnets,
        args.datasets,
        args.settings,
        cluster=args.cluster,
        clear=args.clear,
        verbose=args.verbose,
    )
