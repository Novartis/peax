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
#SBATCH -p cox
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --mem=24000
#SBATCH --array=0-$num_definitions
#SBATCH -t 7-12:00
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
cd /n/pfister_lab/lekschas/peax/experiments/
python train.py \\
  --definitions $definitions \\
  --definition-idx $definition_idx \\
  --datasets $datasets \\
  --settings $settings \\
  --epochs $epochs \\
  --batch_size $batch_size \\
  --peak_weight $peak_weight \\
  --silent

# end of program
exit 0;
"""
)


def jobs(
    search,
    datasets: str,
    settings: str,
    epochs: int = 25,
    batch_size: int = 32,
    peak_weight: float = 1,
    base: str = ".",
    clear: bool = False,
    verbose: bool = False,
):
    tqdm = get_tqdm()

    # Create models and slurm directory
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)

    varying = search["hyperparameters"]["varying"]
    fixed = search["hyperparameters"]["fixed"]
    epochs = search["epochs"]
    batch_size = search["batch_size"]
    peak_weight = search["peak_weight"]

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

        return {
            "conv_filters": conv_filters,
            "conv_kernels": conv_kernels,
            "dense_units": prelim_def["dense_units"],
            "dropouts": dropouts,
            "embedding": prelim_def["embedding"],
            "reg_lambda": prelim_def["reg_lambda"],
            "optimizer": prelim_def["optimizer"],
            "learning_rate": prelim_def["learning_rate"],
            "learning_rate_decay": prelim_def["learning_rate_decay"],
            "loss": prelim_def["loss"],
            "metrics": prelim_def["metrics"],
        }

    model_names = []

    for combination in tqdm(combinations, desc="Jobs", unit="job"):
        combined_def = dict({}, **base_def)

        for i, value in enumerate(combination):
            combined_def[varying["params"][i]] = value

        final_def = finalize_def(combined_def)
        model_name = namify(final_def)
        def_file = os.path.join(base, "models", "{}.json".format(model_name))

        if not pathlib.Path(def_file).is_file() or clear:
            with open(def_file, "w") as f:
                json.dump(final_def, f, indent=2)
        else:
            print("Job file already exists. Use `--clear` to overwrite it.")

        model_names.append(model_name)

    definitions_file = os.path.join(base, "definitions.json")
    with open(definitions_file, "w") as f:
        json.dump(model_names, f, indent=2)

    new_slurm_body = slurm_body.substitute(
        datasets=datasets,
        settings=settings,
        definitions="definitions.json",
        definition_idx="$SLURM_ARRAY_TASK_ID",
        epochs=epochs,
        batch_size=batch_size,
        peak_weight=peak_weight,
    )
    slurm = (
        slurm_header.replace("$num_definitions", str(len(model_names) - 1))
        + new_slurm_body
    )

    slurm_file = os.path.join(base, "train.slurm")

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
        with open(args.neuralnets, "r") as f:
            search = json.load(f)
    except FileNotFoundError:
        print("Please provide a neural network search file via `--neuralnets`")
        sys.exit(2)

    try:
        with open(args.datasets, "r") as f:
            json.load(f)
    except FileNotFoundError:
        print("Please provide a datasets file via `--datasets`")
        sys.exit(2)

    try:
        with open(args.settings, "r") as f:
            json.load(f)
    except FileNotFoundError:
        print("Please provide a settings file via `--settings`")
        sys.exit(2)

    if args.ignore_warns:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    jobs(search, args.datasets, args.settings, clear=args.clear, verbose=args.verbose)
