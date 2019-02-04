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
#SBATCH -p holyseasgpu
#SBATCH --gres=gpu
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --mem=24000
#SBATCH -t 7-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu
#SBATCH -o /n/pfister_lab/lekschas/peax/experiments/logs/out-$name.txt
#SBATCH -e /n/pfister_lab/lekschas/peax/experiments/logs/err-$name.txt

# add additional commands needed for Lmod and module loads here
source new-modules.sh
#module load gcc/4.8.2-fasrc01 python/2.7.9-fasrc01
module load Anaconda/2.1.0-fasrc01
#module load cuda/7.5-fasrc01

"""

slurm_body = Template(
    """
# add commands for analyses here
cd /n/pfister_lab/lekschas/peax/experiments/
python train.py --definition $definition --epochs $epochs --batch_size $batch_size --peak_weight $peak_weight

# end of program
exit 0;
"""
)


def jobs(
    search,
    settings,
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
    pathlib.Path("slurm").mkdir(parents=True, exist_ok=True)

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

    for combination in tqdm(combinations, desc="Jobs", unit="job"):
        combined_def = dict({}, **base_def)

        for i, value in enumerate(combination):
            combined_def[varying["params"][i]] = value

        final_def = finalize_def(combined_def)
        model_name = namify(final_def)

        new_slurm_body = slurm_body.substitute(
            definition="{}.json".format(model_name),
            epochs=epochs,
            batch_size=batch_size,
            peak_weight=peak_weight,
        )
        slurm = slurm_header.replace("$name", model_name) + new_slurm_body

        slurm_file = os.path.join(base, "slurm", "{}.slurm".format(model_name))
        def_file = os.path.join(base, "slurm", "{}.json".format(model_name))

        if not pathlib.Path(slurm_file).is_file() or clear:
            with open(slurm_file, "w") as f:
                f.write(slurm)
        else:
            print("Job file already exists. Use `--clear` to overwrite it.")

        if not pathlib.Path(def_file).is_file() or clear:
            with open(def_file, "w") as f:
                json.dump(final_def, f, indent=2)
        else:
            print("Job file already exists. Use `--clear` to overwrite it.")

    print(
        "Created slurm files for training {} neural networks".format(len(combinations))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Job Creator")
    parser.add_argument(
        "-n", "--neuralnets", help="path to the neural network search file", default=""
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
        with open(args.settings, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Please provide a settings file via `--settings`")
        sys.exit(2)

    if args.ignore_warns:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    jobs(search, settings, clear=args.clear, verbose=args.verbose)
