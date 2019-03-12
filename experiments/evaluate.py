#!/usr/bin/env python

import argparse
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import seaborn as sns
import sys
import warnings

from string import Template

# Stupid Keras things is a smart way to always print. See:
# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
from keras.models import load_model

sys.stderr = stderr

from ae.metrics import dtw_metric, r2_min_numpy
from ae.utils import get_tqdm, evaluate_model, plot_windows
from ae.loss import scaled_mean_squared_error, binary_crossentropy_numpy


slurm_header = """#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p $cluster
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --mem=12000
#SBATCH --array=0-$num_definitions
#SBATCH -t 1-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lekschas@g.harvard.edu
#SBATCH -o /n/pfister_lab/lekschas/peax/experiments/logs/evaluate-out-%A-%a.txt
#SBATCH -e /n/pfister_lab/lekschas/peax/experiments/logs/evaluate-err-%A-%a.txt

# add additional commands needed for Lmod and module loads here
source new-modules.sh
module load Anaconda/5.0.1-fasrc01
"""

slurm_body = Template(
    """
# add commands for analyses here
cd /n/pfister_lab/haehn/Projects/peax/experiments/
source activate /n/pfister_lab/haehn/ENVS/peax
python evaluate.py \\
  --model-names $model_names \\
  --model-name-idx $model_name_idx \\
  $datasets \\
  $clear \\
  $incl_dtw \\
  --silent

# end of program
exit 0;
"""
)


def evaluate(
    model_name: str = None,
    model_names: str = None,
    model_name_idx: int = -1,
    datasets: dict = None,
    dataset_name: str = None,
    base: str = ".",
    clear: bool = False,
    silent: bool = False,
    verbose: bool = False,
    incl_dtw: bool = False,
):
    # Create data directory
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)

    tqdm = get_tqdm()

    postfix = "-{}".format(dataset_name) if dataset_name else ""

    print("Evaluate(verbose={})...".format("True" if verbose else "False"))

    if model_name is not None:
        pass
    elif model_names is not None and model_name_idx >= 0:
        try:
            with open(os.path.join(base, model_names), "r") as f:
                model_names = json.load(f)
        except FileNotFoundError:
            sys.stderr.write(
                "Model names file not found: {}\n".format(
                    os.path.join(base, model_names)
                )
            )
            sys.exit(2)

        try:
            model_name = model_names[model_name_idx]
        except IndexError:
            sys.stderr.write("Model name not available: #{}\n".format(model_name_idx))
            sys.exit(2)
    else:
        sys.stderr.write(
            "Either provide a model name or the name of the file with all model names and an index\n"
        )
        sys.exit(2)

    repetition = None
    if len(model_name.split("__")) > 1:
        parts = model_name.split("__")
        model_name = parts[0]
        repetition = parts[1]
        postfix = "{}__{}".format(postfix, repetition)

    autoencoder_filepath = os.path.join(
        base, "models", "{}---autoencoder{}.h5".format(model_name, postfix)
    )
    training_filepath = os.path.join(
        base, "models", "{}---training{}.h5".format(model_name, postfix)
    )
    evaluation_filepath = os.path.join(
        base, "models", "{}---evaluation{}.h5".format(model_name, postfix)
    )
    total_loss_plot_filepath = os.path.join(
        base, "models", "{}---test-loss{}.png".format(model_name, postfix)
    )
    predictions_filepath = os.path.join(
        "models", "{}---predictions{}.png".format(model_name, postfix)
    )

    if not pathlib.Path(autoencoder_filepath).is_file():
        sys.stderr.write("Autoencoder needs to be available\n")
        return

    if pathlib.Path(evaluation_filepath).is_file() and not clear:
        print("Model is already evaluated. Overwrite with `clear`")
        return

    dtw = dtw_metric()

    if silent:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            autoencoder = load_model(autoencoder_filepath)
    else:
        autoencoder = load_model(autoencoder_filepath)

    if dataset_name is not None:
        datasets = {dataset_name: True}

    num_datasets = len(datasets)

    keras_metrics = {
        # "smse-10": scaled_mean_squared_error(10.0),
        # "r2": r2_min,
        # "shuber-10-5": scaled_huber(10.0, 5.0),
        # "slogcosh-10": scaled_logcosh(10.0),
        # "mae": mae,
        # "bce": binary_crossentropy
    }
    numpy_metrics = {
        "smse-10": scaled_mean_squared_error(10.0, True),
        "r2": r2_min_numpy,
        "bce": binary_crossentropy_numpy,
    }

    if incl_dtw:
        numpy_metrics["dtw"] = dtw

    total_loss = None
    total_times = np.zeros((num_datasets, 4))

    datasets_iter = (
        datasets if silent else tqdm(datasets, desc="Datasets", unit="dataset")
    )

    first_dataset_name = None

    i = 0
    for dataset_name in datasets_iter:
        data_filepath = os.path.join(base, "data", "{}.h5".format(dataset_name))

        if first_dataset_name is None:
            first_dataset_name = dataset_name

        with h5py.File(data_filepath, "r") as f:
            data_test = f["data_test"]

            if verbose:
                print("Start evaluate {}".format(dataset_name))

            loss = evaluate_model(
                autoencoder,
                data_test,
                keras_metrics=list(keras_metrics.values()),
                numpy_metrics=list(numpy_metrics.values()),
                verbose=verbose,
            )
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = np.vstack((total_loss, loss))

        with h5py.File(training_filepath, "r") as f:
            times = f["times"][:]

            total_times[i, 0] = np.mean(times)
            total_times[i, 1] = np.median(times)
            total_times[i, 2] = np.min(times)
            total_times[i, 3] = np.max(times)

        i += 1

    if verbose:
        print("Plot windows...")

    # Only plot windows for the first dataset
    window_idx, total_signal, max_signal = plot_windows(
        dataset_name,
        model_name,
        trained_on_single_dataset=num_datasets == 1,
        ds_type="test",
        num=60,
        min_signal=5,
        base=base,
        save_as=predictions_filepath,
        silent=silent,
        repetition=repetition,
    )

    # Plot and save an overview of the total losses
    sns.set(style="whitegrid")

    df = pd.DataFrame(np.mean(total_loss, axis=0).reshape(1, total_loss.shape[1]))
    df.columns = list(keras_metrics.keys()) + list(numpy_metrics.keys())

    fig, _ = plt.subplots(figsize=(1.25 * total_loss.shape[1], 8))
    plot = sns.barplot(data=df)
    plot.set(xlabel="Metrics", ylabel="Total loss")
    fig.savefig(total_loss_plot_filepath, bbox_inches="tight")

    with h5py.File(evaluation_filepath, "w") as f:
        f.create_dataset("total_loss", data=total_loss)
        f.create_dataset("total_loss_metrics", data=",".join(df.columns))
        f.create_dataset("total_times", data=total_times)
        f.create_dataset("plotted_window_indices", data=window_idx)
        f.create_dataset("plotted_window_total_signal", data=total_signal)
        f.create_dataset("plotted_window_max_signal", data=max_signal)

    return total_loss


def create_jobs(
    search_name: str,
    name: str = None,
    datasets: str = None,
    dataset: str = None,
    cluster: str = "cox",
    base: str = ".",
    clear: bool = False,
    incl_dtw: bool = False,
):
    if cluster == "cox":
        pass
    elif cluster == "holyseas":
        cluster = "holyseasgpu"
    elif cluster == "seasdgx1":
        cluster = "seas_dgx1"
    else:
        sys.stderr.write("Unknown cluster: {}\n".format(cluster))
        sys.exit(2)

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

    try:
        with open(
            os.path.join(base, "definitions-{}.json".format(search_name)), "r"
        ) as f:
            model_names = json.load(f)
    except FileNotFoundError:
        sys.stderr.write(
            "Model names file not found: {}\n".format(os.path.join(base, model_names))
        )
        sys.exit(2)

    new_slurm_body = slurm_body.substitute(
        datasets=datasets_arg,
        model_names="definitions-{}.json".format(search_name),
        model_name_idx="$SLURM_ARRAY_TASK_ID",
        clear="--clear" if clear else "",
        incl_dtw="--incl-dtw" if incl_dtw else "",
    )
    slurm = (
        slurm_header.replace("$num_definitions", str(len(model_names) - 1)).replace(
            "$cluster", cluster
        )
        + new_slurm_body
    )

    slurm_name = (
        "evaluate-{}.slurm".format(name) if name is not None else "evaluate.slurm"
    )
    slurm_file = os.path.join(base, slurm_name)

    with open(slurm_file, "w") as f:
        f.write(slurm)

    print(
        "Created slurm file for evaluating {} neural networks".format(len(model_names))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Testor")
    parser.add_argument("-m", "--model-name", help="name of the model")
    parser.add_argument(
        "-n", "--model-names", help="path to the CAE model names file", default=""
    )
    parser.add_argument(
        "-x",
        "--model-name-idx",
        help="index of the CAE model to be evaluated",
        type=int,
        default=-1,
    )
    parser.add_argument("-d", "--datasets", help="path to the datasets file", type=str)
    parser.add_argument(
        "-o", "--dataset", help="name of a single dataset file", type=str
    )
    parser.add_argument(
        "-c", "--clear", action="store_true", help="clears previously downloadeds"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="turn on verbose logging"
    )
    parser.add_argument("-z", "--silent", action="store_true", help="turn logging off")
    parser.add_argument(
        "--incl-dtw",
        action="store_true",
        help="include DTW as a metric (WARNING this is dead slow!)",
    )

    args = parser.parse_args()

    datasets = None

    if args.dataset is not None:
        pass
    elif args.datasets is not None:
        try:
            with open(args.datasets, "r") as f:
                datasets = json.load(f)
        except FileNotFoundError:
            sys.stderr.write("You need to provide a datasets file via `--datasets`\n")
            sys.exit(2)
    else:
        sys.stderr.write(
            "You need to either provide a datasets file (with `-d`) or the name of a single dataset (with `-o`)\n"
        )
        sys.exit(2)

    if args.silent:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    evaluate(
        model_name=args.model_name,
        model_names=args.model_names,
        model_name_idx=args.model_name_idx,
        datasets=datasets,
        dataset_name=args.dataset,
        clear=args.clear,
        verbose=args.verbose,
        silent=args.silent,
        incl_dtw=args.incl_dtw,
    )
