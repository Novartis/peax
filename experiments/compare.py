#!/usr/bin/env python

import argparse
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys


def compare(
    model_names_file: str,
    dataset_name: str = None,
    num_models: int = 10,
    remove_common_prefix_from_df: bool = False,
    base: str = ".",
    clear: bool = False,
    verbose: bool = False,
    silent: bool = False,
):
    postfix = "-{}".format(dataset_name) if dataset_name is not None else ""

    total_loss = None

    model_names_filename = os.path.splitext(model_names_file)[0]

    if model_names_filename.startswith("definitions-"):
        model_names_filename = model_names_filename[len("definitions-") :]

    try:
        with open(os.path.join(base, model_names_file), "r") as f:
            model_names = json.load(f)
    except FileNotFoundError:
        sys.stderr.write("You need to provide a model names file\n")
        sys.exit(2)

    columns = None

    for model_name in model_names:
        filepath = os.path.join(
            base, "models", "{}---evaluation{}.h5".format(model_name, postfix)
        )

        with h5py.File(filepath, "r") as f:
            loss = f["total_loss"][:]
            if columns is None:
                columns = f["total_loss_metrics"][()].split(",")
            loss = np.mean(loss, axis=0).reshape(1, loss.shape[1])

            if total_loss is None:
                total_loss = loss
            else:
                total_loss = np.vstack((total_loss, loss))

    sorting = np.argsort(total_loss, axis=0)

    ranks = np.zeros(sorting.shape)

    # Transpose ordering and indices such that each row (one CNN model) holds the rank
    # in respect to each metric. E.g.:
    # CNN-1 [0 1 1 2]
    # CNN-2 [1 0 4 3]
    # CNN-3 [0 2 0 0]
    # >> Here CNN-3 ranked first for the first, third, and forth metric and ranks second
    # for the second metric.
    # We can use this now to easily compute the Borda count from this matrix using:
    # np.sum(x, axis=1)
    for c in np.arange(sorting.shape[1]):
        ranks[:, c][sorting[:, c]] = np.arange(sorting.shape[0])

    # Merge sortings using Borda count
    cum_sorting = np.sum(ranks, axis=1)
    ordered_models = np.argsort(cum_sorting)

    sns.set(style="whitegrid")

    num_models = np.max((total_loss.shape[0], num_models))

    fig, axes = plt.subplots(num_models, 1, figsize=(10, 3 * num_models), sharex=True)
    for i, ax in enumerate(axes):
        # sns.barplot(data=df.iloc(i), ax=ax)
        ax.bar(columns, total_loss[ordered_models][i])
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Total loss")
        ax.set_ylim(0, np.max(total_loss[ordered_models][0:num_models]) * 1.05)
        ax.set_title(np.array(model_names)[ordered_models][i])

    fig.savefig(
        os.path.join(base, "comparison-{}.png".format(model_names_filename)),
        bbox_inches="tight",
    )

    prefix = os.path.commonprefix(model_names)

    if remove_common_prefix_from_df and len(model_names) > 1:
        for i, model_name in enumerate(model_names):
            model_names[i] = model_name[len(prefix) :]

    return pd.DataFrame(total_loss, index=model_names, columns=columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Comparer")
    parser.add_argument(
        "-m",
        "--model-names",
        help="path to file containing model names to compare",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dataset-name",
        help="name of the dataset the model was trained on",
        type=str,
    )
    parser.add_argument(
        "-n", "--num-models", help="number of models to print", default=10, type=int
    )
    parser.add_argument(
        "--no-common-prefix",
        action="store_true",
        help="remove common prefix from the DataFrame",
    )
    parser.add_argument(
        "-c", "--clear", action="store_true", help="clear previously found datasets"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="turn on verbose logging"
    )
    parser.add_argument("-z", "--silent", action="store_true", help="turn off logging")

    args = parser.parse_args()

    compare(
        args.model_names,
        dataset_name=args.dataset_name,
        num_models=args.num_models,
        remove_common_prefix_from_df=args.no_common_prefix,
        clear=args.clear,
        verbose=args.verbose,
        silent=args.silent,
    )
