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

import tensorflow as tf

# Stupid Keras things is a smart way to always print. See:
# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
from keras.metrics import mse, mae, binary_crossentropy
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = (
    True
)  # to log device placement (on which device the operation ran)
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

sys.stderr = stderr

from ae.metrics import dtw_metric, r2
from ae.utils import get_tqdm, evaluate_model, plot_windows
from ae.loss import scaled_mean_squared_error, scaled_logcosh, scaled_huber


def evaluate(
    model_name,
    datasets: dict = None,
    dataset_name: str = None,
    base: str = ".",
    clear: bool = False,
    silent: bool = False,
    incl_dtw: bool = False,
):
    # Create data directory
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)

    tqdm = get_tqdm()

    postfix = "-{}".format(dataset_name) if dataset_name else ""

    encoder_filepath = os.path.join(
        base, "models", "{}---encoder{}.h5".format(model_name, postfix)
    )
    decoder_filepath = os.path.join(
        base, "models", "{}---decoder{}.h5".format(model_name, postfix)
    )
    evaluation_filepath = os.path.join(
        base, "models", "{}---evaluation{}.h5".format(model_name, postfix)
    )
    total_loss_plot_filepath = os.path.join(
        base, "models", "{}---total-loss{}.png".format(model_name, postfix)
    )
    predictions_filepath = os.path.join(
        base, "models", "{}---predictions{}.png".format(model_name, postfix)
    )

    if (
        not pathlib.Path(encoder_filepath).is_file()
        or not pathlib.Path(decoder_filepath).is_file()
    ):
        sys.stderr.write("Encode and decoder need to be available\n")
        return

    if pathlib.Path(evaluation_filepath).is_file() and not clear:
        print("Model is already evaluated. Overwrite with `clear`")
        return

    dtw = dtw_metric()

    encoder = load_model(encoder_filepath)
    decoder = load_model(decoder_filepath)

    if dataset_name is not None:
        datasets = {dataset_name: True}

    num_datasets = len(datasets)

    keras_metrics = {
        "mse": mse,
        "smse-2": scaled_mean_squared_error(2.0),
        "smse-3": scaled_mean_squared_error(3.0),
        "smse-5": scaled_mean_squared_error(5.0),
        "smse-10": scaled_mean_squared_error(10.0),
        "r2": r2,
        "shuber-10-5": scaled_huber(10.0, 5.0),
        "slogcosh-10": scaled_logcosh(10.0),
        "mae": mae,
        "bce": binary_crossentropy,
    }
    numpy_metrics = {}

    if incl_dtw:
        numpy_metrics["dtw"] = dtw

    total_loss = None

    datasets_iter = (
        datasets if silent else tqdm(datasets, desc="Datasets", unit="dataset")
    )

    first_dataset_name = None

    for dataset_name in datasets_iter:
        data_filename = "{}.h5".format(dataset_name)
        data_filepath = os.path.join(base, "data", data_filename)

        if first_dataset_name is None:
            first_dataset_name = dataset_name

        with h5py.File(data_filepath, "r") as f:
            data_test = f["data_test"][:]

            loss, _ = evaluate_model(
                encoder,
                decoder,
                data_test,
                keras_metrics=list(keras_metrics.values()),
                numpy_metrics=list(numpy_metrics.values()),
            )
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = np.vstack((total_loss, loss))

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
        f.create_dataset("plotted_window_indices", data=window_idx)
        f.create_dataset("plotted_window_total_signal", data=total_signal)
        f.create_dataset("plotted_window_max_signal", data=max_signal)

    return total_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Testor")
    parser.add_argument("-m", "--model-name", help="name of the model")
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
        args.model_name,
        datasets=datasets,
        dataset_name=args.dataset,
        clear=args.clear,
        silent=args.silent,
        incl_dtw=args.incl_dtw,
    )
