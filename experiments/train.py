#!/usr/bin/env python

import argparse
import h5py
import json
import numpy as np
import pandas as pd
import os
import pathlib
import seaborn as sns
import sys
import time

from ae.cnn import create_model
from ae.utils import namify, get_tqdm

from keras.callbacks import Callback, EarlyStopping
from keras.utils.io_utils import HDF5Matrix

# zp = zero point
# This is the point (total signal) at which we're going to increase the weights for
# windows. The weight of windows with less signal will simply not be increased
signal_weightings = {
    "linear": lambda x, zp: (x - zp).clip(min=0),
    "log2": lambda x, zp: np.log2((x - zp).clip(min=1)),
    "logn": lambda x, zp: np.log((x - zp).clip(min=1)),
    "log10": lambda x, zp: np.log10((x - zp).clip(min=1)),
}


class TimeHistory(Callback):
    def __init__(self):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def get_definition(
    definition: dict = None,
    definitions: list = None,
    definition_idx: int = -1,
    base: str = ".",
) -> dict:
    definition_name = None
    if definition is not None:
        pass
    elif definitions is not None and definition_idx >= 0:
        try:
            definition_name = definitions[definition_idx]
        except IndexError:
            sys.stderr.write("Definition not available: #{}\n".format(definition_idx))
            sys.exit(2)

        try:
            definition_filepath = os.path.join(
                base, "models", "{}.json".format(definition_name)
            )
            with open(definition_filepath, "r") as f:
                definition = json.load(f)
        except FileNotFoundError:
            sys.stderr.write("Definition not found: {}\n".format(definition_filepath))
            sys.exit(2)
    else:
        sys.stderr.write(
            "Either provide a definition or a list of definitions together with a definition index\n"
        )
        sys.exit(2)

    return definition, definition_name


def get_sample_weights(
    data: np.ndarray,
    peaks: np.ndarray,
    signal_weighting: str,
    signal_weighting_zero_point_percentage: float = 0.02,
    peak_weight: float = 1,
    train_on_hdf5: bool = False,
) -> np.ndarray:
    signal_weight = np.zeros(data.shape[0])

    if signal_weighting in signal_weightings and not train_on_hdf5:
        zero_point = data.shape[1] * signal_weighting_zero_point_percentage - 1
        total_signal = np.sum(data, axis=1).squeeze()
        signal_weight = signal_weightings[signal_weighting](total_signal, zero_point)

    no_peak_ratio = (data.shape[0] - np.sum(peaks)) / np.sum(peaks)

    # There are `no_peak_ratio` times more no peak samples. To equalize the
    # importance of samples we give samples that contain a peak more weight
    # but we never downweight peak windows!
    return (
        # Equal weights if there are more windows without peaks (i.e., increase weight
        # of windows with peaks)
        (peaks * np.max((0, no_peak_ratio - 1)))
        # Ensure that all windows have a base weight of 1
        + 1
        # Additionally adjust the weight of windows with a peak
        + (peaks * peak_weight - peaks)
        # Finally, add signal-dependent weights
        + signal_weight
    )


def plot_loss_to_file(
    loss,
    val_loss,
    epochs: int,
    model_name: str,
    dataset_name: str = None,
    base: str = ".",
):
    prefix = "-{}".format(dataset_name) if dataset_name is not None else ""
    filepath = os.path.join(
        base, "models", "{}---train-loss{}.png".format(model_name, prefix)
    )

    # Print loss for fast evaluation
    data = np.zeros((epochs, 2))
    data[:, 0] = loss
    data[:, 1] = val_loss

    df = pd.DataFrame(data)
    df.columns = ["loss", "val_loss"]

    ax = sns.lineplot(data=df)
    ax.set_xticks(np.arange(epochs))
    ax.set(xlabel="epochs", ylabel="loss")
    ax.get_figure().savefig(filepath, bbox_inches="tight")


def train_on_single_dataset(
    settings: dict,
    dataset: str,
    definition: dict = None,
    definitions: list = None,
    definition_idx: int = -1,
    epochs: int = 25,
    batch_size: int = 32,
    peak_weight: float = 1,
    signal_weighting: str = None,
    signal_weighting_zero_point_percentage: float = 0.02,
    base: str = ".",
    clear: bool = False,
    silent: bool = False,
    train_on_hdf5: bool = False,
    early_stopping: bool = False,
):
    # Create data directory
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)

    tqdm_keras = get_tqdm(is_keras=True)

    bins_per_window = settings["window_size"] // settings["resolution"]

    definition, definition_name = get_definition(
        definition, definitions, definition_idx, base
    )

    repetition = None
    if definition_name is not None and len(definition_name.split("__")) > 1:
        repetition = definition_name.split("__")[1]

    model_name = namify(definition)
    encoder_name = os.path.join(
        base, "models", "{}---encoder-{}.h5".format(model_name, dataset)
    )
    decoder_name = os.path.join(
        base, "models", "{}---decoder-{}.h5".format(model_name, dataset)
    )

    if (
        pathlib.Path(encoder_name).is_file() or pathlib.Path(decoder_name).is_file()
    ) and not clear:
        print("Encoder/decoder already exists. Use `--clear` to overwrite it.")
        return

    peak_weight = (
        definition["peak_weight"] if "peak_weight" in definition else peak_weight
    )
    signal_weighting = (
        definition["signal_weighting"]
        if "signal_weighting" in definition
        else signal_weighting
    )
    signal_weighting_zero_point_percentage = (
        definition["signal_weighting_zero_point_percentage"]
        if "signal_weighting_zero_point_percentage" in definition
        else signal_weighting_zero_point_percentage
    )

    definition.pop("peak_weight", None)
    definition.pop("signal_weighting", None)
    definition.pop("signal_weighting_zero_point_percentage", None)

    encoder, decoder, autoencoder = create_model(bins_per_window, **definition)

    loss = None
    val_loss = None

    data_filename = "{}.h5".format(dataset)
    data_filepath = os.path.join(base, "data", data_filename)

    if not pathlib.Path(data_filepath).is_file():
        sys.stderr.write("Dataset not found: {}\n".format(data_filepath))
        sys.exit(2)

    if train_on_hdf5:
        data_train = HDF5Matrix(data_filepath, "data_train")
        data_dev = HDF5Matrix(data_filepath, "data_dev")
        peaks_train = HDF5Matrix(data_filepath, "peaks_train")
        shuffle = "batch"
    else:
        with h5py.File(data_filepath, "r") as f:
            data_train = f["data_train"][:]
            peaks_train = f["peaks_train"][:]
            data_dev = f["data_dev"][:]
            peaks_dev = f["peaks_dev"][:]
        shuffle = True

    sample_weight_train = get_sample_weights(
        data_train,
        peaks_train,
        signal_weighting,
        signal_weighting_zero_point_percentage,
        peak_weight,
        train_on_hdf5,
    )
    sample_weight_dev = get_sample_weights(
        data_dev,
        peaks_dev,
        signal_weighting,
        signal_weighting_zero_point_percentage,
        peak_weight,
        train_on_hdf5,
    )

    times_history = TimeHistory()

    callbacks = [times_history]

    if early_stopping:
        callbacks += [
            EarlyStopping(
                monitor="val_loss",
                min_delta=1e-6,
                patience=10,
                restore_best_weights=True,
            )
        ]

    if not silent:
        callbacks += [tqdm_keras(leave_inner=True)]

    history = autoencoder.fit(
        data_train,
        data_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        validation_data=(data_dev, data_dev, sample_weight_dev),
        sample_weight=sample_weight_train,
        verbose=0,
        callbacks=callbacks,
    )

    try:
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        times = times_history.times
    except KeyError:
        pass

    postfix = "__{}".format(repetition) if repetition is not None else ""

    encoder.save(
        os.path.join(
            base, "models", "{}---encoder-{}{}.h5".format(model_name, dataset, postfix)
        )
    )
    decoder.save(
        os.path.join(
            base, "models", "{}---decoder-{}{}.h5".format(model_name, dataset, postfix)
        )
    )

    with h5py.File(
        os.path.join(
            base, "models", "{}---training-{}{}.h5".format(model_name, dataset, postfix)
        ),
        "w",
    ) as f:
        f.create_dataset("loss", data=loss)
        f.create_dataset("val_loss", data=val_loss)
        f.create_dataset("times", data=times)

    plot_loss_to_file(
        loss, val_loss, epochs, model_name, dataset_name=dataset, base=base
    )


def train(
    settings: dict,
    datasets: dict,
    definition: dict = None,
    definitions: list = None,
    definition_idx: int = -1,
    epochs: int = 25,
    batch_size: int = 512,
    peak_weight: float = 1,
    signal_weighting: str = None,
    signal_weighting_zero_point_percentage: float = 0.02,
    base: str = ".",
    clear: bool = False,
    silent: bool = False,
):
    # Create data directory
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)

    tqdm_normal = get_tqdm()
    tqdm_keras = get_tqdm(is_keras=True)

    bins_per_window = settings["window_size"] // settings["resolution"]

    definition, definition_name = get_definition(
        definition, definitions, definition_idx, base
    )

    repetition = None
    if len(definition_name.split("__")) > 1:
        repetition = definition_name.split("__")[1]

    model_name = namify(definition)
    encoder_name = os.path.join(base, "models", "{}---encoder.h5".format(model_name))
    decoder_name = os.path.join(base, "models", "{}---decoder.h5".format(model_name))

    if (
        pathlib.Path(encoder_name).is_file() or pathlib.Path(decoder_name).is_file()
    ) and not clear:
        print("Encoder/decoder already exists. Use `--clear` to overwrite it.")
        return

    peak_weight = (
        definition["peak_weight"] if "peak_weight" in definition else peak_weight
    )
    signal_weighting = (
        definition["signal_weighting"]
        if "signal_weighting" in definition
        else signal_weighting
    )
    signal_weighting_zero_point_percentage = (
        definition["signal_weighting_zero_point_percentage"]
        if "signal_weighting_zero_point_percentage" in definition
        else signal_weighting_zero_point_percentage
    )

    definition.pop("peak_weight", None)
    definition.pop("signal_weighting", None)
    definition.pop("signal_weighting_zero_point_percentage", None)

    encoder, decoder, autoencoder = create_model(bins_per_window, **definition)

    loss = np.zeros((epochs, len(datasets)))
    val_loss = np.zeros((epochs, len(datasets)))
    times = np.zeros((epochs, len(datasets)))

    if silent:
        epochs_iter = range(epochs)
    else:
        epochs_iter = tqdm_normal(range(epochs), desc="Epochs", unit="epoch")

    for e, epoch in enumerate(epochs_iter):
        epochs_iter = range(epochs)

        if silent:
            datasets_iter = datasets
        else:
            datasets_iter = tqdm_normal(
                datasets, desc="Datasets", unit="dataset", leave=False
            )

        for d, dataset_name in enumerate(datasets_iter):
            data_filename = "{}.h5".format(dataset_name)
            data_filepath = os.path.join(base, "data", data_filename)

            if not pathlib.Path(data_filepath).is_file():
                sys.stderr.write("Dataset not found: {}\n".format(data_filepath))
                sys.exit(2)

            with h5py.File(data_filepath, "r") as f:
                data_train = f["data_train"][:]
                data_dev = f["data_dev"][:]
                data_train = data_train.reshape(
                    data_train.shape[0], data_train.shape[1], 1
                )
                data_dev = data_dev.reshape(data_dev.shape[0], data_dev.shape[1], 1)
                peaks_train = f["peaks_train"][:]
                peaks_dev = f["peaks_dev"][:]

                sample_weight_train = get_sample_weights(
                    data_train,
                    peaks_train,
                    signal_weighting,
                    signal_weighting_zero_point_percentage,
                    peak_weight,
                )
                sample_weight_dev = get_sample_weights(
                    data_dev,
                    peaks_dev,
                    signal_weighting,
                    signal_weighting_zero_point_percentage,
                    peak_weight,
                )

                times_history = TimeHistory()

                callbacks = [times_history]

                if early_stopping:
                    callbacks += [
                        EarlyStopping(
                            monitor="val_loss",
                            min_delta=1e-6,
                            patience=10,
                            restore_best_weights=True,
                        )
                    ]

                if not silent:
                    callbacks += [tqdm_keras(leave_inner=True, leave_outer=False)]

                history = autoencoder.fit(
                    data_train,
                    data_train,
                    epochs=1,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(data_dev, data_dev, sample_weight_dev),
                    sample_weight=sample_weight_train,
                    verbose=0,
                    callbacks=callbacks,
                )

                try:
                    loss[e, d] = history.history["loss"][0]
                    val_loss[e, d] = history.history["val_loss"][0]
                    times[e, d] = times_history.times[0]
                except KeyError:
                    pass

    postfix = "__{}".format(repetition) if repetition is not None else ""

    encoder.save(
        os.path.join(base, "models", "{}---encoder{}.h5".format(model_name, postfix))
    )
    decoder.save(
        os.path.join(base, "models", "{}---decoder{}.h5".format(model_name, postfix))
    )

    with h5py.File(
        os.path.join(base, "models", "{}---training{}.h5".format(model_name, postfix)),
        "w",
    ) as f:
        f.create_dataset("loss", data=loss.mean(axis=1))
        f.create_dataset("val_loss", data=val_loss.mean(axis=1))
        f.create_dataset("times", data=times.mean(axis=1))
        f.create_dataset("loss_per_dataset_per_epoch", data=loss)
        f.create_dataset("val_loss_per_dataset_per_epoch", data=val_loss)
        f.create_dataset("times_per_dataset_per_epoch", data=times)

    plot_loss_to_file(
        loss.mean(axis=1), val_loss.mean(axis=1), epochs, model_name, base=base
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Trainer")
    parser.add_argument(
        "-n", "--definition", help="path to neural network definition file", type=str
    )
    parser.add_argument(
        "-d", "--datasets", help="path to datasets file", default="datasets.json"
    )
    parser.add_argument(
        "-o", "--dataset", help="path to a single dataset file", type=str
    )
    parser.add_argument(
        "-s", "--settings", help="path to settings file", default="settings.json"
    )
    parser.add_argument(
        "-N", "--definitions", help="path to neural network definitions file", type=str
    )
    parser.add_argument(
        "-x",
        "--definition-idx",
        help="index of a specific dataset to prepare",
        type=int,
        default=-1,
    )
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=25)
    parser.add_argument(
        "-b", "--batch_size", type=int, help="size of a batch", default=32
    )
    parser.add_argument("-w", "--peak-weight", type=int, help="peak weight", default=1)
    parser.add_argument("--signal-weighting", type=str, help="signal weighting")
    parser.add_argument(
        "--signal-weighting-zero-point-percentage",
        type=float,
        help="signal weighting zero-point percentage",
        default=0.02,
    )
    parser.add_argument(
        "-c", "--clear", action="store_true", help="clears previously downloads"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="turn on verbose logging"
    )
    parser.add_argument(
        "-z", "--silent", action="store_true", help="disable all but error logs"
    )
    parser.add_argument(
        "--early-stopping", action="store_true", help="employ early stopping"
    )

    args = parser.parse_args()

    definition = None
    definitions = None

    if args.definition is not None:
        try:
            with open(args.definition, "r") as f:
                definition = json.load(f)
        except FileNotFoundError:
            sys.stderr.write(
                "Please provide a neural network definition file via `--definition`\n"
            )
            sys.exit(2)
    elif args.definitions is not None and args.definition_idx >= 0:
        try:
            with open(args.definitions, "r") as f:
                definitions = json.load(f)
        except FileNotFoundError:
            sys.stderr.write(
                "Please provide a neural network definitions file via `--definitions`\n"
            )
            sys.exit(2)
    else:
        sys.stderr.write(
            "Either provide a definition file (via `-n`) or a file with all definitions (via `-N`) and a definition index (via `-x`)\n"
        )
        sys.exit(2)

    try:
        with open(args.settings, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        sys.stderr.write("Please provide a settings file via `--settings`\n")
        sys.exit(2)

    if args.dataset is None:
        try:
            with open(args.datasets, "r") as f:
                datasets = json.load(f)
        except FileNotFoundError:
            sys.stderr.write("Please provide a datasets file via `--datasets`\n")
            sys.exit(2)

    if args.silent:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    batch_size = args.batch_size
    epochs = args.epochs
    peak_weight = args.peak_weight
    signal_weighting = args.signal_weighting
    signal_weighting_zero_point_percentage = args.signal_weighting_zero_point_percentage

    if args.dataset:
        train_on_single_dataset(
            settings,
            args.dataset,
            definition=definition,
            definitions=definitions,
            definition_idx=args.definition_idx,
            epochs=epochs,
            batch_size=batch_size,
            peak_weight=peak_weight,
            signal_weighting=signal_weighting,
            signal_weighting_zero_point_percentage=signal_weighting_zero_point_percentage,
            clear=args.clear,
            silent=args.silent,
            early_stopping=args.early_stopping,
        )
    else:
        datasets = list(datasets.keys())

        if args.verbose:
            print(
                "Train neural networks for {} epochs on {} datasets".format(
                    epochs, len(datasets)
                )
            )

        train(
            settings,
            datasets,
            definition=definition,
            definitions=definitions,
            definition_idx=args.definition_idx,
            epochs=epochs,
            batch_size=batch_size,
            peak_weight=peak_weight,
            signal_weighting=signal_weighting,
            signal_weighting_zero_point_percentage=signal_weighting_zero_point_percentage,
            clear=args.clear,
            silent=args.silent,
            early_stopping=args.early_stopping,
        )
