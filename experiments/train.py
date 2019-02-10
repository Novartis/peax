#!/usr/bin/env python

import argparse
import h5py
import json
import numpy as np
import os
import pathlib
import sys

from ae.cnn import create_model
from ae.utils import namify, get_tqdm

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


def get_definition(
    definition: dict = None,
    definitions: list = None,
    definition_idx: int = -1,
    base: str = ".",
) -> dict:
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

    return definition


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
):
    # Create data directory
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)

    tqdm_keras = get_tqdm(is_keras=True)

    bins_per_window = settings["window_size"] // settings["resolution"]

    definition = get_definition(definition, definitions, definition_idx, base)

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
            data_dev = f["data_dev"][:]
            peaks_train = f["peaks_train"][:]
        shuffle = True

    if signal_weighting in signal_weightings and not train_on_hdf5:
        zero_point = data_train.shape[1] * signal_weighting_zero_point_percentage - 1
        total_signal = np.sum(data_train, axis=1).squeeze()
        signal_weight = signal_weightings[signal_weighting](total_signal, zero_point)

    no_peak_ratio = (data_train.shape[0] - np.sum(peaks_train)) / np.sum(peaks_train)
    # There are `no_peak_ratio` times more no peak samples. To equalize the
    # importance of samples we give samples that contain a peak more weight
    # but we never downweight peak windows!
    sample_weight = (
        # Equal weights if there are more windows without peaks (i.e., increase weight
        # of windows with peaks)
        (peaks_train * np.max((0, no_peak_ratio - 1)))
        # Ensure that all windows have a base weight of 1
        + 1
        # Additionally adjust the weight of windows with a peak
        + (peaks_train * peak_weight - peaks_train)
        # Finally, add signal-dependent weights
        + signal_weight
    )

    if silent:
        callbacks = []
    else:
        callbacks = [tqdm_keras(show_outer=False)]

    history = autoencoder.fit(
        data_train,
        data_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        validation_data=(data_dev, data_dev),
        sample_weight=sample_weight,
        verbose=0,
        callbacks=callbacks,
    )

    try:
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
    except KeyError:
        pass

    encoder.save(
        os.path.join(base, "models", "{}---encoder-{}.h5".format(model_name, dataset))
    )
    decoder.save(
        os.path.join(base, "models", "{}---decoder-{}.h5".format(model_name, dataset))
    )

    with h5py.File(
        os.path.join(base, "models", "{}---training-{}.h5".format(model_name, dataset)),
        "w",
    ) as f:
        f.create_dataset("loss", data=loss)
        f.create_dataset("val_loss", data=val_loss)


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

    definition = get_definition(definition, definitions, definition_idx, base)

    model_name = namify(definition)
    encoder_name = os.path.join(base, "models", "{}---encoder.h5".format(model_name))
    decoder_name = os.path.join(base, "models", "{}---decoder.h5".format(model_name))

    if (
        pathlib.Path(encoder_name).is_file() or pathlib.Path(decoder_name).is_file()
    ) and not clear:
        print("Encoder/decoder already exists. Use `--clear` to overwrite it.")
        return

    encoder, decoder, autoencoder = create_model(bins_per_window, **definition)

    loss = np.zeros(epochs * len(datasets))
    val_loss = np.zeros(epochs * len(datasets))

    if silent:
        epochs_iter = range(epochs)
    else:
        epochs_iter = tqdm_normal(range(epochs), desc="Epochs", unit="epoch")

    i = 0
    for epoch in epochs_iter:
        epochs_iter = range(epochs)

        if silent:
            datasets_iter = datasets
        else:
            datasets_iter = tqdm_normal(
                datasets, desc="Datasets", unit="dataset", leave=False
            )

        for dataset_name in datasets_iter:
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

                if signal_weighting in signal_weightings:
                    zero_point = (
                        data_train.shape[1] * signal_weighting_zero_point_percentage - 1
                    )
                    total_signal = np.sum(data_train, axis=1).squeeze()
                    signal_weight = signal_weightings[signal_weighting](
                        total_signal, zero_point
                    )

                no_peak_ratio = (data_train.shape[0] - np.sum(peaks_train)) / np.sum(
                    peaks_train
                )
                # There are `no_peak_ratio` times more no peak samples. To equalize the
                # importance of samples we give samples that contain a peak more weight
                # but we never downweight peak windows!
                sample_weight = (
                    # Equal weights if there are more windows without peaks (i.e., increase weight
                    # of windows with peaks)
                    (peaks_train * np.max((0, no_peak_ratio - 1)))
                    # Ensure that all windows have a base weight of 1
                    + 1
                    # Additionally adjust the weight of windows with a peak
                    + (peaks_train * peak_weight - peaks_train)
                    # Finally, add signal-dependent weights
                    + signal_weight
                )

                if silent:
                    callbacks = []
                else:
                    callbacks = [tqdm_keras(show_outer=False)]

                history = autoencoder.fit(
                    data_train,
                    data_train,
                    epochs=1,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(data_dev, data_dev),
                    sample_weight=sample_weight,
                    verbose=0,
                    callbacks=callbacks,
                )

                try:
                    loss[i] = history.history["loss"][0]
                    val_loss[i] = history.history["val_loss"][0]
                except KeyError:
                    pass

                i += 1

    encoder.save(os.path.join(base, "models", "{}---encoder.h5".format(model_name)))
    decoder.save(os.path.join(base, "models", "{}---decoder.h5".format(model_name)))

    with h5py.File(
        os.path.join(base, "models", "{}---training.h5".format(model_name)), "w"
    ) as f:
        f.create_dataset("loss", data=loss)
        f.create_dataset("val_loss", data=val_loss)


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
        )
