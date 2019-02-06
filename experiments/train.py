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


def train(
    definition,
    settings,
    datasets,
    epochs: int = 25,
    batch_size: int = 32,
    peak_weight: int = 1,
    base: str = ".",
    clear: bool = False,
):
    # Create data directory
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    pathlib.Path("logs").mkdir(parents=True, exist_ok=True)
    tqdm_normal = get_tqdm()
    tqdm_keras = get_tqdm(is_keras=True)

    bins_per_window = settings["window_size"] // settings["resolution"]

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

    i = 0
    for epoch in tqdm_normal(range(epochs), desc="Epochs"):
        for dataset_name in tqdm_normal(datasets, desc="Datasets"):
            data_filename = "{}.h5".format(dataset_name)
            data_filepath = os.path.join(base, "data", data_filename)

            with h5py.File(data_filepath, "r") as f:
                data_train = f["data_train"][:]
                data_dev = f["data_dev"][:]
                data_train = data_train.reshape(
                    data_train.shape[0], data_train.shape[1], 1
                )
                data_dev = data_dev.reshape(data_dev.shape[0], data_dev.shape[1], 1)
                peaks_train = f["peaks_train"][:]

                no_peak_ratio = (data_train.shape[0] - np.sum(peaks_train)) / np.sum(
                    peaks_train
                )
                # There are `no_peak_ratio` times more no peak samples. To equalize the
                # importance of samples we give samples that contain a peak more weight
                # but we never downweight peak windows!
                sample_weight = (peaks_train * np.max((0, no_peak_ratio - 1))) + 1

                history = autoencoder.fit(
                    data_train,
                    data_train,
                    initial_epoch=epoch,
                    epochs=1,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(data_dev, data_dev),
                    sample_weight=sample_weight,
                    verbose=0,
                    callbacks=[tqdm_keras(show_outer=False)],
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
        "-n",
        "--definition",
        help="path to neural network definition file",
        default="definition.json",
    )
    parser.add_argument(
        "-d", "--datasets", help="path to datasets file", default="datasets.json"
    )
    parser.add_argument(
        "-s", "--settings", help="path to settings file", default="settings.json"
    )
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs", default=25)
    parser.add_argument(
        "-b", "--batch_size", type=int, help="size of a batch", default=32
    )
    parser.add_argument("-w", "--peak_weight", type=int, help="peak weight", default=1)
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
        with open(args.definition, "r") as f:
            definition = json.load(f)
    except FileNotFoundError:
        print("Please provide a neural network definition file via `--definition`")
        sys.exit(2)

    try:
        with open(args.settings, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Please provide a settings file via `--settings`")
        sys.exit(2)

    try:
        with open(args.datasets, "r") as f:
            datasets = json.load(f)
    except FileNotFoundError:
        print("Please provide a datasets file via `--datasets`")
        sys.exit(2)

    if args.ignore_warns:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    batch_size = args.batch_size
    epochs = args.epochs
    peak_weight = args.peak_weight

    datasets = list(datasets.keys())

    print(
        "Train neural networks for {} epochs on {} datasets".format(
            epochs, len(datasets)
        )
    )

    train(
        definition,
        settings,
        datasets,
        epochs,
        batch_size,
        peak_weight,
        clear=args.clear,
    )
