#!/usr/bin/env python

import argparse
import h5py
import json
import numpy as np
import os
import pathlib

from keras.metrics import mse, binary_crossentropy
from keras.models import load_model

from ae.metrics import dtw_metric
from ae.utils import get_tqdm, evaluate_model


def evaluate(model_name, datasets: dict, base: str = ".", clear: bool = False):
    # Create data directory
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)

    tqdm = get_tqdm()

    encoder_filepath = os.path.join(
        base, "models", "{}---encoder.h5".format(model_name)
    )
    decoder_filepath = os.path.join(
        base, "models", "{}---decoder.h5".format(model_name)
    )
    evaluation_filepath = os.path.join(
        base, "models", "{}---evaluation.h5".format(model_name)
    )

    if (
        not pathlib.Path(encoder_filepath).is_file()
        or not pathlib.Path(decoder_filepath).is_file()
    ):
        print("Encode and decoder need to be available")
        return

    if pathlib.Path(evaluation_filepath).is_file() and not clear:
        print("Model is already evaluated. Overwrite with `--clear`")
        return

    dtw = dtw_metric()

    encoder = load_model(encoder_filepath)
    decoder = load_model(decoder_filepath)

    total_loss = np.zeros((len(datasets), 3))

    for dataset_name in tqdm(datasets, desc="Datasets", unit="dataset"):
        data_filename = "{}.h5".format(dataset_name)
        data_filepath = os.path.join(base, "data", data_filename)

        with h5py.File(data_filepath, "r") as f:
            data_test = f["data_test"][:]
            data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], 1)

            loss = evaluate_model(
                encoder,
                decoder,
                data_test,
                numpy_metrics=[dtw],
                keras_metrics=[mse, binary_crossentropy],
            )
            total_loss = np.vstack((total_loss, loss))
            # predicted = predicted.reshape(predicted.shape[0], predicted.shape[1])

    with h5py.File(evaluation_filepath, "w") as f:
        f.create_dataset("total_loss", data=total_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peax Testor")
    parser.add_argument("-m", "--model-name", help="name of the model")
    parser.add_argument(
        "-d", "--datasets", help="path to your datasets file", default="datasets.json"
    )
    parser.add_argument(
        "-c", "--clear", action="store_true", help="clears previously downloadeds"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="turn on verbose logging"
    )

    args = parser.parse_args()

    try:
        with open(args.datasets, "r") as f:
            datasets = json.load(f)
    except FileNotFoundError:
        print("You need to provide a datasets file via `--datasets`")
        raise

    evaluate(args.model_name, datasets, clear=args.clear)
