#!/usr/bin/env python

import argparse
import h5py
import itertools as it
import json
import numpy as np
import os
import pathlib
import sys

from ae.cnn import create_model


parser = argparse.ArgumentParser(description="Peax Trainer")
parser.add_argument(
    "-d", "--definition", help="path to the neural network definition file", default=""
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

# Create data directory
pathlib.Path("models").mkdir(parents=True, exist_ok=True)

bins_per_window = settings["window_size"] // settings["resolution"]
datasets = settings["datasets"]

varying = definition["hyperparameters"]["varying"]
fixed = definition["hyperparameters"]["fixed"]
epochs = definition["epochs"]
batch_size = definition["batch_size"]
peak_weight = definition["peak_weight"]

base_def = dict({}, **fixed)

# Get the product of all possible combinations
varying_params = varying["params"]
varying_values = varying["values"]
combinations = []
for values in varying["values"]:
    combinations += list(it.product(*values))

print("Train {} neural networks for {} epochs each".format(len(combinations), epochs))


def finalize_def(prelim_def: dict) -> dict:
    conv_layers = prelim_def["conv_layers"]
    conv_filter_size = prelim_def["conv_filter_size"]
    conv_filter_size_reverse_order = prelim_def["conv_filter_size_reverse_order"]
    conv_kernel_size = prelim_def["conv_kernel_size"]
    conv_kernel_size_reverse_order = prelim_def["conv_kernel_size_reverse_order"]

    if isinstance(conv_filter_size, int) or isinstance(conv_filter_size, float):
        conv_filter_size = int(conv_filter_size)
        conv_filters = [conv_filter_size] * conv_layers
    elif isinstance(conv_filter_size, list) and len(conv_filter_size) == conv_layers:
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
    elif isinstance(conv_kernel_size, list) and len(conv_kernel_size) == conv_layers:
        if conv_kernel_size_reverse_order:
            conv_kernels = list(reversed(conv_kernel_size))
        else:
            conv_kernels = conv_kernel_size
    else:
        print("ERROR: conv_filter_size needs to be an int or a list of ints")
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


# To make model names more concise but still meaningful
abbr = {
    "conv_filters": "cf",
    "conv_kernels": "ck",
    "dense_units": "du",
    "dropouts": "do",
    "embedding": "e",
    "reg_lambda": "rl",
    "optimizer": "o",
    "learning_rate": "lr",
    "learning_rate_decay": "lrd",
    "loss": "l",
    "metrics": "m",
    "binary_crossentropy": "bce",
}


def namify(definition):
    name = ""
    for i, key in enumerate(definition):
        value = definition[key]
        key = abbr[key] if key in abbr else key
        name += "--" + key + "-" if i > 0 else key + "-"
        if isinstance(value, list):
            name += "-".join([str(v) for v in value])
        else:
            name += str(abbr[value]) if value in abbr else str(value)
    return name


for combination in combinations:
    combined_def = dict({}, **base_def)

    for i, value in enumerate(combination):
        combined_def[varying["params"][i]] = value

    final_def = finalize_def(combined_def)
    model_name = namify(final_def)
    encoder, decoder, autoencoder = create_model(
        bins_per_window, **final_def  # input_dim
    )

    for dataset_name in datasets:
        dataset = datasets[dataset_name]

        data_filename = "{}.h5".format(dataset_name)
        data_filepath = os.path.join("data", data_filename)

        with h5py.File(data_filepath, "r") as f:
            data_train = f["data_train"][:]
            data_test = f["data_test"][:]
            data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], 1)
            data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], 1)

            peaks_train = f["peaks_train"]
            no_peak_ratio = (data_train.shape[0] - peaks_train.size) / peaks_train.size
            # There are `no_peak_ratio` times more no peak samples. To equalize the
            # importance of samples we give samples that contain a peak more weight but
            # we never downweight peak windows!
            sample_weight = (peaks_train * np.max((0, no_peak_ratio - 1))) + 1

            # autoencoder.fit(
            #     data_train,
            #     data_train,
            #     epochs=epochs,
            #     batch_size=batch_size,
            #     shuffle=True,
            #     validation_data=(data_test, data_test),
            #     sample_weight=sample_weight,
            #     verbose=args.verbose,
            # )

    model_filename = "{}.h5".format(model_name)
    model_filepath = os.path.join("models", model_filename)

    encoder.save(os.path.join("models", "{}---encoder.h5".format(model_name)))
    decoder.save(os.path.join("models", "{}---decoder.h5".format(model_name)))
