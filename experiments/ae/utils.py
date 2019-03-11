"""
Copyright 2018 Novartis Institutes for BioMedical Research Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import bbi
import h5py
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import re
import sys
import warnings

from matplotlib.cm import copper
from typing import Tuple
from tqdm import tqdm, tqdm_notebook
from IPython.display import Image, display

# Stupid Keras things is a smart way to always print. See:
# https://github.com/keras-team/keras/issues/1406
stderr = sys.stderr
sys.stderr = open(os.devnull, "w")
from keras_tqdm import TQDMCallback, TQDMNotebookCallback
from keras import backend as K
from keras.layers import Input
from keras.models import Model, load_model

sys.stderr = stderr

from ae.loss import get_loss


def train(
    autoencoder,
    train,
    test,
    epochs=50,
    batch_size=256,
    shuffle=True,
    sample_weights=None,
    verbose=1,
):
    return autoencoder.fit(
        train,
        train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        validation_data=(test, test),
        sample_weight=sample_weights,
        verbose=verbose,
    )


def predict(encoder, decoder, test, validator=None):
    encoded = encoder.predict(test.reshape(test.shape[0], test.shape[1], 1))
    decoded = decoder.predict(encoded)

    loss = None
    if validator is not None:
        loss = K.eval(
            validator(
                K.variable(test.reshape(test.shape[0], test.shape[1])),
                K.variable(decoded.reshape(decoded.shape[0], decoded.shape[1])),
            )
        )

    return decoded, loss, encoded


def evaluate_model(
    encoder,
    decoder,
    data_test,
    keras_metrics: list = [],
    keras_metric_names: list = [],
    numpy_metrics: list = [],
    numpy_metric_names: list = [],
    batch_size: int = 10240,
    verbose: bool = False,
):
    N = data_test.shape[0]

    if verbose:
        print("Evaluate {} windows... ".format(N), end="", flush=True)

    loss = None

    if data_test.ndim == 3:
        data_test = data_test.squeeze(axis=2)

    for batch_start in np.arange(0, N, batch_size):
        batch = data_test[batch_start : batch_start + batch_size]

        batch_prediction = decoder.predict(
            encoder.predict(batch.reshape(batch.shape[0], batch.shape[1], 1))
        )

        if batch_prediction.ndim == 3:
            batch_prediction = batch_prediction.squeeze(axis=2)

        batch_loss = np.zeros((batch.shape[0], len(numpy_metrics) + len(keras_metrics)))

        i = 0

        if len(keras_metrics) > 0:
            k_data = K.variable(batch)
            k_pred = K.variable(batch_prediction)

        for metric in keras_metrics:
            if verbose:
                try:
                    metric_name = keras_metric_names[i]
                except IndexError:
                    metric_name = "#{}".format(i)
                print(
                    "Compute {} metric (keras) ... ".format(metric_name),
                    end="",
                    flush=True,
                )

            batch_loss[:, i] = K.eval(metric(k_data, k_pred))
            i += 1

            if verbose:
                print("done!")

        for metric in numpy_metrics:
            if verbose:
                try:
                    metric_name = numpy_metric_names[i]
                except IndexError:
                    metric_name = "#{}".format(i)
                print(
                    "Compute {} metric (numpy) ... ".format(metric_name),
                    end="",
                    flush=True,
                )

            batch_loss[:, i] = metric(batch, batch_prediction)
            i += 1

            if verbose:
                print("done!")

        if loss is None:
            loss = batch_loss
        else:
            loss = np.concatenate((loss, batch_loss), axis=0)

    return loss


def predict_2d(encoder, decoder, test, validator=None):
    encoded = encoder.predict(test)
    decoded = decoder.predict(encoded)
    decoded = decoded.reshape(decoded.shape[0], decoded.shape[1], decoded.shape[2])
    test = test.reshape(test.shape[0], test.shape[1], test.shape[2])

    loss = None
    if validator is not None:
        loss = K.eval(validator(K.variable(test), K.variable(decoded)))

    return decoded, loss, encoded


def evaluate(autoencoder, test):
    return autoencoder.evaluate(test)


def to_2d(data_1d, ydim, val_max=255, dtype=int):
    data_1d_scaled = data_1d * ydim

    data_2d = np.zeros((data_1d.shape[0], data_1d.shape[1], ydim)).astype(dtype)
    data_2d[:, :] = np.arange(ydim)

    end_full = np.floor(data_1d_scaled).astype(int)
    end_half = np.ceil(data_1d_scaled - 1).astype(int)
    half_ints = np.round(
        ((data_1d_scaled) - np.floor(data_1d_scaled)) * val_max
    ).astype(dtype)
    half_ints[np.where(half_ints == 0)] = val_max

    full = np.where(
        data_2d < end_full.reshape((end_full.shape[0], end_full.shape[1], 1))
    )
    half = np.where(
        data_2d == end_half.reshape((end_half.shape[0], end_half.shape[1], 1))
    )
    rest = np.where(
        data_2d > end_half.reshape((end_half.shape[0], end_half.shape[1], 1))
    )
    data_2d[full] = val_max
    data_2d[half] = half_ints[np.where(end_half != -1)]
    data_2d[rest] = 0
    data_2d = np.swapaxes(data_2d, 2, 1)[:, ::-1]

    return data_2d


def clamp(x):
    return int(max(0, min(x * 255, 255)))


def rgba_to_hex(rgba):
    return "#{0:02x}{1:02x}{2:02x}".format(
        clamp(rgba[0]), clamp(rgba[1]), clamp(rgba[2])
    )


def plt_bw(ground_truth, predicted, loss, rows, ymax=None):
    """Plot windows and color encode by loss"""

    if ymax is None:
        ymax = np.max(ground_truth[rows])

    lossmax = np.max(loss[rows])
    bg = "#ffffff"
    n = rows.size

    plt.figure(figsize=(20, n * 2))
    for i, k in enumerate(rows):
        # display original
        ax = plt.subplot(n * 2, 1, i * 2 + 1)
        ax.set_facecolor("#888888")
        plt.bar(np.arange(ground_truth[k].size), ground_truth[k], color=bg)
        plt.ylim(0, ymax)
        ax.get_xaxis().set_visible(False)

        c = rgba_to_hex(copper((lossmax - loss[k]) / lossmax))

        # display reconstruction
        ax = plt.subplot(n * 2, 1, i * 2 + 2)
        ax.set_facecolor(c)
        plt.bar(np.arange(predicted[k].size), predicted[k], color=bg)
        plt.ylim(0, ymax)
        ax.get_xaxis().set_visible(False)
    plt.show()


def value_changes(arr):
    """Get locations of value changes in a 2D numpy array

    Arguments:
        arr {np.array} -- 2D numpy array

    Returns:
        {np.array} -- Boolean 2D numpy array where {True} indicates a value
            change from the current to the next value
    """
    changes = np.zeros(arr.shape).astype(bool)
    changes[:, :-1] = arr[:, :-1] != arr[:, 1:]
    return changes


def count_peaks(arr):
    """Count peaks in a Boolean 2D numpy array

    Arguments:
        arr {np.array} -- 2D Boolean array where {1}s define intervals

    Returns:
        {np.array} -- 1D array with the number of consecutive intervals
    """
    changes = value_changes(arr)
    num_peaks = np.sum(changes, axis=1)
    num_peaks[np.where(arr[:, 0] == 1)] += 1
    num_peaks[np.where(arr[:, -1] == 1)] += 1
    return num_peaks // 2


def peak_heights(intervals, values, num_peaks, aggregator):
    n, k = intervals.shape

    heights = np.zeros((n, np.max(num_peaks.astype(int))))
    heights[:] = np.nan

    for i in range(n):
        changes = intervals[i, :-1] != intervals[i, 1:]
        indices = np.append(-1, np.append(np.where(changes), k - 1))
        c = 0
        for j in range(indices.size - 1):
            val_idx = indices[j] + 1
            if intervals[i, val_idx] == 1:
                heights[i, c] = np.max(values[i, val_idx : indices[(j + 1)] + 1])
                c += 1

    return aggregator(heights, axis=1)


def rle(arr):
    changes = arr[:-1] != arr[1:]
    indices = np.append(-1, np.append(np.where(changes), arr.size - 1))
    return np.diff(indices)


def peak_widths(arr, aggregator):
    widths = np.zeros((arr.shape[0],))

    c = 0
    for i in range(arr.shape[0]):
        window_widths = rle(arr[i])
        if arr[i, 0] == 0:
            if window_widths.size < 2:
                widths[c] = np.nan
            else:
                widths[c] = aggregator(window_widths[1::2])
        else:
            widths[c] = aggregator(window_widths[::2])

        c += 1

    return widths


def peak_distances(arr, aggregator):
    dists = np.zeros((arr.shape[0],))

    c = 0
    for i in range(arr.shape[0]):
        window_widths = rle(arr[i])

        if arr[i, 0] == 0:
            window_widths = window_widths[::2]
            window_widths = window_widths[1:]
        else:
            window_widths = window_widths[1::2]

        if arr[i, -1] == 0:
            window_widths = window_widths[:-1]

        if window_widths.size == 0:
            dists[c] = np.nan
        else:
            dists[c] = aggregator(window_widths)

        c += 1

    return dists


def get_stats(bigwig, bigbed, norm_vals, window_size, step_size, aggregation, chrom):
    base_bins = math.ceil(window_size / aggregation)

    if chrom not in bbi.chromsizes(bigwig):
        print(
            "Skipping chrom (not in bigWig file):", chrom, bbi.chromsizes(bigwig)[chrom]
        )
        return None

    chrom_size = bbi.chromsizes(bigwig)[chrom]

    intervals = np.zeros((math.ceil((chrom_size - step_size) / step_size), base_bins))
    starts = np.arange(0, chrom_size - step_size, step_size)
    ends = np.append(np.arange(window_size, chrom_size, step_size), chrom_size)
    bins = window_size / aggregation

    # Extract all but the last window in one fashion (faster than `fetch`
    # with loops)
    intervals[:-1] = bbi.stackup(
        bigbed, [chrom] * (starts.size - 1), starts[:-1], ends[:-1], bins=bins
    )

    final_bins = math.ceil((ends[-1] - starts[-1]) / aggregation)
    # Extract the last window separately because it's size is likely to be
    # different from the others
    intervals[-1, :final_bins] = bbi.fetch(
        bigbed, chrom, starts[-1], ends[-1], bins=final_bins, missing=0.0
    )

    intervals = np.round(intervals).astype(int)

    # 0. Number of intevals
    # 1. Min width of peaks
    # 2. Max width of peaks
    # 3. Median width of peaks
    # 4. Min distance of peaks
    # 5. Max distance pf peaks
    # 6. Median distance of peaks
    # 7. Sum of height of peaks
    # 8. Max height of peaks
    # 9. Median height of peaks
    # 10. Median signal
    # 11. Total signal
    # 12. Peak coverage
    stats = np.zeros((norm_vals.shape[0], 13))

    stats[:, 0] = count_peaks(intervals)

    stats[:, 1] = peak_widths(intervals, np.min)
    stats[:, 2] = peak_widths(intervals, np.max)
    stats[:, 3] = peak_widths(intervals, np.median)

    stats[:, 4] = peak_distances(intervals, np.min)
    stats[:, 5] = peak_distances(intervals, np.max)
    stats[:, 6] = peak_distances(intervals, np.median)

    stats[:, 7] = peak_heights(intervals, norm_vals, stats[:, 0], np.nansum)
    stats[:, 8] = peak_heights(intervals, norm_vals, stats[:, 0], np.nanmax)
    stats[:, 9] = peak_heights(intervals, norm_vals, stats[:, 0], np.nanmedian)

    stats[:, 10] = np.median(norm_vals, axis=1)
    stats[:, 11] = np.sum(norm_vals, axis=1)
    stats[:, 12] = peak_widths(intervals, np.sum) / base_bins

    return stats, np.round(intervals).astype(int)


def chunk_beds_binary(
    bigBed: str,
    window_size: int,
    step_size: int,
    chroms: list,
    verbose: bool = True,
    print_per_chrom: callable = None,
) -> np.ndarray:
    """Chunk a bed file of binary annotations into windows

    Extract a single boolean value for genomic windows representing whether a bed
    annotation is present or now. This is for example useful to quickly determine if a
    window contains a peak annotation or not. If you need to know about the actual value
    of the annotation you might need a more involved method.

    Arguments:
        bigBed {str} -- path to the bigBed file
        window_size {int} -- size of the genomic windows in base pairs
        step_size {int} -- size of the steps in base pairs
        chroms {list} -- list of chromosomes from which windows should be extracted

    Keyword Arguments:
        verbose {bool} -- if ``True`` print some stuff (default: {True})
        print_per_chrom {callable} -- if ``True`` print stuff per iteration (default: {None})

    Returns:
        {np.ndarray} -- a 1D array indicating which windows contain at least one annotations
    """
    base_bins = 1
    num_total_windows = 0

    chrom_sizes = bbi.chromsizes(bigBed)
    step_freq = int(window_size / step_size)

    for chrom in chroms:
        chrom_size = chrom_sizes[chrom]
        num_total_windows += (
            np.ceil((chrom_size - window_size) / step_size).astype(int) + 1
        )

    values = np.zeros((num_total_windows, base_bins))

    start = 0
    for chrom in chroms:
        if chrom not in chrom_sizes:
            print("Skipping chrom (not in bigBed file):", chrom, chrom_sizes[chrom])
            continue

        chrom_size = chrom_sizes[chrom]
        bins = np.ceil(chrom_size / window_size).astype(int)
        num_windows = np.ceil((chrom_size - window_size) / step_size).astype(int) + 1

        start_pos = np.arange(0, step_size * step_freq, step_size)
        end_pos = np.arange(
            bins * window_size, bins * window_size + step_size * step_freq, step_size
        )

        end = start + num_windows

        tmp = (
            np.transpose(
                bbi.stackup(
                    bigBed,
                    [chrom] * start_pos.size,
                    start_pos,
                    end_pos,
                    bins=bins,
                    missing=0,
                )
            )
            .reshape((bins * step_freq, base_bins))
            .astype(int)
        )

        values[start:end] = tmp[0:num_windows]

        if verbose and not print_per_chrom:
            print(
                "Extracted",
                "{} windows".format(num_windows),
                "from {}".format(chrom),
                "with a max value of {}.".format(np.max(values[start:end])),
            )

        if print_per_chrom:
            print_per_chrom()

        start = end

    return values.astype(int)


def filter_windows_by_peaks(
    signal: np.ndarray,
    narrow_peaks: np.ndarray,
    broad_peaks: np.ndarray,
    incl_pctl_total_signal: float = 25,
    incl_pct_no_signal: float = 5,
    peak_ratio: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """Filter windows by peak annotations and their total signal

    This method filters windows based on whether they contain at least 1 narrow or broad
    peak annotation or whether their total signal is equal or greater than a certain
    percentile of the averaged total signal of windows containing a peak annotation. The
    goal of this method is to balance the datasets such that roughly half of the windows
    contain some signal or patterns that is worth to be learned.

    Arguments:
        signal {np.ndarray} -- 2D array with the windows' signal
        narrow_peaks {np.ndarray} -- 1D array specifying whether a window contains a
            narrow peak annotation
        broad_peaks {np.ndarray} -- 1D array specifying whether a window contains a
            broad peak annotation

    Keyword Arguments:
        incl_pctl_total_signal {float} -- percentile of the averaged total signal of
            windows containing some peak annotation that should determine if a window
            without a peak annotation is included or filtered out (default: {25})
        incl_pct_no_signal {float} -- percent of empty window that should remain and not
            be filtered out (default: {5})
        peak_ratio {float} -- ratio of windows having a peak annotation vs windows
            having no peak annotation (default: {1})
        verbose {bool} -- if ``True`` print some more stuff (default: {False})

    Returns:
        {np.ndarray} -- 1D Boolean array for selected the remaining windows
    """
    win_with_narrow_or_broad_peaks = (narrow_peaks + broad_peaks).flatten()

    # Total signal per window
    win_total_signal = np.sum(signal, axis=1)

    has_peaks = win_with_narrow_or_broad_peaks > 0

    # Select all windows where the total signal is at least 25 percentile
    # of the windows containing at least 1 peak.
    win_total_signal_gt_pctl = win_total_signal > np.percentile(
        win_total_signal[has_peaks], incl_pctl_total_signal
    )
    win_total_signal_is_zero = win_total_signal == 0

    num_total_win = signal.shape[0]

    pos_win = has_peaks | win_total_signal_gt_pctl
    pos_win_idx = np.arange(num_total_win)[pos_win]
    neg_not_empty_win = ~pos_win & ~win_total_signal_is_zero
    neg_not_empty_win_idx = np.arange(num_total_win)[neg_not_empty_win]
    neg_empty_win = ~pos_win & win_total_signal_is_zero
    neg_empty_win_idx = np.arange(num_total_win)[neg_empty_win]

    if verbose:
        print(
            "Windows: total = {} | with peaks = {} | with signal gt {} pctl = {}".format(
                num_total_win,
                np.sum(has_peaks),
                incl_pctl_total_signal,
                np.sum(win_total_signal_gt_pctl),
            )
        )

    pct_not_empty = 1 - (incl_pct_no_signal / 100)

    # The total number of windows as a factor of the number of windows with peaks.
    # number of total windows = number of windows with peaks * total_win_by_peaks
    total_win_by_peaks = (1 / peak_ratio) + 1

    to_be_sampled = np.min(
        [
            # Do not sample more than the complete number of negative but non-empty
            # windows
            np.max([0, np.sum(neg_not_empty_win)]),
            np.max(
                [
                    # Sample at most half as many windows as windows with peak
                    # annotations. This is only necessary when there are few windows
                    # with annotated peaks but a lot of windows where the total signal
                    # is larger than 25-percentile of the total signal of windows with
                    # peaks
                    np.sum(has_peaks) * 0.5,
                    np.min(
                        [
                            total_win_by_peaks * np.sum(has_peaks) - np.sum(pos_win),
                            np.sum(~pos_win),
                        ]
                    ),
                ]
            ),
        ]
    ).astype(int)
    to_be_sampled_not_empty = np.ceil(to_be_sampled * pct_not_empty).astype(int)
    to_be_sampled_empty = to_be_sampled - to_be_sampled_not_empty

    neg_not_empty_win_no_subsample = np.random.choice(
        neg_not_empty_win_idx, to_be_sampled_not_empty, replace=False
    )

    neg_empty_win_no_subsample = np.random.choice(
        neg_empty_win_idx, to_be_sampled_empty, replace=False
    )

    return np.sort(
        np.concatenate(
            (pos_win_idx, neg_not_empty_win_no_subsample, neg_empty_win_no_subsample)
        )
    )


def split_train_dev_test(
    data: np.ndarray,
    peaks: np.ndarray,
    dev_set_size: float,
    test_set_size: float,
    rnd_seed: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train, dev, and test set

    Arguments:
        data {np.ndarray} -- 2D array with the data to be split by rows
        peaks {np.ndarray} -- 1D array with Boolean annotations for the rows that should
            be split in the same manner as ``data``
        dev_set_size {float} -- percent of the rows of ``data`` that should be used for
            development
        test_set_size {float} -- percent of the rows of ``data`` that should be used for
            testing
        rnd_seed {int} -- np.random seed. needed to ensure reproducibility

    Keyword Arguments:
        verbose {bool} -- if ``True`` print some more stuff (default: {False})
    """
    assert data.shape[0]

    total_num_filtered_windows = data.shape[0]
    shuffling = np.arange(total_num_filtered_windows)

    # Shuffle window ids and use the shuffled ids to shuffle the window data and window peaks
    np.random.seed(rnd_seed)
    np.random.shuffle(shuffling)
    data_shuffled = data[shuffling]
    peaks_shuffled = peaks[shuffling]

    # Split into train, dev, and test set
    split_1 = int((1.0 - dev_set_size - test_set_size) * total_num_filtered_windows)
    split_2 = int((1.0 - test_set_size) * total_num_filtered_windows)

    data_train = data_shuffled[:split_1]
    peaks_train = peaks_shuffled[:split_1]
    data_dev = data_shuffled[split_1:split_2]
    peaks_dev = peaks_shuffled[split_1:split_2]
    data_test = data_shuffled[split_2:]
    peaks_test = peaks_shuffled[split_2:]

    if verbose:
        print(
            "Train: {} (with {:.2f}% peaks) Dev: {} (with {:.2f}% peaks) Test: {} (with {:.2f}% peaks)".format(
                data_train.shape[0],
                np.sum(peaks_train) / peaks_train.shape[0] * 100,
                data_dev.shape[0],
                np.sum(peaks_dev) / peaks_dev.shape[0] * 100,
                data_test.shape[0],
                np.sum(peaks_test) / peaks_test.shape[0] * 100,
            )
        )

    return (
        data_train,
        peaks_train,
        data_dev,
        peaks_dev,
        data_test,
        peaks_test,
        shuffling,
    )


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
    "batch_norm": "bn",
    "batch_norm_input": "bni",
    "peak_weight": "pw",
    "signal_weighting": "sw",
    "signal_weighting_zero_point_percentage": "swz",
}


def namify(definition):
    def to_str(value):
        if isinstance(value, bool):
            return str(int(value))
        else:
            return str(value)

    name = ""
    for i, key in enumerate(definition):
        value = definition[key]
        key = abbr[key] if key in abbr else key
        name += "--" + key + "-" if i > 0 else key + "-"
        if isinstance(value, list):
            name += "-".join([to_str(v) for v in value])
        else:
            name += str(abbr[value]) if value in abbr else to_str(value)
    return name


def is_ipynb():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_tqdm(is_keras: bool = False):
    # Determine which tqdm to use
    if is_ipynb():
        if is_keras:
            return TQDMNotebookCallback
        else:
            return tqdm_notebook
    else:
        if is_keras:
            return TQDMCallback
        else:
            return tqdm


def get_models(ae_filepath: str, silent: bool = True):
    """Get encoder, decoder, and autoencoder from stored file

    This method loads the autoencoder model and creates the related encoder and decoder
    models.

    Arguments:
        ae_filepath {str} -- Path to the stored autoencoder model

    Keyword Arguments:
        silent {bool} -- If {True} Keras warnings from {load_model} are silenced. (default: {True})

    Returns:
        {tuple} -- Tuple with encoder, decoder, autoencoder models
    """
    # Since Keras does not store custom loss functions we have to first determine which
    # loss function we've used for training to pass that into load_model()
    regex = r"--l-(([a-z0-9]+\-?)+)--"
    matches = re.search(regex, ae_filepath)

    loss = None
    custom_objects = {}
    if matches is not None:
        loss = matches.group(1)
        loss = get_loss(matches.group(1))
        if hasattr(loss, "__name__"):
            custom_objects = {loss.__name__: loss}
    else:
        print("Could not determine loss function")
        return None, None, None

    if silent:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            autoencoder = load_model(ae_filepath, custom_objects=custom_objects)
    else:
        autoencoder = load_model(ae_filepath, custom_objects=custom_objects)

    # Find embedding layer
    embedding_layer = None
    embedding_layer_idx = None
    for i, layer in enumerate(autoencoder.layers):
        if layer.name == "embed":
            embedding_layer = layer
            embedding_layer_idx = i

    # Create encoder
    inputs = autoencoder.input
    encoded = inputs
    for i in range(1, embedding_layer_idx + 1):
        encoded = autoencoder.layers[i](encoded)

    encoder = Model(inputs, encoded)

    embedding = embedding_layer.output_shape[1]

    encoded_input = Input(shape=(embedding,), name="input")
    decoded_input = encoded_input
    for i in range(embedding_layer_idx + 1, len(autoencoder.layers)):
        decoded_input = autoencoder.layers[i](decoded_input)
    decoder = Model(encoded_input, decoded_input)

    return encoder, decoder, autoencoder


def plot_total_signal(dataset: str, base: str = "."):
    """Plot total signal of the train, dev, and test set

    Arguments:
        dataset {str} -- Name of the dataset

    Keyword Arguments:
        base {str} -- Path to the base directory (default: {"."})
    """
    with h5py.File(os.path.join(base, "data", "{}.h5".format(dataset)), "r") as f:
        total_signal_train = np.sum(f["data_train"], axis=1)
        total_signal_dev = np.sum(f["data_dev"], axis=1)
        total_signal_test = np.sum(f["data_test"], axis=1)

        num_win = f["data_train"].shape[0]
        num_empty_win = np.sum(total_signal_train == 0)

        print(
            "{} ({:.2f}%) out of {} windows are empty".format(
                num_empty_win, num_empty_win / num_win * 100, num_win
            )
        )

        fig = plt.figure(figsize=(12, 4))
        sns.distplot(total_signal_train, bins=np.arange(40), label="Train")
        sns.distplot(total_signal_dev, bins=np.arange(40), label="Dev")
        sns.distplot(total_signal_test, bins=np.arange(40), label="Test")
        plt.xlabel("Total signal per window")
        fig.legend()


def plot_windows(
    dataset: str,
    model_name: str = None,
    ds_type: str = "train",
    with_peaks: bool = True,
    num: int = 10,
    min_signal: float = 0,
    max_signal: float = math.inf,
    base: str = ".",
    save_as: str = None,
    trained_on_single_dataset: bool = False,
    silent: bool = False,
    repetition: str = None,
    custom_postfix: str = None,
    batch_size: int = 10240,
):
    with h5py.File(os.path.join(base, "data", "{}.h5".format(dataset)), "r") as f:
        data_type = "data_{}".format(ds_type)
        peaks_type = "peaks_{}".format(ds_type)

        if data_type not in f or peaks_type not in f:
            sys.stderr.write("Dataset type not available: {}\n".format(ds_type))
            return

        N = f[data_type].shape[0]
        L = f[data_type].shape[1]
        total_signal = None

        for batch_start in np.arange(0, N, batch_size):
            batch_data = np.squeeze(
                f[data_type][batch_start : batch_start + batch_size], axis=2
            )

            batch_total_signal = np.sum(batch_data, axis=1)

            if total_signal is None:
                total_signal = batch_total_signal
            else:
                total_signal = np.concatenate((total_signal, batch_total_signal))

        gt_min_signal = total_signal > min_signal
        st_max_signal = total_signal < max_signal

        num_windows_to_be_sampled = np.sum(gt_min_signal & st_max_signal)

        choices = np.random.choice(num_windows_to_be_sampled, num, replace=False)

        sampled_wins = f[data_type][gt_min_signal & st_max_signal][choices]
        sampled_peaks = f[peaks_type][gt_min_signal & st_max_signal][choices]

        if model_name:
            postfix = "-{}".format(dataset) if trained_on_single_dataset else ""

            if repetition is not None:
                postfix = "{}__{}".format(postfix, repetition)

            if custom_postfix is not None:
                postfix = "{}-{}".format(postfix, custom_postfix)

            autoencoder_filepath = os.path.join(
                base, "models", "{}---autoencoder{}.h5".format(model_name, postfix)
            )

            encoder, decoder, _ = get_models(autoencoder_filepath)

            sampled_encodings, _, _ = predict(encoder, decoder, sampled_wins)
            sampled_encodings = sampled_encodings.squeeze(axis=2)

        cols = max(math.floor(math.sqrt(num) * 3 / 5), 1)
        rows = math.ceil(num / cols)

        x = np.arange(L)

        fig, axes = plt.subplots(
            rows, cols, figsize=(8 * cols, 1.25 * rows), sharex=True
        )
        fig.patch.set_facecolor("white")

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="gray", label="Ground truth"),
            Patch(facecolor="mediumblue", label="Prediction"),
            Patch(facecolor="green", label="Prediction (w/ peak annotation)"),
        ]

        sampled_window_idx = np.arange(N)[gt_min_signal & st_max_signal][choices]

        i = 0
        for c in np.arange(cols):
            for r in np.arange(rows):
                if i >= num:
                    break

                primary_color = "green" if sampled_peaks[i] == 1 else "mediumblue"
                secondary_color = "gray"
                ground_truth_color = secondary_color if model_name else primary_color
                prediction_color = primary_color

                axes[r, c].bar(x, sampled_wins[i], width=1.0, color=ground_truth_color)
                if model_name:
                    axes[r, c].bar(
                        x,
                        sampled_encodings[i],
                        width=1.0,
                        color=prediction_color,
                        alpha=0.5,
                    )
                axes[r, c].set_xticks(x[5::10])
                axes[r, c].set_xticklabels(x[5::10])

                axes[r, c].spines["top"].set_color("silver")
                axes[r, c].spines["right"].set_color("silver")
                axes[r, c].spines["bottom"].set_color("silver")
                axes[r, c].spines["left"].set_color(primary_color)
                axes[r, c].spines["left"].set_linewidth(4)
                axes[r, c].tick_params(axis="x", colors=secondary_color)
                axes[r, c].tick_params(axis="y", colors=secondary_color)
                axes[r, c].set_ylim(0, 1)
                axes[r, c].set_title(sampled_window_idx[i])
                i += 1

        fig.legend(handles=legend_elements, loc="lower center")
        fig.tight_layout()

        if save_as is not None:
            fig.savefig(os.path.join(base, save_as), bbox_inches="tight")

        return (
            # Window indices
            sampled_window_idx,
            # Total signal of the windows
            total_signal[gt_min_signal & st_max_signal][choices],
            # Max signal in the window
            np.max(sampled_wins, axis=1),
        )


def plot_windows_from_data(
    data,
    min_signal: float = 0,
    max_signal: float = math.inf,
    num: int = 10,
    window_ids: list = None,
):
    if window_ids is not None:
        num = len(window_ids)
        sampled_wins = data[window_ids]
        sampled_window_idx = window_ids
    else:
        total_signal = np.sum(data, axis=1)

        gt_min_signal = total_signal > min_signal
        st_max_signal = total_signal < max_signal

        num_windows_to_be_sampled = np.sum(gt_min_signal & st_max_signal)

        choices = np.random.choice(num_windows_to_be_sampled, num, replace=False)

        sampled_wins = data[gt_min_signal & st_max_signal][choices]

        sampled_window_idx = np.arange(data.shape[0])[gt_min_signal & st_max_signal][
            choices
        ]

    cols = max(math.floor(math.sqrt(num) * 3 / 5), 1)
    rows = math.ceil(num / cols)

    x = np.arange(data.shape[1])

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 1.25 * rows), sharex=True)
    fig.patch.set_facecolor("white")

    i = 0
    for c in np.arange(cols):
        for r in np.arange(rows):
            if i >= num:
                break

            axis = axes[r] if cols == 1 else axes[r, c]

            axis.bar(x, sampled_wins[i], width=1.0, color="#008ca8")
            axis.set_xticks(x[5::10])
            axis.set_xticklabels(x[5::10])

            axis.spines["top"].set_color("silver")
            axis.spines["right"].set_color("silver")
            axis.spines["bottom"].set_color("silver")
            axis.spines["left"].set_color("silver")
            axis.tick_params(axis="x", colors="silver")
            axis.tick_params(axis="y", colors="silver")
            axis.set_ylim(0, 1)
            axis.set_title(sampled_window_idx[i])
            axis.set_xticks([], [])
            axis.set_yticks([], [])
            i += 1

    fig.tight_layout()


def create_hdf5_dset(f, name, data, extendable: bool = False, dtype: str = None):
    if dtype is not None:
        try:
            dtype = data.dtype
        except AttributeError:
            pass

    if name in f.keys():
        if extendable:
            f[name].resize((f[name].shape[0] + data.shape[0]), axis=0)
            f[name][-data.shape[0] :] = data
        else:
            # Overwrite existing dataset
            del f[name]
            f.create_dataset(name, data=data, dtype=dtype)
    else:
        if extendable:
            maxshape = (None, *data.shape[1:])
            f.create_dataset(name, data=data, maxshape=maxshape, dtype=dtype)
        else:
            f.create_dataset(name, data=data, dtype=dtype)


def check_status(
    name: str, step: str, dataset: str = None, base: str = ".", show_loss: bool = False
):
    with open(os.path.join(base, "definitions-{}.json".format(name)), "r") as f:
        model_names = json.load(f)

    not_found = []
    outdated = []

    for model_name in model_names:
        try:
            postfix = "-{}".format(dataset) if dataset is not None else ""
            repetition = None
            if len(model_name.split("__")) > 1:
                parts = model_name.split("__")
                model_name = parts[0]
                repetition = parts[1]
                postfix = "{}__{}".format(postfix, repetition)
            filepath = os.path.join(
                base, "models", "{}---{}{}.h5".format(model_name, step, postfix)
            )
            with h5py.File(filepath, "r") as f:
                try:
                    times = None
                    loss = None
                    val_loss = None
                    if step == "training":
                        times = f["times"][:]
                        loss = f["loss"][:]
                        val_loss = f["val_loss"][:]
                    elif step == "total_times":
                        times = f["total_times"][:]
                    times = times
                    loss = loss
                    val_loss = val_loss
                except KeyError:
                    outdated.append(filepath)
                    continue
            if show_loss:
                filename = "{}---train-loss{}.png".format(model_name, postfix)
                filepath = os.path.join(base, "models", filename)
                print(filename)
                print("Loss: {} Val loss: {}".format(np.mean(loss), np.mean(val_loss)))
                display(Image(filename=filepath))
        except OSError:
            not_found.append(filepath)

    return len(not_found) == 0, not_found, outdated
