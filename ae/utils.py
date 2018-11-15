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
import math
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib.cm import copper


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
    encoded = encoder.predict(test)
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
