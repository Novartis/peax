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


def chunk(bigwig, window_size, step_size, aggregation, chroms, verbose=False):
    base_bins = math.ceil(window_size / aggregation)

    chrom_values = []

    for chrom in chroms:
        if chrom not in bbi.chromsizes(bigwig):
            print(
                "Skipping chrom (not in bigWig file):",
                chrom,
                bbi.chromsizes(bigwig)[chrom],
            )
            continue

        chrom_size = bbi.chromsizes(bigwig)[chrom]

        values = np.zeros(
            (math.ceil((chrom_size - step_size) / step_size), base_bins)
        )
        starts = np.arange(0, chrom_size - step_size, step_size)
        ends = np.append(
            np.arange(window_size, chrom_size, step_size), chrom_size
        )
        bins = window_size / aggregation

        # Extract all but the last window in one fashion (faster than `fetch`
        # with loops)
        values[:-1] = bbi.stackup(
            bigwig,
            [chrom] * (starts.size - 1),
            starts[:-1],
            ends[:-1],
            bins=bins,
            missing=0.0,
        )
        final_bins = math.ceil((ends[-1] - starts[-1]) / aggregation)
        # Extract the last window separately because it's size is likely to be
        # different from the others
        values[-1, :final_bins] = bbi.fetch(
            bigwig, chrom, starts[-1], ends[-1], bins=final_bins, missing=0.0
        )

        if verbose:
            print(
                "Chrom: {}".format(chrom),
                "# win: {}".format(values.shape[0]),
                "Max:   {}".format(np.max(values)),
            )

        chrom_values.append(values)

    return chrom_values
