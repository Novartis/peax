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

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from server import bigwig, utils
from server.autoencoders import Autoencoders


def load(
    bigwig_path: str,
    window_size: int,
    resolution: int,
    step_freq: int,
    chroms: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    return bigwig.chunk(
        bigwig_path,
        window_size,
        resolution,
        step_freq,
        chroms,
        verbose=verbose,
    )


def prepare(
    aes: Autoencoders, dataset_defs: list, config: dict, verbose: bool = False
) -> list:
    datasets = {}

    # Extract individual datasets
    for dataset_def in dataset_defs:
        ae = aes.get(dataset_def["content_type"])
        if ae is not None:
            datasets[dataset_def["filepath"]] = dataset_def.copy()
            dataset = datasets[dataset_def["filepath"]]
            dataset["chromsizes"] = bigwig.get_chromsizes(
                dataset_def["filepath"]
            )
            # Data is normalized by chromosomes
            dataset["chunked_data"] = load(
                dataset_def["filepath"],
                ae.window_size,
                ae.resolution,
                ae.window_size // config["step_freq"],
                config["chroms"],
                verbose=verbose,
            )
            if ae.input_dim == 3 and dataset["chunked_data"].ndim == 2:
                dataset["chunked_data"] = dataset["chunked_data"].reshape(
                    *dataset["chunked_data"].shape, ae.channels
                )
            dataset["encoded_data"] = ae.encode(dataset["chunked_data"])
            dataset["autoencoded_data"] = ae.autoencode(
                dataset["chunked_data"]
            )

    # Check that all datasets have the same number of windows
    num_windows = set(
        map(lambda x: len(x["chunked_data"]), list(datasets.values()))
    )
    assert len(num_windows) == 1

    # Concatenate data
    total_len = 0
    total_len_encoded = 0
    for ae in aes.aes:
        total_len += int(aes.window_size // ae.resolution)
        total_len_encoded += ae.latent_dim

    N = datasets[dataset_defs[0]["filepath"]]["chunked_data"].shape[0]

    concat_data = np.zeros((N, total_len))
    concat_data_autoencoded = np.zeros((N, total_len))
    concat_data_encoded = np.zeros((N, total_len_encoded))

    current_pos = 0
    current_pos_encoded = 0
    for i, dataset_id in enumerate(datasets):
        dataset = datasets[dataset_id]
        ae = aes.get(dataset["content_type"])
        bins = int(aes.window_size // ae.resolution)

        concat_data[:, current_pos : current_pos + bins] = np.squeeze(
            dataset["chunked_data"]
        )

        concat_data_autoencoded[
            :, current_pos : current_pos + bins
        ] = np.squeeze(dataset["autoencoded_data"])

        concat_data_encoded[
            :, current_pos_encoded : current_pos_encoded + ae.latent_dim
        ] = np.squeeze(dataset["encoded_data"])

    # Remove individual data of dataset
    for dataset_id in datasets:
        dataset = datasets[dataset_id]

        dataset.pop("chunked_data", None)
        dataset.pop("autoencoded_data", None)
        dataset.pop("encoded_data", None)

    # Merge autoencoded regions
    concat_data_autoencoded = utils.merge_interleaved_mat(
        concat_data_autoencoded,
        config["step_freq"],
        utils.get_norm_sym_norm_kernel(int(ae.window_size / ae.resolution)),
    )

    datasets["__concat__"] = concat_data
    datasets["__concat_autoencoded__"] = concat_data_autoencoded
    datasets["__concat_encoded__"] = concat_data_encoded

    return datasets


def normalize(data, percentile=99.9):
    cutoff = np.percentile(data, (0, percentile))
    data_norm = np.copy(data)
    data_norm[np.where(data_norm < cutoff[0])] = cutoff[0]
    data_norm[np.where(data_norm > cutoff[1])] = cutoff[1]

    return MinMaxScaler().fit_transform(data_norm)
