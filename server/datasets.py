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


class Datasets:
    def __init__(self):
        self.datasets = []
        self.datasets_by_id = {}
        self.datasets_by_type = {}
        self.chromsizes = None
        self.concat_data = None
        self.concat_encoding = None
        self.concat_autoencoding = None

    def __iter__(self):
        return iter(self.datasets)

    def add(self, dataset):
        if not self.chromsizes:
            self.chromsizes = dataset.chromsizes
            self.chromsizes_cum = np.cumsum(self.chromsizes) - self.chromsizes

        if not self.chromsizes.equals(dataset.chromsizes):
            raise ValueError(
                "Incorrect coordinates: all datasets need to have the same coordinates."
            )

        self.datasets.append(dataset)
        self.datasets_by_id[dataset.id] = dataset

        try:
            self.datasets_by_type[dataset.content_type].append(dataset)
        except KeyError:
            self.datasets_by_type[dataset.content_type] = [dataset]

    def export(self, use_uuid: bool = False):
        return [dataset.export(use_uuid) for dataset in self.datasets]

    def get(self, dataset_id: str):
        if dataset_id in self.datasets:
            return self.datasets[dataset_id]
        raise KeyError("No dataset with ID '{}' found".format(dataset_id))

    def size(self):
        return len(self.datasets)

    def get_by_type(self, dtype: str):
        if dtype in self.datasets_by_type:
            return self.datasets_by_type[dtype]
        raise KeyError("No datasets of type '{}' found".format(dtype))

    def prepare(self, encoders, config, verbose: bool = False) -> list:
        # Used for assertion checking
        set_num_windows = set()

        for encoder in encoders:
            try:
                for dataset in self.get_by_type(encoder.content_type):
                    dataset.prepare(config, encoder, verbose=verbose)
                    set_num_windows.add(dataset.num_windows)
            except KeyError:
                # If there's no data for the encoder we simply continue with
                # our lives
                pass

        # Check that all datasets have the same number of windows
        assert len(set_num_windows) == 1

        num_windows = set_num_windows.pop()

        # Concatenate data
        total_len_windows = 0
        total_len_encoded = 0
        for encoder in encoders.encoders:
            total_len_windows += int(encoders.window_size // encoder.resolution)
            total_len_encoded += encoder.latent_dim

        self.concat_data = np.zeros((num_windows, total_len_windows))
        self.concat_encoding = np.zeros((num_windows, total_len_encoded))

        current_pos = 0
        current_pos_encoded = 0
        for dataset in self.datasets:
            encoder = encoders.get(dataset.content_type)

            self.concat_data[
                :, current_pos : current_pos + dataset.num_bins
            ] = np.squeeze(dataset.chunks)

            self.concat_encoding[
                :, current_pos_encoded : current_pos_encoded + encoder.latent_dim
            ] = np.squeeze(dataset.encoding)

        # Remove individual data of dataset
        for dataset in self.datasets:
            dataset.chunks = None
            dataset.encoding = None
