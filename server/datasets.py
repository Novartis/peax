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

import h5py
import hashlib
import numpy as np
import os
import re


class Datasets:
    def __init__(self):
        self.datasets = []
        self.datasets_by_id = {}
        self.datasets_by_type = {}
        self.chromsizes = None
        self.concat_data = None
        self.concat_encoding = None
        self.concat_autoencoding = None
        self._cache_filename = None

    def __iter__(self):
        return iter(self.datasets)

    @property
    def cache_filename(self):
        return self._cache_filename

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

    def export(self, use_uuid: bool = False, autoencodings: bool = False):
        return [
            dataset.export(use_uuid=use_uuid, autoencodings=autoencodings)
            for dataset in self.datasets
            if not autoencodings or dataset.is_autoencoded
        ]

    def get(self, dataset_id: str):
        try:
            return self.datasets_by_id[dataset_id]
        except KeyError:
            raise KeyError("No dataset with ID '{}' found".format(dataset_id))

    def size(self):
        return len(self.datasets)

    def get_by_type(self, dtype: str):
        if dtype in self.datasets_by_type:
            return self.datasets_by_type[dtype]
        raise KeyError("No datasets of type '{}' found".format(dtype))

    def prepare(self, encoders, config, clear: bool = False, verbose: bool = False):
        # Used for assertion checking
        set_num_windows = set()

        for encoder in encoders:
            try:
                for dataset in self.get_by_type(encoder.content_type):
                    num_windows = dataset.prepare(
                        config, encoder, clear=clear, verbose=verbose
                    )
                    set_num_windows.add(num_windows)
            except KeyError:
                # If there's no data for the encoder we simply continue with our lives
                pass

        # Check that all datasets have the same number of windows
        assert len(set_num_windows) == 1

        num_windows = set_num_windows.pop()

        encoder_filenames = [encoder.encoder_filename for encoder in encoders]
        dataset_filenames = [dataset.filename for dataset in self.datasets]
        all_filenames = ":".join(encoder_filenames + dataset_filenames)

        # Generate filename from the set of datasets and encoders
        md5 = hashlib.md5()
        md5.update(all_filenames.encode())
        self._cache_filename = "{}.hdf5".format(md5.hexdigest())

        cache_filepath = os.path.join(config.cache_dir, self.cache_filename)

        # Concatenate data
        mode = "w" if clear else "w-"

        try:
            with h5py.File(cache_filepath, mode) as f:
                w = f.create_group("windows")
                e = f.create_group("encodings")
                for chromosome in config.chroms:
                    chr_str = str(chromosome)

                    pos = 0
                    pos_encoded = 0

                    concat_data = np.zeros((num_windows, encoders.total_len_windows))
                    concat_encoding = np.zeros(
                        (num_windows, encoders.total_len_encoded)
                    )

                    for dataset in self.datasets:
                        with h5py.File(dataset.cache_filename, "r") as ds:
                            encoder = encoders.get(dataset.content_type)
                            windows = ds["windows/{}".format(chr_str)]
                            encodings = ds["encodings/{}".format(chr_str)]

                            concat_data[
                                :, pos : pos + encoder.window_num_bins
                            ] = np.squeeze(windows)

                            concat_encoding[
                                :, pos_encoded : pos_encoded + encoder.latent_dim
                            ] = np.squeeze(encodings)

                    w[chr_str] = concat_data
                    w[chr_str].attrs["total_len_windows"] = encoders.total_len_windows

                    e[chr_str] = concat_encoding
                    e[chr_str].attrs["total_len_encoded"] = encoders.total_len_encoded

                    # Lets write to disk
                    f.flush()
        except OSError as error:
            # When `clear` is `False` and the data is already prepared then we expect to
            # see error number 17 as we opened the file in `w-` mode.
            if not clear:
                # Stupid h5py doesn't populate `error.errno` so we have to parse it out
                # manually
                matches = re.search(r"errno = (\d+)", str(error))
                if matches and int(matches.group(1)) == 17:
                    pass
                else:
                    raise
            else:
                raise
