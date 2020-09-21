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

from contextlib import contextmanager, suppress
from scipy.spatial.distance import cdist

from server import chromsizes, utils


class Datasets:
    def __init__(self):
        self.datasets = []
        self.datasets_by_id = {}
        self.datasets_by_type = {}
        self.custom_chromosomes = None
        self.chromsizes = None
        self.coords = None
        self._cache_filename = None
        self._total_len_windows = -1
        self._total_len_encoded = -1

    def __iter__(self):
        return iter(self.datasets)

    @property
    def cache_filename(self):
        return self._cache_filename

    @property
    def cache_filepath(self):
        return self._cache_filepath

    @property
    def total_len_windows(self):
        return self._total_len_windows

    @property
    def total_len_encoded(self):
        return self._total_len_encoded

    @property
    def length(self):
        return len(self.datasets)

    @contextmanager
    def cache(self):
        cache = h5py.File(self.cache_filepath, "r")
        try:
            yield DatasetsCache(cache)
        finally:
            cache.close()

    def add(self, dataset):
        if self.chromsizes is None:
            self.chromsizes = dataset.chromsizes
            self.chromsizes_cum = np.cumsum(self.chromsizes) - self.chromsizes

        if self.coords is None:
            self.coords = dataset.coords

        if dataset.custom_chromosomes is not None and self.custom_chromosomes is None:
            self.custom_chromosomes = dataset.custom_chromosomes

        if not chromsizes.equals(
            self.chromsizes, dataset.chromsizes, self.coords, self.custom_chromosomes
        ):
            raise ValueError(
                "Incorrect coordinates: all datasets need to have the same coordinates."
            )

        self.datasets.append(dataset)
        self.datasets_by_id[dataset.id] = dataset

        try:
            self.datasets_by_type[dataset.content_type].append(dataset)
        except KeyError:
            self.datasets_by_type[dataset.content_type] = [dataset]

    def export(
        self,
        use_uuid: bool = False,
        autoencodings: bool = False,
        ignore_chromsizes: bool = False,
    ):
        return [
            dataset.export(
                use_uuid=use_uuid,
                autoencodings=autoencodings,
                ignore_chromsizes=ignore_chromsizes,
            )
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

    def createCacheHash(self, encoders, config):
        # Generate filename from the set of datasets and encoders
        encoder_filenames = [encoder.encoder_filename for encoder in encoders]
        dataset_filenames = [dataset.filename for dataset in self.datasets]
        all_filenames = ":".join(encoder_filenames + dataset_filenames + config.chroms)

        md5 = hashlib.md5()
        md5.update(all_filenames.encode())
        return md5.hexdigest()

    def remove_cache(self):
        for dataset in self.datasets:
            dataset.remove_cache()

        with suppress(FileNotFoundError):
            os.remove(self.cache_filepath)

    def compute_encodings_dist(
        self,
        target: np.ndarray,
        dist_metric: str = "euclidean",
        batch_size: int = 10000,
        verbose: bool = False,
    ):
        with h5py.File(self.cache_filepath, "r+") as f:
            if verbose:
                print(
                    "Compute distance of encoded windows to the encoded target",
                    end="",
                    flush=True,
                )

            encodings = f["encodings"][:]
            N = encodings.shape[0]
            target = target.reshape((1, -1))

            dist = None
            for batch_start in np.arange(0, N, batch_size):
                if verbose:
                    print(".", end="", flush=True)

                encodings_batch = encodings[batch_start : batch_start + batch_size]

                try:
                    batch_dist = cdist(encodings_batch, target, dist_metric).flatten()
                except ValueError:
                    batch_dist = cdist(encodings_batch, target).flatten()

                if dist is None:
                    dist = batch_dist
                else:
                    dist = np.concatenate((dist, batch_dist))

            f["encodings_dist"][:] = dist

    def get_encodable(self, encoders):
        return utils.flatten(
            [self.get_by_type(encoder.content_type) for encoder in encoders]
        )

    def prepare(
        self,
        encoders,
        config,
        clear: bool = False,
        verbose: bool = False,
    ):
        # Used for assertion checking
        total_num_windows = None
        chrom_num_windows = None

        encodable_datasets = list(self.get_encodable(encoders))

        for encoder in encoders:
            try:
                if verbose:
                    print("Prepare all datasets just for you...", flush=True)

                for dataset in encodable_datasets:

                    ds_total_num_windows, ds_chrom_num_windows = dataset.prepare(
                        config, encoder, clear=clear, verbose=verbose
                    )

                    if total_num_windows is None:
                        total_num_windows = ds_total_num_windows

                    if chrom_num_windows is None:
                        chrom_num_windows = ds_chrom_num_windows

                    if verbose:
                        print(
                            "Make sure that all windows are correctly prepared...",
                            flush=True,
                        )

                    # Check that all datasets have the same number of windows
                    assert (
                        total_num_windows == ds_total_num_windows
                    ), "The total number of windows should be the same for all datasets"

                    # Check that all datasets have the same number of windows
                    assert ds_chrom_num_windows.equals(
                        chrom_num_windows
                    ), "The number of windows per chromosome should be the same for all datasets"

            except KeyError:
                # If there's no data for the encoder we simply continue with our lives
                # pass
                raise

        self._cache_filename = "{}.hdf5".format(self.createCacheHash(encoders, config))
        self._cache_filepath = os.path.join(config.cache_dir, self.cache_filename)

        if verbose:
            print(f'Caching dataset under {self._cache_filename}')

        self._total_len_windows = 0
        self._total_len_encoded = 0

        for dataset in encodable_datasets:
            encoder = encoders.get(dataset.content_type)

            self._total_len_encoded += encoder.latent_dim
            self._total_len_windows += int(encoders.window_size // encoder.resolution)

        # Concatenate data
        mode = "w" if clear else "w-"

        try:
            with h5py.File(self.cache_filepath, mode) as f:
                if verbose:
                    print("Concatenate and save windows...")

                w = f.create_dataset(
                    "windows",
                    (total_num_windows, self.total_len_windows),
                    dtype=np.float32,
                )
                w_max = f.create_dataset(
                    "windows_max", (total_num_windows,), dtype=np.float32
                )
                w_sum = f.create_dataset(
                    "windows_sum", (total_num_windows,), dtype=np.float32
                )
                w_mean = f.create_dataset(
                    "windows_mean", (total_num_windows,), dtype=np.float32
                )
                e = f.create_dataset(
                    "encodings",
                    (total_num_windows, self.total_len_encoded),
                    dtype=np.float32,
                )
                f.create_dataset(
                    "encodings_dist", (total_num_windows,), dtype=np.float32
                )
                e_knn_density = f.create_dataset(
                    "encodings_knn_density", (total_num_windows,), dtype=np.float32
                )

                # Metadata
                w.attrs["total_num_windows"] = total_num_windows
                w.attrs["total_len_windows"] = self._total_len_windows
                e.attrs["total_len_encoded"] = self._total_len_encoded

                pos_window_from = 0
                pos_window_to = 0
                pos_encoded_from = 0
                pos_encoded_to = 0

                for dataset in encodable_datasets:
                    encoder = encoders.get(dataset.content_type)

                    pos_window_to = pos_window_from + encoder.window_num_bins
                    pos_encoded_to = pos_encoded_from + encoder.latent_dim

                    with dataset.cache() as dataset_cache:

                        w[:, pos_window_from:pos_window_to] = np.squeeze(
                            dataset_cache.windows
                        )

                        e[:, pos_encoded_from:pos_encoded_to] = np.squeeze(
                            dataset_cache.encodings
                        )

                        # Write to disk
                        f.flush()

                    pos_window_from = pos_window_to
                    pos_encoded_from = pos_encoded_to

                # Compute simple stats to speed up online calculation down the road
                pos_chrom_from = 0
                pos_chrom_to = 0

                if verbose:
                    print("Compute per-chromosome statistics...")

                for i, chromosome in enumerate(config.chroms):
                    pos_chrom_to = pos_chrom_from + chrom_num_windows[i]

                    w_max[pos_chrom_from:pos_chrom_to] = np.nanmax(
                        w[pos_chrom_from:pos_chrom_to, :], axis=1
                    )
                    w_sum[pos_chrom_from:pos_chrom_to] = np.nansum(
                        w[pos_chrom_from:pos_chrom_to, :], axis=1
                    )
                    w_mean[pos_chrom_from:pos_chrom_to] = np.nanmean(
                        w[pos_chrom_from:pos_chrom_to, :], axis=1
                    )

                    pos_chrom_from = pos_chrom_to

                    # Write to disk
                    f.flush()

                if verbose:
                    print("Compute the encoded windows' knn density...")

                e_knn_density[:] = utils.knn_density(e[:])

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

        if verbose:
            print("All datasets have been prepared! Thanks for waiting.")

    @contextmanager
    def prepared_data(self):
        if not self.cache_filepath:
            raise ValueError("Data not prepared")
        f = h5py.File(self.cache_filepath, "r")
        try:
            yield f
        finally:
            f.close()


class DatasetsCache:
    def __init__(self, cache):
        self.cache = cache

    @property
    def windows(self):
        return self.cache["windows"]

    @property
    def windows_max(self):
        return self.cache["windows_max"]

    @property
    def windows_sum(self):
        return self.cache["windows_sum"]

    @property
    def windows_mean(self):
        return self.cache["windows_mean"]

    @property
    def encodings(self):
        return self.cache["encodings"]

    @property
    def encodings_dist(self):
        return self.cache["encodings_dist"]

    @property
    def computed_dist_to_target(self):
        return np.sum(self.cache["encodings_dist"]) > 0

    @property
    def encodings_knn_density(self):
        return self.cache["encodings_knn_density"]
