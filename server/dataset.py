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

import os
import h5py
import hashlib
import numpy as np
import pandas as pd
import re

from contextlib import contextmanager, suppress

from server import bigwig, utils
from server.chromsizes import get as get_chromsizes


class Dataset:
    def __init__(
        self,
        filepath: str,
        content_type: str,
        id: str,
        name: str,
        filetype: str = None,
        fill: str = None,
        height: int = None,
        chromsizes=None,
        custom_chromosomes=None,
        coords: str = None,
        clear_cache: bool = False,
    ):
        self.filepath = filepath
        self.filetype = filetype
        self.content_type = content_type
        self.id = id
        self.name = name
        self.num_bins = -1
        self.num_windows = -1
        self.fill = fill
        self.height = height
        self.custom_chromosomes = custom_chromosomes
        self.chromsizes = chromsizes
        self.clear_cache = clear_cache
        self.coords = coords

        self._cache = None
        self._is_autoencoded = False

        if self.chromsizes is None:
            self.chromsizes = get_chromsizes(self.coords, self.filepath)

    @property
    def is_autoencoded(self):
        return self._is_autoencoded

    @property
    def filename(self):
        return os.path.basename(self.filepath)

    @property
    def cache_filepath(self):
        return self._cache_filepath

    def get_cache_filename(self, window_size: int, step_freq: int, chroms: list):
        md5 = hashlib.md5()
        md5.update(":".join(chroms).encode())
        chroms_hash = md5.hexdigest()

        filename, _ = os.path.splitext(self.filename)

        return f"{filename}_w-{window_size}_f-{step_freq}_chr-{chroms_hash[:6]}.hdf5"

    @contextmanager
    def cache(self):
        cache = h5py.File(self.cache_filepath, "r")
        try:
            yield DatasetCache(cache)
        finally:
            cache.close()

    def export(
        self,
        use_uuid: bool = False,
        autoencodings: bool = False,
        ignore_chromsizes: bool = False,
    ):
        # Only due to some weirdness in HiGlass
        idKey = "uuid" if use_uuid else "id"
        out = {
            "filepath": None if autoencodings else self.filepath,
            "filetype": "__autoencoding__" if autoencodings else self.filetype,
            "content_type": self.content_type,
            idKey: "{}|ae".format(self.id) if autoencodings else self.id,
            "name": self.name,
            "coords": self.coords,
        }

        if not ignore_chromsizes:
            out["chromsizes"] = self.chromsizes

        return out

    def remove_cache(self):
        with suppress(FileNotFoundError):
            os.remove(self.cache_filepath)

    def prepare(
        self, config, encoder, clear: bool = False, verbose: bool = False
    ) -> int:
        if verbose:
            print("Prepare {}...".format(self.name), flush=True)

        assert (
            self.content_type == encoder.content_type
        ), "Content type of the encoder must match the dataset's content type"

        if self.chromsizes is None:
            self.chromsizes = get_chromsizes(self.coords, self.filepath)

        mode = "w" if clear else "w-"
        step_size = encoder.window_size // config.step_freq
        global_num_bins = None

        # Determine number of windows per chromsome
        num_windows_per_chrom = []
        total_num_windows = 0
        res_size_per_chrom = []
        total_res_sizes = 0

        for chromosome in config.chroms:
            num_windows = utils.get_num_windows(
                self.chromsizes[chromosome], encoder.window_size, step_size
            )
            num_windows_per_chrom.append(num_windows)
            total_num_windows += num_windows
            res_size = int(self.chromsizes[chromosome] // encoder.resolution)
            res_size_per_chrom.append(res_size)
            total_res_sizes += res_size

        chrom_num_windows = pd.Series(
            num_windows_per_chrom, index=config.chroms, dtype=int
        )

        chrom_res_sizes = pd.Series(res_size_per_chrom, index=config.chroms, dtype=int)

        # chroms + encoder.window_size + config.step_freq
        cache_filename = self.get_cache_filename(
            encoder.window_size, config.step_freq, config.chroms
        )

        self._cache_filepath = os.path.join(config.cache_dir, cache_filename)

        ascii_chroms = [n.encode("ascii", "ignore") for n in config.chroms]

        try:
            with h5py.File(self.cache_filepath, mode) as f:
                w = f.create_dataset(
                    "windows",
                    (total_num_windows, encoder.window_num_bins),
                    dtype=np.float32,
                )
                e = f.create_dataset(
                    "encodings",
                    (total_num_windows, encoder.latent_dim),
                    dtype=np.float32,
                )

                # Metadata
                w.attrs["window_size"] = encoder.window_size
                w.attrs["resolution"] = encoder.resolution
                w.attrs["step_size"] = step_size
                w.attrs["step_freq"] = config.step_freq
                w.attrs["chrom_order"] = ascii_chroms
                w.attrs["chrom_num_windows"] = chrom_num_windows
                e.attrs["file_name"] = encoder.encoder_filename
                e.attrs["chrom_num_windows"] = chrom_num_windows
                e.attrs["chrom_order"] = ascii_chroms

                if hasattr(encoder, "autoencode"):
                    a = f.create_dataset(
                        "autoencodings", (total_res_sizes,), dtype=np.float32
                    )
                    a.attrs["chrom_num_windows"] = chrom_num_windows
                    a.attrs["chrom_res_sizes"] = chrom_res_sizes
                    a.attrs["chrom_order"] = ascii_chroms
                    a.attrs["file_name"] = encoder.encoder_filename

                if verbose:
                    print("Extract windows for {}".format(self.id), flush=True)

                pos = 0
                pos_ae = 0

                if verbose:
                    print(
                        "Prepare chromosomes: {}...".format(", ".join(config.chroms)),
                        flush=True,
                    )

                for chromosome in config.chroms:
                    chr_str = str(chromosome)

                    if verbose:
                        print("Extract windows...", flush=True)

                    # Extract the windows
                    windows = bigwig.chunk(
                        self.filepath,
                        encoder.window_size,
                        encoder.resolution,
                        step_size,
                        [chromosome],
                        chromsizes=self.chromsizes,
                        verbose=verbose,
                    )
                    num_windows, num_bins = windows.shape

                    if global_num_bins is None:
                        global_num_bins = num_bins

                    assert (
                        global_num_bins == num_bins
                    ), "Changing number of bins between chromosomes is not allowed"

                    assert (
                        encoder.window_num_bins == global_num_bins
                    ), "Encoder should have the same number of bins as the final data"

                    if encoder.input_dim == 3 and windows.ndim == 2:
                        # Keras expects 3 input dimensions:
                        # 1. number of samples (== number of windows)
                        # 2. sample size (== number of bins per window)
                        # 3. sample dim  (== 1 because each window just has 1 dim)
                        windows = windows.reshape(*windows.shape, encoder.channels)

                    if verbose:
                        print("Encode windows...", flush=True)

                    encoding = encoder.encode(
                        windows,
                        chrom=chr_str,
                        step_freq=config.step_freq
                    )

                    # Data is organized by chromosomes. Currently interchromosomal
                    # patterns are not allowed
                    w[pos : pos + num_windows] = np.squeeze(windows)
                    e[pos : pos + num_windows] = encoding

                    pos += num_windows

                    if hasattr(encoder, "decode"):
                        if verbose:
                            print(
                                "Decode encoded windows, i.e., get the reconstructions...",
                                flush=True,
                            )

                        autoencoding = encoder.decode(encoding)

                        if verbose:
                            print(
                                "Merge interleaved reconstructed windows...", flush=True
                            )

                        # Merge interleaved autoencoded windows to one continuous track
                        autoencoding = utils.merge_interleaved_mat(
                            autoencoding,
                            config.step_freq,
                            utils.get_norm_sym_norm_kernel(
                                encoder.window_size // encoder.resolution
                            ),
                        )

                        a_len = min(chrom_res_sizes[chr_str], autoencoding.shape[0])

                        a[pos_ae : pos_ae + a_len] = autoencoding[:a_len]

                        pos_ae += chrom_res_sizes[chr_str]

                    # Lets write to disk
                    f.flush()
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

        # If we got until here and the encoder is an autoencoder the track was
        # autoencoded.
        self._is_autoencoded = hasattr(encoder, "autoencode")

        # For convenience
        return total_num_windows, chrom_num_windows


class DatasetCache:
    def __init__(self, cache):
        self.cache = cache

    @property
    def windows(self):
        return self.cache["windows"]

    @property
    def chrom_num_windows(self):
        return self.cache["windows"].attrs["chrom_num_windows"]

    @property
    def encodings(self):
        return self.cache["encodings"]

    @property
    def autoencodings(self):
        return self.cache["autoencodings"]

    def num_windows_by_chrom(self, chromosome, config):
        chr_str = str(chromosome).encode("ascii", "ignore")
        for index, chrom in enumerate(self.cache["windows"].attrs["chrom_order"]):
            if chrom == chr_str:
                return self.cache["windows"].attrs["chrom_num_windows"][index]
        return None
