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
import re

from server import bigwig
from server import utils


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
        self.chromsizes = chromsizes
        self.clear_cache = clear_cache

        self.chunks = None
        self.encoding = None
        self.autoencoding = None

        if not self.chromsizes:
            self.chromsizes = bigwig.get_chromsizes(self.filepath)

    @property
    def is_autoencoded(self):
        return self.autoencoding is not None

    @property
    def filename(self):
        return os.path.basename(self.filepath)

    @property
    def cache_filename(self):
        return "cache/{}.hdf5".format(self.filename)

    def export(self, use_uuid: bool = False, autoencodings: bool = False):
        # Only due to some weirdness in HiGlass
        idKey = "uuid" if use_uuid else "id"
        return {
            "filepath": None if autoencodings else self.filepath,
            "filetype": "__autoencoding__" if autoencodings else self.filetype,
            "content_type": self.content_type,
            idKey: "{}|ae".format(self.id) if autoencodings else self.id,
            "name": self.name,
        }

    def prepare(self, config, encoder, clear: bool = False, verbose: bool = False):
        assert self.content_type == encoder.content_type

        self.chromsizes = bigwig.get_chromsizes(self.filepath)

        mode = "w" if clear else "w-"
        step_size = encoder.window_size // config.step_freq
        num_total_windows = 0
        global_num_bins = None

        try:
            with h5py.File(self.cache_filename, mode) as f:
                w = f.create_group("windows")
                e = f.create_group("encodings")
                a = f.create_group("autoencodings")
                for chromosome in config.chroms:
                    chr_str = str(chromosome)

                    # Extract the windows
                    windows = bigwig.chunk(
                        self.filepath,
                        encoder.window_size,
                        encoder.resolution,
                        step_size,
                        [chromosome],
                        verbose=verbose,
                    )

                    num_windows, num_bins = windows.shape

                    num_total_windows += num_windows

                    if global_num_bins is None:
                        global_num_bins = num_bins

                    assert (
                        global_num_bins == num_bins
                    ), "Changing number of bins between chromosomes is not allowed"

                    assert (
                        encoder.window_num_bins == global_num_bins
                    ), "Encoder should have the same number of bins as the final data"

                    if encoder.input_dim == 3 and windows.ndim == 2:
                        windows = windows.reshape(*windows.shape, encoder.channels)

                    encoding = encoder.encode(windows)

                    # Data is organized by chromosomes. Currently interchromosomal
                    # patterns are not allowed
                    w[chr_str] = windows
                    w[chr_str].attrs["window_size"] = encoder.window_size
                    w[chr_str].attrs["resolution"] = encoder.resolution
                    w[chr_str].attrs["step_size"] = step_size
                    w[chr_str].attrs["step_freq"] = config.step_freq

                    e[chr_str] = encoding
                    e[chr_str].attrs["file_name"] = encoder.encoder_filename

                    if getattr(encoder, "autoencode"):
                        autoencoding = encoder.autoencode(windows)

                        # Merge interleaved autoencoded windows to one continuous track
                        autoencoding = utils.merge_interleaved_mat(
                            autoencoding,
                            config.step_freq,
                            utils.get_norm_sym_norm_kernel(
                                encoder.window_size // encoder.resolution
                            ),
                        )

                        a[chr_str] = autoencoding
                        a[chr_str].attrs["file_name"] = encoder.encoder_filename

                    # Lets write to disk
                    f.flush()

                w.attrs["num_total_windows"] = num_total_windows
                print("global_num_bins2", global_num_bins)
                w.attrs["num_bins"] = encoder.window_num_bins
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

        # Return for convenience
        return num_total_windows
