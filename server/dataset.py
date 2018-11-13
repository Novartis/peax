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

        if not self.chromsizes:
            self.chromsizes = bigwig.get_chromsizes(self.filepath)

    def export(self, use_uuid: bool = False):
        # Only due to some weirdness in HiGlass
        idKey = "uuid" if use_uuid else "id"
        return {
            "filepath": self.filepath,
            "filetype": self.filetype,
            "content_type": self.content_type,
            idKey: self.id,
            "name": self.name,
        }

    def prepare(self, config, encoder, verbose: bool = False) -> list:
        assert self.content_type == encoder.content_type

        # Check if file is cached
        self.cache_filename = "{}.arrow".format(os.path.basename(self.filepath))

        self.chromsizes = bigwig.get_chromsizes(self.filepath)

        # Extract the windows
        self.chunks = bigwig.chunk(
            self.filepath,
            encoder.window_size,
            encoder.resolution,
            encoder.window_size // config.step_freq,
            config.chroms,
            verbose=verbose,
        )

        self.num_windows, self.num_bins = self.chunks.shape

        if encoder.input_dim == 3 and self.chunks.ndim == 2:
            self.chunks = self.chunks.reshape(*self.chunks.shape, encoder.channels)
        self.encoding = encoder.encode(self.chunks)
        self.autoencoding = encoder.autoencode(self.chunks)

        # Merge interleaved autoencoded windows to one continuous track
        self.autoencoding = utils.merge_interleaved_mat(
            self.autoencoding,
            config.step_freq,
            utils.get_norm_sym_norm_kernel(encoder.window_size // encoder.resolution),
        )
