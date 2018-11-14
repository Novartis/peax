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

import keras
import numpy as np
import os


class Encoder:
    def __init__(
        self,
        encoder_filepath: str,
        content_type: str,
        window_size: int,
        resolution: int,
        channels: int,
        input_dim: int,
        latent_dim: int,
    ):
        self.encoder_filepath = encoder_filepath
        self.content_type = content_type
        self.window_size = window_size
        self.resolution = resolution
        self.channels = channels
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.window_num_bins = int(self.window_size // self.resolution)

        self.encoder = keras.models.load_model(self.encoder_filepath)

    @property
    def encoder_filename(self):
        return os.path.basename(self.encoder_filepath)

    def encode(self, data: np.ndarray) -> np.ndarray:
        return self.encoder.predict(data)

    def export(self):
        return {
            "encoder": self.encoder_filepath,
            "content_type": self.content_type,
            "window_size": self.window_size,
            "resolution": self.resolution,
            "channels": self.channels,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
        }


class Autoencoder(Encoder):
    def __init__(
        self,
        encoder_filepath: str,
        decoder_filepath: str,
        content_type: str,
        window_size: int,
        resolution: int,
        channels: int,
        input_dim: int,
        latent_dim: int,
    ):
        super(Autoencoder, self).__init__(
            encoder_filepath,
            content_type,
            window_size,
            resolution,
            channels,
            input_dim,
            latent_dim,
        )
        self.decoder_filepath = decoder_filepath
        self.decoder = keras.models.load_model(self.decoder_filepath)

    @property
    def decoder_filename(self):
        return os.path.basename(self.decoder_filepath)

    def autoencode(self, data: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(data))

    def decode(self, data: np.ndarray) -> np.ndarray:
        return self.decoder.predict(data)

    def export(self):
        export = super(Autoencoder, self).export()
        export["decoder"] = self.decoder_filepath
        return export
