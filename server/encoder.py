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


class Encoder:
    def __init__(
        self,
        encoder: str,
        content_type: str,
        window_size: int,
        resolution: int,
        channels: int,
        input_dim: int,
        latent_dim: int,
    ):
        self.en_path = encoder
        self.content_type = content_type
        self.window_size = window_size
        self.resolution = resolution
        self.channels = channels
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.en = keras.models.load_model(self.en_path)

    def encode(self, data: np.ndarray) -> np.ndarray:
        return self.en.predict(data)


class Autoencoder(Encoder):
    def __init__(
        self,
        encoder: str,
        decoder: str,
        content_type: str,
        window_size: int,
        resolution: int,
        channels: int,
        input_dim: int,
        latent_dim: int,
    ):
        super(Autoencoder, self).__init__(
            encoder,
            content_type,
            window_size,
            resolution,
            channels,
            input_dim,
            latent_dim,
        )
        self.de_path = decoder
        self.de = keras.models.load_model(self.de_path)

    def autoencode(self, data: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(data))

    def decode(self, data: np.ndarray) -> np.ndarray:
        return self.de.predict(data)
