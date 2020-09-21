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
import os

from server.utils import load_model, get_encoder, get_decoder


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
        model_args: list = None,
    ):
        self.encoder_filepath = encoder_filepath
        self.content_type = content_type
        self.window_size = window_size
        self.resolution = resolution
        self.channels = channels
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.window_num_bins = int(self.window_size // self.resolution)
        self.model_args = [] if model_args is None else model_args

        self._encoder = None

    @property
    def encoder_filename(self):
        if self.encoder_filepath is not None:
            return os.path.basename(self.encoder_filepath)
        else:
            return os.path.basename(self.autoencoder_filepath)

    @property
    def encoder(self):
        # Lazy load model
        if self._encoder is None:
            if self.encoder_filepath is not None:
                self._encoder = load_model(
                    self.encoder_filepath,
                    silent=True,
                    additional_args=self.model_args,
                )
            else:
                if self._autoencoder is None:
                    self._autoencoder = load_model(
                        self.autoencoder_filepath,
                        silent=True,
                        additional_args=self.model_args,
                    )
                self._encoder = get_encoder(self._autoencoder)
        return self._encoder

    def encode(
        self,
        data: np.ndarray = None,
        chrom: str = None,
        start: int = None,
        end: int = None,
        step_freq: int = None,
    ) -> np.ndarray:
        if hasattr(self.encoder, 'is_data_agnostic'):
            # Custom encoder model
            return self.encoder.predict(
                chrom=chrom,
                start=start,
                end=end,
                window_size=self.window_size,
                step_size=self.window_size // step_freq
            )

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
        content_type: str,
        window_size: int,
        resolution: int,
        channels: int,
        input_dim: int,
        latent_dim: int,
        autoencoder_filepath: str = None,
        encoder_filepath: str = None,
        decoder_filepath: str = None,
        model_args: list = None,
    ):
        super(Autoencoder, self).__init__(
            encoder_filepath,
            content_type,
            window_size,
            resolution,
            channels,
            input_dim,
            latent_dim,
            model_args,
        )

        self.autoencoder_filepath = autoencoder_filepath
        self._autoencoder = None

        self.decoder_filepath = decoder_filepath
        self._decoder = None

    @property
    def decoder_filename(self):
        return os.path.basename(self.decoder_filepath)

    @property
    def decoder(self):
        # Lazy load model
        if self._decoder is None:
            if self.decoder_filepath is not None:
                self._decoder = load_model(self.decoder_filepath, silent=True)
            else:
                if self._autoencoder is None:
                    self._autoencoder = load_model(
                        self.autoencoder_filepath, silent=True
                    )
                self._decoder = get_decoder(self._autoencoder)
        return self._decoder

    def autoencode(self, data: np.ndarray) -> np.ndarray:
        if self._autoencoder is not None:
            return self._autoencoder.predict(data)
        return self.decode(self.encode(data))

    def decode(self, data: np.ndarray) -> np.ndarray:
        return self.decoder.predict(data)

    def export(self):
        export = super(Autoencoder, self).export()
        export["decoder"] = self.decoder_filepath
        return export
